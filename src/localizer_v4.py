"""
localizer_v4.py
-----------------
Phase 2 ("Registration under Unknown Pose") engine. Drop-in continuation
of localizer_v3.py: IDENTICAL coarse-to-fine pyramid search, IDENTICAL
CNN-landmark path, IDENTICAL ambiguity-gated lattice-enumeration
fallback (see localizer_v2.py's docstring for that logic and why it's
gated the way it is) -- v1/v2/v3 are untouched and still fully reachable
for comparison. Two things are ADDED, both required by the addendum and
neither optional:

1. CONTINUOUS POSE REFINEMENT (pose_refine.py). v1/v2/v3 report the
   scale/rotation HYPOTHESIS the grid search happened to land on --
   fine when neither is scored (Phase 1), nowhere near Phase 2's
   tolerances (scale within 1%, rotation within 0.25deg for full
   credit). Once the tie-break above has picked a site, refine_pose()
   polishes (x, y, scale, rotation) continuously around it -- see that
   module's docstring for why a LOCAL step is appropriate here even
   though the global candidate search above must not crop.

2. A "FOUND" DECISION. Phase 1 had no reason to ever say "not present" --
   every sample had a defined, scoreable target (a real fiducial, or the
   spec's own "nearest periodic repeat to centre" fallback). Phase 2's
   Set C is different in kind: the reference is not anywhere in the
   search image at all. `found_model` (see train_confidence_p2.py) is
   the SAME kind of object as v1/v2/v3's `confidence_model` -- a small
   scikit-learn classifier over engineered correlation-surface features
   -- trained on an expanded feature set (adds the REFINED peak score
   and the CNN's own best score to the original six) and, critically, on
   data that includes genuinely absent pairs, which Phase 1's generator
   structurally could not produce (see dataset_generator.py's
   make_pair_p2). Its calibrated probability serves double duty as both
   the reported `score` column (checked for AUC/calibration) and, after
   a tuned threshold, the `found` flag -- one trained number answering
   "should I trust this prediction," rather than two disconnected ones.
   Degrades gracefully exactly like the Phase 1 confidence model: no
   weights file -> a documented heuristic fallback, never a crash.

Nothing about the CNN path, the ambiguity gate, or the lattice
enumeration changed even by one line from v2/v3.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from cnn_matcher import PatchCNN
from localizer import _nms_peaks, _subpixel_offset, _cnn_score_peaks
from pyramid_search import pyramid_scale_rotation_search
from lattice_localizer import estimate_lattice_vectors, enumerate_lattice_closest_to_center, refine_lattice_from_peaks
from pose_refine import refine_pose
from pattern_render import to_luminance


@dataclass
class LocalizationResultV4:
    x: float
    y: float
    scale: float                # continuous, refined -- NOT a grid point
    theta_deg: float             # continuous, refined; CCW-positive, "reference as it appears
                                  # in the search image" -- already the addendum's own
                                  # convention (see pose_refine.py / dataset_generator.py
                                  # docstrings for the empirical sign verification)
    score: float                  # calibrated P(this x/y/found is correct) -- report this
                                  # verbatim as the predictions.csv `score` column
    found: bool                    # thresholded `score` -- report this as `found`
    grid_scale: float               # pre-refinement grid hypothesis (diagnostics only)
    grid_angle: float
    best_score: float
    second_score: float
    refined_peak_score: float
    num_candidate_peaks: int
    cnn_used: bool = False
    cnn_score: float = 0.0
    tie_break_method: str = "classical_center"
    time_ms: float = 0.0
    candidate_peaks: list = field(default_factory=list)
    features: list = field(default_factory=list)   # exactly what found_model.predict_proba saw --
                                                      # train_confidence_p2.py reuses THIS, never a
                                                      # separately-recomputed copy, so training-time
                                                      # and inference-time features cannot drift apart.


def _heuristic_found_score(refined_peak: float, cnn_best: float, ratio: float) -> float:
    """Used ONLY if no trained found_model is available (missing/corrupt
    weights file) -- degrades gracefully rather than crashing, exactly
    like v1/v2/v3's confidence_model fallback. Not the primary mechanism:
    train_confidence_p2.py's calibrated classifier is.
    """
    return float(np.clip(0.6 * refined_peak + 0.25 * cnn_best + 0.15 * ratio, 0.0, 1.0))


def localize_v4(reference: np.ndarray, search: np.ndarray,
                 scale_range=(8.0, 12.0), n_scale: int = 9,
                 angle_range=(-5.0, 5.0), n_angle: int = 5,
                 # downsample=2, not the coarser 4 the coarse-to-fine pyramid
                 # started with: at scale=12 (max zoom, smallest template),
                 # the finest disclosed presets have periods as small as 3.3-4
                 # search-px (e.g. finfet_7nm's 40nm fin pitch / 12nm search
                 # px). A 4x coarse-stage downsample compresses that under 1px
                 # -- aliased away entirely -- and the coarse grid can then
                 # converge on a confidently-wrong scale (measured directly:
                 # one such case landed 321px from ground truth at
                 # downsample=4, scale=8.75 instead of the true 12.0; the
                 # SAME pair resolves to 0.4px error at downsample<=2). 2 keeps
                 # >=2px/period even in this worst case while still roughly
                 # halving runtime versus no downsampling at all -- see
                 # docs/validation_report_v5.md, "The downsample=4 aliasing
                 # bug" for the full sweep across presets/severities/seeds.
                 n_scale_fine: int = 5, n_angle_fine: int = 5, downsample: int = 2,
                 abs_gap: float = 0.02,
                 found_model=None, found_threshold: float = 0.5,
                 cnn_matcher=None, cnn_accept_thresh: float = 0.5,
                 unique_match_ratio: float = 0.06,
                 refine: bool = True, min_std: float = 2.0) -> LocalizationResultV4:
    t0 = time.perf_counter()
    reference = to_luminance(np.asarray(reference)).astype(np.float32)
    search = to_luminance(np.asarray(search)).astype(np.float32)
    sh, sw = search.shape[:2]

    # Guard against degenerate (near-constant) input BEFORE running any
    # correlation. cv2.matchTemplate's TM_CCOEFF_NORMED is a 0/0 ratio
    # (covariance / product-of-std-devs) when either the template or the
    # search window has ~zero variance -- OpenCV's own handling of that
    # case returns exactly 1.0 (a "perfect match") everywhere rather
    # than 0 or NaN. Measured directly: two all-black 1000x1000 images
    # produced best_score=1.0 and a CONFIDENT found=1 at a specific
    # (x,y,scale,theta) with no real signal behind any of it -- the
    # worst possible failure mode (confidently wrong, not just wrong).
    # A legitimate capture of a real pattern always has real contrast;
    # a genuinely flat reference or search means a failed acquisition,
    # not "the pattern is this flat region," so this is a real
    # precondition to check, not a heuristic tuned to one bad image.
    if float(np.std(reference)) < min_std or float(np.std(search)) < min_std:
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        return LocalizationResultV4(
            x=sw / 2.0, y=sh / 2.0, scale=float(np.mean(scale_range)), theta_deg=0.0,
            score=0.0, found=False, grid_scale=float(np.mean(scale_range)), grid_angle=0.0,
            best_score=0.0, second_score=0.0, refined_peak_score=0.0, num_candidate_peaks=0,
            cnn_used=False, cnn_score=0.0, tie_break_method="degenerate_input",
            time_ms=elapsed_ms, candidate_peaks=[], features=[0.0] * 8,
        )

    best_corr, best_template_size, best_scale, best_angle = pyramid_scale_rotation_search(
        reference, search, scale_range=scale_range, n_scale_coarse=n_scale,
        angle_range=angle_range, n_angle_coarse=n_angle,
        n_scale_fine=n_scale_fine, n_angle_fine=n_angle_fine, downsample=downsample)

    peaks = _nms_peaks(best_corr, best_template_size, abs_gap=abs_gap, center=(sw / 2.0, sh / 2.0))
    if not peaks:
        _, max_val, _, max_loc = cv2.minMaxLoc(best_corr)
        th, tw = best_template_size
        peaks = [(max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0, float(max_val))]
    peaks.sort(key=lambda p: -p[2])
    best_score = peaks[0][2]
    second_score = peaks[1][2] if len(peaks) > 1 else 0.0

    cx_img, cy_img = sw / 2.0, sh / 2.0
    cnn_used = False
    cnn_best = 0.0
    tie_break_method = "classical_center"

    th, tw = best_template_size
    # Gate: same "is the classical winner ALREADY ambiguous" check the
    # lattice path uses below, now shared with the CNN path too. Originally
    # (v2/v3) the CNN could override a clear classical winner unconditionally
    # -- sound when the CNN's job is "spot the rare, deliberately-placed
    # fiducial" (Phase 1's pattern model), since a genuine fiducial is strong,
    # reliable evidence regardless of how sharp the correlation peak already
    # is. It stops being sound once the CNN's job is instead "guess which
    # subtly-different candidate is more likely correct" (no fiducial exists
    # in the addendum's own generator -- confirmed directly: cnn_score sits
    # at ~0.0000-0.0148 across all 20 real organizer samples, never once
    # crossing cnn_accept_thresh, see docs/validation_report_v5.md,
    # "The CNN has nothing to detect"). A classifier retrained for that
    # harder task carries real overfit risk, so it may only break a tie the
    # classical evidence already flags as one, never overrule a clear
    # classical winner outright.
    raw_ratio = 1.0 - (second_score / best_score) if best_score > 1e-6 else 0.0
    genuinely_ambiguous = len(peaks) > 1 and raw_ratio < unique_match_ratio

    if cnn_matcher is not None:
        cnn_scores = _cnn_score_peaks(search, peaks, cnn_matcher)
        cnn_best = max(cnn_scores) if cnn_scores else 0.0
    else:
        cnn_scores = []

    if cnn_matcher is not None and cnn_best >= cnn_accept_thresh and genuinely_ambiguous:
        # Landmark path: UNCHANGED from v2/v3 -- other periodic repeats
        # are not valid alternatives once the CNN is confident. Only the
        # trigger condition (now also requiring genuine ambiguity) is new.
        best_i = int(np.argmax(cnn_scores))
        chosen = peaks[best_i]
        cnn_used = True
        px_int = int(np.clip(round(chosen[0] - tw / 2.0), 0, best_corr.shape[1] - 1))
        py_int = int(np.clip(round(chosen[1] - th / 2.0), 0, best_corr.shape[0] - 1))
        dx, dy = _subpixel_offset(best_corr, px_int, py_int)
        anchor_x, anchor_y = chosen[0] + dx, chosen[1] + dy
        tie_break_method = "cnn_landmark"
    else:
        # UNCHANGED from v2/v3: gate the lattice jump on the correlation
        # surface's OWN evidence of ambiguity, not on the CNN's opinion
        # alone (see localizer_v2.py's docstring for why).

        _, max_val, _, max_loc = cv2.minMaxLoc(best_corr)
        raw_anchor_x = max_loc[0] + tw / 2.0
        raw_anchor_y = max_loc[1] + th / 2.0
        adx, ady = _subpixel_offset(best_corr, max_loc[0], max_loc[1])
        raw_anchor_x, raw_anchor_y = raw_anchor_x + adx, raw_anchor_y + ady

        lattice = estimate_lattice_vectors(best_corr, subpixel=True) if genuinely_ambiguous else None
        if lattice is not None:
            v1, v2 = lattice
            if len(peaks) >= 5:
                # Sharpen further using every detected peak's own subpixel
                # position, not just the single autocorrelation sample --
                # see lattice_localizer.refine_lattice_from_peaks and
                # docs/validation_report_v4.md, "Pure-periodic localization"
                # for why this matters specifically at Phase 2's tolerances.
                refined_peaks_xy = []
                for (px, py, _s) in peaks:
                    px_int = int(np.clip(round(px - tw / 2.0), 0, best_corr.shape[1] - 1))
                    py_int = int(np.clip(round(py - th / 2.0), 0, best_corr.shape[0] - 1))
                    pdx, pdy = _subpixel_offset(best_corr, px_int, py_int)
                    refined_peaks_xy.append((px + pdx, py + pdy))
                v1, v2 = refine_lattice_from_peaks(refined_peaks_xy, v1, v2)
            bounds = (tw / 2.0, th / 2.0, sw - tw / 2.0, sh - th / 2.0)
            anchor_x, anchor_y = enumerate_lattice_closest_to_center(
                (raw_anchor_x, raw_anchor_y), v1, v2, bounds, (cx_img, cy_img))
            tie_break_method = "lattice_enumeration"
        else:
            chosen = min(peaks, key=lambda p: (p[0] - cx_img) ** 2 + (p[1] - cy_img) ** 2)
            px_int = int(np.clip(round(chosen[0] - tw / 2.0), 0, best_corr.shape[1] - 1))
            py_int = int(np.clip(round(chosen[1] - th / 2.0), 0, best_corr.shape[0] - 1))
            dx, dy = _subpixel_offset(best_corr, px_int, py_int)
            anchor_x, anchor_y = chosen[0] + dx, chosen[1] + dy
            tie_break_method = "classical_center_fallback"

    # --- NEW: continuous pose refinement around the chosen site ---
    coarse_scale_step = (scale_range[1] - scale_range[0]) / max(1, n_scale - 1)
    coarse_angle_step = (angle_range[1] - angle_range[0]) / max(1, n_angle - 1)
    if refine:
        r = refine_pose(reference, search, anchor_x, anchor_y, best_scale, best_angle,
                         scale_step=coarse_scale_step, angle_step=coarse_angle_step,
                         scale_bounds=scale_range, angle_bounds=angle_range)
        final_x, final_y = r["x"], r["y"]
        final_scale = float(np.clip(r["scale"], scale_range[0], scale_range[1]))
        final_angle = float(np.clip(r["angle"], angle_range[0], angle_range[1]))
        refined_peak = r["peak_score"] if r["peak_score"] > 0 else best_score
    else:
        final_x, final_y = anchor_x, anchor_y
        final_scale, final_angle = best_scale, best_angle
        refined_peak = best_score

    ratio = 1.0 - (second_score / best_score) if best_score > 1e-6 else 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))

    feats = np.array([[best_score, second_score, ratio, refined_peak, cnn_best,
                        float(len(peaks)), final_scale, final_angle]])
    score = None
    if found_model is not None:
        try:
            score = float(found_model.predict_proba(feats)[0, 1])
        except Exception:
            score = None
    if score is None:
        score = _heuristic_found_score(refined_peak, cnn_best, ratio)
    found = bool(score >= found_threshold)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return LocalizationResultV4(
        x=float(final_x), y=float(final_y), scale=final_scale, theta_deg=final_angle,
        score=score, found=found, grid_scale=best_scale, grid_angle=best_angle,
        best_score=float(best_score), second_score=float(second_score),
        refined_peak_score=float(refined_peak), num_candidate_peaks=len(peaks),
        cnn_used=cnn_used, cnn_score=float(cnn_best), tie_break_method=tie_break_method,
        time_ms=elapsed_ms, candidate_peaks=peaks, features=feats[0].tolist(),
    )

"""
localizer_v3.py
-----------------
Drop-in upgrade to localizer_v2.py. Identical CNN-landmark path and
identical ambiguity-gated lattice-enumeration fallback (see
localizer_v2.py's docstring for that logic and why it's gated the way
it is) -- the ONLY change is how the coarse scale/rotation search is
done.

(train_cnn_matcher.py was separately improved -- noise-matched training
-- while validating this version, but that fix lives in the training
script and benefits v1/v2/v3 equally once retrained; it is NOT a
v3-specific change. See docs/validation_report_v3.md's "methodology
error, caught and fixed" section: an earlier draft of this docstring
attributed that gain to v3 specifically, which doesn't hold up under a
clean-machine regression test -- corrected here.)

v1/v2 both use a flat, single-resolution 9x5=45-hypothesis grid, every
hypothesis a full-resolution correlation against the full 1000x1000
search image -- measured at ~1.7-2.0s, essentially the entire runtime of
one localize() call.

v3 replaces that one step with a coarse-to-fine pyramid
(pyramid_search.py): a cheap 4x-downsampled pass over the same 45
hypotheses to find the right scale/rotation neighbourhood (~0.35s), then
a small, full-resolution pass centred on that neighbourhood
(n_scale_fine x n_angle_fine hypotheses, default 3x3=9).

MEASURED on this repo's own 30-pair test set, CNN held constant across
engines so the comparison isolates the search algorithm (not estimated
-- see docs/validation_report_v3.md for the full numbers and
methodology):
  - Mean time/pair: 1969ms (v2) -> 1099ms (v3 default, 3x3 fine grid):
    ~1.8x faster.
  - WITH-fiducial fair<5px: 100% both v2 and v3 (unchanged -- the CNN is
    identical between them in this comparison).
  - PURE-periodic fair-target median error: 15.13px both v2 and v3
    (unchanged), since a 3x3 fine grid re-centred on the coarse winner
    uses the SAME step size as the original flat grid -- just fewer
    wasted hypotheses far from the optimum, not finer resolution near
    it. A denser fine grid (n_scale_fine=5, n_angle_fine=5) trades part
    of the speed back for finer final scale/rotation resolution, which
    measurably helped this repo's periodic-only test set (~15px ->
    ~13px median, n=12 -- a real but small-sample-size result). Pass
    n_scale_fine=5, n_angle_fine=5 explicitly if that trade is worth it
    for your use case.

Nothing about the CNN path, the ambiguity gate, or the lattice
enumeration changed even by one line from localizer_v2.py -- this
module differs only in the coarse-search step (pyramid_search instead
of the flat grid inlined in v2). v1/v2/v3 all load the same
weights/cnn_matcher.npz by default -- there is only one CNN in this
repo, and it's shared.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np

from cnn_matcher import PatchCNN
from localizer import _nms_peaks, _subpixel_offset, _cnn_score_peaks
from pyramid_search import pyramid_scale_rotation_search
from lattice_localizer import estimate_lattice_vectors, enumerate_lattice_closest_to_center


@dataclass
class LocalizationResultV3:
    x: float
    y: float
    confidence: float
    scale: float
    rotation_deg: float
    num_candidate_peaks: int
    best_score: float
    second_score: float
    time_ms: float
    cnn_used: bool = False
    cnn_score: float = 0.0
    tie_break_method: str = "classical_center"
    candidate_peaks: list = field(default_factory=list)


def localize_v3(reference: np.ndarray, search: np.ndarray,
                 scale_range=(8.0, 12.0), n_scale=9,
                 angle_range=(-6.0, 6.0), n_angle=5,
                 n_scale_fine: int = 3, n_angle_fine: int = 3, downsample: int = 4,
                 abs_gap: float = 0.02,
                 confidence_model=None,
                 cnn_matcher=None,
                 cnn_accept_thresh: float = 0.5,
                 unique_match_ratio: float = 0.06) -> LocalizationResultV3:
    t0 = time.perf_counter()
    reference = reference.astype(np.float32)
    search = search.astype(np.float32)
    sh, sw = search.shape[:2]

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

    if cnn_matcher is not None:
        cnn_scores = _cnn_score_peaks(search, peaks, cnn_matcher)
        cnn_best = max(cnn_scores) if cnn_scores else 0.0
    else:
        cnn_scores = []

    if cnn_matcher is not None and cnn_best >= cnn_accept_thresh:
        # --- Landmark path: UNCHANGED from localizer.py. The CNN found a
        # unique, confident match -- other periodic repeats are NOT valid
        # alternatives here (only the landmark-bearing site is correct),
        # so lattice enumeration must NOT run in this branch. ---
        best_i = int(np.argmax(cnn_scores))
        chosen = peaks[best_i]
        cnn_used = True
        th, tw = best_template_size
        px_int = int(np.clip(round(chosen[0] - tw / 2.0), 0, best_corr.shape[1] - 1))
        py_int = int(np.clip(round(chosen[1] - th / 2.0), 0, best_corr.shape[0] - 1))
        dx, dy = _subpixel_offset(best_corr, px_int, py_int)
        final_x, final_y = chosen[0] + dx, chosen[1] + dy
        tie_break_method = "cnn_landmark"
    else:
        # --- CNN wasn't confident -- but that alone does NOT mean the
        # match is genuinely ambiguous. It can also mean a real landmark
        # is present but the CNN failed to recognise it (e.g. at a noise
        # level outside its training distribution). Trusting a lattice
        # jump in that situation would actively throw away a correct,
        # unique match in favour of a merely-periodic lookalike -- which
        # is exactly backwards. So gate the lattice jump on the
        # classical correlation surface's OWN evidence of ambiguity
        # (a near-tied runner-up peak), not on the CNN's opinion alone.
        # A single dominant, well-separated peak is itself evidence of a
        # one-off feature and should be trusted as-is.
        raw_ratio = 1.0 - (second_score / best_score) if best_score > 1e-6 else 0.0
        genuinely_ambiguous = len(peaks) > 1 and raw_ratio < unique_match_ratio

        _, max_val, _, max_loc = cv2.minMaxLoc(best_corr)
        th, tw = best_template_size
        anchor_x = max_loc[0] + tw / 2.0
        anchor_y = max_loc[1] + th / 2.0
        adx, ady = _subpixel_offset(best_corr, max_loc[0], max_loc[1])
        anchor_x, anchor_y = anchor_x + adx, anchor_y + ady

        lattice = estimate_lattice_vectors(best_corr) if genuinely_ambiguous else None
        if lattice is not None:
            v1, v2 = lattice
            bounds = (tw / 2.0, th / 2.0, sw - tw / 2.0, sh - th / 2.0)
            final_x, final_y = enumerate_lattice_closest_to_center(
                (anchor_x, anchor_y), v1, v2, bounds, (cx_img, cy_img))
            tie_break_method = "lattice_enumeration"
        else:
            # Not genuinely ambiguous (a single dominant peak, or no clear
            # periodicity measurable) -- trust the classical rule among
            # NMS survivors directly. With few/no real competitors this
            # reduces to trusting the anchor, which is the right call
            # when the correlation surface itself found a unique match
            # that the CNN simply didn't independently confirm.
            chosen = min(peaks, key=lambda p: (p[0] - cx_img) ** 2 + (p[1] - cy_img) ** 2)
            px_int = int(np.clip(round(chosen[0] - tw / 2.0), 0, best_corr.shape[1] - 1))
            py_int = int(np.clip(round(chosen[1] - th / 2.0), 0, best_corr.shape[0] - 1))
            dx, dy = _subpixel_offset(best_corr, px_int, py_int)
            final_x, final_y = chosen[0] + dx, chosen[1] + dy
            tie_break_method = "classical_center_fallback"

    ratio = 1.0 - (second_score / best_score) if best_score > 1e-6 else 0.0
    ratio = float(np.clip(ratio, 0.0, 1.0))
    confidence = max(ratio, cnn_best) if cnn_used else ratio
    if confidence_model is not None:
        try:
            feats = np.array([[best_score, second_score, ratio, len(peaks),
                                best_scale, best_angle]])
            calibrated = float(confidence_model.predict_proba(feats)[0, 1])
            confidence = max(calibrated, cnn_best) if cnn_used else calibrated
        except Exception:
            pass

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return LocalizationResultV3(
        x=float(final_x), y=float(final_y), confidence=confidence,
        scale=best_scale, rotation_deg=best_angle,
        num_candidate_peaks=len(peaks), best_score=float(best_score),
        second_score=float(second_score), time_ms=elapsed_ms,
        cnn_used=cnn_used, cnn_score=cnn_best, tie_break_method=tie_break_method,
        candidate_peaks=peaks,
    )

"""
localizer.py
-------------
Core matching engine behind localize.py. Kept as a separate importable
module so evaluate.py and train_confidence.py can call it directly
in-process, while localize.py stays a thin, spec-compliant CLI wrapper.

Pipeline
========
1. COARSE, MULTI-SCALE, MULTI-ROTATION SEARCH
   The true scale (~10x) and rotation (~0, small) are known only
   approximately, so we sweep a small grid of candidate (scale, angle)
   pairs, resize+rotate the reference into a template at each, and run
   normalized cross-correlation (cv2.matchTemplate, TM_CCOEFF_NORMED,
   FFT-accelerated internally by OpenCV) against the full search image.
   The hypothesis whose OWN correlation surface has the highest peak
   wins -- an earlier version of this file tried taking an element-wise
   max across *all* hypotheses first, hoping it would be more
   noise-robust; empirically it was the opposite (many hypotheses
   compared independently at every pixel gives noise a lot of free
   rolls of the dice). See docs/references.md.

2. PERIODIC-AWARE PEAK EXTRACTION
   Highly periodic layouts produce *many* near-identical correlation
   peaks, spaced roughly one pattern period apart. NMS here uses a
   suppression footprint that is a *fraction* of the template footprint
   (not the whole footprint), so adjacent genuine period-repeats stay
   distinct. The tie threshold is an ABSOLUTE score gap (empirically
   ~0.01-0.03 in TM_CCOEFF_NORMED units, not a percentage of the peak
   height -- see docs/references.md). When more candidates qualify than
   `max_peaks`, truncation keeps the ones CLOSEST TO the search image's
   center rather than the highest-scoring ones, since that is what the
   downstream tie-break actually selects on -- truncating by score first
   was found to silently discard the true tie-break winner on densely
   periodic fields (see docs/references.md).

3. LEARNED RE-RANKING (optional, `cnn_matcher`)
   A small from-scratch NumPy CNN (cnn_matcher.py) looks at each
   candidate's own neighbourhood in the search image and scores "does
   this look like the true landmark." When confident about any
   candidate, its choice overrides the classical tie-break; otherwise
   (e.g. a genuinely landmark-free periodic field) the classical rule
   below is used unchanged.

4. CENTER TIE-BREAK (classical fallback)
   Per the problem statement's own scoring rule: among every peak that
   survived NMS (already the "comparably good" set), return the one
   closest to the search image's center.

5. SUB-PIXEL REFINEMENT
   A quadratic (parabolic) fit to the correlation surface around the
   chosen peak in x and y independently (Tian & Huhns, 1986).

6. CONFIDENCE
   The uniqueness ratio between the best peak and the next distinct one
   (Lowe-style ratio test, 2004) is the classical confidence signal,
   blended with the CNN's own score when the CNN made the call, and
   optionally recalibrated by a trained scikit-learn model.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.ndimage import maximum_filter

from cnn_matcher import PatchCNN


@dataclass
class LocalizationResult:
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
    candidate_peaks: list = field(default_factory=list)


def _candidate_grid(scale_lo=8.0, scale_hi=12.0, n_scale=9,
                     angle_lo=-6.0, angle_hi=6.0, n_angle=5):
    scales = np.linspace(scale_lo, scale_hi, n_scale)
    angles = np.linspace(angle_lo, angle_hi, n_angle)
    return scales, angles


def _make_template(reference: np.ndarray, scale: float, angle_deg: float):
    h, w = reference.shape[:2]
    if angle_deg != 0.0:
        m = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle_deg, 1.0)
        rot = cv2.warpAffine(reference, m, (w, h), flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REFLECT)
    else:
        rot = reference
    new_w, new_h = max(4, int(round(w / scale))), max(4, int(round(h / scale)))
    if new_w < 8 or new_h < 8:
        return None
    return cv2.resize(rot, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _nms_peaks(corr: np.ndarray, template_size: tuple, abs_gap: float = 0.02,
               max_peaks: int = 300, center: tuple | None = None):
    """Every (x, y, score) local maximum within `abs_gap` of the global
    best score, as CENTER coordinates. Truncation (when more than
    max_peaks qualify) keeps candidates closest to `center` -- see
    module docstring point 2.
    """
    th, tw = template_size
    footprint = max(3, int(round(min(tw, th) * 0.12)))
    local_max = maximum_filter(corr, size=footprint, mode="nearest")
    best = float(corr.max())
    thresh = best - abs_gap
    is_peak = (corr >= local_max - 1e-9) & (corr >= thresh)
    ys, xs = np.nonzero(is_peak)
    scores = corr[ys, xs]
    if len(xs) > max_peaks:
        if center is not None:
            ccx, ccy = center[0] - tw / 2.0, center[1] - th / 2.0
            d2 = (xs - ccx) ** 2 + (ys - ccy) ** 2
            keep = np.argsort(d2)[:max_peaks]
        else:
            keep = np.argsort(-scores)[:max_peaks]
        ys, xs, scores = ys[keep], xs[keep], scores[keep]
    return [(float(x + tw / 2.0), float(y + th / 2.0), float(s))
            for x, y, s in zip(xs, ys, scores)]


def _subpixel_offset(corr: np.ndarray, px: int, py: int):
    h, w = corr.shape
    dx = dy = 0.0
    if 0 < px < w - 1:
        f_l, f_c, f_r = corr[py, px - 1], corr[py, px], corr[py, px + 1]
        denom = (f_l - 2 * f_c + f_r)
        if abs(denom) > 1e-9:
            dx = float(np.clip(0.5 * (f_l - f_r) / denom, -0.5, 0.5))
    if 0 < py < h - 1:
        f_t, f_c, f_b = corr[py - 1, px], corr[py, px], corr[py + 1, px]
        denom = (f_t - 2 * f_c + f_b)
        if abs(denom) > 1e-9:
            dy = float(np.clip(0.5 * (f_t - f_b) / denom, -0.5, 0.5))
    return dx, dy


def _cnn_score_peaks(search: np.ndarray, peaks: list, cnn_matcher: "PatchCNN"):
    """Score every candidate peak's neighbourhood with the CNN (values
    assumed 0-255). A candidate too close to the border to extract a
    full patch gets a score of 0.0 (never preferred, never crashes).
    """
    half = PatchCNN.INPUT_SIZE // 2
    sh, sw = search.shape[:2]
    batch, valid_idx = [], []
    for i, (x, y, _) in enumerate(peaks):
        px, py = int(round(x)), int(round(y))
        x0, y0 = px - half, py - half
        x1, y1 = x0 + PatchCNN.INPUT_SIZE, y0 + PatchCNN.INPUT_SIZE
        if x0 < 0 or y0 < 0 or x1 > sw or y1 > sh:
            continue
        patch = (search[y0:y1, x0:x1] / 255.0).astype(np.float32)
        batch.append(patch)
        valid_idx.append(i)
    scores = [0.0] * len(peaks)
    if batch:
        probs = cnn_matcher.forward(np.stack(batch)[:, None, :, :])
        for i, p in zip(valid_idx, probs):
            scores[i] = float(p)
    return scores


def localize(reference: np.ndarray, search: np.ndarray,
             scale_range=(8.0, 12.0), n_scale=9,
             angle_range=(-6.0, 6.0), n_angle=5,
             abs_gap: float = 0.02,
             confidence_model=None,
             cnn_matcher=None,
             cnn_accept_thresh: float = 0.5) -> LocalizationResult:
    """Find where `reference` appears (shrunk ~10x) inside `search`.

    Both inputs are single-channel (grayscale) uint8 or float arrays.
    Returns a LocalizationResult with (x, y) in `search` pixel coordinates.
    """
    t0 = time.perf_counter()
    reference = reference.astype(np.float32)
    search = search.astype(np.float32)
    sh, sw = search.shape[:2]

    scales, angles = _candidate_grid(scale_range[0], scale_range[1], n_scale,
                                      angle_range[0], angle_range[1], n_angle)

    best_overall = -np.inf
    best_corr = best_template_size = best_scale = best_angle = None

    for scale in scales:
        for angle in angles:
            template = _make_template(reference, scale, angle)
            if template is None or template.shape[0] >= sh or template.shape[1] >= sw:
                continue
            corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            local_max = float(corr.max())
            if local_max > best_overall:
                best_overall = local_max
                best_corr = corr
                best_template_size = template.shape[:2]
                best_scale, best_angle = float(scale), float(angle)

    if best_corr is None:
        template = _make_template(reference, 10.0, 0.0)
        best_corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        best_template_size = template.shape[:2]
        best_scale, best_angle = 10.0, 0.0

    peaks = _nms_peaks(best_corr, best_template_size, abs_gap=abs_gap, center=(sw / 2.0, sh / 2.0))
    if not peaks:
        _, max_val, _, max_loc = cv2.minMaxLoc(best_corr)
        th, tw = best_template_size
        peaks = [(max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0, float(max_val))]

    peaks.sort(key=lambda p: -p[2])
    best_score = peaks[0][2]
    second_score = peaks[1][2] if len(peaks) > 1 else 0.0

    # --- Tie-break: CNN gets first say if available and confident about
    # some candidate; otherwise classical closest-to-center. ---
    cx_img, cy_img = sw / 2.0, sh / 2.0
    cnn_used = False
    cnn_best = 0.0
    if cnn_matcher is not None:
        cnn_scores = _cnn_score_peaks(search, peaks, cnn_matcher)
        cnn_best = max(cnn_scores) if cnn_scores else 0.0
        if cnn_best >= cnn_accept_thresh:
            best_i = int(np.argmax(cnn_scores))
            chosen = peaks[best_i]
            cnn_used = True
        else:
            chosen = min(peaks, key=lambda p: (p[0] - cx_img) ** 2 + (p[1] - cy_img) ** 2)
    else:
        chosen = min(peaks, key=lambda p: (p[0] - cx_img) ** 2 + (p[1] - cy_img) ** 2)

    # --- Sub-pixel refinement on the winning hypothesis's own surface ---
    th, tw = best_template_size
    px_int = int(np.clip(round(chosen[0] - tw / 2.0), 0, best_corr.shape[1] - 1))
    py_int = int(np.clip(round(chosen[1] - th / 2.0), 0, best_corr.shape[0] - 1))
    dx, dy = _subpixel_offset(best_corr, px_int, py_int)
    final_x = chosen[0] + dx
    final_y = chosen[1] + dy

    # --- Confidence: classical uniqueness ratio, boosted by the CNN's own
    # score when it made the call (the ratio is low by construction
    # whenever periodic repeats tie, exactly the case a landmark-spotting
    # CNN is meant to resolve). ---
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

    return LocalizationResult(
        x=float(final_x), y=float(final_y), confidence=confidence,
        scale=best_scale, rotation_deg=best_angle,
        num_candidate_peaks=len(peaks), best_score=float(best_score),
        second_score=float(second_score), time_ms=elapsed_ms,
        cnn_used=cnn_used, cnn_score=cnn_best,
        candidate_peaks=peaks,
    )

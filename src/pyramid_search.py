"""
pyramid_search.py
-------------------
Coarse-to-fine replacement for localizer.py's flat 45-hypothesis
scale x rotation grid search.

Measured on this machine: the original single-resolution grid (9 scales
x 5 angles = 45 full-resolution cv2.matchTemplate calls against the full
1000x1000 search image) costs ~1.65s and is essentially the entire
runtime of one localize() call -- everything downstream (CNN scoring,
lattice enumeration) is comparatively free. A 4x-downsampled version of
the exact same call costs ~4.7x less per hypothesis (measured: 36.7ms ->
7.9ms). That gap is the lever this module uses.

Design (two stages, both searching the WHOLE search image -- see note
below on why neither stage crops):
  Stage 1 (coarse): the same 9x5 grid, run at 4x downsampling. Cheap
    (~0.35s for all 45 hypotheses). Identifies which (scale, angle)
    neighbourhood is promising, not a final answer.
  Stage 2 (fine): a small grid (default 5x5 = 25 hypotheses, tunable)
    spanning +/-1 coarse grid-step around stage 1's winner, but at FULL
    resolution AND at up to 2x finer step size than the original coarse
    grid (0.25 scale-step vs 0.5, 1.5 degree-step vs 3.0). This is where
    the final answer comes from.

Net effect (see docs/validation_report_v3.md for the actual measured
numbers): fewer expensive full-resolution correlations than the
original flat grid, AND finer final scale/rotation resolution in the
region that matters -- both directions the brute-force grid could only
trade off against each other.

IMPORTANT: stage 2 must still correlate against the FULL search image,
not a cropped region around stage 1's approximate location. It's
tempting to crop for extra speed, but localizer_v2.py's downstream
logic (ambiguity gating via the ratio between the best and second-best
peak, and lattice-vector estimation via autocorrelation of the whole
correlation surface) both need visibility into every competing peak
across the entire image -- cropping would make a genuinely ambiguous,
periodic field look falsely unique. The speedup here comes entirely
from fewer/cheaper hypotheses, never from a smaller search area.
"""
from __future__ import annotations

import cv2
import numpy as np

from localizer import _make_template


def pyramid_scale_rotation_search(reference: np.ndarray, search: np.ndarray,
                                   scale_range=(8.0, 12.0), n_scale_coarse=9,
                                   angle_range=(-6.0, 6.0), n_angle_coarse=5,
                                   n_scale_fine=5, n_angle_fine=5,
                                   downsample=4):
    """Returns (best_corr, best_template_size, best_scale, best_angle),
    exactly like localizer.localize()'s internal coarse search -- this
    is a drop-in replacement for that one step, nothing downstream of it
    changes.
    """
    sh, sw = search.shape[:2]
    coarse_scales = np.linspace(scale_range[0], scale_range[1], n_scale_coarse)
    coarse_angles = np.linspace(angle_range[0], angle_range[1], n_angle_coarse)

    small_h, small_w = max(8, sh // downsample), max(8, sw // downsample)
    search_small = cv2.resize(search, (small_w, small_h), interpolation=cv2.INTER_AREA)

    best_val = -np.inf
    best_scale = best_angle = None
    for scale in coarse_scales:
        template = _make_template(reference, scale, 0.0)
        if template is None:
            continue
        for angle in coarse_angles:
            t = _make_template(reference, scale, angle) if angle != 0.0 else template
            if t is None:
                continue
            tw_small = max(4, round(t.shape[1] / downsample))
            th_small = max(4, round(t.shape[0] / downsample))
            if th_small >= small_h or tw_small >= small_w:
                continue
            t_small = cv2.resize(t, (tw_small, th_small), interpolation=cv2.INTER_AREA)
            corr = cv2.matchTemplate(search_small, t_small, cv2.TM_CCOEFF_NORMED)
            v = float(corr.max())
            if v > best_val:
                best_val = v
                best_scale, best_angle = float(scale), float(angle)

    if best_scale is None:
        best_scale, best_angle = float(np.mean(scale_range)), 0.0

    # Stage 2: narrow, finer, full-resolution -- centred on the coarse winner.
    coarse_scale_step = (scale_range[1] - scale_range[0]) / max(1, n_scale_coarse - 1)
    coarse_angle_step = (angle_range[1] - angle_range[0]) / max(1, n_angle_coarse - 1)
    fine_scales = np.linspace(best_scale - coarse_scale_step, best_scale + coarse_scale_step, n_scale_fine)
    fine_angles = np.linspace(best_angle - coarse_angle_step, best_angle + coarse_angle_step, n_angle_fine)
    fine_scales = np.clip(fine_scales, scale_range[0] - coarse_scale_step, scale_range[1] + coarse_scale_step)
    fine_angles = np.clip(fine_angles, angle_range[0] - coarse_angle_step, angle_range[1] + coarse_angle_step)

    best_overall = -np.inf
    best_corr = best_template_size = None
    final_scale = final_angle = None
    for scale in fine_scales:
        for angle in fine_angles:
            template = _make_template(reference, scale, angle)
            if template is None or template.shape[0] >= sh or template.shape[1] >= sw:
                continue
            corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
            v = float(corr.max())
            if v > best_overall:
                best_overall = v
                best_corr = corr
                best_template_size = template.shape[:2]
                final_scale, final_angle = float(scale), float(angle)

    if best_corr is None:
        # extreme edge case (e.g. reference larger than search at every
        # tried scale) -- fall back to a single centred hypothesis so the
        # caller always gets a usable result.
        template = _make_template(reference, float(np.mean(scale_range)), 0.0)
        best_corr = cv2.matchTemplate(search, template, cv2.TM_CCOEFF_NORMED)
        best_template_size = template.shape[:2]
        final_scale, final_angle = float(np.mean(scale_range)), 0.0

    return best_corr, best_template_size, final_scale, final_angle

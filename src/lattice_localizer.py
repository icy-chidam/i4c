"""
lattice_localizer.py
---------------------
Prototype improvement over localizer.py's classical tie-break.

DIAGNOSIS (verified empirically against the shipped repo's own pure-periodic
test samples -- see accompanying analysis): the classical NCC pipeline's
*match quality* is already excellent even on purely periodic, landmark-free
fields -- predicted positions land within 1-4px of an exact periodic repeat
of the true site in every diagnosed case, often sub-pixel. The large
reported errors (100+ px) are not a matching failure; they are a
*candidate-selection* failure: NMS peak-picking with a fixed absolute-score
threshold and a max_peaks=300 cap cannot reliably enumerate "every valid
periodic repeat" when a ~7-12px period tiles a 1000px search image into
thousands of geometrically valid sites -- many legitimately-periodic sites
score just under the NMS threshold purely from noise and get silently
dropped, so "closest to center among NMS survivors" is picking from an
incomplete, noise-biased sample of the true candidate set.

FIX: once we have ONE reliable anchor match (the classical pipeline's own
best peak, already accurate to its own local cell), measure the pattern's
true periodicity directly and exactly via the 2D autocorrelation of the
*entire* correlation surface (FFT-based) -- this aggregates evidence over
every peak-to-peak spacing in the whole image at once, so it is far more
noise-robust than trusting any single peak's survival. With exact period
vectors in hand, the full lattice of valid candidate centers within the
search image can be enumerated analytically (simple arithmetic, no
threshold, no cap, nothing can be silently missed), and the tie-break rule
("closest to the search image's centre") can be applied EXACTLY over the
complete candidate set instead of an incomplete one.

This is a classical Fourier/lattice-geometry technique, not a CNN --
included specifically to demonstrate that the periodic-ambiguity failure
mode is best addressed with signal-processing/geometry, not more learned
capacity (a CNN patch classifier, by construction, can only ever tell you
"does this look like the landmark" -- useless when there IS no landmark,
which is exactly the regime this module targets).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import maximum_filter

from localizer import _candidate_grid, _make_template, _subpixel_offset  # reuse, don't duplicate
import cv2


def estimate_lattice_vectors(corr: np.ndarray, min_peak_frac: float = 0.12,
                              search_radius_frac: float = 0.45, min_len_px: float = 3.0):
    """Estimate the two shortest independent periodicity vectors present in
    `corr` via its own 2D autocorrelation (FFT-based, zero-padded to avoid
    circular wrap contaminating short lags). Returns (v1, v2) as (dx, dy)
    float pairs in the SAME pixel units as `corr`, or None if no clear
    periodicity is found (e.g. a genuinely aperiodic / single-landmark-only
    correlation surface -- the caller should fall back to the classical
    rule in that case).
    """
    h, w = corr.shape
    c = (corr - corr.mean()).astype(np.float64)
    H, W = 2 * h, 2 * w
    F = np.fft.rfft2(c, s=(H, W))
    power = (F * np.conj(F)).real
    ac = np.fft.irfft2(power, s=(H, W))
    ac = np.fft.fftshift(ac)
    cy, cx = H // 2, W // 2

    R = max(8, int(min(h, w) * search_radius_frac))
    y0, y1 = max(0, cy - R), min(H, cy + R + 1)
    x0, x1 = max(0, cx - R), min(W, cx + R + 1)
    window = ac[y0:y1, x0:x1]
    wcy, wcx = cy - y0, cx - x0

    local_max = maximum_filter(window, size=5, mode="nearest") == window
    peak_val = window.max()
    mask = local_max & (window > min_peak_frac * peak_val)
    ys, xs = np.nonzero(mask)

    cand = []
    for y, x in zip(ys, xs):
        dy, dx = float(y - wcy), float(x - wcx)
        r = float(np.hypot(dx, dy))
        if r < min_len_px:
            continue
        cand.append((dx, dy, r, float(window[y, x])))
    if len(cand) < 2:
        return None

    cand.sort(key=lambda t: t[2])
    v1 = np.array(cand[0][:2], dtype=float)

    v2 = None
    for dx, dy, r, val in cand[1:]:
        v = np.array([dx, dy], dtype=float)
        n1, n2 = np.linalg.norm(v), np.linalg.norm(v1)
        if n1 < 1e-6 or n2 < 1e-6:
            continue
        cosang = abs(float(np.dot(v, v1) / (n1 * n2)))
        if cosang < 0.90:
            v2 = v
            break
    if v2 is None:
        return None
    return v1, v2


def enumerate_lattice_closest_to_center(anchor_xy, v1, v2, bounds, center_xy, max_k: int = 200):
    """Enumerate every lattice site anchor + k1*v1 + k2*v2 that lands fully
    inside `bounds` = (x0, y0, x1, y1), and return the one closest to
    `center_xy`. Fully vectorized (no threshold, no candidate cap -- the
    entire lattice within bounds is considered).
    """
    ax, ay = anchor_xy
    x0, y0, x1, y1 = bounds
    cx, cy = center_xy

    k = np.arange(-max_k, max_k + 1)
    K1, K2 = np.meshgrid(k, k)
    X = ax + K1 * v1[0] + K2 * v2[0]
    Y = ay + K1 * v1[1] + K2 * v2[1]
    inb = (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    if not np.any(inb):
        return anchor_xy
    Xc, Yc = X[inb], Y[inb]
    d2 = (Xc - cx) ** 2 + (Yc - cy) ** 2
    i = int(np.argmin(d2))
    return float(Xc[i]), float(Yc[i])


def localize_lattice(reference: np.ndarray, search: np.ndarray,
                      scale_range=(8.0, 12.0), n_scale=9,
                      angle_range=(-6.0, 6.0), n_angle=5):
    """Same coarse scale/rotation search as localizer.localize(), but the
    tie-break among periodic repeats is done by exact lattice enumeration
    instead of NMS-peak survival. Returns (x, y, diagnostics_dict).
    """
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

    th, tw = best_template_size
    _, max_val, _, max_loc = cv2.minMaxLoc(best_corr)
    anchor_x, anchor_y = max_loc[0] + tw / 2.0, max_loc[1] + th / 2.0
    dx, dy = _subpixel_offset(best_corr, max_loc[0], max_loc[1])
    anchor_x, anchor_y = anchor_x + dx, anchor_y + dy

    lattice = estimate_lattice_vectors(best_corr)
    cx_img, cy_img = sw / 2.0, sh / 2.0
    bounds = (tw / 2.0, th / 2.0, sw - tw / 2.0, sh - th / 2.0)

    if lattice is None:
        final_x, final_y = anchor_x, anchor_y
        method = "no_lattice_found_used_anchor"
    else:
        v1, v2 = lattice
        final_x, final_y = enumerate_lattice_closest_to_center(
            (anchor_x, anchor_y), v1, v2, bounds, (cx_img, cy_img))
        method = "lattice_enumeration"

    return dict(x=final_x, y=final_y, scale=best_scale, rotation_deg=best_angle,
                anchor_x=anchor_x, anchor_y=anchor_y, method=method,
                lattice=None if lattice is None else (lattice[0].tolist(), lattice[1].tolist()),
                best_score=float(max_val))

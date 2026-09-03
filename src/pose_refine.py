"""
pose_refine.py
----------------
NEW for Phase 2. Continuous local refinement of (x, y, scale, rotation)
around an already-CHOSEN candidate site.

Why this module has to exist: v1/v2/v3 only ever report the coarse (or
coarse-to-fine grid's) scale/angle HYPOTHESIS itself -- fine for Phase 1
(rotation/scale were never scored, only x/y), nowhere near Phase 2's
tolerances (scale within 1%, rotation within 0.25 degree for full pose
credit -- see the addendum's credit-tier table). A discrete grid,
however fine, can only ever land exactly on a grid point; Phase 2 needs
a continuous answer, so this is a genuinely new capability, not a
retune of an existing one.

Why doing this with a LOCAL crop is not the mistake pyramid_search.py's
own docstring warns against: that warning ("must still correlate
against the FULL search image, not a cropped region") is about
resolving WHICH periodic repeat is the correct site -- a decision that
needs visibility into every competing peak across the whole image, or a
genuinely ambiguous field looks falsely unique. By the time this module
runs, localizer_v4 has ALREADY made that decision (via the same
CNN-landmark / ambiguity-gated lattice-enumeration tie-break as v2/v3,
completely unchanged). This step only polishes the pose AT the site
already chosen -- a purely local question, since scale and rotation are
properties of the whole capture (one stage tilt, one magnification
error) rather than of any individual site, so refining around any one
correctly-identified site recovers the same true (scale, angle) as any
other. Cropping here trades away nothing the global step needed.

Method: bounded Powell search (scipy.optimize, already a project
dependency via lattice_localizer.py's SciPy use) over (scale, angle),
scoring each proposal by the PEAK normalized cross-correlation
obtainable in a small local crop around the anchor -- matchTemplate's
own correlation surface already maximizes over translation at every
proposal, so (x, y) doesn't need to be a separate optimized parameter
here. Once (scale, angle) are set, one final local correlation surface
gives sub-pixel (x, y) via the same parabolic fit v1/v2/v3 already use
(Tian & Huhns, 1986) -- no new positional method, just applied at a
continuous rather than grid-locked pose.
"""
from __future__ import annotations

import numpy as np
import cv2
from scipy.optimize import minimize

from localizer import _make_template, _subpixel_offset


def make_template_continuous(reference: np.ndarray, scale: float, angle_deg: float, out_size: int,
                              prefiltered: dict | None = None) -> np.ndarray:
    """Build a template of EXACTLY `out_size` x `out_size` pixels via a
    single rotate+scale affine sample of `reference`, with `scale`
    acting as a continuous parameter -- unlike localizer._make_template,
    which rotates via warpAffine and then resizes to
    `round(reference_size / scale)`, an INTEGER pixel count.

    That integer rounding is invisible for Phase 1 (scale/rotation were
    never scored, only x/y), but it quantizes the achievable scale
    resolution to roughly `ref_size / template_px**2` -- for this
    project's 1000px reference and an 8-12x template (83-125px), that
    is ~0.06-0.15, the SAME ORDER as Phase 2's 1%-tolerance requirement
    (~0.08-0.12 absolute at scale 8-12). Confirmed empirically, not
    assumed: sweeping scale at the true angle produces a visible
    staircase whose plateau boundaries land exactly on integer values of
    round(1000/scale) (see docs/validation_report_v4.md, "Why a second
    template builder"). A discrete-output-size template genuinely cannot
    resolve finer than that, no matter how good the optimizer sitting on
    top of it is -- so `refine_pose` needs an objective that can
    actually SEE sub-quantum scale changes, which means holding the
    output canvas size FIXED and letting the sampling density vary
    continuously with `scale` instead.

    Anti-aliasing: sampling an ~8-12x downscale with plain bilinear
    interpolation (no pre-filtering) skips most of the source footprint
    each output pixel should be averaging over, which both loses real
    signal and can manufacture its own high-frequency noise -- so the
    source is Gaussian-pre-blurred (sigma scaled to the CURRENT `scale`)
    before the affine sample, standard mipmap-style prefiltering.
    `prefiltered`, if given, is a small cache dict this function fills in
    on first use (keyed by a rounded sigma) so repeated calls across a
    single refine_pose optimization (where scale moves only slightly
    within a bounded box) don't reblur the full reference at every
    evaluation.

    `out_size` should be held constant across an entire refine_pose call
    -- it is the fixed canvas this function samples into; only the
    geometry sampled from `reference` changes as `scale`/`angle_deg`
    vary. Rotation sign/handedness matches localizer._make_template's
    (confirmed by direct comparison at non-quantized scales in
    docs/validation_report_v4.md) -- this is a continuous-scale
    REPLACEMENT of the same geometric operation, not a different
    convention.
    """
    sigma = max(0.0, float(scale) * 0.5)
    key = round(sigma, 2)
    if prefiltered is not None and key in prefiltered:
        src = prefiltered[key]
    else:
        src = cv2.GaussianBlur(reference.astype(np.float32), (0, 0), sigmaX=sigma) if sigma >= 0.15 \
            else reference.astype(np.float32)
        if prefiltered is not None:
            prefiltered[key] = src

    h, w = reference.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    ocx, ocy = out_size / 2.0, out_size / 2.0
    theta = np.deg2rad(angle_deg)
    c, s = np.cos(theta), np.sin(theta)
    a, b = scale * c, -scale * s
    d, e = scale * s, scale * c
    tx = cx - a * ocx - b * ocy
    ty = cy - d * ocx - e * ocy
    M = np.array([[a, b, tx], [d, e, ty]], dtype=np.float64)
    return cv2.warpAffine(src, M, (out_size, out_size),
                           flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
                           borderMode=cv2.BORDER_REFLECT)


def _local_crop(search: np.ndarray, cx: float, cy: float, half_size: float):
    h, w = search.shape[:2]
    x0 = int(max(0, round(cx - half_size)))
    y0 = int(max(0, round(cy - half_size)))
    x1 = int(min(w, round(cx + half_size)))
    y1 = int(min(h, round(cy + half_size)))
    return search[y0:y1, x0:x1], x0, y0


def _peak_at_pose(reference: np.ndarray, crop: np.ndarray, scale: float, angle: float,
                   out_size: int, prefiltered: dict, expected_xy: tuple[float, float] | None = None,
                   window_radius: float = 4.0):
    """Best NCC achievable at (scale, angle) inside `crop`, using the
    FIXED-canvas continuous template builder (see make_template_continuous)
    so the objective `refine_pose` optimizes is smooth in (scale, angle)
    rather than staircased. Returns (peak_value, corr_surface_or_None,
    max_loc_or_None). Never raises -- a degenerate case just scores -1.0
    so the optimizer walks away from it instead of crashing.

    `expected_xy`, if given, restricts the argmax search to a small
    `window_radius`-px box around that crop-local position instead of
    cv2.minMaxLoc's GLOBAL max over the whole crop. This matters for
    small-period patterns: the crop has to be sized off the TEMPLATE
    (which can be much larger than the period at scale~8-12), so for a
    ~10-15px period the crop can easily span a dozen-plus repeats of it
    -- an unconstrained global max lets a neighbouring repeat's peak
    (higher purely from noise) silently steal the match, undoing
    whichever site localizer_v4's lattice tie-break deliberately chose.
    Confirmed empirically, not theoretically: an early version without
    this constraint was measured landing EXACTLY an integer number of
    periods away from the intended site (see
    docs/validation_report_v4.md, "The refine_pose regression"). This
    step is meant to polish pose AT an already-chosen site (see this
    module's docstring), never to re-litigate which site that is.
    """
    if out_size >= crop.shape[0] or out_size >= crop.shape[1] or out_size < 4:
        return -1.0, None, None
    t = make_template_continuous(reference, scale, angle, out_size, prefiltered=prefiltered)
    corr = cv2.matchTemplate(crop, t.astype(np.float32), cv2.TM_CCOEFF_NORMED)

    if expected_xy is not None:
        ecx, ecy = expected_xy[0] - out_size / 2.0, expected_xy[1] - out_size / 2.0
        r = window_radius
        x0 = int(np.clip(np.floor(ecx - r), 0, corr.shape[1] - 1))
        x1 = int(np.clip(np.ceil(ecx + r) + 1, x0 + 1, corr.shape[1]))
        y0 = int(np.clip(np.floor(ecy - r), 0, corr.shape[0] - 1))
        y1 = int(np.clip(np.ceil(ecy + r) + 1, y0 + 1, corr.shape[0]))
        sub = corr[y0:y1, x0:x1]
        if sub.size == 0:
            return -1.0, None, None
        _, max_val, _, sub_loc = cv2.minMaxLoc(sub)
        max_loc = (sub_loc[0] + x0, sub_loc[1] + y0)
    else:
        _, max_val, _, max_loc = cv2.minMaxLoc(corr)

    if not np.isfinite(max_val):
        return -1.0, None, None
    return float(max_val), corr, max_loc


def refine_pose(reference: np.ndarray, search: np.ndarray,
                 anchor_x: float, anchor_y: float, scale0: float, angle0: float,
                 scale_step: float = 0.5, angle_step: float = 1.5,
                 scale_bounds: tuple[float, float] | None = None,
                 angle_bounds: tuple[float, float] | None = None,
                 crop_radius_factor: float = 1.6, max_iter: int = 80) -> dict:
    """Locally refine (scale, angle, x, y) around (anchor_x, anchor_y,
    scale0, angle0). `scale_step`/`angle_step` should be the STEP of
    whatever grid stage produced (scale0, angle0) -- refinement here is
    bounded to +/- that step either side (the grid already localized the
    right neighbourhood; this is about resolving WITHIN it, not
    re-searching globally, so a tight box both keeps this fast and stops
    it from drifting onto some other periodic repeat's pose).
    `scale_bounds`/`angle_bounds`, if given, additionally hard-clip the
    box to the disclosed Phase 2 range -- so refinement can never wander
    outside [8,12] / +/-5deg even if the local optimum would sit just
    past the edge due to noise, and so a starting point the grid stage
    pushed slightly outside those bounds (its own clipping is
    deliberately a bit wider, see pyramid_search.py) is pulled back
    in-bounds before this ever calls the optimizer.

    Returns dict(x, y, scale, angle, peak_score, improved). `improved`
    is False if refinement could not do at least as well as the
    starting grid point (rare -- degenerate crop, usually near an image
    border) in which case (x, y, scale, angle) fall back to the
    UNREFINED anchor/grid values and peak_score is the original grid
    peak, so the caller always gets a usable, never-worse answer.
    """
    reference = reference.astype(np.float32)
    search = search.astype(np.float32)

    out_size = max(8, int(round(reference.shape[0] / scale0)))
    half = out_size * crop_radius_factor
    crop, x0, y0 = _local_crop(search, anchor_x, anchor_y, half)
    prefiltered: dict = {}
    expected_xy = (anchor_x - x0, anchor_y - y0)  # anchor's position WITHIN the crop -- held
                                                     # fixed throughout; see _peak_at_pose's
                                                     # docstring for why this must not float free.

    # Clip the starting point into bounds FIRST -- pyramid_search's own
    # clipping is deliberately wider than the disclosed range (it lets
    # the fine grid explore slightly past its own coarse edge), so
    # (scale0, angle0) can arrive here a hair outside [8,12]/+/-5deg.
    # Powell requires x0 inside its bounds, and clipping the objective
    # value's own starting point is more predictable than relying on
    # whatever an out-of-bounds Powell call happens to do.
    if scale_bounds is not None:
        scale0 = float(np.clip(scale0, scale_bounds[0], scale_bounds[1]))
    if angle_bounds is not None:
        angle0 = float(np.clip(angle0, angle_bounds[0], angle_bounds[1]))

    base_val, base_corr, base_loc = _peak_at_pose(reference, crop, scale0, angle0, out_size,
                                                     prefiltered, expected_xy=expected_xy)
    if base_corr is None:
        return dict(x=float(anchor_x), y=float(anchor_y), scale=float(scale0),
                    angle=float(angle0), peak_score=-1.0, improved=False)

    lo_s, hi_s = scale0 - scale_step, scale0 + scale_step
    lo_a, hi_a = angle0 - angle_step, angle0 + angle_step
    if scale_bounds is not None:
        lo_s, hi_s = max(lo_s, scale_bounds[0]), min(hi_s, scale_bounds[1])
    if angle_bounds is not None:
        lo_a, hi_a = max(lo_a, angle_bounds[0]), min(hi_a, angle_bounds[1])

    def objective(params):
        s, a = params
        val, _, _ = _peak_at_pose(reference, crop, s, a, out_size, prefiltered, expected_xy=expected_xy)
        return -val

    best_scale, best_angle, best_val = scale0, angle0, base_val
    try:
        result = minimize(objective, x0=np.array([scale0, angle0]), method="Powell",
                           bounds=[(lo_s, hi_s), (lo_a, hi_a)],
                           options=dict(maxiter=max_iter, xtol=1e-4, ftol=1e-6))
        cand_scale, cand_angle = float(result.x[0]), float(result.x[1])
        cand_val, cand_corr, cand_loc = _peak_at_pose(reference, crop, cand_scale, cand_angle,
                                                         out_size, prefiltered, expected_xy=expected_xy)
        if cand_corr is not None and cand_val >= base_val:
            best_scale, best_angle, best_val = cand_scale, cand_angle, cand_val
    except Exception:
        pass  # Powell failed for any reason -> keep the grid point, never crash the pair.

    peak_val, corr, max_loc = _peak_at_pose(reference, crop, best_scale, best_angle, out_size,
                                               prefiltered, expected_xy=expected_xy)
    if corr is None:
        peak_val, corr, max_loc = base_val, base_corr, base_loc
        best_scale, best_angle = scale0, angle0

    dx, dy = _subpixel_offset(corr, max_loc[0], max_loc[1])
    final_x = x0 + max_loc[0] + out_size / 2.0 + dx
    final_y = y0 + max_loc[1] + out_size / 2.0 + dy

    return dict(x=float(final_x), y=float(final_y), scale=float(best_scale),
                angle=float(best_angle), peak_score=float(peak_val),
                improved=bool(peak_val > base_val + 1e-9))

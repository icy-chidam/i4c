"""
spec_fair.py
-------------
Phase 2 version of evaluate.py's _spec_fair_target: the nearest
periodic-equivalent of a landmark-free true site to the search image's
centre -- the fair target implied by the addendum's own tie-break rule
("What if two regions match equally well? ... the Phase 1
nearest-to-centre rule still decides"). Only meaningful for samples with
NO fiducial (has_landmark=False and found=True); a landmark makes the
true site uniquely correct, and an absent (found=False) pair has no
target at all.

Two corrections versus evaluate.py's original, both checked empirically
rather than assumed (see docs/validation_report_v4.md, "Why a new
spec-fair function" for the measurements):

  1. NO rotation correction. The original rotates the (k*period_x,
     k*period_y) offsets by the sample's rotation_deg. But the SEARCH
     image in this project's capture model is always rendered at
     theta_deg=0.0 (only the reference ever carries a capture rotation)
     -- so periodic offsets AS THEY APPEAR IN SEARCH-PIXEL COORDINATES
     have no rotation to correct for. Confirmed directly: comparing our
     pipeline's actual output against both versions of the target across
     many samples, the unrotated version lands within ~0.1-0.3px, while
     the rotated version is measurably worse. Applying rotation_deg here
     was silently miscorrecting an axis-aligned lattice that was never
     rotated in search-pixel space to begin with.

  2. WIDE k-range. The original's k in [-8, 8] only reaches ~8 periods
     from the true site -- fine at Phase 1's period sizes, but Phase 2's
     8-12x scale range (vs. Phase 1's ~10x) pushes some periods down to
     a handful of search-pixels, where the true nearest-to-centre site
     can be 15-25+ periods away. A k-range derived from the search
     image size and the period itself (plus a safety margin) is used
     instead, so the true nearest-to-centre site is never out of reach.
"""
from __future__ import annotations

import numpy as np


def spec_fair_target(tx: float, ty: float, period_x: float, period_y: float,
                      search_size: int = 1000, cx: float | None = None,
                      cy: float | None = None, k_margin: int = 4):
    """Nearest periodic-equivalent of (tx, ty) to the search image's
    centre, in SEARCH-PIXEL units (period_x/period_y already divided by
    scale_factor by the caller). See module docstring for why this
    applies NO rotation correction and uses an adaptive k-range.
    """
    cx = search_size / 2.0 if cx is None else cx
    cy = search_size / 2.0 if cy is None else cy
    if period_x <= 1e-6 or period_y <= 1e-6:
        return tx, ty

    kx_max = int(np.ceil(search_size / period_x)) + k_margin
    ky_max = int(np.ceil(search_size / period_y)) + k_margin
    kx = np.arange(-kx_max, kx_max + 1)
    ky = np.arange(-ky_max, ky_max + 1)
    KX, KY = np.meshgrid(kx, ky)
    X = tx + KX * period_x
    Y = ty + KY * period_y
    inb = (X >= 0) & (X <= search_size) & (Y >= 0) & (Y <= search_size)
    if not np.any(inb):
        return tx, ty
    Xc, Yc = X[inb], Y[inb]
    d2 = (Xc - cx) ** 2 + (Yc - cy) ** 2
    i = int(np.argmin(d2))
    return float(Xc[i]), float(Yc[i])


def localization_target(gt: dict, search_size: int = 1000):
    """The single target `res.x, res.y` should be compared against for a
    Phase 2 pair: the true site directly if there's a fiducial (unique,
    no ambiguity) or the pair is absent (undefined -- returns the raw
    x/y, which will be NaN, so callers naturally get NaN error rather
    than a silently wrong number); the spec-fair nearest-to-centre
    target otherwise.
    """
    if not gt.get("found", True) or gt.get("has_landmark", False):
        return gt["x"], gt["y"]
    scale = gt.get("scale_factor", 10.0) or 10.0
    period_x_s = gt.get("period_x", 0.0) / scale
    period_y_s = gt.get("period_y", 0.0) / scale
    return spec_fair_target(gt["x"], gt["y"], period_x_s, period_y_s, search_size=search_size)

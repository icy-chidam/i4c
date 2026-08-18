"""
pattern_render.py
------------------
Analytic, continuous-coordinate renderer for periodic semiconductor die
layouts, plus the SEM-realistic degradation models (independent sensor
noise, edge brightening, blur) required by the Drift-Sense synthetic
dataset. Three layout styles:

  * dram          -- orthogonal word-line/bit-line grid + via contacts.
  * dram_arcuate  -- a second, more realistic DRAM style grounded in an
                     actual COB-DRAM layout patent (arcuate moats, wavy
                     bit lines; see docs/references.md, EP0780901A2).
                     Added after cross-checking the official problem
                     statement's own example images, which use this
                     diagonal/non-orthogonal style rather than a plain
                     grid.
  * finfet        -- dense parallel fins + periodic gate bars.

Design idea (why this file looks the way it does)
===================================================
Instead of rendering one giant high-resolution canvas and cropping /
downsampling pieces out of it, every image is produced by evaluating a
single continuous function `intensity = pattern(world_x, world_y)`
directly at whatever pixel grid a given "capture" needs. A capture is
just three numbers: where its top-left corner sits in WORLD coordinates,
how many world-units one of its pixels spans (`scale`), and an optional
rotation of its sampling grid relative to the canonical (world-aligned)
frame.

    * The Reference image is a capture at scale = ``scale_ref`` (fine,
      "100x"-like sampling) and a small random rotation (models the stage
      not being perfectly re-aligned between visits).
    * The Search image is a capture at scale = ``scale_ref * K`` with
      K ~ 10 (the "10x lower magnification" in the problem statement),
      axis aligned.

Because both are just samples of the *same* underlying periodic function,
the ground-truth location of the reference inside the search image is
exact by construction, and there is no upper limit on how much data can
be produced.

Anti-aliasing: a naive point-sample of a fine periodic function at a
coarse pixel pitch would alias badly (Moire fringes). We avoid this two
ways: (1) the smoothing width of every edge is expressed in WORLD units
and scaled by the capture's own `scale`, so apparent sharpness in OUTPUT
pixels stays roughly constant regardless of magnification, and (2) every
capture is rendered at `supersample`x linear resolution and area-averaged
back down (cv2.INTER_AREA).
"""
from __future__ import annotations

import numpy as np
import cv2


# ----------------------------------------------------------------------
# Low-level smooth building blocks
# ----------------------------------------------------------------------

def _smooth_edge(d: np.ndarray, half_width: float, softness: float) -> np.ndarray:
    """Smooth step: ~1 where d << half_width, ~0 where d >> half_width."""
    softness = max(float(softness), 1e-3)
    z = np.clip((d - half_width) / softness, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(z))


def _wrap_dist(v: np.ndarray, period: float) -> np.ndarray:
    """Distance from v to the nearest multiple of `period` (>=0)."""
    return np.abs(((v + period / 2.0) % period) - period / 2.0)


def _rotate(u: np.ndarray, v: np.ndarray, theta_deg: float):
    if theta_deg == 0:
        return u, v
    t = np.deg2rad(theta_deg)
    c, s = np.cos(t), np.sin(t)
    return u * c - v * s, u * s + v * c


# ----------------------------------------------------------------------
# Alignment fiducial -- the disambiguating landmark a subset of samples
# get (see default_params); a filled square, large enough (radius on the
# order of one pattern period) to dominate whole-template correlation by
# a wide, consistent margin. A single missing via/fin and a "box-in-box"
# bright+dark design were both tried first and performed worse -- see
# docs/references.md "Design attempts that did not work".
# ----------------------------------------------------------------------

def _fiducial_overlay(wx: np.ndarray, wy: np.ndarray, p: dict, softness: float):
    if not p.get("defect"):
        z = np.zeros_like(wx)
        return z, z
    fx, fy = p["defect_xy"]
    r = p["defect_radius"]
    d = np.maximum(np.abs(wx - fx), np.abs(wy - fy))
    bright = _smooth_edge(d, r, softness)
    return bright, np.zeros_like(wx)


# ----------------------------------------------------------------------
# The three die-architecture styles
# ----------------------------------------------------------------------

def dram_intensity(wx: np.ndarray, wy: np.ndarray, p: dict, softness: float) -> np.ndarray:
    """Periodic word-line / bit-line grid with a via/contact dot at every
    intersection. All geometry is periodic in both axes, which is exactly
    what makes this style prone to the ambiguous, repeating-pattern
    failure mode Applied Materials describes.
    """
    dx = _wrap_dist(wx - p["phase_x"], p["period_x"])
    dy = _wrap_dist(wy - p["phase_y"], p["period_y"])

    line_v = _smooth_edge(dx, p["period_x"] * p["linewidth_frac"] / 2, softness)
    line_h = _smooth_edge(dy, p["period_y"] * p["linewidth_frac"] / 2, softness)
    lines = np.maximum(line_v, line_h) * 0.72

    via_r = min(p["period_x"], p["period_y"]) * p["via_frac"]
    via = _smooth_edge(np.sqrt(dx ** 2 + dy ** 2), via_r, softness) * 0.95

    base = np.clip(np.maximum(lines, via), 0.0, 1.0)
    bright, _ = _fiducial_overlay(wx, wy, p, softness)
    return base * (1 - bright) + 1.0 * bright


def dram_arcuate_intensity(wx: np.ndarray, wy: np.ndarray, p: dict, softness: float) -> np.ndarray:
    """DRAM 'arcuate moat / wavy bit line' layout style, grounded in an
    actual COB-DRAM layout patent (arcuate capacitor-under-bit-line
    moats connected by bit lines laid out in a wavy, crest-and-trough
    half-pitch pattern; see docs/references.md, EP0780901A2).
    Approximated as two families of diagonal capsule ("stadium") shapes
    on a herringbone offset -- captures the visual/statistical character
    (non-orthogonal periodicity, elongated rounded features) rather than
    the literal patent geometry, documented as an approximation.
    """
    def capsule_field(x, y, angle_deg, phase_u, phase_v):
        t = np.deg2rad(angle_deg)
        c, s = np.cos(t), np.sin(t)
        u = x * c - y * s
        v = x * s + y * c
        du = _wrap_dist(u - phase_u, p["period_u"])
        dv = _wrap_dist(v - phase_v, p["period_v"])
        dx = np.maximum(du - p["half_len"], 0.0)
        d = np.sqrt(dx ** 2 + dv ** 2)
        return _smooth_edge(d, p["radius"], softness)

    fieldA = capsule_field(wx, wy, p["tilt_deg"], p["phase_x"], p["phase_y"])
    fieldB = capsule_field(wx, wy, -p["tilt_deg"], p["phase_x"], p["phase_y"] + p["period_v"] / 2)
    base = np.maximum(fieldA, fieldB) * 0.62

    bright, _ = _fiducial_overlay(wx, wy, p, softness)
    return np.clip(base * (1 - bright) + 1.0 * bright, 0.0, 1.0)


def finfet_intensity(wx: np.ndarray, wy: np.ndarray, p: dict, softness: float) -> np.ndarray:
    """Dense parallel fin lines crossed by periodic horizontal gate bars.
    Gate bars repeat at a coarser vertical period than the fins do (many
    fins per gate pitch, as in real device layouts) -- again a source of
    periodic ambiguity.
    """
    dx = _wrap_dist(wx - p["phase_x"], p["period_x"])
    fins = _smooth_edge(dx, p["period_x"] * p["fin_width_frac"] / 2, softness) * 0.68

    dy = _wrap_dist(wy - p["phase_y"], p["period_gate"])
    gates = _smooth_edge(dy, p["period_gate"] * p["gate_width_frac"] / 2, softness * 1.4) * 0.92

    crossing_boost = fins * gates * 0.25
    base = np.clip(fins + gates - fins * gates * 0.4 + crossing_boost, 0.0, 1.0)
    bright, _ = _fiducial_overlay(wx, wy, p, softness)
    return np.clip(base * (1 - bright) + 1.0 * bright, 0.0, 1.0)


STYLE_FUNCS = {
    "dram": dram_intensity,
    "dram_arcuate": dram_arcuate_intensity,
    "finfet": finfet_intensity,
}


def default_params(style: str, rng: np.random.Generator, defect_prob: float = 0.5) -> dict:
    """Randomised, style-appropriate geometry. `defect_prob` controls how
    often a deliberate square alignment fiducial is placed at the
    reference's own world origin (so it also lands inside the search
    image at the true location) -- realistic (fabs place these for
    exactly this navigation purpose; the official example images each
    show one) and necessary for a subset of samples to be solvable at
    all, since a purely periodic field of view has no information-
    theoretic way to recover a unique location. Periods are sized so
    that period/10 (the nominal scale ratio) still clears the visibility
    floor after the search capture's anti-aliasing step.
    """
    if style == "dram":
        period_x = rng.uniform(85.0, 120.0)
        period_y = rng.uniform(85.0, 120.0)
        params = dict(
            period_x=period_x, period_y=period_y,
            phase_x=rng.uniform(0, period_x), phase_y=rng.uniform(0, period_y),
            linewidth_frac=rng.uniform(0.22, 0.30),
            via_frac=rng.uniform(0.14, 0.20),
        )
        if rng.uniform() < defect_prob:
            params["defect"] = True
            params["defect_xy"] = (0.0, 0.0)
            params["defect_radius"] = min(period_x, period_y) * 1.1
        return params
    elif style == "dram_arcuate":
        period_u = rng.uniform(80.0, 110.0)
        period_v = rng.uniform(60.0, 85.0)
        params = dict(
            period_u=period_u, period_v=period_v,
            phase_x=rng.uniform(0, period_u), phase_y=rng.uniform(0, period_v),
            half_len=period_u * rng.uniform(0.24, 0.30),
            radius=period_v * rng.uniform(0.11, 0.14),
            tilt_deg=rng.uniform(25.0, 40.0),
        )
        if rng.uniform() < defect_prob:
            params["defect"] = True
            params["defect_xy"] = (0.0, 0.0)
            params["defect_radius"] = min(period_u, period_v) * 1.2
        return params
    elif style == "finfet":
        period_x = rng.uniform(55.0, 80.0)
        period_gate = period_x * rng.uniform(4.0, 6.0)
        params = dict(
            period_x=period_x, period_gate=period_gate,
            phase_x=rng.uniform(0, period_x), phase_y=rng.uniform(0, period_gate),
            fin_width_frac=rng.uniform(0.30, 0.40),
            gate_width_frac=rng.uniform(0.16, 0.22),
        )
        if rng.uniform() < defect_prob:
            params["defect"] = True
            params["defect_xy"] = (0.0, 0.0)
            params["defect_radius"] = period_x * 1.4
        return params
    raise ValueError(f"unknown style {style!r}")


# ----------------------------------------------------------------------
# Capture model: render one image (reference OR search) of the pattern
# ----------------------------------------------------------------------

def render_capture(
    style: str,
    params: dict,
    out_size: int,
    center_world: tuple[float, float],
    scale: float,
    theta_deg: float = 0.0,
    base_softness_px: float = 0.9,
    supersample: int = 2,
) -> np.ndarray:
    """Render one out_size x out_size capture of the periodic pattern."""
    fn = STYLE_FUNCS[style]
    n = out_size * supersample
    lin = (np.arange(n) + 0.5) / supersample - out_size / 2.0
    u, v = np.meshgrid(lin, lin)
    u, v = _rotate(u, v, theta_deg)
    wx = center_world[0] + u * scale
    wy = center_world[1] + v * scale

    softness_world = base_softness_px * scale
    img = fn(wx, wy, params, softness_world).astype(np.float32)

    if supersample > 1:
        img = cv2.resize(img, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return np.clip(img, 0.0, 1.0)


# ----------------------------------------------------------------------
# Degradation models
# ----------------------------------------------------------------------

def add_edge_brightening(img: np.ndarray, strength: float = 0.22) -> np.ndarray:
    """SEM images show brighter contrast right along feature edges (the
    'edge effect' from enhanced secondary-electron escape probability
    near topographic edges; Reimer, *Scanning Electron Microscopy*,
    Springer, 2nd ed., 1998, ch. 4-5; Goldstein et al., *Scanning
    Electron Microscopy and X-Ray Microanalysis*, Springer, 2018, ch. 3).
    """
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    grad = np.sqrt(gx ** 2 + gy ** 2)
    m = grad.max()
    if m > 1e-6:
        grad = grad / m
    return np.clip(img + strength * grad, 0.0, 1.0)


def add_sensor_noise(img: np.ndarray, rng: np.random.Generator,
                      shot_gain: float = 0.05, read_noise_std: float = 0.012) -> np.ndarray:
    """Independent, per-image shot + read noise (Gaussian approximation of
    a Poissonian-Gaussian sensor noise model; Foi, Trimeche, Katkovnik &
    Egiazarian, "Practical Poissonian-Gaussian Noise Modeling and Fitting
    for Single-Image Raw-Data", IEEE TIP 17(10), 2008). Independent `rng`
    per call is what keeps reference and search noise independent, as the
    problem statement requires -- they are separate physical captures.
    """
    signal = np.clip(img, 0.0, 1.0)
    shot_std = np.sqrt(np.maximum(signal, 1e-4)) * shot_gain
    noisy = signal + rng.normal(0.0, 1.0, size=img.shape) * shot_std \
        + rng.normal(0.0, read_noise_std, size=img.shape)
    return noisy


def to_uint8(img: np.ndarray) -> np.ndarray:
    return np.clip(img * 255.0, 0, 255).astype(np.uint8)

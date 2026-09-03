"""
dataset_generator.py
---------------------
Standalone synthetic dataset generator for the Drift-Sense problem
statement (Applied Materials / IESA-SEMICON Hackathon 2026, Track 2).

No proprietary wafer-inspection data is available, so every training and
self-evaluation pair is generated procedurally: a Reference image (fine,
"100x" sampling of a periodic die layout) and a Search image (a 1000x1000
capture of the *same* underlying periodic pattern at ~10x lower
magnification, with the reference's true location recorded exactly).

Usage
-----
    python dataset_generator.py --style dram          --num-pairs 200 --split train --output-dir ../data
    python dataset_generator.py --style dram_arcuate  --num-pairs 40  --split val   --output-dir ../data
    python dataset_generator.py --style both          --num-pairs 36  --split test  --output-dir ../data

`--style` accepts dram, dram_arcuate, finfet, or both (rotates through
all three registered styles). Every run is fully parametrized from the
command line and requires no manual edits.

PHASE 2 ADDENDUM (Registration under Unknown Pose)
---------------------------------------------------
Phase 1's generator (`make_pair`, `generate`, below) is UNCHANGED and
still reachable exactly as before -- v1/v2/v3 and train_confidence.py
all still call `make_pair` directly and behave identically. Phase 2
needs data this generator never had a reason to produce (unknown zoom
in the disclosed [8,12] range rather than "~10x jitter", unknown +/-5
degree rotation, and -- the one thing Phase 1's generator structurally
cannot make -- pairs where the reference is genuinely ABSENT, not just
periodic-and-ambiguous). `make_pair_p2` / `generate_p2` add exactly
that as new, additive functions; pass `--phase2` to reach them from
this same CLI.

    python dataset_generator.py --phase2 --style both --num-pairs 200 \\
        --split val --output-dir ../data_p2 --absent-frac 0.22 --degraded-frac 0.5
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import cv2
import numpy as np

from pattern_render import default_params, render_capture, add_edge_brightening, add_sensor_noise, to_uint8
from pattern_render import perturb_geometry, apply_degradation, to_pseudo_optical_rgb

SEARCH_SIZE = 1000   # fixed by the problem statement (1000x1000, both images)
REF_SIZE = 1000      # see pattern_render.py docstring: this is what makes the
                      # reference occupy ~100x100 px (~1%) inside the search image at K~10.
ALL_STYLES = ["dram", "dram_arcuate", "finfet"]


def make_pair(style: str, rng: np.random.Generator, out_size_search: int = SEARCH_SIZE,
              out_size_ref: int = REF_SIZE, margin: int = 120, defect_prob: float = 0.5,
              vary_noise: bool = True):
    """Render one (reference, search, ground_truth) sample.

    `vary_noise`: draw a per-sample noise multiplier (0.4x-2.5x the base
    shot/read noise levels below) instead of using a single fixed noise
    strength for every sample. Without this, "results across multiple
    noise levels" (an explicit validation requirement) is impossible to
    report because every generated pair would be equally noisy. The
    search image's noise is always scaled up relative to the reference's
    (search_mult = ref_mult * 1.6, clipped) so the search stays the
    noisier of the two at every level, per the problem statement.
    """
    params = default_params(style, rng, defect_prob=defect_prob)

    scale_ref = 1.0
    scale_factor = float(rng.uniform(8.5, 11.5))          # "~10x", with jitter
    scale_search = scale_ref * scale_factor
    theta_deg = float(rng.uniform(-6.0, 6.0))              # small stage rotation

    ref_center_world = (0.0, 0.0)
    # True placement: bounded, roughly-Gaussian offset from the search
    # window's center (realistic drift), not uniform anywhere -- a real
    # re-acquisition search window is centered on where the tool expects
    # the site to be.
    center = out_size_search / 2.0
    drift_std = out_size_search * 0.09
    if rng.uniform() < 0.15:
        drift_std *= 2.6  # occasional large-drift outlier
    lo, hi = margin, out_size_search - margin
    true_x = float(np.clip(rng.normal(center, drift_std), lo, hi))
    true_y = float(np.clip(rng.normal(center, drift_std), lo, hi))
    search_center_world = (
        ref_center_world[0] - scale_search * (true_x - out_size_search / 2.0),
        ref_center_world[1] - scale_search * (true_y - out_size_search / 2.0),
    )

    reference = render_capture(style, params, out_size_ref, ref_center_world,
                                scale_ref, theta_deg=theta_deg, base_softness_px=0.8)
    search = render_capture(style, params, out_size_search, search_center_world,
                             scale_search, theta_deg=0.0, base_softness_px=0.9)

    noise_mult = float(rng.uniform(0.4, 2.5)) if vary_noise else 1.0
    search_noise_mult = min(noise_mult * 1.6, 4.0)  # search stays noisier than reference

    reference = cv2.GaussianBlur(reference, (0, 0), sigmaX=0.5)
    reference = add_edge_brightening(reference, strength=0.20)
    ref_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    reference = add_sensor_noise(reference, ref_rng, shot_gain=0.045 * noise_mult,
                                  read_noise_std=0.010 * noise_mult)

    search = cv2.GaussianBlur(search, (0, 0), sigmaX=0.8)
    search = add_edge_brightening(search, strength=0.20)
    search_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    search = add_sensor_noise(search, search_rng, shot_gain=0.075 * search_noise_mult,
                               read_noise_std=0.016 * search_noise_mult)

    gt = dict(
        style=style, x=round(true_x, 3), y=round(true_y, 3),
        rotation_deg=round(theta_deg, 3), scale_factor=round(scale_factor, 4),
        period_x=round(params.get("period_x", params.get("period_u", 0.0)), 3),
        period_y=round(params.get("period_y", params.get("period_gate", params.get("period_v", 0.0))), 3),
        has_landmark=bool(params.get("defect", False)),
        noise_level=round(noise_mult, 3),
        ref_size=out_size_ref, search_size=out_size_search,
    )
    return to_uint8(reference), to_uint8(search), gt


def generate(style: str, num_pairs: int, output_dir: Path, split: str | None,
             seed: int, out_size_search: int, out_size_ref: int, defect_prob: float = 0.5):
    root = output_dir / split if split else output_dir
    ref_dir, search_dir = root / "references", root / "searches"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(seed)
    styles = ALL_STYLES if style == "both" else [style]

    rows = []
    t0 = time.time()
    for i in range(num_pairs):
        s = styles[i % len(styles)] if style == "both" else style
        sample_seed = int(master_rng.integers(0, 2 ** 31))
        rng = np.random.default_rng(sample_seed)
        ref_img, search_img, gt = make_pair(s, rng, out_size_search, out_size_ref,
                                             defect_prob=defect_prob)

        sample_id = f"{split or 'sample'}_{i:05d}"
        cv2.imwrite(str(ref_dir / f"{sample_id}.png"), ref_img)
        cv2.imwrite(str(search_dir / f"{sample_id}.png"), search_img)

        row = {"id": sample_id, "seed": sample_seed, **gt}
        rows.append(row)

    elapsed = time.time() - t0

    csv_path = root / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = root / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_landmark = sum(1 for r in rows if r["has_landmark"])
    print(f"Generated {num_pairs} pairs ({style}) -> {root}  [{elapsed:.1f}s, "
          f"{elapsed / num_pairs:.2f}s/pair]")
    print(f"  {n_landmark}/{num_pairs} have a unique local landmark "
          f"(a deliberate alignment fiducial); the rest are purely periodic.")
    print(f"  references/  searches/  metadata.csv  metadata.json")
    return csv_path


# ----------------------------------------------------------------------
# PHASE 2 ADDITIONS
# ----------------------------------------------------------------------

from pattern_render import DEGRADATION_SEVERITIES  # noqa: E402


def make_pair_p2(style: str, rng: np.random.Generator, out_size_search: int = SEARCH_SIZE,
                  out_size_ref: int = REF_SIZE, margin: int = 120, defect_prob: float = 0.5,
                  scale_range: tuple[float, float] = (8.0, 12.0), rotation_max: float = 5.0,
                  absent: bool = False, severity: int | None = None, optical: bool = False):
    """SUPERSEDED for training/validation use -- kept only so earlier
    scripts that import it (evaluate_p2.py, spec_fair.py, train_confidence_p2.py's
    ORIGINAL run) still execute; do not use this for NEW training data.

    Once Applied Materials shared their actual Phase 2 generator
    (organizer_gen/, vendored verbatim) plus 20 real ground-truthed
    samples, direct testing turned up a real modeling mismatch here:
    this function rotates the REFERENCE and keeps the search axis-
    aligned. The organizer's real generator does the opposite --
    rotates the SEARCH raster by theta about the canvas centre and
    keeps the reference an unrotated crop (confirmed three independent
    ways: matching theta directly, not negated, against 5 real samples'
    correlation peaks; their own baseline_zncc.py using
    `cv2.getRotationMatrix2D(center, theta, 1/zoom)` with theta
    unnegated; and their own generator-build prompt document stating
    the convention explicitly). That's a real train/deployment
    distribution mismatch for anything trained on this function's
    output (a CNN never sees a rotated search patch here, but real
    search images carry up to +/-5deg of it), not just a sign
    convention to fix in isolation -- see
    docs/validation_report_v5.md, "The rotation model was inverted"
    for the full derivation and the empirical check.

    Use `organizer_gen.phase2_pipeline.generate_phase2_sample` (see
    gen_batch_organizer.py-style scripts, and
    docs/validation_report_v5.md) for anything new: training data,
    threshold tuning, or validation. This function's OWN registration
    ALGORITHM target (`localizer_v4.py`) is unaffected by any of this --
    it searches for whichever angle maximizes correlation regardless of
    which side of the generative process carries the rotation, and
    converges correctly either way (confirmed against all 20 real
    samples: theta error median 0.045deg, worst 0.208deg). Only DATA
    GENERATED BY THIS FUNCTION for training/scoring purposes carries
    the mismatch.

    Original docstring follows, describing what this function models
    (still accurate for what it actually does, just not what the real
    generator does):

    Phase 2 pair generator. Same physical capture model as `make_pair`
    above (world-coordinate renderer, independent reference/search
    noise) -- extended with the three things Phase 1's generator has no
    reason to model:

      scale_range / rotation_max -- the addendum's DISCLOSED bounds
        ([8,12], +/-5deg) instead of Phase 1's ~10x-with-jitter. Default
        values here already match the addendum exactly.

      absent -- if True, reference and search are rendered from TWO
        INDEPENDENTLY DRAWN parameter sets of the SAME `style` (own
        period, phase, linewidths...) -- "a different die region of the
        same architecture" per the addendum's Set C description -- so no
        true instance of the reference exists anywhere in the search
        image. `defect_prob=0.0` on the search side's own draw
        specifically avoids a look-alike fiducial accidentally landing
        near the search's own center, which would blur the line between
        "genuinely absent" and "present but ambiguous."

      severity -- None for a clean (Set-A-like) pair, or 0-3 for a
        degraded (Set-B-like) one: geometry ("polygon scaling") jitter
        via perturb_geometry PLUS the pixel-level defocus/charging/
        scan-distortion/noise stack via apply_degradation, applied to
        the SEARCH side only -- the reference stays the clean "golden"
        capture, matching the addendum's own Set B description.

      optical -- 3-channel pseudo-optical RGB output (Set D bonus)
        instead of grayscale, both sides (register.py's own image
        loader converts either back to luminance before matching, so
        this is the only place channel count is a design choice at all).

    IMPORTANT -- rotation sign, read before scoring `theta_report_deg`:
    `rotation_deg` here is `theta_deg`, the physical capture-rotation
    parameter handed to render_capture for the reference (kept for
    continuity with make_pair's own field of the same name). That is
    NOT the value localizer_v4.py will recover and report. A controlled
    test (fixed, known theta_deg; sweep candidate angles fed to
    localizer._make_template; take the argmax) shows the angle that
    actually maximizes correlation -- i.e. what any matcher, including
    this repo's, will converge to -- is -theta_deg, not theta_deg (see
    docs/validation_report_v4.md, "Rotation sign convention": verified
    both by that controlled sweep and independently against OpenCV's own
    documented +angle=CCW convention for cv2.getRotationMatrix2D). This
    also matches the addendum's own definition once you work through it
    ("rotation of the reference pattern AS IT APPEARS IN the search
    image" is, by construction, the rotation _make_template must apply
    TO the reference to reproduce that appearance -- not the rotation
    the reference's own capture underwent relative to world axes, which
    is what `theta_deg`/`rotation_deg` actually is). `theta_report_deg`
    (= -theta_deg) is the pre-computed, correctly-signed column;
    score/tune against THAT column, never against `rotation_deg` itself.
    """
    params_ref = default_params(style, rng, defect_prob=defect_prob)
    params_search = params_ref
    found = True
    if absent:
        params_search = default_params(style, rng, defect_prob=0.0)
        found = False
    elif severity is not None:
        geom_frac = DEGRADATION_SEVERITIES[int(np.clip(severity, 0, 3))]["geom_frac"]
        params_search = perturb_geometry(style, params_ref, rng, max_frac=geom_frac)

    scale_factor = float(rng.uniform(scale_range[0], scale_range[1]))
    theta_deg = float(rng.uniform(-rotation_max, rotation_max))

    ref_center_world = (0.0, 0.0)
    center = out_size_search / 2.0
    drift_std = out_size_search * 0.09
    if rng.uniform() < 0.15:
        drift_std *= 2.6
    lo, hi = margin, out_size_search - margin
    true_x = float(np.clip(rng.normal(center, drift_std), lo, hi))
    true_y = float(np.clip(rng.normal(center, drift_std), lo, hi))
    search_center_world = (
        ref_center_world[0] - scale_factor * (true_x - out_size_search / 2.0),
        ref_center_world[1] - scale_factor * (true_y - out_size_search / 2.0),
    )
    if absent:
        # No true site -- re-anchor the "different die region" on a
        # fresh random world offset so it isn't left sitting at world
        # (0,0) purely because that's the reference's own origin.
        search_center_world = (search_center_world[0] + float(rng.uniform(-2000, 2000)),
                                search_center_world[1] + float(rng.uniform(-2000, 2000)))

    reference = render_capture(style, params_ref, out_size_ref, ref_center_world,
                                1.0, theta_deg=theta_deg, base_softness_px=0.8)
    search = render_capture(style, params_search, out_size_search, search_center_world,
                             scale_factor, theta_deg=0.0, base_softness_px=0.9)

    noise_mult = float(rng.uniform(0.4, 2.5))
    search_noise_mult = min(noise_mult * 1.6, 4.0)
    if severity is not None and not absent:
        search_noise_mult *= (1.0 + 0.2 * DEGRADATION_SEVERITIES[int(np.clip(severity, 0, 3))]["noise_mult"])

    reference = cv2.GaussianBlur(reference, (0, 0), sigmaX=0.5)
    reference = add_edge_brightening(reference, strength=0.20)
    ref_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    reference = add_sensor_noise(reference, ref_rng, shot_gain=0.045 * noise_mult,
                                  read_noise_std=0.010 * noise_mult)

    search = cv2.GaussianBlur(search, (0, 0), sigmaX=0.8)
    search = add_edge_brightening(search, strength=0.20)
    search_rng = np.random.default_rng(rng.integers(0, 2 ** 31))
    if severity is not None and not absent:
        search = apply_degradation(search, search_rng, severity)
    search = add_sensor_noise(search, search_rng, shot_gain=0.075 * search_noise_mult,
                               read_noise_std=0.016 * search_noise_mult)

    reference = np.clip(reference, 0.0, 1.0)
    search = np.clip(search, 0.0, 1.0)
    ref_out, search_out = to_uint8(reference), to_uint8(search)
    if optical:
        ref_out = to_uint8(to_pseudo_optical_rgb(reference, ref_rng))
        search_out = to_uint8(to_pseudo_optical_rgb(search, search_rng))

    gt = dict(
        style=style,
        x=round(true_x, 3) if found else float("nan"),
        y=round(true_y, 3) if found else float("nan"),
        rotation_deg=round(theta_deg, 3),
        theta_report_deg=round(-theta_deg, 3) if found else float("nan"),
        scale_factor=round(scale_factor, 4) if found else float("nan"),
        period_x=round(params_ref.get("period_x", params_ref.get("period_u", 0.0)), 3),
        period_y=round(params_ref.get("period_y", params_ref.get("period_gate", params_ref.get("period_v", 0.0))), 3),
        has_landmark=bool(params_ref.get("defect", False)) and found,
        noise_level=round(noise_mult, 3),
        found=found,
        severity=-1 if severity is None else int(severity),
        optical=bool(optical),
        ref_size=out_size_ref, search_size=out_size_search,
    )
    return ref_out, search_out, gt


def generate_p2(style: str, num_pairs: int, output_dir: Path, split: str | None, seed: int,
                 out_size_search: int, out_size_ref: int, defect_prob: float,
                 scale_lo: float, scale_hi: float, rotation_max: float,
                 absent_frac: float, degraded_frac: float, optical_frac: float):
    root = output_dir / split if split else output_dir
    ref_dir, search_dir = root / "references", root / "searches"
    ref_dir.mkdir(parents=True, exist_ok=True)
    search_dir.mkdir(parents=True, exist_ok=True)

    master_rng = np.random.default_rng(seed)
    styles = ALL_STYLES if style == "both" else [style]

    rows = []
    t0 = time.time()
    for i in range(num_pairs):
        s = styles[i % len(styles)] if style == "both" else style
        sample_seed = int(master_rng.integers(0, 2 ** 31))
        rng = np.random.default_rng(sample_seed)

        absent = rng.uniform() < absent_frac
        optical = rng.uniform() < optical_frac
        severity = None
        if not absent and rng.uniform() < degraded_frac:
            severity = int(rng.integers(0, 4))

        ref_img, search_img, gt = make_pair_p2(
            s, rng, out_size_search, out_size_ref, defect_prob=defect_prob,
            scale_range=(scale_lo, scale_hi), rotation_max=rotation_max,
            absent=absent, severity=severity, optical=optical)

        sample_id = f"{split or 'sample'}_{i:05d}"
        cv2.imwrite(str(ref_dir / f"{sample_id}.png"), ref_img)
        cv2.imwrite(str(search_dir / f"{sample_id}.png"), search_img)
        rows.append({"id": sample_id, "seed": sample_seed, **gt})

    elapsed = time.time() - t0
    csv_path = root / "metadata.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    json_path = root / "metadata.json"
    with open(json_path, "w") as f:
        json.dump(rows, f, indent=2)

    n_absent = sum(1 for r in rows if not r["found"])
    n_degraded = sum(1 for r in rows if r["severity"] >= 0)
    n_optical = sum(1 for r in rows if r["optical"])
    n_landmark = sum(1 for r in rows if r["has_landmark"])
    print(f"Generated {num_pairs} Phase-2 pairs ({style}) -> {root}  "
          f"[{elapsed:.1f}s, {elapsed / num_pairs:.2f}s/pair]")
    print(f"  scale range=[{scale_lo},{scale_hi}]  rotation=+/-{rotation_max}deg")
    print(f"  {n_absent}/{num_pairs} absent (Set-C-like), {n_degraded}/{num_pairs} degraded "
          f"(Set-B-like), {n_optical}/{num_pairs} optical/RGB (Set-D-like)")
    print(f"  {n_landmark}/{num_pairs} present pairs additionally carry a unique fiducial.")
    print(f"  references/  searches/  metadata.csv  metadata.json")
    return csv_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--style", choices=["dram", "dram_arcuate", "finfet", "both"], default="both",
                    help="Die architecture style to generate (default: both, rotating through all three).")
    ap.add_argument("--num-pairs", type=int, default=30,
                    help="Number of reference/search pairs to generate (default: 30, the "
                         "hackathon-required self-evaluation minimum).")
    ap.add_argument("--output-dir", type=Path, default=Path("../data"),
                    help="Root output directory (default: ../data).")
    ap.add_argument("--split", choices=["train", "val", "test"], default=None,
                    help="If given, writes to <output-dir>/<split>/ instead of directly to "
                         "<output-dir>/ -- run once per split to build train/val/test.")
    ap.add_argument("--seed", type=int, default=None,
                    help="Master RNG seed. Default: auto-derived from --split (42/holdover "
                         "train=42, val=142042, test=242042; 42 if no --split given) so that "
                         "running the README Quickstart's three commands as-is produces three "
                         "INDEPENDENT splits. Earlier versions of this script defaulted every "
                         "split to the same fixed seed=42, which made data/train, data/val and "
                         "data/test pixel-identical for their first overlapping N samples -- "
                         "confirmed by direct comparison of generated metadata. Pass --seed "
                         "explicitly to reproduce that old behaviour or for any other reason.")
    ap.add_argument("--search-size", type=int, default=SEARCH_SIZE,
                    help=f"Search image side length in pixels (default: {SEARCH_SIZE}).")
    ap.add_argument("--ref-size", type=int, default=REF_SIZE,
                    help=f"Reference image side length in pixels (default: {REF_SIZE}).")
    ap.add_argument("--defect-prob", type=float, default=0.5,
                    help="Fraction of samples given a unique alignment fiducial marker "
                         "at the true site, vs. left purely periodic (default: 0.5).")
    ap.add_argument("--phase2", action="store_true",
                    help="Use the Phase 2 generator (make_pair_p2/generate_p2) instead of "
                         "Phase 1's -- adds unknown scale/rotation bounds, absent pairs, "
                         "degraded severities and optical/RGB pairs. Default OFF, so existing "
                         "Phase 1 usage (README Quickstart, train_confidence.py, etc.) is "
                         "completely unaffected.")
    ap.add_argument("--scale-lo", type=float, default=8.0, help="Phase 2: lower scale bound.")
    ap.add_argument("--scale-hi", type=float, default=12.0, help="Phase 2: upper scale bound.")
    ap.add_argument("--rotation-max", type=float, default=5.0,
                    help="Phase 2: rotation sampled uniformly in +/- this many degrees.")
    ap.add_argument("--absent-frac", type=float, default=0.0,
                    help="Phase 2: fraction of pairs where the reference is genuinely absent "
                         "(Set-C-like). The addendum's own Set C is 40/180=22%% of the "
                         "grayscale pairs; --absent-frac 0.22 matches that.")
    ap.add_argument("--degraded-frac", type=float, default=0.0,
                    help="Phase 2: fraction of the PRESENT pairs given a degradation severity "
                         "(Set-B-like: charging/scan-distortion/defocus/elevated noise/polygon "
                         "scaling). Severity 0-3 is drawn uniformly per degraded pair.")
    ap.add_argument("--optical-frac", type=float, default=0.0,
                    help="Phase 2: fraction of pairs rendered as 3-channel pseudo-optical RGB "
                         "(Set-D-like bonus).")
    args = ap.parse_args()

    if args.seed is not None:
        resolved_seed = args.seed
    else:
        resolved_seed = {"train": 42, "val": 142042, "test": 242042}.get(args.split, 42)

    if args.phase2:
        generate_p2(args.style, args.num_pairs, args.output_dir, args.split, resolved_seed,
                    args.search_size, args.ref_size, args.defect_prob,
                    args.scale_lo, args.scale_hi, args.rotation_max,
                    args.absent_frac, args.degraded_frac, args.optical_frac)
    else:
        generate(args.style, args.num_pairs, args.output_dir, args.split,
                  resolved_seed, args.search_size, args.ref_size, args.defect_prob)


if __name__ == "__main__":
    main()

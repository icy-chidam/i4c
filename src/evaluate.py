"""
evaluate.py
------------
Self-evaluation harness. Runs localize() over a generated test split,
reports accuracy and timing, and saves annotated visualizations for a
genuine SUCCESS case and a genuine HONEST FAILURE case.

v2 changes (see docs/validation_report.md for the full rationale and the
before/after numbers that motivated each one):
  * --engine {v2,v1} selects the localization pipeline. v2 (default) is
    localizer_v2.localize_v2 -- identical CNN-landmark path, but the
    classical fallback (used when no landmark is confidently detected)
    picks the closest-to-centre site by exact lattice enumeration instead
    of NMS-peak survival. v1 is the original localizer.localize, kept
    only so the two can be A/B compared on demand.
  * Threshold-wise accuracy now reports 5-, 4-, 2- and 1-pixel pass
    rates (the problem statement's own validation requirement; the
    previous version of this file only computed a 5px and a separately-
    named-but-actually-configurable "20px" bucket) plus a 0.5px
    sub-pixel bucket.
  * Mean, median AND worst-case (max) error are now all reported (worst-
    case was previously missing entirely).
  * Results are stratified by style, scale-factor bucket, rotation-angle
    bucket (specifically separating the spec's own stated "~1-2 degrees"
    range from the wider range this generator produces for robustness
    testing) and noise-level tertile (low/med/high) -- this requires
    dataset_generator.py's noise_level field, so re-generate your data
    directory with the current generator if you're re-using an older one
    that predates it.
  * results.csv is now the single, complete manifest deliverable
    (reference/search paths, true x/y, predicted x/y, error, AND every
    per-pair generation parameter) instead of splitting that information
    across metadata.csv and results.csv with several fields in neither.
  * The output JSON records hardware, Python version and the timing
    method used, per the explicit validation requirement.

Because the dataset mixes two genuinely different regimes -- fields of
view with a unique alignment fiducial (solvable) and fields that are
purely, repeatingly periodic (not solvable from this image pair alone;
see docs/references.md for why that is a property of the *pattern*, not
a shortcoming of the matcher) -- headline "accuracy" is reported for
each regime separately.

Two ways to score periodic samples: the problem statement's own
tie-break rule (closest to the search image's center, when more than
one tile matches) implies a specific "correct" answer even for a purely
periodic field: whichever periodic repeat of the true site is nearest to
center. That is generally NOT the arbitrary location this generator
happened to place the true site at. So error is reported against BOTH
the generator's true placement AND that spec-fair target -- see
_spec_fair_target(). NOTE: for the dram_arcuate style specifically, this
target is itself only approximate (it treats the lattice as axis-aligned
using period_x/period_y, when the true dram_arcuate lattice is a tilted
two-family rhombic lattice) -- see docs/validation_report.md, "Known
limitation" section, before reading too much into dram_arcuate-specific
numbers near the threshold.

Usage
-----
    python evaluate.py --data-dir ../data/test --output-dir ../results
    python evaluate.py --data-dir ../data/test --output-dir ../results_v1 --engine v1
"""
from __future__ import annotations

import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from localizer import localize as localize_v1


def _load_confidence_model(path: Path):
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception:
        return None


def _load_cnn_matcher(path: Path):
    if not path.exists():
        return None
    try:
        from cnn_matcher import PatchCNN
        return PatchCNN.load(path)
    except Exception:
        return None


def _spec_fair_target(tx: float, ty: float, period_x: float, period_y: float,
                       search_size: int = 1000, rotation_deg: float = 0.0,
                       cx: float | None = None, cy: float | None = None):
    """The nearest periodic-equivalent of the true site to the search
    image's center -- the fair target implied by the spec's own
    tie-break rule for a landmark-free (purely periodic) sample. Only
    meaningful there; a landmark makes the true site uniquely correct
    (see call site). Accounts for the sample's own small capture
    rotation. Approximate for the dram_arcuate style, whose true
    periodicity is a rotated (tilted, two-family) lattice rather than
    the axis-aligned one assumed here -- see module docstring.
    """
    cx = search_size / 2.0 if cx is None else cx
    cy = search_size / 2.0 if cy is None else cy
    if period_x <= 1e-6 or period_y <= 1e-6:
        return tx, ty
    t = np.deg2rad(rotation_deg)
    c, s = np.cos(t), np.sin(t)
    best = (tx, ty)
    best_d2 = (tx - cx) ** 2 + (ty - cy) ** 2
    k_range = range(-8, 9)
    for kx in k_range:
        for ky in k_range:
            ox = kx * period_x * c - ky * period_y * s
            oy = kx * period_x * s + ky * period_y * c
            cxp, cyp = tx + ox, ty + oy
            if not (0 <= cxp <= search_size and 0 <= cyp <= search_size):
                continue
            d2 = (cxp - cx) ** 2 + (cyp - cy) ** 2
            if d2 < best_d2:
                best_d2 = d2
                best = (cxp, cyp)
    return best


def _scale_bucket(scale_factor: float) -> str:
    if scale_factor < 9.5:
        return "scale<9.5"
    if scale_factor > 10.5:
        return "scale>10.5"
    return "scale9.5-10.5"


def _rotation_bucket(rotation_deg: float) -> str:
    a = abs(rotation_deg)
    if a <= 2.0:
        return "rot<=2deg (spec-nominal)"
    return "rot>2deg (stress test)"


def _noise_bucket_edges(levels: list[float]) -> tuple[float, float]:
    arr = np.array(levels)
    return float(np.percentile(arr, 33)), float(np.percentile(arr, 67))


def run_eval(data_dir: Path, output_dir: Path, weights: Path, cnn_weights: Path,
             n_scale: int = 9, n_angle: int = 5, engine: str = "v2"):
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(data_dir / "metadata.csv")))
    confidence_model = _load_confidence_model(weights)
    cnn_matcher = _load_cnn_matcher(cnn_weights)
    print(f"Engine: {engine}   CNN matcher: {'loaded' if cnn_matcher else 'not found -- classical-only'}")

    if engine in ("v2", "v3"):
        module = "localizer_v3" if engine == "v3" else "localizer_v2"
        fn_name = "localize_v3" if engine == "v3" else "localize_v2"
        _localize_fn = getattr(__import__(module, fromlist=[fn_name]), fn_name)
    else:
        _localize_fn = None  # uses localize_v1 directly below

    results = []
    for row in rows:
        sid = row["id"]
        ref_path = data_dir / "references" / f"{sid}.png"
        search_path = data_dir / "searches" / f"{sid}.png"
        ref = cv2.imread(str(ref_path), cv2.IMREAD_GRAYSCALE)
        search = cv2.imread(str(search_path), cv2.IMREAD_GRAYSCALE)
        if ref is None or search is None:
            print(f"skipping {sid}: image(s) missing")
            continue

        t0 = time.perf_counter()
        if engine in ("v2", "v3"):
            res = _localize_fn(ref, search, n_scale=n_scale, n_angle=n_angle,
                                confidence_model=confidence_model, cnn_matcher=cnn_matcher)
            tie_break = res.tie_break_method
        else:
            res = localize_v1(ref, search, n_scale=n_scale, n_angle=n_angle,
                               confidence_model=confidence_model, cnn_matcher=cnn_matcher)
            tie_break = "cnn_landmark" if res.cnn_used else "classical_center"
        dt_ms = (time.perf_counter() - t0) * 1000.0

        tx, ty = float(row["x"]), float(row["y"])
        has_landmark = row.get("has_landmark", "False") == "True"
        scale_factor = float(row.get("scale_factor", 10.0)) or 10.0
        rotation_deg = float(row.get("rotation_deg", 0.0))
        noise_level = float(row.get("noise_level", 1.0)) if row.get("noise_level") else None
        if has_landmark:
            fair_x, fair_y = tx, ty
        else:
            period_x_s = float(row.get("period_x", 0.0)) / scale_factor
            period_y_s = float(row.get("period_y", 0.0)) / scale_factor
            fair_x, fair_y = _spec_fair_target(tx, ty, period_x_s, period_y_s,
                                                search_size=int(row.get("search_size", 1000)),
                                                rotation_deg=rotation_deg)

        err = float(np.hypot(res.x - tx, res.y - ty))
        err_fair = float(np.hypot(res.x - fair_x, res.y - fair_y))

        results.append(dict(
            id=sid, reference_path=str(ref_path), search_path=str(search_path),
            style=row["style"], has_landmark=has_landmark,
            scale_factor=round(scale_factor, 4), rotation_deg=round(rotation_deg, 3),
            noise_level=noise_level, seed=row.get("seed", ""),
            true_x=tx, true_y=ty, pred_x=round(res.x, 3), pred_y=round(res.y, 3),
            error_px=round(err, 4), error_px_spec_fair=round(err_fair, 4),
            confidence=round(res.confidence, 4), cnn_used=res.cnn_used,
            tie_break_method=tie_break,
            num_candidate_peaks=res.num_candidate_peaks, time_ms=round(dt_ms, 1),
        ))

    df = pd.DataFrame(results)
    df["scale_bucket"] = df["scale_factor"].apply(_scale_bucket)
    df["rotation_bucket"] = df["rotation_deg"].apply(_rotation_bucket)
    if df["noise_level"].notna().any():
        lo, hi = _noise_bucket_edges(df["noise_level"].dropna().tolist())
        df["noise_bucket"] = df["noise_level"].apply(
            lambda v: "n/a" if pd.isna(v) else ("low" if v <= lo else ("high" if v >= hi else "mid")))
    else:
        df["noise_bucket"] = "n/a (pre-noise-variation dataset)"

    def _summary_row(sub: pd.DataFrame, label: str) -> dict:
        if sub.empty:
            return {}
        e = sub["error_px"].to_numpy()
        ef = sub["error_px_spec_fair"].to_numpy()
        t = sub["time_ms"].to_numpy()
        return dict(
            label=label, n=int(len(sub)),
            mean_error_px=round(float(e.mean()), 3),
            median_error_px=round(float(np.median(e)), 3),
            worst_error_px=round(float(e.max()), 3),
            within_5px=round(float((e < 5).mean()), 4),
            within_4px=round(float((e < 4).mean()), 4),
            within_2px=round(float((e < 2).mean()), 4),
            within_1px=round(float((e < 1).mean()), 4),
            within_0_5px_subpixel=round(float((e < 0.5).mean()), 4),
            mean_error_px_spec_fair=round(float(ef.mean()), 3),
            median_error_px_spec_fair=round(float(np.median(ef)), 3),
            worst_error_px_spec_fair=round(float(ef.max()), 3),
            within_5px_spec_fair=round(float((ef < 5).mean()), 4),
            within_4px_spec_fair=round(float((ef < 4).mean()), 4),
            within_2px_spec_fair=round(float((ef < 2).mean()), 4),
            within_1px_spec_fair=round(float((ef < 1).mean()), 4),
            mean_confidence=round(float(sub["confidence"].mean()), 4),
            mean_time_ms=round(float(t.mean()), 1),
            p95_time_ms=round(float(np.percentile(t, 95)), 1),
        )

    summary = dict(
        all=_summary_row(df, "ALL"),
        with_fiducial=_summary_row(df[df["has_landmark"]], "WITH fiducial (solvable)"),
        pure_periodic=_summary_row(df[~df["has_landmark"]], "PURE periodic (ambiguous)"),
    )
    for s in sorted(df["style"].unique()):
        summary[f"style_{s}"] = _summary_row(df[df["style"] == s], f"style={s}")
    for b in sorted(df["scale_bucket"].unique()):
        summary[f"scale_{b}"] = _summary_row(df[df["scale_bucket"] == b], b)
    for b in sorted(df["rotation_bucket"].unique()):
        summary[f"rotation_{b}"] = _summary_row(df[df["rotation_bucket"] == b], b)
    if df["noise_bucket"].iloc[0] != "n/a (pre-noise-variation dataset)":
        for b in ["low", "mid", "high"]:
            sub = df[df["noise_bucket"] == b]
            if not sub.empty:
                summary[f"noise_{b}"] = _summary_row(sub, f"noise={b}")

    print()
    for key in ["all", "with_fiducial", "pure_periodic"]:
        s = summary[key]
        if not s:
            continue
        print(f"{s['label']:26s} n={s['n']:3d}  mean={s['mean_error_px']:8.2f}px  "
              f"median={s['median_error_px']:8.2f}px  worst={s['worst_error_px']:8.2f}px  "
              f"within[5/4/2/1]px={s['within_5px']:.0%}/{s['within_4px']:.0%}/"
              f"{s['within_2px']:.0%}/{s['within_1px']:.0%}  "
              f"fair_median={s['median_error_px_spec_fair']:7.2f}px  "
              f"fair_within5px={s['within_5px_spec_fair']:.0%}")

    high_conf = df[df["confidence"] >= 0.5]
    if not high_conf.empty:
        hc_e, hc_ef = high_conf["error_px"].to_numpy(), high_conf["error_px_spec_fair"].to_numpy()
        print(f"\nCalibration check: {len(high_conf)}/{len(df)} predictions had confidence "
              f">= 0.5; of those, {(hc_e < 5).mean():.0%} were within 5px of the generator's "
              f"true placement, {(hc_ef < 5).mean():.0%} within 5px of the spec-fair target.")

    env_info = dict(
        python_version=sys.version.split()[0],
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
        cpu_count=__import__("os").cpu_count(),
        engine=engine,
        timing_method="time.perf_counter() wrapped tightly around the localize()/localize_v2() "
                       "call only (excludes image I/O); wall-clock, single-threaded, CPU-only, "
                       "no GPU used or available to this pipeline.",
    )

    with open(output_dir / "results.json", "w") as f:
        json.dump(dict(environment=env_info, per_sample=results, summary=summary), f, indent=2)

    manifest_cols = ["id", "reference_path", "search_path", "style", "has_landmark",
                      "scale_factor", "rotation_deg", "noise_level", "seed",
                      "true_x", "true_y", "pred_x", "pred_y",
                      "error_px", "error_px_spec_fair", "confidence", "cnn_used",
                      "tie_break_method", "num_candidate_peaks", "time_ms"]
    df[manifest_cols].to_csv(output_dir / "results.csv", index=False)

    print(f"\nEnvironment: Python {env_info['python_version']}  {env_info['platform']}  "
          f"engine={engine}")
    print(f"Wrote {output_dir/'results.json'} and {output_dir/'results.csv'} "
          f"(single complete manifest: paths + truth + predictions + generation metadata).")
    return results


def save_case_figure(data_dir: Path, output_dir: Path, sample_id: str,
                      pred_x: float, pred_y: float, true_x: float, true_y: float,
                      confidence: float, label: str):
    ref = cv2.imread(str(data_dir / "references" / f"{sample_id}.png"))
    search = cv2.imread(str(data_dir / "searches" / f"{sample_id}.png"))
    disp = 420
    ref_d = cv2.resize(ref, (disp, disp))
    search_d = cv2.resize(search, (disp, disp))

    def mark(img, x, y, color, marker):
        mx, my = int(x * disp / 1000), int(y * disp / 1000)
        cv2.drawMarker(img, (mx, my), color, markerType=marker, markerSize=26, thickness=3)

    mark(search_d, true_x, true_y, (0, 200, 0), cv2.MARKER_SQUARE)
    mark(search_d, pred_x, pred_y, (0, 0, 255), cv2.MARKER_CROSS)

    pad = np.full((disp, 10, 3), 255, dtype=np.uint8)
    banner = np.full((56, disp * 2 + 10, 3), 255, dtype=np.uint8)
    cv2.putText(banner, f"{label}  |  conf={confidence:.3f}  "
                         f"(green square = true, red cross = predicted)",
                (10, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)
    sheet = np.vstack([banner, np.hstack([ref_d, pad, search_d])])
    out_path = output_dir / f"case_{sample_id}.png"
    cv2.imwrite(str(out_path), sheet)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("../data/test"))
    ap.add_argument("--output-dir", type=Path, default=Path("../results"))
    ap.add_argument("--weights", type=Path, default=Path("../weights/confidence_model.joblib"))
    ap.add_argument("--cnn-weights", type=Path, default=Path("../weights/cnn_matcher.npz"),
                    help="Trained CNN patch-matcher (.npz, see train_cnn_matcher.py). "
                         "Same file is used by all engines (v1/v2/v3) so the comparison "
                         "isolates the search-algorithm difference. If missing, the "
                         "classical/lattice tie-break is used automatically.")
    ap.add_argument("--fast", action="store_true", help="Smaller search grid for quick iteration.")
    ap.add_argument("--engine", choices=["v3", "v2", "v1"], default="v3",
                    help="v3 (default): coarse-to-fine pyramid search (faster) + lattice "
                         "tie-break. v2: lattice tie-break, original flat search grid. v1: "
                         "original NMS-closest-to-centre tie-break, original grid. All three "
                         "for A/B comparison -- see docs/validation_report_v3.md.")
    args = ap.parse_args()

    n_scale, n_angle = (5, 3) if args.fast else (9, 5)
    results = run_eval(args.data_dir, args.output_dir, args.weights, args.cnn_weights,
                        n_scale, n_angle, engine=args.engine)

    landmark_ok = [r for r in results if r["has_landmark"] and r["error_px"] < 5]
    landmark_fail = [r for r in results if not r["has_landmark"]]
    if landmark_ok:
        r = landmark_ok[len(landmark_ok) // 2]
        save_case_figure(args.data_dir, args.output_dir, r["id"], r["pred_x"], r["pred_y"],
                          r["true_x"], r["true_y"], r["confidence"], "SUCCESS (fiducial present)")
        print(f"Saved SUCCESS exhibit: case_{r['id']}.png")
    if landmark_fail:
        r = landmark_fail[len(landmark_fail) // 2]
        save_case_figure(args.data_dir, args.output_dir, r["id"], r["pred_x"], r["pred_y"],
                          r["true_x"], r["true_y"], r["confidence"],
                          "HONEST FAILURE (pure periodic, no fiducial -- low confidence reported)")
        print(f"Saved HONEST FAILURE exhibit: case_{r['id']}.png")


if __name__ == "__main__":
    main()

"""
evaluate_p2.py
----------------
Internal validation harness that simulates the addendum's ACTUAL 100pt
(+10 bonus) rubric on generated Phase 2 data, so the submission can be
checked against something close to the real scoring formula before
submitting -- not just proxy metrics (F1 alone, AUC alone) in isolation.
Same spirit as evaluate.py's Phase 1 role, extended for Phase 2's own
scored components.

Simulates, per the addendum's Scoring page:
  - Localization (40 pts): Sets A+B present pairs, tiered 1/2/3/5px
    credit, weighted 0.45*A + 0.55*B. Compared against
    spec_fair.localization_target, NOT raw generator x/y, for
    landmark-free pairs (see spec_fair.py for why).
  - Pose recovery (20 pts): scale (10) + rotation (10), tiered credit,
    scored only where localization credit > 0 (per the addendum), mean
    taken over ALL present pairs (so a pair with zero localization
    credit contributes zero pose credit too, not an excluded case).
  - Rejection (15 pts): F1 on the found flag across every grayscale
    (Set A+B+C) pair.
  - Calibration (10 pts): AUC of the reported score column against a
    per-pair correctness label (found decided right AND, if present,
    localization credit > 0) -- the addendum's own "any scale works, so
    long as it's monotonic with your own correctness."
  - Efficiency: reported (median/p95 wall clock), not scored -- there's
    no other teams' distribution to rank against here.
  - Generator/citations/failure-analysis (10 pts): not something this
    script can grade; carried forward from Phase 1 as the addendum
    says, tracked as a checklist elsewhere (see README.md).

Usage
-----
    python evaluate_p2.py --n-a 24 --n-b 24 --n-c 16 --n-d 6 --seed 7
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score

from dataset_generator import make_pair_p2
from localizer_v4 import localize_v4
from cnn_matcher import PatchCNN
from spec_fair import localization_target

try:
    import joblib
except ImportError:
    joblib = None

STYLES = ["dram", "dram_arcuate", "finfet"]


def loc_credit(err_px: float) -> float:
    if err_px <= 1.0:
        return 1.00
    if err_px <= 2.0:
        return 0.80
    if err_px <= 3.0:
        return 0.60
    if err_px <= 5.0:
        return 0.40
    return 0.0


def scale_credit(pred: float, true: float) -> float:
    rel = abs(pred - true) / true if true else 1.0
    if rel <= 0.01:
        return 1.00
    if rel <= 0.02:
        return 0.60
    if rel <= 0.05:
        return 0.30
    return 0.0


def rotation_credit(pred: float, true: float) -> float:
    err = abs(pred - true)
    if err <= 0.25:
        return 1.00
    if err <= 0.5:
        return 0.60
    if err <= 1.0:
        return 0.30
    return 0.0


def run_set(label: str, n: int, absent: bool, degraded: bool, optical: bool,
            master_rng, cnn_matcher, found_model, found_threshold):
    rows = []
    for i in range(n):
        style = STYLES[i % len(STYLES)]
        rng = np.random.default_rng(int(master_rng.integers(0, 2 ** 31)))
        severity = int(rng.integers(0, 4)) if degraded else None
        ref, search, gt = make_pair_p2(style, rng, absent=absent, severity=severity, optical=optical)

        t0 = time.perf_counter()
        res = localize_v4(ref, search, cnn_matcher=cnn_matcher, found_model=found_model,
                           found_threshold=found_threshold)
        elapsed = time.perf_counter() - t0

        row = dict(set=label, style=style, found_true=bool(gt["found"]), found_pred=bool(res.found),
                   score=float(res.score), time_s=elapsed)
        if gt["found"]:
            if res.found:
                # What the REAL predictions.csv would actually contain.
                px, py, pscale, ptheta = res.x, res.y, res.scale, res.theta_deg
            else:
                # Contract: "found: 1 or 0. When 0, write 0 in the pose
                # columns" -- a false negative on a present pair does
                # NOT get credit for whatever we internally computed;
                # the submitted row is genuinely (0,0,0,0).
                px, py, pscale, ptheta = 0.0, 0.0, 0.0, 0.0
            fx, fy = localization_target(gt)
            row["xy_err"] = float(np.hypot(px - fx, py - fy))
            row["loc_credit"] = loc_credit(row["xy_err"])
            row["scale_credit"] = scale_credit(pscale, gt["scale_factor"]) if row["loc_credit"] > 0 else 0.0
            row["rotation_credit"] = rotation_credit(ptheta, gt["theta_report_deg"]) if row["loc_credit"] > 0 else 0.0
        else:
            row["xy_err"] = None
            row["loc_credit"] = None
            row["scale_credit"] = None
            row["rotation_credit"] = None
        rows.append(row)
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-a", type=int, default=24, help="Set A (nominal, present) pair count")
    ap.add_argument("--n-b", type=int, default=24, help="Set B (degraded, present) pair count")
    ap.add_argument("--n-c", type=int, default=16, help="Set C (absent) pair count")
    ap.add_argument("--n-d", type=int, default=6, help="Set D (optical, bonus) pair count")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--cnn-weights", default="../weights/cnn_matcher_p2.npz")
    ap.add_argument("--confidence-weights", default="../weights/confidence_model_p2.joblib")
    args = ap.parse_args()

    cnn = PatchCNN.load(args.cnn_weights)
    found_model, found_threshold = None, 0.5
    if joblib is not None:
        bundle = joblib.load(args.confidence_weights)
        found_model = bundle["model"]
        found_threshold = float(bundle["threshold"])
        print(f"Loaded found_model={bundle.get('model_name')}  threshold={found_threshold:.3f}  "
              f"(val_auc={bundle.get('val_auc'):.4f} val_f1={bundle.get('val_f1'):.4f} at training time)")

    master_rng = np.random.default_rng(args.seed)
    t0 = time.time()
    rows = []
    rows += run_set("A", args.n_a, absent=False, degraded=False, optical=False,
                     master_rng=master_rng, cnn_matcher=cnn, found_model=found_model, found_threshold=found_threshold)
    rows += run_set("B", args.n_b, absent=False, degraded=True, optical=False,
                     master_rng=master_rng, cnn_matcher=cnn, found_model=found_model, found_threshold=found_threshold)
    rows += run_set("C", args.n_c, absent=True, degraded=False, optical=False,
                     master_rng=master_rng, cnn_matcher=cnn, found_model=found_model, found_threshold=found_threshold)
    rows += run_set("D", args.n_d, absent=False, degraded=False, optical=True,
                     master_rng=master_rng, cnn_matcher=cnn, found_model=found_model, found_threshold=found_threshold)
    total_time = time.time() - t0

    def present_rows(sets):
        return [r for r in rows if r["set"] in sets and r["found_true"]]

    A_present, B_present = present_rows(["A"]), present_rows(["B"])
    A_credit = np.mean([r["loc_credit"] for r in A_present]) if A_present else 0.0
    B_credit = np.mean([r["loc_credit"] for r in B_present]) if B_present else 0.0
    loc_score = 40.0 * (0.45 * A_credit + 0.55 * B_credit)

    AB_present = A_present + B_present
    scale_score = 10.0 * np.mean([r["scale_credit"] for r in AB_present]) if AB_present else 0.0
    rot_score = 10.0 * np.mean([r["rotation_credit"] for r in AB_present]) if AB_present else 0.0

    grayscale = [r for r in rows if r["set"] in ("A", "B", "C")]
    y_true_found = [1 if r["found_true"] else 0 for r in grayscale]
    y_pred_found = [1 if r["found_pred"] else 0 for r in grayscale]
    # STANDARD F1 (found=1/present positive, scikit-learn default). v4 used
    # pos_label=0 (reject positive), reasoned from the addendum's own prose;
    # superseded once the organizer's actual score_baseline.py became
    # available -- it computes standard F1, and its own
    # baseline_calibration.txt numbers only reproduce under that
    # convention. See docs/validation_report_v5.md, section 4.
    f1 = f1_score(y_true_found, y_pred_found)
    rejection_score = 15.0 * f1

    y_correct = [1 if (r["found_true"] and r["found_pred"] and (r["loc_credit"] or 0) > 0)
                    or (not r["found_true"] and not r["found_pred"]) else 0 for r in grayscale]
    y_score = [r["score"] for r in grayscale]
    try:
        auc = roc_auc_score(y_correct, y_score)
    except ValueError:
        auc = float("nan")
    calibration_score = 10.0 * auc if not np.isnan(auc) else float("nan")

    times = [r["time_s"] for r in rows]
    scored_total = loc_score + scale_score + rot_score + rejection_score + \
        (calibration_score if not np.isnan(calibration_score) else 0.0)

    print()
    print(f"{'SET':4s} {'n':>4s} {'found_true':>11s} {'found_pred':>11s} {'mean_xy_err':>12s} {'mean_loc_cred':>14s}")
    for label in ["A", "B", "C", "D"]:
        sub = [r for r in rows if r["set"] == label]
        if not sub:
            continue
        n_found_true = sum(r["found_true"] for r in sub)
        n_found_pred = sum(r["found_pred"] for r in sub)
        errs = [r["xy_err"] for r in sub if r["xy_err"] is not None]
        creds = [r["loc_credit"] for r in sub if r["loc_credit"] is not None]
        print(f"{label:4s} {len(sub):4d} {n_found_true:11d} {n_found_pred:11d} "
              f"{np.mean(errs) if errs else float('nan'):12.2f} {np.mean(creds) if creds else float('nan'):14.3f}")

    print()
    print(f"Localization (40):     {loc_score:6.2f}   (A_credit={A_credit:.3f} n={len(A_present)}, "
          f"B_credit={B_credit:.3f} n={len(B_present)})")
    print(f"Pose - scale (10):     {scale_score:6.2f}")
    print(f"Pose - rotation (10):  {rot_score:6.2f}")
    print(f"Rejection F1 (15):     {rejection_score:6.2f}   (F1={f1:.4f}, "
          f"{sum(y_true_found)}/{len(y_true_found)} truly found)")
    print(f"Calibration AUC (10):  {calibration_score:6.2f}   (AUC={auc:.4f})")
    print(f"---------------------------------")
    print(f"SUBTOTAL (of 95, excl. efficiency+carried-forward): {scored_total:6.2f}")
    print()
    print(f"Efficiency: median={np.median(times):.2f}s/pair  p95={np.percentile(times,95):.2f}s/pair  "
          f"max={np.max(times):.2f}s/pair  (budget: median<=5s, hard timeout 20s)")
    print(f"Total wall clock for {len(rows)} pairs: {total_time:.1f}s")


if __name__ == "__main__":
    main()

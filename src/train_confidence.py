"""
train_confidence.py
---------------------
Trains the optional confidence-calibration model referenced by
localizer.py / localize.py (weights/confidence_model.joblib).

Why this exists given the classical uniqueness-ratio confidence already
works: the ratio is a reasonable, zero-training-data heuristic, but it is
not a *calibrated* probability. A small logistic-regression head, trained
on engineered features from the correlation surface against our own
(unlimited, self-labelled) synthetic data, turns those features into an
actual probability-correct estimate. A failure of this file (missing,
corrupted, or just not retrained yet) degrades gracefully to the
classical ratio rather than breaking inference -- see localize.py's
loader.

Usage
-----
    python train_confidence.py --num-pairs 150 --output ../weights/confidence_model.joblib
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from dataset_generator import make_pair
from localizer import localize


def build_training_data(num_pairs: int, seed: int, n_scale: int, n_angle: int, tolerance_px: float):
    rng = np.random.default_rng(seed)
    styles = ["dram", "dram_arcuate", "finfet"]
    X, y = [], []
    for i in range(num_pairs):
        style = styles[i % len(styles)]
        sample_rng = np.random.default_rng(int(rng.integers(0, 2 ** 31)))
        ref, search, gt = make_pair(style, sample_rng, defect_prob=0.5)
        res = localize(ref, search, n_scale=n_scale, n_angle=n_angle)
        correct = int(np.hypot(res.x - gt["x"], res.y - gt["y"]) < tolerance_px)
        X.append([res.best_score, res.second_score,
                   1.0 - (res.second_score / res.best_score if res.best_score > 1e-6 else 0.0),
                   res.num_candidate_peaks, res.scale, res.rotation_deg])
        y.append(correct)
        if (i + 1) % 20 == 0:
            print(f"  generated {i + 1}/{num_pairs} training samples "
                  f"({sum(y)}/{len(y)} correct so far)")
    return np.array(X), np.array(y)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--num-pairs", type=int, default=150,
                    help="Synthetic pairs to train the calibrator on (default: 150).")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tolerance-px", type=float, default=20.0,
                    help="Pixel error under which a prediction counts as 'correct' (default: 20).")
    ap.add_argument("--output", type=Path, default=Path("../weights/confidence_model.joblib"))
    ap.add_argument("--fast", action="store_true", help="Smaller search grid, for quicker (re)training.")
    args = ap.parse_args()

    n_scale, n_angle = (5, 3) if args.fast else (9, 5)
    print(f"Generating {args.num_pairs} labelled synthetic samples "
          f"(n_scale={n_scale}, n_angle={n_angle})...")
    X, y = build_training_data(args.num_pairs, args.seed, n_scale, n_angle, args.tolerance_px)
    print(f"Label balance: {y.sum()}/{len(y)} positive (correct within {args.tolerance_px:.0f}px).")

    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=1000))

    if len(np.unique(y)) < 2:
        print("WARNING: only one class present in the generated labels -- try more --num-pairs "
              "or a different --seed. Falling back to the classical ratio at inference time "
              "(see localize.py's graceful fallback); not saving an unfitted model.")
        return

    scores = cross_val_score(model, X, y, cv=min(5, int(y.sum()), int((1 - y).sum() + 1)))
    print(f"5-fold CV accuracy of the calibrator: {scores.mean():.3f} +/- {scores.std():.3f}")

    model.fit(X, y)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(model, args.output)
    print(f"Saved calibrated confidence model -> {args.output}")


if __name__ == "__main__":
    main()

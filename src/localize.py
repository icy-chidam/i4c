"""
localize.py
-------------
THE mandatory inference script for the Drift-Sense submission. Applied
Materials runs this directly on their held-out test pairs, so it must
work with no manual edits: given a reference image path and a search
image path, it prints the predicted center (x, y) of the reference
pattern inside the search image, plus a confidence score.

Usage
-----
    python localize.py --reference ref.png --search search.png
    python localize.py --reference ref.png --search search.png --output result.json
    python localize.py --reference-dir data/test/references --search-dir data/test/searches \
                        --output-dir results/   # convenience batch mode
    python localize.py --reference ref.png --search search.png --no-cnn   # classical-only
    python localize.py --reference ref.png --search search.png --engine v1   # old tie-break, for comparison

Output (stdout, always) is a single JSON object:
    {"x": 546.4, "y": 227.1, "confidence": 0.83,
     "scale": 8.9, "rotation_deg": 1.2, "time_ms": 812.4,
     "num_candidate_peaks": 3, "cnn_used": true, "cnn_score": 0.97,
     "tie_break_method": "cnn_landmark"}

`num_candidate_peaks` > 1 means the periodic layout produced more than
one near-equally-good match. `cnn_used` reports whether the learned
patch matcher (cnn_matcher.py) made the final call (it did, if it was
confident about some candidate) or the pipeline fell back to the
classical/lattice tie-break. `tie_break_method` is one of:
"cnn_landmark" (a confident landmark was found -- unambiguous),
"lattice_enumeration" (no landmark; the closest-to-centre periodic
repeat was picked by exact lattice enumeration -- see localizer_v2.py
and docs/validation_report.md), or "classical_center_fallback" (rare --
only if no clear periodicity could be measured either).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2

from localizer import localize as localize_v1
from localizer_v2 import localize_v2
from localizer_v3 import localize_v3

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_WEIGHTS = REPO_ROOT / "weights" / "confidence_model.joblib"
DEFAULT_CNN_WEIGHTS = REPO_ROOT / "weights" / "cnn_matcher.npz"


def _load_confidence_model(path: Path):
    """Best-effort load of the optional learned confidence calibrator.
    Never raises -- a missing or unreadable weights file just means the
    classical (uniqueness-ratio) confidence is used instead.
    """
    if not path.exists():
        return None
    try:
        import joblib
        return joblib.load(path)
    except Exception as e:
        print(f"[localize.py] note: confidence model not loaded ({e}); "
              f"falling back to classical confidence.", file=sys.stderr)
        return None


def _load_cnn_matcher(path: Path):
    """Best-effort load of the optional learned CNN patch matcher. Never
    raises -- a missing or unreadable weights file just means the
    pipeline falls back to the classical closest-to-center tie-break.
    """
    if not path.exists():
        return None
    try:
        from cnn_matcher import PatchCNN
        return PatchCNN.load(path)
    except Exception as e:
        print(f"[localize.py] note: CNN matcher not loaded ({e}); "
              f"falling back to the classical tie-break.", file=sys.stderr)
        return None


def _read_gray(path: Path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"could not read image: {path}")
    return img


def run_one(reference_path: Path, search_path: Path, confidence_model=None, cnn_matcher=None,
            scale_range=(8.0, 12.0), n_scale=9, angle_range=(-6.0, 6.0), n_angle=5,
            engine: str = "v3") -> dict:
    reference = _read_gray(reference_path)
    search = _read_gray(search_path)
    if engine == "v3":
        result = localize_v3(reference, search, scale_range=scale_range, n_scale=n_scale,
                              angle_range=angle_range, n_angle=n_angle,
                              confidence_model=confidence_model, cnn_matcher=cnn_matcher)
        tie_break = result.tie_break_method
    elif engine == "v2":
        result = localize_v2(reference, search, scale_range=scale_range, n_scale=n_scale,
                              angle_range=angle_range, n_angle=n_angle,
                              confidence_model=confidence_model, cnn_matcher=cnn_matcher)
        tie_break = result.tie_break_method
    else:
        result = localize_v1(reference, search, scale_range=scale_range, n_scale=n_scale,
                              angle_range=angle_range, n_angle=n_angle,
                              confidence_model=confidence_model, cnn_matcher=cnn_matcher)
        tie_break = "cnn_landmark" if result.cnn_used else "classical_center"
    return {
        "reference": str(reference_path), "search": str(search_path),
        "x": round(result.x, 2), "y": round(result.y, 2),
        "confidence": round(result.confidence, 4),
        "scale": round(result.scale, 3), "rotation_deg": round(result.rotation_deg, 2),
        "num_candidate_peaks": result.num_candidate_peaks,
        "cnn_used": result.cnn_used, "cnn_score": round(result.cnn_score, 4),
        "tie_break_method": tie_break,
        "time_ms": round(result.time_ms, 1),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--reference", type=Path, help="Path to the reference image.")
    ap.add_argument("--search", type=Path, help="Path to the search image.")
    ap.add_argument("--output", type=Path, default=None,
                    help="Optional path to also write the JSON result to.")
    ap.add_argument("--reference-dir", type=Path, default=None,
                    help="Batch mode: directory of reference images (paired by filename).")
    ap.add_argument("--search-dir", type=Path, default=None,
                    help="Batch mode: directory of search images (paired by filename).")
    ap.add_argument("--output-dir", type=Path, default=None,
                    help="Batch mode: directory to write one JSON result per pair.")
    ap.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS,
                    help="Optional trained confidence-calibration model (.joblib). "
                         "If missing, classical confidence is used automatically.")
    ap.add_argument("--cnn-weights", type=Path, default=DEFAULT_CNN_WEIGHTS,
                    help="Optional trained CNN patch-matcher (.npz, see "
                         "train_cnn_matcher.py). If missing, the classical/lattice "
                         "tie-break is used automatically.")
    ap.add_argument("--no-cnn", action="store_true",
                    help="Disable the CNN matcher even if weights are present "
                         "(classical-only pipeline).")
    ap.add_argument("--fast", action="store_true",
                    help="Smaller scale/rotation search grid for quick iteration.")
    ap.add_argument("--engine", choices=["v3", "v2", "v1"], default="v3",
                    help="v3 (default): coarse-to-fine pyramid search (faster) + lattice "
                         "tie-break + noise-matched CNN -- see docs/validation_report_v3.md. "
                         "v2: lattice tie-break, original flat search grid. v1: original "
                         "NMS-closest-to-centre tie-break, original grid. All kept for A/B "
                         "comparison.")
    args = ap.parse_args()

    confidence_model = _load_confidence_model(args.weights)
    cnn_matcher = None if args.no_cnn else _load_cnn_matcher(args.cnn_weights)
    n_scale, n_angle = (5, 3) if args.fast else (9, 5)

    if args.reference and args.search:
        result = run_one(args.reference, args.search, confidence_model, cnn_matcher,
                          n_scale=n_scale, n_angle=n_angle, engine=args.engine)
        text = json.dumps(result)
        print(text)
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(text)
        return

    if args.reference_dir and args.search_dir:
        out_dir = args.output_dir or Path("results")
        out_dir.mkdir(parents=True, exist_ok=True)
        ref_files = sorted(args.reference_dir.glob("*.png"))
        for ref_path in ref_files:
            search_path = args.search_dir / ref_path.name
            if not search_path.exists():
                print(f"[localize.py] skipping {ref_path.name}: no matching search image",
                      file=sys.stderr)
                continue
            result = run_one(ref_path, search_path, confidence_model, cnn_matcher,
                              n_scale=n_scale, n_angle=n_angle, engine=args.engine)
            print(json.dumps(result))
            (out_dir / f"{ref_path.stem}.json").write_text(json.dumps(result, indent=2))
        return

    ap.error("provide either --reference/--search (single pair) or "
              "--reference-dir/--search-dir (batch mode)")


if __name__ == "__main__":
    main()

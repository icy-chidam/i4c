"""
register.py (src/) -- Phase 2 mandatory entry point
------------------------------------------------------
    python register.py --input pairs.csv --output predictions.csv

Implements the addendum's exact output contract: one row per pair_id in
`pairs.csv`, columns `pair_id,x,y,theta,scale,found,score`, found=1/0
with pose columns forced to 0 when found=0, every pair_id written
exactly once even if that pair's own processing fails outright.

On pairs.csv's OWN schema: the addendum ships pairs.csv and three
ground-truthed sample pairs at T+2, two days after this addendum -- this
script is being written at T+0, so the exact column names are not yet
known. What IS fixed regardless of that schema: `pair_id`, and needing
to locate a reference image and a search image per row. `_resolve_pairs`
below therefore tries, in order: (1) common explicit path column names,
(2) any column whose name contains "ref"/"search" case-insensitively,
(3) the directory-convention fallback this project's own generator
already uses (references/<pair_id>.png, searches/<pair_id>.png next to
pairs.csv). ACTION ITEM once the real pairs.csv arrives: run this
against the three sample pairs immediately (exactly as the addendum
suggests) and confirm _resolve_pairs actually finds the right files --
see docs/validation_report_v4.md, "register.py's pairs.csv assumption,"
and adjust `_resolve_pairs` if the real column names differ from all
three guesses above.

Per-pair robustness, both required by "a missing row scores zero" and
by the hard 20s/pair timeout:
  - every pair is wrapped in a SIGALRM-based hard timeout (default 15s,
    5s of margin under the addendum's 20s cutoff) -- if a pair is
    somehow still running past that, we abandon it and emit a safe
    found=0 row rather than risk the full 20s wall (which scores zero
    AND, unlike an early abandon, leaves no row at all if it also takes
    the whole process down).
  - any other exception for a single pair (corrupt image, unexpected
    array shape, ...) is caught individually; the run continues and
    that pair gets the same safe found=0 fallback row. One bad pair
    never costs the other 199.

Weights (weights/cnn_matcher_p2.npz, weights/confidence_model_p2.joblib)
load once at startup; either failing to load degrades this exactly like
v1/v2/v3 already degrade without a CNN or confidence model -- a
documented heuristic step in, never a crash (see localizer_v4.py).
"""
from __future__ import annotations

import argparse
import csv
import signal
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from pattern_render import to_luminance
from cnn_matcher import PatchCNN
from localizer_v4 import localize_v4

try:
    import joblib
except ImportError:
    joblib = None

OUTPUT_COLUMNS = ["pair_id", "x", "y", "theta", "scale", "found", "score"]
HARD_TIMEOUT_S = 15.0          # addendum's own cutoff is 20s/pair; this leaves 5s of margin.
SCALE_RANGE = (8.0, 12.0)      # addendum's disclosed bounds, exactly.
ANGLE_RANGE = (-5.0, 5.0)


class _PairTimeout(Exception):
    pass


def _alarm_handler(signum, frame):
    raise _PairTimeout()


def _load_models(weights_dir: Path):
    cnn_matcher = None
    try:
        cnn_matcher = PatchCNN.load(weights_dir / "cnn_matcher_p2.npz")
    except Exception as e:
        print(f"[register.py] WARNING: could not load CNN weights ({e}); "
              f"continuing without the CNN landmark path.", file=sys.stderr)

    found_model, found_threshold = None, 0.5
    if joblib is None:
        print("[register.py] WARNING: joblib not importable; using the heuristic score/found "
              "fallback for every pair.", file=sys.stderr)
    else:
        try:
            bundle = joblib.load(weights_dir / "confidence_model_p2.joblib")
            found_model = bundle["model"]
            found_threshold = float(bundle.get("threshold", 0.5))
        except Exception as e:
            print(f"[register.py] WARNING: could not load confidence model ({e}); "
                  f"using the heuristic score/found fallback for every pair.", file=sys.stderr)
    return cnn_matcher, found_model, found_threshold


def _find_col(columns, candidates):
    lower = {c.lower(): c for c in columns}
    for cand in candidates:
        if cand in lower:
            return lower[cand]
    return None


def _find_col_substring(columns, substrings):
    for c in columns:
        cl = c.lower()
        if any(s in cl for s in substrings) and "id" not in cl:
            return c
    return None


def _resolve_pairs(input_csv: Path):
    """Returns a list of dicts: {pair_id, reference_path, search_path}.
    See the module docstring for the column-detection strategy.
    """
    with open(input_csv, newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return []
    columns = list(rows[0].keys())

    id_col = _find_col(columns, ["pair_id", "id", "pairid"]) or columns[0]
    ref_col = _find_col(columns, ["reference_path", "reference", "ref_path", "ref",
                                    "reference_image", "reference_file"])
    search_col = _find_col(columns, ["search_path", "search", "search_image", "search_file",
                                       "wide_search_path", "wide_search"])
    if ref_col is None:
        ref_col = _find_col_substring(columns, ["ref"])
    if search_col is None:
        search_col = _find_col_substring(columns, ["search", "wide"])

    base_dir = input_csv.resolve().parent
    out = []
    for row in rows:
        pair_id = row[id_col]
        if ref_col is not None and row.get(ref_col):
            ref_path = Path(row[ref_col])
        else:
            ref_path = Path("references") / f"{pair_id}.png"
        if search_col is not None and row.get(search_col):
            search_path = Path(row[search_col])
        else:
            search_path = Path("searches") / f"{pair_id}.png"
        if not ref_path.is_absolute():
            ref_path = (base_dir / ref_path).resolve()
        if not search_path.is_absolute():
            search_path = (base_dir / search_path).resolve()
        out.append(dict(pair_id=pair_id, reference_path=ref_path, search_path=search_path))
    return out


def _safe_row(pair_id) -> dict:
    return dict(pair_id=pair_id, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=0.0)


def _process_one(pair, cnn_matcher, found_model, found_threshold) -> dict:
    pair_id = pair["pair_id"]
    ref_raw = cv2.imread(str(pair["reference_path"]), cv2.IMREAD_UNCHANGED)
    search_raw = cv2.imread(str(pair["search_path"]), cv2.IMREAD_UNCHANGED)
    if ref_raw is None or search_raw is None:
        raise FileNotFoundError(f"could not read images for pair {pair_id} "
                                 f"(reference={pair['reference_path']}, search={pair['search_path']})")

    reference = to_luminance(ref_raw.astype(np.float32))
    search = to_luminance(search_raw.astype(np.float32))

    res = localize_v4(reference, search, scale_range=SCALE_RANGE, angle_range=ANGLE_RANGE,
                       found_model=found_model, found_threshold=found_threshold,
                       cnn_matcher=cnn_matcher)

    if res.found:
        return dict(pair_id=pair_id, x=res.x, y=res.y, theta=res.theta_deg, scale=res.scale,
                    found=1, score=res.score)
    else:
        return dict(pair_id=pair_id, x=0.0, y=0.0, theta=0.0, scale=0.0, found=0, score=res.score)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", type=Path, required=True, help="Path to pairs.csv")
    ap.add_argument("--output", type=Path, required=True, help="Path to write predictions.csv")
    ap.add_argument("--weights-dir", type=Path, default=None,
                    help="Defaults to <repo_root>/weights next to this script.")
    ap.add_argument("--timeout", type=float, default=HARD_TIMEOUT_S,
                    help=f"Per-pair hard wall-clock cutoff in seconds (default {HARD_TIMEOUT_S}, "
                         f"vs. the addendum's own 20s/pair).")
    args = ap.parse_args()

    weights_dir = args.weights_dir or (Path(__file__).resolve().parent.parent / "weights")
    cnn_matcher, found_model, found_threshold = _load_models(weights_dir)

    try:
        pairs = _resolve_pairs(args.input)
    except Exception as e:
        # Never let a bad --input crash before a single row is written --
        # a missing/unreadable pairs.csv should still leave a valid
        # (empty) predictions.csv behind, the same as the "0 rows"
        # branch just below, not an unhandled traceback and no output
        # file at all.
        print(f"[register.py] ERROR: could not read {args.input} ({type(e).__name__}: {e})",
              file=sys.stderr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS).writeheader()
        return

    if not pairs:
        print(f"[register.py] ERROR: no rows found in {args.input}", file=sys.stderr)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=OUTPUT_COLUMNS).writeheader()
        return

    have_alarm = hasattr(signal, "SIGALRM")
    if have_alarm:
        signal.signal(signal.SIGALRM, _alarm_handler)
    else:
        print("[register.py] WARNING: signal.SIGALRM unavailable on this platform; "
              "per-pair hard timeout is disabled (relying on typical runtime only).",
              file=sys.stderr)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Written and FLUSHED after every single pair, not accumulated in memory
    # and written once at the end -- so a crash, external kill, or any other
    # interruption after pair N still leaves the first N rows on disk as a
    # valid, readable predictions.csv rather than losing the whole run's
    # work to whatever happened on pair N+1 (or to the final write itself:
    # this also means an unwritable --output is caught on the very FIRST
    # row rather than after minutes of otherwise-successful computation).
    out_f = open(args.output, "w", newline="")
    writer = csv.DictWriter(out_f, fieldnames=OUTPUT_COLUMNS)
    writer.writeheader()
    out_f.flush()

    times = []
    n_written = 0
    t_start = time.time()
    try:
        for i, pair in enumerate(pairs):
            t0 = time.time()
            try:
                if have_alarm:
                    signal.alarm(int(args.timeout))
                row = _process_one(pair, cnn_matcher, found_model, found_threshold)
            except _PairTimeout:
                print(f"[register.py] WARNING: pair {pair['pair_id']} exceeded {args.timeout}s, "
                      f"emitting a found=0 fallback row.", file=sys.stderr)
                row = _safe_row(pair["pair_id"])
            except Exception as e:
                print(f"[register.py] WARNING: pair {pair['pair_id']} failed ({type(e).__name__}: {e}), "
                      f"emitting a found=0 fallback row.", file=sys.stderr)
                row = _safe_row(pair["pair_id"])
            finally:
                if have_alarm:
                    signal.alarm(0)
            dt = time.time() - t0
            times.append(dt)
            writer.writerow(row)
            out_f.flush()
            n_written += 1
            if (i + 1) % 20 == 0 or (i + 1) == len(pairs):
                print(f"[register.py] {i + 1}/{len(pairs)} pairs done "
                      f"({time.time() - t_start:.0f}s elapsed, median {np.median(times):.2f}s/pair)",
                      file=sys.stderr)
    finally:
        out_f.close()

    if times:
        print(f"[register.py] wrote {n_written} rows -> {args.output}  "
              f"(median {np.median(times):.2f}s/pair, max {np.max(times):.2f}s/pair)", file=sys.stderr)
    else:
        print(f"[register.py] wrote {n_written} rows -> {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()

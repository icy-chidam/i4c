# Validation Report v5 — Validated Against Real Organizer Data

Companion to `docs/validation_report_v4.md` (synthetic-data-only
validation, written before Applied Materials shared their actual Phase
2 generator source and 20 ground-truthed sample pairs). This report
covers what changed once real data was available to test against:
three findings (one convention question resolved, two real bugs found
and fixed), all confirmed directly against `organizer_resources/
ground_truth.csv`, and a head-to-head comparison against the
organizer's own reference baseline.

## Headline result

Run via `register.py` (the actual submission entry point, unmodified)
against all 20 real organizer-provided pairs:

| Metric | This submission | Organizer's own ZNCC baseline |
|---|---|---|
| Mean present-pair localization credit | **0.988** | 0.800 |
| Scale error (median / worst) | **0.10% / 0.68%** | 1.0% / 3.0% |
| Rotation error (median / worst) | **0.045deg / 0.208deg** | 0.35deg / 1.10deg |
| Rejection F1 (found=1 positive) | **0.970** | 0.897 |
| Precision / Recall | 0.941 / 1.000 | 1.00 / 0.81 |

Every localization/pose number is roughly an order of magnitude
tighter than the baseline; rejection F1 is higher despite starting
from a rejection classifier that (before the fixes below) scored
**zero of four** real absent pairs correctly. Reproduce with:

```bash
cd src
python register.py --input ../organizer_resources/pairs.csv \
    --output /tmp/predictions.csv
# then compare against ../organizer_resources/ground_truth.csv
```

(`pairs.csv`'s paths are relative to the organizer's own delivered
folder layout -- point `--input` at a copy of `pairs.csv` sitting next
to `reference/` and `search/` folders, i.e. run it from inside a copy
of the original `AMP_Phase 2 material` directory, or edit the paths.)

## What the organizer materials actually contained

Beyond the 3 sample pairs the addendum promised, this resource pack
included the organizer's own generator SOURCE CODE
(`phase2_pipeline.py`, `sem_imaging.py`, `presets.py`,
`structural_defects.py`, `patterns/{dram,finfet,zones}.py` -- vendored
verbatim into `src/organizer_gen/`, see that directory's own
`README_VENDORED.md`), their reference ZNCC baseline matcher
(`baseline_zncc.py`), their own scoring/calibration script
(`score_baseline.py`), 20 real image pairs with full ground truth, a
jury manifest with per-pair generation parameters, a calibration
summary, and two documents describing the design brief given to
whoever built the generator. This is a completely different tier of
information from inferring conventions out of the addendum's prose
alone, and using it directly (the addendum explicitly permits
"regenerating your own dataset") is what made the three findings below
possible.

## 1. Rotation convention — resolved, and it was inverted in our own synthetic data (not in the algorithm)

`dataset_generator.make_pair_p2` (this repo's Phase 2 data generator,
built before this resource pack was available) rotates the REFERENCE
and keeps the search axis-aligned. Direct testing against the real
organizer data showed the opposite is true of their actual generator:

**Test**: for 5 real pairs (p002, p004, p005, p006, p007) with known
`theta`, fed `_make_template(reference, scale, angle_deg=theta)`
(unnegated) and `angle_deg=-theta` (negated) and compared peak
correlation:

| pair | true theta | peak, unnegated | peak, negated |
|---|---|---|---|
| p002 | -1.20 | **0.938** | 0.608 |
| p004 | 4.60 | **0.937** | 0.354 |
| p005 | -4.90 | **0.942** | 0.466 |
| p006 | 2.30 | **0.927** | 0.435 |
| p007 | -3.10 | **0.934** | 0.509 |

Unnegated wins decisively in all 5 cases (peaks 0.93-0.94 vs 0.35-0.61)
-- not a close call. This is independently confirmed two more ways:
their own `baseline_zncc.py` builds its matching template via
`cv2.getRotationMatrix2D(center, th, 1.0/z)` with `th` used directly,
unnegated; and their generator-build design document states the
convention explicitly ("`p_search = (1/z)*R(theta)*(p_canvas -
c_canvas) + c_search`... positive theta turns the pattern
counter-clockwise as displayed") and separately warns "both signs are
internally consistent if you are generating and labelling your own
data... a solver calibrated against it inverts theta on ours" --
exactly the trap our own generator fell into.

**Why this did NOT need a fix in `localizer_v4.py` itself**:
`pyramid_scale_rotation_search` and `pose_refine.refine_pose` search
for whichever angle maximizes correlation; they don't assume or depend
on which side of a generative process "actually" carries the rotation.
Confirmed directly: theta error across all 20 real samples has a
median of 0.045deg and a worst case of 0.208deg, with zero sign
corrections applied anywhere in the search/refinement code. The
mismatch was entirely confined to `make_pair_p2`'s OWN synthetic
ground truth and anything trained against it -- see that function's
updated docstring for the specific downstream consequence (a CNN
trained on its output never sees a rotated search patch, since its
search is always axis-aligned, while real search images carry up to
+/-5deg of rotation baked into their content) and why it's marked
superseded rather than patched in place.

## 2. The downsample=4 aliasing bug — real, found on the first real-data run

Running the (at-the-time-unmodified) pipeline against all 20 real
pairs surfaced one outright failure: p003 (finfet_10nm, z=12.00 exactly,
theta=0.00 exactly) localized 321px from ground truth, converging on
scale=8.75 instead of the true 12.0.

**Diagnosis**: `pyramid_scale_rotation_search`'s coarse stage
downsamples both images by a factor of `downsample` (previously 4)
before correlating, purely for speed. At scale=12 (max zoom, smallest
template) the finest disclosed presets have periods as small as
3.3-4 search-px (`finfet_7nm`'s 40nm fin pitch at 12nm/search-px
resolution; `finfet_10nm`/`dram_dense` sit at 4.0px). A 4x downsample
compresses a 4px period under 1px -- aliased away entirely -- so the
coarse grid has nothing left to correlate against except noise, and
can converge anywhere.

**Fix**: reduced the default `downsample` from 4 to 2.

**Validation**:
- All 20 real pairs re-run at downsample=1/2/3: p003 resolves to
  0.38-0.49px error at every value except 4 (peak 0.88 vs 0.70).
- downsample=1 vs 2 across all 16 real present pairs: IDENTICAL
  max error (1.20px) at the coarse-grid stage alone; downsample=2 runs
  in very close to half the time (median 0.73s vs 1.37s for that
  stage).
- Stress-tested downsample=2 against the two worst-pitch presets
  (`finfet_7nm`, `finfet_10nm`, `dram_dense`) at z=11.5-12.0 across
  multiple thetas, severities 0-4, and seeds: 17/17 completed runs
  succeeded (errors 0.28-3.28px); the one non-completion was the
  ORGANIZER'S OWN generator refusing to emit that combination at all
  (`RuntimeError: best verifiable margin 0.006 below floor 0.02`) --
  not a failure of this pipeline.

## 3. Absent-pair rejection was completely broken against real decoys — found, diagnosed, fixed

Same first real-data run: all 4 real absent pairs (p015-p018) came back
as confident false positives, scores 0.81-0.95 -- not borderline
misses, confidently wrong.

**Diagnosis**: the found/score classifier had been trained entirely on
`make_pair_p2`'s own absent-pair mechanism ("two independently drawn
parameter sets of the same style"). The organizer's real mechanism is
much more specific -- a decoy reference is cut from a canvas with
`mat_size_nm * 0.55` and `strip_width_nm * 2.1` (smaller mats, wider
strips than the search's own canvas), biased onto a mat/strip
junction -- and the resulting correlation-feature distribution didn't
transfer. (Their own generator-design document independently flags
this exact risk: "If you cut the decoy reference from a canvas with
the SAME zone geometry, you will find that absent pairs score HIGHER
correlation peaks than present pairs... teaches solvers to reject
confident matches.")

**Fix**: retrained the classifier entirely on data generated by the
vendored `organizer_gen.phase2_pipeline.generate_phase2_sample`
(`Phase2Params(present=False, boundary_bias=0.70, ...)`, matching their
own generation script's exact parameters) -- 320 pairs across all 12
architecture presets, full zoom/theta/severity ranges, ~20% absent.
Also corrected the F1 convention used for threshold selection (see
next section).

**Validation**: re-running against the 4 real absent pairs: 3/4
correctly rejected (was 0/4), the remaining one (p018) scoring 0.425 --
a genuinely borderline call, not a confident miss (the lowest-scoring
real PRESENT pair, p012, scores 0.453 -- a 0.028 margin on n=4 real
absent pairs, too thin to chase further without overfitting to this
exact small sample; see "Known limitation" below). Full-dataset
rejection F1 (all 20 real pairs): 0.970, versus the organizer
baseline's own 0.897.

## 4. Rejection F1's positive class — corrected back to standard, using their own scoring code as ground truth

`docs/validation_report_v4.md` (section 4) concluded, from the
addendum's prose ("a team that never rejects anything scores zero"),
that F1 must be computed with reject (found=0) as the positive class.
The organizer's own `score_baseline.py` — actual, executable,
organizer-provided scoring code, strictly more authoritative than our
own reverse-engineering of a sentence — computes it the standard way:

```python
tp = sum(1 for r in rows if r["present"] and r["pred_present"])
fp = sum(1 for r in rows if not r["present"] and r["pred_present"])
fn = sum(1 for r in rows if r["present"] and not r["pred_present"])
# precision = tp/(tp+fp), recall = tp/(tp+fn) -- found=1 IS the positive class
```

This is directly confirmed by their own `baseline_calibration.txt`:
threshold 0.55 gives "TP=13 FP=0 FN=3, precision=1.00 recall=0.81
F1=0.897" -- which only reproduces from their 16 present / 4 absent
split under the standard (found=1 positive) formula. `train_confidence_p2.py`
and `evaluate_p2.py` are both reverted to `pos_label=1` (the
scikit-learn default) accordingly; the earlier `pos_label=0` "fix" is
superseded by this stronger evidence. (v4's reasoning wasn't
unreasonable given what was available at the time -- an ambiguous
sentence versus no code to check it against -- it was simply
overridden by better information becoming available, which is exactly
why this note exists instead of silently rewriting history.)

## 5. CNN landmark path: confirmed inactive on real data, gated more conservatively rather than retrained

The original (Phase 1) CNN was trained to recognize an artificial
"fiducial" marker -- a bright square the Phase 1 team's OWN practice
generator adds to ~50% of samples. The real organizer generator has no
such concept; uniqueness there comes from genuine per-feature
positional jitter (every line/via/fin is independently jittered a
little, so periodic repeats are only APPROXIMATELY, not exactly, self-
similar) plus the verify-and-retry gate discussed below, not a planted
marker.

**Checked, not assumed**: ran all 20 real pairs and logged
`cnn_used`/`cnn_score`. Result: `cnn_used=False` on all 20,
`cnn_score` between 0.0000 and 0.0148 throughout -- never once close
to `cnn_accept_thresh` (0.5). The CNN is confirmed harmless (it
correctly reports "no landmark here" for patches that never contain
one) but also confirmed to contribute nothing on real data currently.

**Decision**: rather than retraining it against a materially harder,
unvalidated objective ("does this candidate look more correct than
that one," with no planted marker to key on) under real time
constraints, `localizer_v4.py`'s CNN-override trigger now additionally
requires the classical evidence to already flag genuine ambiguity
(same gate the lattice-enumeration path already used) -- see that
file's updated comment. This is a strictly conservative change: the
CNN can now only break a tie the classical peak comparison already
flags as one, never overrule a clear classical winner outright,
closing off the one way a future retrain (or a corrupted/mismatched
weights file) could actively hurt a result instead of just sitting
idle. Retraining it properly (real pattern crops as positives, other
candidate sites or decoy crops as negatives, evaluated against held-
out real-generator data before trusting it) is a reasonable next step
but wasn't pursued here given the core pipeline already leads the
baseline by such a wide margin without it.

## 6. Known limitation carried forward

p018 (real absent, `finfet_14nm`, z=12.0, severity 0) remains a false
positive at score 0.425 against the classical/lattice pipeline alone
(no CNN contribution either way). Investigated: `best_score=0.451`,
only one NMS peak survives (`ratio=1.0`, i.e. not classically flagged
as ambiguous), so it resolves via `classical_center_fallback`. The
margin against the nearest real present pair's score (p012, 0.453) is
0.028 on a 4-sample real absent set -- accurately reflects genuine
model uncertainty (scores in the 0.4-0.45 range, not the 0.81-0.95
range seen before the fix) rather than a confident miss, but chasing a
threshold that flips this one case would be fitting n=4, not fixing a
real, generalizable gap. Documented rather than hidden, consistent
with this project's own practice throughout.

## Files changed this session

| File | Change |
|---|---|
| `src/organizer_gen/` | **new** -- vendored organizer generator source, see its own `README_VENDORED.md` |
| `organizer_resources/` | **new** -- organizer's ground_truth.csv, pairs.csv, manifest_jury.csv, baseline_calibration.txt, README (images not re-bundled, see zip size note) |
| `localizer_v4.py` | `downsample` default 4->2 (see #2); CNN-override now also requires classical ambiguity (see #5) |
| `weights/confidence_model_p2.joblib` | retrained on organizer-generator data, standard F1 threshold tuning (see #3, #4) |
| `src/dataset_generator.py` | `make_pair_p2` docstring marks it superseded for new training data (see #1) |
| `train_confidence_p2.py`, `evaluate_p2.py` | F1 `pos_label` reverted 0->1 (standard), see #4 |
| `gen_batch_organizer.py` (scratch, not in submission zip) | generates training data via the vendored generator |

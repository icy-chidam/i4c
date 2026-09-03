# Validation Report v4 — Phase 2 (Registration under Unknown Pose)

Companion to `docs/validation_report_v3.md` (Phase 1, untouched). This
report covers only what changed for the addendum: four findings from
direct measurement (two were real bugs, two were conventions that
needed checking rather than assuming), the resulting fixes, and the
evidence for each. Everything here is reproducible from this repo's own
scripts — commands are given inline.

## 0. What stayed the same

`localizer.py` (v1), `localizer_v2.py`, `localizer_v3.py`,
`dataset_generator.py`'s `make_pair`/`generate`, and every existing
call site are **byte-for-byte unaffected** by this addendum's changes —
checked, not assumed:

```
$ python evaluate.py --data-dir ../data/test --output-dir /tmp/v3_recheck
# mean=54.08px median=0.07px worst=257.54px with_fiducial=100.0% pure_periodic=8.3%
# identical to the numbers already in validation_report_v3.md, before AND after
# every edit in this addendum (re-checked specifically after lattice_localizer.py's
# subpixel=True parameter was added, since that file is shared with v2/v3).
```

`localizer_v4.py` is additive: same coarse-to-fine pyramid search, same
CNN-landmark path, same ambiguity-gated lattice-enumeration fallback as
v2/v3. Two capabilities are added (continuous pose refinement, a found/
score model) — see §2–4.

## 1. Rotation sign convention — checked, not a bug

Phase 2 scores rotation to ±0.25° for full credit. Before relying on
`_make_template`'s `angle_deg` as "the value to report," we verified
what it actually represents, since a sign error here would silently
zero the entire 10-pt rotation component.

**Test 1** (`dataset_generator.make_pair` + a fine angle sweep fed to
`localizer._make_template`, holding scale fixed at the true value):
generated `dram`/`finfet` pairs at known `theta_deg` (+5.0, −4.0), swept
candidate angles at 0.25° resolution. The argmax angle was **exactly
the negation** of the generator's `theta_deg` in all four trials, with
peak correlation 0.91–0.93 confirming a genuine (not noise) match.

**Test 2** (independent check of OpenCV's own convention): built a
test image with a single bright point due east of center, applied
`cv2.getRotationMatrix2D(center, +90, 1.0)` + `warpAffine`. The point
moved to due north — confirming +angle = CCW as normally displayed,
matching OpenCV's documented convention.

**Conclusion**: `_make_template`'s `angle_deg`, when it maximizes
correlation, already equals "the CCW rotation that makes the reference
look like its appearance in the search" — exactly the addendum's own
definition of θ. **No change to the search/report logic.** The only
consequence: our own synthetic ground truth's `rotation_deg` field
(the physical capture-rotation parameter) is the *negative* of what
should be scored, so `dataset_generator.make_pair_p2` stores a second,
unambiguous field, `theta_report_deg = -rotation_deg`, and every
Phase 2 script compares against that field, never `rotation_deg`
directly.

## 2. Scale quantization — real bug, fixed

**Symptom**: sweeping scale at the exact true angle (dram sample,
period ≈13px) showed a visible staircase — `s=8.75` and `s=8.80` gave
*identical* correlation (0.3985), as did several other adjacent pairs.

**Diagnosis**: `localizer._make_template` resizes to
`new_w = round(reference_size / scale)`, an integer. `round(1000/9.25)
= round(1000/9.30) = 108` — many distinct `scale` values collapse onto
the same integer template size, and `cv2.resize` to that fixed integer
size discards the rest of the information about which exact `scale`
was requested. Plateau width is ≈`ref_size / template_px²`, which at
this project's 8–12x scale (83–125px templates) is **0.06–0.15** — the
same order as Phase 2's 1% top-credit tolerance (≈0.08–0.12 absolute).
No amount of optimization on top of a quantized objective can resolve
finer than the plateau it's standing on.

**Fix** (`pose_refine.make_template_continuous`): a single rotate+scale
affine (`cv2.warpAffine`, `WARP_INVERSE_MAP`) samples directly into a
**fixed-size** output canvas, with `scale` acting purely on the
*sampling density*, never the canvas's own integer dimensions. Source
is Gaussian-pre-filtered with `sigma = 0.5 * scale` before sampling
(mipmap-style) since an ~8–12x downsample via plain bilinear
interpolation aliases badly otherwise. Used only inside the new local
refinement stage (`pose_refine.refine_pose`); the coarse/fine grid
stages still use the original `_make_template` (their precision needs
are already met by the existing design — this is a refinement-stage
fix, not a search-stage one).

**Validation** (7 controlled trials, known scale/angle, all three
styles, scale spanning 8.05–11.95, rotation spanning −4.95° to +4.95°):

| Metric | Before fix | After fix |
|---|---|---|
| Scale within 1% (top credit tier) | not measured pre-fix in this form* | 7/7 (100%) |
| Rotation within 0.25° (top credit tier) | 4/7 clean, 3/7 landed on the wrong side of a plateau boundary | 7/7 (100%) |

*The pre-fix staircase made "scale error" itself an artifact of which
plateau the coarse grid happened to land on rather than a meaningful
number; the fix is what made this table meaningful to compute at all.
Reproduce: `pose_refine.refine_pose` docstring has the exact trial
setup; ad hoc reproduction script used during development swept
`np.linspace(scale-0.6, scale+0.6, 25)` at the true angle and printed
the correlation value at each point — the plateau is visible directly.

## 3. Local refinement jumping to a competing periodic repeat — real bug, fixed

**Symptom**: after fix #2, a landmark-free `dram` pair (period ≈13px)
that the lattice-enumeration tie-break had correctly localized (0.13px
from the true nearest-to-centre site, verified by hand) came out of
`localize_v4` **116px away** — `tie_break_method` still reported
`"lattice_enumeration"`, so the error was downstream of the tie-break.

**Diagnosis**: `refine_pose`'s local crop is sized off the *template*
(`out_size * crop_radius_factor`, ≈1.6–2x template radius, ~150–200px)
— reasonable when template size and period are comparable, but at
8–12x scale a small-period pattern's template (~100px) can be 10–15x
its own period (~7–13px). The crop can contain a dozen-plus repeats of
the pattern, and `_peak_at_pose`'s unconstrained `cv2.minMaxLoc` finds
the *global* max inside that crop — which, from noise alone, can be a
neighboring repeat rather than the one at the crop's center. Confirmed
precisely: `(451.97 − 504.01) / 12.88 = −4.04` and
`(608.66 − 505.12) / 12.99 = +7.97` — the wrong answer sat *exactly* 4
and 8 whole periods from the intended site in x and y respectively.
This is not a scale/rotation refinement problem; it silently re-litigates
which site is correct, undoing work the global tie-break had already
done (see `pose_refine.py`'s module docstring for why that distinction
matters).

**Fix**: `_peak_at_pose` gained an `expected_xy`/`window_radius`
parameter (default 4px) that restricts the argmax search to a small
box around the crop's own center, instead of the crop's global max.
`refine_pose` now always passes the anchor's own crop-local position
as `expected_xy`, held fixed for the whole optimization.

**Validation**: re-ran the exact 20-trial pure-periodic test (fixed
seed 4242, `dram`/`dram_arcuate`/`finfet`, no fiducial) that first
exposed this, scoring against the corrected nearest-to-centre target
(`spec_fair.localization_target`, §5):

| | Before fix | After fix |
|---|---|---|
| Within 5px | 0/20 (0%) | 17/20 (85%) |
| Within 1px | 0/20 (0%) | 13/20 (65%) |
| Median error | 104.8px | 0.74px |

Re-ran the fix #2 controlled trials (7/7 within top scale/rotation
tier) afterward to confirm no regression — unchanged, 7/7 both metrics.

## 4. Rejection F1's positive class — verified against the addendum's own claim

The addendum states: *"A team that never rejects anything scores zero
[on the 15pt rejection metric]."* We checked what F1 definition this
implies rather than guessing:

```python
y_true = [1]*140 + [0]*40   # 140 present, 40 absent (Set A+B / Set C proportions)
y_pred = [1]*180             # "never rejects anything"
f1_score(y_true, y_pred, pos_label=1)  # -> 0.875  (found=1 is positive)
f1_score(y_true, y_pred, pos_label=0)  # -> 0.0    (reject/absent is positive)
```

Only `pos_label=0` (rejection — correctly detecting *absence* — as the
positive class) reproduces the addendum's stated zero. This is used
throughout (`register.py` doesn't compute F1 itself, but
`train_confidence_p2.py` and `evaluate_p2.py` both do, and both now use
`pos_label=0`). It also matters for threshold selection: a
component-level F1 tuned the wrong way picked threshold 0.14 from an
F1-vs-found-as-positive curve; the corrected metric plus a full-formula
sweep (next section) moved this to 0.12.

## 5. Choosing the found threshold

Component-level F1 alone doesn't capture the addendum's own asymmetry:
a false negative on a present pair also forfeits that pair's *entire*
localization + pose credit (`found=0` writes zeros in the pose columns
per the output contract), while a false positive on an absent pair
only costs the rejection metric (Set C isn't scored for localization at
all). `evaluate_p2.py` simulates the full formula (weighted
localization + pose + rejection F1 + calibration AUC) across a
threshold sweep on ~165 held-out synthetic pairs at realistic Set A/B/C/D
proportions (35/35/20/10%):

| threshold | loc (40) | pose (20) | rejection (15) | calibration (10) | **total (95)** |
|---|---|---|---|---|---|
| 0.00 | 30.85 | 15.50 | 0.00 | 8.21 | 54.56 |
| 0.05 | 30.70 | 15.46 | 3.64 | 7.86 | 57.65 |
| **0.10** | **30.12** | **15.21** | **7.14** | **7.42** | **59.89** |
| 0.14 | 29.60 | 15.00 | 7.66 | 7.41 | 59.67 |
| 0.25 | 28.42 | 14.37 | 8.95 | 7.35 | 59.08 |
| 0.40 | 26.41 | 13.50 | 9.40 | 7.47 | 56.79 |
| 0.70 | 19.23 | 9.40 | 7.36 | 7.59 | 43.58 |

Total peaks in the 0.10–0.25 range and falls off on both sides — low
enough to avoid needlessly forfeiting localization credit on
uncertain-but-present pairs, high enough to still catch a meaningful
share of absent ones. Shipped threshold: **0.12** (`spec_fair.py`'s
tolerance and the classifier's own validation split were used together
to land inside this plateau rather than exactly on the single-sample
optimum, which is somewhat noisy at this sample size).

This IS the main remaining soft spot: rejection F1 lands at 0.48–0.75
across independent seeds (see `failure_analysis_p2.pdf` §5–6) —
`finfet`-style absent pairs specifically show higher residual
correlation than `dram`/`dram_arcuate` even when genuinely absent
(simple, repetitive geometry gives less to discriminate on), and this
is disclosed rather than hidden. Improving this further (style-specific
features or thresholds) was identified but not pursued given time —
flagged here for anyone picking this back up.

## 6. Dependency check for the reference machine's Python 3.11

This repo was developed/tested on Python 3.12.3 (matching
`results/results.json`'s recorded environment); the addendum's
reference machine runs Python 3.11. Checked directly against PyPI's
package metadata (not assumed) that every pinned version in
`requirements.txt` ships a working wheel for `cp311`/manylinux x86_64:
`numpy==2.4.4`, `scipy==1.17.1`, `scikit-learn==1.8.0` ship explicit
`cp311` wheels; `opencv-python-headless==4.13.0.92` ships `cp37-abi3`
(forward-compatible stable-ABI wheels covering 3.11); `joblib==1.5.3`
and `pandas==3.0.2` ship pure-Python / `cp311` wheels respectively. No
pin changes needed.

## 7. What's new, file by file

| File | Status | Purpose |
|---|---|---|
| `pattern_render.py` | extended (additive) | `perturb_geometry`, `apply_degradation`, `add_charging_artifact`, `add_scan_distortion`, `add_defocus`, `to_pseudo_optical_rgb`, `to_luminance` |
| `dataset_generator.py` | extended (additive, `--phase2` flag) | `make_pair_p2`/`generate_p2`: disclosed scale/rotation bounds, absent pairs, degraded severities, optical pairs |
| `lattice_localizer.py` | extended (additive `subpixel=` param, default False) | subpixel autocorrelation peaks; new `refine_lattice_from_peaks` (least-squares sharpening from NMS peaks) |
| `pose_refine.py` | **new** | continuous scale/rotation/subpixel-xy refinement (§2, §3) |
| `localizer_v4.py` | **new** | Phase 2 engine: v3's tie-break logic + continuous refinement + found/score model |
| `train_cnn_matcher_p2.py` | **new** | retrains the *same* `PatchCNN` architecture on the disclosed scale range + degraded severities (rotation untouched — CNN patches come from the search image, which is never rotated in this capture model) |
| `train_confidence_p2.py` | **new** | trains the found/score classifier: pure-presence label (§ design note below), 8-feature vector |
| `spec_fair.py` | **new** | nearest-to-centre comparison target for landmark-free pairs, corrected vs. `evaluate.py`'s original (no false rotation correction — search is never rotated in this capture model; adaptive k-range for small periods) |
| `register.py` (+ root wrapper) | **new** | mandatory entry point; flexible `pairs.csv` column detection, SIGALRM hard per-pair timeout, per-pair exception isolation |
| `evaluate_p2.py` | **new** | full addendum-rubric scoring simulation (§5) |

**Design note on the found/score label**: trained on pure presence
(`gt["found"]`), *not* "present and we localized it correctly." Those
are different questions — a present pair where the periodic tie-break
picked a different-but-valid repeat is a localization problem (scored
separately), not a presence problem, and reporting `found=0` on it
gains nothing (localization credit is already gone once we're
uncertain) while losing rejection F1 for no reason. Verified this
doesn't just paper over the distinction: mean scores on a
never-shown-as-its-own-class sanity split still order correctly
(present+accurate > present+missed > absent) — see
`train_confidence_p2.py`'s `build_dataset` docstring.

## 8. Adversarial pressure-testing pass (post-submission hardening)

Run after the initial build, specifically trying to break `register.py`
rather than confirm it works. Found and fixed three real issues; a
dozen+ other adversarial cases (listed below) held up with no change
needed. All fixes re-verified against the existing test suite and
`evaluate_p2.py` afterward — no regressions.

**Fixed:**

1. **Top-level crash on an unreadable `pairs.csv`.** A missing file,
   bad permissions, or any other read error before the per-pair loop
   started threw an unhandled traceback — zero rows written, not even
   the ones we could have processed. `_resolve_pairs` is now wrapped;
   any failure there produces a valid (empty) `predictions.csv` and a
   clear stderr message instead of a crash.
2. **All-or-nothing final write.** Predictions were held in memory and
   written once at the end — a crash, kill, or write failure on the
   very last line would have lost every row computed before it, no
   matter how long the run had already taken. Rewritten to open the
   output file once, write the header, and write + flush each row
   immediately after that pair finishes. Verified directly: killed a
   simulated run mid-way through pair 3 of 4, and the first 2 rows were
   already valid and complete on disk.
3. **Degenerate (near-blank) input scored a confident false positive.**
   Two all-black 1000x1000 images — zero information, no possible match
   — came back as `found=1`, `score=0.92`, with a specific-looking
   (x, y, scale, theta). Root cause, confirmed directly: `cv2.matchTemplate`'s
   `TM_CCOEFF_NORMED` is a covariance-over-std-devs ratio; when either
   image has ~zero variance that's a 0/0 division, and OpenCV's own
   handling of that case returns exactly `1.0` everywhere — a "perfect
   match" artifact, not a real signal. This is the worst kind of
   failure (confidently wrong, not just wrong), and nothing in the
   pipeline previously guarded against it. Fixed: `localize_v4` now
   checks `std(reference)` and `std(search)` before running any
   correlation; near-constant input short-circuits straight to
   `found=False, score=0.0` (and runs in ~0.02s instead of ~1.3s, a
   free efficiency win on top of correctness). Re-verified this does
   NOT false-trigger on real patterns (std ~50-95 on ordinary generated
   pairs, comfortably above the threshold).

**Tried and held up (no change needed):** empty/header-only `pairs.csv`;
unrecognized column names (substring match still finds them); pair_ids
containing slashes, spaces, or commas (CSV-escaped correctly on output;
per-pair failures degrade to a safe row rather than crashing); duplicate
pair_ids; corrupted/truncated image files; a 20x20 image (hits an
internal OpenCV assertion, caught by the per-pair handler); non-square
and mismatched reference/search dimensions; RGBA input (alpha silently
and harmlessly dropped by `to_luminance`); the SIGALRM timeout actually
firing under a simulated 30s hang (recovered at the configured cutoff,
continued correctly to the next pair, no cascading failure across
multiple timeouts in one run); corrupted or entirely missing CNN/
confidence-model weight files (each degrades to its documented
fallback); exact boundary scale values (8.0, 12.0 precisely, output
correctly stays clipped in-range).

**Not fixed, lower priority:** RGBA input works by dropping alpha
rather than by deliberate design (harmless in practice — alpha carries
no image information for this problem, but noted as incidental, not
intentional, correctness). Path-traversal-style pair_ids (e.g.
containing `../`) aren't specially sanitized — not a realistic risk
against an organizer-controlled `pairs.csv`, so left as-is rather than
adding complexity against a threat model that doesn't apply here.

## 9. Known open item

The addendum's own `pairs.csv` and 3 ground-truthed sample pairs ship
at T+2, two days after this addendum — after this development pass.
`register.py`'s `_resolve_pairs` handles this by trying, in order,
several plausible explicit column names, then a substring match on
"ref"/"search", then falling back to this project's own
`references/<pair_id>.png` / `searches/<pair_id>.png` convention.
**Action item**: the moment the real `pairs.csv` arrives, run
`register.py` against it immediately (exactly as the addendum
suggests) and confirm `_resolve_pairs` finds the right files before
the T+7 freeze; adjust the column-name candidates in `register.py` if
needed. Nothing else in the pipeline depends on this assumption.

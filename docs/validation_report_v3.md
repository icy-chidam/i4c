# v3 addendum: coarse-to-fine pyramid search + noise-matched CNN

Follow-up to `docs/validation_report.md` (read that first for the v1->v2
diagnosis and lattice-enumeration rationale, which v3 keeps unchanged).
This addendum covers two further changes, and -- importantly -- a
methodology error I made while first validating them, caught by a final
clean-machine regression test before shipping. Leaving that in rather
than quietly fixing it, because it's a useful example of exactly the
kind of check this whole exercise has been about.

## Change 1: coarse-to-fine pyramid search (real, isolated, reproducible)

`pyramid_search.py` / `localizer_v3.py` replace v1/v2's flat 45-
hypothesis single-resolution grid (9 scales x 5 angles, every hypothesis
a full-resolution correlation against the full 1000x1000 search image --
measured at ~1.6-2.0s, essentially the entire runtime of one
`localize()` call) with a two-stage search: a cheap 4x-downsampled pass
over the same 45 hypotheses to find the right scale/rotation
neighbourhood (~0.35s), then a small full-resolution pass (default
3x3=9 hypotheses) centred on it.

The CNN-landmark decision path and the ambiguity-gated lattice-
enumeration fallback are untouched, line for line, from
`localizer_v2.py` -- see that file's diff against `localizer_v3.py` if
you want to confirm that directly.

## Change 2: noise-matched CNN training (real, but NOT v3-exclusive -- see correction below)

Adding per-sample noise-level variation to `dataset_generator.py` (done
in the v2 pass, to close a different validation-checklist gap -- see
`docs/validation_report.md` section 1.3) exposed that
`train_cnn_matcher.py` trained the CNN at a single fixed noise level,
sometimes lower than what the generator can now produce (up to 2.5x/4x
multiplier). Some genuinely fiducial-bearing test samples at high noise
were failing the CNN's confidence gate purely because they were noisier
than anything the CNN had seen in training. Fixed by widening the CNN's
training noise range to match the generator's.

**This is a data/training fix, unrelated to which search algorithm is
used.** `train_cnn_matcher.py` now always trains this way -- there was
no reason to keep the worse, noise-mismatched training procedure around
as a selectable option. So this benefit applies equally whichever engine
(`v1`, `v2`, or `v3`) loads the resulting `cnn_matcher.npz`.

## A methodology error, caught and fixed before shipping

My first pass at validating change 2 compared the OLD `cnn_matcher.npz`
(trained before this session, at the original fixed noise level) against
a NEW `cnn_matcher_v3.npz` (trained after patching `train_cnn_matcher.py`)
saved under a different filename, and got a real, honestly-measured
83% -> 100% fiducial-accuracy jump from that comparison. That number is
real -- I did measure it, with two genuinely different weight files.

But when I then ran a full clean-machine regression test (delete
everything, regenerate from nothing, exactly what re-running this
repo's own Quickstart does), both `cnn_matcher.npz` AND
`cnn_matcher_v3.npz` came from training runs of the SAME (already-
patched) script, with the same default seed -- so they came out
byte-identical (confirmed: matching md5 hashes). The "v2 uses the old
CNN, v3 uses the retrained one" framing I'd written docs around does
NOT reproduce from a clean checkout, because there is no "old CNN" left
in the repo to compare against -- I'd only kept one training script, and
it now always produces the improved version.

Net effect: the 83% -> 100% jump is real as a measurement of "old
training procedure vs new training procedure," but it is **not** a
property of `--engine v3` specifically, and presenting it as one (as an
earlier draft of this file did) would have been misleading. Fixed by
consolidating to one `cnn_matcher.npz` used by all three engines, so the
comparison below isolates exactly what actually differs between them:
the search algorithm, nothing else.

## Validated results (corrected methodology: same test set, same single CNN, engine-only comparison)

| Engine | Mean time/pair | Fiducial fair-target within 5px | Periodic fair-target median error |
|---|---|---|---|
| v1 (original submission) | 1743ms | 100% | 20.52px |
| v2 (lattice enumeration) | 1969ms | 100% | 15.13px |
| **v3 (+ pyramid search)** | **1099ms** | 100% | 15.13px |

Reading this honestly, with the CNN now held constant across all three:

- **v1 -> v2** (lattice enumeration replacing NMS-survivor tie-break):
  the periodic-case improvement documented in `docs/validation_report.md`
  holds up (20.52px -> 15.13px fair-target median, ~26% better), at a
  real cost in runtime (the FFT autocorrelation step adds ~200-400ms
  when it fires).
- **v2 -> v3** (pyramid search replacing the flat grid): **~1.8x
  faster/pair** (1969ms -> 1099ms), with **no accuracy change** in
  either regime at the default 3x3 fine-grid setting -- a 3x3 grid
  re-centred on the coarse winner uses the same step size as the
  original flat grid (fewer wasted far-away hypotheses, not finer
  resolution near the optimum), so this change is honestly efficiency-
  only at its default setting, not an accuracy win. A denser fine grid
  (`n_scale_fine=5, n_angle_fine=5`, pass explicitly) trades part of
  that speed back for a small additional periodic-case gain -- see the
  code comment in `localizer_v3.py` for the specific numbers from that
  setting.
- **The CNN noise-retrain's real, measured 83% -> 100% fiducial
  improvement applies uniformly** once `train_cnn_matcher.py` is (re)run
  with the current code -- it is not attributable to, or exclusive to,
  any one engine.

## What this does and doesn't claim

The honest headline: **v3 is meaningfully faster than v2 (and v1) with
no accuracy regression anywhere**, and **the CNN training fix is a real,
separate improvement that benefits the whole pipeline regardless of
engine choice**. Neither claim depends on the other, and neither should
be overstated as more than it is -- the pyramid search at its default
setting is an efficiency change, not an accuracy one; the periodic
(ambiguous) case's accuracy ceiling is still set by the v1->v2 lattice
change and by the inherent geometry of the `dram_arcuate` scoring
caveat already documented in `docs/validation_report.md`.

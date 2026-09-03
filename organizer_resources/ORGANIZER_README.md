# Drift-Sense Phase 2 — 20 Sample Pairs

Applied Materials Confidential · jury/organizer reference material

These 20 pairs are a **worked reference implementation** of the Phase 2 addendum: unknown zoom, unknown rotation, and reference-absent pairs. They are here so the jury can see what the spec produces before committing to the full 200-pair set, and so the three unscored sample pairs given to participants can be cut from something already validated.

---

## 1. Provenance

Built on top of the generator published at
`huggingface.co/spaces/aayushraina21/drift-sense-synthetic-data`.

The container running this had no outbound network, so the upstream modules were transcribed from that space's published source rather than cloned: `src/presets.py`, `src/structural_defects.py`, `src/patterns/{dram,finfet,zones}.py`, `src/sem_imaging.py`. Those files are unmodified in substance — the SEM physics (Gaussian beam PSF, Poisson shot noise, raster drift/shear, charging streaks, astigmatism, vignetting, barrel distortion, speckle, impulse noise), the DRAM 6F² and FinFET structure models, the 6-preset-per-family scaling ladder, and the mat/strip zone composition are all upstream work. `app.py`, `generate_family_dataset.py` and the layer-visualisation tooling weren't needed and weren't transcribed.

**Everything Phase-2-specific is in `src/phase2_pipeline.py`**, which is new.

---

## 2. What Phase 2 changes in the pipeline

Upstream fixes the zoom at exactly 10× — not by resizing, but because the reference is 1 nm/px and the search image is 10 nm/px on a 10 000² fine canvas. Three consequences, and three changes:

| | Upstream (Phase 1) | Here (Phase 2) |
|---|---|---|
| Zoom | 10×, from the 1 vs 10 nm/px ratio | `z ~ U[8,12]`, so the fine canvas is `1000·z` px — 8 675² to 13 009² |
| Rotation | An imaging artifact only | A pose parameter. Search raster rotated by `θ ~ U[-5,+5]°`, and θ is ground truth |
| Reference | Always cut from the search canvas | 4 of 20 pairs cut from an independent canvas — no true instance |

Geometry is one affine, `canvas → search`: rotate by +θ about the canvas centre, scale by 1/z, translate to centre a 1000×1000 output. Ground truth is the reference crop's centre pushed through that same affine, so location, θ and scale are consistent by construction. The affine round-trips to 2.5e-13 px and recovers z and θ exactly (see §6).

Two implementation notes worth knowing:

- **Anti-aliasing.** Upstream used `INTER_AREA` for the integer 10× downsample. `warpAffine` has no `INTER_AREA`, so a `z`-wide box prefilter stands in for it. The beam PSF alone is not a sufficient low-pass at z=12.
- **Ground truth tracks the distortions.** Raster drift and barrel distortion are applied *after* the pose affine, so they displace the true feature away from the affine coordinate — worth 1–2 px at high severity, which would have been unexplained label error sitting right inside the scoring tiers. The exact per-row drift vector is captured and the GT point is pushed through both maps (Newton inversion for the barrel cubic). Labels are exact.

---

## 3. Files

```
pairs.csv                 give to participants — pair_id, search_path, reference_path
ground_truth.csv          WITHHOLD — pair_id, present, x, y, theta, scale
manifest_jury.csv         jury only — 32 columns: every generation parameter, per pair
baseline_calibration.txt  naive-baseline run over all 20 pairs
contact_sheet.png         all 20 search images with GT pose boxes + reference insets
reference/pNNN.png        1000×1000, 1 nm/px
search/pNNN.png           1000×1000, z nm/px
```

`ground_truth.csv` for absent pairs carries `present=0` and zeros in the pose columns.
Set D (`p019`, `p020`) is 3-channel; everything else is single-channel.

---

## 4. Coverage

20 pairs, poses hand-specified rather than sampled so the span is provable rather than probable.

| Set | Pairs | Content |
|---|---|---|
| A — nominal | 8 | Reference present, mild noise, full pose range |
| B — degraded | 6 | Reference present, severity levels 1/2/3/4/2/3 |
| C — absent | 4 | No true instance. `present=0` |
| D — optical | 2 | RGB analogue, bonus only |

- **Zoom** spans both endpoints: 8.00, 8.25, 8.45, 8.60, 9.05, 9.15, 9.40, 9.80, 10.00, 10.30, 10.60, 10.75, 11.20, 11.30, 11.75, 11.90, 12.00 (×3)
- **Rotation** spans both endpoints and zero: −4.90 to +4.90, including three pairs at exactly 0.00
- **Presence**: 16 present / 4 absent = 20%, matching the addendum
- **9 architecture presets** across both families, so no team can tune to one pitch

---

## 5. Calibration — read this before scaling to 200

The naive ZNCC baseline (brute-force over a 0.5× / 1.0° grid) scores:

| | Mean credit | Median error |
|---|---|---|
| Set A | **1.000** | 0.40 px |
| Set B | 0.467 | 0.51 px |
| Set D | 1.000 | 0.15 px |
| **Overall (present)** | **0.800** | — |

**Set A is too easy.** The naive method gets full credit on all 8 pairs. My jury note targeted a naive baseline of 0.30–0.55 on the weighted total; 0.80 is well above that, which means a 200-pair set built to this recipe would not spread 30 teams. Set B is doing all the discriminating — severity 3 and 4 defeat the baseline outright (p011, p012, p014 → credit 0).

**Recommendation:** for the real set, shift the Set B severity distribution toward levels 3–4 and raise Set A's floor to roughly the current level 1. As a sample set intended for I/O validation, easy is fine — the addendum says the three participant samples are Set-A-like and unscored.

**Rejection is not trivially threshold-separable, and that is deliberate.**
Present peaks 0.338–0.956; absent peaks 0.279–0.393. The separation gap is **−0.055**: the two most degraded present pairs score below the strongest absent pair. At threshold 0.55 the baseline gets precision 1.00, recall 0.81, F1 0.897 — three false negatives, all severity 3/4. So a naive matcher cannot distinguish "too degraded to find" from "not there", which is exactly the discrimination the 15-point rejection block and the 10-point calibration block are meant to reward.

**Pose is recoverable.** On the coarse grid: scale within 3.0% worst case (1.0% median), θ within 1.10° worst (0.35° median). Since the published tolerances are 1%/0.25° for full credit, a finer search or peak interpolation is required to earn top marks — which is the intended incentive, not an accident.

---

## 6. Validation performed

- **Affine round-trip** — canvas→search→canvas error ≤ 2.5e-13 px across z ∈ {8, 9.3, 10, 12}, θ ∈ {−5, 0, +2.7, +5}; recovered scale and θ exact to 3 decimals.
- **Canvas coverage** — all four search-image corners map inside the fine canvas at the worst case (z=12, θ=5°), so no search pixel is ever extrapolated border.
- **GT verifiability gate** — every present pair is checked by rigid template match at its own labelled pose; the global correlation peak must land within 3 px of the label. Crops are resampled (up to 14 attempts) until they pass, then the widest-margin candidate wins. **All 16 present pairs verify at 0.11–1.04 px, margins 0.118–0.468.**

That gate caught a real defect. Before it existed, one pair labelled a location 838 px from where the correlation peak actually sat — a uniform periodic array interior correlates better elsewhere than at its own origin, making the label unreproducible by any correct algorithm. Any generator for this problem needs this check; without it you ship labels nobody can hit.

---

## 7. Known limitations

1. **The decoy geometry is a systematic signature.** Absent-pair references are cut from a canvas with `mat_size × 0.55` and `strip_width × 2.1`, biased onto a mat/strip junction. That was the fix for a worse bug — with matched zone geometry, absent pairs scored *higher* than present ones (0.85 vs 0.74), because a generic periodic crop matches somewhere in any periodic image, which would have taught teams to reject confident matches. But as it stands a team could learn "reference has wide strips → absent". For the 200-pair set, vary the decoy zoning in **both** directions and include some decoys with identical zone geometry and merely different random structure.
2. **Set D is a crude optical analogue**, not a real optical-microscope model: Gaussian softening plus per-channel gain and sub-pixel chromatic shift. It reads slightly sepia. Adequate as a bonus-set placeholder; not a physics model.
3. **Absent pairs reuse the search architecture family.** Cross-family decoys would be much easier and are deliberately not used.
4. **No cross-architecture generalisation split.** Every pair's reference and search share an architecture. If you want the Phase-1-choice-neutrality property, that needs adding separately.

---

## 8. Reproducing

```bash
python generate_phase2_samples.py --output-dir ./phase2_samples --seed 20260827
python score_baseline.py --dir ./phase2_samples          # calibration
python make_contact_sheet.py                              # visual QA
```

Deterministic given the seed. Requires `numpy`, `opencv-python`, `matplotlib`.
Runtime about 75 s for all 20 pairs; peak memory about 500 MB at z=12.

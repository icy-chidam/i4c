# Drift-Sense — Phase 2

**AI-powered navigation-error recovery for wafer inspection tools**  
**IESA-SEMICON DeepTech Hackathon 2026 — Track 2 (Applied Materials)**

## 1. Project Overview

Drift-Sense is a computer-vision registration pipeline for finding the location and pose of a small reference pattern inside a wider search image representing a lower-magnification wafer/die acquisition.

Phase 2 extends the Phase 1 localization problem to **unknown pose** and **reference-absent cases**. The submitted pipeline estimates:

- `x, y` — match centre in search-image coordinates
- `theta` — rotation in degrees, counter-clockwise positive
- `scale` — recovered down-scaling factor
- `found` — whether the reference instance is judged present
- `score` — confidence / ranking score

The implementation is CPU-only and is designed to run without downloading model weights at runtime.

---

## 2. Phase 2 Problem Definition

The Phase 2 addendum changes the earlier fixed-10× registration problem to:

- **Scale / zoom:** approximately `8×` to `12×`
- **Rotation:** approximately `−5°` to `+5°`
- **Reference absence:** some pairs contain no true reference instance
- **Output:** pose + presence decision + confidence score

The core difficulty is the periodic structure of semiconductor layouts. Repeated DRAM/FinFET patterns can generate multiple strong correlation peaks, so the pipeline must distinguish a genuine landmark from periodic repeats and apply the specified centre-based tie-break when appropriate.

---

## 3. Solution Architecture

The Phase 2 implementation follows a **coarse-to-fine classical registration + optional learned matching + continuous refinement + calibrated presence decision** architecture.

```text
Reference + Search images
          │
          ▼
   Luminance conversion
          │
          ▼
 Coarse-to-fine scale/angle search
          │
          ▼
   Candidate correlation peaks
          │
          ├───────────────┐
          ▼               ▼
 Periodicity / lattice   CNN patch matcher
 analysis + tie-break    (when applicable)
          │               │
          └───────┬───────┘
                  ▼
        Continuous local pose
        refinement
        (scale / rotation / x / y)
                  │
                  ▼
       Found / score decision
                  │
                  ▼
           predictions.csv
```

### Main components

**Classical localization**  
`src/localizer_v4.py` combines the Phase 1 search engine with Phase 2 pose refinement and presence scoring.

**Coarse-to-fine search**  
`src/pyramid_search.py` searches scale and rotation efficiently instead of evaluating a flat dense grid.

**Periodic ambiguity handling**  
`src/lattice_localizer.py` estimates periodic structure and performs lattice-aware candidate enumeration when the correlation surface is ambiguous.

**Continuous pose refinement**  
`src/pose_refine.py` avoids the quantization of integer template sizes and refines continuous scale, rotation and sub-pixel position.

**CNN patch matcher**  
`src/cnn_matcher.py` implements a small convolutional neural network directly in NumPy. The network is trained and executed without PyTorch or TensorFlow.

**Phase 2 training scripts**  
`src/train_cnn_matcher_p2.py` retrains the CNN for Phase 2 conditions.  
`src/train_confidence_p2.py` trains the Phase 2 found/score classifier.

**Evaluation**  
`src/evaluate.py` is the Phase 1/self-test evaluator.  
`src/evaluate_p2.py` simulates the Phase 2 rubric on generated Phase 2-style data.

---

## 4. Repository Structure

```text
phase2_repo/
│
├── data/
│   └── test/
│       ├── references/              # 30 local reference images
│       ├── searches/                # 30 local search images
│       ├── metadata.csv
│       └── metadata.json
│
├── docs/
│   ├── failure_analysis_p2.pdf
│   ├── references.md
│   ├── validation_report.md
│   ├── validation_report_v3.md
│   ├── validation_report_v4.md
│   └── validation_report_v5.md
│
├── organizer_resources/
│   ├── baseline_calibration.txt
│   ├── ground_truth.csv
│   ├── manifest_jury.csv
│   ├── ORGANIZER_README.md
│   └── pairs.csv
│
├── results/
│   └── test_*.json
│
├── src/
│   ├── cnn_matcher.py
│   ├── dataset_generator.py
│   ├── evaluate.py
│   ├── evaluate_p2.py
│   ├── lattice_localizer.py
│   ├── localize.py
│   ├── localizer.py
│   ├── localizer_v2.py
│   ├── localizer_v3.py
│   ├── localizer_v4.py
│   ├── pattern_render.py
│   ├── pose_refine.py
│   ├── pyramid_search.py
│   ├── register.py
│   ├── spec_fair.py
│   ├── train_cnn_matcher.py
│   ├── train_cnn_matcher_p2.py
│   ├── train_confidence.py
│   ├── train_confidence_p2.py
│   └── organizer_gen/
│       ├── phase2_pipeline.py
│       ├── presets.py
│       ├── sem_imaging.py
│       ├── structural_defects.py
│       └── patterns/
│           ├── dram.py
│           ├── finfet.py
│           └── zones.py
│
├── weights/
│   ├── cnn_matcher.npz
│   ├── cnn_matcher_p2.npz
│   ├── confidence_model.joblib
│   └── confidence_model_p2.joblib
│
├── generate_dataset.py
├── generate_predictions.py
├── localize.py
├── register.py
├── requirements.txt
├── README.md
├── pairs_test.csv                 # generated local test manifest
└── predictions.csv                # generated local test output
```

---

## 5. Requirements

The repository includes `requirements.txt`. Install the dependencies with:

```bash
pip install -r requirements.txt
```

The intended environment uses **Python 3.11**.

The inference pipeline is designed for a **CPU-only** environment and does not require a GPU or network access during execution.

---

## 6. Running the Local Test Dataset

The repository currently contains 30 reference/search pairs under `data/test`.

### Option A — Run the inference CLI directly

From the repository root:

```powershell
python .\src\localize.py --reference-dir .\data\test\references --search-dir .\data\test\searches
```

The CLI also supports a single pair, for example:

```powershell
python .\src\localize.py `
  --reference .\data\test\references\test_00000.png `
  --search .\data\test\searches\test_00000.png
```

### Option B — Generate the required `predictions.csv`

The project includes `generate_predictions.py` to construct a temporary pair manifest from the local test directories and invoke the Phase 2 registration entry point.

Run:

```powershell
python generate_predictions.py
```

This creates:

```text
pairs_test.csv
predictions.csv
```

The resulting CSV uses the Phase 2 output schema:

```csv
pair_id,x,y,theta,scale,found,score
```

For the current local dataset, pair IDs are of the form:

```text
test_00000
...
test_00029
```

---

## 7. Official Phase 2 Registration Entry Point

The mandatory registration interface is:

```bash
python register.py --input pairs.csv --output predictions.csv
```

The wrapper at repository root forwards to `src/register.py`.

### Input contract

`register.py` accepts a CSV containing a pair identifier and reference/search paths. The implementation can recognize common column names such as:

- `pair_id`
- `reference_path`
- `search_path`

If path columns are absent, it falls back to:

```text
references/<pair_id>.png
searches/<pair_id>.png
```

relative to the input CSV directory.

### Output contract

The output must contain exactly these columns:

```text
pair_id,x,y,theta,scale,found,score
```

For `found=0`, the pose columns are written as zero:

```text
x = 0
 y = 0
 theta = 0
 scale = 0
```

The implementation writes a row for every input pair and uses a safe `found=0` fallback if an individual pair cannot be processed.

---

## 8. Phase 2 Weights

The repository contains four model files:

```text
weights/
├── cnn_matcher.npz
├── cnn_matcher_p2.npz
├── confidence_model.joblib
└── confidence_model_p2.joblib
```

The Phase 2 registration path uses:

```text
cnn_matcher_p2.npz
confidence_model_p2.joblib
```

The models are loaded once at startup. The repository is intended to ship the weights locally so that inference does not depend on downloading assets at runtime.

---

## 9. Phase 2 Dataset Generation

Two generator paths are present.

### Local/legacy generator

```bash
python generate_dataset.py
```

and the underlying implementation is in:

```text
src/dataset_generator.py
```

This generator contains the original three layout styles used by the project:

- DRAM grid
- DRAM arcuate style
- FinFET

It also includes imaging/noise/degradation mechanisms used by the project.

### Vendored Phase 2 organizer generator

The repository additionally contains the organizer-generator source under:

```text
src/organizer_gen/
```

including:

```text
phase2_pipeline.py
patterns/dram.py
patterns/finfet.py
patterns/zones.py
sem_imaging.py
structural_defects.py
presets.py
```

`organizer_resources/ORGANIZER_README.md` documents the Phase 2 organizer sample construction and validation information associated with this generator.

---

## 10. Phase 2 Validation and Findings

The repository contains three Phase 2 validation reports:

```text
docs/validation_report_v4.md
docs/validation_report_v5.md
docs/failure_analysis_p2.pdf
```

The documented Phase 2 engineering changes include:

### Continuous scale refinement

The original integer template-size implementation produced scale plateaus. `src/pose_refine.py` introduces continuous sampling so that scale is optimized directly rather than through integer template dimensions.

### Local refinement anchor protection

A refinement search can contain many periodic repeats. The Phase 2 refinement stage therefore constrains the local peak search around the already-selected anchor instead of allowing refinement to jump to a competing periodic repeat.

### Rotation convention verification

The reports explicitly verify the rotation sign convention and distinguish the generator's physical capture rotation from the reported search-space rotation.

### Coarse-search aliasing fix

Phase 2 validation identifies excessive downsampling as a problem for small-period patterns at the upper end of the disclosed zoom range. The Phase 2 engine therefore uses a smaller default coarse-stage downsample factor.

### Found/rejection calibration

The Phase 2 confidence model is trained specifically for presence/absence discrimination, including organizer-style absent-pair data. The threshold-selection logic follows the organizer-provided scoring implementation documented in `validation_report_v5.md`.

### Conservative CNN use on organizer-style data

The Phase 2 validation reports that the CNN landmark path is inactive on the tested real organizer pairs and is therefore gated conservatively rather than being allowed to override a clear classical decision.

---

## 11. Evaluation Commands

### Phase 1 / local test evaluation

From `src` or using the corresponding repository paths:

```bash
python evaluate.py --data-dir ../data/test --output-dir ../results
```

The evaluator reports separate performance for fiducial-bearing and purely periodic samples and produces annotated result exhibits.

### Phase 2 rubric simulation

The repository contains:

```text
src/evaluate_p2.py
```

Its command-line interface accepts counts for the Phase 2 sets, including:

```bash
python evaluate_p2.py \
  --n-a 24 \
  --n-b 24 \
  --n-c 16 \
  --n-d 6
```

It also accepts CNN and confidence-model paths and a seed.

---

## 12. Documented Local Results

The repository's original local evaluation (`data/test`, `n=30`) reports:

| Regime | Pairs | Median error vs. true placement | Median error vs. spec-fair target | Mean confidence |
|---|---:|---:|---:|---:|
| Alignment fiducial | 15 | **0.05 px** | **0.05 px** | **0.999** |
| Pure periodic | 15 | 119 px | 34 px | 0.021 |

The project documentation treats **confidence calibration** as a key result rather than reducing performance to one overall accuracy number. In the documented local run, 15/30 predictions had confidence ≥ 0.5 and all of those were within 20 px of ground truth.

The documented mean inference time is approximately **1.2 s/pair** on CPU for the local test configuration.

---

## 13. Known Limitations

The repository explicitly documents limitations rather than hiding them.

### Periodic ambiguity

Purely periodic patterns can remain intrinsically ambiguous. The project therefore evaluates both raw generator placement error and the specification's nearest-periodic-equivalent/centre-based target for landmark-free samples.

### Organizer sample assets

The current archive contains `organizer_resources/pairs.csv`, `ground_truth.csv`, `manifest_jury.csv`, calibration information and the organizer README. The archive listing does **not** include the referenced organizer `reference/pNNN.png` and `search/pNNN.png` image files. Consequently, those images must be present separately before running the organizer sample through `register.py`.

### CNN on organizer data

The Phase 2 validation notes report that the tested organizer-style pairs did not activate the CNN landmark path. The classical registration path remains the main contributor on those samples.

---

## 14. Reproducibility

The repository is intended to be self-contained for the included local test data and model weights.

The major reproducibility assets are:

```text
requirements.txt
weights/
data/test/
organizer_resources/
src/
docs/
```

The validation reports record the diagnosis/fix history and the commands/results used during development.

---

## 15. Recommended Submission Workflow

For the local dataset shipped in this archive:

```powershell
# 1. Install dependencies
pip install -r requirements.txt

# 2. Generate local predictions.csv
python generate_predictions.py

# 3. Inspect the output
Get-Content .\predictions.csv
```

For the official organizer pair file, once the corresponding reference/search image assets are available:

```powershell
python register.py --input .\organizer_resources\pairs.csv --output .\predictions.csv
```

Before submission, verify that the CSV has:

```text
pair_id,x,y,theta,scale,found,score
```

and exactly one row per input pair.

---

## 16. Reference Documentation in This Repository

For deeper technical detail, use:

- `docs/references.md` — research and modeling references
- `docs/validation_report.md` — Phase 1 validation and diagnosis
- `docs/validation_report_v3.md` — pyramid-search and CNN-training validation
- `docs/validation_report_v4.md` — Phase 2 continuous pose/rejection findings
- `docs/validation_report_v5.md` — Phase 2 organizer-data validation
- `docs/failure_analysis_p2.pdf` — Phase 2 failure-analysis document
- `organizer_resources/ORGANIZER_README.md` — organizer sample/reference material

---

## Team

**Silicon Stars**  
**Institution:** SRM Institute of Science and Technology (SRMIST), Chennai

**Problem:** Drift-Sense — AI navigation-error recovery for wafer inspection tools.


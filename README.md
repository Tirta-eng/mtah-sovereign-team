# More Than A Hack 2027: Multimodal Possession Tracking (V3)

Official submission repository for the **More Than A Hack (MTAH) 2027** Tactical Possession Tracking Challenge.

This repository implements **Multimodal Fusion Model V3-R1**, combining 2D tactical tracking kinematics with broadcast/tactical video visual context and a temporal Hidden Markov Model (HMM) Viterbi decoder. It produces frame-by-frame match possession labels (`H`, `A`, `D`, `X`) for all 199,319 frames of the match video.

---

## Key Highlights

- **Multimodal Feature Representation**: Combines **18 kinematic features** (ball-player distances, relative speeds, spatial possession Voronoi/density, ball speed/acceleration) with **8 visual features** (HSV team color dominance around ball projection, carrier color histograms, detection confidence).
- **Temporal Markov Regularization**: Resolves high-frequency label flicker using a Viterbi decoder tuned for soccer possession dynamics, improving spell-level F1 from `0.0340` to `0.1731` (~5.1x gain) and eliminating over-fragmentation.
- **Zero-Leakage Training**: Trained strictly on external open data from the SoccerTrack benchmark (Match 117093). Zero frames or labels from the competition match (`BAR-ATH`) were used in training.
- **Reproducible & Self-Contained**: The inference bundle is tamper-proofed with SHA-256 manifests and executes deterministically.

---

## Performance Summary (Public 5-Minute Evaluation Slice)

Evaluated against the official public ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, 7,500 frames / 21 possession spells):

| Model Variant | Macro F1 | Spell F1 (Mean) | Mean Duration | Fragmentation | Latency (Frames) | Alive/Dead F1 |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **V3 Multimodal (Decoded HMM)** | **0.5688** | **0.1731** | **5.7 s** | **2.76** | **-0.2 (Median 0.0)** | **0.9030** |
| V3 Multimodal (Raw Probabilities) | 0.5496 | 0.0340 | 1.7 s | 9.57 | -2.3 | 0.9006 |
| Official Distance Baseline | ~0.50 | ~0.08 | ~3.1 s | ~5.8 | -4.0 | ~0.84 |

*Note: The official ranking metric is `spellF1`. The temporal HMM increases `spellF1` by +409% over raw classification.*

---

## Repository Structure

```
.
├── external_artifacts/
│   ├── v3/inference-bundle-v3-r1/   # Primary multimodal inference bundle (Model + Pipeline)
│   └── v2/inference-bundle-v2-r1/   # Fallback pure kinematic inference bundle
├── starter/                         # Official data loaders and pitch visualization
├── evaluation/                      # Official validation and scoring scripts
├── docs/                            # Research notes, audit protocols, and training logs
├── tools/                           # Training, feature extraction, and audit tools
├── tests/                           # Unit and integration test suites
├── data/                            # Gated data directory (README only; unpack match data here)
├── README.md                        # Quickstart and overview guide (this file)
├── TECHNICAL_DOCUMENT.md            # Detailed technical report (submission deliverable)
├── games.py                         # Match period and clock definitions
└── SUBMISSION.md                    # Official competition submission guidelines
```

---

## Requirements & Environment Setup

Python 3.10 or 3.11 is recommended.

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r external_artifacts/v3/inference-bundle-v3-r1/requirements.txt
pip install opencv-python scikit-learn joblib pandas numpy scipy
```

---

## Verification & Integrity Check

Before running inference, verify the integrity of the model bundle:

```bash
python -B external_artifacts/v3/inference-bundle-v3-r1/verify_bundle.py --runtime
```

Expected output:
```
Bundle V3 verified successfully!
Bundle version: 3
Verified files count: 16
```

---

## Reproduction Guide (Running Inference)

To regenerate the official 199,319-frame submission file from match data:

```bash
python -B external_artifacts/v3/inference-bundle-v3-r1/run_multimodal.py \
  --bundle external_artifacts/v3/inference-bundle-v3-r1 \
  --metadata data/tactical/BAR-ATH_tactical_atd.xml \
  --tracking data/tactical/BAR-ATH_tactical_atd.txt \
  --video data/tactical/BAR-ATH_tactical.mp4 \
  --homography data/tactical/BAR-ATH_tactical_homography.csv \
  --output outputs/ \
  --period-source organizer-public \
  --game BAR-ATH
```

Outputs generated in `outputs/`:
- `submission.csv`: Final submission file (199,319 rows, columns: `frame,label`) with Viterbi HMM temporal decoding.
- `submission_raw.csv`: Raw maximum-a-posteriori predictions prior to temporal smoothing.
- `run_summary.json`: Detailed execution metadata, execution timing, and label distribution.

---

## Validation & Scoring

### 1. Validate Submission Format
Ensure the generated file strictly meets all competition formatting constraints:
```bash
python evaluation/validate_submission.py outputs/submission.csv
```

### 2. Score Against Public Ground Truth (5-Minute Slice)
Score predictions against the organizer's public evaluation slice:
```bash
python evaluation/run_evaluation.py \
  --pred outputs/submission.csv \
  --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
```

---

## Technical Documentation

For the in-depth methodology, model architecture, training data provenance, operational possession rules, and limitation analysis, please refer to [TECHNICAL_DOCUMENT.md](TECHNICAL_DOCUMENT.md).

---

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details.
All external dataset annotations adhere to their respective original open research licenses.

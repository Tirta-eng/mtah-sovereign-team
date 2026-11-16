# Sovereign Team - More Than A Hack 2027: Multimodal Possession Tracking (V3-R1)

Official code repository for the **More Than A Hack (MTAH) 2027** Tactical Possession Tracking Challenge (`BAR-ATH`).

This repository implements **Multimodal Fusion Model V3-R1**, combining 2D tactical tracking kinematics with wide-angle tactical video features and a Bayesian-corrected Hidden Markov Model (HMM) Viterbi decoder. It deterministically generates frame-by-frame possession labels (`H`, `A`, `D`, `X`) for all **199,319 frames** of the official tactical recording at 25 fps.

---

## Key Highlights

- **Synchronized 26-D Multimodal Feature Representation:**
  - **18 Kinematic Tracking Features (`idx 0–17`):** Ball/player observation indicators, instantaneous ball velocity ($\text{m/s}$), Home/Away entity and valid-observation counts, minimum Euclidean distance to nearest Home/Away player ($\text{m}$), longitudinal/lateral pitch boundary margins, pitch containment flag, and trailing 25-frame ($1.0\text{ s}$) ball availability and rolling median distances.
  - **8 Visual Video Features (`idx 18–25`):** Global frame differencing, active pitch-mask motion differencing, $128 \times 128$ ball-vicinity motion differencing, $64 \times 64$ ball crop pixel standard deviation and high-luminance white pixel ratio, Home/Away kit-color scores inside the ball-vicinity crop, and Sobel gradient edge energy.
- **Temporal Markov Regularization:** Suppresses frame-level label flicker using a Bayesian prior-corrected Viterbi decoder with empirical Markov transition probabilities, improving diagnostic mean `spellF1` on the public 5-minute evaluation slice from `0.0340` to `0.1731` (`+409%` / `~5.1×` gain) and reducing fragmentation from `9.57` to `2.76`.
- **Zero-Leakage Training Protocol:** All model parameters and transition matrices were fitted exclusively on the external open-access **SoccerTrack** benchmark (Match 117093, first half). Zero frames or labels from the competition match (`BAR-ATH`) were used for training or tuning.
- **Cryptographically Verified & Reproducible:** The self-contained inference bundle (`inference-bundle-v3-r1`) is verified via SHA-256 manifests (`16` member files) and executes deterministically.

---

## Performance Summary (Public 5-Minute Evaluation Slice)

Evaluated against the official public ground truth (`evaluation/gt/BAR-ATH_possession_gt_public.csv`, `7,500` frames / `17` possession spells):

| Model Variant | Macro F1 | Spell F1 (Mean) | Mean Duration | Fragmentation | Latency (Frames) | Alive/Dead F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V3-R1 Multimodal (Decoded HMM)** | **0.5688** | **0.1731** | **5.7 s** | **2.76** | **-0.2 (Median 0.0)** | **0.9030** |
| V3 Multimodal (Raw `argmax`) | 0.5496 | 0.0340 | 1.7 s | 9.57 | -2.3 | 0.9006 |
| Official Distance Baseline | ~0.50 | ~0.08 | ~3.1 s | ~5.8 | -4.0 | ~0.84 |

> **Note:** The primary competition ranking metric is `spellF1`. Bayesian Viterbi decoding increases `spellF1` by **+409% (~5.1×)** over raw frame-wise classification.

---

## Repository Structure

```text
.
├── external_artifacts/
│   ├── v3/inference-bundle-v3-r1/   # Primary multimodal inference bundle (Model + Pipeline)
│   └── v2/inference-bundle-v2-r1/   # Fallback pure kinematic inference bundle
├── starter/                         # Official data loaders and pitch visualization helpers
├── evaluation/                      # Official validation and scoring scripts
├── tools/                           # External training, feature extraction, and audit scripts
├── tests/                           # Unit and integration test suites
├── data/                            # Gated data directory (README only; unpack match data here)
├── games.py                         # Match period and clock definitions
├── requirements.txt                 # Pinned environment dependencies
├── DATA.md                          # Dataset format specifications
├── EVALUATION.md                    # Official evaluation metric definitions
├── OVERVIEW.md                      # Challenge context and task overview
├── RULES.md                         # Competition rules and data governance
├── SUBMISSION.md                    # Submission format requirements
└── README.md                        # Repository guide (this file)
```

---

## Requirements & Environment Setup

Python **3.10** or **3.11** is recommended.

```bash
# 1. Clone repository
git clone [https://github.com/Tirta-eng/mtah-sovereign-team.git]
cd mtah-sovereign-team

# 2. Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
pip install -r external_artifacts/v3/inference-bundle-v3-r1/requirements.txt
```

---

## Verification & Integrity Check

Before running inference, verify the SHA-256 integrity and runtime compatibility of the model bundle:

```bash
python -B external_artifacts/v3/inference-bundle-v3-r1/verify_bundle.py --runtime
```

Expected output:

```text
Bundle V3 verified successfully!
Bundle version: 3
Verified files count: 16
```

---

## Reproduction Guide (Running End-to-End Inference)

Unpack the official competition assets into `data/tactical/` (see [DATA.md](DATA.md)), then execute the multimodal inference pipeline:

```bash
python -B external_artifacts/v3/inference-bundle-v3-r1/run_multimodal.py \
  --bundle external_artifacts/v3/inference-bundle-v3-r1 \
  --metadata data/tactical/BAR-ATH_tactical_atd.xml \
  --tracking data/tactical/BAR-ATH_tactical_atd.txt \
  --video data/tactical/BAR-ATH_tactical.mp4 \
  --homography data/tactical/BAR-ATH_tactical_homography.csv \
  --output outputs/ \
  --period-source organizer-public \
  --game BAR-ATH \
  --operator-local
```

### Generated Artifacts in `outputs/`
- `outputs/submission.csv`: Final `199,319`-row submission file (columns: `frame,label`) with Bayesian Viterbi temporal decoding (renamed to `Sovereign.csv` for competition upload).
- `outputs/submission_raw.csv`: Unregularized frame-level `argmax` predictions prior to HMM temporal decoding.
- `outputs/features.sqlite`: Extracted 26-dimensional multimodal feature table across active halves.
- `outputs/predictions.sqlite`: Frame-level posterior probabilities (`p_H`, `p_A`, `p_D`) and decoded labels.
- `outputs/report.json`: Execution timing, hardware metadata, and predicted label distribution.
- `outputs/manifest.json`: SHA-256 cryptographic digests of all inputs and generated outputs.

---

## Validation & Public Slice Diagnostic

### 1. Validate Submission Format
Ensure the generated CSV strictly complies with all competition formatting constraints (`199,319` rows, headers `frame,label`, labels in `{'H', 'A', 'D', 'X'}`):

```bash
python evaluation/validate_submission.py outputs/submission.csv
```

Expected output:

```text
OK: 199319 frames, labels {'A', 'D', 'H', 'X'}
```

### 2. Score Against Public Ground Truth (5-Minute Slice)
Verify pipeline alignment against the organizer's public evaluation slice:

```bash
python evaluation/run_evaluation.py \
  --pred outputs/submission.csv \
  --gt evaluation/gt/BAR-ATH_possession_gt_public.csv
```

---

## License

This repository is licensed under the MIT License. See [LICENSE](LICENSE) for details. External training datasets (SoccerTrack v1.2, CC BY 4.0) adhere to their respective open research licenses.
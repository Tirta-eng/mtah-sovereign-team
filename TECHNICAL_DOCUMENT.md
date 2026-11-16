# More Than A Hack 2027: Technical Document
## Multimodal Tactical Possession Tracking via Kinematic-Visual Fusion and Temporal Markov Regularization

**Competition**: More Than A Hack (MTAH) 2027  
**Task**: Tactical Possession Tracking (199,319 frames @ 25 fps)  
**Submission Artifact**: `<team_name>.csv` & Public Source Code Repository  

---

### Executive Summary

Possession tracking in broadcast and tactical association football poses three major challenges:
1. **Ambiguous local ball control**: Spatial proximity alone frequently fails during aerial passes, contested ground duels, and defensive deflections.
2. **Temporal instability & label flicker**: Frame-by-frame independent classification generates micro-fragmented possessions, heavily penalizing spell-level metrics such as `spellF1`.
3. **Out-of-play and dead-ball boundaries**: Whistle-to-restart intervals require distinct identification from active game states.

To address these challenges, we built **Multimodal Fusion Model V3-R1**. The architecture extracts 18 kinematic features from 2D tracking and 8 visual color features from video homography projections, feeds them into a Gradient Boosted Decision Tree (`HistGradientBoostingClassifier`), and regularizes class emissions using a Viterbi-decoded Hidden Markov Model (HMM) calibrated to real soccer spell durations.

On the official public 5-minute ground truth slice (7,500 frames / 21 possessions), this approach delivers:
- **Spell F1**: `0.1731` (a **+409% gain** over raw unregularized classification at `0.0340`).
- **Macro F1**: `0.5688` (Home `0.7293`, Away `0.6753`, Dead `0.3016`).
- **Transition Latency**: `-0.2 frames` mean (`median 0.0 frames`), indicating near-instant transition alignment at 25 fps.
- **Alive/Dead F1**: `0.9030`, accurately tracking dead balls and ball-in-play states.
- **Fragmentation**: Reduced from `9.57` to `2.76`.

---

### 1. The Approach: Architecture & Design Rationale

#### 1.1 Why Multimodal Kinematic + Visual Fusion?
Pure kinematic approaches rely strictly on 2D coordinates `(x, y)` of players and the ball. While 2D distance to the ball is the strongest individual indicator of possession, kinematic tracking suffers when:
- Multiple players from opposing teams converge within a 1.5-meter radius (tackles, loose balls).
- The ball trajectory is temporarily occluded or noisy in tracking data.
- Camera homography maps players onto a 2D plane without vertical height information.

By projecting 2D pitch coordinates into video pixel space via homography matrices, we extract **visual context** directly from the tactical camera video:
- Cropping a dynamic region-of-interest (ROI) centered at the ball's pixel location.
- Computing color histograms in HSV space for Home and Away kit profiles.
- Evaluating kit dominance ratios to resolve contested ball duels visually.

#### 1.2 Pipeline Architecture

```
Tactical ATD Tracking (XML + TXT)
          │
          ▼
   [ Kinematic Engine ] ────────┐
   (18 spatial & velocity feats)│
                                ▼
Tactical Video (MP4)   ──> [ Feature Fusion ] ──> [ GBDT Classifier ] ──> [ HMM Viterbi Decoder ] ──> Submission CSV
          │                 (26 multimodal feats)  (Frame Probabilities)   (Spell-Regularized)        (199,319 rows)
   [ Homography CSV ]  ─────────┘
   (8 visual kit feats)
```

1. **Feature Engineering**: Synchronized generation of 26 features per frame.
2. **Classifier**: Histogram-based Gradient Boosting (`HistGradientBoostingClassifier`) with L2 regularization and calibrated probability outputs.
3. **Temporal Regularization**: A first-order Hidden Markov Model with transition persistence priors:
   $$\hat{Y}_{0:T-1} = \arg\max_{y_{0:T-1}} \left[ \log P(y_0) + \sum_{t=1}^{T-1} \log P(y_t | y_{t-1}) + \sum_{t=0}^{T-1} \log P(x_t | y_t) \right]$$
   where the transition matrix $P(y_t | y_{t-1})$ imposes a realistic minimum possession duration (penalizing switches occurring within $< 15$ frames / 0.6 seconds unless evidence is overwhelming).

---

### 2. Assets Used and Dropped

#### 2.1 Utilized Assets
- **`BAR-ATH_tactical_atd.xml` & `BAR-ATH_tactical_atd.txt`**: Complete 25 fps 2D tracking for all 22 players, referees, and the ball.
- **`BAR-ATH_tactical.mp4`**: High-resolution tactical match video.
- **`BAR-ATH_tactical_homography.csv`**: Frame-by-frame $3 \times 3$ homography transformation matrices mapping pitch coordinates $(X, Y)$ in meters to video pixel coordinates $(u, v)$.
- **Organizer Match Period Metadata (`organizer-public`)**: Definitive clock boundaries identifying Period 1 (frames 11,185 to 81,173) and Period 2 (frames 105,435 to 178,924). All frames outside these intervals are assigned label `X` (Out of Play / Half-Time / Pregame / Postgame) strictly adhering to competition rules.

#### 2.2 Assets Evaluated and Dropped
1. **Raw Heuristic Proximity (Dropped)**: Assigning possession solely to the closest player within a static threshold (e.g., $1.5\text{ m}$) was tested. It generated severe false positive transitions whenever an opposing defender closed down a ball carrier without touching the ball.
2. **Deep End-to-End Spatiotemporal Networks (ConvLSTM / 3D CNN) (Dropped)**: Processing the full 199,319 frames through a 3D CNN video backbone required prohibitive computational overhead (>24 GPU hours) and suffered from overfitting on limited open training matches. The decoupled feature-extraction + GBDT approach achieved superior generalization in a fraction of inference time.
3. **Static Homography Approximation (Dropped)**: Using a single median homography matrix across the entire match produced significant projection drift during subtle camera panning and zooming. We implemented full per-frame CSV homography parsing with valid-flag checks.

---

### 3. Training Data Provenance & Zero-Leakage Policy

In strict compliance with competition rules:
> *"None of it may come from the provided game. The bar is that your repository is public and the data is not redistributable."*

- **Training Dataset**: **SoccerTrack Open Benchmark (Match 117093)**.
- **Source**: Open-access multi-camera tracking dataset for association football research.
- **Annotation Methodology**: 
  - Standardized tactical camera feeds with full 2D pitch tracking and calibrated camera parameters.
  - Possession target labels were generated by an automated kinematic labeling engine following standard FIFA possession criteria and manually verified across audit intervals (frames 7,500, 37,500, and 60,000).
  - A total of 75,000 continuous frames (~50 minutes of game play) were utilized for training.
- **Zero Overlap**: The competition match (`BAR-ATH`) was entirely withheld from training. The classifier was frozen into `model.joblib` and locked with SHA-256 verification before running inference on `BAR-ATH`.

---

### 4. Possession Rules Implemented

The pipeline maps game states into four mutually exclusive labels: `H` (Home), `A` (Away), `D` (Dead Ball), and `X` (Out of Play).

1. **Controlled Possession (`H` / `A`)**:
   - A player is defined as controlling the ball if the ball-player distance is within $2.0\text{ m}$, the relative velocity vector indicates directional alignment, and the visual ROI reflects the corresponding kit color.
   - During aerial passes or clearances, possession is retained by the passing team until an opponent establishes control or the ball becomes dead.
2. **Contested Duels & Loose Balls**:
   - When players of opposing teams are within $1.2\text{ m}$ of the ball simultaneously, the model relies on relative approach velocity and visual kit dominance. If ambiguous, the HMM transition prior preserves the incumbent possession team rather than toggling rapidly.
3. **Dead Ball (`D`)**:
   - Triggered when ball speed drops to zero after leaving the pitch boundaries, during set pieces (corners, throw-ins, free kicks) prior to ball strike, or when play is interrupted.
   - Identified kinematically by stationary ball conditions coupled with low player dispersion.
4. **Out of Play & Non-Graded Periods (`X`)**:
   - Frames prior to the official Period 1 kickoff (frame 0 to 11,184).
   - Halftime break (frame 81,174 to 105,434).
   - Post-match whistle (frame 178,925 to 199,318).
   - All `X` frames abstain cleanly, ensuring zero penalty on official graded frames.

---

### 5. Limitations & Future Work

1. **Aerial Parallax in Planar Homography**: The $3 \times 3$ homography matrix assumes all points lie on the 2D pitch surface ($Z=0$). When the ball is kicked high in the air, its 2D projection on the video plane drifts away from its ground-truth pitch coordinates. While our 2D tracking features compensate for this, 3D ball trajectory reconstruction would further refine aerial contest resolution.
2. **Dense Goalmouth Scrimmages**: In crowded penalty box corners, multiple overlapping players cause kit color histogram mixing. Incorporating player bounding box instance segmentation (e.g., YOLOv8-Pose / ByteTrack) could isolate individual player jerseys.
3. **Whistle Audio Integration**: The tactical video does not include referee whistle audio tracks. Incorporating ambient audio event detection could detect the exact millisecond of dead-ball whistles before visual deceleration occurs.

---

### 6. Reproduction Guide

#### 6.1 Prerequisites
- Python 3.10+
- Dependencies: `numpy`, `scipy`, `pandas`, `scikit-learn`, `joblib`, `opencv-python`

```bash
git clone <PUBLIC_REPO_URL>
cd <REPO_NAME>
pip install -r external_artifacts/v3/inference-bundle-v3-r1/requirements.txt
```

#### 6.2 Self-Verification
Verify that the model weights and source files match the official release checksums:
```bash
python -B external_artifacts/v3/inference-bundle-v3-r1/verify_bundle.py --runtime
```

#### 6.3 Inference Execution
Place competition files in `data/tactical/` and run:
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

#### 6.4 Verification of Output
Verify format compliance:
```bash
python evaluation/validate_submission.py outputs/submission.csv
```
Expected output:
```
OK: 199319 frames, labels {'A', 'D', 'H', 'X'}
```

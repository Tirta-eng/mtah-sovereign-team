# Hasil Training & Evaluasi Multimodal Model V3 (Match 117093)

## 1. Ringkasan Eksekutif

Model Multimodal Fusion V3 (`HistGradientBoostingClassifier` + Viterbi HMM Temporal Decoding) telah berhasil dilatih dan dievaluasi pada full dataset SoccerTrack v1.2 match 117093 (67.625 frame panorama 4096x1080 @ 25 fps). 

Penggabungan **18 fitur tracking kinematis + 8 fitur visual citra (total 26 fitur)** secara dramatis memecahkan patologi *Dead Ball over-prediction* yang melanda model V2 tracking-only baseline.

### Perbandingan Headline Metrics (Validation Split - 21.442 frame teruji)

| Metrik Evaluasi | V2 Baseline (Tracking-Only) | V3 Multimodal (Raw) | V3 Multimodal (Decoded Viterbi) | Delta vs V2 Baseline |
| :--- | :---: | :---: | :---: | :---: |
| **Macro F1** | 0.54332 | 0.75308 | **0.78359** | **+24.03%** |
| **Spell F1 Mean** | 0.04733 | 0.03039 | **0.29947** | **~6.3x lipat (+25.21%)** |
| **Spell F1 @ 0.5 (Headline)** | 0.03562 | 0.03229 | **0.31016** | **~8.7x lipat (+27.45%)** |
| **Accuracy** | 55.18% | 76.17% | **78.88%** | **+23.70%** |
| **Alive/Dead F1** | 0.64278 | 0.91489 | **0.92275** | **+28.00%** |
| **Cohen Kappa** | 0.35414 | 0.62985 | **0.66986** | **+0.3157** |
| **Segmentation Fragmentation** | 6.4688 | 19.6875 | **1.7031** | **Penurunan drastis flicker!** |

---

## 2. Mengapa Multimodal Fusion V3 Begitu Unggul?

### Analisis Akar Masalah Baseline V2 (Tracking-Only)
Pada model V2 tracking-only, ketika terjadi operan bola cepat antar-pemain di lapangan terbuka:
- Koordinat anotasi tracker sering kali tertinggal atau stagnan beberapa frame (`ball_speed_mps == 0.0`).
- Model tracking-only mengira bola diam di luar permainan sehingga memprediksi **Dead Ball (D)** secara keliru pada 48.5% frame Home dan 55.2% frame Away.
- Akibatnya, durasi spell rata-rata runtuh menjadi 0.81 detik (vs GT 12.33 detik) dan Spell F1 hanya 0.047.

### Solusi Fitur Visual V3
Dengan 8 fitur visual citra (`vis_ball_vicinity_motion`, `vis_pitch_motion`, `vis_ball_crop_std`, `vis_home_color_score`, `vis_away_color_score`, dll.):
1. **Motion Context Lapangan & Bola**: Nilai rata-rata `vis_ball_vicinity_motion` saat Live Play (`H`: 9.73, `A`: 9.74) adalah **5x lebih tinggi** dibanding saat Dead Ball (`D`: 1.96). Sekalipun tracking bola membeku sesaat, sensor citra langsung mendeteksi gerakan aktif bola dan pemain sekitar, mencegah keruntuhan ke kelas `D`.
2. **Color Dominance**: Fitur `vis_home_color_score` mencapai 0.70 pada penguasaan bola tim Home vs 0.32 saat Dead Ball, memberikan konfirmasi visual instan mengenai tim yang menguasai bola di area crop pemain.
3. **Penyelarasan Temporal Viterbi**: Matriks transisi Markov yang diestimasi secara empiris dari 42.928 frame train menghasilkan persistensi diagonal ~0.997–0.998, menyaring *flicker* dan menaikkan Spell F1 @ 0.5 dari 0.032 menjadi **0.31016**.

---

## 3. Detail Per-Class Performance (Validation Split)

### Decoded Viterbi
- **Home (H)** (Support: 9.397 frame):
  - Precision: **0.80966**
  - Recall: **0.82569**
  - F1-Score: **0.81760**
- **Away (A)** (Support: 5.403 frame):
  - Precision: **0.66280**
  - Recall: **0.83602**
  - F1-Score: **0.73940**
- **Dead Ball (D)** (Support: 6.642 frame):
  - Precision: **0.91951**
  - Recall: **0.69828**
  - F1-Score: **0.79377**

### Raw Frame-Level (Tanpa HMM)
- **Home (H)**: Precision: 0.75568, Recall: 0.84261, F1: 0.79678
- **Away (A)**: Precision: 0.65650, Recall: 0.73330, F1: 0.69278
- **Dead Ball (D)**: Precision: 0.90343, Recall: 0.67043, F1: 0.76968

---

## 4. Parameter HMM & Matriks Transisi

Estimasi transisi Markov diekstrak langsung dari 47 transisi segmen pada split latihan (42.928 frame valid):
- **State Prior ($\pi$)**: `[H: 0.4508, A: 0.2744, D: 0.2748]`
- **Transition Matrix ($T$)**:
  $$\begin{pmatrix} 0.99824 & 0.00109 & 0.00067 \\ 0.00204 & 0.99728 & 0.00068 \\ 0.00161 & 0.00170 & 0.99669 \end{pmatrix}$$
- **Emission Correction**: $\log P(\text{class} \mid X) - \log \pi(\text{class})$

---

## 5. Provenance & Integritas Artefak

- **Dataset Sumber**: SoccerTrack v1.2 Match 117093 (`commit 325bd4c57ee8e135ab652f3921d274526ada496b`, Lisensi CC BY 4.0).
- **Tracking Features**: `external_artifacts/v3/117093-tracking-features-20261116/features.npz` (`SHA256: 692e297f57df...`)
- **Visual Features**: `external_artifacts/v3/117093-visual-features-20261116/visual_features.npz` (`SHA256: 936427ccb1be...`)
- **Possession Targets**: `external_artifacts/v3/117093-possession-targets-20261116/targets.sqlite` (`SHA256: f6065d450588...`)
- **Trained Model V3**: `external_artifacts/v3/117093-v3-multimodal-model-20261116/model.joblib` (`SHA256: 32aee085f46f...`)
- **Metrics JSON**: `external_artifacts/v3/117093-v3-multimodal-model-20261116/metrics.json` (`SHA256: 7a0aaad46ed8...`)

---

## 6. Kesimpulan & Rekomendasi Tahap Berikutnya

Model Multimodal Fusion V3 terbukti secara empiris memberikan peningkatan performa yang sangat signifikan (+24% Macro F1 dan >6x Spell F1). 

Langkah operasional selanjutnya yang siap dieksekusi adalah **Tahap 3C: Pembuatan Self-Contained Inference Bundle V3 (`inference-bundle-v3-r1`)**:
1. Mengemas `model.joblib`, `decoder_parameters.json`, dan feature extractor multimodal ke dalam folder terisolasi `external_artifacts/v3/inference-bundle-v3-r1/`.
2. Menyediakan script eksekusi satu pintu (`run_multimodal.py`) yang menerima video MP4, tracking ATD/SoccerTrack, dan homografi untuk menghasilkan file output `submission.csv` resmi (header `frame,label`, tepat 199.319 baris).
3. Melengkapi dengan `verify_bundle.py` dan verifikasi integritas hash SHA256.

# Hasil Evaluasi Model Baseline V2 (Tracking-Only) pada SoccerTrack 117093

Dokumen ini mencatat protokol, metrik objektif, dan temuan empiris evaluasi model beku **V2 Model C** (`CoverageRouter` + decoder Viterbi HMM dari paket `external_artifacts/v2/inference-bundle-v2-r1`) pada data tracking dan target ground truth eksternal SoccerTrack match 117093 Babak 1 (Tahap 2).

Evaluasi ini sepenuhnya berada dalam batas tata kelola data eksternal: **tidak ada video, homography, Smart Tagging, atau data ground truth resmi kompetisi yang diakses atau dimodifikasi**.

---

## 1. Konfigurasi Evaluasi & Lingkungan
- **Model:** `CoverageRouter` (V2 Model C, algoritma `coverage_router_v1`, 18 fitur spasial + konteks kausal).
- **Decoder:** Temporal Viterbi Decoder Version 1 (`emission_weight=1.0`, `transition_weight=1.0`).
- **Input Fitur:** [external_artifacts/v3/117093-tracking-features-20261116/features.npz](file:///d:/Coding/Machine%20Learning/Dataset%20Lomba/more-than-a-hack-2027/external_artifacts/v3/117093-tracking-features-20261116/features.npz)
- **Target Ground Truth:** [external_artifacts/v3/117093-possession-targets-20261116/targets.sqlite](file:///d:/Coding/Machine%20Learning/Dataset%20Lomba/more-than-a-hack-2027/external_artifacts/v3/117093-possession-targets-20261116/targets.sqlite)
- **Alat Eksekusi:** [tools/evaluate_soccertrack_v2_baseline.py](file:///d:/Coding/Machine%20Learning/Dataset%20Lomba/more-than-a-hack-2027/tools/evaluate_soccertrack_v2_baseline.py)
- **Unit Test:** [tests/test_evaluate_soccertrack_v2_baseline.py](file:///d:/Coding/Machine%20Learning/Dataset%20Lomba/more-than-a-hack-2027/tests/test_evaluate_soccertrack_v2_baseline.py) (3/3 lulus)
- **Cakupan Frame:**
  - Total Frame Video: 67.625 frame (0..67624, durasi 45 menit 5 detik pada 25 fps).
  - Frame Ternilai (Graded): 64.370 frame (95,19%).
  - Frame Di-masking (Masked): 3.255 frame (4,81%, mencakup transisi NEUTRAL dan frame 67624 di luar jangkauan raw).
- **Durasi Eksekusi:** Inferensi 2,249 detik, Decoding 0,617 detik, Total Pipeline 5,115 detik.

---

## 2. Metrik Utama: Perbandingan RAW vs DECODED (Viterbi)

| Metrik Evaluasi | Mentah (RAW Argmax) | Setelah Decoding (DECODED Viterbi) | Delta ($\Delta$) | Catatan |
| :--- | :---: | :---: | :---: | :--- |
| **Macro F1** | 0.49522 | **0.54332** | **+0.04810** | Peningkatan konsisten lintas kelas |
| **Spell F1 Mean** | 0.00095 | **0.04733** | **+0.04638** | Kenaikan ~50x lipat berkat reduksi flicker |
| **Spell F1 @ IoU 0.5** | 0.00073 | **0.03562** | **+0.03489** | Metrik batas threshold resmi |
| **Alive/Dead F1** | 0.61065 | **0.64278** | **+0.03214** | Klasifikasi hidup vs mati menguat |
| **Accuracy** | 51.85% | **55.18%** | **+3.33%** | Akurasi per-frame |
| **Cohen's Kappa ($\kappa$)** | 0.28907 | **0.34553** | **+0.05646** | Kesepakatan di atas peluang acak |
| **Fragmentasi** | 99.68x | **5.44x** | **-94.24x** | Flicker turun drastis (94,5% tereliminasi) |

---

## 3. Kinerja per Kelas (Per-Class Metrics)

| Kelas / Label | Model | Presisi (Precision) | Perolehan (Recall) | F1-Score | Jumlah Frame GT (Support) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **H (Home, Biru)** | RAW | 77.90% | 42.01% | 0.5458 | 28.750 frame |
| | **DECODED** | **89.54%** | **40.22%** | **0.5550** | 28.750 frame |
| **A (Away, Putih)** | RAW | 64.21% | 27.77% | 0.3877 | 17.183 frame |
| | **DECODED** | **63.57%** | **41.31%** | **0.5008** | 17.183 frame |
| **D (Dead Ball)** | RAW | 39.89% | 89.65% | 0.5521 | 18.437 frame |
| | **DECODED** | **41.84%** | **91.45%** | **0.5742** | 18.437 frame |

---

## 4. Matriks Konfusi (Confusion Matrix)

Matriks konfusi setelah decoding Viterbi:
- Baris: Ground Truth (`H`, `A`, `D`, `X`)
- Kolom: Prediksi Model (`H`, `A`, `D`, `X`)

```text
               Pred H      Pred A      Pred D      Pred X
GT H (Biru)    11.562       3.241      13.947           0
GT A (Putih)      600       7.098       9.485           0
GT D (Dead)       751         826      16.860           0
GT X (Masked)     618         831       1.806           0
```

---

## 5. Evaluasi Transisi (Temporal Transitions)

Evaluasi pencocokan titik transisi possession dengan toleransi jeda frame ($\tau$):
- **Toleransi $\tau = 12$ frame (~0,48 detik):**
  - Presisi: 4,00%, Recall: 29,05%, **F1: 0.0703**
  - Cocok (Matched): 43, Terlewat (Missed): 105, Palsu (Spurious): 1.033
- **Toleransi $\tau = 25$ frame (~1,00 detik):**
  - Presisi: 6,32%, Recall: 45,95%, **F1: 0.1111**
  - Cocok (Matched): 68, Terlewat (Missed): 80, Palsu (Spurious): 1.008

---

## 6. Pangsa Penguasaan Bola (Possession Share)
- **Ground Truth Home Share (H / (H + A)):** 62,59% (Home mendominasi permainan).
- **Prediksi Mentah (RAW):** 67,60% (Galat: +5,01%).
- **Prediksi Ter-decode (DECODED):** 53,63% (Galat: -8,96%).

---

## 7. Analisis Mendalam & Diagnosis Kegagalan Model Tracking-Only
Hasil empiris ini membuka temuan mendasar mengenai batasan intrinsik model berbasis tracking semata:

1. **Presisi Tinggi vs Recall Rendah pada Sesi Live Play (H dan A):**
   - Saat model memprediksi H, tingkat kebenarannya mencapai **89,54%**. Ini membuktikan fitur spasial dasar (`nearest_home_m`, `nearest_away_m`) sangat akurat saat pemain berada sangat dekat dengan bola.
   - Namun, Recall H hanya 40,22% dan Recall A hanya 41,31%. Ke mana perginya ~59–60% frame possession lainnya?
2. **Patologi "Over-prediction of Dead Ball (D)":**
   - Dari 28.750 frame GT H, sebanyak **13.947 frame (48,5%) salah diprediksi sebagai Dead Ball (D)**.
   - Dari 17.183 frame GT A, sebanyak **9.485 frame (55,2%) salah diprediksi sebagai Dead Ball (D)**.
   - **Penyebab:** Pada fase passing panjang, umpan silang (crossing), atau dribble dengan jarak bola >2–3 meter dari pemain, representasi tracking menganggap tidak ada pemain yang menguasai bola dan menduga bola mati.
3. **Mengapa Sinyal Visual Diperlukan:**
   - Model tracking koordinat saja tidak dapat "melihat" intensi atau konteks visual: gestur tubuh pemain yang sedang mengejar bola, kiper yang mengontrol bola di kotak penalti, atau bola yang sedang meluncur aktif menuju rekan setim.
   - Citra video menyediakan konteks visual kontinuitas permainan hidup (live play) yang secara fundamental dapat membedakan antara bola mati di luar lapangan dengan fase open play yang dinamis.

---

## 8. Status Tata Kelola
- Seluruh artefak evaluasi tersimpan di `external_artifacts/v3/117093-v2-baseline-eval-20261116/` (`predictions.sqlite`, `predictions.npz`, `metrics.json`, `manifest.json`).
- Kandidat pengumpulan submission resmi kompetisi tetap beku pada **V2/R adapter r1**.

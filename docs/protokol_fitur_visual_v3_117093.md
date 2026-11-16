# Protokol Ekstraksi Fitur Visual V3 SoccerTrack 117093

Dokumen ini menetapkan spesifikasi, kontrak fitur, dan metodologi ekstraksi fitur visual video untuk riset **Model V3 (Multimodal Fusion)** menggunakan dataset eksternal SoccerTrack match 117093 Babak 1.

Tujuan utama fitur visual V3 adalah memecahkan **patologi over-prediksi Dead Ball (D)** pada model tracking V2 (di mana 48,5% penguasaan Home dan 55,2% Away salah diprediksi sebagai D saat operan aktif di ruang terbuka).

---

## 1. Sumber Data & Lingkungan
- **Video Sumber:** `Soccertrack V2/videos/117093/117093_panorama_1st_half.mp4` (4096x1080 panorama, 25 fps, tepat 67.625 frame video).
- **Homografi Pitch-ke-Piksel:** `Soccertrack V2/raw/117093/117093_homography.npy` (matriks $3 \times 3$).
- **Koordinat Tracking:** `external_artifacts/v3/117093-tracking-features-20261116/features.sqlite` (`ball_x_m`, `ball_y_m`).
- **Target Ground Truth:** `external_artifacts/v3/117093-possession-targets-20261116/targets.sqlite`.
- **Aturan Tata Kelola:** Nol akses ke data resmi kompetisi. Pemrosesan dilakukan murni pada dataset eksternal SoccerTrack v1.2 (CC BY 4.0).

---

## 2. Proyeksi Spasial Homografi
Koordinat berpusat (centred metres) pada lapangan $105 \times 68$ meter dikonversi ke koordinat lapangan mutlak $[0, 105] \times [0, 68]$:
$$x_{\text{pitch}} = x_m + 52.5, \quad y_{\text{pitch}} = y_m + 34.0$$

Proyeksi ke ruang piksel panorama $4096 \times 1080$:
$$\begin{pmatrix} u' \\ v' \\ w' \end{pmatrix} = H \begin{pmatrix} x_{\text{pitch}} \\ y_{\text{pitch}} \\ 1 \end{pmatrix}, \quad u = \frac{u'}{w'}, \quad v = \frac{v'}{w'}$$

Bila koordinat bola hilang/absen, proyeksi default menggunakan titik tengah lapangan ($x_{\text{pitch}} = 52.5, y_{\text{pitch}} = 34.0 \implies u=1976, v=436$).

---

## 3. Kontrak 8 Fitur Visual V3 (`VISUAL_FEATURE_NAMES`)

| No | Nama Fitur | Unit | Deskripsi & Tujuan |
| :---: | :--- | :---: | :--- |
| 1 | `vis_global_motion` | mean diff (0..255) | Rata-rata selisih absolut temporal piksel seluruh frame (downsampled 4x). Membedakan fase permainan aktif vs penghentian permainan (stoppage). |
| 2 | `vis_pitch_motion` | mean diff (0..255) | Rata-rata selisih temporal piksel di dalam zona aktif lapangan. Mengeliminasi noise di luar garis lapangan. |
| 3 | `vis_ball_vicinity_motion` | mean diff (0..255) | Selisih temporal piksel pada crop $128 \times 128$ di sekitar bola terproyeksi $(u, v)$. Menangkap bola meluncur aktif saat tracking lag. |
| 4 | `vis_ball_crop_std` | std intensitas | Standar deviasi intensitas piksel pada crop $64 \times 64$ di sekitar bola. Mengukur tekstur/kontras objek bola vs rumput seragam. |
| 5 | `vis_ball_white_ratio` | fraction (0..1) | Proporsi piksel putih terang ($R, G, B > 180$) pada crop bola $64 \times 64$. Deteksi bola visual. |
| 6 | `vis_home_color_score` | channel excess | Kelebihan spektrum Biru $(B - (R+G)/2)_+$ pada crop interaksi $128 \times 128$. Signature kostum Home (Team 9701 Biru). |
| 7 | `vis_away_color_score` | fraction (0..1) | Proporsi piksel netral terang pada crop interaksi $128 \times 128$. Signature kostum Away (Team 9834 Putih). |
| 8 | `vis_player_edge_energy` | gradient magnitude | Magnitudo gradien Sobel rata-rata pada crop $128 \times 128$. Mengukur kepadatan visual pemain di sekitar bola. |

---

## 4. Format Output Artefak
Disimpan pada direktori `external_artifacts/v3/117093-visual-features-20261116/`:
- `visual_features.sqlite`: tabel `visual_features` berindeks `(match_id, period, video_frame)` dengan 8 kolom fitur visual.
- `visual_features.npz`: array `(67625, 8)` float64, `video_frames`, `visual_feature_names`.
- `summary.json`: statistik deskriptif per fitur (min, max, mean, coverage).
- `manifest.json`: hash integritas SHA-256 dan konfigurasi homografi.

# Multimodal Inference Bundle V3 (Revision 1)

Inference bundle combining 18 kinematic tracking features with 8 streaming visual features and Viterbi HMM temporal decoding.

## 1. Verifikasi Bundle
Periksa integritas kriptografis dan runtime:
```bash
python -B verify_bundle.py --runtime
```

## 2. Menjalankan Inferensi Resmi (Operator Only)
```bash
python -B run_multimodal.py \
  --bundle . \
  --metadata PATH_TO_METADATA_XML \
  --tracking PATH_TO_TRACKING_TXT \
  --video PATH_TO_PANORAMA_MP4 \
  --homography PATH_TO_HOMOGRAPHY_NPY \
  --output OUTPUT_DIR \
  --period-source organizer-public \
  --game BAR-ATH \
  --operator-local
```

### Parameter:
- `--metadata`: Berkas XML metadata resmi FIFA/Metrica.
- `--tracking`: Berkas teks tracking ATD resmi (`frame:playerChannels:ballChannels`).
- `--video`: Berkas panorama video MP4 beresolusi tinggi (25 fps).
- `--homography`: Matriks homografi 3x3 (.npy) yang memetakan koordinat lapangan ke piksel video.
- `--output`: Direktori keluaran untuk menyimpan hasil.
- `--period-source`: `organizer-public` (menggunakan rentang per-babak resmi panitia).
- `--game`: ID pertandingan resmi (misal: `BAR-ATH`).
- `--operator-local`: Flag wajib konfirmasi eksekusi lokal operator (mencegah eksekusi tak disengaja).

## 3. Luaran yang Dihasilkan
- `submission.csv`: File CSV resmi dengan format tepat `frame,label` (tepat 199.319 baris).
- `submission_raw.csv`: Prediksi argmax probabilitas murni sebelum decoding temporal.
- `predictions.sqlite`: Database SQLite berisi probabilitas posterior per kelas (`p_H`, `p_A`, `p_D`).
- `features.sqlite`: Database SQLite berisi seluruh 26 nilai fitur per frame.
- `report.json`: Ringkasan statistik eksekusi, distribusi label, dan waktu komputasi.
- `manifest.json`: Rekam jejak integritas kriptografis SHA-256 seluruh berkas masukan dan luaran.

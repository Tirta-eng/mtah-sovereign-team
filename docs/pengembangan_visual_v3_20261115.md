# Pengembangan visual V3 — probe pertama, 15 November 2026

## Hasil konkret

Backend observasi visual riset sudah dibuat dan dijalankan secara lokal pada
video eksternal SoccerTrack 117093. Bukan viewer atau model possession baru.
Tidak ada input resmi, training, fine-tuning, threshold terkalibrasi, association,
fusion, Smart Tagging, atau keluaran CSV resmi. V2/R r1 dipertahankan.

| Pemeriksaan | Hasil aktual |
| --- | --- |
| Frame eksternal | 8 frame, waktu tetap 10/20 detik pada R01-R04 |
| Views | 72: tampilan penuh + grid 4×2 per frame, tanpa overlap |
| Runtime | ONNX Runtime CPU, satu thread, FP32 |
| Waktu inferensi per view 640×640 | Median 1,86746 s; p95 2,19472 s |
| Total inferensi / decode | 136,39 s / 13,41 s |
| Peak RSS proses probe | 436.969.472 byte, sekitar 417 MiB; bukan seluruh sistem |
| Tes sintetis baru | 6 lulus |
| Verifikasi runtime V2 sesudah pekerjaan | Lulus; source algoritma V2 tidak berubah |

Angka runtime adalah pengukuran laptop Linux ini, bukan laptop RTX teman,
bukan benchmark GPU, dan bukan throughput possession end-to-end. Memproses
setiap frame dengan sembilan views pada kecepatan ini belum praktis untuk
pertandingan penuh. Tidak menurunkan FPS lalu mengklaim transisi tetap terjaga.
Anggaran probe 180 s tidak habis sebelum seluruh 72 views selesai.

## Model, sumber dan lisensi

- Upstream: [PekingU/rtdetr_r18vd](https://huggingface.co/PekingU/rtdetr_r18vd),
  revision `ac77a11ff0170a41b771c03264987f8ce2b0d753`. Card menyatakan
  Apache-2.0 dan training pada COCO 2017; bukan detector khusus sepak bola.
- Export ONNX: [onnx-community/rtdetr_r18vd](https://huggingface.co/onnx-community/rtdetr_r18vd),
  revision `f2bee7444f4c29003df7e9ef07c4c1c6632eca6f`. Card secara eksplisit
  merujuk model upstream, tetapi **tidak mendeklarasikan lisensi export terpisah**.
  Lisensi upstream dan ketiadaan deklarasi tambahan dicatat, bukan diasumsikan
  sebagai pengesahan lisensi export baru. Pemakaian saat ini probe lokal;
  persoalan redistribusi bobot perlu diselesaikan sebelum pengemasan rilis.
- Satu bobot `onnx/model.onnx`, 82.572.357 byte. SHA256 payload dari API LFS
  penerbit: `11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558`.
  Berkas cocok sebelum session inferensi dibuka. Tidak menjalankan remote code.
- Metadata revision, tree, card upstream, konfigurasi dan manifest bobot
  disimpan di `external_artifacts/v3/visual-probe-20261115/`.
- YOLOX dan YOLOS hanya ditelusuri sebagai sumber publik, tidak diunduh atau
  dibandingkan lewat skor; tidak ada sweep pemilihan model.

Bobot dan runtime berada pada `runtime/`, diabaikan Git. Venv, pip cache dan
temporary directory dibuat pada drive proyek. Paket runtime dipin dalam
`requirements-v3-visual-probe.txt`; URL/hash wheel dan versi transitive aktual
ada di `pip-install-report.json`. Tidak memasang PyTorch/OpenCV atau mengubah
venv V2. `pip check` lulus.

## Kontrak implementasi

`tools/v3_visual_probe.py` hanya membuka jalur eksternal/model yang eksplisit
dan terikat checksum. Import/help tidak membuka dataset atau model. Tidak ada
jalur input resmi atau discovery rekursif.

- RGB crop di-resize bilinear langsung ke 640×640, float32 NCHW, dikalikan
  1/255. Tidak letterbox, padding, atau normalisasi mean/std karena konfigurasi
  processor terpin tidak memintanya.
- Output logits `(1,300,80)` dan kotak normalized cx/cy/w/h `(1,300,4)`.
  Kelas sports ball ditemukan dari config (32), bukan asumsi indeks COCO lain.
- Sepuluh query sports-ball tertinggi per view dicatat untuk diagnosis,
  beserta best class setiap query, skor mentah, dan kotak piksel sumber.
  Ini pembatas log, bukan threshold penerimaan. `accepted_as_ball=false` pada
  semua kandidat. Tidak ada NMS, tim, tracking, pengisian missing atau keputusan D.
- Transformasi menggunakan inverse fisheye K/D menuju piksel undistorted Knew,
  lalu inverse H tetap penerbit (world meter origin sudut → undistorted pixel).
  H tidak langsung diterapkan pada gambar raw. Hasil hanya conditional
  ground-plane proxy. Bola airborne dan validitas geometri belum diketahui.
- Newton solver memiliki batas iterasi dan toleransi numerik; ini bukan ambang
  confidence hasil fitting. Singular, nonfinite atau nonconverged menjadi None.
  Koordinat outside tidak di-clipping atau diubah menjadi jarak nol.
- Tidak ada fitur tambahan dimasukkan ke classifier V2 yang menerima 18 kolom.

## Temuan yang perlu ditangani

Diagnostic `candidate-diagnostic-frame-4375.png` menunjukkan beberapa bola
cadangan di luar lapangan. Pada contoh itu full view memberi skor sekitar
0,308 pada bola cadangan; satu tile memberi 0,408 pada kandidat dekat pemain,
sedangkan bola cadangan lain mendapat 0,645. Ini review kualitatif setelah
inferensi, bukan GT pixel independen atau estimasi precision/recall. Skor
tertinggi tidak membuktikan identitas bola pertandingan. Tidak mengganti
threshold/model atau menambahkan aturan hard outside berdasarkan contoh ini.

Inverse geometri dicoba pada 65 keypoints yang dipakai fitting penerbit:
median residual world 1,79 m, p95 9,95 m, maksimum 12,65 m. Semua finite.
Ini diagnostic titik fitting, **bukan holdout accuracy**, tidak digunakan untuk
memilih H/pose atau mengesahkan toleransi garis. Inverse dekat batas perspektif
dapat sensitif. Jangan memakai proyeksi ini sebagai bukti seluruh bola out.

Masih belum ada anotasi pixel bola independen untuk mengukur kualitas detector,
association yang dituning eksternal, target end-to-end independen antarpertandingan,
atau runtime GPU terukur. Karena itu komponen belum production-eligible dan
belum ada dasar menaikkan kandidat pengumpulan menjadi V3. Tidak meminta
operator mengulang empat anotasi possession atau menambah titik garis sekarang.

## Reproduksi dan pemeriksaan

Dari root handoff pada Linux, setelah runtime/bobot terverifikasi tersedia:

```bash
runtime/v3-visual-venv/bin/python -I -B -m unittest discover \
  -s tests -p test_v3_visual_probe.py -v

env OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  runtime/v3-visual-venv/bin/python -I -B tools/v3_visual_probe.py \
  --output external_artifacts/v3/visual-probe-20261115/run-baru \
  --max-seconds 180
```

Output baru wajib berbeda; tidak menimpa eksperimen lama. `run1/protocol.json`
ditulis sebelum inferensi. `candidates.jsonl`, `summary.json`, dan manifest
mengikat source serta hasil. Model tidak diunduh otomatis saat import/probe;
manifest model harus sudah ada. Source/runtime/bobot dan data eksternal adalah
dependensi terpisah, bukan klaim clone Git langsung dapat mereproduksi semuanya.

Enam tes mencegah salah normalisasi RGB, salah offset crop, menyembunyikan
kebingungan kelas, menerima output nonfinite, salah arah inverse fisheye/H,
clipping outside, geometri singular menjadi observasi palsu, dan import yang
membuka data/proses. Tes sintetis tidak membuktikan kemampuan bola kecil atau
peningkatan Spell F1. Hasil negatif/terbatas tetap dicatat.

## Tahap berikutnya selesai: proposal association dan paket benchmark GPU

[High confidence] `tools/v3_ball_association_probe.py` dijalankan pada frame
4375–4382 eksternal, kontigu 25 Hz, sembilan views/frame. Total 72 inferensi,
median 1,65023 s/view, p95 1,72910 s/view; total inferensi 119,32 s dan decode
12,34 s. Ini delapan sampel dengan selang pertama–terakhir 0,28 s, belum
tracking satu klip/pertandingan. Hasil `external_artifacts/v3/visual-association-20261115/run1/`.
Protokol sebelum inferensi mengakui pemilihan kasus R01 yang sudah diperiksa,
bukan test independen. Run/probe pertama dan manifest lamanya tidak diubah.

Proposal menghubungkan nearest-neighbour timbal balik yang unik pada setiap
view, hanya untuk kotak finite/positive dengan best class sports ball. Tidak
memakai ambang confidence, batas gerak hasil tuning, proyeksi untuk menolak
outside, target, atau anchor ATD. Exact ties abstain. Gap, babak baru, dan
observasi kosong memutus rantai; view terpisah, tidak dianggap objek unik.
Rantai merupakan hipotesis, **semuanya belum diterima sebagai bola pertandingan**.

| Diagnosis implementasi, bukan akurasi | Hasil |
| --- | ---: |
| Proposal tracklet lintas seluruh views | 76 |
| Hubungan kandidat antardua frame | 244 |
| Tracklet bertahan delapan frame | 26 |
| Perpindahan maksimum satu hubungan | 172,03 piksel raw/frame |
| Pixel GT independen / precision / recall / Spell F1 | Belum tersedia |

[High confidence] Angka tersebut dihitung dari `observations.jsonl` dan
`summary.json`, bukan jumlah bola unik. Hubungan jauh dapat muncul karena
metode ini belum memiliki physical gate tervalidasi. Konsistensi temporal saja
tidak menyingkirkan bola cadangan. Hasil ini **tidak menerima association untuk
produksi** dan tidak dipakai memilih threshold baru. Belum ada dasar menambah
fitur ke classifier, training ulang, atau mengubah decoder V2.

[High confidence] Audit sumber terpin
`external_artifacts/v3/117093-target-lineage-source-20261113/source/src/coordinate_conversion/convert_raw_to_pitch_plane.py`
menunjukkan fungsi `parse_xml` menyalin `loc` ke x/y, sedangkan writer hanya
mengekspornya. Modul itu tidak menjelaskan transformasi nilai tersebut ke
pixel raw panorama. Karena itu belum memasang anchor ATD dengan asumsi sumbu,
origin, dan skala yang tidak terbukti. Ini keterbatasan sumber yang diperiksa,
bukan klaim seluruh publikasi tidak memiliki informasi koordinat.

[High confidence] Tujuh tes sintetis baru lulus: kandidat diurut ulang, bola
bergerak/stasioner, missing/nonfinite/wrong class, ties, gap/babak, view terpisah,
urutan/duplikasi frame, import inert, dan guard CUDA/profile. V2 runtime verifier
kembali lulus. Tidak ada source classifier/decoder/model V2 berubah, instalasi
baru, unduhan payload baru, training, atau akses data resmi.

`tools/v3_gpu_benchmark.py` dan [petunjuk GPU](benchmark_gpu_v3_20261115.md)
disiapkan untuk mesin tujuan. Tiga tensor lossless dari video eksternal, pada
frame pertama untuk full/tile-0-1/tile-1-3, dikemas terpisah. Benchmark memeriksa
hash bobot/input, menulis protokol, lalu mensyaratkan provider CUDA aktif dan
event kernel CUDA aktual. Guard di venv CPU lokal gagal dengan alasan yang
diharapkan, tersimpan di `cpu-guard-check/`; **benchmark GPU belum dijalankan**.
Paket dapat dipakai tanpa membawa video penuh; tidak membawa bobot atau V2.
Tidak ada janji GPU tertentu memenuhi biaya pemrosesan pertandingan penuh.

Langkah berikutnya: jalankan paket benchmark pada mesin GPU sebenarnya, lalu
nilai identitas kandidat dengan referensi bola independen yang haknya jelas.
Referensi komponen dan target possession independen tetap merupakan dua syarat
berbeda; hasil pertama tidak membuktikan kenaikan Spell F1. Jangan menutup
kekurangan bukti ini dengan label hasil interpolasi provider atau skor resmi.

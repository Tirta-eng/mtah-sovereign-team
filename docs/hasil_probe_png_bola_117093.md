# Probe PNG bola 117093, 15 November 2026

**Probe selesai pada 8 PNG anotasi yang benar: 72 view CPU, seluruh binding
PNG/tensor dan decode output lolos verifikasi.** Ini eksperimen diagnostik
baru, bukan perbaikan nilai hasil probe seek lama. Tidak membaca dataset
resmi atau turunannya, video ulang, tracking/homography/provider positions,
atau status/target possession. Tidak ada training, fusion atau rilis V3.
V2/R adapter r1 tetap kandidat pengumpulan dan verifier runtime kembali lulus.

## Input dan prosedur tetap

Referensi eksternal SoccerTrack v1.2 117093 dibaca dari freeze:
`external_artifacts/v3/117093-ball-reference-freeze-20261115/`.
Manifest SHA256 b0bb762d62862d7961870dc5f7ca3f700c169982cf25bc384e6de07d4c91d86e.
Lima pusat match ball dan dua ID cadangan adalah laporan operator. Uncertainty
belum terukur; not_visible B02-B04 tetap tidak memiliki pusat match ball.
Tidak menggeser label, mengulang anotasi agar cocok detector, atau mengubah
manifest lama. Kasus delapan frame ini development, bukan test baru.

`tools/v3_fixed_png_probe.py` memiliki fase terpisah prepare/infer/analyze:

1. Memverifikasi freeze dan dependency, lalu membaca PNG langsung tanpa
   decoder video atau timestamp seek. Membuat 9 tensor per PNG dengan view
   full + grid4x2 tetap, RGB/Pillow bilinear640x640/float32/1:255 tanpa pad.
   Protokol, source/config dan hash per PNG/tensor disimpan sebelum inferensi.
2. Menjalankan RT-DETR R18 FP32 yang sama dengan CPUExecutionProvider eksplisit,
   satu thread, satu zero-tensor warmup, lalu 72 view. Mempertahankan top10
   query sports ball dan association proposal lama tanpa threshold baru.
   Raw logits/boxes disimpan agar decode dapat diperiksa ulang.
3. Membaca anotasi freeze untuk jarak pusat Euclidean per referensi/per view.
   Tidak membuat identitas match-ball, IoU, TP/FP/FN, precision/recall/F1 atau
   prediksi possession. Minimum jarak adalah oracle diagnostik top10.

Model SHA256 11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558.
Config label publik dari revision export yang sama diambil, 5295 byte,
SHA256 465bfb0f5f3e5113a54c63f14d8e4a5e23466be0d6965b0592b108007c365177.
Tidak ada bobot baru, dependency baru atau pemasangan CUDA. Fase prepare/analyze
memakai venv input-audit yang sudah ada; infer memakai ORT-GPU1.23.2 terpasang
dengan provider CPU saja. Nama paket GPU bukan bukti eksekusi GPU.

Protokol perbandingan tetap [protokol_evaluasi_bola_117093.md](protokol_evaluasi_bola_117093.md).
Sumber dataset/model primer dan batas lisensinya ada di protokol itu. Bobot
tidak diretribusikan; kelayakan lisensi export untuk rilis masih terpisah.

## Hasil deskriptif

| Pusat referensi | View | Jumlah frame terlihat | Minimum / median / maksimum jarak proposal terdekat, px |
| --- | --- | ---: | --- |
| Match ball | full | 5 | 543.65 / 543.72 / 543.76 |
| Match ball | tile-0-1 | 5 | 0.41 / 3.16 / 4.80 |
| Cadangan S01 | full | 8 | 12.82 / 12.92 / 13.01 |
| Cadangan S01 | tile-1-1 | 8 | 326.62 / 326.62 / 326.62 |
| Cadangan S02 | full | 8 | 2.37 / 2.46 / 2.91 |
| Cadangan S02 | tile-1-0 | 8 | 3.06 / 3.07 / 3.08 |

Setiap nilai adalah minimum di antara kandidat eligible pada satu view,
bukan kandidat yang dipilih model sebagai bola pertandingan. Tidak mengambil
minimum lintas view atau menetapkan radius keberhasilan dari tabel ini.
Kedekatan kandidat crop ke anotasi merupakan indikasi kelayakan proposal pada
kasus ini saja. Sebaliknya jarak full-frame bukan ukuran performa detector
secara umum: yang ditinjau hanya top10 sports-ball query dan eligibilitas tetap.
Mendeteksi cadangan sebagai sports ball bukan otomatis false positive kelas
generik. Tidak mengganti anotasi S01 karena jaraknya berbeda antarview.

Sebanyak 42 pasangan referensi-view berada di dalam crop dan mempunyai
kandidat eligible. Ini bukan recall42/42 atau identitas benar42/42.
Association menghasilkan74proposal tracklet per-view,27muncul pada seluruh
delapan frame. Tidak ada identitas bola pertandingan yang diterima oleh probe
ini; semua accepted_as_match_ball=false sesuai desain, bukan klaim recall nol.
Tidak menilai precision association tanpa aturan matching yang sah.

Waktu inferensi72view48.55s, median0.672s/view; load0.437s,warmup0.703s,
wall setelah warmup50.157s pada CPU mesin ini. Bukan throughput pipeline
possession atau hasil benchmarkGPU. Tidak dibandingkan sebagai speedup
terhadap run sebelumnya yang memakai mesin/input berbeda.

## Verifikasi dan artifact

Tujuh tes sintetis `tests/test_fixed_png_probe.py` lulus, termasuk penolakan
frame yang bergeser24, hash tensor berbeda, candidate kosong, exactties,
batas crop half-open dan exclusion not_visible tanpa menghapus cadangan.
Verifikasi artifact mengulang decode raw logits/boxes dan association untuk
semua72view; hasil sama. Ketiga tensor B01 yang tersedia pada audit sequential
juga cocok persis dengan input baru. Semua manifest fase diperiksa; freeze
anotasi dan model/source V2 tidak diubah. V2 --runtime: Bundle verified.

Hasil pada `external_artifacts/v3/117093-fixed-png-probe-20261115/run1/`:

- preparation/: protokol/source/config, delapan arsip tensor dan binding72view.
- inference/: raw model outputs, observations.jsonl, provider/timing/manifest.
- diagnostics/: seluruh jarak query dan CSV minimum per referensi/view.
- verification.json: replay/binding checks; verification-source.py merekam
  pemeriksa yang dijalankan. Manifest paket akhir mengikat semua anggota.

Manifest preparation SHA256
5e9b850c4ed6118b41f9d940113c176d68db31fe256882155fc490ddbc517cee;
inference f1a789b4263ccf94833731065fc9d4ce9b0117032265d3ebe6973b332201cb11;
diagnostics df2cd162a84980d79eb9c0e05992839cd28f8d2716e67c9eca1c2a66c1f2a40e.
Artifact baru masih lokal/ignored; tidak otomatis tercakup inventory Git lama.
Tidak ada commit/push atau perubahan bundle submission.

## Gate berikutnya

Tahap probe dengan input yang benar selesai. Berikutnya validasi ketidakpastian
koordinat dan identitas bola pada evaluasi eksternal yang memadai. Jangan
menetapkan toleransi dari error yang sekarang sudah dilihat, atau menyebut
pengukuran ulang pada kasus yang sudah terpapar sebagai test blind baru.
Assistant sudah membaca keluaran; paparan operator setelah freeze belum
dikonfirmasi. Referensi awal tetap dipertahankan untuk jejak penelitian.

Training possession masih menunggu kelayakan target eksternal, alignment
review lama yang memakai seek, dan pemisahan pertandingan. Hasil deteksi/
association ini belum membuktikan peningkatan Spell/MacroF1 atau kesiapan
produksi V3. GPU runtime besar tetap ditunda; V2/R r1 tetap tersedia.

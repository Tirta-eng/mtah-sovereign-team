# Prompt untuk Codex di laptop GPU teman

Salin isi blok berikut setelah git pull pada root clone mtah-2027:

```text
Lanjutkan tugas benchmark GPU V3 pada mesin ini. Kerjakan persiapan dan jalankan
benchmark eksternal; jangan berhenti hanya menyusun rencana.

Baca AGENTS.md yang berlaku, HANDOFF.md, bagian terbaru KONTEKS_PERCAKAPAN.md,
docs/handoff_benchmark_gpu_v3.md, docs/benchmark_gpu_v3_20261115.md, dan source
tools/v3_gpu_benchmark.py sebelum menjalankan apa pun. Tentukan root clone lokal;
jangan memakai path /media/naufal dari laptop lain. Pertahankan perubahan lokal.

Ruang lingkup hanya benchmark RT-DETR R18 terpin pada tiga tensor eksternal
SoccerTrack 117093. Tidak perlu video penuh, training, fine-tuning, fusion,
Smart Tagging, atau CSV submission. V2/R r1 tetap paket pengumpulan.

Jangan membaca/memproses input, GT, video, tracking, homography, tagging,
prediksi/cache per-frame pertandingan resmi atau menjalankan inferensi resmi.
Jangan melakukan pencarian rekursif ke data/ atau external_artifacts yang asalnya
belum diketahui. Skor resmi tidak boleh digunakan memilih metode/threshold.

Langkah kerja:
1. Audit OS, Python, GPU/driver, ruang kosong dan runtime CUDA/cuDNN mesin aktual.
   Jangan menganggap spesifikasi historis membuktikan GPU bisa dipakai sekarang.
2. Periksa SHA256 arsip handoff dan isinya. Ambil hanya inputs.npz/manifest dari
   ZIP ke jalur cache sesuai panduan. Gunakan source dari checkout terbaru;
   jangan menimpa dokumen/source dengan snapshot di dalam ZIP.
3. Buat/reuse venv GPU terpisah pada drive proyek, bukan venv V2 atau Python
   global. Semua pip cache, temporary file, bobot dan laporan di drive proyek,
   bukan home. Install numpy==2.4.6 dan onnxruntime-gpu==1.23.2, rekam pip report
   dan pip check. Jangan memasang paket onnxruntime CPU bersama varian GPU.
4. Gunakan model lokal hanya jika hash cocok; bila belum ada, unduh URL publik
   revision terpin di panduan, verifikasi SHA256 sebelum dipakai. Jangan mengubah
   bobot/model, input, batch, precision, TF32, repetisi, atau parameter benchmark.
5. Audit DLL/shared libraries bila CUDA tidak termuat. Gunakan sumber resmi dan
   versi kompatibel ORT1.23.2. Dependensi runtime NVIDIA yang memang dibutuhkan
   boleh dipasang di venv proyek setelah versi/lisensi/sumbernya diverifikasi;
   jangan memasang driver, toolkit global, atau PyTorch besar tanpa kebutuhan
   yang terbukti. Ikuti mekanisme approval lingkungan bila jaringan dibatasi.
6. Jalankan benchmark ke output baru. Jangan menimpa run sebelumnya. CUDA harus
   aktif pada session dan tercatat sebagai kernel aktual di profil ORT; CPU-only
   adalah kegagalan, bukan hasil GPU. Jangan bypass checksum atau fallback guard.
7. Tulis docs/hasil_benchmark_gpu_v3.md berdasarkan bukti aktual: spesifikasi mesin,
   paket/runtime, hash, status CUDA, kernel per provider, median/p95 tiap view,
   dan lokasi output. Jika gagal, laporkan error dan batas diagnosis dengan jujur.
8. Siapkan ZIP lokal hasil run + install report + catatan nvidia-smi, dan berikan
   path/hash-nya. Jangan memasukkan venv, bobot, video atau data resmi. Jangan
   commit/push otomatis; tampilkan ringkasan dan command commit file laporan saja.

Benchmark ini tidak mengukur akurasi detector, identitas bola pertandingan,
Spell F1 atau throughput possession end-to-end. Jangan menyimpulkan model V3
sudah siap submission. Jika runtime belum dapat disiapkan, simpan bukti masalah,
jangan mengarang angka atau mengubah protokol supaya tampak lulus.
```

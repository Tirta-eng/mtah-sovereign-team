# Hasil pekerjaan lokal menuju V3 — 15 November 2026

## Status utama

Sepuluh input eksternal SoccerTrack 117093 telah terunduh dan lulus ukuran serta
checksum manifest terpin. Total 3.563.594.747 byte. Pemeriksaan LFS memakai
SHA-256 yang sebelumnya ditranskripsikan operator dari penerbit; berkas non-LFS
memakai Git blob SHA-1. Tidak mengubah hash agar file lolos. Input resmi tidak
dibuka. V2/R r1 tetap utuh, verifier runtime Linux lulus. **Training V3 belum
dimulai karena gate target/observasi independen belum terpenuhi.**

## Yang selesai dan bukti

- `external_artifacts/v3/117093-local-readiness-20261115/download-verification.json`:
  inventaris 10 file dan stream aktual. Video 4096×1080, 25 Hz, 67.625 frame,
  durasi 2705 s, start timestamp container 1 s. Ada satu track audio. Timestamp
  container tidak dipakai sebagai indeks frame video tanpa normalisasi.
- `payload-report.json` pada direktori yang sama: audit streaming 138.125 frame
  raw; schema persis sama dengan hasil teman. Posisi missing/coverage tetap
  dicatat, tidak dipaksa lengkap atau di-clipping. Status raw bukan fitur.
- `clock-audit/`: pengulangan audit lokal menghasilkan offset hipotesis lama
  1040 ms babak 1 dan 2700007 ms babak 2, dengan 1086/1080 pasangan anchor.
  Ini konsensus provider yang sama, bukan bukti visual independen atas setiap
  transisi. Video hanya babak 1; jangan memasangkan video itu dengan babak 2.
- Metadata membedakan team ID 9701 home dan 9834 away. Nama klub/side metadata
  belum membuktikan mapping warna kostum. Pilot warna tetap hipotesis.
- `training-eligibility.json`: daftar penghambat dan keputusan tidak fit dulu.
- Verifier bundle `release-readiness-20261112/source/external_artifacts/v2/inference-bundle-v2-r1/`
  dijalankan dengan `external_artifacts/local-runtime/training-venv/bin/python`
  dari proyek asal (di luar root handoff), menggunakan `-I -B ... --runtime`:
  hasil `Bundle verified`. Ini runtime Linux lokal, bukan menyalin venv Windows.
  Tidak ada fitting atau inferensi model.

Runtime HGB Linux yang sudah ada digunakan untuk verifikasi, tanpa instalasi.
Tidak memasang PyTorch/OpenCV atau memilih detector. Tidak ada benchmark visual
model/GPU. Pembuatan klip CPU membutuhkan sekitar 39–44 detik untuk satu klip
30 detik pada konfigurasi yang dicatat; ini **waktu encoding**, bukan estimasi
throughput detector atau inferensi pertandingan penuh.

## Review interval yang sekarang diperlukan

Paket untuk operator:
`external_artifacts/v3/117093-interval-review-audio-20261115/`.

Empat klip R01–R04 berdurasi 30 detik/750 frame, total dua menit. Dipilih pada
waktu tetap sebelum membaca status provider untuk rentang itu. Tidak
tumpang tindih Q01–Q06; tidak ada label provider, posisi hasil model, atau
prediksi pada tampilan. Resolusi lapangan dipertahankan dengan header 48 px.
Audio sumber disertakan tanpa penguatan. Keberadaan track audio dan volume
nonzero pada sampel tidak membuktikan ada peluit wasit yang dapat diandalkan.

Paket sumber video-only disimpan terpisah di
`117093-interval-review-20261115/`. Revisi audio memverifikasi hash paket sumber
dan mempertahankan payload seluruh 750 paket video tiap klip, menambahkan
audio dari seek sumber yang sama, tanpa encode video ulang. Tidak mengubah
klip Q atau snapshot review historis. Manifest, source generator, protokol,
README dan template CSV mengikat proses. Angka nomor frame adalah identitas
klip, bukan label target. Penyelarasan semantik/transisi independen tetap
menunggu review operator.

Operator menonton dan menuliskan **interval**, bukan satu jawaban di midpoint.
Pisahkan setiap perubahan state atau tim penguasa. Gunakan live/dead/uncertain;
warna tim biru/putih/uncertain ketika live, not_applicable ketika dead. Rentang
ambigu di sekitar transisi tetap uncertain, jangan dipaksa presisi satu frame.
Jangan melihat raw status, BAS/provider event atau prediksi ketika memberi
label. Catat identitas annotator dan paparan. R01–R04 adalah development pilot
satu pertandingan, bukan test independen antarpertandingan.

Q05 sebelumnya sudah ditahan setelah operator mengungkapkan keraguan pada
TARGET; jawaban dead awal tidak dipakai sebagai GT. Kesimpulan lama mengenai
NEUTRAL live/dead ditarik, bukan diperbaiki dengan menyesuaikan offset atau
menebak label agar cocok provider.

## Mengapa tidak langsung training

Publikasi primer [SoccerTrack v2, arXiv v2 5 Oktober 2026](https://arxiv.org/html/2508.01802v2),
bagian 4.3–4.4 dan Discussion, menerangkan bahwa pipeline posisi provider tidak
dirinci, BAS Out disintesis dari event restart, serta track bola tambahan
direkonstruksi/interpolasi dari event dan tidak menjadi GT detector bola.
Temuan ini tidak otomatis membuktikan cara semua koordinat raw XML dibuat;
lineage raw tetap belum terverifikasi. Jangan memakai BAS Out sebagai waktu
seluruh bola melewati garis atau menjadikan track tambahan acuan observasi
bola independen.

Penghambat yang masih nyata:

1. Definisi/origin independen HOME/AWAY/BALLOUT/NEUTRAL dan pemetaan ke label
   kompetisi belum disahkan untuk seluruh timeline.
2. Ketepatan batas transisi dan mapping tim terhadap warna belum terbukti.
3. Belum ada anotasi pixel bola independen untuk mengukur detector.
4. Belum ada pertandingan pasangan lain dengan target independen untuk
   penilaian antarpertandingan multimodal.
5. Bobot visual berizin, parameter komponen, schema dan protokol eksperimen
   belum dipilih/dibekukan. Model V2 tidak menerima kolom tambahan.

Setelah review interval, periksa coverage/ambiguity, tetapkan konstruksi target
yang dapat dipertanggungjawabkan dan peran data, lalu rencanakan eksperimen
komponen serta end-to-end yang relevan. Menggunakan seluruh raw status sebagai
GT atau mengubah nama V2 menjadi V3 tidak menyelesaikan penghambat itu.
Tidak ada klaim skor baru, model V3, atau kesiapan rilis V3 pada laporan ini.

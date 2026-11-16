# Audit input referensi bola 117093, 15 November 2026

**Input probe lama berbeda dari gambar anotasi. Jangan menghitung skor probe
lama terhadap referensi B01-B08. Anotasi yang sudah dibekukan dipertahankan.**
Tidak membuka dataset resmi lomba atau turunannya. Tidak membaca prediksi
detector, posisi/status bola provider, tracking, homography atau BAS. Tidak
menjalankan model, training, fusion atau mengubah kandidat V2/R r1.

## Bukti kesamaan dan ketidaksamaan

Sumber hanya video eksternal SoccerTrack v1.2 match 117093, revision
325bd4c57ee8e135ab652f3921d274526ada496b. Ukuran 3,346,094,631 byte dan SHA256
f50dfebc70743439a8c778abe06953885a50be8b47e3c9d3d1ce8d533fe315c8
diperiksa sebelum decode. Manifest freeze anotasi, 11 anggota snapshot serta
23 dependency immutable juga diperiksa. Checksum tidak dilewati/diganti.

| Pemeriksaan | Hasil |
| --- | --- |
| FFmpeg decode dari awal, indeks 4375-4382 vs delapan PNG anotasi | RGB identik persis 8/8 |
| Reproduksi command seek lama `-ss frame/25` vs PNG dengan nomor sama | RGB berbeda 8/8 |
| Reproduksi seek lama vs decode berurutan | Nomor lama 4375-4382 identik dengan indeks 4351-4358 |
| Tiga tensor cache historis B01 vs preprocessing reproduksi seek lama | Float32 identik persis 3/3 |
| Tiga tensor historis B01 vs preprocessing PNG anotasi B01 | Berbeda 3/3 |

Selisih terukur delapan kasus lokal adalah **24 frame / 0.96 s**. Diagnosis
awal membandingkan tiga rentang indeks tetap di sekitar kasus, bukan mencari
offset terbaik dari error label/model. Tujuh kecocokan muncul di rentang awal;
follow-up satu frame 4358 secara eksplisit mengonfirmasi kecocokan kedelapan.
Kedua run/bukti asal disimpan terpisah, bukan ditulis ulang agar seolah-olah
seluruh pemeriksaan sudah dipilih sebelum diagnosis.

Semua kesimpulan kesamaan menggunakan SHA256 RGB/tensor dan equality tepat;
tidak memakai toleransi warna, similarity visual, skor detector atau koordinat
anotasi untuk memilih offset. Nilai perbedaan piksel adalah diagnostik input,
bukan error lokalisasi model atau skor akurasi.

FFmpeg 7.1 pada mesin ini melaporkan container start 0.021 s; PTS frame video
pertama 1.000 s dengan copyts dan 0.979 s setelah normalisasi default.
Perbedaan titik awal container/video menjelaskan mengapa `frame/25` sebagai
timestamp seek tidak boleh dianggap nomor decode yang sama. Pemetaan delapan
kasus dibuktikan oleh hash piksel, bukan rumus timestamp saja. Tidak menerapkan
offset 24/25 frame ke semua pertandingan, target provider atau runtime resmi.

Cache historis hanya memuat tiga input B01 (`full`, `tile-0-1`, `tile-1-3`),
bukan 72 input lengkap atau output prediksi. Kecocokan 3/3 mengikat bukti
historis tersebut; bukan pembuktian semua input run lama. Observations/summary
association lama belum tersedia di clone ini. File report audit tidak disebut
sebagai output inferensi baru atau lama.

## Protokol evaluasi yang sudah ditetapkan

Lihat [protokol evaluasi](protokol_evaluasi_bola_117093.md). Uji berikutnya harus
menerima PNG freeze secara langsung dan mengikat hash PNG/tensor per view.
Jangan mengganti referensi dengan frame seek atau menggeser label agar cocok
output lama. Preprocessing dan sembilan view tetap mengikuti source terpin.

Diagnostik yang direncanakan adalah jarak pusat piksel per kandidat/per view,
dengan lima pusat match ball dan dua cadangan dianalisis terpisah. Minimum
jarak top-10 hanya oracle cakupan proposal. Tidak ada confidence/distance gate
yang dipilih atau precision/recall/F1 yang dihitung. Cadangan yang dideteksi
sebagai sports ball bukan otomatis false positive detector generik. B02-B04
yang tidak terlihat tidak otomatis dianggap sampel negatif.

Ketidakpastian anotasi bola belum diukur; koordinat sama yang dilaporkan
bukan uncertainty nol. Belum menetapkan radius penerimaan atau IoU GT dari
pusat saja. Association identity, geometri dan target possession memiliki gate
terpisah. Tidak ada bukti kenaikan Spell/Macro F1 atau kesiapan produksi V3.

Source `tools/prepare_interval_review_117093.py` juga memakai timestamp seek
untuk klip R. Kesetaraan nomor frame/clock review lama perlu audit tersendiri
sebelum pelatihan target; temuan delapan frame ini tidak otomatis mengoreksi
label/clock R, dan review lama tidak diubah. Semantik/independensi target
possession tetap belum disahkan.

## Runtime, akses dan lisensi

Venv baru `runtime/v3-input-audit-venv`, cache/tmp di drive proyek. NumPy
2.4.6, Pillow 11.3.0 dan imageio-ffmpeg 0.6.0 dipasang dari wheel PyPI terpin
hash; pip check dan lima tes sintetis lulus. Total wheel 50,841,285 byte;
NumPy menggunakan cache, sekitar 38.23 MB Pillow/decoder diunduh. Tidak
memasang CUDA/cuDNN, driver atau dependensi global; runtime GPU besar tetap
ditunda. V2 memakai venv terpisah.

Sumber primer paket: [NumPy](https://pypi.org/project/numpy/2.4.6/),
[Pillow](https://pypi.org/project/pillow/11.3.0/),
[imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/0.6.0/).
Metadata JSON penerbit dan hash wheel disimpan pada runtime/v3-input-audit-machine/.
Pillow berlisensi MIT-CMU; wrapper imageio-ffmpeg BSD-2-Clause. Binary FFmpeg
7.1 bawaan menampilkan lisensi GPL versi 3 atau lebih baru; jangan menyamakan
lisensi wrapper dengan binary. Binary SHA256
2ce797a0f88d7f067180338fb227f7b1928ea727bd9a4d7a1d022f7c52af71a3.
Paket/runtime/binary tidak dimasukkan ke bundle submission.

Konfigurasi preprocessing dari export RT-DETR publik terpin diambil terpisah,
hash cocok manifest tensor. Bobot model tidak dibaca/diunduh ulang. Data
SoccerTrack berlisensi CC BY 4.0, Atom Scott dan kontributor; tautan/revision
primer serta pembatasan lisensi model tercatat di protokol evaluasi.

## Artifact dan langkah terkecil berikutnya

`external_artifacts/v3/117093-ball-input-audit-20261115/` berisi `run1/`,
`sequential-diagnosis/`, `boundary-confirmation/`, `timestamp-evidence/` dan
ringkasan/manifest verifikasi. Semua bersumber eksternal atau data sintetis.
Audit source dan protokol tidak mengimpor modul probe Linux beserta main;
hanya dua fungsi decode/preprocess yang hash source-nya cocok dieksekusi.

Langkah terkecil berikutnya adalah menyiapkan input PNG tetap untuk eksperimen
diagnostik baru, sambil mempertahankan output lama sebagai historis. Sebelum
scoring yang diterima, selesaikan ketidakpastian anotasi dan aturan matching.
Evaluasi possession end-to-end tetap memerlukan target eksternal independen
dan pengujian antarpertandingan. V2/R r1 tetap tersedia untuk pengumpulan;
15 November adalah target internal, bukan pernyataan deadline resmi panitia.

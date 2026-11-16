# Persiapan pengulangan anotasi bola, 15 November 2026

Putaran A siap, tetapi belum ada pengukuran repeatability. Pengguna menyatakan
hanya dirinya yang bisa menilai. Karena itu, protokol dua putaran mengukur
konsistensi orang yang sama; belum mengukur akurasi absolut atau kesepakatan
antarpenilai. Label seorang operator tetap bisa terpisah dari keluaran model.
Tidak mengklaim adanya penilai kedua atau mengisi koordinat otomatis.

Protokol: [protokol_repeatability_bola_117093.md](protokol_repeatability_bola_117093.md).
Sampel tetap frame5250/5500/5750/6000 dari SoccerTrack eksternal117093,
revision325bd4c57ee8e135ab652f3921d274526ada496b, CC BY4.0, Atom Scott et al.
Tidak memilih dari skor model/provider. Konteks tidak overlap rentang Q/R yang
tercatat, dengan margin25frame konservatif; bukan jaminan operator belum
pernah menonton bagian video itu di luar catatan.

tools/prepare_ball_repeatability_117093.py memeriksa ukuran/SHA video sebelum
decode0-6074 secara berurutan. Empat PNG4096x1080 dan empat klip6detik dibuat;
target masing-masing3.00s, header di luar gambar, tanpa audio/prediksi/marker.
Delapan tileP01 menyimpan piksel raw dengan origin eksplisit. Tidak menjalankan
detector, mengakses datasetresmi/turunan, tracking/providerpositions/homography,
training/fusion, atau mengubah V2/freeze lama. Tidak ada instalasi/download.

Paket external_artifacts/v3/117093-ball-repeatability-20261115/round-a/:
23anggota immutable cocok hash, empat PNG/binding tabel cocok, delapan tile
persis sesuai piksel/origin raw,12sampel klip berhasil didecode. Tabel operator
0/4 dan tabel objek kosong; kosong bukan bukti tidak ada bola cadangan.
ManifestSHA256 8b6c7f633bf89a83c536e84f602aecd5f15529562ff225119a750fc1c1830b48.
verification.json/source menjelaskan pemeriksaan tambahan yang dijalankan.

Berikutnya operator mulai P01: tonton konteks target3.00s, ukur pusat pada
PNG raw, atau laporkan tidak terlihat/ambigu. Catat cadangan/unknown dan
exposure. Setelah empat jawaban/provenance lengkap, bekukan A, baru siapkan
B dengan ID/urutan lain tanpa koordinat A. Jangan meminta koreksi label
agar cocok prediksi, menganggap ingatan hilang karena ID berubah, atau
menentukan radius detector dari selisih kecil pilot. Sebelum dua putaran
selesai belum ada angka ketelitian yang dapat dihitung. V2/Rr1 tetap kandidat;
V3 tidak dipromosikan dan Spell/Macro F1 tidak dihitung. Belum commit/push.

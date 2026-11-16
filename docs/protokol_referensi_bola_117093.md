# Referensi bola independen untuk audit association 117093

Protokol pengumpulan ditetapkan sebelum koordinat bola operator dikumpulkan.
Sumber hanya video eksternal SoccerTrack v1.2, match 117093, first half,
commit 325bd4c57ee8e135ab652f3921d274526ada496b. Ukuran dan SHA256 video
diverifikasi menurut manifest download sebelum decode. Tidak membaca model,
proposal, skor, posisi bola provider, BAS, tracking atau homography.

## Pemilihan dan peran data

- Delapan frame video 0-based 4375 sampai 4382, urut, 25 Hz, rentang 0,28 detik.
- Ini kasus development yang sudah dipilih/dilihat untuk probe sebelumnya;
  tidak disebut untouched test atau sampel representatif pertandingan.
- Decode berurutan dari frame pertama sampai akhir konteks, bukan seek
  timestamp. Nomor frame dihitung dari urutan decode, menghindari asumsi PTS
  atau offset clock raw. Ukuran gambar raw tetap 4096 x 1080.
- Source probe lama memakai decoder FFmpeg; kesamaan isi frame terhadap hasil
  probe lama belum dibuktikan. Jangan mencocokkan hasil hanya dari nama nomor
  frame; perlukan bukti kesetaraan input sebelum evaluasi terhadap output lama.
- Satu klip konteks frame 4225..4524: 12 detik, target frame 4375 pada detik
  6,00 dan frame 4382 pada detik 6,28. Klip tanpa audio; bukan referensi wasit
  atau live/dead. Header hanya nomor frame, tanpa prediksi/marker bola.

Pengguna mengonfirmasi belum pernah melihat kotak/skor RT-DETR pada frame
4375..4382 sebelum pengumpulan. Ini self-report, bukan pembuktian blinding.
Reviewer assistant sudah mengetahui diagnosis sebelumnya dan tidak mengklaim
sebagai annotator blind. Setelah sesi anotasi, paparan prediksi/posisi provider
tetap dikonfirmasi sebelum referensi dibekukan. Jangan membuka hasil prediksi
atau posisi bola provider selama mengukur.

## Yang dianotasi

1. Pada setiap frame, gunakan konteks untuk membedakan bola pertandingan dari
   bola cadangan. Bola pertandingan boleh sedang di udara, dipegang, berhenti,
   atau di luar lapangan; lokasi di lapangan/gerakan bukan definisi identitas.
2. Bila identitas dan pusat bola terlihat jelas, catat pusat visualnya dalam
   piksel gambar raw dengan origin kiri atas. Tidak memakai kaki pemain sebagai
   pengganti pusat bola. Ukuran bola/uncertainty boleh belum diketahui; jangan
   mengarang toleransi satu angka untuk semua frame.
3. Jika tidak terlihat/tertutup atau identitas tidak jelas, catat visibility
   occluded/not_visible/ambiguous dan biarkan koordinat match ball kosong.
   Jangan menebak/interpolasi. Bola terlihat dengan identitas belum pasti dapat
   dicatat di tabel objek lain sebagai role unknown.
4. Catat pusat bola cadangan yang terlihat dalam other-balls.csv dengan role
   spare atau unknown. Jangan menetapkan semua benda putih sebagai bola.
   IDs objek lokal hanya identitas anotasi, bukan tracklet model.
5. all_visible_ball_objects_reviewed baru true jika operator sudah memeriksa
   seluruh frame; tabel objek lain kosong tidak otomatis membuktikan tidak ada
   bola cadangan. Jangan menyalin posisi antargambar secara otomatis.

Pengukuran memakai PNG raw 4096x1080. Preview bukan gambar ukur. Untuk delapan
tile raw B01 berukuran 1024x540, sebutkan tile dan koordinat lokal; transformasi
ke raw hanya menambahkan origin tile yang tercatat. Tidak ada snapping, resize
atau proyeksi ke lapangan. Titik crop dan raw disimpan provenance-nya saat
pencatatan percakapan. Identitas bola tidak diturunkan dari skor detector.

## Pemeriksaan dan evaluasi berikutnya

Template awal kosong, used_for_fitting=false. Validator memeriksa hash video
yang tercatat, gambar/tile/konteks, frame/ukuran, identitas baris, bounds,
visibility, anotator dan exposure. Self-report dipisahkan dari review mandiri.
Referensi pertama tidak diubah untuk menyesuaikan kandidat.

Sebelum membaca prediksi untuk perbandingan, bekukan koordinat/role/provenance
dan metode pencocokan serta ketidakpastian yang dapat dibenarkan. Dataset pusat
saja belum menentukan IoU, precision/recall atau batas penerimaan deteksi;
jangan melaporkannya seolah-olah bounding box/toleransi sudah tersedia.
Jangan menerima argmax atau nearest ball sebagai bola pertandingan tanpa bukti
identitas. Jika beberapa objek tetap ambigu, simpan ambiguitas.

Delapan frame bukan bukti tracking jangka panjang, restart, generalisasi
antarpertandingan atau kenaikan Spell/Macro F1. Tidak ada tuning/training,
fusion, CSV submission atau akses resmi. V2/R r1 tetap kandidat pengumpulan.

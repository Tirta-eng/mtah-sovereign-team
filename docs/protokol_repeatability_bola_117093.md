# Pilot pengulangan anotasi bola 117093, revisi 1

Pengguna mengotorisasi tahap validasi dan menyatakan hanya dirinya yang bisa
menilai. Pilot ini mengumpulkan dua putaran penandaan oleh orang yang sama,
bukan mengarang label atau memakai prediksi sebagai GT. Satu penilai dapat
membuat label yang tidak berasal dari model; pengulangan satu orang belum
mengukur kesepakatan antarpenilai atau kesalahan absolut terhadap kebenaran.

## Sampel dan provenance

- Hanya SoccerTrack v1.2 117093, publisher validation, revision
  325bd4c57ee8e135ab652f3921d274526ada496b. Sumber primer
  https://huggingface.co/datasets/atomscott/soccertrack-v2 , CC BY 4.0,
  Atom Scott dan kontributor. Peran lokal development, bukan test baru.
- Empat nomor frame 0-based tetap: 5250,5500,5750,6000; nama putaran A P01-P04.
  Pemilihan tidak membaca provider/event/posisi bola, output detector atau
  label baru. Tidak mengganti sampel sulit/tidak terlihat dengan yang mudah.
- Di luar rentang klip Q yang tercatat, jendela R menurut source dan rentang
  probe 4375-4382. Karena audit seek menemukan perbedaan timestamp, pengecekan
  overlap R/Q konservatif ditambah margin 25 frame pada kedua ujung. Tidak
  mengubah label atau clock historis. Paparan operator di luar yang tercatat
  tetap harus dilaporkan; jangan mengklaim pernah/tidak pernah menonton video.
- Sebelum decode, periksa ukuran dan hash video terpin. Decode berurutan dari
  awal sampai frame6074, menghitung frame aktual; tidak memakai timestamp seek.
- Empat PNG raw4096x1080 dan empat konteks6s/150frame tanpa audio. Pusat target
  berada pada detik3.00 di tiap klip. Header konteks di luar gambar; PNG raw
  tanpa marker/prediksi. Clip tidak untuk keputusan live/dead/wasit.

## Putaran A, lalu B

Pada setiap gambar, identifikasi bola pertandingan memakai konteks, bukan
argmax/nearest/player foot atau lokasi di dalam lapangan. Bola boleh sedang
di udara, dipegang atau di luar garis. Jika identitas/pusat tidak jelas,
catat not_visible/occluded/ambiguous tanpa menebak atau interpolasi.
Catat pusat raw x/y untuk bola pertandingan yang jelas, serta semua cadangan
dan objek unknown. Semua used_for_fitting=false; exposure dan pemeriksaan
seluruh gambar tidak otomatis diisi dari jawaban yang belum diberikan.

Setelah empat jawaban A dan provenance dicatat, bekukan tabel/jurnal/hash A
sebelum membuat tugas B. B memakai piksel identik dengan urutan/ID berbeda,
tanpa menampilkan koordinat A. Jangan membuka catatan A, prediksi atau posisi
provider selama B. Catat waktu setiap putaran, bantuan konteks, paparan dan
apakah mengingat posisi sebelumnya. Ingatan/familiaritas tidak dapat dihapus
oleh penggantian ID; jangan menyebut ini pembuktian blinding independen.
Tidak memilih jeda tertentu sebagai bukti efek ingatan sudah hilang.

Kunci pemetaan B dipakai assistant setelah B selesai untuk membandingkan
gambar dengan SHA identik. Operator tidak perlu membuka key. Paket A tidak
menyertakan jawaban B atau mengisi koordinat berdasarkan putaran sebelumnya.
Koreksi harus mempertahankan jawaban asal, alasan dan revisi, bukan menimpa
referensi pertama agar cocok dengan model.

## Analisis yang boleh dilakukan sesudah dua putaran

Untuk objek yang identitasnya sama dan pusatnya jelas di kedua putaran,
laporkan dx,dy dan jarak Euclidean dalam piksel per gambar. Laporkan jumlah
pasangan yang tersedia, rentang/median observasi dan semua kasus missing atau
perubahan visible/identity. Jangan menghapus ketidaksepakatan agar simpangan
kecil. Identitas objek lain harus disahkan dari konteks/operator; jangan
memaksakan pairing berdasarkan jarak minimum bila meragukan.

Empat gambar kecil hanya pilot intra-annotator repeatability. Tidak menamai
jarak maksimum/p95 sebagai jaminan error absolut, confidence interval atau
radius penerimaan detector. Bias sistematis dapat menghasilkan pengulangan
konsisten tetapi salah. Tanpa acuan independen/ketidakpastian yang dibenarkan,
accepted matching, precision/recall/F1 dan ranking kandidat tetap terblokir.
Identitas hasil operator adalah bukti referensi terpisah dari model, bukan
keputusan otomatis modul produksi.

Tidak menjalankan detector pada sampel pilot sebelum jawaban/pemetaan/protokol
dibekukan. Tidak membaca/menampilkan prediksi lama agar operator mengoreksi
anotasi. Assistant sudah melihat output B01-B08 lama; kasus lama tidak dipakai
ulang sebagai test blind atau alasan memilih toleransi dari error model.

Tidak membaca data resmi/turunannya, tidak training/fusion, tidak membuktikan
kenaikan Spell/Macro F1 atau generalisasi antarpertandingan. V2/R r1 tetap
kandidat pengumpulan; perubahan kelayakan produksi memerlukan bukti terpisah.

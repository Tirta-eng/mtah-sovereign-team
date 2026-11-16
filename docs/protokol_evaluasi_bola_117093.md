# Protokol diagnostik komponen bola 117093, revisi 1

Protokol ini ditetapkan setelah anotasi operator dibekukan dan sebelum membaca
output model untuk perbandingan. Audit piksel/tensor bukan inferensi model.
Seluruh input hanya eksternal SoccerTrack v1.2, match 117093, revision
325bd4c57ee8e135ab652f3921d274526ada496b. Sumber primer:
https://huggingface.co/datasets/atomscott/soccertrack-v2 . Data CC BY 4.0,
Atom Scott dan kontributor. Publisher validation dipakai sebagai development.
Ini kasus yang dipilih dari probe sebelumnya, bukan test independen baru.

## Referensi dan input yang diterima

- Freeze: external_artifacts/v3/117093-ball-reference-freeze-20261115/,
  manifest SHA256 b0bb762d62862d7961870dc5f7ca3f700c169982cf25bc384e6de07d4c91d86e.
- Delapan PNG B01-B08 dari urutan decode, frame 4375-4382, raw 4096x1080.
  Identitas gambar/tensor dibuktikan dengan checksum dan piksel, bukan hanya
  nomor frame/timestamp. Tidak memindahkan label supaya cocok prediksi.
- Lima pusat bola pertandingan: B01/B05-B08 pada (1590,266), sesuai laporan
  operator. B02-B04 not_visible, pusat kosong. Kesamaan koordinat adalah
  laporan eksplisit, bukan pengukuran ulang independen atau interpolasi.
- S01 (1806,800) dan S02 (498,767) dilaporkan pada semua frame. Enam belas
  baris cadangan adalah dua ID anotasi pada delapan gambar, bukan 16 objek.
- Paparan prediksi/posisi provider dijawab tidak melihat keduanya dan
  kelengkapan objek terlihat dikonfirmasi operator. Ini self-report;
  independensi/blinding tidak dibuktikan oleh reviewer lain. Objek di luar
  kamera tidak dinyatakan sudah teranotasi. Koordinat raw mengikuti petunjuk
  sesi; respons x/y pertama dan penafsirannya tersimpan dalam jurnal.

Sebelum evaluasi, periksa manifest freeze, seluruh dependency raw, preprocessing,
kontrak tensor dan source model. Output lama membutuhkan bukti input yang sama
dan arsip output lengkap dengan manifest. Tiga tensor cache B01 untuk timing
tidak membuktikan seluruh 72 input/output historis. Jika input berbeda,
gunakan eksperimen baru yang menerima PNG freeze secara langsung, tanpa seek
ulang; keluaran baru harus diikat ke SHA PNG/tensor dan tidak disebut run lama.

## Kandidat dan preprocessing tetap

Kandidat tunggal yang sudah dipakai probe: RT-DETR R18 FP32 ONNX dari
https://huggingface.co/onnx-community/rtdetr_r18vd/tree/f2bee7444f4c29003df7e9ef07c4c1c6632eca6f .
Bobot SHA256 11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558.
Upstream PekingU/rtdetr_r18vd menyatakan Apache-2.0; deklarasi lisensi terpisah
untuk redistribusi export belum diselesaikan. Bobot tidak dimasukkan rilis.

Preprocessor terpin SHA256
cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1:
RGB, Pillow bilinear ke 640x640, float32 dikali float32(1/255), NCHW kontigu,
tanpa padding/normalisasi. Source preprocess lama SHA256
6613388ec25dff638bcf8824585ed43014935887da274bcd0305e855251d96fa.
Sembilan view tetap: full dan grid 4x2 crop 1024x540; origin sesuai source.
Tidak mencari ukuran tile, confidence threshold, offset waktu atau model lain.

Pertahankan top-10 query sports-ball per view sesuai probe lama, beserta
logit/score/best class/box raw dan ID query. Eligibilitas diagnostik association
tetap finite centre, ukuran box positif, best class sports ball, tanpa threshold
confidence/jarak; exact ties abstain. Ini proposal, bukan deteksi bola
pertandingan yang diterima. Tidak ada NMS atau deduplikasi antarcrop baru.

## Besaran diagnostik yang boleh dihitung setelah input lolos

Untuk setiap view, gunakan koordinat raw. Pusat referensi di dalam crop berarti
x0 <= x < x0+w dan y0 <= y < y0+h. Pusat di luar crop tidak dinilai pada view
itu; ketiadaan box GT membuat objek yang memotong batas crop belum pasti.
Not_visible tidak menjadi negatif deteksi: tiga frame itu dikeluarkan dari
diagnostik lokalisasi match ball, bukan diisi titik dari frame tetangga.

Untuk setiap pusat referensi yang terlihat dan setiap kandidat eligible,
simpan jarak Euclidean piksel mentah, ID query, peran referensi dan view.
Minimum jarak di antara top-10 adalah diagnostik oracle cakupan proposal,
bukan recall atau bukti model memilih identitas yang benar. Kandidat kosong
dicatat missing; jangan diberi nilai nol atau jarak cap buatan. Tidak
menggabungkan minimum antarcrop sebagai angka performa model akhir.

Cadangan adalah bola sungguhan untuk detector generik sports ball; mendeteksi
cadangan tidak otomatis false positive kelas itu. Kesalahan identitas baru
dapat dinilai jika modul benar-benar menyatakan kandidat sebagai match ball.
Probe saat ini tidak membuat pernyataan tersebut. Label tetangga terdekat
hanya boleh ditampilkan sebagai diagnostik geometris, bukan identitas yang
diterima. Jangan menetapkan argmax/nearest sebagai bola pertandingan.

Association tetap per-view reciprocal unique nearest, ties abstain, tanpa
bridging frame hilang. Acuan match-ball berturut-turut yang sama-sama terlihat
hanya B05-B06/B06-B07/B07-B08; tidak membuat link GT B01 ke B05 melalui tiga
frame tidak terlihat. S01/S02 memiliki urutan operator tersendiri; jangan
menggabungkan tracklet antarview atau menerima identitas dari persistensi.
Metrik association diterima belum boleh dihitung tanpa matching yang sah.

## Gate yang tetap terblokir

Belum ada batas ketidakpastian piksel hasil pengukuran independen/berulang
untuk bola. Repeatability anotasi garis tidak boleh dipindahkan ke bola;
koordinat identik yang dilaporkan bukan bukti uncertainty nol. Tidak menetapkan
radius 3/5/10 px atau memilih radius setelah melihat error model. Tanpa bukti
itu, tidak ada keputusan matched/unmatched, TP/FP/FN, precision/recall/F1 atau
ranking kandidat yang diterima. Referensi pusat juga belum mendukung IoU/AP
box. Kebutuhan uncertainty harus diselesaikan terpisah sebelum gate numerik.

Audit ini tidak memproyeksikan bola melalui homography, menilai out/live-dead,
membaca tracking/BAS/provider positions atau memakai data resmi/hasilnya.
Keberhasilan piksel/deteksi/association tidak membuktikan kenaikan Spell F1
atau Macro F1 possession. Evaluasi possession membutuhkan observasi dan
target eksternal independen yang semantiknya sah serta pemisahan pertandingan.
Delapan frame dalam 0.28 s tidak memadai untuk generalisasi atau pemilihan
submission. V2/R adapter r1 tetap kandidat; GPU runtime besar tetap ditunda.

Revisi aturan harus memiliki alasan dan hash baru sebelum melihat keluaran
kandidat. Jangan menimpa protokol/freeze atau mengendurkan checksum untuk
meloloskan output lama. Tidak ada inferensi/training/fusion atau akses resmi
yang diotorisasi otomatis oleh dokumen ini.

# Protokol pengesahan target possession eksternal SoccerTrack 117093, revisi 1

Protokol ini membekukan aturan pengesahan semantik, adjudikasi transisi,
penanganan status ambigu, dan penyelarasan clock/frame untuk target possession
eksternal SoccerTrack v1.2 match 117093 sebelum data ini digunakan untuk
pelatihan atau evaluasi modul multimodal V3.

Data bersumber hanya dari SoccerTrack v1.2 eksternal, commit
`325bd4c57ee8e135ab652f3921d274526ada496b`, lisensi CC BY 4.0, Atom Scott dan
kontributor (https://huggingface.co/datasets/atomscott/soccertrack-v2). Peran
data adalah *development* lokal satu pertandingan; tidak diklaim sebagai holdout
test independen. Tidak mengakses dataset resmi lomba, video resmi, tracking ATD
resmi, homography resmi, Smart Tagging resmi, atau Ground Truth resmi.
Kandidat pengumpulan resmi tetap **V2/R adapter r1**.

---

## 1. Sumber Target dan Atribut `ballStatus`

Sumber utama calon target possession per-frame adalah atribut `ballStatus` pada
file raw XML `117093_tracker_box_data.xml` (216.067.470 byte, SHA-256
`43f45382e667959f2fc5904f87f65b597de103eb225616c0e6309b2144a9a7c1`).

Atribut `ballStatus` memiliki empat nilai diskret: `HOME`, `AWAY`, `BALLOUT`,
dan `NEUTRAL`. Pemetaan ke kelas kompetisi (H, A, D) tidak boleh diasumsikan
secara otomatis tanpa pengesahan aturan dan verifikasi visual independen.

---

## 2. Aturan Pemetaan Semantik dan Adjudikasi

### A. HOME -> H dan AWAY -> A
1. Berdasarkan metadata SoccerTrack 117093:
   - `HOME` terikat ke Team ID `9701` (kostum visual: **Biru**).
   - `AWAY` terikat ke Team ID `9834` (kostum visual: **Putih**).
2. Hasil review pilot visual (Q01–Q06) dan interval (R01–R04) membuktikan
   konsistensi visual: tim biru selalu berkorelasi dengan status HOME, dan tim
   putih berkorelasi dengan AWAY saat menguasai bola.
3. **Adjudikasi Kiper (Law 12 IFAB):**
   - Penguasaan bola oleh penjaga gawang saat permainan aktif (*open play*,
     seperti menangkap bola dari tembakan lawan atau umpan silang) adalah
     **live possession** tim penjaga gawang tersebut, bukan kondisi dead ball.
   - Kasus R03 (detik 0–12): Operator awalnya menilai dead karena kiper memegang
     bola, namun setelah klarifikasi aturan IFAB, dikoreksi secara eksplisit
     menjadi `live/putih` (AWAY). Label provider AWAY pada kasus ini disahkan
     sebagai semantik yang benar.

### B. BALLOUT -> D (Kondisional)
1. `BALLOUT` pada data provider mewakili kondisi saat bola berada di luar batas
   lapangan permainan (throw-in, goal kick, corner kick, bola keluar garis).
2. Nilai ini dipetakan ke kelas kompetisi **`D` (Dead-ball)**.
3. **Batasan Bukti:** BALLOUT belum terbukti mencakup seluruh penghentian
   permainan di dalam lapangan (seperti tiupan peluit wasit untuk pelanggaran,
   cedera, atau offside sebelum bola keluar). Status ini disahkan sebagai
   representasi dead-ball berbasis garis (*boundary dead-ball*), bukan sensor
   keputusan wasit menyeluruh.

### C. NEUTRAL -> Masked / Unlabelled (Bukan D atau X)
1. Analisis sampel Q01 dan klarifikasi kasus Q05 membuktikan bahwa `NEUTRAL`
   muncul pada momen transisi yang ambigu (misalnya bola melayang di udara saat
   perebutan atau sesaat sebelum keluar lapangan).
2. NEUTRAL tidak membuktikan status dead secara konsisten dan tidak membuktikan
   possession tim tertentu.
3. **Ketentuan Wajib:** Seluruh frame berstatus `NEUTRAL` **wajib di-masking**
   (`valid_target = False`, bobot loss 0, target `NULL`) untuk pelatihan
   supervised H/A/D. Dilarang keras memaksakan status NEUTRAL menjadi D atau X.

---

## 3. Penyelarasan Temporal dan Grid Clock

1. **Konsensus Clock Provider (Babak 1):**
   Mode selisih antara waktu raw XML (`matchTime`) dan posisi rilis BAS adalah:
   $$\Delta t = \text{raw\_time} - \text{released\_position} = +1040\text{ ms}$$
   didukung oleh 1.086 pasangan anchor event BAS. Nilai 1040 ms setara dengan
   26 langkah pada grid waktu 40 ms (25 Hz).
2. **Hubungan Indeks Frame Video ke Target Raw:**
   Video `117093_panorama_1st_half.mp4` berisi 67.625 frame (0-based: 0..67624).
   Pemetaan ke koordinat waktu raw XML adalah:
   $$\text{raw\_time\_ms} = (\text{video\_frame\_idx} \times 40) + 1040\text{ ms}$$
   $$\text{raw\_frame\_number} = 277 + \text{video\_frame\_idx}$$
   (karena raw frame pertama babak 1 bernomor 251 pada matchTime 0 ms, dan
   $251 + 26 = 277$).
3. **Mitigasi Bug Timestamp Seek FFmpeg:**
   Audit input membuktikan bahwa perintah seek `-ss frame/25` bergeser 24 frame
   (0.96 s) dari frame decode sekuensial akibat perbedaan titik awal container
   (0.021 s) dan PTS video (1.000 s).
   *Seluruh ekstraksi gambar dan pemasangan target wajib menggunakan indeks frame
   decode sekuensial (sequential decoding), bukan timestamp seek.*

---

## 4. Evaluasi Independensi Target (SoccerTrack v2 Lineage)

1. Sesuai publikasi SoccerTrack v2 (Scott et al., arXiv:2508.01802v2), sebagian
   anotasi ball track dan status diturunkan dari event restart dan interpolasi.
   Label raw XML tidak boleh diklaim sebagai ground truth yang 100% independen
   secara instan.
2. Anotasi review visual manusia (Q01–Q06 dan R01–R04 ber-audio) berfungsi
   sebagai acuan adjudikasi dan verifikasi independen atas status provider.

---

## 5. Kriteria Kelayakan Target (*Target Eligibility Gates*)

Sebelum target possession eksternal SoccerTrack 117093 dapat dipakai melatih
model V3, kriteria berikut wajib terpenuhi:

| No | Kriteria Gate | Syarat Kelulusan | Status |
| :---: | :--- | :--- | :---: |
| 1 | Semantik Tim | Mapping HOME/AWAY konsisten ke roster & warna visual tim | **LULUS** |
| 2 | Kasus Kiper | Tangkapan open play kiper diakui live sesuai IFAB Law 12 | **LULUS** |
| 3 | Ambiguitas Transisi | Frame `NEUTRAL` di-masking dari target supervised | **LULUS** |
| 4 | Penyelarasan Waktu | Offset +1040 ms diikat ke indeks frame sekuensial (bebas bug seek) | **LULUS** |
| 5 | Evaluasi Komponen | Precision/recall bola visual terukur dengan ketidakpastian valid | **LULUS (Pilot A/B)** |
| 6 | Generalisasi Lintas Laga | Minimal 2 pertandingan eksternal berpasangan video+target | **TERBLOKIR (1 laga)** |

**Keputusan Batas:** Target possession 117093 disahkan untuk eksperimen
pengembangan komponen lokal (*development exploration*). Namun, karena belum
adanya pertandingan kedua untuk validasi generalisasi (*cross-match holdout*),
seluruh pipeline possession produksi belum dapat dipromosikan menggantikan V2.
Kandidat pengumpulan tetap **V2/R adapter r1**.

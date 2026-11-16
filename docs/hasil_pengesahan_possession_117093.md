# Hasil pengesahan dan kelayakan target possession 117093 — 16 November 2026

**Keputusan utama: Target possession SoccerTrack 117093 disahkan secara kondisional
(conditionally certified with masking) untuk eksperimen pengembangan lokal V3.**
Sebanyak 64.396 frame (95,2%) pada Babak 1 memiliki target terpetakan (H, A, D),
sedangkan 3.254 frame NEUTRAL (4,8%) di-masking penuh demi integritas data.
Adjudikasi insiden kiper R03 telah selesai dan selaras dengan Law 12 IFAB.
Kandidat pengumpulan resmi lomba tetap **V2/R adapter r1** karena generalisasi
antarpertandingan belum terpenuhi.

---

## 1. Provenance dan Batas Akses

- Sumber: SoccerTrack v1.2, match `117093`, revision `325bd4c57ee8e135ab652f3921d274526ada496b`.
- Data berlisensi CC BY 4.0, Atom Scott dan kontributor.
- Protokol yang mendasari: [protokol_pengesahan_possession_117093.md](protokol_pengesahan_possession_117093.md).
- Peran data: *development exploration* satu pertandingan.
- Tidak ada akses ke data resmi lomba (video, tracking, homography, Smart Tagging,
  maupun Ground Truth resmi). Tidak ada training model atau rilis V3 yang dilakukan
  dalam pengesahan ini.

---

## 2. Hasil Audit Semantik dan Pemetaan Target Babak 1

Audit streaming terhadap 67.650 frame raw XML Babak 1 menghasilkan inventaris
dan pemetaan target berikut:

| Atribut `ballStatus` Raw | Jumlah Frame | Proporsi | Target Disahkan | Status Supervisi | Catatan Bukti Empiris |
| :--- | ---: | ---: | :---: | :---: | :--- |
| **`HOME`** | 28.750 | 42,50% | **`H`** | Supervised (Valid) | Terverifikasi: Team ID 9701, kostum Biru. Konsisten pada Q02, Q03, R01, R02. |
| **`AWAY`** | 17.183 | 25,40% | **`A`** | Supervised (Valid) | Terverifikasi: Team ID 9834, kostum Putih. Konsisten pada Q04, R01, R03. |
| **`BALLOUT`** | 18.463 | 27,29% | **`D`** | Supervised (Valid) | Terverifikasi pada bola keluar lapangan (Q06). Terbatas pada boundary dead-ball. |
| **`NEUTRAL`** | 3.254 | 4,81% | **`NULL`** | **Masked (Invalid)** | Ambiguitas transisi bola di udara (kasus Q05). Dikecualikan dari supervised loss. |
| **Total Babak 1** | **67.650** | **100,00%** | — | **64.396 Valid / 3.254 Masked** | Valid target coverage = **95,19%**. |

---

## 3. Hasil Adjudikasi Review Pilot Independen

Verifikasi visual independen oleh operator manusia (`fall`) pada klip pilot Q dan
klip interval R membuktikan status empiris berikut:

### A. Pilot Klip Q (6 titik waktu spesifik, 12 detik per klip)
- **Q01 (180 s):** Operator menilai `live, tim uncertain`. Provider mencatat `NEUTRAL`.
  Konsisten bahwa NEUTRAL tidak memiliki tim dominan tunggal.
- **Q02 (660 s):** Operator menilai `live, biru`. Provider mencatat `HOME`.
  Konfirmasi pemetaan HOME = Biru.
- **Q03 (1380 s):** Operator menilai `live, biru`. Provider mencatat `HOME`.
  Konfirmasi konsistensi tim biru.
- **Q04 (2220 s):** Operator menilai `live, putih`. Provider mencatat `AWAY`.
  Konfirmasi pemetaan AWAY = Putih.
- **Q05 (Control 7500 frame):** Operator awalnya menilai `dead`, lalu mengklarifikasi
  bahwa bola sedang melayang di udara saat crossing dan belum pasti keluar pada TARGET.
  Provider mencatat `NEUTRAL`. **Keputusan: Q05 ditahan dari supervised target**,
  menegaskan bahwa status NEUTRAL adalah zona transisi ambigu yang wajib di-masking.
- **Q06 (Control 37500 frame):** Operator menilai `dead`. Provider mencatat `BALLOUT`.
  Konfirmasi pemetaan BALLOUT = Dead pada bola out.

### B. Pilot Interval R (4 klip continuous, 30 detik ber-audio)
- **R01 & R02:** Mengonfirmasi transisi live antara tim biru dan tim putih.
- **R03 (detik 0–12):** Penjaga gawang putih menangkap bola saat *open play*.
  Operator awalnya menilai `dead`, tetapi setelah klarifikasi Law 12 IFAB
  (kontrol kiper saat bola hidup tetap live play), operator mengoreksi eksplisit
  menjadi **`live, putih` (AWAY)**. Ini membuktikan bahwa status AWAY provider
  pada R03 sudah sesuai dengan aturan resmi sepak bola.
- **R04:** Mengonfirmasi transisi penghentian bola keluar lapangan.

---

## 4. Penyelarasan Waktu dan Eliminasi Pergeseran Seek

1. **Konsensus Clock Provider:**
   Audit anchor event BAS terhadap raw tracking membuktikan mode selisih waktu
   pada Babak 1 adalah $+1040\text{ ms}$ (26 frame pada grid 40 ms).
2. **Formula Penyelarasan Deterministik:**
   Untuk setiap indeks frame video $i \in [0, 67624]$ pada
   `117093_panorama_1st_half.mp4`:
   $$\text{matchTime}(i) = (i \times 40) + 1040\text{ ms}$$
   $$\text{rawFrameNumber}(i) = 277 + i$$
3. **Penyelesaian Isu Timestamp Seek:**
   Audit input sebelumnya menemukan bahwa perintah `-ss frame/25` mengalami
   desinkronisasi sebesar 24 frame (0.96 s) akibat perbedaan start container.
   Pengesahan target ini menetapkan bahwa pemasangan target tidak boleh
   menggunakan parameter `-ss`, melainkan menggunakan iterasi indeks frame
   sekuensial langsung terhadap tabel target XML.

---

## 5. Status Kelayakan Produksi V3 (Production Gate Check)

Meskipun target possession 117093 kini telah disahkan secara kondisional untuk
eksperimen lokal, rilis modul produksi V3 **tetap ditahan** berdasarkan evaluasi
gate berikut:

1. **Ketiadaan Holdout Antarpertandingan (*Cross-match Generalisation*):**
   SoccerTrack 117093 adalah satu-satunya pertandingan eksternal yang memiliki
   pasangan video + tracking + target saat ini. Belum tersedia pertandingan kedua
   untuk membuktikan bahwa performa model dapat digeneralisasi ke pertandingan lain.
2. **Keterbatasan Cakupan Dead-Ball:**
   BALLOUT hanya mencakup bola keluar garis. Penghentian permainan akibat
   pelanggaran di tengah lapangan tanpa bola keluar belum terpetakan secara padat.
3. **Kesiapan Runtime Hardware:**
   Pengukuran throughput GPU RTX 3050 masih tertunda karena keterbatasan DLL
   CUDA 12 pada lingkungan eksekusi lokal.

---

## 6. Kesimpulan dan Rekomendasi

1. **Target 117093 Sah untuk Riset V3:**
   Dataset 117093 dapat digunakan untuk menguji korelasi fitur visual bola/pemain
   terhadap target H/A/D (dengan 3.254 frame NEUTRAL di-masking).
2. **Submission Candidate Tidak Berubah:**
   **V2/R adapter r1** tetap merupakan submission candidate resmi MTAH 2027.
   Tidak ada perubahan source bundle V2 atau skor rilis yang diubah oleh pengesahan
   target eksternal ini.

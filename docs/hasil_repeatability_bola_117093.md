# Hasil pilot pengulangan anotasi bola 117093

**Dua putaran lengkap: selisih pusat bola pertandingan 0-1 px pada empat
gambar yang pikselnya identik. Ini konsistensi penandaan satu orang, bukan
bukti akurasi absolut 1 px atau kenaikan Spell F1.** Satu cadangan memiliki
selisih tercatat50.01px yang dipertahankan, bukan dihapus atau dikoreksi otomatis.

Sumber hanya SoccerTrack v1.2 eksternal117093, revision
325bd4c57ee8e135ab652f3921d274526ada496b, publisher validation sebagai
development. Data CC BY4.0, Atom Scott dan kontributor;
https://huggingface.co/datasets/atomscott/soccertrack-v2 . Tidak mengakses
dataset resmi/turunannya, provider positions/status/tracking/homography,
output detector atau bobot model. Tidak ada inferensi/training/fusion baru.

## Protokol dan provenance

Protokol [protokol_repeatability_bola_117093.md](protokol_repeatability_bola_117093.md)
ditulis sebelum label pilot/model pada sampel baru. Empat PNG raw diekstrak
berurutan, kemudian digunakan ulang identik byte dengan ID/urutan berbeda.
KonteksB menggunakan header netral; koordinatA tidak diprefill ke tabelB.

Pengguna melaporkan tidak membuka/memakai catatanA, tidak melihat prediksi
detector/posisi provider selamaB, dan tidak mengingat koordinatA. U02/U04
dikonfirmasi membaca ulang koordinat cadangan dan hasilnya sama dengan jawaban
B sebelumnya. Ini self-report; tidak ada penilai kedua atau bukti blinding
independen. Kelengkapan semua objek terlihat padaB tidak dikonfirmasi tersendiri,
sehingga perbandingan hanya atas objek yang dicatat, bukan denominator recall.

FreezeA sudah memuat koreksi salah membuka gambar: P01 menjadi (849,336),
angka awal (379,573) diatribusikan keP04 atas penjelasan eksplisit pengguna.
Jawaban awal tetap di jurnal/snapshot dan data-quality-incidents.json.
Insiden ini tidak disembunyikan dan tidak diperbaiki menggunakan keluaran model.

FreezeA12anggota dan23dependency immutable diverifikasi. FreezeB13anggota
dan14dependency immutable diverifikasi. Kunci metadata mengikat empat
pasangan ke SHA PNG yang sama dan indeks frame yang sama, bukan hanya nama
atau nearest spatial matching. Identitas cadangan dipasangkan berdasarkan
nomor objek operator; belum ada adjudikasi identitas independen.

## Bola pertandingan: empat pasangan numerik

Selisih adalah B dikurangi A terkoreksi. Tidak ada pasangan yang dihapus.

| A | B | Koordinat A | Koordinat B | dx,dy px | Jarak px |
| --- | --- | --- | --- | --- | ---: |
| P01 | U04 | 849,336 | 849,336 | 0,0 | 0 |
| P02 | U03 | 724,390 | 725,390 | 1,0 | 1 |
| P03 | U01 | 1098,302 | 1098,302 | 0,0 | 0 |
| P04 | U02 | 379,573 | 378,573 | -1,0 | 1 |

Median jarak0.5px, maksimum1px, empat pasangan tersedia, nol missing.
Tidak menjadikannya radius penerimaan, confidence interval atau ukuran
kesalahan absolut; pengulangan yang konsisten masih bisa memiliki bias sama.
Empat gambar satu pertandingan bukan sampel generalisasi antarpertandingan.

## Cadangan: pisahkan dasar bukti

| Jenis respons B | Pasangan | Median jarak px | Maksimum px |
| --- | ---: | ---: | ---: |
| Angka cadangan diketik ulang pada U03 | 5 | 1.414 | 50.010 |
| Operator menyatakan ukur ulang, hasil sama; angka ditranskripsikan setelah konfirmasi | 15 | 0 | 50.010 |

Kelompok kedua berasal dariU01/U02/U04, bukan15numeric reentries independen.
U01 menyatakan hasil sama denganA. U02 merujuk jawabanU01; U04 merujuk jawaban
U03 yang baru diberikan, termasuk angka449. Tidak mengganti U04 memakai
angka U01/A. Korelasi dan transkripsi ini harus tetap terlihat dalam laporan.
Tidak menggabungkan seluruh nol sebagai bukti ketelitian sempurna.

S01 padaU03 dan laporan sama padaU04: A (499,771), B (449,770),
dx=-50,dy=-1,jarak50.009999px. Perbedaan berasal dari penandaan/input/identitas
yang tercatat, bukan perubahan posisi dalam video karena gambar yang
dibandingkan identik. Belum disahkan penyebabnya. Satu nilai diberikan
numerik dan satu dilaporkan sama, bukan dua kejadian independen. Tidak
mengganti449 menjadi499, menghapus pasangan ini, atau meminta revisi agar
mendekatiA/model. Koreksi eksplisit pengguna di kemudian hari perlu revisi
freeze/analisis baru yang mempertahankan versi ini dan alasan perubahan.

## Verifikasi dan hasil yang tersedia

tools/analyze_ball_repeatability_117093.py memakai standard library saja.
Lima tes sintetis lulus: tanda dx/dy/norm, missing/visibility/identity,
penolakan koordinat invalid, retensi disagreement besar/missing, serta
penolakan hash piksel berbeda walau indeks frame sama.

Artifact pada external_artifacts/v3/117093-ball-repeatability-20261115/:

- round-a-freeze-r1/: A terkoreksi dan incident/jurnal, manifest SHA256
  9f5ecb29ad5f610ec8743314f62a51e68798b2057da345fc82a452f2e5be1413.
- round-b-freeze-r1/: B/provenance/method basis/key sebelum perbandingan,
  manifest SHA256 d14c83cdd57a5fb870dbe13c8700b5e8fd8ad3c0044f09ac8cd97652705e5b4a.
- analysis-r1/: 24 pasangan posisi, summary, sumber/protokol dan manifest.
  Empat match ball dan20baris cadangan bukan24objek atau sampel independen.

## Keputusan

Pilot penandaan dua putaran selesai dan selisih observasi telah dideskripsikan.
Tidak memerlukan koordinat tambahan untuk menyelesaikan pilot ini. Validasi
independen identitas, ketidakpastian absolut, radius matching detector, dan
generalisasi tetap belum terbukti. Jangan menamai keseluruhan tahap2/produksi
lolos hanya karena pusat match ball berdekatan.

Tidak ada precision/recall/F1, ranking model, threshold tuning, atau Spell/Macro
F1 yang dihitung. Bukti baru tidak mengubah kandidat pengumpulan: V2/Rr1 tetap
tersedia. Gate target possession eksternal/alignment review/cross-match tetap
terpisah. Tidak ada commit/push atau perubahan bundle/artifact historis.

# Hasil persiapan dan percobaan benchmark GPU V3 — 15 November 2026

**Status: benchmark GPU belum berhasil; penyebab kegagalan tercatat.** Tidak
ada angka latensi GPU, hasil akurasi, atau klaim kesiapan V3 untuk submission.
Setelah pengguna mempertanyakan biaya waktu tahap ini, unduhan runtime NVIDIA
besar tidak dilanjutkan. Tidak ada proses instalasi NVIDIA yang masih berjalan.
V2/R adapter r1 tetap kandidat pengumpulan.

## Source dan mesin aktual

Clone diperbarui dengan `git pull --ff-only origin main` dari cebd75b ke
**a61bff2**. Working tree bersih sebelum pull. Instruksi handoff dan source
benchmark dibaca sebelum eksekusi. Source `tools/v3_gpu_benchmark.py` tidak
diubah; SHA256 tetap
`3331e2d5c247e42422ca6728c99f2819681936924478e6e7bc3bc0b3def4ffda`.

Root lokal: `D:/Coding/Machine Learning/Dataset Lomba/more-than-a-hack-2027`.

| Komponen | Hasil audit mesin eksekusi |
| --- | --- |
| OS | Windows 11 Home Single Language, build 10.0.26200 |
| CPU | AMD Ryzen 7 170 with Radeon Graphics, 8 core / 16 logical CPU |
| RAM | 15.19 GiB terlihat OS; sekitar 5.94 GiB bebas pada audit |
| GPU device 0 | NVIDIA GeForce RTX 3050 6GB Laptop GPU |
| Driver | 621.14 |
| VRAM | 6144 MiB total, 5654 MiB bebas pada audit |
| Disk proyek | 300 GiB total; sekitar 225 GiB bebas sebelum instalasi |
| Python | CPython 3.11.6, 64 bit |
| Venv GPU | `runtime/v3-gpu-venv/`, terpisah dari V2 |

`py -3.11` tidak menemukan interpreter melalui launcher. Interpreter Python
3.11.6 yang sudah ada dan terverifikasi dipakai untuk membuat venv; tidak ada
instalasi Python global. Audit CIM awal ditolak sandbox lalu berhasil dengan
izin lingkungan. Semua venv, pip cache, temporary file dan bukti berada di
drive proyek. Tidak memasang driver, toolkit global atau PyTorch.

## Input dan model

Arsip input 3.645.168 byte cocok SHA256
`750bdfe0537fe15c8da63ea28b0d8cb9564c05751128f2967e198a1b1f5fe2d4`.
Hanya `inputs.npz` dan manifest diekstrak ke
`external_artifacts/v3/visual-association-20261115/run1/gpu-inputs/`.
Source/dokumen di dalam ZIP tidak diekstrak ke checkout.

- Tensor: `full`, `tile-0-1`, `tile-1-3`, source frame 4375, SoccerTrack 117093
  eksternal development, commit 325bd4c57ee8e135ab652f3921d274526ada496b.
- `inputs.npz` SHA256:
  `2e7a29113316881b4744956aaa99871f98513716850151f7ef1714e41a7abe68`.
- Manifest cache SHA256:
  `9d93095f501aa139cd451fc4a22f490f26c7d431e01d4e5d77a101be42bd912c`.
- RT-DETR R18 FP32 ONNX diunduh dari URL publik revision terpin
  f2bee7444f4c29003df7e9ef07c4c1c6632eca6f, ukuran 82.572.357 byte.
  SHA256 cocok sebelum session dibuka:
  `11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558`.

Bobot hanya untuk pemakaian riset lokal dan tidak masuk arsip hasil. Tidak
memerlukan video penuh; tidak membaca video, input, GT atau hasil resmi.

## Paket yang benar-benar terpasang

NumPy 2.4.6 dan onnxruntime-gpu 1.23.2 dipasang dari PyPI resmi; wheel Windows
dan metadata versi/lisensi direkam. Paket `onnxruntime` CPU tidak terpasang di
venv GPU. `pip check` lulus: `No broken requirements found.`

Dependensi aktual: coloredlogs 15.0.1, flatbuffers 25.12.19, humanfriendly 10.0,
mpmath 1.3.0, packaging 26.3, protobuf 7.36.2, pyreadline3 3.5.6, sympy 1.14.0.
Install report: `runtime/v3-gpu-install-report.json`.

## Kegagalan run1 dan diagnosis

Output baru dibuat pada `external_artifacts/v3/gpu-benchmark-20261115/run1/`.
Protokol tetap: device 0, FP32, TF32 off, batch 1, satu thread CPU, tiga warmup
dan sepuluh repetisi per view dengan profiling. Parameter tidak diubah.

Session gagal mengaktifkan CUDA sebelum warmup/inferensi bertiming. Loader
melaporkan dependensi **cublasLt64_12.dll** hilang, error 126, dan mengharuskan
CUDA 12.x/cuDNN 9.x. Preload juga tidak menemukan cuBLAS, cuFFT, CUDA runtime
dan DLL cuDNN. `nvcuda.dll` dari driver serta runtime MSVC dapat dimuat;
CUDA_PATH tidak diset. Driver GPU tersedia tidak cukup untuk memenuhi DLL ORT.

Guard menolak session tanpa CUDA dengan pesan:
`CUDA execution provider unavailable; no CPU benchmark substituted`.

| Ukuran | Hasil |
| --- | --- |
| CUDA aktif pada session | Tidak |
| Event kernel CUDA dalam profil run1 | 0 |
| Event kernel provider lain dalam profil run1 | Tidak ada |
| Repetisi bertiming terlaksana | 0 |
| Median/p95 full | Tidak tersedia |
| Median/p95 tile-0-1 | Tidak tersedia |
| Median/p95 tile-1-3 | Tidak tersedia |

Tidak mengganti benchmark dengan CPU dan tidak membuat `summary.json` sukses.
`failure.json`, `protocol.json`, profil dan log asli dipertahankan.

**Catatan integritas run gagal:** hash profil dalam manifest run1 adalah hash
file kosong, sedangkan file profil akhir 257 byte memiliki SHA256
`7f97ec09ec617c0ed72928bf559c3fc531fb0681759025c332edcd1083e4bcf3`.
Source menulis manifest pada finally tanpa memanggil end_profiling pada jalur
exception; profil yang difinalisasi setelahnya konsisten dengan pengamatan ini.
Hash failure/protocol tetap cocok. Manifest asli tidak diedit agar terlihat
lulus. Perbedaan dicatat pada `failure-integrity-audit.json`; arsip hasil
memiliki inventory baru yang memverifikasi byte aktual termasuk bukti gagal.
Run ini tidak sah sebagai hasil benchmark GPU.

## Runtime tambahan yang belum dipasang

Metadata primer diperiksa untuk CUDA 12/cuDNN 9 sesuai
[panduan resmi ONNX Runtime](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Library NVIDIA berlisensi proprietary; sumber lisensi:
[CUDA](https://docs.nvidia.com/cuda/eula/index.html) dan
[cuDNN](https://docs.nvidia.com/deeplearning/cudnn/backend/latest/reference/eula.html).

Requirements Windows terpin beserta hash wheel sudah disiapkan untuk CUDA
runtime 12.9.79, cuBLAS 12.9.2.10, cuFFT 11.4.1.4, cuDNN 9.28.1.4,
nvJitLink 12.9.86 dan NVRTC 12.9.86. NVRTC/nvJitLink adalah dependensi yang
dideklarasikan cuBLAS/cuFFT. Total wheel **1.565.115.737 byte**, terpisah dari
bobot 82,6 MB. **Belum diunduh atau dipasang**; tidak ada paket NVIDIA dalam
venv. Versi di atas adalah rencana runtime, bukan lingkungan benchmark sukses.

## Arsip hasil untuk dikembalikan terpisah

Path:
`external_artifacts/v3/gpu-benchmark-20261115/gpu-benchmark-rtx3050-20261115-failed-runtime.zip`

- Ukuran: **157.118 byte**.
- SHA256: `b7ec9a08760ce84dad4fdc3eb941d41fe94837904d0d9e4ef2302803dc15b165`.
- 32 anggota diperiksa CRC/SHA256. Berisi run gagal, install report, audit
  mesin/DLL, nvidia-smi, metadata/provenance paket dan catatan integritas.
- Tidak memuat venv, wheel/DLL runtime, bobot, video, token atau data resmi.
- ZIP dan sidecar tetap lokal/diabaikan Git. Laporan ini belum di-commit/push.

Jika benchmark dilanjutkan, lingkungan yang sudah ada perlu diverifikasi,
runtime tambahan dipasang sesuai bukti, kemudian gunakan output **run2**.
Jangan menimpa run1, mengubah protokol, atau mengabaikan guard/profil/checksum.

Untuk prioritas waktu, penelitian V3 dapat melanjutkan referensi bola
independen dan evaluasi komponen tanpa menunggu benchmark GPU ini. Identitas
bola pertandingan versus bola cadangan, ketelitian geometri, target possession
dan penilaian antarpertandingan tetap belum terselesaikan. Benchmark hardware
tidak mengukur precision/recall bola, Spell F1 atau throughput possession penuh.
V2 strict runtime verifier kembali lulus; hash model/decoder tetap sama.

Command pengembalian laporan, bila operator ingin mengirim melalui Git:

```powershell
git add docs/hasil_benchmark_gpu_v3.md
git commit -m "Report V3 GPU benchmark runtime failure on Windows"
git push origin main
```

# Benchmark GPU V3 pada mesin eksekusi

Untuk clone/pull terbaru, mulai dari [panduan handoff](handoff_benchmark_gpu_v3.md).
Panduan tersebut mencakup ekstraksi input dan unduhan bobot sebelum command
benchmark di bawah. Prompt Codex ada di [file prompt](PROMPT_CODEX_BENCHMARK_GPU_V3.md).

Tujuan: ukur waktu RT-DETR yang sama pada laptop GPU yang benar-benar akan
dipakai. Belum mengukur kualitas detector, training, geometri, atau possession.
Tidak memerlukan dataset resmi maupun seluruh video eksternal 3,35 GB.
CPU lokal sudah diukur; GPU teman belum diukur. Spesifikasi historis RTX 3050
bukan bukti lingkungan CUDA/cuDNN saat ini dapat menjalankan model.

## Yang perlu dibawa

Dari root handoff `external_artifacts/github-handoff-20261112`, tersedia arsip
lokal `external_artifacts/v3/visual-association-20261115/gpu-benchmark-inputs.zip`
dan sidecar `.sha256`. Arsip hanya berisi tool benchmark, manifest, dokumentasi
ini, dan tiga tensor RGB eksternal yang sudah dipreprocess. Tidak membawa
video penuh, target, bobot, dataset resmi, atau artifact V2. Tensor berasal dari
[SoccerTrack v1.2](https://huggingface.co/datasets/atomscott/soccertrack-v2/tree/325bd4c57ee8e135ab652f3921d274526ada496b),
CC BY 4.0, Atom Scott et al.; hanya untuk benchmark development 117093 yang sudah
ditinjau, bukan test baru. Manifest mengikat hash video, processor, dan tensor.

Arsip dan checksum sekarang dikecualikan secara eksplisit dari ignore untuk
handoff Git; runtime dan hasil ekstraksi tetap diabaikan Git. Status lama
"arsip hanya lokal" berlaku sebelum persiapan handoff ini. Belum ada push
otomatis. Ekstrak ke temporary directory proyek dan salin hanya dua file
gpu-inputs, mengikuti panduan handoff; jangan menimpa source/dokumen terkini
dengan snapshot historis di dalam arsip. `tools/v3_gpu_benchmark.py` dapat
berjalan di Windows dan Linux, tanpa modul `resource` khusus Linux.

Bobot yang diperlukan adalah RT-DETR R18 FP32 ONNX revision
`f2bee7444f4c29003df7e9ef07c4c1c6632eca6f`, 82.572.357 byte. Gunakan salinan
lokal yang cocok hash atau unduh sendiri dari
[export penerbit yang dipin](https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/f2bee7444f4c29003df7e9ef07c4c1c6632eca6f/onnx/model.onnx).
Letakkan pada `runtime/v3-visual-models/rtdetr-r18vd-f2bee744/model.onnx`.
SHA256 wajib:
`11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558`.
Tool memeriksanya sebelum membuka model. Lisensi upstream Apache-2.0;
export merujuk upstream tetapi deklarasi lisensi export tersendiri belum ada.
Ini pemakaian riset lokal; persoalan redistribusi bobot rilis tetap terbuka.

## Persiapan Windows PowerShell, dijalankan teman pada drive proyek

Jalankan dari direktori yang memuat `tools/` hasil ekstraksi. Jangan memakai
venv V2 atau memasang paket global. Contoh ini belum dieksekusi pada Windows.

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
py -3.11 --version

New-Item -ItemType Directory -Force runtime/pip-cache-v3-gpu, runtime/tmp-v3-gpu
$env:PIP_CACHE_DIR = Join-Path $PWD 'runtime/pip-cache-v3-gpu'
$env:TMP = Join-Path $PWD 'runtime/tmp-v3-gpu'
$env:TEMP = $env:TMP
py -3.11 -m venv runtime/v3-gpu-venv
runtime/v3-gpu-venv/Scripts/python.exe -m pip install numpy==2.4.6 onnxruntime-gpu==1.23.2 --report runtime/v3-gpu-install-report.json
runtime/v3-gpu-venv/Scripts/python.exe -m pip check

Get-FileHash runtime/v3-visual-models/rtdetr-r18vd-f2bee744/model.onnx -Algorithm SHA256
runtime/v3-gpu-venv/Scripts/python.exe -I -B tools/v3_gpu_benchmark.py --output external_artifacts/v3/gpu-benchmark-20261115/run1
```

[Paket resmi onnxruntime-gpu 1.23.2](https://pypi.org/project/onnxruntime-gpu/1.23.2/)
berlisensi MIT dan tersedia di PyPI. Versi disamakan dengan probe CPU.
Jangan memasang paket `onnxruntime` CPU bersama varian GPU pada venv ini.
Dependensi transitif GPU belum dipin melalui hasil instalasi mesin tujuan;
laporan pip merekam versi, URL dan hash aktual untuk audit berikutnya.
Tidak ada paket GPU diunduh/diinstal pada pekerjaan lokal ini.

Driver yang terlihat di `nvidia-smi` belum membuktikan semua DLL tersedia.
Ikuti [persyaratan resmi CUDA Execution Provider dan preload DLLs](https://onnxruntime.ai/docs/execution-providers/CUDA-ExecutionProvider.html).
Dokumentasi memasangkan keluarga ORT ini dengan CUDA 12 dan cuDNN 9; detail
minor/runtime harus cocok dengan build dan sistem tujuan. Windows juga
memerlukan runtime MSVC yang sesuai. Jangan mengunduh toolkit/PyTorch besar
sebelum lingkungan mesin tersebut diperiksa. `preload_dlls()` hanya memuat
library yang sudah tersedia, bukan menginstalnya. Jika gagal, bawa log error
inisialisasi untuk diagnosis; jangan mengganti versi/precision/model tanpa
protokol eksperimen baru.

## Apa yang diukur dan syarat bukti

Protokol ditulis sebelum session dibuka: device 0, FP32, TF32 dimatikan,
satu thread CPU, batch 1, tiga warmup, sepuluh repetisi pada setiap tiga view.
Waktu `session.run` termasuk transfer host/device dan output kembali ke CPU;
tidak termasuk decode video, preprocessing, association, classifier, atau decoder.
Tidak mengasumsikan latensi GPU sama dengan CPU atau memenuhi pertandingan penuh.

CUDA harus terdaftar dan aktif pada session. Profil ORT harus merekam event
kernel `CUDAExecutionProvider`; CPU-only ditolak, dan fallback exception saat
run dimatikan. Operator shape tertentu boleh ditempatkan di CPU; jumlah event
masing-masing provider dilaporkan. Ini membuktikan GPU dipakai, bukan semua
operator dieksekusi di GPU. Profiling menambah overhead, sehingga hasil diberi
label waktu probe dengan profiling, bukan benchmark throughput murni.

Jika berhasil, kirim `summary.json`, `protocol.json`, `manifest.json`, profil
ORT dan install report. Jika gagal, kirim `failure.json` dan pesan inisialisasi.
Sertakan hasil `nvidia-smi` mesin yang benar-benar menjalankannya. Tidak perlu
mengirim kredensial atau data resmi. Tool mencatat platform/Python/CPU/GPU yang
dilaporkan sistem, dan gagal secara eksplisit tanpa mengarang angka kecepatan.

Lolos benchmark hanya membuka pemeriksaan biaya komputasi. Tidak membuktikan
bola pertandingan dipilih dengan benar atau Spell F1 meningkat. V2/R r1 tetap
kandidat pengumpulan sampai validasi V3 yang relevan terpenuhi.

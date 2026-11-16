# Handoff benchmark GPU V3

Tugas teman sekarang adalah mengukur runtime detector pada mesin GPU aktual.
Tidak training, tidak membuat CSV submission, tidak membuka data resmi, dan
tidak mengganti kandidat V2. Tidak perlu video eksternal penuh. Gunakan prompt
[PROMPT_CODEX_BENCHMARK_GPU_V3.md](PROMPT_CODEX_BENCHMARK_GPU_V3.md) jika dikerjakan
melalui Codex; langkah terminal manual tersedia di bawah.

## 1. Push dari laptop Naufal, Linux

Repo kolaborasi yang remote-nya sudah sesuai berada di lokasi berikut.
Jangan menjalankan git init atau remote add lagi.

```bash
cd /media/naufal/Linux1/new-mtah/external_artifacts/github-handoff-20261112
git status --short
git add --pathspec-from-file=docs/daftar_file_handoff_gpu_v3.txt
git diff --cached --stat
git commit -m "Prepare external V3 GPU benchmark handoff"
git push origin main
```

Daftar tersebut mencakup source/test/dokumentasi pekerjaan lokal sebelumnya,
arsip benchmark sekitar 3,65 MB, checksum, dan catatan verifikasi. Tidak
memasukkan seluruh external_artifacts, runtime, video penuh, atau data resmi.
Periksa daftar staged sebelum commit. Push/commit belum dijalankan oleh agent.

Jika push ditolak karena teman sudah menambah commit, gunakan:

```bash
git pull --rebase origin main
git push origin main
```

Jika ada conflict, selesaikan dahulu; jangan force push atau membuang perubahan.
Jika Git menyebut identitas author belum diatur, tetapkan nama/email Git milikmu
secara lokal pada repo ini. Jangan menggunakan token sebagai nama/email author.

## 2. Teman menarik perubahan

Dari root clone teman, yaitu folder yang memuat .git, tools dan docs:

```bash
git status --short
git pull --ff-only origin main
```

Jika ada perubahan lokal atau pull ditolak, pertahankan pekerjaan teman dan
selesaikan kondisi Git tersebut dahulu. Path root clone teman dapat berbeda
dari path laptop Naufal; jangan menyalin path /media/naufal ke laptop Windows.
Gunakan source benchmark dari checkout terbaru, bukan salinan di dalam ZIP.

## 3. Windows PowerShell: ekstrak input eksternal

Jalankan blok terpisah secara berurutan. Hentikan jika ada error. Jika mengulang,
gunakan folder unpack baru yang belum ada; jangan menimpa file tanpa verifikasi.

```powershell
$ErrorActionPreference = 'Stop'
$archive = 'external_artifacts/v3/visual-association-20261115/gpu-benchmark-inputs.zip'
$expected = ((Get-Content "$archive.sha256" -Raw).Trim() -split '\s+')[0]
if ((Get-FileHash $archive -Algorithm SHA256).Hash.ToLowerInvariant() -ne $expected) { throw 'Archive checksum mismatch' }

New-Item -ItemType Directory -Force runtime | Out-Null
$unpack = 'runtime/v3-benchmark-unpack-1'
if (Test-Path $unpack) { throw 'Choose a fresh unpack directory' }
Expand-Archive -LiteralPath $archive -DestinationPath $unpack
$cache = 'external_artifacts/v3/visual-association-20261115/run1/gpu-inputs'
if (Test-Path $cache) { throw 'Input cache exists: verify it before reusing; do not overwrite' }
New-Item -ItemType Directory -Force $cache | Out-Null
Copy-Item "$unpack/$cache/inputs.npz" "$cache/inputs.npz"
Copy-Item "$unpack/$cache/manifest.json" "$cache/manifest.json"
$inputManifest = Get-Content "$cache/manifest.json" -Raw | ConvertFrom-Json
if ((Get-FileHash "$cache/inputs.npz" -Algorithm SHA256).Hash.ToLowerInvariant() -ne $inputManifest.inputs_sha256) { throw 'Input checksum mismatch' }
```

Arsip tepat 3.645.168 byte; SHA256:
`750bdfe0537fe15c8da63ea28b0d8cb9564c05751128f2967e198a1b1f5fe2d4`.
Hanya `inputs.npz` dan manifest diekstrak ke jalur yang dibaca benchmark.
Dokumen/source di dalam arsip adalah snapshot historis; tetap gunakan checkout.

## 4. Siapkan venv dan bobot, seluruhnya pada drive proyek

Catat GPU yang benar-benar dipakai. Python 3.11 harus tersedia. Jangan memakai
venv V2 atau memasang ke Python global. Jika venv tujuan sudah ada, audit dahulu;
jangan menghapus atau menggantinya secara otomatis.

```powershell
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
py -3.11 --version
New-Item -ItemType Directory -Force runtime/pip-cache-v3-gpu, runtime/tmp-v3-gpu | Out-Null
$env:PIP_CACHE_DIR = Join-Path $PWD 'runtime/pip-cache-v3-gpu'
$env:TMP = Join-Path $PWD 'runtime/tmp-v3-gpu'
$env:TEMP = $env:TMP
$env:PYTHONNOUSERSITE = '1'
if (Test-Path runtime/v3-gpu-venv) { throw 'GPU venv exists: audit/reuse it explicitly' }
py -3.11 -m venv runtime/v3-gpu-venv
runtime/v3-gpu-venv/Scripts/python.exe -m pip install numpy==2.4.6 onnxruntime-gpu==1.23.2 --report runtime/v3-gpu-install-report.json
if ($LASTEXITCODE -ne 0) { throw 'Dependency installation failed' }
runtime/v3-gpu-venv/Scripts/python.exe -m pip check
if ($LASTEXITCODE -ne 0) { throw 'Dependency check failed' }
```

Unduh bobot terpin sekitar 82,6 MB, hanya jika salinan yang cocok belum ada.
Tidak membutuhkan token untuk URL model publik ini. Jangan unduh ulang jika
file sudah cocok; checksum tidak cocok harus diselidiki, bukan diabaikan.

```powershell
$modelDir = 'runtime/v3-visual-models/rtdetr-r18vd-f2bee744'
$model = "$modelDir/model.onnx"
$modelSha = '11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558'
New-Item -ItemType Directory -Force $modelDir | Out-Null
if (-not (Test-Path $model)) {
    $partial = "$model.partial"
    if (Test-Path $partial) { throw 'Partial exists: inspect previous download first' }
    Invoke-WebRequest -Uri 'https://huggingface.co/onnx-community/rtdetr_r18vd/resolve/f2bee7444f4c29003df7e9ef07c4c1c6632eca6f/onnx/model.onnx' -OutFile $partial
    if ((Get-FileHash $partial -Algorithm SHA256).Hash.ToLowerInvariant() -ne $modelSha) { throw 'Downloaded model checksum mismatch' }
    Move-Item $partial $model
}
if ((Get-FileHash $model -Algorithm SHA256).Hash.ToLowerInvariant() -ne $modelSha) { throw 'Model checksum mismatch' }
```

Library CUDA/cuDNN/MSVC dapat masih kurang meskipun driver terpasang. Audit
kebutuhan runtime mengikuti [panduan teknis](benchmark_gpu_v3_20261115.md)
dan sumber resmi yang ditautkan di sana. Instalasi pip di atas belum menjamin
semua DLL tersedia. Jangan mengganti model, versi ORT, precision, atau mengizinkan
CPU fallback agar hasil terlihat sukses. Tidak perlu memasang PyTorch/toolkit
global tanpa diagnosis. Codex dapat memeriksa dependensi runtime NVIDIA yang
benar-benar dibutuhkan dan memasangnya di venv proyek, dengan provenance tercatat.

## 5. Jalankan benchmark dan catat hasil

```powershell
New-Item -ItemType Directory -Force runtime/v3-gpu-machine | Out-Null
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | Out-File runtime/v3-gpu-machine/nvidia-smi.txt -Encoding utf8
runtime/v3-gpu-venv/Scripts/python.exe -I -B tools/v3_gpu_benchmark.py --output external_artifacts/v3/gpu-benchmark-20261115/run1
```

Jika `run1` sudah ada, simpan dan gunakan `run2`, bukan menimpanya. Untuk sesi
terminal baru, set kembali PIP_CACHE_DIR/TMP/TEMP ke drive proyek sebelum
instalasi tambahan. Benchmark memerlukan CUDA aktif dan kernel CUDA di profil.
CPU-only dianggap gagal secara eksplisit. Durasi ini mencakup profiling dan
transfer tensor, bukan video decode atau possession end-to-end.

Codex menulis **docs/hasil_benchmark_gpu_v3.md** dengan hasil aktual:

- Mesin/OS/Python/GPU/driver yang benar-benar menjalankan benchmark.
- Versi paket, hasil pip check, runtime CUDA/cuDNN yang berhasil dimuat atau error.
- Status CUDA, hitungan kernel per provider, median/p95 per view, serta batas interpretasinya.
- Hash source/model/input dan lokasi output, termasuk kegagalan jika terjadi.
- Tidak mengklaim kualitas bola, training V3, atau kenaikan Spell F1.

Jika berhasil, paketkan folder output run yang dipakai, install report, serta
runtime/v3-gpu-machine/nvidia-smi.txt ke ZIP lokal. Jika gagal, sertakan failure.json
jika ada dan log error terminal; error sebelum validasi input/session mungkin
belum menghasilkan failure.json. Periksa log agar tidak membawa kredensial.
Bagikan arsip hasil kepada Naufal; hasil runtime masih diabaikan Git. Untuk
mengembalikan ringkasan melalui repo, teman dapat menjalankan:

```bash
git add docs/hasil_benchmark_gpu_v3.md
git commit -m "Report V3 GPU benchmark on execution machine"
git push origin main
```

Jika ditolak karena remote lebih baru, rebase dengan aman dahulu. Jangan
force-add runtime, bobot, video, atau seluruh external_artifacts. Agent tidak
mengirim pesan/file ke orang lain tanpa instruksi eksplisit.

## Linux pada mesin GPU

Gunakan prosedur yang sama, mengganti interpreter dengan python3.11 dan path
venv menjadi runtime/v3-gpu-venv/bin/python. Dari root clone, blok ini memeriksa
checksum ZIP dan mengekstrak hanya input, tanpa menimpa cache yang sudah ada:

```bash
python3 -I -B - <<'PY'
from pathlib import Path
import hashlib, zipfile
root = Path.cwd()
stage = root/'external_artifacts/v3/visual-association-20261115'
archive = stage/'gpu-benchmark-inputs.zip'
expected = (stage/'gpu-benchmark-inputs.zip.sha256').read_text().split()[0]
assert hashlib.sha256(archive.read_bytes()).hexdigest() == expected
cache = stage/'run1/gpu-inputs'
assert not cache.exists(), 'Cache exists: verify before reusing'
with zipfile.ZipFile(archive) as zipped:
    cache.mkdir(parents=True)
    for name in ('inputs.npz', 'manifest.json'):
        relative = str((cache/name).relative_to(root))
        (cache/name).write_bytes(zipped.read(relative))
PY
```

Pasang runtime pada drive proyek; audit dahulu jika venv tujuan sudah ada:

```bash
mkdir -p runtime/pip-cache-v3-gpu runtime/tmp-v3-gpu
export PIP_CACHE_DIR="$PWD/runtime/pip-cache-v3-gpu"
export TMPDIR="$PWD/runtime/tmp-v3-gpu"
python3.11 -m venv runtime/v3-gpu-venv
runtime/v3-gpu-venv/bin/python -m pip install numpy==2.4.6 onnxruntime-gpu==1.23.2 --report runtime/v3-gpu-install-report.json
runtime/v3-gpu-venv/bin/python -m pip check
```

Sediakan bobot di jalur yang sama dan cocokkan SHA256 sebelum menjalankan:

```bash
runtime/v3-gpu-venv/bin/python -I -B tools/v3_gpu_benchmark.py --output external_artifacts/v3/gpu-benchmark-20261115/run1
```

Tidak ada klaim langkah Windows/GPU telah diuji di mesin teman. Yang sudah
diuji lokal: tujuh tes sintetis dan guard penolakan CPU-only; V2 tetap terverifikasi.

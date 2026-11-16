"""Cross-platform hardware timing on fixed external tensors; no video/GT input.

Requires an isolated ONNX Runtime GPU environment on the execution machine.
No download, training, threshold selection, or possession output. CPU-only
execution fails explicitly. Some shape operators may legitimately run on CPU;
the profile reports both providers and must contain CUDA kernel events.
"""
import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
MODEL_SHA = '11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558'


def sha(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def require_cuda(providers):
    """Reject silent CPU fallback; registered provider alone is not kernel proof."""
    if 'CUDAExecutionProvider' not in providers:
        raise RuntimeError('CUDA execution provider unavailable; no CPU benchmark substituted')


def profile_providers(events):
    """Count actual profiled node/kernel provider events; require CUDA execution."""
    counts = Counter(e.get('args', {}).get('provider') for e in events
                     if e.get('cat') == 'Node' and e.get('args', {}).get('provider'))
    if not counts['CUDAExecutionProvider']:
        raise RuntimeError('No CUDA kernel events in profile; GPU throughput is unverified')
    return dict(counts)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT/'external_artifacts/v3').resolve())
    if output.exists():
        raise FileExistsError(output)
    # No arbitrary data path: accept only this externally generated cache.
    cache = ROOT/'external_artifacts/v3/visual-association-20261115/run1/gpu-inputs'
    model = ROOT/'runtime/v3-visual-models/rtdetr-r18vd-f2bee744/model.onnx'
    manifest = json.loads((cache/'manifest.json').read_text())
    if (manifest['official_data'] is not False or manifest['source_frame_0_based'] != 4375
            or manifest['dataset_revision'] != '325bd4c57ee8e135ab652f3921d274526ada496b'
            or manifest['model_sha256'] != MODEL_SHA or sha(model) != MODEL_SHA
            or sha(cache/'inputs.npz') != manifest['inputs_sha256']):
        raise ValueError('External/model checksum or provenance mismatch')
    import numpy as np
    import onnxruntime as ort
    if ort.__version__ != '1.23.2':
        raise RuntimeError('Use pinned onnxruntime-gpu==1.23.2 in an isolated environment')
    with np.load(cache/'inputs.npz', allow_pickle=False) as archive:
        tensors = {name: archive[name] for name in archive.files}
    if set(tensors) != {'full', 'tile-0-1', 'tile-1-3'} or set(manifest['views']) != set(tensors):
        raise ValueError('Incomplete or unexpected benchmark samples')
    for tensor in tensors.values():
        if (tensor.shape != (1, 3, 640, 640) or tensor.dtype != np.float32
                or not np.isfinite(tensor).all() or tensor.min() < 0 or tensor.max() > 1):
            raise ValueError('Invalid cached input tensor')
    output.mkdir(parents=True)
    protocol = dict(ort_version='1.23.2', model_sha256=MODEL_SHA,
        cache_manifest_sha256=sha(cache/'manifest.json'), sample_names=list(tensors),
        device_id=0, warmup_runs=3, repeats_per_sample=10, batch_size=1,
        cpu_threads=1, tf32=False, includes_host_device_transfer=True,
        preprocessing_and_decode_timed=False, cuda_kernel_proof_required=True,
        full_possession_throughput=False, official_data_access=False)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    try:
        # Supported ORT loader; no package download or external model code.
        if hasattr(ort, 'preload_dlls'):
            ort.preload_dlls()
        require_cuda(ort.get_available_providers())
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        options.enable_profiling = True
        options.profile_file_prefix = str(output/'ort-profile')
        tick = time.perf_counter()
        session = ort.InferenceSession(str(model), options, providers=[
            ('CUDAExecutionProvider', {'device_id': 0, 'use_tf32': 0}), 'CPUExecutionProvider'])
        load_seconds = time.perf_counter()-tick
        session.disable_fallback()
        require_cuda(session.get_providers())
        if {i.name for i in session.get_inputs()} != {'pixel_values'}:
            raise ValueError('Unexpected model inputs')
        for _ in range(protocol['warmup_runs']):
            session.run(None, {'pixel_values': next(iter(tensors.values()))})
        timings = {name: [] for name in tensors}
        for _ in range(protocol['repeats_per_sample']):
            for name, tensor in tensors.items():
                tick = time.perf_counter()
                values = session.run(None, {'pixel_values': tensor})
                elapsed = time.perf_counter()-tick
                if not all(np.isfinite(v).all() for v in values):
                    raise ValueError('Nonfinite model outputs')
                timings[name].append(elapsed)
        profile = Path(session.end_profiling())
        providers = profile_providers(json.loads(profile.read_text()))
        try:
            gpu = subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,memory.total',
                '--format=csv,noheader'], text=True, stderr=subprocess.STDOUT, timeout=10).strip()
        except (OSError, subprocess.SubprocessError):
            gpu = 'nvidia-smi unavailable; GPU model/driver not verified'
        try:
            gpu_package = importlib.metadata.version('onnxruntime-gpu')
        except importlib.metadata.PackageNotFoundError:
            gpu_package = None
        summary = dict(status='actual_cuda_kernels_observed', machine=dict(
            system=platform.platform(), python=platform.python_version(),
            processor=platform.processor(), logical_cpus=os.cpu_count(), nvidia_smi=gpu),
            onnxruntime_gpu_package=gpu_package, numpy_version=np.__version__,
            session_providers=session.get_providers(), kernel_events_by_provider=providers,
            model_load_seconds=load_seconds,
            inference_seconds_by_view={name: dict(median=float(np.median(t)),
                p95=float(np.quantile(t, .95)), runs=len(t)) for name, t in timings.items()},
            raw_seconds_by_view=timings, quality_metrics_available=False,
            full_pipeline_throughput=False, official_data_access=False)
        (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    except Exception as exc:
        (output/'failure.json').write_text(json.dumps(dict(status='benchmark_not_completed',
            error_type=type(exc).__name__, message=str(exc), gpu_speed_claim=False), indent=2)+'\n')
        raise
    finally:
        (output/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),
            files_sha256={p.name: sha(p) for p in output.iterdir() if p.is_file() and p.name != 'manifest.json'},
            official_data_access=False), indent=2)+'\n')


if __name__ == '__main__':
    main()

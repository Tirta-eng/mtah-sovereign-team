"""Audit explicitly pinned external pixels/tensors; no model/official inputs."""
import argparse
import ast
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
FREEZE = ROOT / 'external_artifacts/v3/117093-ball-reference-freeze-20261115'
PACKET = ROOT / 'external_artifacts/v3/117093-ball-reference-task-20261115'
CACHE = ROOT / 'external_artifacts/v3/visual-association-20261115/run1/gpu-inputs'
PROCESSOR = ROOT / 'runtime/v3-input-audit-machine/primary-metadata/preprocessor_config.json'
FREEZE_SHA = 'b0bb762d62862d7961870dc5f7ca3f700c169982cf25bc384e6de07d4c91d86e'
VISUAL_SHA = '6613388ec25dff638bcf8824585ed43014935887da274bcd0305e855251d96fa'
CACHE_MANIFEST_SHA = '9d93095f501aa139cd451fc4a22f490f26c7d431e01d4e5d77a101be42bd912c'
CACHE_SHA = '2e7a29113316881b4744956aaa99871f98513716850151f7ef1714e41a7abe68'
PROCESSOR_SHA = 'cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def verify(path, expected):
    if sha(path) != expected:
        raise ValueError('Checksum mismatch: ' + str(path))


def contained(base, name):
    path = (base / name).resolve(strict=True)
    path.relative_to(base.resolve())
    return path


def compare_arrays(a, b):
    """Exact equality and diagnostic numeric difference, never quality scores."""
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    if a.shape != b.shape or a.dtype != b.dtype:
        raise ValueError('Array contracts differ')
    if not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError('Nonfinite audit array')
    difference = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return dict(exact_equal=bool(np.array_equal(a, b)), shape=list(a.shape), dtype=str(a.dtype),
        differing_values=int(np.count_nonzero(a != b)),
        maximum_absolute_difference=float(difference.max()),
        mean_absolute_difference=float(difference.mean()),
        a_sha256=hashlib.sha256(a.tobytes(order='C')).hexdigest(),
        b_sha256=hashlib.sha256(b.tobytes(order='C')).hexdigest())


def main(output):
    import importlib.metadata
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    output = output.resolve()
    output.relative_to((ROOT / 'external_artifacts/v3').resolve())
    if output.exists():
        raise FileExistsError(output)
    verify(FREEZE / 'freeze-manifest.json', FREEZE_SHA)
    frozen = json.loads((FREEZE / 'freeze-manifest.json').read_text())
    for name, entry in frozen['snapshot_files'].items():
        path = contained(FREEZE, name)
        if path.stat().st_size != entry['bytes']:
            raise ValueError('Frozen file size mismatch')
        verify(path, entry['sha256'])
    for name, expected in frozen['immutable_source_files_sha256'].items():
        verify(contained(PACKET, name), expected)
    plan = json.loads((ROOT / 'docs/soccertrack_v1_2_117093_download.json').read_text())
    if plan['revision'] != frozen['dataset_revision'] or plan['match'] != '117093':
        raise ValueError('Wrong external dataset provenance')
    entry, = [e for e in plan['files'] if e['path'] == 'videos/117093/117093_panorama_1st_half.mp4']
    video = contained(ROOT / 'Soccertrack V2', entry['path'])
    if video.stat().st_size != entry['bytes'] or entry['payload_sha256'] != frozen['source_video_sha256']:
        raise ValueError('Wrong external video')
    verify(video, entry['payload_sha256'])  # before any video decoder opens it
    verify(CACHE / 'manifest.json', CACHE_MANIFEST_SHA)
    verify(CACHE / 'inputs.npz', CACHE_SHA)
    cache_manifest = json.loads((CACHE / 'manifest.json').read_text())
    if (cache_manifest['official_data'] is not False or cache_manifest['source_frame_0_based'] != 4375
            or cache_manifest['video_sha256'] != entry['payload_sha256']
            or cache_manifest['dataset_revision'] != plan['revision']):
        raise ValueError('Unexpected tensor-cache provenance')
    verify(PROCESSOR, PROCESSOR_SHA)
    visual = ROOT / 'tools/v3_visual_probe.py'
    verify(visual, VISUAL_SHA)
    if cache_manifest['preprocessing_source_sha256'] != VISUAL_SHA or cache_manifest['processor_sha256'] != PROCESSOR_SHA:
        raise ValueError('Cache preprocessing source/config differs')
    tree = ast.parse(visual.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in ('preprocess', 'decode')]
    if {node.name for node in functions} != {'preprocess', 'decode'}:
        raise ValueError('Expected exactly the frozen preprocess/decode functions')
    executables = list((Path(imageio_ffmpeg.__file__).parent / 'binaries').glob('*.exe'))
    ffmpeg, = executables
    ffmpeg.resolve().relative_to(Path(sys.prefix).resolve())
    version = subprocess.run([str(ffmpeg), '-version'], check=True, capture_output=True, text=True, timeout=10).stdout
    source_manifest = json.loads((FREEZE / 'source-packet-manifest.json').read_text())
    images = source_manifest['images']
    if [e['source_frame_0_based'] for e in images] != list(range(4375, 4383)):
        raise ValueError('Unexpected frozen frame sequence')
    output.mkdir(parents=True, exist_ok=False)
    (output / 'legacy-seek').mkdir()
    protocol = dict(kind='external_input_equivalence_audit',
        source_video_sha256=entry['payload_sha256'], frozen_annotation_manifest_sha256=FREEZE_SHA,
        visual_source_sha256=VISUAL_SHA, cache_sha256=CACHE_SHA, processor_sha256=PROCESSOR_SHA,
        ffmpeg_sha256=sha(ffmpeg), frames=list(range(4375,4383)), fps=25,
        pixel_comparison='exact RGB uint8 equality; report differences without acceptance tolerance',
        cache_comparison='exact float32 C-order tensor equality at B01 only, three historical views',
        historical_scope='Actual old raw frames/72 outputs absent; current decoder recreation is not historical execution proof.',
        no_timestamp_offset_search=True, no_model_inference=True, no_threshold_tuning=True,
        no_provider_position_access=True, official_data_access=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n', encoding='utf-8')
    (output / 'ffmpeg-version.txt').write_text(version, encoding='utf-8')
    commands = []
    current_frame = None

    def run_legacy(command):
        expected = ['ffmpeg','-nostdin','-v','error','-threads','1','-ss',str(current_frame/25),
            '-i',str(video),'-map','0:v:0','-an','-sn','-frames:v','1',
            '-filter_threads','1','-f','image2pipe','-vcodec','png','pipe:1']
        if command != expected:
            raise ValueError('Legacy decode command changed')
        actual = [str(ffmpeg)] + command[1:]
        result = subprocess.run(actual, check=True, capture_output=True, timeout=45)
        if len(result.stdout) > 30_000_000:
            raise ValueError('Unexpected decode output size')
        (output / f'legacy-seek/frame-{current_frame}.png').write_bytes(result.stdout)
        commands.append(dict(frame=current_frame, argv=actual, stderr=result.stderr.decode('utf-8', errors='replace')))
        return result.stdout

    namespace = dict(io=io, subprocess=SimpleNamespace(check_output=run_legacy))
    # Never import the Linux-only probe module or execute main/model access.
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(visual), 'exec'), namespace)
    preprocess, decode = namespace['preprocess'], namespace['decode']
    processor = json.loads(PROCESSOR.read_text())
    comparisons, seq_samples, seek_samples = [], {}, {}
    views = [('full',(0,0,4096,1080)),('tile-0-1',(1024,0,1024,540)),('tile-1-3',(3072,540,1024,540))]
    for image in images:
        current_frame = image['source_frame_0_based']
        raw_path = contained(PACKET, image['path'])
        verify(raw_path, image['sha256'])
        with Image.open(raw_path) as raw:
            reference = raw.convert('RGB')
        recreated = decode(video, current_frame)
        check = compare_arrays(np.asarray(reference), np.asarray(recreated))
        comparisons.append(dict(image_id=image['image_id'], source_frame_0_based=current_frame,
            raw_png_sha256=image['sha256'], local_legacy_seek_png_sha256=sha(output / f'legacy-seek/frame-{current_frame}.png'),
            pixels=check))
        if current_frame == 4375:
            for view, (x,y,w,h) in views:
                seq_samples[view] = preprocess(reference.crop((x,y,x+w,y+h)), processor)
                seek_samples[view] = preprocess(recreated.crop((x,y,x+w,y+h)), processor)
        print(json.dumps(dict(frame=current_frame, exact_rgb_equal=check['exact_equal'], max_difference=check['maximum_absolute_difference'])), flush=True)
    with np.load(CACHE / 'inputs.npz', allow_pickle=False) as cache:
        if set(cache.files) != {name for name,_ in views}:
            raise ValueError('Unexpected historical cache views')
        tensors = []
        for name,_ in views:
            array = cache[name]
            if array.shape != (1,3,640,640) or array.dtype != np.float32:
                raise ValueError('Unexpected historical tensor contract')
            tensors.append(dict(view=name, sequential_vs_historical=compare_arrays(seq_samples[name],array),
                local_legacy_seek_vs_historical=compare_arrays(seek_samples[name],array)))
    np.savez_compressed(output / 'b01-sequential-tensors.npz', **seq_samples)
    np.savez_compressed(output / 'b01-local-legacy-seek-tensors.npz', **seek_samples)
    report = dict(status='external_input_equivalence_audit_completed',
        current_legacy_seek_vs_frozen_raw_exact_all_8=all(r['pixels']['exact_equal'] for r in comparisons),
        b01_three_historical_tensors_exact_sequential=all(r['sequential_vs_historical']['exact_equal'] for r in tensors),
        b01_three_historical_tensors_exact_current_legacy_seek=all(r['local_legacy_seek_vs_historical']['exact_equal'] for r in tensors),
        historical_72_inference_inputs_verified=False, historical_model_outputs_present=False,
        frame_comparisons=comparisons, tensor_comparisons=tensors,
        packages={n:importlib.metadata.version(n) for n in ('numpy','Pillow','imageio-ffmpeg')},
        decoder_cross_platform_equivalence_asserted=False, annotations_changed=False,
        model_inference=False, official_data_access=False, ready_for_component_scoring=False,
        missing_gates=['localization uncertainty/acceptance protocol','all-frame historical inference-input evidence and outputs'])
    (output / 'commands.json').write_text(json.dumps(commands,indent=2)+'\n',encoding='utf-8')
    (output / 'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    members = {p.relative_to(output).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(output.rglob('*')) if p.is_file()}
    (output / 'manifest.json').write_text(json.dumps(dict(kind='external_input_equivalence_audit',
        audit_source_sha256=sha(Path(__file__)),files=members,official_data_access=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ('status','current_legacy_seek_vs_frozen_raw_exact_all_8',
        'b01_three_historical_tensors_exact_sequential','b01_three_historical_tensors_exact_current_legacy_seek',
        'historical_72_inference_inputs_verified','model_inference','official_data_access')},indent=2))


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)

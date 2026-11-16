"""Sequential decoder diagnosis on fixed external frames, never clock fitting."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]


def main(confirm_boundary=False):
    import imageio_ffmpeg
    import numpy as np
    from PIL import Image
    spec=importlib.util.spec_from_file_location('input_audit',ROOT/'tools/audit_ball_reference_inputs_117093.py')
    audit=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    packet=audit.PACKET
    first=ROOT/'external_artifacts/v3/117093-ball-input-audit-20261115/run1'
    output=ROOT/'external_artifacts/v3/117093-ball-input-audit-20261115'/('boundary-confirmation' if confirm_boundary else 'sequential-diagnosis')
    if output.exists():
        raise FileExistsError(output)
    audit.verify(audit.FREEZE/'freeze-manifest.json',audit.FREEZE_SHA)
    freeze=json.loads((audit.FREEZE/'freeze-manifest.json').read_text())
    for name,entry in freeze['snapshot_files'].items():
        audit.verify(audit.contained(audit.FREEZE,name),entry['sha256'])
    for name,expected in freeze['immutable_source_files_sha256'].items():
        audit.verify(audit.contained(packet,name),expected)
    previous=json.loads((first/'manifest.json').read_text())
    for name,entry in previous['files'].items():
        audit.verify(audit.contained(first,name),entry['sha256'])
    plan=json.loads((ROOT/'docs/soccertrack_v1_2_117093_download.json').read_text())
    entry,=[e for e in plan['files'] if e['path']=='videos/117093/117093_panorama_1st_half.mp4']
    video=audit.contained(ROOT/'Soccertrack V2',entry['path'])
    if entry['payload_sha256']!=freeze['source_video_sha256'] or video.stat().st_size!=entry['bytes']:
        raise ValueError('Wrong external video')
    audit.verify(video,entry['payload_sha256'])
    ffmpeg,=list((Path(imageio_ffmpeg.__file__).parent/'binaries').glob('*.exe'))
    ffmpeg.resolve().relative_to(Path(sys.prefix).resolve())
    frames=[4358] if confirm_boundary else list(range(4350,4358))+list(range(4375,4383))+list(range(4400,4408))
    output.mkdir(parents=True,exist_ok=False)
    (output/'source.py').write_bytes(Path(__file__).read_bytes())
    references={}
    legacy={}
    for i in range(8):
        with Image.open(packet/f'B{i+1:02}-frame-{4375+i}-raw.png') as image:
            references[4375+i]=np.array(image.convert('RGB'))
        with Image.open(first/f'legacy-seek/frame-{4375+i}.png') as image:
            legacy[4375+i]=np.array(image.convert('RGB'))
    hashes=lambda images:{hashlib.sha256(a.tobytes()).hexdigest():n for n,a in images.items()}
    reference_hashes,legacy_hashes=hashes(references),hashes(legacy)
    expression='select=eq(n\\,4358),showinfo' if confirm_boundary else 'select=between(n\\,4350\\,4357)+between(n\\,4375\\,4382)+between(n\\,4400\\,4407),showinfo'
    command=[str(ffmpeg),'-nostdin','-hide_banner','-threads','1','-i',str(video),'-map','0:v:0','-an','-sn',
        '-vf',expression,'-filter_threads','1','-fps_mode','passthrough','-frames:v',str(len(frames)),
        '-pix_fmt','rgb24','-f','rawvideo','pipe:1']
    protocol=dict(kind='external_decoder_sequence_diagnosis',source_video_sha256=entry['payload_sha256'],
        frames=frames,selection=('Explicit follow-up: test frame 4358, immediately after the first diagnostic group, to confirm the eighth legacy decode; no annotation/model-score access or clock adoption.' if confirm_boundary else 'Fixed three groups: requested indices and +/-25 frames to diagnose a possible 1 s start-PTS effect; no best-offset optimisation.'),
        decode='From beginning, select by FFmpeg n; exact RGB matching only, no annotation-coordinate comparison',
        source_sha256=audit.sha(Path(__file__)),ffmpeg_sha256=audit.sha(ffmpeg),command=command,
        official_data_access=False,model_inference=False,clock_or_annotations_changed=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    records=[]
    started=time.monotonic()
    with (output/'ffmpeg-showinfo.txt').open('wb') as log:
        process=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=log,stdin=subprocess.DEVNULL)
        try:
            for number in frames:
                parts=[]
                remaining=4096*1080*3
                while remaining:
                    chunk=process.stdout.read(remaining)
                    if not chunk:
                        raise ValueError('Sequential decoder ended early')
                    parts.append(chunk)
                    remaining-=len(chunk)
                payload=b''.join(parts)
                digest=hashlib.sha256(payload).hexdigest()
                array=np.frombuffer(payload,dtype=np.uint8).reshape(1080,4096,3)
                record=dict(ffmpeg_sequential_frame_0_based=number,rgb_sha256=digest,
                    exact_match_reference_frame=reference_hashes.get(digest),
                    exact_match_legacy_requested_frame=legacy_hashes.get(digest))
                if number in references:
                    record['same_index_reference_comparison']=audit.compare_arrays(array,references[number])
                    record['same_index_legacy_comparison']=audit.compare_arrays(array,legacy[number])
                    Image.fromarray(array).save(output/f'sequential-frame-{number}.png')
                records.append(record)
                print(json.dumps({k:v for k,v in record.items() if not k.endswith('comparison')}),flush=True)
            if process.stdout.read(1):
                raise ValueError('Unexpected extra selected frame')
            if process.wait(timeout=15):
                raise RuntimeError('FFmpeg sequential decode failed')
        except BaseException:
            process.kill()
            process.wait()
            raise
        finally:
            process.stdout.close()
    report=dict(status='sequential_diagnosis_completed',selected_frames=len(records),seconds=time.monotonic()-started,
        records=records,clock_or_annotations_changed=False,official_data_access=False,model_inference=False,
        no_automatic_reference_relabel_or_offset_adoption=True)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    members={p.name:dict(bytes=p.stat().st_size,sha256=audit.sha(p)) for p in output.iterdir() if p.is_file()}
    (output/'manifest.json').write_text(json.dumps(dict(files=members,official_data_access=False),indent=2)+'\n',encoding='utf-8')


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--confirm-boundary',action='store_true')
    main(parser.parse_args().confirm_boundary)

"""External-only single-annotator pilot preparation; never executes a model."""
import argparse
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
FRAMES=(5250,5500,5750,6000)


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def overlap(a,b):
    return max(a[0],b[0])<=min(a[1],b[1])


def load_fields():
    spec=importlib.util.spec_from_file_location('reference_fields',ROOT/'tools/prepare_ball_reference_117093.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FRAME_FIELDS,module.OBJECT_FIELDS


def prepare(output):
    import cv2
    plan_path=ROOT/'docs/soccertrack_v1_2_117093_download.json'
    plan=json.loads(plan_path.read_text(encoding='utf-8'))
    if plan['match']!='117093' or plan['revision']!='325bd4c57ee8e135ab652f3921d274526ada496b':
        raise ValueError('Wrong external provenance')
    entry,=[e for e in plan['files'] if e['path']=='videos/117093/117093_panorama_1st_half.mp4']
    video=(ROOT/'Soccertrack V2'/entry['path']).resolve(strict=True)
    video.relative_to((ROOT/'Soccertrack V2').resolve())
    if video.stat().st_size!=entry['bytes'] or sha(video)!=entry['payload_sha256']:
        raise ValueError('External video checksum/size mismatch')
    prior_path=ROOT/'external_artifacts/v3/117093-possession-review-pilot-20261113/manifest.json'
    prior=json.loads(prior_path.read_text(encoding='utf-8'))
    if prior['match']!='117093' or prior['dataset_revision']!=plan['revision'] or prior['source_video_sha256']!=entry['payload_sha256']:
        raise ValueError('Unexpected prior-exposure metadata')
    excluded=[(int(c['source_frame_start_0_based'])-25,int(c['source_frame_end_inclusive_0_based'])+25) for c in prior['clips']]
    # Read only the fixed centre constants from source AST, not review answers.
    import ast
    interval_source=ROOT/'tools/prepare_interval_review_117093.py'
    tree=ast.parse(interval_source.read_text(encoding='utf-8'))
    centres,=[ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CENTRES_SECONDS' for t in node.targets)]
    excluded.extend(((c-15)*25-25,(c+15)*25+24) for c in centres)
    excluded.extend([(4225,4524),(7500,7500),(37500,37500),(60000,60000)])
    windows=[(f-75,f+74) for f in FRAMES]
    if any(overlap(w,e) for w in windows for e in excluded):
        raise ValueError('Pilot context overlaps recorded old review/exposure window')
    if output.exists():raise FileExistsError(output)
    if shutil.disk_usage(ROOT).free<1024**3:raise RuntimeError('Insufficient project space')
    frame_fields,object_fields=load_fields()
    output.mkdir(parents=True,exist_ok=False)
    (output/'tiles-P01').mkdir()
    shutil.copyfile(ROOT/'docs/protokol_repeatability_bola_117093.md',output/'protocol.md')
    shutil.copyfile(Path(__file__),output/'preparation-source.py')
    protocol=dict(kind='single_annotator_ball_repeatability_pilot',match='117093',dataset_revision=plan['revision'],
        source_video_sha256=entry['payload_sha256'],frames=list(FRAMES),context_frames=windows,
        round='A',round_b_not_yet_prepared=True,reviewers_available=1,
        initial_reply='Hanya saya yang bisa menilai',prior_exposure_manifest_sha256=sha(prior_path),
        prior_interval_source_sha256=sha(interval_source),excluded_windows=excluded,
        sampling='fixed frame numbers; no model/provider labels or clarity-based replacement',
        decode='sequential from frame 0, no timestamp seek',protocol_sha256=sha(output/'protocol.md'),
        new_frame_model_predictions_read=False,provider_labels_or_positions_read=False,official_data_access=False,
        trained=False,inter_annotator_agreement_measurable=False,absolute_accuracy_established=False,
        ready_for_numeric_acceptance=False,production_eligible=False)
    (output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n',encoding='utf-8')
    initial=dict(source='operator_conversation_reply',reply='Hanya saya yang bisa menilai',reviewers_available=1,
        initial_prior_exposure_on_new_samples='unknown_pending_confirmation',coordinate_space='raw4096x1080',
        independent_second_reviewer_available=False,labels_generated=False,model_predictions_generated=False)
    (output/'initial-provenance.json').write_text(json.dumps(initial,indent=2)+'\n',encoding='utf-8')
    (output/'operator-measurements.jsonl').write_text(json.dumps(initial)+'\n',encoding='utf-8')
    cap=cv2.VideoCapture(str(video))
    if (not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT))!=67625
        or abs(cap.get(cv2.CAP_PROP_FPS)-25)>1e-6
        or (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))!=(4096,1080)
        or round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=0):
        cap.release()
        raise ValueError('Unexpected external stream contract')
    writers={}
    images=[]
    tiles=[]
    counts={f:0 for f in FRAMES}
    try:
        for i,f in enumerate(FRAMES,1):
            writer=cv2.VideoWriter(str(output/f'P{i:02}-context.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),25,(4096,1128))
            if not writer.isOpened():raise RuntimeError('Context writer unavailable')
            writers[f]=writer
        for n in range(FRAMES[-1]+75):
            ok,image=cap.read()
            if not ok or image.shape!=(1080,4096,3) or round(cap.get(cv2.CAP_PROP_POS_FRAMES))!=n+1:
                raise ValueError('Sequential decode/index failure')
            if n in FRAMES:
                image_id=f'P{FRAMES.index(n)+1:02}'
                name=f'{image_id}-raw.png'
                if not cv2.imwrite(str(output/name),image):raise OSError('PNG write failed')
                images.append(dict(image_id=image_id,source_frame_0_based=n,path=name,sha256=sha(output/name),
                    width=4096,height=1080,context_path=f'{image_id}-context.mp4',target_context_seconds=3.0))
                if n==FRAMES[0]:
                    for r in range(2):
                        for c in range(4):
                            x,y=c*1024,r*540
                            path=f'tiles-P01/P01-r{r}-c{c}.png'
                            if not cv2.imwrite(str(output/path),image[y:y+540,x:x+1024]):raise OSError('Tile write failed')
                            tiles.append(dict(path=path,image_id=image_id,x_origin=x,y_origin=y,width=1024,height=540,sha256=sha(output/path)))
            for i,f in enumerate(FRAMES,1):
                if f-75<=n<=f+74:
                    context=cv2.copyMakeBorder(image,48,0,0,0,cv2.BORDER_CONSTANT,value=(0,0,0))
                    caption=f'P{i:02} | context only | target at 3.00 s'
                    if n==f:caption+=' | TARGET'
                    cv2.putText(context,caption,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),1)
                    writers[f].write(context)
                    counts[f]+=1
            if n%1500==0:print(json.dumps(dict(decoded_through_frame=n)),flush=True)
    finally:
        cap.release()
        for writer in writers.values():writer.release()
    if len(images)!=4 or any(c!=150 for c in counts.values()):raise ValueError('Incomplete pilot media')
    for image in images:
        decoded=cv2.imread(str(output/image['path']),cv2.IMREAD_COLOR)
        if decoded is None or decoded.shape!=(1080,4096,3):raise ValueError('Saved raw PNG invalid')
    checks=[]
    for image in images:
        clip=cv2.VideoCapture(str(output/image['context_path']))
        try:
            if not clip.isOpened() or round(clip.get(cv2.CAP_PROP_FRAME_COUNT))!=150:raise ValueError('Invalid clip')
            for n in (0,75,149):
                clip.set(cv2.CAP_PROP_POS_FRAMES,n)
                ok,frame=clip.read()
                if not ok or frame.shape!=(1128,4096,3):raise ValueError('Invalid context sample')
                checks.append(dict(image_id=image['image_id'],clip_frame=n))
        finally:clip.release()
    for name in ('frames.csv','frames-template.csv'):
        with (output/name).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=frame_fields)
            writer.writeheader()
            writer.writerows(dict(image_id=i['image_id'],source_frame_0_based=i['source_frame_0_based'],image_sha256=i['sha256'],used_for_fitting='false') for i in images)
    for name in ('other-balls.csv','other-balls-template.csv'):
        with (output/name).open('x',newline='',encoding='utf-8') as stream:csv.DictWriter(stream,fieldnames=object_fields).writeheader()
    (output/'README.md').write_text('''# Pilot bola, putaran A: 4 gambar

Mulai P01. Tonton [konteks P01](P01-context.mp4), target pada detik3.00,
lalu buka [P01 raw](P01-raw.png) di Paint. Ukuran4096x1080, origin kiri atas.
Kirim pusat bola pertandingan x,y bila jelas, serta bola cadangan/unknown.
Kalau tidak terlihat atau identitas ragu, jawab tidak terlihat/tidak jelas.
Jangan menebak dari kaki pemain, menyalin koordinat antarframe, atau membuka
hasil detector/provider. Penilai dapat memakai konteks tetapi jangan membaca
jawaban sebelumnya. Semua foto baru, bukan B01-B08 yang sudah dibandingkan.

Contoh format tanpa angka jawaban:
P01 match: (x,y) / tidak terlihat / tidak jelas
cadangan S01: (x,y); unknown U01: (x,y)

Setelah P01, lanjut [P02](P02-raw.png), [P03](P03-raw.png), [P04](P04-raw.png).
Masing-masing [konteks P02](P02-context.mp4), [P03](P03-context.mp4),
[P04](P04-context.mp4) menempatkan target pada3.00s. Tidak perlu status
possession atau anotasi garis. Tiles-P01 tersedia bila bola sulit dilihat;
jika memakai koordinat tile, kirim nama tile dan koordinat lokal.

Setelah A selesai dan dibekukan, akan dibuat putaran B dengan ID/urutan lain
tanpa koordinat A. Pengulangan satu orang mengukur repeatability saja,
bukan akurasi absolut/kesepakatan antarpenilai. Ingatan/paparan harus dicatat.
Tidak ada model yang dijalankan untuk gambar ini dan tabel masih kosong.

Source: SoccerTrack v1.2, Atom Scott et al., CC BY4.0,
https://huggingface.co/datasets/atomscott/soccertrack-v2 revision
325bd4c57ee8e135ab652f3921d274526ada496b. Satu pertandingan development,
bukan test umum atau bukti Spell F1. Lihat protocol.md. V2/Rr1 tetap kandidat.
''',encoding='utf-8',newline='\n')
    immutable=[i['path'] for i in images]+[i['context_path'] for i in images]+[t['path'] for t in tiles]
    immutable+=['protocol.md','protocol.json','preparation-source.py','initial-provenance.json','README.md','frames-template.csv','other-balls-template.csv']
    manifest=dict(kind='external_ball_repeatability_round_a',created_at_utc=datetime.now(timezone.utc).isoformat(),
        match='117093',dataset_revision=plan['revision'],source_video_sha256=entry['payload_sha256'],source_video_bytes=entry['bytes'],
        images=images,tiles=tiles,files_sha256={n:sha(output/n) for n in immutable},
        official_data_access=False,model_inference=False,annotation_responses=0,reviewer_count=1)
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (output/'manifest.sha256').write_text(sha(output/'manifest.json')+'  manifest.json\n',encoding='utf-8')
    for name,digest in manifest['files_sha256'].items():
        if sha(output/name)!=digest:raise ValueError('Prepared immutable checksum mismatch')
    report=dict(status='round_a_media_verified_annotations_pending',raw_images=4,context_clips=4,
        context_samples_decoded=len(checks),context_frames_each=150,immutable_members=len(immutable),
        checksum_members_verified=len(immutable),manifest_sha256=sha(output/'manifest.json'),
        coordinate_labels_generated=False,model_inference=False,official_data_access=False,
        reviewer_count=1,independent_inter_annotator_validation_ready=False,accepted_scoring_ready=False)
    (output/'preflight.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    (output/'annotation-progress.json').write_text(json.dumps(dict(status='awaiting_round_a_operator',round='A',
        responses=0,frames=4,round_b_prepared=False,second_reviewer_available=False,next_image_id='P01',
        next_task='context at3.00s then raw centre measurements; exposure/completeness need operator confirmation',
        ready_for_uncertainty_estimate=False,ready_for_component_scoring=False),indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to((ROOT/'external_artifacts/v3').resolve())
    prepare(output)

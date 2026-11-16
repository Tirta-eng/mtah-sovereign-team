"""Prepare identical external pixels under new IDs after corrected round A freeze."""
import csv
from datetime import datetime,timezone
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import shutil

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external_artifacts/v3/117093-ball-repeatability-20261115'
FREEZE=BASE/'round-a-freeze-r1'
PACKET=BASE/'round-a'
OUTPUT=BASE/'round-b'
KEY=BASE/'internal-mapping/round-b-key.json'
FREEZE_SHA='9f5ecb29ad5f610ec8743314f62a51e68798b2057da345fc82a452f2e5be1413'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2);stream.write('\n')


def main():
    import cv2
    if OUTPUT.exists() or KEY.exists():raise FileExistsError('Round B/key already exists; do not overwrite')
    if sha(FREEZE/'freeze-manifest.json')!=FREEZE_SHA:raise ValueError('Corrected A freeze mismatch')
    frozen=json.loads((FREEZE/'freeze-manifest.json').read_text(encoding='utf-8'))
    if frozen['kind']!='external_ball_repeatability_round_a_freeze' or frozen['match']!='117093':raise ValueError('Wrong external freeze')
    for base,members in [(FREEZE,{n:e['sha256'] for n,e in frozen['snapshot_files'].items()}),(PACKET,frozen['immutable_source_files_sha256'])]:
        for name,digest in members.items():
            path=(base/name).resolve(strict=True)
            path.relative_to(base.resolve())
            if sha(path)!=digest:raise ValueError('Source mismatch: '+name)
    source=json.loads((FREEZE/'source-packet-manifest.json').read_text(encoding='utf-8'))
    images={i['image_id']:i for i in source['images']}
    if set(images)!={'P01','P02','P03','P04'}:raise ValueError('Unexpected source images')
    order=sorted(images)
    random.Random(11709320261115).shuffle(order)
    if order==sorted(images):order=order[1:]+order[:1]
    # Permutation uses IDs and a fixed seed, never image coordinates or scores.
    OUTPUT.mkdir(exist_ok=False)
    KEY.parent.mkdir(exist_ok=True)
    protocol=dict(kind='external_ball_repeatability_round_b',round='B',
        source_round_a_freeze_sha256=FREEZE_SHA,source_dataset_revision=source['dataset_revision'],
        reviewer_count=1,input='byte-identical raw PNGs; new IDs/order; no A coordinates displayed',
        context='re-encode existing 150-frame context; replace header only; raw measurement PNG unchanged',
        target_context_seconds=3.0,preparation_source_sha256=sha(Path(__file__)),
        permutation='fixed-seed shuffle of source IDs, rotate if unchanged; mapping stored separately',
        memory_exposure_confirmation_pending=True,independently_verified_blinding=False,
        provider_or_model_predictions_read=False,model_inference=False,training=False,official_data_access=False,
        absolute_accuracy_established=False,inter_annotator_agreement_measurable=False,accepted_scoring_ready=False)
    write_json(OUTPUT/'protocol.json',protocol)
    shutil.copyfile(Path(__file__),OUTPUT/'preparation-source.py')
    entries=[]
    mapping=[]
    clip_checks=0
    for index,old_id in enumerate(order,1):
        image=images[old_id]
        image_id=f'U{index:02}'
        png=f'{image_id}-raw.png'
        clip=f'{image_id}-context.mp4'
        shutil.copyfile(PACKET/image['path'],OUTPUT/png)
        if sha(OUTPUT/png)!=image['sha256']:raise ValueError('Repeated raw pixels changed')
        cap=cv2.VideoCapture(str(PACKET/image['context_path']))
        writer=cv2.VideoWriter(str(OUTPUT/clip),cv2.VideoWriter_fourcc(*'mp4v'),25,(4096,1128))
        if not cap.isOpened() or not writer.isOpened():
            cap.release();writer.release()
            raise RuntimeError('Round B context encoder/decoder unavailable')
        count=0
        try:
            while True:
                ok,frame=cap.read()
                if not ok:break
                if frame.shape!=(1128,4096,3):raise ValueError('Context shape changed')
                frame[:48,:,:]=0
                caption=f'{image_id} | round B | target at 3.00 s'
                if count==75:caption+=' | TARGET'
                cv2.putText(frame,caption,(12,32),cv2.FONT_HERSHEY_SIMPLEX,.85,(255,255,255),1)
                writer.write(frame)
                count+=1
        finally:cap.release();writer.release()
        if count!=150:raise ValueError('Unexpected context length')
        check=cv2.VideoCapture(str(OUTPUT/clip))
        try:
            if not check.isOpened() or round(check.get(cv2.CAP_PROP_FRAME_COUNT))!=150:raise ValueError('Invalid B context')
            for sample in (0,75,149):
                check.set(cv2.CAP_PROP_POS_FRAMES,sample)
                ok,frame=check.read()
                if not ok or frame.shape!=(1128,4096,3):raise ValueError('Invalid B sample')
                clip_checks+=1
        finally:check.release()
        entries.append(dict(image_id=image_id,path=png,sha256=image['sha256'],width=4096,height=1080,
            context_path=clip,target_context_seconds=3.0))
        mapping.append(dict(round_b_image_id=image_id,round_a_image_id=old_id,source_frame_0_based=image['source_frame_0_based'],raw_png_sha256=image['sha256']))
    spec=importlib.util.spec_from_file_location('reference_fields',ROOT/'tools/prepare_ball_reference_117093.py')
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fields=[f for f in module.FRAME_FIELDS if f!='source_frame_0_based']
    for name in ('frames.csv','frames-template.csv'):
        with (OUTPUT/name).open('x',newline='',encoding='utf-8') as stream:
            writer=csv.DictWriter(stream,fieldnames=fields)
            writer.writeheader()
            writer.writerows(dict(image_id=i['image_id'],image_sha256=i['sha256'],used_for_fitting='false') for i in entries)
    for name in ('other-balls.csv','other-balls-template.csv'):
        with (OUTPUT/name).open('x',newline='',encoding='utf-8') as stream:csv.DictWriter(stream,fieldnames=module.OBJECT_FIELDS).writeheader()
    write_json(KEY,dict(kind='round_b_to_frozen_a_pixel_identity_key',source_freeze_sha256=FREEZE_SHA,
        permutation_seed=11709320261115,mapping=mapping,no_coordinates_in_key=True,
        restriction='Operator should not inspect key/A answers while measuring B; this is a behavioural instruction, not independently verified access control.'))
    initial=dict(round='B',annotator='operator_user',reviewer_count=1,coordinates_from_a_not_prefilled=True,
        model_or_provider_exposure_during_b='unknown_pending_confirmation',memory_of_a='unknown_pending_confirmation',
        independently_verified_blinding=False,source_round_a_freeze_sha256=FREEZE_SHA)
    write_json(OUTPUT/'initial-provenance.json',initial)
    (OUTPUT/'operator-measurements.jsonl').write_text(json.dumps(initial)+'\n',encoding='utf-8')
    (OUTPUT/'README.md').write_text('''# Putaran B: ukur ulang tanpa membuka jawaban A

Mulai [konteks U01](U01-context.mp4), target pada3.00s, lalu ukur
[U01 raw](U01-raw.png) di Paint. Raw4096x1080 tidak diubah; origin kiri atas.
Kirim pusat bola pertandingan dan cadangan/unknown yang terlihat. Kalau tidak
jelas, sebutkan tidak terlihat/ambigu. Jangan mengambil koordinat dari A atau
memakai prediksi/provider. Jangan membuka internal mapping, catatan atau
koordinat A ketika mengukur. Ingatan sebelumnya harus dilaporkan; perubahan
ID tidak membuktikan ingatan hilang atau blinding independen.

Ukurlah cadangan kembali pada tiap gambar, jangan menyalin angka dari putaran
A. Bila sulit, sebutkan ketidakpastiannya, bukan memaksa konsisten.
Sesudah U01 lanjut [U02](U02-raw.png), [U03](U03-raw.png), [U04](U04-raw.png),
masing-masing memiliki klip konteks dengan nama ID yang sama dan target3.00s.

Semua tabel masih kosong. Sesudah B selesai, piksel identik akan dipasangkan
ke A yang sudah dikoreksi/dikunci untuk menghitung selisih observasi. Pengulangan
satu orang tidak membuktikan akurasi absolut/kesepakatan antarpenilai atau
radius penerimaan detector. Tidak ada model dijalankan untuk pilot ini.
Data eksternal SoccerTrack v1.2/117093, CC BY4.0, Atom Scott et al., revision
325bd4c57ee8e135ab652f3921d274526ada496b. V2/Rr1 tetap kandidat submission.
''',encoding='utf-8',newline='\n')
    immutable=[i['path'] for i in entries]+[i['context_path'] for i in entries]+['protocol.json','preparation-source.py',
        'frames-template.csv','other-balls-template.csv','initial-provenance.json','README.md']
    manifest=dict(kind='external_ball_repeatability_round_b',created_at_utc=datetime.now(timezone.utc).isoformat(),
        match='117093',dataset_revision=source['dataset_revision'],source_video_sha256=source['source_video_sha256'],
        source_round_a_freeze_sha256=FREEZE_SHA,images=entries,files_sha256={n:sha(OUTPUT/n) for n in immutable},
        mapping_key_sha256=sha(KEY),model_inference=False,official_data_access=False,reviewer_count=1)
    write_json(OUTPUT/'manifest.json',manifest)
    (OUTPUT/'manifest.sha256').write_text(sha(OUTPUT/'manifest.json')+'  manifest.json\n',encoding='utf-8')
    for name,digest in manifest['files_sha256'].items():
        if sha(OUTPUT/name)!=digest:raise ValueError('B checksum mismatch')
    write_json(OUTPUT/'verification.json',dict(status='round_b_prepared_from_frozen_a',raw_pngs_byte_identical=4,
        context_samples_decoded=clip_checks,immutable_members_verified=len(immutable),source_freeze_members_verified=len(frozen['snapshot_files']),
        original_immutable_media_verified=23,operator_coordinates_prefilled=False,round_b_responses=0,
        manifest_sha256=sha(OUTPUT/'manifest.json'),model_inference=False,official_data_access=False))
    write_json(OUTPUT/'annotation-progress.json',dict(status='awaiting_round_b_operator',round='B',responses=0,frames=4,
        next_image_id='U01',source_round_a_frozen=True,memory_confirmation_pending=True,session_exposure_confirmation_pending=True,
        ready_for_repeatability_comparison=False,absolute_accuracy_established=False,ready_for_component_scoring=False))
    progress=json.loads((PACKET/'annotation-progress.json').read_text(encoding='utf-8'))
    progress.update(round_b_prepared=True,next_task='Measure round B U01 without consulting A coordinates; report memory/exposure.',round_b_manifest_sha256=sha(OUTPUT/'manifest.json'))
    (PACKET/'annotation-progress.json').write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(status='round_b_ready',raw_pngs_byte_identical=4,context_samples_decoded=clip_checks,
        immutable_members_verified=len(immutable),manifest_sha256=sha(OUTPUT/'manifest.json'),model_inference=False,official_data_access=False),indent=2))


if __name__=='__main__':main()

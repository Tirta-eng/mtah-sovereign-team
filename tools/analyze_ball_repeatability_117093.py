"""Descriptive comparison of frozen external human annotations; no model access."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import shutil
import statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'external_artifacts/v3/117093-ball-repeatability-20261115'
A=BASE/'round-a-freeze-r1'
B=BASE/'round-b-freeze-r1'
A_SHA='9f5ecb29ad5f610ec8743314f62a51e68798b2057da345fc82a452f2e5be1413'
B_SHA='d14c83cdd57a5fb870dbe13c8700b5e8fd8ad3c0044f09ac8cd97652705e5b4a'


def sha(path):
    digest=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):digest.update(chunk)
    return digest.hexdigest()


def read_json(path):return json.loads(path.read_text(encoding='utf-8'))


def write_json(path,value):
    with path.open('x',encoding='utf-8') as stream:json.dump(value,stream,indent=2,allow_nan=False);stream.write('\n')


def contained(base,name):
    path=(base/name).resolve(strict=True);path.relative_to(base.resolve());return path


def verify_freeze(folder,expected,kind):
    if sha(folder/'freeze-manifest.json')!=expected:raise ValueError('Freeze checksum mismatch')
    manifest=read_json(folder/'freeze-manifest.json')
    if manifest['kind']!=kind or manifest['match']!='117093' or manifest['dataset_revision']!='325bd4c57ee8e135ab652f3921d274526ada496b':
        raise ValueError('Unexpected external provenance')
    for name,entry in manifest['snapshot_files'].items():
        path=contained(folder,name)
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['sha256']:raise ValueError('Frozen snapshot mismatch: '+name)
    packet=contained(ROOT,manifest['source_packet']);packet.relative_to((ROOT/'external_artifacts/v3').resolve())
    for name,digest in manifest['immutable_source_files_sha256'].items():
        if sha(contained(packet,name))!=digest:raise ValueError('Raw dependency mismatch: '+name)
    return manifest


def centre(x,y):
    if isinstance(x,bool) or isinstance(y,bool):raise ValueError('Boolean coordinate')
    x,y=float(x),float(y)
    if not math.isfinite(x) or not math.isfinite(y) or not 0<=x<4096 or not 0<=y<1080:
        raise ValueError('Nonfinite/out-of-bounds centre')
    return x,y


def compare(a,b):
    if a is None or b is None:return dict(status='unavailable',reason='missing_annotation',dx=None,dy=None,distance_px=None)
    if a['visibility']!='visible' or b['visibility']!='visible':
        return dict(status='unavailable',reason='visibility_not_clear_in_both',dx=None,dy=None,distance_px=None)
    if a['role']!=b['role'] or a['role'] not in ('match','spare'):
        return dict(status='unavailable',reason='identity_role_not_confirmed_in_both',dx=None,dy=None,distance_px=None)
    ax,ay=centre(*a['centre']);bx,by=centre(*b['centre'])
    return dict(status='paired_observed_positions',reason=None,dx=bx-ax,dy=by-ay,distance_px=math.hypot(bx-ax,by-ay),
        a_x=ax,a_y=ay,b_x=bx,b_y=by)


def verify_pixel_identity(key,a_image,b_image):
    if (key['raw_png_sha256']!=a_image['sha256'] or key['raw_png_sha256']!=b_image['sha256']
        or key['source_frame_0_based']!=a_image['source_frame_0_based']):raise ValueError('Cross-round pixel/frame mismatch')


def describe(rows):
    values=[r['distance_px'] for r in rows if r['distance_px'] is not None]
    return dict(recorded_pairs=len(rows),available_pairs=len(values),unavailable_pairs=len(rows)-len(values),
        minimum_px=min(values) if values else None,median_px=statistics.median(values) if values else None,
        maximum_px=max(values) if values else None,
        interpretation='observed differences only, not absolute error or acceptance radius')


def read_csv(path):
    with path.open(newline='',encoding='utf-8') as stream:return list(csv.DictReader(stream))


def as_match(row):
    if row is None:return None
    return dict(visibility=row['match_ball_visibility'],role='match' if row['match_ball_identity']=='match_confirmed' else 'uncertain',
        centre=[row['match_center_x_raw'],row['match_center_y_raw']])


def as_other(row):
    if row is None:return None
    return dict(visibility=row['visibility'],role=row['role'],centre=[row['center_x_raw'],row['center_y_raw']])


def main(output):
    output=output.resolve();output.relative_to((ROOT/'external_artifacts/v3').resolve())
    if output.exists():raise FileExistsError(output)
    am=verify_freeze(A,A_SHA,'external_ball_repeatability_round_a_freeze')
    bm=verify_freeze(B,B_SHA,'external_ball_repeatability_round_b_freeze')
    if bm['source_round_a_freeze_sha256']!=A_SHA:raise ValueError('Wrong A parent')
    ai={i['image_id']:i for i in read_json(A/'source-packet-manifest.json')['images']}
    bi={i['image_id']:i for i in read_json(B/'source-packet-manifest.json')['images']}
    key=read_json(B/'pixel-identity-key.json')
    if key['source_freeze_sha256']!=A_SHA:raise ValueError('Wrong mapping source')
    mappings=key['mapping']
    if (len(mappings)!=4 or {m['round_a_image_id'] for m in mappings}!=set(ai)
        or {m['round_b_image_id'] for m in mappings}!=set(bi)):raise ValueError('Mapping is not a bijection')
    af={r['image_id']:r for r in read_csv(A/'frames.csv')};bf={r['image_id']:r for r in read_csv(B/'frames.csv')}
    ao={(r['image_id'],r['object_id']):r for r in read_csv(A/'other-balls.csv')}
    bo={(r['image_id'],r['object_id']):r for r in read_csv(B/'other-balls.csv')}
    if set(af)!=set(ai) or set(bf)!=set(bi) or len(ao)!=20 or len(bo)!=20:raise ValueError('Incomplete/duplicate expected annotations')
    for rows,images in [(af,ai),(bf,bi)]:
        for image_id,row in rows.items():
            if row['image_sha256']!=images[image_id]['sha256'] or row['used_for_fitting']!='false':raise ValueError('Annotation image/fitting mismatch')
    basis={(r['image_id'],r['object_id']):r['basis'] for r in read_json(B/'measurement-basis.json')['observations']}
    output.mkdir(parents=True,exist_ok=False)
    shutil.copyfile(Path(__file__),output/'source.py')
    shutil.copyfile(A/'protocol.md',output/'protocol.md')
    protocol=dict(kind='descriptive_single_annotator_ball_repeatability',a_freeze_sha256=A_SHA,b_freeze_sha256=B_SHA,
        source_sha256=sha(Path(__file__)),pairing='exact pixel identity key; then operator local object IDs, never nearest matching',
        sign_convention='B minus corrected frozen A',missingness='retain unavailable/role/visibility discrepancies',
        numeric_and_equality_reports_stratified=True,no_outlier_trimming=True,no_detector_threshold_selected=True,
        no_model_or_official_data_access=True,absolute_accuracy_established=False,accepted_scoring_ready=False)
    write_json(output/'protocol.json',protocol)
    pairs=[]
    for mapping in mappings:
        a_id,b_id=mapping['round_a_image_id'],mapping['round_b_image_id']
        verify_pixel_identity(mapping,ai[a_id],bi[b_id])
        common=dict(round_a_image_id=a_id,round_b_image_id=b_id,source_frame_0_based=mapping['source_frame_0_based'],
            raw_png_sha256=mapping['raw_png_sha256'])
        pairs.append(dict(**common,object_id='match',role='match',measurement_basis=basis[(b_id,'match')],**compare(as_match(af[a_id]),as_match(bf[b_id]))))
        ids=sorted({k[1] for k in ao if k[0]==a_id}|{k[1] for k in bo if k[0]==b_id})
        for object_id in ids:
            pairs.append(dict(**common,object_id=object_id,role='spare',
                measurement_basis=basis.get((b_id,object_id),'missing'),
                **compare(as_other(ao.get((a_id,object_id))),as_other(bo.get((b_id,object_id))))))
    with (output/'paired-differences.jsonl').open('x',encoding='utf-8') as stream:
        for row in pairs:stream.write(json.dumps(row,allow_nan=False)+'\n')
    columns=['round_a_image_id','round_b_image_id','source_frame_0_based','object_id','role','measurement_basis','status','reason','a_x','a_y','b_x','b_y','dx','dy','distance_px']
    with (output/'paired-differences.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=columns,extrasaction='ignore',lineterminator='\n');writer.writeheader();writer.writerows(pairs)
    match=[r for r in pairs if r['role']=='match'];spares=[r for r in pairs if r['role']=='spare']
    explicit=[r for r in spares if r['measurement_basis']=='explicit_numeric_b_response']
    equal=[r for r in spares if r['measurement_basis']=='operator_confirms_remeasurement_equality_without_numeric_reentry']
    largest=max((r['distance_px'] for r in spares if r['distance_px'] is not None),default=None)
    summary=dict(status='single_annotator_pilot_comparison_complete',same_pixel_image_pairs_verified=4,
        all_recorded_position_pairs=len(pairs),match_ball=describe(match),spare_explicit_numeric_reentry=describe(explicit),
        spare_equality_remeasurement_reports=describe(equal),largest_recorded_spare_difference_px=largest,
        largest_spare_difference_pairs=[r for r in spares if r['distance_px']==largest] if largest is not None else [],
        all_unavailable_pairs=[r for r in pairs if r['distance_px'] is None],
        a_wrong_image_attribution_incidents_retained=len(read_json(A/'data-quality-incidents.json')['incidents']),
        provenance=read_json(B/'session-provenance.json')['replies'],reviewer_count=1,
        independently_verified_blinding=False,spare_identity_pairing='operator numbering, not independently adjudicated',
        exhaustive_object_inventory_for_b_certified=False,observed_repeatability_described=True,
        absolute_accuracy_established=False,detector_acceptance_radius=None,confidence_interval_established=False,
        inter_annotator_agreement_measured=False,precision_recall_f1_computed=False,spell_macro_f1_computed=False,
        model_or_threshold_selection=False,production_eligible=False,submission_candidate='V2/R adapter r1',
        official_data_access=False,model_inference=False,training=False,fusion=False,
        limitations=['Four images from one external development match, not independent test samples',
            'Numeric reentry and equality reports have different evidence bases and remain separate',
            'No disagreement removed; no coordinate altered to agree with A or a model',
            'No absolute error bound follows from small intra-annotator differences',
            'A corrected wrong-image attribution is retained as a workflow incident'])
    write_json(output/'summary.json',summary)
    write_json(output/'verification.json',dict(a_snapshot_members_verified=len(am['snapshot_files']),b_snapshot_members_verified=len(bm['snapshot_files']),
        a_immutable_source_members_verified=len(am['immutable_source_files_sha256']),b_immutable_source_members_verified=len(bm['immutable_source_files_sha256']),
        pixel_identity_pairs_verified=4,paired_rows=len(pairs),all_disagreements_retained=True,official_data_access=False))
    files={p.name:dict(bytes=p.stat().st_size,sha256=sha(p)) for p in output.iterdir() if p.is_file()}
    write_json(output/'manifest.json',dict(kind='external_ball_repeatability_analysis',files=files,official_data_access=False))
    (output/'manifest.sha256').write_text(sha(output/'manifest.json')+'  manifest.json\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['status','match_ball','spare_explicit_numeric_reentry','spare_equality_remeasurement_reports','largest_recorded_spare_difference_px','absolute_accuracy_established','official_data_access']},indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True)
    main(parser.parse_args().output)

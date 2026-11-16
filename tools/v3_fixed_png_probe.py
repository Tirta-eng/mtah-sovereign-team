"""Fixed external PNG research probe; no video seek, official inputs or production."""
import argparse
import ast
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import time

ROOT=Path(__file__).resolve().parents[1]
FREEZE=ROOT/'external_artifacts/v3/117093-ball-reference-freeze-20261115'
PACKET=ROOT/'external_artifacts/v3/117093-ball-reference-task-20261115'
METADATA=ROOT/'runtime/v3-input-audit-machine/primary-metadata'
MODEL=ROOT/'runtime/v3-visual-models/rtdetr-r18vd-f2bee744/model.onnx'
FREEZE_SHA='b0bb762d62862d7961870dc5f7ca3f700c169982cf25bc384e6de07d4c91d86e'
VISUAL_SHA='6613388ec25dff638bcf8824585ed43014935887da274bcd0305e855251d96fa'
PROCESSOR_SHA='cd38cd59999e7a95d68e487fbe5132df3d4e5c32a0836add57e6126ba0c4eaf1'
CONFIG_SHA='465bfb0f5f3e5113a54c63f14d8e4a5e23466be0d6965b0592b108007c365177'
MODEL_SHA='11843b02455cc24009aed24d4c40db721b1093be5ccd6bbe7b9c441abb1d0558'
PROTOCOL_SHA='94b8d7acfb60761d02b9c97fcc8342119224c657aaed1e5487c85f64809dd34c'
VIEWS=[('full',(0,0,4096,1080))]+[(f'tile-{r}-{c}',(c*1024,r*540,1024,540)) for r in range(2) for c in range(4)]


def sha(path):
    result=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):
            result.update(chunk)
    return result.hexdigest()


def verify(path,expected):
    if sha(path)!=expected:raise ValueError('Checksum mismatch: '+str(path))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def write_json(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,indent=2,allow_nan=False)
        stream.write('\n')


def contained(base,name):
    path=(base/name).resolve(strict=True)
    path.relative_to(base.resolve())
    return path


def tensor_sha(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


def seal(folder):
    files={p.relative_to(folder).as_posix():dict(bytes=p.stat().st_size,sha256=sha(p))
        for p in sorted(folder.rglob('*')) if p.is_file()}
    write_json(folder/'manifest.json',dict(files=files,official_data_access=False))
    (folder/'manifest.sha256').write_text(sha(folder/'manifest.json')+'  manifest.json\n',encoding='utf-8')


def verify_seal(folder):
    verify(folder/'manifest.json',(folder/'manifest.sha256').read_text().split()[0])
    manifest=read_json(folder/'manifest.json')
    for name,entry in manifest['files'].items():
        path=contained(folder,name)
        if path.stat().st_size!=entry['bytes']:raise ValueError('File size mismatch')
        verify(path,entry['sha256'])
    return manifest


def verify_reference():
    verify(FREEZE/'freeze-manifest.json',FREEZE_SHA)
    frozen=read_json(FREEZE/'freeze-manifest.json')
    if frozen['match']!='117093' or frozen['dataset_revision']!='325bd4c57ee8e135ab652f3921d274526ada496b':
        raise ValueError('Wrong external provenance')
    for name,entry in frozen['snapshot_files'].items():
        verify(contained(FREEZE,name),entry['sha256'])
    for name,digest in frozen['immutable_source_files_sha256'].items():
        verify(contained(PACKET,name),digest)
    return read_json(FREEZE/'source-packet-manifest.json')


def probe_functions():
    source=ROOT/'tools/v3_visual_probe.py'
    verify(source,VISUAL_SHA)
    functions=[n for n in ast.parse(source.read_text(encoding='utf-8')).body
        if isinstance(n,ast.FunctionDef) and n.name in ('preprocess','candidate_boxes')]
    if {n.name for n in functions}!={'preprocess','candidate_boxes'}:raise ValueError('Missing frozen functions')
    namespace={}
    exec(compile(ast.Module(body=functions,type_ignores=[]),str(source),'exec'),namespace)
    return namespace


def eligible(candidate):
    xy=candidate.get('center_raw_px')
    return bool(candidate.get('positive_box_size') and candidate.get('best_class')=='sports ball'
        and isinstance(xy,(tuple,list)) and len(xy)==2
        and all(isinstance(v,(int,float)) and not isinstance(v,bool) and math.isfinite(v) for v in xy))


def in_crop(x,y,crop):
    x0,y0,w,h=crop
    return x0<=x<x0+w and y0<=y<y0+h


def distances_to_reference(candidates,reference,crop):
    x,y=reference['centre_raw_px']
    if not in_crop(x,y,crop):return None
    distances=[dict(candidate_index=i,query=c['query'],distance_raw_px=math.hypot(c['center_raw_px'][0]-x,c['center_raw_px'][1]-y))
        for i,c in enumerate(candidates) if eligible(c)]
    minimum=min((d['distance_raw_px'] for d in distances),default=None)
    return dict(reference=reference,eligible_candidates=len(distances),distances=distances,
        minimum_raw_px=minimum,minimising_queries=[d['query'] for d in distances if d['distance_raw_px']==minimum],
        oracle_proposal_diagnostic_only=True,accepted_match=False)


def check_binding(record,image):
    if (record['image_id']!=image['image_id'] or record['source_frame_0_based']!=image['source_frame_0_based']
            or record['raw_png_sha256']!=image['raw_png_sha256']):raise ValueError('Prediction/image binding mismatch')
    view,=[v for v in image['views'] if v['view']==record['view']]
    if record['tensor_sha256']!=view['tensor_sha256'] or record['crop_raw_xywh']!=view['crop_raw_xywh']:
        raise ValueError('Prediction/tensor binding mismatch')


def reference_points(frame,others):
    points=[]
    if frame['match_ball_visibility']=='visible' and frame['match_ball_identity']=='match_confirmed':
        points.append(dict(object_id='match',role='match',centre_raw_px=[float(frame['match_center_x_raw']),float(frame['match_center_y_raw'])]))
    for other in others:
        if other['image_id']==frame['image_id']:
            points.append(dict(object_id=other['object_id'],role=other['role'],centre_raw_px=[float(other['center_x_raw']),float(other['center_y_raw'])]))
    return points


def prepare(output):
    import importlib.metadata
    import numpy as np
    from PIL import Image
    if output.exists():raise FileExistsError(output)
    source=verify_reference()
    verify(METADATA/'preprocessor_config.json',PROCESSOR_SHA)
    verify(METADATA/'rtdetr-config.json',CONFIG_SHA)
    verify(ROOT/'docs/protokol_evaluasi_bola_117093.md',PROTOCOL_SHA)
    if MODEL.stat().st_size!=82572357:raise ValueError('Wrong model size')
    verify(MODEL,MODEL_SHA)
    functions=probe_functions()
    output.mkdir(parents=True,exist_ok=False)
    prep=output/'preparation'
    (prep/'inputs').mkdir(parents=True)
    for path in [Path(__file__),ROOT/'tools/v3_visual_probe.py',ROOT/'tools/v3_ball_association_probe.py',
                 ROOT/'docs/protokol_evaluasi_bola_117093.md',METADATA/'preprocessor_config.json',METADATA/'rtdetr-config.json']:
        shutil.copyfile(path,prep/path.name)
    protocol=dict(kind='fixed_external_png_ball_proposal_probe',schema=1,
        frozen_reference_manifest_sha256=FREEZE_SHA,source_frames=list(range(4375,4383)),
        input='Exact frozen raw PNGs; no video decoding, timestamp seek or offset correction',
        visual_source_sha256=VISUAL_SHA,processor_sha256=PROCESSOR_SHA,model_config_sha256=CONFIG_SHA,
        model_sha256=MODEL_SHA,tool_sha256=sha(Path(__file__)),association_source_sha256=sha(ROOT/'tools/v3_ball_association_probe.py'),
        comparison_protocol_sha256=PROTOCOL_SHA,views=VIEWS,top_k=10,
        provider='CPUExecutionProvider',threads=1,warmup_zero_tensors=1,inference_wall_budget_seconds=240,
        candidate_acceptance=False,confidence_threshold=None,distance_threshold=None,
        association='unchanged reciprocal unique nearest per view; exact ties abstain; no frame gap bridging',
        diagnostics='per-reference per-view eligible query distances only; minimum is oracle, not recall',
        uncertainty_established=False,precision_recall_f1_allowed=False,possession_scoring_allowed=False,
        training=False,fusion=False,official_data_access=False,
        packages={n:importlib.metadata.version(n) for n in ('numpy','Pillow')})
    write_json(prep/'protocol.json',protocol)  # before inference or reading model output
    images=[]
    processor=read_json(METADATA/'preprocessor_config.json')
    for item in source['images']:
        raw=contained(PACKET,item['path'])
        verify(raw,item['sha256'])
        with Image.open(raw) as decoded:
            image=decoded.convert('RGB')
        if image.size!=(4096,1080):raise ValueError('Wrong raw PNG shape')
        samples,views={},[]
        for name,(x,y,w,h) in VIEWS:
            tensor=functions['preprocess'](image.crop((x,y,x+w,y+h)),processor)
            if tensor.shape!=(1,3,640,640) or tensor.dtype!=np.float32 or not tensor.flags.c_contiguous:
                raise ValueError('Invalid tensor contract')
            samples[name]=tensor
            views.append(dict(view=name,crop_raw_xywh=[x,y,w,h],tensor_sha256=tensor_sha(tensor)))
        archive=prep/f"inputs/{item['image_id']}.npz"
        np.savez_compressed(archive,**samples)
        images.append(dict(image_id=item['image_id'],source_frame_0_based=item['source_frame_0_based'],
            raw_png_sha256=item['sha256'],raw_png_path=item['path'],views=views,
            tensor_archive_path=archive.relative_to(prep).as_posix(),tensor_archive_sha256=sha(archive)))
    write_json(prep/'inputs.json',dict(images=images,protocol_sha256=sha(prep/'protocol.json'),
        frozen_reference_manifest_sha256=FREEZE_SHA,model_sha256=MODEL_SHA,official_data_access=False))
    seal(prep)
    print(json.dumps(dict(status='fixed_png_inputs_prepared',images=len(images),views=sum(len(i['views']) for i in images),
        manifest_sha256=sha(prep/'manifest.json'),official_data_access=False)),flush=True)


def infer(output):
    import importlib.metadata
    import numpy as np
    import onnxruntime as ort
    prep=output/'preparation'
    verify_seal(prep)
    inputs=read_json(prep/'inputs.json')
    protocol=read_json(prep/'protocol.json')
    verify(Path(__file__),protocol['tool_sha256'])
    if protocol['frozen_reference_manifest_sha256']!=FREEZE_SHA or inputs['model_sha256']!=MODEL_SHA:
        raise ValueError('Wrong prepared external inputs')
    verify_reference()
    verify(MODEL,MODEL_SHA)
    verify(prep/'rtdetr-config.json',CONFIG_SHA)
    if ort.__version__!='1.23.2':raise ValueError('Unexpected ORT version')
    config=read_json(prep/'rtdetr-config.json')
    ball_id,=[int(k) for k,v in config['id2label'].items() if v=='sports ball']
    association=ROOT/'tools/v3_ball_association_probe.py'
    verify(association,protocol['association_source_sha256'])
    spec=importlib.util.spec_from_file_location('frozen_association',association)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    tracker=module.TrackletProposals()
    functions=probe_functions()
    destination=output/'inference'
    destination.mkdir(exist_ok=False)
    (destination/'raw-outputs').mkdir()
    count=0
    timings=[]
    tick=time.monotonic()
    try:
        options=ort.SessionOptions()
        options.intra_op_num_threads=options.inter_op_num_threads=1
        options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
        session=ort.InferenceSession(str(MODEL),sess_options=options,providers=['CPUExecutionProvider'])
        if session.get_providers()!=['CPUExecutionProvider']:raise RuntimeError('Unexpected provider')
        if len(session.get_inputs())!=1 or session.get_inputs()[0].name!='pixel_values':raise ValueError('Unexpected model input')
        names=[o.name for o in session.get_outputs()]
        if set(names)!={'logits','pred_boxes'}:raise ValueError('Unexpected model output')
        load_seconds=time.monotonic()-tick
        tick=time.monotonic()
        session.run(None,{'pixel_values':np.zeros((1,3,640,640),dtype=np.float32)})
        warmup_seconds=time.monotonic()-tick
        start=time.monotonic()
        with (destination/'observations.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
            for image in inputs['images']:
                archive=contained(prep,image['tensor_archive_path'])
                verify(archive,image['tensor_archive_sha256'])
                raw_outputs={}
                with np.load(archive,allow_pickle=False) as tensors:
                    if set(tensors.files)!={v['view'] for v in image['views']}:raise ValueError('Unexpected view set')
                    for view in image['views']:
                        if time.monotonic()-start>240:raise TimeoutError('Research budget exceeded; no partial scoring')
                        tensor=tensors[view['view']]
                        if tensor.shape!=(1,3,640,640) or tensor.dtype!=np.float32 or tensor_sha(tensor)!=view['tensor_sha256']:
                            raise ValueError('Tensor bytes changed')
                        tick=time.monotonic()
                        values=dict(zip(names,session.run(None,{'pixel_values':tensor})))
                        elapsed=time.monotonic()-tick
                        candidates=functions['candidate_boxes'](values['logits'],values['pred_boxes'],ball_id,
                            view['crop_raw_xywh'],config['id2label'],top_k=10)
                        record=dict(image_id=image['image_id'],source_frame_0_based=image['source_frame_0_based'],
                            raw_png_sha256=image['raw_png_sha256'],tensor_sha256=view['tensor_sha256'],
                            view=view['view'],crop_raw_xywh=view['crop_raw_xywh'],inference_seconds=elapsed,candidates=candidates,
                            association=tracker.update('FIRST_HALF',image['source_frame_0_based'],view['view'],candidates))
                        check_binding(record,image)
                        for name,array in values.items():raw_outputs[view['view']+'__'+name]=array
                        stream.write(json.dumps(record,allow_nan=False)+'\n')
                        stream.flush()
                        timings.append(elapsed)
                        count+=1
                np.savez_compressed(destination/f"raw-outputs/{image['image_id']}.npz",**raw_outputs)
                print(json.dumps(dict(image_id=image['image_id'],views_completed=count)),flush=True)
        if count!=72:raise ValueError('Incomplete view sequence')
        summary=dict(status='fixed_png_research_inference_complete',images=8,views=72,providers=session.get_providers(),
            model_load_seconds=load_seconds,warmup_seconds=warmup_seconds,inference_seconds_sum=sum(timings),
            inference_seconds_median=float(np.median(timings)),wall_seconds=time.monotonic()-start,
            preparation_manifest_sha256=sha(prep/'manifest.json'),packages={n:importlib.metadata.version(n) for n in ('numpy','onnxruntime-gpu')},
            candidate_matches_accepted=False,quality_f1_computed=False,production_eligible=False,
            training=False,fusion=False,official_data_access=False)
        write_json(destination/'summary.json',summary)
    except BaseException as error:
        write_json(destination/'failure.json',dict(error_type=type(error).__name__,message=str(error),views_completed=count,
            official_data_access=False,no_partial_scoring=True))
        raise
    finally:
        seal(destination)
    print(json.dumps(summary,indent=2),flush=True)


def analyze(output):
    prep=output/'preparation'
    inference=output/'inference'
    verify_seal(prep)
    verify_seal(inference)
    verify_reference()
    summary=read_json(inference/'summary.json')
    if summary['status']!='fixed_png_research_inference_complete' or summary['views']!=72:
        raise ValueError('No complete inference for analysis')
    if summary['preparation_manifest_sha256']!=sha(prep/'manifest.json'):raise ValueError('Wrong preparation binding')
    images={r['image_id']:r for r in read_json(prep/'inputs.json')['images']}
    with (FREEZE/'frames.csv').open(newline='',encoding='utf-8') as stream:
        references={r['image_id']:r for r in csv.DictReader(stream)}
    with (FREEZE/'other-balls.csv').open(newline='',encoding='utf-8') as stream:
        others=list(csv.DictReader(stream))
    records=[json.loads(line) for line in (inference/'observations.jsonl').read_text(encoding='utf-8').splitlines()]
    if len(records)!=72 or len({(r['image_id'],r['view']) for r in records})!=72:
        raise ValueError('Missing/duplicate image/view records')
    destination=output/'diagnostics'
    destination.mkdir(exist_ok=False)
    diagnostics=[]
    lengths={}
    for record in records:
        check_binding(record,images[record['image_id']])
        ref=references[record['image_id']]
        refs=reference_points(ref,others)
        for reference in refs:
            comparison=distances_to_reference(record['candidates'],reference,record['crop_raw_xywh'])
            if comparison is not None:
                diagnostics.append(dict(image_id=record['image_id'],source_frame_0_based=record['source_frame_0_based'],
                    view=record['view'],**comparison))
        for proposal in record['association']['proposals']:
            if proposal['accepted_as_match_ball'] is not False:raise ValueError('Accepted identity is forbidden')
            key=(record['view'],proposal['tracklet_id'])
            lengths[key]=lengths.get(key,0)+1
    with (destination/'reference-query-distances.jsonl').open('x',encoding='utf-8',newline='\n') as stream:
        for diagnostic in diagnostics:stream.write(json.dumps(diagnostic,allow_nan=False)+'\n')
    rows=[dict(image_id=r['image_id'],view=r['view'],object_id=r['reference']['object_id'],role=r['reference']['role'],
        eligible_candidates=r['eligible_candidates'],minimum_raw_px=r['minimum_raw_px'],
        minimising_queries='|'.join(map(str,r['minimising_queries']))) for r in diagnostics]
    with (destination/'oracle-proposal-distances.csv').open('x',encoding='utf-8',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]),lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    result=dict(status='descriptive_proposal_diagnostics_only',images=8,views=72,
        raw_png_and_tensor_bindings_verified=72,reference_view_pairs=len(diagnostics),
        oracle_distance_minima_with_candidates=sum(r['minimum_raw_px'] is not None for r in diagnostics),
        oracle_distance_minima_missing=sum(r['minimum_raw_px'] is None for r in diagnostics),
        proposed_tracklets=len(lengths),tracklets_observed_all_eight_frames=sum(n==8 for n in lengths.values()),
        accepted_match_ball_identities=0,matching_radius=None,uncertainty_established=False,
        generic_detector_spare_is_automatically_false_positive=False,not_visible_is_detector_negative=False,
        precision_recall_f1_computed=False,spell_macro_f1_computed=False,training=False,fusion=False,
        official_data_access=False,production_eligible=False,
        limitations=['Oracle minimum distance is not model-selected ball identity or recall',
            'No cross-view pooling/deduplication or threshold selection',
            'Eight correlated development frames; centres are operator self-report, uncertainty unmeasured',
            'Do not reuse these already exposed images as a new blind annotation test'])
    write_json(destination/'summary.json',result)
    seal(destination)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--phase',choices=['prepare','infer','analyze'],required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    output=args.output.resolve()
    output.relative_to((ROOT/'external_artifacts/v3').resolve())
    {'prepare':prepare,'infer':infer,'analyze':analyze}[args.phase](output)

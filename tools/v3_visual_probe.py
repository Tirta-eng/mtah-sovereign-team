"""External-only visual/geometric feasibility probe; not a possession pipeline.

No import-time dataset/model access. CLI accepts an output under v3 only and
opens explicit pinned SoccerTrack/model paths. Never discovers official inputs.
Retains raw sports-ball query scores; no tuned acceptance threshold, tracking,
fusion, H/A/D output, training, or data-dependent model selection is performed.
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def preprocess(image, config):
    """RGB PIL crop -> float32 (1,3,640,640), publisher bilinear resize/1:255.

    No padding or mean/std normalization for this pinned RT-DETR processor.
    Aspect ratio is stretched exactly as declared; box decoding reverses this
    with crop width/height. Unknown processor contracts fail, never guess.
    """
    import numpy as np
    from PIL import Image
    if (config['size'] != {'height': 640, 'width': 640}
            or config['do_normalize'] or config['do_pad']
            or config['resample'] != 2 or not config['do_rescale']
            or not config['do_resize'] or config['rescale_factor'] != 1 / 255):
        raise ValueError('Unsupported preprocessing contract')
    data = np.asarray(image.convert('RGB').resize((640, 640), Image.Resampling.BILINEAR), dtype=np.float32)
    return np.ascontiguousarray((data * np.float32(1 / 255)).transpose(2, 0, 1)[None])


def candidate_boxes(logits, boxes, ball_id, crop, names, top_k=10):
    """Keep top sports-ball query diagnostics, not thresholded detections.

    logits=(1,Q,C), boxes=(1,Q,4) normalized cx,cy,w,h. Return raw-image pixel
    boxes using crop=(x,y,w,h). Scores are model sigmoid outputs, uncalibrated.
    Retain outside coordinates and the query's best class to expose confusion.
    No NMS, confidence acceptance, team identity or ball-presence decision.
    """
    import numpy as np
    logits, boxes = np.asarray(logits), np.asarray(boxes)
    if (logits.ndim != 3 or logits.shape[0] != 1 or boxes.shape != (*logits.shape[:2], 4)
            or not np.isfinite(logits).all() or not np.isfinite(boxes).all()
            or not 0 <= ball_id < logits.shape[-1]):
        raise ValueError('Invalid detector output contract')
    scores = np.exp(-np.logaddexp(0., -logits[0]))
    order = np.argsort(-scores[:, ball_id], kind='stable')[:top_k]
    x0, y0, width, height = crop
    result = []
    for index in order:
        cx, cy, w, h = map(float, boxes[0, index])
        best = int(np.argmax(scores[index]))
        result.append(dict(query=int(index), score=float(scores[index, ball_id]),
            raw_ball_logit=float(logits[0, index, ball_id]),best_class=names[str(best)],
            best_class_score=float(scores[index, best]),
            center_raw_px=[x0+cx*width,y0+cy*height],
            box_raw_xyxy=[x0+(cx-w/2)*width,y0+(cy-h/2)*height,
                          x0+(cx+w/2)*width,y0+(cy+h/2)*height],
            positive_box_size=bool(w>0 and h>0),accepted_as_ball=False))
    return result


def ground_plane_proxy(raw_pixel, k, d, knew, homography):
    """Raw fisheye pixel -> corner-origin metres, conditional ground-plane proxy.

    Invert the fixed 4-coefficient fisheye polynomial with Newton iterations,
    then H(world-metres -> undistorted-pixels). Numerical convergence tolerance
    is a solver check, not a learned geometric confidence threshold. No clipping
    to pitch, no covariance claim. None means missing/invalid transform. Even a
    valid result is not a verified ball location: ball height is unknown.
    """
    import numpy as np
    point = np.asarray(raw_pixel,dtype=float)
    k,d,knew,homography = [np.asarray(x,dtype=float) for x in (k,d,knew,homography)]
    if (point.shape != (2,) or k.shape != (3,3) or knew.shape != (3,3)
            or homography.shape != (3,3) or d.size != 4
            or any(not np.isfinite(v).all() for v in (point,k,d,knew,homography))):
        return None
    try:
        ray=np.linalg.solve(k,np.r_[point,1.])
        if abs(ray[2])<1e-12:return None
        xy=ray[:2]/ray[2];rd=float(np.linalg.norm(xy));theta=rd;coeff=d.ravel()
        for _ in range(40):
            t2=theta*theta
            f=theta*(1+coeff[0]*t2+coeff[1]*t2**2+coeff[2]*t2**3+coeff[3]*t2**4)-rd
            derivative=1+3*coeff[0]*t2+5*coeff[1]*t2**2+7*coeff[2]*t2**3+9*coeff[3]*t2**4
            if abs(f)<1e-10:break
            if abs(derivative)<1e-12:return None
            theta-=f/derivative
        else:return None
        if not 0<=theta<np.pi/2:return None
        undistorted=xy if rd<1e-12 else xy*(np.tan(theta)/rd)
        pixel=knew@np.r_[undistorted,1.]
        world=np.linalg.solve(homography,pixel)
        if abs(world[2])<1e-12:return None
        value=world[:2]/world[2]
        return value.tolist() if np.isfinite(value).all() else None
    except (np.linalg.LinAlgError,FloatingPointError,OverflowError):
        return None


def decode(video, frame):
    """Decode one 0-based CFR frame through accurate input seek, no disk cache."""
    from PIL import Image
    command=['ffmpeg','-nostdin','-v','error','-threads','1','-ss',str(frame/25),
             '-i',str(video),'-map','0:v:0','-an','-sn','-frames:v','1',
             '-filter_threads','1','-f','image2pipe','-vcodec','png','pipe:1']
    image=Image.open(io.BytesIO(subprocess.check_output(command))).convert('RGB')
    if image.size!=(4096,1080):raise ValueError('Source video shape differs')
    return image


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--max-seconds',type=int,default=180,
                        help='Research wall budget after warmup, not production sampling')
    args=parser.parse_args()
    output=args.output.resolve();output.relative_to((ROOT/'external_artifacts/v3').resolve())
    if output.exists():raise FileExistsError(output)
    import numpy as np
    import onnxruntime as ort
    from PIL import Image
    modeldir=ROOT/'runtime/v3-visual-models/rtdetr-r18vd-f2bee744'
    manifest=json.loads((modeldir/'manifest.json').read_text())
    for name,digest in manifest['files_sha256'].items():
        path=(modeldir/name).resolve();path.relative_to(modeldir.resolve())
        if sha(path)!=digest:raise ValueError('Model/config checksum mismatch')
    config=json.loads((modeldir/'config.json').read_text());processor=json.loads((modeldir/'preprocessor_config.json').read_text())
    ball_id=next(int(k) for k,v in config['id2label'].items() if v=='sports ball')
    plan=json.loads((ROOT/'docs/soccertrack_v1_2_117093_download.json').read_text())
    raw=ROOT/'Soccertrack V2'
    needed=[e for e in plan['files'] if e['path'].startswith('videos/') or e['path'].endswith(('_camera_intrinsics.npz','_homography.npy'))]
    for entry in needed:
        path=raw/entry['path']
        if path.stat().st_size!=entry['bytes'] or sha(path)!=entry['payload_sha256']:
            raise ValueError('External input checksum mismatch')
    video=raw/next(e['path'] for e in needed if e['path'].startswith('videos/'))
    with np.load(raw/'raw/117093/117093_camera_intrinsics.npz',allow_pickle=False) as archive:
        k,d,knew=[archive[name] for name in ('K','D','Knew')]
    homography=np.load(raw/'raw/117093/117093_homography.npy',allow_pickle=False)
    task=json.loads((ROOT/'external_artifacts/v3/117093-interval-review-audio-20261115/manifest.json').read_text())
    # Fixed clip-local times, independent of annotations, statuses and scores.
    frames=[(c['clip_id'],c['source_start_frame']+250) for c in task['clips']]
    frames += [(c['clip_id'],c['source_start_frame']+500) for c in task['clips']]
    views=[('full',(0,0,4096,1080))]+[(f'tile-{row}-{col}',(col*1024,row*540,1024,540)) for row in range(2) for col in range(4)]
    protocol=dict(model_manifest_sha256=sha(modeldir/'manifest.json'),detector='RT-DETR R18 FP32 ONNX CPU',
        frames=frames,views=views,top_k=10,confidence_threshold=None,threshold_calibrated=False,
        candidate_box_acceptance=False,tracking_or_association=False,classifier_or_fusion=False,
        sampling='feasibility only; not possession timeline or frame-rate design',
        budget_seconds=args.max_seconds,threads=1,geometry='fixed publisher H + fisheye inverse; unverified ground-plane proxy only',
        geometry_accuracy_certified=False,ground_truth_read=False,official_data_access=False,trained=False)
    output.mkdir(parents=True);(output/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    options=ort.SessionOptions();options.intra_op_num_threads=1;options.inter_op_num_threads=1
    options.execution_mode=ort.ExecutionMode.ORT_SEQUENTIAL
    start=time.monotonic();session=ort.InferenceSession(str(modeldir/'model.onnx'),sess_options=options,providers=['CPUExecutionProvider'])
    load_seconds=time.monotonic()-start
    inputs=[dict(name=v.name,shape=v.shape,type=v.type) for v in session.get_inputs()]
    outputs=[dict(name=v.name,shape=v.shape,type=v.type) for v in session.get_outputs()]
    if len(inputs)!=1 or inputs[0]['name']!='pixel_values':raise ValueError('Unexpected ONNX inputs')
    if {v['name'] for v in outputs}!={'logits','pred_boxes'}:raise ValueError('Unexpected ONNX outputs')
    tensor=preprocess(Image.new('RGB',(640,640)),processor)
    start=time.monotonic();session.run(None,{'pixel_values':tensor});warm_seconds=time.monotonic()-start
    deadline=time.monotonic()+args.max_seconds;timings=[];records=[];decode_times=[];truncated=False
    with (output/'candidates.jsonl').open('w',encoding='utf-8') as stream:
        for clip_id,frame in frames:
            if time.monotonic()>=deadline:truncated=True;break
            start=time.monotonic();image=decode(video,frame);decode_times.append(time.monotonic()-start)
            for view,(x,y,w,h) in views:
                if time.monotonic()>=deadline:truncated=True;break
                tensor=preprocess(image.crop((x,y,x+w,y+h)),processor)
                start=time.monotonic();values=session.run(None,{'pixel_values':tensor});elapsed=time.monotonic()-start
                actual=dict(zip([o['name'] for o in outputs],values))
                candidates=candidate_boxes(actual['logits'],actual['pred_boxes'],ball_id,(x,y,w,h),config['id2label'])
                for c in candidates:
                    c['conditional_ground_corner_m']=ground_plane_proxy(c['center_raw_px'],k,d,knew,homography)
                    c['ball_height_known']=False;c['geometry_validity']='not certified; planar assumption only'
                record=dict(clip_id=clip_id,source_frame_0_based=frame,view=view,crop_raw_xywh=[x,y,w,h],inference_seconds=elapsed,candidates=candidates)
                stream.write(json.dumps(record,allow_nan=False)+'\n');stream.flush()
                records.append(record);timings.append(elapsed)
                print(json.dumps(dict(frame=frame,view=view,seconds=round(elapsed,3),top_ball_score=round(candidates[0]['score'],5),top_query_best_class=candidates[0]['best_class'])),flush=True)
    summary=dict(status='external_observation_probe_only',model_load_seconds=load_seconds,warmup_seconds=warm_seconds,
        views_run=len(records),unique_frames=len({r['source_frame_0_based'] for r in records}),budget_exhausted=truncated,
        inference_seconds=dict(median=float(np.median(timings)),p95=float(np.quantile(timings,.95)),sum=float(sum(timings))) if timings else None,
        decode_seconds_sum=sum(decode_times),peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,
        score_quantiles=np.quantile([r['candidates'][0]['score'] for r in records],[0,.5,1]).tolist() if records else [],
        input_contract=inputs,output_contract=outputs,packages={n:__import__('importlib.metadata',fromlist=['version']).version(n) for n in ('numpy','Pillow','onnxruntime')},
        quality_metrics_computable=False,independent_ball_labels_available=False,recall_or_precision_claim=None,
        production_eligible=False,training_performed=False,official_data_access=False,
        limitations=['COCO generic detector, not football-calibrated','scores are not calibrated confidence',
        'no ball identity/association or out/restart inference','planar geometry cannot resolve airborne ball',
        'no component GT or end-to-end F1; probe timing is not full-pipeline throughput'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    (output/'manifest.json').write_text(json.dumps(dict(source_sha256=sha(Path(__file__)),files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()},official_data_access=False),indent=2)+'\n')


if __name__=='__main__':
    main()

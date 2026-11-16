"""Bounded external-only tracklet proposals, not match-ball identification.

Eight contiguous 25 Hz frames from the already inspected SoccerTrack R01 case.
No target, ATD, model fitting, production threshold, fusion or possession output.
Import is inert. All payloads are explicit, checksum-bound external paths.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[1]


class TrackletProposals:
    """Reciprocal nearest raw-pixel proposals, independently for each crop.

    update(period, frame, view, candidates) takes an integer source frame (0-based)
    and candidate dictionaries from the frozen visual probe. Only finite centres,
    positive boxes and best-class sports ball enter this research comparison.
    Missing/empty observations, non-contiguous frames and period changes reset
    links. No cross-view deduplication or permanent ball identity is claimed.
    Exact distance ties abstain. No physical displacement gate was calibrated:
    a link may be arbitrarily wrong and ALWAYS remains an unaccepted proposal.
    Pixel displacement is diagnostic, never a live/dead or possession feature.
    """

    def __init__(self):
        self.previous = {}
        self.counter = 0

    def update(self, period, frame, view, candidates):
        if not isinstance(frame, int) or isinstance(frame, bool) or frame < 0:
            raise ValueError('Expected nonnegative integer source frame')
        previous = self.previous.get(view)
        if previous and previous['period'] == period and frame <= previous['frame']:
            raise ValueError('Duplicate or unordered view frame')
        contiguous = bool(previous and previous['period'] == period and frame == previous['frame'] + 1)
        current = []
        for index, candidate in enumerate(candidates):
            xy = candidate.get('center_raw_px')
            if (candidate.get('positive_box_size') and candidate.get('best_class') == 'sports ball'
                    and isinstance(xy, (list, tuple)) and len(xy) == 2
                    and all(isinstance(x, (int, float)) and math.isfinite(x) for x in xy)):
                current.append(dict(candidate_index=index, xy=list(xy)))
        old = previous['points'] if contiguous else []
        distances = [[sum((a-b)**2 for a, b in zip(p['xy'], q['xy'])) for q in current] for p in old]

        def unique_minimum(values):
            if not values:
                return None
            smallest = min(values)
            indices = [i for i, v in enumerate(values) if v == smallest]
            return indices[0] if len(indices) == 1 else None

        forward = [unique_minimum(row) for row in distances]
        backward = [unique_minimum([row[j] for row in distances]) for j in range(len(current))]
        for j, point in enumerate(current):
            i = backward[j]
            linked = i is not None and forward[i] == j
            if linked:
                point['tracklet_id'] = old[i]['tracklet_id']
                step = math.sqrt(distances[i][j])
            else:
                self.counter += 1
                point['tracklet_id'] = self.counter
                step = None
            point.update(previous_frame=frame-1 if linked else None,
                         step_raw_px=step, accepted_as_match_ball=False)
        self.previous[view] = dict(period=period, frame=frame, points=current)
        return dict(period=period, source_frame_0_based=frame, view=view,
                    contiguous_predecessor=contiguous, proposals=current)


def load_visual():
    """Explicit local source import at execution time; never at module import."""
    spec = importlib.util.spec_from_file_location('v3_visual_probe', ROOT/'tools/v3_visual_probe.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT/'external_artifacts/v3').resolve())
    if output.exists():
        raise FileExistsError(output)
    import numpy as np
    import onnxruntime as ort
    probe = load_visual()
    modeldir = ROOT/'runtime/v3-visual-models/rtdetr-r18vd-f2bee744'
    model_manifest = json.loads((modeldir/'manifest.json').read_text())
    for name, digest in model_manifest['files_sha256'].items():
        path = (modeldir/name).resolve()
        path.relative_to(modeldir.resolve())
        if probe.sha(path) != digest:
            raise ValueError('Model/config checksum mismatch')
    config = json.loads((modeldir/'config.json').read_text())
    processor = json.loads((modeldir/'preprocessor_config.json').read_text())
    plan_path = ROOT/'docs/soccertrack_v1_2_117093_download.json'
    plan = json.loads(plan_path.read_text())
    entry, = [e for e in plan['files'] if e['path'] == 'videos/117093/117093_panorama_1st_half.mp4']
    video = ROOT/'Soccertrack V2'/entry['path']
    if video.stat().st_size != entry['bytes'] or probe.sha(video) != entry['payload_sha256']:
        raise ValueError('External video checksum mismatch')
    frames = list(range(4375, 4383))
    views = [('full', (0, 0, 4096, 1080))] + [
        (f'tile-{r}-{c}', (c*1024, r*540, 1024, 540)) for r in range(2) for c in range(4)]
    protocol = dict(source_frames_0_based=frames, period='FIRST_HALF', fps=25,
        duration_first_to_last_seconds=7/25, views=views, top_k_per_view=10,
        selection='Purposive known R01 spare-ball case, not a held-out test',
        association='per-view reciprocal unique nearest pixel centre; ties abstain',
        eligibility='positive box, finite centre, best class sports ball; no score threshold',
        distance_gate=None, physical_plausibility_validated=False,
        no_confidence_or_geometry_gate_fitted=True, model_manifest_sha256=probe.sha(modeldir/'manifest.json'),
        visual_source_sha256=probe.sha(ROOT/'tools/v3_visual_probe.py'),
        source_video_sha256=entry['payload_sha256'], download_plan_sha256=probe.sha(plan_path),
        cpu_threads=1, max_views=72, wall_budget_seconds=210,
        no_gap_bridging=True, accepted_match_ball=False, official_data_access=False,
        training=False, production_eligible=False, independent_ground_truth=False,
        gpu_cache_views=['full', 'tile-0-1', 'tile-1-3'])
    output.mkdir(parents=True)
    (output/'protocol.json').write_text(json.dumps(protocol, indent=2)+'\n')
    options = ort.SessionOptions()
    options.intra_op_num_threads = options.inter_op_num_threads = 1
    options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    session = ort.InferenceSession(str(modeldir/'model.onnx'), options, providers=['CPUExecutionProvider'])
    names = [o.name for o in session.get_outputs()]
    if {v.name for v in session.get_inputs()} != {'pixel_values'} or set(names) != {'logits', 'pred_boxes'}:
        raise ValueError('Unexpected ONNX interface')
    ball_id = next(int(i) for i, label in config['id2label'].items() if label == 'sports ball')
    tracker = TrackletProposals()
    samples, records, timings, decode_times = {}, [], [], []
    started = time.monotonic()
    complete = True
    with (output/'observations.jsonl').open('w') as stream:
        for frame in frames:
            if time.monotonic()-started >= protocol['wall_budget_seconds']:
                complete = False
                break
            tick = time.monotonic()
            image = probe.decode(video, frame)
            decode_times.append(time.monotonic()-tick)
            for view, (x, y, w, h) in views:
                if time.monotonic()-started >= protocol['wall_budget_seconds']:
                    complete = False
                    break
                tensor = probe.preprocess(image.crop((x, y, x+w, y+h)), processor)
                if frame == frames[0] and view in protocol['gpu_cache_views']:
                    samples[view] = tensor.copy()
                tick = time.monotonic()
                values = dict(zip(names, session.run(None, {'pixel_values': tensor})))
                elapsed = time.monotonic()-tick
                candidates = probe.candidate_boxes(values['logits'], values['pred_boxes'], ball_id,
                                                    (x, y, w, h), config['id2label'])
                proposals = tracker.update('FIRST_HALF', frame, view, candidates)
                record = dict(candidates=candidates, association=proposals, inference_seconds=elapsed)
                stream.write(json.dumps(record, allow_nan=False)+'\n')
                stream.flush()
                records.append(record)
                timings.append(elapsed)
            print(json.dumps(dict(frame=frame, views_run=len(records), elapsed_seconds=round(time.monotonic()-started, 2))), flush=True)
    cache = output/'gpu-inputs'
    cache.mkdir()
    np.savez_compressed(cache/'inputs.npz', **samples)
    cached = dict(schema=1, source='SoccerTrack v1.2 external 117093; CC BY 4.0; Atom Scott et al.',
        source_url='https://huggingface.co/datasets/atomscott/soccertrack-v2',
        dataset_revision='325bd4c57ee8e135ab652f3921d274526ada496b',
        source_frame_0_based=frames[0], video_sha256=entry['payload_sha256'],
        preprocessing_source_sha256=probe.sha(ROOT/'tools/v3_visual_probe.py'),
        processor_sha256=probe.sha(modeldir/'preprocessor_config.json'),
        model_sha256=model_manifest['sha256'], shape=[1, 3, 640, 640], dtype='float32',
        views=list(samples), inputs_sha256=probe.sha(cache/'inputs.npz'),
        official_data=False, intended_use='fixed preprocessed external tensors for hardware timing, no quality labels')
    (cache/'manifest.json').write_text(json.dumps(cached, indent=2)+'\n')
    lengths, steps = {}, []
    for record in records:
        for p in record['association']['proposals']:
            key = str(p['tracklet_id'])
            lengths[key] = lengths.get(key, 0)+1
            if p['step_raw_px'] is not None:
                steps.append(p['step_raw_px'])
    summary = dict(complete=complete and len(records) == 72, views_run=len(records),
        frames_run=len({r['association']['source_frame_0_based'] for r in records}),
        inference_seconds=dict(median=float(np.median(timings)), p95=float(np.quantile(timings, .95)), sum=sum(timings)) if timings else None,
        decode_seconds_sum=sum(decode_times), wall_seconds=time.monotonic()-started,
        tracklet_lengths=lengths, tracklets=len(lengths), linked_steps=len(steps),
        max_step_raw_px=max(steps) if steps else None,
        complete_eight_frame_tracklets=sum(n == 8 for n in lengths.values()),
        quality_metrics_available=False, match_ball_identified=False, production_eligible=False,
        training_performed=False, official_data_access=False,
        limitations=['No independent pixel GT or identity labels',
        'Nearest neighbour can link distant unrelated candidates; no physical gate is validated',
        'Duplicate hypotheses across views remain separate; no global identity',
        'Eight frames cannot establish long-term tracking/restart or possession benefit',
        'No geometry, ATD anchor, classifier or decoder integration'])
    (output/'summary.json').write_text(json.dumps(summary, indent=2)+'\n')
    files = [p for p in output.rglob('*') if p.is_file()]
    (output/'manifest.json').write_text(json.dumps(dict(source_sha256=probe.sha(Path(__file__)),
        files_sha256={str(p.relative_to(output)): probe.sha(p) for p in files},
        official_data_access=False), indent=2)+'\n')


if __name__ == '__main__':
    main()

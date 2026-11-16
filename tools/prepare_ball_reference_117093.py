"""Prepare raw external ball-reference images; never read predictions/targets."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
FRAMES = tuple(range(4375, 4383))
FRAME_FIELDS = ['image_id', 'source_frame_0_based', 'image_sha256', 'scene_reviewed',
    'match_ball_visibility', 'match_ball_identity', 'match_center_x_raw',
    'match_center_y_raw', 'center_uncertainty_px', 'all_visible_ball_objects_reviewed',
    'annotator', 'model_predictions_seen', 'provider_ball_positions_seen',
    'used_for_fitting', 'notes']
OBJECT_FIELDS = ['image_id', 'object_id', 'role', 'visibility', 'center_x_raw',
    'center_y_raw', 'center_uncertainty_px', 'annotator', 'model_predictions_seen',
    'provider_ball_positions_seen', 'used_for_fitting', 'notes']


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def prepare(output):
    import cv2
    import numpy as np
    plan_path = ROOT / 'docs/soccertrack_v1_2_117093_download.json'
    plan = json.loads(plan_path.read_text())
    if plan['match'] != '117093' or plan['revision'] != '325bd4c57ee8e135ab652f3921d274526ada496b':
        raise ValueError('Unexpected source identity')
    entry, = [e for e in plan['files'] if e['path'] == 'videos/117093/117093_panorama_1st_half.mp4']
    video = (ROOT / 'Soccertrack V2' / entry['path']).resolve(strict=True)
    video.relative_to((ROOT / 'Soccertrack V2').resolve())
    if video.stat().st_size != entry['bytes'] or sha(video) != entry['payload_sha256']:
        raise ValueError('Source video size/checksum mismatch')
    if shutil.disk_usage(output.parent).free < 1024 ** 3:
        raise RuntimeError('Need at least 1 GiB project-drive space for bounded preparation')
    output.mkdir(exist_ok=False)
    (output / 'tiles-B01').mkdir()
    protocol_path = ROOT / 'docs/protokol_referensi_bola_117093.md'
    shutil.copyfile(protocol_path, output / 'protocol.md')
    protocol = dict(source_frames_0_based=FRAMES, context_frames_inclusive=[4225, 4524],
        decode='sequential from first decoded frame; count actual decoded images; no timestamp seek',
        source_video_sha256=entry['payload_sha256'], source_plan_sha256=sha(plan_path),
        sampling='known eight-frame development case, not a new held-out test',
        input_equivalence_to_previous_ffmpeg_probe_verified=False,
        protocol_sha256=sha(output / 'protocol.md'), model_predictions_read=False,
        provider_positions_read=False, geometric_projections_read=False, training=False,
        official_data_read=False, preview_is_measurement_image=False)
    (output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n', encoding='utf-8')
    cap = cv2.VideoCapture(str(video))
    if (not cap.isOpened() or round(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != 67625
            or abs(cap.get(cv2.CAP_PROP_FPS) - 25) > 1e-6
            or (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))) != (4096, 1080)
            or round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != 0):
        cap.release()
        raise ValueError('Source stream contract differs')
    context_path = output / 'context-frames-4225-4524.mp4'
    writer = cv2.VideoWriter(str(context_path), cv2.VideoWriter_fourcc(*'mp4v'), 25, (4096, 1128))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError('Context video encoder unavailable')
    images, tiles, previews = [], [], []
    context_count = 0
    try:
        for source_frame in range(4525):
            ok, image = cap.read()
            if not ok or image.shape != (1080, 4096, 3):
                raise ValueError('Sequential source decode failed')
            if round(cap.get(cv2.CAP_PROP_POS_FRAMES)) != source_frame + 1:
                raise ValueError('Decoder index disagrees with sequential count')
            if source_frame in FRAMES:
                image_id = f'B{source_frame - FRAMES[0] + 1:02d}'
                name = f'{image_id}-frame-{source_frame}-raw.png'
                if not cv2.imwrite(str(output / name), image):
                    raise OSError('Could not write raw PNG')
                images.append(dict(image_id=image_id, source_frame_0_based=source_frame,
                    path=name, sha256=sha(output / name), width=4096, height=1080,
                    context_frame_0_based=source_frame - 4225,
                    context_seconds=(source_frame - 4225) / 25))
                thumb = cv2.resize(image, (1024, 270), interpolation=cv2.INTER_AREA)
                thumb = cv2.copyMakeBorder(thumb, 28, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                cv2.putText(thumb, image_id + ' | preview only', (8, 20), cv2.FONT_HERSHEY_SIMPLEX, .5, (255, 255, 255), 1)
                previews.append(thumb)
                if image_id == 'B01':
                    for row in range(2):
                        for col in range(4):
                            x, y = col * 1024, row * 540
                            tile = f'tiles-B01/B01-r{row}-c{col}-raw.png'
                            if not cv2.imwrite(str(output / tile), image[y:y + 540, x:x + 1024]):
                                raise OSError('Could not write raw tile')
                            tiles.append(dict(path=tile, image_id='B01', x_origin=x, y_origin=y,
                                              width=1024, height=540, sha256=sha(output / tile)))
            if source_frame >= 4225:
                header = cv2.copyMakeBorder(image, 48, 0, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))
                text = f'Source frame {source_frame} | targets B01-B08: clip 6.00-6.28 s'
                if source_frame in FRAMES:
                    text += f' | B{source_frame - FRAMES[0] + 1:02d} TARGET'
                cv2.putText(header, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, .85, (255, 255, 255), 1)
                writer.write(header)
                context_count += 1
            if source_frame % 1000 == 0:
                print(json.dumps(dict(decoded_through_frame=source_frame)), flush=True)
    finally:
        writer.release()
        cap.release()
    if len(images) != 8 or context_count != 300:
        raise ValueError('Incomplete preparation')
    preview_path = output / 'preview-contact-sheet.jpg'
    if not cv2.imwrite(str(preview_path), np.vstack([np.hstack(previews[i:i+2]) for i in range(0,8,2)])):
        raise OSError('Could not write preview')
    for name in ('frames.csv', 'frames-template.csv'):
        with (output / name).open('x', newline='', encoding='utf-8') as stream:
            csv_writer = csv.DictWriter(stream, fieldnames=FRAME_FIELDS)
            csv_writer.writeheader()
            csv_writer.writerows(dict(image_id=e['image_id'], source_frame_0_based=e['source_frame_0_based'],
                                     image_sha256=e['sha256'], used_for_fitting='false') for e in images)
    for name in ('other-balls.csv', 'other-balls-template.csv'):
        with (output / name).open('x', newline='', encoding='utf-8') as stream:
            csv.DictWriter(stream, fieldnames=OBJECT_FIELDS).writeheader()
    initial = dict(source='operator_conversation_reply', reply='Belum pernah',
        scope='prior RT-DETR box/score exposure on frames 4375-4382 before annotation',
        annotator='operator_user', prior_model_predictions_seen=False,
        independently_verified_blinding=False, session_exposure_confirmation_pending=True,
        provider_ball_positions_exposure_unknown=True, labels_or_coordinates_generated=False)
    (output / 'initial-provenance.json').write_text(json.dumps(initial, indent=2)+'\n', encoding='utf-8')
    (output / 'operator-measurements.jsonl').write_text(json.dumps(initial)+'\n', encoding='utf-8')
    clip = cv2.VideoCapture(str(context_path))
    if not clip.isOpened() or round(clip.get(cv2.CAP_PROP_FRAME_COUNT)) != 300:
        raise ValueError('Context clip cannot be decoded')
    for sample in (0, 150, 157, 299):
        clip.set(cv2.CAP_PROP_POS_FRAMES, sample)
        ok, frame = clip.read()
        if not ok or frame.shape != (1128, 4096, 3):
            raise ValueError('Context sample decode failed')
    clip.release()
    immutable = [e['path'] for e in images] + [e['path'] for e in tiles] + [
        context_path.name, preview_path.name, 'protocol.md', 'protocol.json',
        'frames-template.csv', 'other-balls-template.csv', 'initial-provenance.json']
    manifest = dict(kind='external_match_ball_reference_collection', schema=1, match='117093',
        dataset_revision=plan['revision'], publisher_split='validation', project_role='component_development',
        created_at=datetime.now(timezone.utc).isoformat(), source_video_sha256=entry['payload_sha256'],
        source_video_bytes=entry['bytes'], source_manifest='docs/soccertrack_v1_2_117093_download.json',
        preparation_source_sha256=sha(Path(__file__)), images=images, tiles=tiles,
        files_sha256={name: sha(output / name) for name in immutable},
        context_audio_preserved=False, context_frames=300, independent_cameras=1,
        reference_measurements_completed=0, model_predictions_read=False, official_data_read=False,
        ready_for_component_scoring=False, ready_for_possession_evaluation=False)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(dict(status='empty_reference_package_prepared', frames=8,
                         source_frames=FRAMES, context_frames=context_count, coordinates_filled=0)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT / 'external_artifacts/v3').resolve())
    if output.exists():
        raise FileExistsError(output)
    prepare(output)

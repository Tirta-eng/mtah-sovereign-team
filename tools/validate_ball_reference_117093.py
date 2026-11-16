"""Read-only validation of external ball-reference annotation provenance."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

ROOT = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_center(x, y, width=4096, height=1080):
    if isinstance(x, bool) or isinstance(y, bool):
        raise ValueError('Boolean is not a pixel coordinate')
    x, y = float(x), float(y)
    if not (math.isfinite(x) and math.isfinite(y) and 0 <= x < width and 0 <= y < height):
        raise ValueError('Centre outside raw image or nonfinite')
    return x, y


def tile_to_raw(x, y, tile):
    x, y = check_center(x, y, tile['width'], tile['height'])
    return check_center(x + tile['x_origin'], y + tile['y_origin'])


def provenance(row):
    if row['used_for_fitting'] != 'false':
        raise ValueError('Reference marked for fitting')
    if not row['annotator']:
        raise ValueError('Annotator missing')
    keys = ('model_predictions_seen', 'provider_ball_positions_seen')
    if any(row[k] not in ('', 'true', 'false') for k in keys):
        raise ValueError('Invalid exposure value')
    return all(row[k] in ('true', 'false') for k in keys)


def validate_frame(row):
    if row['used_for_fitting'] != 'false':
        raise ValueError('Reference marked for fitting')
    if row['scene_reviewed'] not in ('', 'false', 'true'):
        raise ValueError('Invalid scene-review flag')
    xy = (row['match_center_x_raw'], row['match_center_y_raw'])
    if row['scene_reviewed'] != 'true':
        if any(xy) or row['match_ball_visibility'] or row['match_ball_identity']:
            raise ValueError('Unreviewed image must not have inferred labels/coordinates')
        return dict(reviewed=False, visible_match_centre=False, provenance_complete=False)
    complete = provenance(row)
    if row['all_visible_ball_objects_reviewed'] not in ('', 'false', 'true'):
        raise ValueError('Invalid exhaustive-review flag')
    visibility, identity = row['match_ball_visibility'], row['match_ball_identity']
    if visibility not in ('visible', 'occluded', 'not_visible', 'ambiguous'):
        raise ValueError('Invalid visibility')
    if identity not in ('match_confirmed', 'uncertain', 'not_assessable'):
        raise ValueError('Invalid match-ball identity')
    measured = visibility == 'visible' and identity == 'match_confirmed'
    if measured:
        check_center(*xy)
    elif any(xy) or row['center_uncertainty_px']:
        raise ValueError('Unavailable/ambiguous match ball must not have guessed coordinates')
    if row['center_uncertainty_px']:
        uncertainty = float(row['center_uncertainty_px'])
        if not math.isfinite(uncertainty) or uncertainty <= 0:
            raise ValueError('Invalid localization uncertainty')
    return dict(reviewed=True, visible_match_centre=measured, provenance_complete=complete)


def read_rows(path):
    with path.open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def validate(task):
    task = task.resolve(strict=True)
    task.relative_to((ROOT / 'external_artifacts/v3').resolve())
    manifest = json.loads((task / 'manifest.json').read_text())
    if (manifest['kind'] != 'external_match_ball_reference_collection' or manifest['match'] != '117093'
            or manifest['dataset_revision'] != '325bd4c57ee8e135ab652f3921d274526ada496b'):
        raise ValueError('Unexpected reference provenance')
    plan = json.loads((ROOT / 'docs/soccertrack_v1_2_117093_download.json').read_text())
    video, = [e for e in plan['files'] if e['path'] == 'videos/117093/117093_panorama_1st_half.mp4']
    if manifest['source_video_sha256'] != video['payload_sha256'] or manifest['source_video_bytes'] != video['bytes']:
        raise ValueError('Source video identity differs from pinned manifest')
    for name, expected in manifest['files_sha256'].items():
        path = (task / name).resolve(strict=True)
        path.relative_to(task)
        if sha(path) != expected:
            raise ValueError('Immutable reference checksum mismatch: ' + name)
    images = {e['image_id']: e for e in manifest['images']}
    if len(images) != 8 or set(images) != {f'B{i:02d}' for i in range(1,9)}:
        raise ValueError('Unexpected image slots')
    for index, name in enumerate(sorted(images)):
        image = images[name]
        if image['source_frame_0_based'] != 4375 + index:
            raise ValueError('Source frame sequence mismatch')
        path = (task / image['path']).resolve(strict=True)
        path.relative_to(task)
        data = path.read_bytes()
        if (sha(path) != image['sha256'] or data[:8] != b'\x89PNG\r\n\x1a\n'
                or struct.unpack('>II', data[16:24]) != (4096,1080)):
            raise ValueError('Raw PNG identity/dimensions mismatch')
    tiles = manifest['tiles']
    if (len(tiles) != 8 or len({t['path'] for t in tiles}) != 8
            or {(t['x_origin'],t['y_origin']) for t in tiles} != {(c*1024,r*540) for r in range(2) for c in range(4)}):
        raise ValueError('Unexpected tile grid')
    for tile in tiles:
        if tile['image_id'] != 'B01' or (tile['width'],tile['height']) != (1024,540):
            raise ValueError('Tile provenance differs from raw image')
        path = (task / tile['path']).resolve(strict=True)
        path.relative_to(task)
        data = path.read_bytes()
        if sha(path) != tile['sha256'] or struct.unpack('>II',data[16:24]) != (1024,540):
            raise ValueError('Tile checksum/dimensions mismatch')
    rows = read_rows(task / 'frames.csv')
    if len(rows) != 8 or len({r['image_id'] for r in rows}) != 8 or {r['image_id'] for r in rows} != set(images):
        raise ValueError('Missing/duplicate frame annotations')
    states = []
    for row in rows:
        image = images[row['image_id']]
        if row['image_sha256'] != image['sha256'] or int(row['source_frame_0_based']) != image['source_frame_0_based']:
            raise ValueError('Annotation/image provenance mismatch')
        states.append(validate_frame(row))
    objects = read_rows(task / 'other-balls.csv')
    identifiers = set()
    for row in objects:
        key = (row['image_id'], row['object_id'])
        if row['image_id'] not in images or not row['object_id'] or key in identifiers:
            raise ValueError('Missing/duplicate/unexpected ball-object identity')
        identifiers.add(key)
        provenance(row)
        if row['role'] not in ('spare', 'unknown') or row['visibility'] != 'visible':
            raise ValueError('Other-ball object must be visible and explicitly spare/unknown')
        check_center(row['center_x_raw'], row['center_y_raw'])
        if row['center_uncertainty_px']:
            v = float(row['center_uncertainty_px'])
            if not math.isfinite(v) or v <= 0:
                raise ValueError('Invalid other-ball uncertainty')
    reviewed = sum(s['reviewed'] for s in states)
    complete = sum(s['provenance_complete'] for s in states)
    object_provenance_complete = all(provenance(r) for r in objects)
    return dict(status='reference_preflight_passed', frame_slots=8, reviewed_frames=reviewed,
        visible_match_centres=sum(s['visible_match_centre'] for s in states),
        other_visible_ball_objects=len(objects), frame_exposure_provenance_complete=complete,
        exhaustive_object_review_frames=sum(r['all_visible_ball_objects_reviewed']=='true' for r in rows),
        frames_with_reported_prediction_or_provider_exposure=sum(
            r['model_predictions_seen']=='true' or r['provider_ball_positions_seen']=='true' for r in rows),
        objects_with_reported_prediction_or_provider_exposure=sum(
            r['model_predictions_seen']=='true' or r['provider_ball_positions_seen']=='true' for r in objects),
        raw_image_checksums_and_dimensions_verified=8, immutable_files_verified=len(manifest['files_sha256']),
        independently_verified_blinding=False, uncertainty_established=False,
        ready_for_reference_freeze=reviewed==8 and complete==8 and object_provenance_complete
            and all(r['all_visible_ball_objects_reviewed']=='true' for r in rows),
        ready_for_component_scoring=False, ready_for_precision_recall=False,
        ready_for_possession_evaluation=False, model_predictions_read=False,
        source_manifest_sha256=sha(task / 'manifest.json'), frames_csv_sha256=sha(task / 'frames.csv'),
        other_balls_csv_sha256=sha(task / 'other-balls.csv'), official_data_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT / 'external_artifacts/v3').resolve())
    if output.exists():
        raise FileExistsError(output)
    report = validate(args.task)
    with output.open('x', encoding='utf-8') as stream:
        json.dump(report, stream, indent=2)
        stream.write('\n')
    print(json.dumps(report, indent=2))

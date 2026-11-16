"""Prepare new external interval review clips, without reading provider labels.

Four fixed 30-second windows, selected by video time rather than predictions,
events or status. Output keeps 4096x1080 source imagery plus a 48-pixel header.
Frame spans are zero-based, end-exclusive at 25 Hz; no official paths, model,
target inference or fitting. Operator must label intervals, not one midpoint.
This small single-match development task cannot establish cross-match quality.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import time

ROOT = Path(__file__).resolve().parents[1]
CENTRES_SECONDS = (180, 660, 1380, 2220)


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def probe(path):
    command = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=width,height,avg_frame_rate,nb_frames,start_time,duration', '-of', 'json', str(path)]
    return json.loads(subprocess.check_output(command))['streams'][0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    output.relative_to((ROOT / 'external_artifacts/v3').resolve())
    if output.exists():
        raise FileExistsError(output)
    plan = json.loads((ROOT / 'docs/soccertrack_v1_2_117093_download.json').read_text())
    entry = next(e for e in plan['files'] if e['path'].startswith('videos/'))
    video = ROOT / 'Soccertrack V2' / entry['path']
    if video.stat().st_size != entry['bytes'] or sha(video) != entry['payload_sha256']:
        raise ValueError('External source video checksum mismatch')
    stream = probe(video)
    if (stream['width'], stream['height'], stream['avg_frame_rate'], int(stream['nb_frames'])) != (4096, 1080, '25/1', 67625):
        raise ValueError('Unexpected external video stream')
    windows = [dict(clip_id=f'R{i:02}',source_start_frame=(c-15)*25,
                    source_stop_frame_exclusive=(c+15)*25, frames=750, fps=25)
               for i,c in enumerate(CENTRES_SECONDS,1)]
    # Exclude every old Q window by frame bounds, never consult its status key.
    previous = json.loads((ROOT / 'external_artifacts/v3/117093-possession-review-pilot-20261113/manifest.json').read_text())
    for window in windows:
        for old in previous['clips']:
            if max(window['source_start_frame'],old['source_frame_start_0_based']) < min(window['source_stop_frame_exclusive'],old['source_frame_end_inclusive_0_based']+1):
                raise ValueError('Review window overlaps a previously reviewed Q clip')
    output.mkdir(parents=True)
    protocol = dict(kind='external_interval_review_protocol',match='117093',dataset_revision=plan['revision'],
        source_video_sha256=entry['payload_sha256'],selection='four fixed clock windows; no provider labels, event anchors or predictions consulted',
        candidate_scores_read=False,role='development_annotation_pilot_not_test',windows=windows,
        video_frame_convention='zero-based end-exclusive; source_frame=start+clip_frame',
        seeking='FFmpeg accurate input seek, seconds relative to file start; not absolute container PTS',
        operator_target='all intervals in each clip; no default provider/team labels',
        label_values=['live','dead','uncertain'],team_values=['biru','putih','uncertain','not_applicable'],
        provider_labels_rendered=False,model_predictions_rendered=False,independent_target_certified=False,
        trained=False,production_eligible=False,official_data_access=False)
    (output / 'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    with (output/'intervals.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.writer(f,lineterminator='\n')
        writer.writerow(['clip_id','start_s','end_s_exclusive','game_state','controlling_team_visual','annotator','provider_labels_seen','model_predictions_seen','notes'])
        for window in windows:
            writer.writerow([window['clip_id'],0,30,'','','','','','Split into separate rows for every state/team change; uncertain where boundaries are unclear.'])
    (output/'README.md').write_text('''# Review interval baru SoccerTrack 117093

Empat klip R01-R04, masing-masing 30 detik, dipilih pada waktu tetap sebelum
membaca status provider. Tidak tumpang tindih klip Q01-Q06. Gambar lapangan
tetap pada resolusi sumber, hanya ditambah header. Klip ini tanpa audio.

Tonton keseluruhan klip dan catat rentang waktu lokal pada intervals.csv.
Pisahkan baris setiap perubahan live/dead atau tim yang menguasai. Gunakan
live/dead/uncertain; warna biru/putih/uncertain ketika live, not_applicable
ketika dead. Rentang [start_s,end_s) tidak boleh tumpang tindih. Cakup 0-30 s;
bagian tidak jelas diberi uncertain. Jangan mengisi satu label untuk seluruh
klip jika status berubah. Nomor 0,04 detik adalah satu frame, bukan kewajiban
menebak ketelitian transisi: rentang transisi yang tidak jelas tetap uncertain.

Contoh format saja, bukan jawaban klip: Rxx,0,8,live,biru; Rxx,8,9,uncertain,
uncertain; Rxx,9,30,dead,not_applicable. Tambahkan annotator dan paparan label/
prediksi. Jangan melihat raw status, event provider, overlay atau key saat
menilai. Jika pernah terpapar, catat apa dan kapan; jangan klaim blind baru.

Source: SoccerTrack v2 v1.2, Atom Scott dan kontributor, CC BY 4.0.
https://huggingface.co/datasets/atomscott/soccertrack-v2
Perubahan: ekstraksi ulang, encoding lossy dan header; tanpa fitting/label dari
model. Ini pilot development satu pertandingan, bukan test antarpertandingan,
anotasi bola independen, atau bukti Spell F1. Paket baru bukan model V3.
''',encoding='utf-8')
    entries=[]
    for window in windows:
        dest=output/(window['clip_id']+'.mp4')
        start=window['source_start_frame']/25
        caption=f"{window['clip_id']} | source start frame {window['source_start_frame']} | local 0-30s | 25fps"
        vf=f"pad=iw:ih+48:0:48:color=black,drawtext=fontfile=/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf:text='{caption}':x=16:y=8:fontsize=26:fontcolor=white"
        command=['ffmpeg','-nostdin','-v','error','-threads','1','-ss',str(start),'-i',str(video),
                 '-map','0:v:0','-an','-sn','-dn','-frames:v','750','-filter_threads','1','-vf',vf,
                 '-c:v','libx264','-preset','veryfast','-crf','18','-pix_fmt','yuv420p','-threads','1',
                 '-movflags','+faststart','-n',str(dest)]
        before=time.monotonic()
        subprocess.run(command,check=True)
        info=probe(dest)
        if (int(info['nb_frames']),info['width'],info['height'],info['avg_frame_rate']) != (750,4096,1128,'25/1'):
            raise ValueError('Encoded clip frame/shape mismatch')
        item=dict(window,path=dest.name,sha256=sha(dest),bytes=dest.stat().st_size,
                  encode_wall_seconds=time.monotonic()-before,stream=info)
        entries.append(item)
        print(json.dumps(dict(clip=dest.name,frames=750,bytes=item['bytes'],encode_seconds=item['encode_wall_seconds'])),flush=True)
    manifest=dict(protocol_sha256=sha(output/'protocol.json'),source_video_sha256=entry['payload_sha256'],
        generator_sha256=sha(Path(__file__)),clips=entries,operator_measurements_completed=0,
        source_status_read_for_selection=False,training_performed=False,official_data_access=False,
        production_eligible=False,files_sha256={p.name:sha(p) for p in output.iterdir() if p.is_file()})
    (output/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


if __name__ == '__main__':
    main()

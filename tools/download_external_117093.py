"""Download only the ten pinned external SoccerTrack 117093 audit inputs.

Python standard library only; import/help/dry-run perform no network requests.
Outputs preserve raw/bas/videos paths under project-local 'Soccertrack V2'.
An optional HF token is read with getpass, never persisted or logged. Interrupted
downloads retain .partial bytes and resume with validated HTTP byte ranges.
Files become final only after exact size and SHA256 (LFS) or Git blob SHA1
(non-LFS) verification. These checks do not certify target semantics or quality.
No dataset parsing, training, model inference or official input access occurs.
"""

import argparse
import getpass
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import time
import urllib.error
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
REVISION = '325bd4c57ee8e135ab652f3921d274526ada496b'
MANIFESTS = ('soccertrack_v1_2_117093_download.json',
             'soccertrack_v1_2_117093_possession_verified.json')


class SafeRedirect(urllib.request.HTTPRedirectHandler):
    """Follow HTTPS redirects; never forward the HF bearer token to a CDN."""

    def redirect_request(self, req, fp, code, msg, headers, newurl):
        target = urllib.parse.urlsplit(newurl)
        if target.scheme != 'https' or target.username or target.password:
            raise ValueError('Refusing unsafe download redirect')
        result = super().redirect_request(req, fp, code, msg, headers, newurl)
        if result is not None and target.hostname != 'huggingface.co':
            result.remove_header('Authorization')
        return result


def load_specs(without_video=False):
    """Read pinned local plans, validate paths/URLs/checksums, return file specs."""
    result = []
    for name in MANIFESTS:
        manifest = json.loads((ROOT / 'docs' / name).read_text())
        if manifest['revision'] != REVISION or manifest['match'] != '117093':
            raise ValueError('Unexpected dataset revision or match')
        for spec in manifest['files']:
            path = spec['path']
            if (not path.startswith(('raw/117093/', 'bas/117093/', 'videos/117093/'))
                    or '..' in Path(path).parts or Path(path).is_absolute()):
                raise ValueError('Unexpected external input path')
            expected = ('https://huggingface.co/datasets/atomscott/soccertrack-v2/'
                        'resolve/' + REVISION + '/' + path)
            if spec['url'] != expected or type(spec['bytes']) is not int or spec['bytes'] <= 0:
                raise ValueError('Unexpected download URL or size')
            digest = spec.get('payload_sha256') if spec['lfs'] else spec['git_oid']
            if not re.fullmatch(r'[0-9a-f]{64}' if spec['lfs'] else r'[0-9a-f]{40}', digest or ''):
                raise ValueError('Missing pinned payload checksum')
            if not (without_video and path.startswith('videos/')):
                result.append(spec)
    if len({s['path'] for s in result}) != len(result):
        raise ValueError('Duplicate download path')
    return result


def verify(path, spec):
    """Stream exact-byte verification; native .npy/.npz/XML contents stay opaque."""
    if path.is_symlink() or path.stat().st_size != spec['bytes']:
        raise ValueError('Size mismatch: ' + spec['path'])
    digest = hashlib.sha256() if spec['lfs'] else hashlib.sha1()
    if not spec['lfs']:
        digest.update(b'blob ' + str(spec['bytes']).encode() + b'\0')
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    expected = spec['payload_sha256'] if spec['lfs'] else spec['git_oid']
    if digest.hexdigest() != expected:
        raise ValueError('Checksum mismatch; file preserved, not accepted: ' + spec['path'])


def transfer(spec, destination, token, opener):
    """Resume one opaque payload; validate response range and bounded byte count."""
    part = destination.with_name(destination.name + '.partial')
    if destination.is_symlink() or part.is_symlink():
        raise ValueError('Refusing symlink destination')
    if destination.exists():
        verify(destination, spec)
        print('Verified existing:', spec['path'], flush=True)
        return
    start = part.stat().st_size if part.exists() else 0
    if start > spec['bytes']:
        raise ValueError('Oversized partial; inspect before retry: ' + spec['path'])
    destination.parent.mkdir(parents=True, exist_ok=True)
    if start < spec['bytes']:
        headers = {'Accept-Encoding': 'identity', 'User-Agent': 'MTAH-external-audit/1.0'}
        if token:
            headers['Authorization'] = 'Bearer ' + token
        if start:
            headers['Range'] = 'bytes=' + str(start) + '-'
        request = urllib.request.Request(spec['url'], headers=headers)
        with opener.open(request, timeout=90) as response:
            if response.status == 206:
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', response.headers.get('Content-Range', ''))
                if (not match or int(match[1]) != start or int(match[2]) != spec['bytes'] - 1
                        or int(match[3]) != spec['bytes']):
                    raise ValueError('Unexpected resume range: ' + spec['path'])
                mode = 'ab' if start else 'wb'
            elif response.status == 200:
                # Range ignored: truncate partial, do not append duplicate bytes.
                start, mode = 0, 'wb'
            else:
                raise ValueError('Unexpected response status')
            count, last = start, time.monotonic()
            with part.open(mode) as stream:
                while True:
                    block = response.read(1024 * 1024)
                    if not block:
                        break
                    if count + len(block) > spec['bytes']:
                        raise ValueError('Response exceeds expected size: ' + spec['path'])
                    stream.write(block)
                    count += len(block)
                    if time.monotonic() - last >= 5:
                        print(f"{spec['path']}: {count:,}/{spec['bytes']:,} bytes ({count/spec['bytes']:.1%})", flush=True)
                        last = time.monotonic()
    verify(part, spec)
    part.rename(destination)
    print('Downloaded and verified:', spec['path'], flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run', action='store_true', help='List files/space; no network or writes')
    parser.add_argument('--without-video', action='store_true', help='Download nine non-video inputs first')
    args = parser.parse_args()
    specs = load_specs(args.without_video)
    output = ROOT / 'Soccertrack V2'
    destinations = []
    remaining = 0
    for spec in specs:
        destination = output / spec['path']
        destination.resolve().relative_to(output.resolve())
        destinations.append(destination)
        if destination.exists():
            verify(destination, spec)
        else:
            part = destination.with_name(destination.name + '.partial')
            existing = part.stat().st_size if part.exists() else 0
            if existing > spec['bytes']:
                raise ValueError('Oversized partial: ' + spec['path'])
            remaining += spec['bytes'] - existing
        print(f"{spec['path']}  {spec['bytes']:,} bytes")
    free = shutil.disk_usage(ROOT).free
    print(f'Destination: {output}\nTotal: {sum(s["bytes"] for s in specs):,} bytes; remaining: {remaining:,}; free: {free:,}')
    if args.dry_run:
        return
    if free < remaining + 512 * 1024 * 1024:
        raise ValueError('Insufficient project-drive space (requires remaining bytes + 512 MiB reserve)')
    needs_network = any(not p.exists() and (not p.with_name(p.name + '.partial').exists()
                        or p.with_name(p.name + '.partial').stat().st_size < s['bytes'])
                        for s, p in zip(specs, destinations))
    if needs_network and not sys.stdin.isatty():
        raise ValueError('Run from an interactive terminal for the hidden token prompt')
    token = getpass.getpass('HF read token (hidden; Enter if not required): ').strip() if needs_network else ''
    if token and not re.fullmatch(r'hf_[A-Za-z0-9]+', token):
        raise ValueError('Invalid HF token format; token value is not displayed')
    opener = urllib.request.build_opener(SafeRedirect())
    for spec, destination in zip(specs, destinations):
        transfer(spec, destination, token, opener)
    print('All requested inputs verified. No dataset parsing or training performed.')


if __name__ == '__main__':
    try:
        main()
    except urllib.error.HTTPError as error:
        print(f'HTTP {error.code}: check dataset access/token (401/403), or retry later. Partial files retained.', file=sys.stderr)
        sys.exit(1)
    except (ValueError, OSError, urllib.error.URLError) as error:
        # Avoid logging signed redirect URLs, bearer headers or token contents.
        message = str(error) if isinstance(error, ValueError) else type(error).__name__
        print('Stopped: ' + message + '. Partial files retained; rerun to resume.', file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print('\nInterrupted. Rerun the same command to resume.', file=sys.stderr)
        sys.exit(130)

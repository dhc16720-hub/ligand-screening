#!/usr/bin/env python3
"""Package only audited course artifacts; never walk the workspace for uploads."""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import tempfile
import zipfile

import reproduce_topic1 as runner


def safe_name(name):
    path = PurePosixPath(name)
    if path.is_absolute() or '..' in path.parts or '\\' in name or not path.parts:
        raise RuntimeError(f'Unsafe archive path: {name}')
    if any(p in {'.git', '.venv-topic1', '.env', '.env.local'} for p in path.parts):
        raise RuntimeError(f'Non-deliverable archive path: {name}')
    return name


def zip_checked(root, destination, entries):
    """Check the same bytes we archive; an incomplete ZIP is never published."""
    if destination.exists():
        raise RuntimeError(f'Refusing to replace release asset: {destination}')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=destination.parent, suffix='.partial', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
            for name, expected in sorted(entries.items()):
                safe_name(name)
                path = root/name
                if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
                    raise RuntimeError(f'Unsafe source: {name}')
                data = path.read_bytes()
                if hashlib.sha256(data).hexdigest() != expected:
                    raise RuntimeError(f'Changed artifact while packaging: {name}')
                archive.writestr(name, data)
            archive.writestr(f'release_manifests/{destination.stem}.json', json.dumps(entries, indent=2)+'\n')
        verify_zip(temporary)
        # Do not replace an existing version, even if another process created it meanwhile.
        import os
        os.link(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return dict(name=destination.name, bytes=destination.stat().st_size,
                sha256=runner.digest(destination), files=len(entries))


def verify_zip(path):
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)): raise RuntimeError('Duplicate archive entries')
        for name in names: safe_name(name)
        manifests = [n for n in names if n.startswith('release_manifests/') and n.endswith('.json')]
        if len(manifests) != 1: raise RuntimeError('Expected one internal file manifest')
        entries = json.loads(archive.read(manifests[0]))
        if not entries or set(names) != set(entries) | {manifests[0]}:
            raise RuntimeError('Incomplete or unexpected archive coverage')
        for name, expected in entries.items():
            with archive.open(name) as stream:
                actual = hashlib.file_digest(stream, 'sha256').hexdigest()
            if actual != expected: raise RuntimeError(f'Archive checksum mismatch: {name}')
    return len(entries)


def verify_release(folder):
    record = json.loads((folder/'release.json').read_text())
    if record.get('kind') != 'topic1-audited-results-and-models' or len(record.get('assets', [])) != 2:
        raise RuntimeError('Invalid release descriptor')
    for asset in record['assets']:
        name = safe_name(asset['name'])
        if len(PurePosixPath(name).parts) != 1: raise RuntimeError('Asset must be in release directory')
        path = folder/name
        if path.stat().st_size != asset['bytes'] or runner.digest(path) != asset['sha256']:
            raise RuntimeError(f'Release asset checksum mismatch: {name}')
        if verify_zip(path) != asset['files']: raise RuntimeError('Wrong file count')
    print('RELEASE VERIFIED: two archives, every internal file and archive checksum', flush=True)


def build(version, folder):
    if not re.fullmatch(r'topic1-v\d+\.\d+\.\d+(?:-[A-Za-z0-9.]+)?', version):
        raise RuntimeError('Use a version such as topic1-v1.0.0')
    if folder.exists(): raise RuntimeError('Use a new release directory; published assets are immutable')
    root = runner.ROOT
    runner.check_public()
    record_path = root/'outputs/project_verification/latest.json'
    record = json.loads(record_path.read_text())
    if record.get('status') != 'passed' or record.get('pass_all') is not True or \
            set(record.get('stages', {})) != {'historical_and_final', 'small_data', 'external_diagnostics'} or \
            not all(s.get('pass_all') is True for s in record['stages'].values()):
        raise RuntimeError('A successful check-project is required before packaging')
    if record['source_manifest_sha256'] != runner.digest(root/runner.SOURCE_MANIFEST):
        raise RuntimeError('Public files changed since project verification; rerun check-project')
    import verify_topic1_project as project
    if project.snapshot(root) != record['artifact_hashes']:
        raise RuntimeError('Artifacts changed since project verification; rerun check-project')
    models = {p:h for p,h in record['artifact_hashes'].items() if p.endswith('.joblib')}
    results = {p:h for p,h in record['artifact_hashes'].items() if p not in models}
    results[record_path.relative_to(root).as_posix()] = runner.digest(record_path)
    folder.mkdir(parents=True)
    assets = [zip_checked(root, folder/f'{version}-results.zip', results),
              zip_checked(root, folder/f'{version}-models.zip', models)]
    descriptor = dict(kind='topic1-audited-results-and-models', version=version,
                      source_manifest_sha256=record['source_manifest_sha256'],
                      verified_platform=record['platform'], assets=assets,
                      instructions='Download repository source at the matching release tag. Extract results and models into that repository root. Install locked environment, then run check-project. Archives contain saved macOS results, not a new training run or universal cross-platform numerical equivalence.')
    (folder/'release.json').write_text(json.dumps(descriptor, indent=2)+'\n')
    (folder/'SHA256SUMS.txt').write_text(''.join(f"{a['sha256']}  {a['name']}\n" for a in assets)
                                  +f"{runner.digest(folder/'release.json')}  release.json\n")
    verify_release(folder)
    print(json.dumps(descriptor, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('build'); p.add_argument('--version', required=True); p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify'); p.add_argument('folder', type=Path)
    args = parser.parse_args()
    if args.command == 'build': build(args.version, args.output)
    else: verify_release(args.folder)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, str(runner.ROOT))
    main()

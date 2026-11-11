"""Standard-library integrity check; optionally restore only missing files."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--restore-missing-from',type=Path);args=parser.parse_args()
    root=Path(__file__).resolve().parents[1]
    manifests=sorted((root/'outputs').glob('topic1*/delivery_manifest.json'))
    archive=zipfile.ZipFile(args.restore_missing_from) if args.restore_missing_from else None
    problems=[];checked=0;restored=[]
    try:
        for manifest in manifests:
            entries=json.loads(manifest.read_text())
            for relative,expected in entries.items():
                destination=(root/relative).resolve()
                if not destination.is_relative_to(root):raise ValueError('Unsafe manifest path')
                if not destination.exists() and archive and relative in archive.namelist():
                    data=archive.read(relative)
                    if hashlib.sha256(data).hexdigest()!=expected:raise ValueError(f'Archive checksum mismatch: {relative}')
                    destination.parent.mkdir(parents=True,exist_ok=True);destination.write_bytes(data);restored.append(relative)
                if not destination.is_file():problems.append(dict(path=relative,status='missing'))
                elif hashlib.sha256(destination.read_bytes()).hexdigest()!=expected:problems.append(dict(path=relative,status='changed; never overwritten'))
                checked+=1
    finally:
        if archive:archive.close()
    print(json.dumps(dict(passed=not problems,checked_manifest_entries=checked,restored=restored,problems=problems),ensure_ascii=False,indent=2))
    raise SystemExit(1 if problems else 0)


if __name__=='__main__':main()

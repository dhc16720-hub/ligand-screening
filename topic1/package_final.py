"""Build an allowlisted research delivery and validate every archive member."""
import hashlib,json,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/topic1_final'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def main():
    files=set()
    for version in ['topic1','topic1_v2','topic1_v3','topic1_v4']:
        manifest=ROOT/'outputs'/version/'delivery_manifest.json'
        for name,expected in json.loads(manifest.read_text()).items():
            path=ROOT/name
            assert sha(path)==expected,name
            files.add(path)
        files.add(manifest)
    files.update(p for p in (ROOT/'topic1').iterdir() if p.is_file())
    files.update((ROOT/'tests').glob('test_topic1*.py'))
    files.update(p for p in OUT.rglob('*') if p.is_file() and 'qa' not in p.relative_to(OUT).parts and p.name!='delivery_manifest.json')
    for p in files:
        assert p.resolve().is_relative_to(ROOT) and not p.is_symlink(),p
        assert p.is_file(),p
    entries={p.relative_to(ROOT).as_posix():sha(p) for p in sorted(files)}
    manifest=OUT/'delivery_manifest.json'
    manifest.write_text(json.dumps(entries,ensure_ascii=False,indent=2))
    files.add(manifest)
    archive=ROOT/'outputs/Topic1_最终实验与报告_完整交付.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(files):z.write(p,p.relative_to(ROOT))
    with zipfile.ZipFile(archive) as z:
        assert z.testzip() is None
        assert len(z.namelist())==len(set(z.namelist()))==len(files)
        for name,expected in entries.items():assert hashlib.sha256(z.read(name)).hexdigest()==expected,name
    checksum=sha(archive)
    archive.with_suffix('.zip.sha256').write_text(checksum+'  '+archive.name+'\n')
    print(json.dumps(dict(archive=str(archive),files=len(files),bytes=archive.stat().st_size,sha256=checksum,all_hashes_verified=True),ensure_ascii=False,indent=2))
if __name__=='__main__':main()

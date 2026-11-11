#!/usr/bin/env python3
"""Public-clone setup and orchestration; frozen scientific modules stay unchanged."""
import argparse
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
COMMIT = '07e50bcac791ce7660684d71e59646a1b6b82faa'
FOLDS_SHA = '94c17bc3dac32c67d33d69289927a5fec174497e74bbc47293e23d3d147e9a6d'
VERSIONS = ['topic1', 'topic1_v2', 'topic1_v3', 'topic1_v4']
TASKS = ['matbench_steels', 'matbench_jdft2d', 'matbench_expt_gap', 'matbench_expt_is_metal', 'matbench_glass']
SOURCE_MANIFEST = 'results/topic1-final/source_manifest.json'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run(args):
    print('+ ' + ' '.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), cwd=ROOT, check=True)


def module(name, *args):
    run(['bash', ROOT/'topic1/python.sh', '-m', name, *args])


def check_manifest(root, manifest):
    entries = json.loads(manifest.read_text())
    if not isinstance(entries, dict) or not entries:
        raise RuntimeError(f'Empty/invalid manifest: {manifest}')
    for relative, expected in entries.items():
        path = (root/relative).resolve()
        if not path.is_relative_to(root.resolve()):
            raise RuntimeError(f'Unsafe manifest path: {relative}')
        if not path.is_file() or digest(path) != expected:
            raise RuntimeError(f'Missing or changed file: {relative}')
    return len(entries)


def check_public():
    count = check_manifest(ROOT, ROOT/SOURCE_MANIFEST)
    print(f'PUBLIC FILES OK: {count}. This verifies published bytes, not a training run.')


def check_outputs(versions):
    count = 0
    for version in versions:
        manifest = ROOT/'outputs'/version/'delivery_manifest.json'
        if not manifest.is_file():
            raise RuntimeError(f'Missing {manifest.relative_to(ROOT)}; run the corresponding stage first.')
        count += check_manifest(ROOT, manifest)
    print(f'OUTPUT FILES OK: {len(versions)} required manifests, {count} entries.')


def audit_passed(version):
    folder = ROOT/'outputs'/version
    audit = json.loads((folder/'requirement_audit.json').read_text())
    if audit.get('pass_all') is not True:
        raise RuntimeError(f'{version}: scientific audit did not pass')
    if version == 'topic1':
        tasks = audit.get('tasks', [])
        if {t['task'] for t in tasks} != set(TASKS) or any(t['official_folds'] != 5 or len(t['substantive_models']) < 3 for t in tasks):
            raise RuntimeError('Minimum requirements are incomplete')
        if audit.get('total_main_fits') != 320 or audit.get('learning_points_verified') != 50:
            raise RuntimeError('Baseline or learning-curve results are incomplete')
    elif version == 'topic1_v2':
        if set(audit.get('official_tasks', [])) != set(TASKS) or audit.get('verified_prediction_folds') != 260:
            raise RuntimeError('V2 results are incomplete')


def freeze(version):
    """Only after scientific audit; never replace an existing historical manifest."""
    audit_passed(version)
    folder = ROOT/'outputs'/version
    manifest = folder/'delivery_manifest.json'
    if manifest.exists():
        check_manifest(ROOT, manifest)
        return
    paths = [p for p in (ROOT/'topic1').iterdir() if p.is_file()]
    paths += list((ROOT/'tests').glob('test_topic1*.py'))
    paths += [p for p in folder.rglob('*') if p.is_file() and not any(x in p.parts for x in ['mpl-cache', '__pycache__', 'qa']) and p != manifest]
    entries = {p.relative_to(ROOT).as_posix(): digest(p) for p in sorted(set(paths))}
    manifest.write_text(json.dumps(entries, ensure_ascii=False, indent=2)+'\n')
    check_manifest(ROOT, manifest)
    print(f'FROZEN {version}: {len(entries)} entries; this run has its own hashes.')


def setup():
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Use Python 3.12: python3.12 scripts/reproduce_topic1.py setup')
    check_public()
    checkout = ROOT/'third_party/matbench'
    if not checkout.exists():
        checkout.parent.mkdir(parents=True, exist_ok=True)
        run(['git', 'clone', 'https://github.com/materialsproject/matbench.git', checkout])
        run(['git', '-C', checkout, 'checkout', '--detach', COMMIT])
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError('Existing Matbench checkout has a different commit; use a fresh project copy.')
    env = ROOT/'.venv-topic1'
    if not env.exists():
        run([sys.executable, '-m', 'venv', env])
    python = env/'bin/python'
    run([python, '-m', 'pip', 'install', '-r', ROOT/'topic1/requirements-v4.lock.txt'])
    run([python, '-m', 'pip', 'check'])
    run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/reproduce_topic1.py', 'preflight'])


def preflight():
    if not __debug__ or os.environ.get('PYTHONOPTIMIZE'):
        raise RuntimeError('Do not disable assertions; they enforce scientific audits.')
    if sys.version_info[:2] != (3, 12):
        raise RuntimeError('Python 3.12 is required')
    check_public()
    checkout = ROOT/'third_party/matbench'
    actual = subprocess.check_output(['git', '-C', str(checkout), 'rev-parse', 'HEAD'], text=True).strip()
    if actual != COMMIT:
        raise RuntimeError('Wrong Matbench source commit')
    for archive in ['matbench_v0.1_modnet_v0.1.12', 'matbench_v0.1_CrabNet']:
        for filename in ['results.json.gz', 'info.json']:
            if not (checkout/'benchmarks'/archive/filename).is_file():
                raise RuntimeError(f'Missing reference archive: {archive}/{filename}')
    for line in (ROOT/'topic1/requirements-v4.lock.txt').read_text().splitlines():
        if '==' in line:
            name, expected = line.split('==')
            actual = importlib.metadata.version(name)
            if actual != expected:
                raise RuntimeError(f'{name}: expected {expected}, found {actual}')
    from matbench.constants import MBV01_VALIDATION_DATA_PATH
    if digest(MBV01_VALIDATION_DATA_PATH) != FOLDS_SHA:
        raise RuntimeError('Installed official folds have changed')
    # Imports catch platform/OpenMP loading errors before a long run.
    import catboost, xgboost  # noqa: F401
    print('PREFLIGHT OK: Python, locked packages, Matbench source, archives, official folds and native libraries.')


def download_external():
    # Fetch bytes only; external labels are first read by the frozen evaluator after source fitting.
    destination = ROOT/'outputs/topic1_final/datasets/glass_ternary_hipt.json.gz'
    expected = '4a4c60ef227891203570dbc3decb606649b877f6b75289afd4e78fe8e8095e63'
    if destination.exists():
        if digest(destination) != expected:
            raise RuntimeError('External cache checksum mismatch; refusing to overwrite')
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.download')
    try:
        import requests
        from requests.adapters import HTTPAdapter
        from urllib3.util.retry import Retry
        with requests.Session() as session:
            session.mount('https://', HTTPAdapter(max_retries=Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
            with session.get('https://ndownloader.figshare.com/files/13298018', timeout=(15, 60), stream=True) as response, temporary.open('wb') as output:
                response.raise_for_status()
                for chunk in response.iter_content(1024*1024):
                    output.write(chunk)
        if digest(temporary) != expected:
            raise RuntimeError('External download checksum mismatch')
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    print('EXTERNAL CACHE OK: verified SHA-256; no labels used by this downloader.')


def stage(name):
    dependencies = {'minimum': [], 'v2': VERSIONS[:1], 'v3': VERSIONS[:2], 'v4': VERSIONS[:3], 'final': VERSIONS}
    check_outputs(dependencies[name]) if dependencies[name] else None
    if name == 'minimum':
        if (ROOT/'outputs/topic1/delivery_manifest.json').exists():
            check_outputs(['topic1']); audit_passed('topic1'); return
        module('topic1.run', 'prepare')
        module('topic1.run', 'run')
        module('topic1.run', 'learning', '--tasks', 'matbench_steels', 'matbench_expt_gap')
        module('topic1.diagnose'); module('topic1.report'); freeze('topic1')
    elif name == 'v2':
        if (ROOT/'outputs/topic1_v2/delivery_manifest.json').exists():
            check_outputs(['topic1_v2']); audit_passed('topic1_v2'); return
        for step in ['main', 'ablation', 'robustness']:
            module('topic1.advanced', step)
        for mod in ['conformal', 'linear_ablation', 'references', 'advanced_report']:
            module('topic1.'+mod)
        freeze('topic1_v2')
    elif name == 'v3':
        if (ROOT/'outputs/topic1_v3/delivery_manifest.json').exists():
            check_outputs(['topic1_v3']); return
        module('topic1.refine'); module('topic1.refine_report')
    elif name == 'v4':
        if (ROOT/'outputs/topic1_v4/delivery_manifest.json').exists():
            check_outputs(['topic1_v4']); return
        module('topic1.physics'); module('topic1.physics_report')
    else:
        if (ROOT/'outputs/topic1_final/verification_manifest.json').exists():
            run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/verify_topic1_complete.py'])
            return
        download_external()
        module('topic1.final_equivalence')
        for step in ['conditional', 'reselection', 'conformal', 'common_pool']:
            module('topic1.final_validation', step)
        for mod in ['final_external', 'final_audit', 'final_report_data']:
            module('topic1.'+mod)
        run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/verify_topic1_complete.py', '--seal'])
        print('FINAL SCIENTIFIC AUDIT COMPLETE. DOCX/native slides are optional separate presentation builds.')



def project_command(name):
    """Invalidate old success before preflight or any training stage can fail."""
    import datetime
    import time
    from verify_topic1_project import atomic_json, REPORT
    target = ROOT/REPORT
    started = time.monotonic()
    record = dict(status='running', pass_all=False, command=name,
                  started_at=datetime.datetime.now(datetime.timezone.utc).isoformat())
    atomic_json(target, record)
    try:
        preflight()
        if name == 'check-project':
            run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/verify_topic1_project.py'])
        else:
            for child in ['all', 'small-data', 'external-diagnostics', 'check-project']:
                run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/reproduce_topic1.py', child])
        # A zero exit code without the complete science record is insufficient.
        result = json.loads(target.read_text())
        if result.get('status') != 'passed' or result.get('pass_all') is not True or \
                set(result.get('stages', {})) != {'historical_and_final', 'small_data', 'external_diagnostics'} or \
                not all(v.get('pass_all') is True for v in result['stages'].values()):
            raise RuntimeError('Project command did not produce a complete successful audit')
    except BaseException as error:
        record.update(status='failed', pass_all=False, elapsed_seconds=time.monotonic()-started,
                      finished_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      error={'type':type(error).__name__, 'message':str(error)})
        atomic_json(target, record)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage', choices=['setup','check-public','preflight','test','minimum','v2','v3','v4','final','all','check-outputs','check-complete','seal-final','download-external','small-data','check-small-data','external-diagnostics','project','check-project'])
    args = parser.parse_args()
    if args.stage == 'setup': setup()
    elif args.stage == 'check-public': check_public()
    elif args.stage == 'check-outputs': check_outputs(VERSIONS)
    elif args.stage == 'download-external': download_external()
    elif args.stage in ['project', 'check-project']: project_command(args.stage)
    else:
        preflight()
        if args.stage == 'preflight': return
        if args.stage in ['check-complete', 'seal-final']:
            extra = ['--seal'] if args.stage == 'seal-final' else []
            run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/verify_topic1_complete.py', *extra])
            return
        if args.stage in ['small-data', 'check-small-data']:
            check_outputs(['topic1']); audit_passed('topic1')
            manifest=ROOT/'outputs/topic1_small_data/delivery_manifest.json'
            if manifest.exists(): check_manifest(ROOT,manifest)
            elif args.stage=='check-small-data': raise RuntimeError('Missing small-data delivery manifest; run small-data first')
            if args.stage=='small-data': module('experiments.small_data','run')
            module('experiments.small_data','audit')
            module('experiments.study_report')
            check_manifest(ROOT,manifest)
            return
        if args.stage=='external-diagnostics':
            run(['bash', ROOT/'topic1/python.sh', ROOT/'scripts/verify_topic1_complete.py'])
            module('experiments.external_diagnostics')
            return
        if args.stage == 'test':
            module('unittest', 'discover', '-s', 'tests', '-p', 'test_topic1*.py')
            module('unittest', 'discover', '-s', 'tests', '-p', 'test_reproduction.py')
            module('unittest', 'discover', '-s', 'tests', '-p', 'test_small_data.py')
            module('unittest', 'discover', '-s', 'tests', '-p', 'test_external_diagnostics.py')
            module('unittest', 'discover', '-s', 'tests', '-p', 'test_project_delivery.py')
        elif args.stage == 'all':
            for name in ['minimum','v2','v3','v4','final']: stage(name)
        else: stage(args.stage)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, FileNotFoundError, subprocess.CalledProcessError) as error:
        print(f'REPRODUCTION FAILED: {error}', file=sys.stderr)
        sys.exit(1)

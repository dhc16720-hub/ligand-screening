"""Compare archived Linux summaries with macOS; never round predictions to force equality."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import pandas as pd

TABLES={
 'minimum':('topic1/summary.csv',['task','representation','model'],['mae_mean','roc_auc_mean']),
 'v2':('topic1_v2/summary.csv',['task','representation','model'],['mae_mean','roc_auc_mean']),
 'v3':('topic1_v3/summary.csv',['task','representation','model','metric','unit'],['mean']),
 'v4':('topic1_v4/summary.csv',['task','representation','model','metric','unit'],['mean']),
 'final':('topic1_final/validation_summary.csv',['task','stage','seed'],['mae_mean','roc_auc_mean','coverage_mean','width_mean']),
 'small_data':('topic1_small_data/summary.csv',['task','model','representation','fraction'],['mean']),
 'small_data_paired':('topic1_small_data/paired_comparisons.csv',['task','model','fraction','comparison'],['mean_improvement','improved_folds']),
 'external':('topic1_external_diagnostics/system_metrics.csv',['system','model'],['roc_auc','brier','balanced_accuracy','predicted_positive_fraction']),
}


def compare(mac,linux,destination,require_linux=True,recovered=False):
    destination.mkdir(parents=True,exist_ok=True)
    (destination/'comparison.json').write_text(json.dumps(dict(status='running',pass_all=False))+'\n')
    evidence_path=linux/('linux_recovery/replay.json' if recovered else 'project_verification/latest.json')
    evidence=json.loads(evidence_path.read_text())
    if recovered:
        if (evidence.get('status')!='passed' or evidence.get('replay_passed') is not True
            or evidence.get('final_model_replays')!=86
            or evidence.get('small_data',{}).get('replayed_models')!=1050
            or evidence.get('small_data',{}).get('pass_all') is not True):
            raise RuntimeError('Successful saved-evidence replay required')
    elif (evidence.get('status') != 'passed' or evidence.get('pass_all') is not True
        or set(evidence.get('stages',{}))!={'historical_and_final','small_data','external_diagnostics'}
        or not all(v.get('pass_all') is True for v in evidence['stages'].values())):
        raise RuntimeError('Linux complete scientific audit required')
    if require_linux and not evidence.get('platform','').startswith('Linux'):
        raise RuntimeError('Expected a Linux scientific audit')
    mac_evidence=json.loads((mac/'project_verification/latest.json').read_text())
    if mac_evidence.get('pass_all') is not True or mac_evidence.get('status')!='passed':
        raise RuntimeError('Baseline scientific audit required')
    def checked_bytes(root, file, record):
        value=(root/file).read_bytes()
        if hashlib.sha256(value).hexdigest()!=record['artifact_hashes'].get('outputs/'+file):
            raise RuntimeError(f'Unverified or changed artifact: {root}/{file}')
        return value
    input_identity={}
    tasks=['matbench_steels','matbench_jdft2d','matbench_expt_gap','matbench_expt_is_metal','matbench_glass']
    for task in tasks:
        for name in ['official_folds.json','samples.csv']:
            file=f'topic1/{task}/{name}'
            a=checked_bytes(mac,file,mac_evidence);b=checked_bytes(linux,file,evidence)
            if a!=b:raise RuntimeError(f'Input identity differs: {file}')
            input_identity[file]=hashlib.sha256(a).hexdigest()
    feature_checks=[]
    for task in tasks:
        names=['A','B','C']+(['C_structure'] if task=='matbench_jdft2d' else [])
        for name in names:
            file=f'topic1/{task}/features_{name}.npy'
            a_bytes=checked_bytes(mac,file,mac_evidence);b_bytes=checked_bytes(linux,file,evidence)
            a=np.load(mac/file,allow_pickle=False);b=np.load(linux/file,allow_pickle=False)
            if a.shape!=b.shape or not np.array_equal(np.isfinite(a),np.isfinite(b)):
                raise RuntimeError(f'Feature shape or missing-value pattern differs: {file}')
            mask=np.isfinite(a)&np.isfinite(b)
            feature_checks.append(dict(task=task,representation=name,shape=list(a.shape),
                byte_identical=a_bytes==b_bytes,max_abs_difference=float(np.max(np.abs(a[mask]-b[mask]))) if mask.any() else 0.0))
    (destination/'feature_comparison.json').write_text(json.dumps(feature_checks,indent=2)+'\n')
    rows=[];coverage={}

    for stage,(file,keys,metrics) in TABLES.items():
        linux_file=file.replace('topic1_external_diagnostics/','linux_recovery/external_diagnostics/') if recovered else file
        checked_bytes(mac,file,mac_evidence);checked_bytes(linux,linux_file,evidence)
        left=pd.read_csv(mac/file);right=pd.read_csv(linux/linux_file)
        if left.duplicated(keys).any() or right.duplicated(keys).any():raise RuntimeError(f'Duplicate keys: {stage}')
        joined=left.merge(right,on=keys,how='outer',suffixes=('_mac','_linux'),indicator=True,validate='one_to_one')
        if not (joined['_merge']=='both').all():raise RuntimeError(f'Unmatched rows: {stage}')
        coverage[stage]=len(joined)
        for _,row in joined.iterrows():
            identity={('condition' if k=='stage' else k):row[k] for k in keys}
            for metric in metrics:
                a=row[metric+'_mac'];b=row[metric+'_linux']
                if pd.isna(a) and pd.isna(b):continue
                if not np.isfinite([a,b]).all():raise RuntimeError(f'Invalid metric coverage: {stage}, {identity}, {metric}')
                rows.append(dict(stage=stage,**identity,score_column=metric,mac=float(a),linux=float(b),
                                 delta_linux_minus_mac=float(b-a),abs_difference=float(abs(b-a))))
    frame=pd.DataFrame(rows);frame.to_csv(destination/'cross_platform_metrics.csv',index=False)
    frame['dataset']=frame['task'].fillna(frame['system'])
    maxima=frame.groupby(['stage','dataset','score_column'],as_index=False).abs_difference.max()
    maxima.to_csv(destination/'cross_platform_max_differences.csv',index=False)
    paired=frame[(frame.stage=='small_data_paired')&(frame.score_column=='mean_improvement')]
    flips=paired[(paired.mac*paired.linux)<0]
    flips.to_csv(destination/'paired_direction_changes.csv',index=False)
    # Explicitly compare the descriptive low-data steel result rather than generalising.
    steel=paired[(paired.task=='matbench_steels')&(paired.model=='ET')&(paired.comparison=='C-A')&(paired.fraction==.2)]
    result=dict(status='passed',pass_all=True,self_check=not require_linux,linux_saved_evidence_replay_passed=require_linux,linux_unified_check_project_passed=require_linux and not recovered,
                validation_scope=evidence.get('scope','complete check-project'),
                source_run=evidence.get('source_run'),
                evidence_record_sha256=hashlib.sha256(evidence_path.read_bytes()).hexdigest(),linux_platform=evidence['platform'],
                linux_source_manifest_sha256=evidence.get('source_manifest_sha256'),
                mac_source_manifest_sha256=mac_evidence.get('source_manifest_sha256'),
                checked_input_identity=input_identity,summary_rows=coverage,numeric_comparisons=len(frame),
                paired_mean_direction_changes=len(flips),steel_low_data_ET=steel[['mac','linux','delta_linux_minus_mac']].to_dict('records'),
                exact_numeric_equality=bool((frame.abs_difference==0).all()),
                limitation='Descriptive paired table comparison, not independent statistical inference or a new external benchmark. Different releases may contain non-scientific runner/doc changes; compare pinned source metadata separately.')
    (destination/'comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--mac',type=Path,required=True);p.add_argument('--linux',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--recovered-evidence',action='store_true');a=p.parse_args()
    try:
        compare(a.mac,a.linux,a.output,recovered=a.recovered_evidence)
    except BaseException as exc:
        a.output.mkdir(parents=True,exist_ok=True)
        (a.output/'comparison.json').write_text(json.dumps(dict(status='failed',pass_all=False,error=str(exc)),indent=2)+'\n')
        raise

"""Read-only audit of historical results; independent checks of final validation."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from matminer.datasets import load_dataset
from sklearn.metrics import roc_auc_score,brier_score_loss
from topic1.run import ROOT,TASKS,write_json,sha,scores
from topic1.final_validation import OUT,V1,V2,V3,V4,read,register,data,take,historical_recipe
from topic1.final_models import specifications
from topic1.refine_models import select,combine
from topic1.physics_models import gate
from topic1.advanced import mean_inner_loss
from topic1.final_external import composition_key


def audit_selection(folder,task,expected_ids):
    r=json.loads((folder/'selection.json').read_text());oof=read(folder/'inner_oof.csv')
    assert r['train_ids']==list(expected_ids)==list(oof.mbid)
    y=task.df.loc[list(expected_ids),task.metadata.target].to_numpy();cl=task.metadata.task_type=='classification'
    np.testing.assert_array_equal(oof.target,np.asarray(y,float))
    inner=[(np.array(s['train']),np.array(s['validation'])) for s in r['inner']]
    for j,(a,b) in enumerate(inner):
        assert set(a).isdisjoint(b) and set(a)|set(b)==set(range(len(y)))
        assert (oof.inner_fold.iloc[b]==j).all()
    pool={k:oof[k].to_numpy() for k in r['search']}
    for key,search in r['search'].items():
        for candidate in search:
            assert len(candidate['losses'])==3
            np.testing.assert_allclose(candidate['mean'],np.mean(candidate['losses']))
        winner=min(search,key=lambda c:c['mean'])
        assert winner['parameters']==r['parameters'][key]
        np.testing.assert_allclose(mean_inner_loss(y,pool[key],inner,cl),winner['mean'])
    c={k:v for k,v in pool.items() if k.startswith('C:')}
    recipe,candidates=select(y,c,inner,cl)
    assert recipe==r['old_recipe'] and candidates==r['old_candidates']
    if 'new_recipe' in r:
        d={k:v for k,v in pool.items() if k.startswith('D:')}
        new,new_candidates=select(y,d,inner,cl)
        decision=gate(y,combine(c,recipe),combine(d,new),inner,cl)
        assert decision==r['gate'] and new==r['new_recipe'] and new_candidates==r['new_candidates']
        if decision['use_new']:recipe=new
    assert recipe==r['recipe']
    return r


def audit():
    register();preserved={};rows=[];replays=0;inner_fits=0;refits=0
    for version,folder in [('V1',V1),('V2',V2),('V3',V3),('V4',V4)]:
        m=json.loads((folder/'delivery_manifest.json').read_text())
        for p,h in m.items():assert sha(ROOT/p)==h,p
        preserved[version]=len(m)
    for name in TASKS:
        task,matrices,info,rep=data(name);cl=task.metadata.task_type=='classification'
        for stage in ['conditional','reselection','conformal']:
            if stage=='reselection' and name not in TASKS[:2]:continue
            if stage=='conformal' and name not in TASKS[:3]:continue
            folders=sorted((OUT/name/stage).glob('*'))
            assert len(folders)==(5 if stage=='conformal' else 10),(name,stage,len(folders))
            for folder in folders:
                r=json.loads((folder/'result.json').read_text());fold=r['fold']
                train,ytrain=task.get_train_and_val_data(fold);test,target=task.get_test_data(fold,include_target=True)
                p=read(folder/'predictions.csv');assert list(p.mbid)==list(test.index)
                np.testing.assert_array_equal(p.target,np.asarray(target,float))
                s=scores(target,p.prediction.to_numpy(),cl)
                for k,v in s.items():np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-12,atol=1e-12)
                bundle=joblib.load(folder/'model.joblib')
                actual=bundle.predict(take(matrices,task.df.index.get_indexer(test.index)))
                np.testing.assert_allclose(actual,p.prediction,rtol=1e-11,atol=1e-8);replays+=1
                if stage=='conditional':
                    recipe,parameters=historical_recipe(name,rep,fold)
                    assert recipe==r['recipe'] and parameters==r['parameters'] and r['train_ids']==list(train.index)
                    refits+=r['refits']
                else:
                    ids=train.index if stage=='reselection' else r['proper_train_ids']
                    sr=audit_selection(folder,task,ids);inner_fits+=sr['inner_fits'];refits+=sr['refits']
                    if stage=='conformal':
                        a,b,c=map(set,[r['proper_train_ids'],r['calibration_ids'],list(test.index)])
                        assert a.isdisjoint(b) and (a|b)==set(train.index) and (a|b).isdisjoint(c)
                        assert set(info.loc[list(a),'formula']).isdisjoint(info.loc[list(b),'formula'])
                        cal=read(folder/'calibration.csv');assert list(cal.mbid)==r['calibration_ids']
                        np.testing.assert_array_equal(cal.target,task.df.loc[cal.mbid,task.metadata.target].to_numpy())
                        cal_actual=bundle.predict(take(matrices,task.df.index.get_indexer(cal.mbid)))
                        np.testing.assert_allclose(cal_actual,cal.prediction,rtol=1e-11,atol=1e-8)
                        np.testing.assert_allclose(cal.residual,np.abs(cal.target-cal.prediction))
                        rank=int(np.ceil((len(cal)+1)*.9));q=float(np.sort(cal.residual)[rank-1])
                        np.testing.assert_allclose(q,r['radius'])
                        intervals=read(folder/'intervals.csv');assert list(intervals.mbid)==list(test.index)
                        np.testing.assert_allclose(intervals.lower,p.prediction-q);np.testing.assert_allclose(intervals.upper,p.prediction+q)
                        coverage=float(((p.target>=intervals.lower)&(p.target<=intervals.upper)).mean())
                        np.testing.assert_allclose(coverage,r['coverage'])
                rows.append(dict(task=name,stage=stage,seed=r.get('seed',42),fold=fold,**s,coverage=r.get('coverage'),width=r.get('mean_width')))
        eq=json.loads((OUT/name/'equivalence/result.json').read_text());assert eq['pass_all']
        train,_=task.get_train_and_val_data(0);sr=audit_selection(OUT/name/'equivalence',task,train.index)
        inner_fits+=sr['inner_fits'];refits+=sr['refits']
        print('FINAL AUDIT',name,flush=True)
    # The comparison with the same candidate pool is reconstructed from OOF.
    for name in ['matbench_steels','matbench_jdft2d','matbench_glass']:
        task,matrices,info,rep=data(name);cl=task.metadata.task_type=='classification'
        from topic1.advanced import inner_folds
        for fold in range(5):
            train,y=task.get_train_and_val_data(fold);test,target=task.get_test_data(fold,include_target=True)
            inner,_=inner_folds(np.asarray(y),info.loc[train.index,'formula'].to_numpy(),cl,fold)
            for letter in ['C','D']:
                oofs={};predictions={}
                for f in ['RF','ET','CatBoost']:
                    folder=V4/name/'runs/D'/f if letter=='D' else (V3 if f=='CatBoost' else V2)/name/'runs'/rep/f
                    oofs[f]=read(folder/f'inner_oof_{fold}.csv').prediction.to_numpy()
                    predictions[f]=read(folder/f'fold_{fold}.csv').prediction.to_numpy()
                recipe,_=select(np.asarray(y),oofs,inner,cl)
                folder=OUT/name/'common_pool'/f'{letter}_fold_{fold}';r=json.loads((folder/'result.json').read_text());p=read(folder/'predictions.csv')
                assert recipe==r['recipe'] and list(p.mbid)==list(test.index)
                np.testing.assert_array_equal(p.prediction,combine(predictions,recipe))
                np.testing.assert_array_equal(p.target,np.asarray(target,float))
                s=scores(target,p.prediction.to_numpy(),cl)
                for k,v in s.items():np.testing.assert_allclose(v,r['metrics'][k])
                rows.append(dict(task=name,stage='common_pool_'+letter,seed=42,fold=fold,**s))
    # External provenance, composition exclusion, raw labels and model replay.
    folder=OUT/'external';r=json.loads((folder/'result.json').read_text());p=read(folder/'predictions.csv')
    raw=OUT/'datasets/glass_ternary_hipt.json.gz';assert sha(raw)==r['raw_sha256']
    df=load_dataset('glass_ternary_hipt',data_home=str(OUT/'datasets'),download_if_missing=False)
    assert list(p.id)==[f'hipt-{i:05d}' for i in range(len(df))]
    np.testing.assert_array_equal(p.target,df.gfa.astype(int))
    task,matrices,info,rep=data('matbench_glass')
    sr=audit_selection(folder/'full_training',task,task.df.index);inner_fits+=sr['inner_fits'];refits+=sr['refits']
    bundle=joblib.load(folder/'full_training/model.joblib')
    external_matrices={k:np.load(folder/f'features_{k}.npy') for k in ['C','D']}
    np.testing.assert_allclose(bundle.predict(external_matrices),p.prediction,rtol=1e-11,atol=1e-8);replays+=1
    keys=df.formula.map(composition_key);overlap=keys.isin(set(info.formula.map(composition_key))).to_numpy()
    a=external_matrices['C'][:,:118];b=matrices['C'][:,:118]
    nearest=np.concatenate([cdist(a[i:i+256],b,'cityblock').min(axis=1) for i in range(0,len(a),256)])
    np.testing.assert_allclose(p.min_fraction_L1_distance,nearest)
    primary=(~overlap)&(nearest>.02)
    np.testing.assert_array_equal(primary,p.primary)
    np.testing.assert_array_equal(overlap,p.exact_training_overlap)
    for name,mask in [('primary',primary),('nonexact_overlap_removed',~overlap)]:
        s=scores(p.target[mask],p.prediction[mask],True)
        for k,v in s.items():np.testing.assert_allclose(v,r[name][k])
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'validation_fold_metrics.csv',index=False)
    summary=frame.groupby(['task','stage','seed']).agg({k:['mean','std'] for k in ['mae','roc_auc','coverage','width']}).reset_index()
    summary.columns=['_'.join(c).strip('_') if isinstance(c,tuple) else c for c in summary.columns]
    summary.to_csv(OUT/'validation_summary.csv',index=False)
    result=dict(pass_all=True,historical_files_preserved=preserved,official_tasks=TASKS,outer_folds=5,
                validated_outer_prediction_files=len(rows),replayed_bundles=replays,inner_candidate_fits=inner_fits,final_component_refits=refits,
                total_new_fits=inner_fits+refits,external_primary_n=int(primary.sum()),source_checksum_verified=True,
                conformal_selection_calibration_separated=True,common_pool_reconstructed=True,
                limitations=['Full reselection seed experiment covers steels and jdft2d, conditional refit covers five tasks.',
                             'External validation covers glass under changed processing conditions, not all five properties.',
                             'Published strong-model results are rescored archives, not matched-budget local retraining.'])
    write_json(OUT/'requirement_audit.json',result)
    print('AUDIT COMPLETE',result,flush=True)
    return frame,result


if __name__=='__main__':audit()

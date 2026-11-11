"""External sputtering data evaluated once, after protocol/model freezing."""
import json
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from pymatgen.core import Composition
from matminer.datasets import load_dataset
from sklearn.metrics import brier_score_loss,roc_auc_score,balanced_accuracy_score
from topic1.run import features,sha,write_json,scores
from topic1.physics_features import alloy_features
from topic1.final_validation import OUT,register,data,store_fit
from topic1.final_models import fit_procedure


def composition_key(formula):
    c=Composition(formula).fractional_composition.get_el_amt_dict()
    return ';'.join(f'{k}:{v:.8f}' for k,v in sorted(c.items()))


def external():
    register();folder=OUT/'external';folder.mkdir(parents=True,exist_ok=True)
    if (folder/'result.json').exists():return
    task,matrices,info,rep=data('matbench_glass')
    y=task.df[task.metadata.target].to_numpy()
    model_folder=folder/'full_training'
    if (model_folder/'selection.json').exists():
        bundle=joblib.load(model_folder/'model.joblib')
    else:
        bundle,record,oofs=fit_procedure(matrices,y,info.formula.to_numpy(),True,0,42)
        store_fit(model_folder,bundle,record,oofs,task.df.index,y)
    raw=OUT/'datasets/glass_ternary_hipt.json.gz'
    expected=json.loads((OUT/'protocol.json').read_text())['external']['sha256'];assert sha(raw)==expected
    # The external table is loaded only after the source-trained model is saved.
    df=load_dataset('glass_ternary_hipt',data_home=str(OUT/'datasets'),download_if_missing=False)
    df.index=[f'hipt-{i:05d}' for i in range(len(df))]
    keys=df.formula.map(composition_key);train_keys=set(info.formula.map(composition_key))
    matrix_path=folder/'features_C.npy'
    if matrix_path.exists():
        c=np.load(matrix_path);d=np.load(folder/'features_D.npy');a=c[:,:118]
    else:
        featurized,names,metadata=features(df.formula)
        extra,labels,issues=alloy_features(df.formula);extra[~np.isfinite(extra)]=np.nan
        c=featurized['C'];d=np.column_stack([c,extra]);a=featurized['A'][:,:118]
        np.save(matrix_path,c);np.save(folder/'features_D.npy',d)
        write_json(folder/'features.json',dict(ids=list(df.index),names_C=names['C'],names_D=names['C']+labels,issues=issues))
    train_a=matrices['C'][:,:118]
    nearest=np.concatenate([cdist(a[i:i+256],train_a,metric='cityblock').min(axis=1) for i in range(0,len(a),256)])
    exact=keys.isin(train_keys).to_numpy();primary=(~exact)&(nearest>.02)
    p=bundle.predict({'C':c,'D':d})
    assert np.isfinite(p).all() and ((p>=0)&(p<=1)).all()
    # Only now are external targets used, for scoring; no subsequent fitting.
    target=df.gfa.to_numpy(dtype=bool)
    frame=pd.DataFrame(dict(id=df.index,formula=df.formula,composition_key=keys,system=df.system,processing=df.processing,
                            exact_training_overlap=exact,min_fraction_L1_distance=nearest,primary=primary,target=target.astype(int),prediction=p))
    frame.to_csv(folder/'predictions.csv',index=False)
    def evaluate(mask):
        y0=target[mask];p0=p[mask]
        if not len(y0):return dict(n=0)
        metrics=scores(y0,p0,True) if len(np.unique(y0))==2 else dict(roc_auc=None,balanced_accuracy=None)
        return dict(n=len(y0),positive_fraction=float(y0.mean()),brier=float(brier_score_loss(y0,p0)),**metrics)
    results=dict(primary=evaluate(primary),nonexact_overlap_removed=evaluate(~exact),all_rows_diagnostic=evaluate(np.ones(len(df),bool)),
                 n_total=len(df),n_exact_overlap=int(exact.sum()),n_nonexact_near_overlap=int(((~exact)&(nearest<=.02)).sum()),
                 raw_sha256=sha(raw),protocol_sha256=sha(OUT/'protocol.json'),model_sha256=sha(model_folder/'model.joblib'),
                 per_system={str(s):evaluate(primary&(df.system.to_numpy()==s)) for s in sorted(df.system.unique())},
                 interpretation='Cross-source and processing-domain shift. No external labels enter fitting, tuning, thresholding or calibration.')
    # Composition-cluster bootstrap conditions on this trained model and source.
    sub=frame[frame.primary].reset_index(drop=True);clusters=[g.index.to_numpy() for _,g in sub.groupby('composition_key',sort=True)]
    rng=np.random.default_rng(20261110);auc=[]
    for _ in range(1000):
        indices=np.concatenate([clusters[i] for i in rng.integers(0,len(clusters),len(clusters))])
        boot=sub.iloc[indices]
        if boot.target.nunique()==2:auc.append(float(roc_auc_score(boot.target,boot.prediction)))
    results['conditional_cluster_bootstrap_auc_95pct']=np.quantile(auc,[.025,.975]).tolist()
    results['bootstrap_note']='Composition-cluster bootstrap; conditional on four observed chemical systems and fitted model, not a population-level guarantee.'
    write_json(folder/'result.json',results)
    print('EXTERNAL COMPLETE',results['primary'],flush=True)


if __name__=='__main__':external()

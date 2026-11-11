"""Assemble one verified evidence snapshot for the report and presentation."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss,roc_curve
from topic1.run import ROOT,TASKS,write_json
from topic1.final_validation import OUT,V1,V2,V3,V4,read,data,historical_recipe
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

NAMES={'matbench_steels':'钢材屈服强度','matbench_jdft2d':'二维材料剥离能','matbench_expt_gap':'实验带隙','matbench_expt_is_metal':'金属性','matbench_glass':'玻璃形成能力'}
UNITS=['MPa','meV/atom','eV','ROC-AUC','ROC-AUC']


def prepare():
    audit=json.loads((OUT/'requirement_audit.json').read_text());assert audit['pass_all']
    selected=read(V4/'summary.csv');baseline=read(V2/'summary.csv');metrics=read(OUT/'validation_fold_metrics.csv')
    references=read(V2/'reference_fold_metrics.csv');learning=read(V1/'learning_curve.csv')
    payload=dict(audit=audit,tasks=[],learning={},stability=[],conformal=[],common_pool=[],representation=[],references=[])
    for i,name in enumerate(TASKS):
        metric='roc_auc' if i>=3 else 'mae';rep='C_structure' if i==1 else 'C'
        current=selected[(selected.task==name)&(selected.representation=='V4')].iloc[0]
        task,matrices,info,_=data(name)
        entry=dict(id=name,label=NAMES[name],n=len(task.df),unit=UNITS[i],metric=metric,representation=rep,
                   mean=float(current['mean']),std=float(current['std']),baselines={})
        for model in ['Linear','RF','ET','XGB']:
            r=baseline[(baseline.task==name)&(baseline.representation==rep)&(baseline.model==model)].iloc[0]
            entry['baselines'][model]=dict(mean=float(r[f'{metric}_mean']),std=float(r[f'{metric}_std']))
        payload['tasks'].append(entry)
        ab=baseline[(baseline.task==name)&(baseline.model=='ET')&baseline.representation.isin(['A','B','C'])]
        payload['representation'].append(dict(task=name,label=NAMES[name],metric=metric,values={r.representation:float(getattr(r,f'{metric}_mean')) for r in ab.itertuples()}))
        for stage in ['conditional','reselection']:
            sub=metrics[(metrics.task==name)&(metrics.stage==stage)]
            if sub.empty:continue
            vals={42:float(current['mean'])}
            for seed,g in sub.groupby('seed'):vals[int(seed)]=float(g[metric].mean())
            changed=None
            if stage=='reselection':
                count=0
                for folder in (OUT/name/stage).glob('*'):
                    r=json.loads((folder/'result.json').read_text());s=json.loads((folder/'selection.json').read_text())
                    old,_=historical_recipe(name,rep,r['fold'])
                    if sorted(old['members'])!=sorted(s['recipe']['members']) or old['operation']!=s['recipe']['operation']:count+=1
                changed=count
            payload['stability'].append(dict(task=name,label=NAMES[name],stage=stage,metric=metric,seed_means=vals,
                                              minimum=min(vals.values()),maximum=max(vals.values()),recipe_changes=changed))
        if i<3:
            frames=[];widths=[];proper=[]
            for folder in sorted((OUT/name/'conformal').glob('*')):
                p=read(folder/'predictions.csv');interval=read(folder/'intervals.csv');r=json.loads((folder/'result.json').read_text())
                p['covered']=(p.target>=interval.lower)&(p.target<=interval.upper);p['width']=interval.upper-interval.lower;frames.append(p)
                proper.append(len(r['proper_train_ids']))
            allp=pd.concat(frames);s=metrics[(metrics.task==name)&(metrics.stage=='conformal')]
            payload['conformal'].append(dict(task=name,label=NAMES[name],coverage=float(allp.covered.mean()),width=float(allp.width.mean()),
                                              fold_min=float(s.coverage.min()),fold_max=float(s.coverage.max()),mae=float((allp.target-allp.prediction).abs().mean()),proper_train_mean=float(np.mean(proper))))
        for model in ['MODNet','CrabNet']:
            sub=references[(references.task==name)&(references.model==model)]
            if len(sub):payload['references'].append(dict(task=name,model=model,mean=float(sub[metric].mean())))
    for name,g in learning.groupby('task'):
        payload['learning'][name]=[dict(fraction=float(f),mean=float(a.mae.mean()),std=float(a.mae.std())) for f,a in g.groupby('fraction')]
    for name in ['matbench_steels','matbench_jdft2d','matbench_glass']:
        metric='roc_auc' if name.endswith('glass') else 'mae'
        values={rep:float(metrics[(metrics.task==name)&(metrics.stage=='common_pool_'+rep)][metric].mean()) for rep in ['C','D']}
        payload['common_pool'].append(dict(task=name,label=NAMES[name],metric=metric,**values))
    external=json.loads((OUT/'external/result.json').read_text());p=read(OUT/'external/predictions.csv');sub=p[p.primary]
    task,_,_,_=data('matbench_glass');prior=float(task.df[task.metadata.target].mean())
    external['source_positive_fraction']=prior
    external['source_prior_dummy_brier']=float(brier_score_loss(sub.target,np.full(len(sub),prior)))
    external['source_prior_dummy_accuracy']=float((sub.target==(prior>.5)).mean())
    external['source_prior_dummy_note']='Post-hoc diagnostic reference with source-only class frequency; no external fitting or threshold selection.'
    payload['external']=external
    payload['physics_controls']=read(V4/'matched_representation_controls.csv').to_dict('records')
    payload['ridge_diagnostic']=json.loads((V1/'ridge_failure_diagnostic.json').read_text())
    payload['sources']=[
        dict(id=1,title='SDSC6020 Topics 课程说明',url='用户提供的 Topics.pdf，第 2–4 页'),
        dict(id=2,title='Dunn et al. Benchmarking materials property prediction methods',url='https://doi.org/10.1038/s41524-020-00406-3'),
        dict(id=3,title='Matbench official repository, fixed commit 07e50bc',url='https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa'),
        dict(id=4,title='Ward et al. Matminer An open source toolkit for materials data mining',url='https://doi.org/10.1016/j.commatsci.2018.05.018'),
        dict(id=5,title='Matminer composition and site featurizer documentation',url='https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html'),
        dict(id=6,title='Ren et al. Accelerated discovery of metallic glasses',url='https://pmc.ncbi.nlm.nih.gov/articles/PMC5898831/'),
        dict(id=7,title='Angelopoulos and Bates A Gentle Introduction to Conformal Prediction',url='https://arxiv.org/abs/2107.07511'),
        dict(id=8,title='MODNet published Matbench benchmark archive',url='https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa/benchmarks/matbench_v0.1_modnet_v0.1.12'),
        dict(id=9,title='CrabNet published Matbench benchmark archive',url='https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa/benchmarks/matbench_v0.1_CrabNet')]
    write_json(OUT/'evidence.json',payload)
    plots(payload,p)
    return payload


def plots(d,external_predictions):
    directory=OUT/'figures';directory.mkdir(exist_ok=True)
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,3.5))
    for ax,name,unit in zip(axes,['matbench_steels','matbench_expt_gap'],['MPa','eV']):
        r=d['learning'][name];ax.errorbar([x['fraction']*100 for x in r],[x['mean'] for x in r],yerr=[x['std'] for x in r],marker='o',capsize=3,color='#087f8c')
        ax.set_title(name.replace('matbench_',''));ax.set_xlabel('Training fraction (%)');ax.set_ylabel(f'MAE ({unit})');ax.grid(alpha=.15)
    fig.suptitle('Fixed RF baseline on C; bars show outer-fold standard deviation');fig.tight_layout();fig.savefig(directory/'learning.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.6))
    names=[x['task'].replace('matbench_','') for x in d['conformal']]
    axes[0].bar(names,[x['coverage'] for x in d['conformal']],color='#087f8c');axes[0].axhline(.9,color='#b75b3f',ls='--',label='Nominal 90%');axes[0].set_ylim(0,1);axes[0].set_ylabel('Empirical held-out coverage');axes[0].legend()
    sub=external_predictions[external_predictions.primary]
    fpr,tpr,_=roc_curve(sub.target,sub.prediction);axes[1].plot(fpr,tpr,color='#087f8c',label=f"External AUC {d['external']['primary']['roc_auc']:.3f}")
    axes[1].plot([0,1],[0,1],'--',color='gray');axes[1].set_xlabel('False positive rate');axes[1].set_ylabel('True positive rate');axes[1].legend()
    fig.tight_layout();fig.savefig(directory/'validation.png',dpi=180);plt.close(fig)
    fig,ax=plt.subplots(figsize=(8,3.6))
    systems=[];values=[]
    for system,r in d['external']['per_system'].items():
        if r.get('roc_auc') is not None:systems.append(system);values.append(r['roc_auc'])
    ax.bar(systems,values,color=['#087f8c','#527c83','#b75b3f']);ax.axhline(.5,color='gray',ls='--');ax.set_ylim(0,1);ax.set_ylabel('External ROC-AUC');ax.set_title('System-specific transfer; FeTiNb has only negative labels')
    fig.tight_layout();fig.savefig(directory/'external_systems.png',dpi=180);plt.close(fig)


if __name__=='__main__':prepare()

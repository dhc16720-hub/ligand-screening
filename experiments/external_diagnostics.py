"""Descriptive failure analysis of already evaluated external glass predictions; no fitting."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score, brier_score_loss, average_precision_score
from matbench.task import MatbenchTask
from topic1.run import ROOT, write_json
from experiments.small_data import sha

OUT=ROOT/'outputs/topic1_external_diagnostics'
SOURCE=ROOT/'outputs/topic1_final/external'


def metrics(target, prediction):
    y=np.asarray(target,int);p=np.asarray(prediction,float)
    if not len(y):return dict(n=0)
    if not np.isfinite(p).all() or ((p<0)|(p>1)).any():raise ValueError('Invalid probabilities')
    tn,fp,fn,tp=confusion_matrix(y,p>.5,labels=[0,1]).ravel()
    two=len(np.unique(y))==2
    return dict(n=len(y),positive_fraction=float(y.mean()),mean_probability=float(p.mean()),
                predicted_positive_fraction=float((p>.5).mean()),tn=int(tn),fp=int(fp),fn=int(fn),tp=int(tp),
                roc_auc=float(roc_auc_score(y,p)) if two else None,
                average_precision=float(average_precision_score(y,p)) if two else None,
                balanced_accuracy=float((tp/(tp+fn)+tn/(tn+fp))/2) if two else None,
                accuracy=float((tp+tn)/len(y)),brier=float(brier_score_loss(y,p)))


def reliability(y,p):
    y=np.asarray(y,int);p=np.asarray(p,float)
    if not len(y) or not np.isfinite(p).all() or ((p<0)|(p>1)).any(): raise ValueError('Invalid reliability input')
    bins=np.minimum((p*10).astype(int),9)
    rows=[]
    for i in range(10):
        mask=bins==i
        rows.append(dict(bin=i,left=i/10,right=(i+1)/10,n=int(mask.sum()),
                         mean_probability=float(p[mask].mean()) if mask.any() else None,
                         observed_fraction=float(y[mask].mean()) if mask.any() else None))
    ece=sum(r['n']/len(y)*abs(r['mean_probability']-r['observed_fraction']) for r in rows if r['n'])
    return rows,ece


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    source_files=[SOURCE/'predictions.csv',SOURCE/'result.json',SOURCE/'full_training/selection.json']
    before={str(p.relative_to(ROOT)):sha(p) for p in source_files}
    frame=pd.read_csv(SOURCE/'predictions.csv');primary=frame[frame.primary].copy()
    frozen=json.loads((SOURCE/'result.json').read_text())
    base=metrics(primary.target,primary.prediction)
    for key in ['n','positive_fraction','roc_auc','balanced_accuracy','accuracy','brier']:
        np.testing.assert_allclose(base[key],frozen['primary'][key],rtol=1e-12,atol=1e-12)
    task=MatbenchTask('matbench_glass');source_y=task.df[task.metadata.target].to_numpy(dtype=int)
    source_prior=float(source_y.mean())
    selections=json.loads((SOURCE/'full_training/selection.json').read_text())
    if selections['train_ids']!=list(task.df.index):raise RuntimeError('Source training identity mismatch')
    groups=[('ALL_PRIMARY',primary),*[(str(s),g) for s,g in primary.groupby('system',sort=True)]]
    rows=[];bins=[]
    for system,group in groups:
        diagnostic,ece=reliability(group.target,group.prediction)
        bins.extend(dict(system=system,**row) for row in diagnostic)
        for name,p in [('frozen_model',group.prediction.to_numpy()),
                       ('source_prevalence',np.full(len(group),source_prior)),
                       ('constant_0.5',np.full(len(group),.5)),('constant_0',np.zeros(len(group)))]:
            rows.append(dict(system=system,model=name,**metrics(group.target,p),
                             ece_10_equal_width_bins=ece if name=='frozen_model' else None))
    table=pd.DataFrame(rows);table.to_csv(OUT/'system_metrics.csv',index=False)
    pd.DataFrame(bins).to_csv(OUT/'reliability_bins.csv',index=False)
    distances=[]
    for low,high in zip([.02,.1,.2,.5],[.1,.2,.5,2.]):
        sub=primary[(primary.min_fraction_L1_distance>low)&(primary.min_fraction_L1_distance<=high)]
        distances.append(dict(distance_lower=low,distance_upper=high,**metrics(sub.target,sub.prediction)))
    pd.DataFrame(distances).to_csv(OUT/'distance_metrics.csv',index=False)
    if before!={str(p.relative_to(ROOT)):sha(p) for p in source_files}:raise RuntimeError('Frozen external evidence changed')
    write_json(OUT/'audit.json',dict(pass_all=True,primary_rows=len(primary),source_training_rows=len(source_y),
        source_positive_fraction=source_prior,source_hashes=before,code_sha256=sha(__file__),new_fits=0,
        threshold='prediction > 0.5, same as frozen scoring',systems=len(groups)-1,
        limitations=['Post-hoc descriptive analysis of previously inspected external data; not a new blind evaluation.',
                     'No calibration or threshold selection on external labels.',
                     'Constant baselines are diagnostic comparisons, not tuned models.',
                     'Single-class system AUC/AP/balanced accuracy undefined; recorded as null.',
                     'ECE depends on ten equal-width bins; no population-level calibration guarantee.']))
    figures(primary,table,pd.DataFrame(bins))
    write_report(table,source_prior)
    print(table[table.model=='frozen_model'].to_string(index=False),flush=True)


def figures(primary,table,bins):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4.2),layout='constrained')
    for system,group in bins.groupby('system',sort=False):
        nonempty=group[group.n>0]
        axes[0].plot(nonempty.mean_probability,nonempty.observed_fraction,'o-',label=system,alpha=.85)
    axes[0].plot([0,1],[0,1],'--',color='gray',linewidth=1)
    axes[0].set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title='External reliability (fixed bins)')
    axes[0].legend(fontsize=7,loc='upper left')
    for target,color in [(0,'#3876a6'),(1,'#b56a36')]:
        axes[1].hist(primary.loc[primary.target==target,'prediction'],bins=np.linspace(0,1,21),alpha=.6,label=f'Observed class {target}',color=color)
    axes[1].axvline(.5,color='black',linestyle='--',linewidth=1)
    axes[1].set(xlabel='Predicted positive probability',ylabel='Number of measurements',title='Probability distribution / fixed threshold')
    axes[1].legend(fontsize=8)
    fig.savefig(OUT/'reliability.png',dpi=180);fig.savefig(OUT/'reliability.pdf');plt.close(fig)
    subset=table[(table.model=='frozen_model')&(table.system!='ALL_PRIMARY')]
    fig,axes=plt.subplots(1,4,figsize=(12,3.8),layout='constrained')
    for ax,row in zip(axes,subset.itertuples()):
        matrix=np.array([[row.tn,row.fp],[row.fn,row.tp]])
        ax.imshow(matrix,cmap='Blues',vmin=0,vmax=max(1,matrix.max()))
        for i,j in itertools_product():
            ax.text(j,i,str(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>.55*matrix.max() else 'black')
        ax.set(xticks=[0,1],yticks=[0,1],xlabel='Predicted class',ylabel='Observed class',title=row.system)
    fig.savefig(OUT/'confusion_matrices.png',dpi=180,bbox_inches='tight',pad_inches=.15)
    fig.savefig(OUT/'confusion_matrices.pdf',bbox_inches='tight',pad_inches=.15);plt.close(fig)


def itertools_product():
    return [(0,0),(0,1),(1,0),(1,1)]


def write_report(table,source_prior):
    from experiments.study_report import markdown_table
    model=table[table.model=='frozen_model'][['system','n','positive_fraction','mean_probability',
          'predicted_positive_fraction','roc_auc','balanced_accuracy','brier']]
    baseline=table[table.system=='ALL_PRIMARY'][['model','roc_auc','accuracy','brier']]
    primary=table[(table.system=='ALL_PRIMARY')&(table.model=='frozen_model')].iloc[0]
    text=f'''# 外部玻璃数据：为什么排序尚可，分类却失效

这是对已经评估过的外部数据的事后描述性分析，不是新的盲测。模型、预测、样本排除规则和阈值沿用原冻结实验；没有用外部标签训练、校准或挑选阈值。

## 核心发现

主集合 {int(primary.n):,} 条，真实正类 {primary.positive_fraction:.2%}，模型平均概率 {primary.mean_probability:.2%}，预测正类 {primary.predicted_positive_fraction:.2%}。混淆矩阵为 TN={int(primary.tn)}、FP={int(primary.fp)}、FN={int(primary.fn)}、TP={int(primary.tp)}：高召回伴随大量假阳性。

源训练集正类占 {source_prior:.2%}，外部主集合占 {primary.positive_fraction:.2%}，且制备条件发生变化。这些是可观察到的分布差异，不能据此证明某个因素单独导致失效。后续应核对标签语义、制备条件和特征覆盖。

## 分体系结果

{markdown_table(model)}

CoFeZr、CoTiZr、CoVZr 三个含正负样本的体系都被全部判为正类，balanced accuracy 均为 0.5。CoVZr 的 AUC 点估计约 0.474，整体 AUC 不能代表每个体系；这里不据此宣称统计上显著劣于随机。FeTiNb 只有负类，AUC、average precision 和 balanced accuracy 均为未定义。

## 概率参照

{markdown_table(baseline)}

Brier 越低越好。原模型高于源训练集正类比例的恒定预测，也高于恒定 0.5 的 0.25。当前外部概率不可靠，不能因为 AUC 较高就解释为可靠的成功概率。恒定参照的多数类 accuracy 高也不代表有辨别能力，其 AUC 为 0.5。阈值沿用 prediction > 0.5，恰好 0.5 判为负类。

## 图表与距离分组

![可靠性与概率分布](reliability.png)

左图比较十个等宽概率箱中的平均预测与实际正类比例，样本数及空箱保存在 reliability_bins.csv。ECE 对分箱方式敏感，不作为跨总体保证。

![分体系混淆矩阵](confusion_matrices.png)

组成 L1 距离分组见 distance_metrics.csv。距离、材料体系和制备条件可能共同变化，不把分组差异解释为单一因果效应。

## 后续验证边界

若开发校准或阈值策略，应在源训练数据内层拟合。当前外部集已被查看，后续改进属于探索性研究；独立迁移结论需要新的留出来源。现有失败结果保留，不用后续调参覆盖。
'''
    (OUT/'REPORT.md').write_text(text)


if __name__=='__main__':main()

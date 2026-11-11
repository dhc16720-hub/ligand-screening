"""Readable figures and Chinese interpretations from verified additive studies."""
import json
from pathlib import Path
import pandas as pd
from topic1.run import ROOT, write_json
from experiments.small_data import OUT, TASKS, sha

NAMES=dict(zip(TASKS,['Steel yield strength','2D exfoliation energy','Experimental band gap','Metal classification','Glass formation']))
UNITS=dict(zip(TASKS,['MAE (MPa)','MAE (meV/atom)','MAE (eV)','ROC-AUC','ROC-AUC']))


def markdown_table(frame):
    def format_cell(x):
        if pd.isna(x):return '—'
        if isinstance(x,float):return f'{x:.5f}'
        return str(x)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |','| '+' | '.join(['---']*len(frame.columns))+' |',
                      *['| '+' | '.join(map(format_cell,row))+' |' for row in frame.itertuples(index=False,name=None)]])


def build():
    manifest=OUT/'delivery_manifest.json'
    if manifest.exists():
        import sys
        sys.path.insert(0,str(ROOT/'scripts'))
        from reproduce_topic1 import check_manifest
        check_manifest(ROOT,manifest)
        print('STUDY REPORT ALREADY FROZEN; original figures and report preserved',flush=True)
        return
    audit=json.loads((OUT/'audit.json').read_text())
    if audit.get('pass_all') is not True or audit.get('replayed_models')!=1050:raise RuntimeError('Complete scientific audit required')
    summary=pd.read_csv(OUT/'summary.csv');paired=pd.read_csv(OUT/'paired_comparisons.csv')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for model in ['RF','ET']:
        fig,axes=plt.subplots(2,3,figsize=(13,7.6),layout='constrained')
        for ax,task in zip(axes.flat,TASKS):
            for rep,color in [('A','#376a9c'),('B','#c07837'),('C','#43856a')]:
                points=summary[(summary.task==task)&(summary.model==model)&(summary.representation==rep)].sort_values('fraction')
                ax.errorbar(points.fraction*100,points['mean'],yerr=points.fold_sd,color=color,label=rep,marker='o',capsize=3)
            ax.set(title=NAMES[task],xlabel='Official training fold used (%)',ylabel=UNITS[task],xticks=[20,50,100])
            ax.grid(alpha=.18);ax.legend(title='Representation',fontsize=8,title_fontsize=8)
        axes.flat[-1].axis('off')
        axes.flat[-1].text(.02,.9,f'{model}: 100 trees, fixed seed 42\n\nA: element fractions + composition statistics\nB: Magpie attribute statistics\nC: concatenation of A and B\n\nSame subsets for every representation.\n3 subsets at 20% and 50%; full point once.\nAverage subsets within each official fold first.\nError bars: fold SD, not confidence intervals.\nNo structural features in this study.',va='top',fontsize=10,linespacing=1.6)
        fig.suptitle(f'{model} | Representation effects across training sizes',fontsize=16)
        fig.savefig(OUT/f'learning_{model}.png',dpi=180)
        fig.savefig(OUT/f'learning_{model}.pdf');plt.close(fig)
    selected=paired[paired.comparison=='C-A'][['task','model','fraction','mean_improvement','fold_sd','improved_folds']]
    interaction=pd.read_csv(OUT/'low_data_interaction.csv');interaction=interaction[interaction.comparison=='C-A']
    lines=['# 小数据 × 科学表示：配对对照实验','',
           '五个 Matbench 任务使用官方五折。固定 RF/ET 各 100 棵树，对比 A（元素比例、元素数、组成熵与比例平方和）、B（Magpie 属性统计）、C（A+B）。每个官方训练折取 20%、50%、100%；前两档各三次随机抽样。相同折、比例、抽样种子在所有模型和表示间共享样本 ID，20% 子集嵌套于 50%。全量点只拟合一次。',
           '', '本轮为既有基准上的探索性补充，未调参、未用测试结果选择新的主方法，也未改变老师要求的官方全量五折主结果。这里仅研究组成表示，二维材料的结构表示另见原实验。',
           '', '## 实测与审计','',
           '共完成 1,050 次拟合及 1,050 个保存模型重放。审计逐项检查官方测试 ID、训练子集、目标、预测、模型预算以及训练内补缺和标准化。先平均每个官方折内的抽样重复，再统计五折均值和标准差。重复抽样并非独立测试集，折间标准差也不是置信区间。',
           '', '## C 相对 A 的收益','',
           '下表统一规定正值代表 C 较好：回归为 MAE(A) − MAE(C)，分类为 AUC(C) − AUC(A)。量纲按任务分别解释，不能把不同任务的数值直接求平均。`improved_folds` 是五折中 C 较好的折数，是描述性证据，不是显著性检验。',
           '', markdown_table(selected),'','## 小样本是否更受益','',
           '下表末列为“20% 时的 C−A 收益减去 100% 时的收益”。正值说明本次配对实验中 C 的优势在小样本档更大；负值说明优势更小或退步更明显。该交互量不证明科学描述符普遍能解决小数据问题。',
           '',markdown_table(interaction[['task','model','0.2','1.0','low_data_gain_minus_full']]),'',
           '## 学习曲线','','![RF 学习曲线](learning_RF.png)','','![ET 学习曲线](learning_ET.png)','',
           '完整 B−A、C−B 对照见 paired_comparisons.csv；每次抽样指标见 fold_metrics.csv。随机训练子集不是按材料体系留出的新域测试；已有组成重复和跨域局限仍需保留。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    entries={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='delivery_manifest.json'}
    if manifest.exists() and json.loads(manifest.read_text())!=entries:
        raise RuntimeError('Study delivery changed; preserve the previous manifest')
    if not manifest.exists():write_json(manifest,entries)
    print(f'STUDY REPORT COMPLETE: {len(entries)} verified delivery files',flush=True)


if __name__=='__main__':build()

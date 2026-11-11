"""Independently verify saved predictions, then generate course evidence."""
import json
import os
import shutil
import matminer.datasets
from pathlib import Path
os.environ.setdefault('MPLCONFIGDIR', str(Path(__file__).resolve().parents[1]/'outputs/topic1/mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from matbench.constants import MBV01_VALIDATION_DATA_PATH
from matplotlib.colors import LogNorm
from topic1.run import TASKS, MODELS, OUT, scores, sha, write_json


def main():
    figures = OUT/'figures'
    figures.mkdir(exist_ok=True)
    folds, metadata, prediction_frames, audit_tasks = [], {}, {}, []
    for name in TASKS:
        task = MatbenchTask(name)
        metadata[name] = dict(task.metadata)
        is_cls = task.metadata.task_type == 'classification'
        manifest = json.loads((OUT/name/'official_folds.json').read_text())
        reps = ['A','B','C'] + (['C_structure'] if name == 'matbench_jdft2d' else [])
        for rep in reps:
            for model in MODELS:
                frames = []
                official = MatbenchTask.from_file(str(OUT/name/'official_results'/f'{rep}_{model}.json.gz'))
                official.validate()
                for fold in task.folds:
                    train,_ = task.get_train_and_val_data(fold)
                    test,y = task.get_test_data(fold,include_target=True)
                    assert manifest[str(fold)] == {'train':list(train.index),'test':list(test.index)}
                    file = OUT/name/'runs'/rep/model/f'fold_{fold}'
                    frame = pd.read_csv(file.with_suffix('.csv'), float_precision='round_trip').rename(columns={'mbid':'id'})
                    result = json.loads(file.with_suffix('.json').read_text())
                    assert list(frame.id) == list(test.index)
                    np.testing.assert_allclose(frame.target,np.asarray(y,dtype=float),rtol=0,atol=1e-12)
                    recalculated = scores(y,frame.prediction.to_numpy(),is_cls)
                    for key,value in recalculated.items():
                        np.testing.assert_allclose(value,result['metrics'][key],rtol=1e-10,atol=1e-12)
                    official_data = official.results[f'fold_{fold}']['data']
                    np.testing.assert_allclose([official_data[i] for i in test.index],frame.prediction,rtol=1e-10,atol=1e-12)
                    if is_cls:
                        assert frame.prediction.between(0,1).all()
                        np.testing.assert_allclose(official.results[f'fold_{fold}']['scores']['rocauc'],recalculated['roc_auc'])
                    folds.append({k:result[k] for k in ['task','representation','model','fold','n_train','n_test','seconds']}|recalculated)
                    frames.append(frame)
                combined = pd.concat(frames,ignore_index=True)
                assert combined.id.is_unique and set(combined.id) == set(task.df.index)
                prediction_frames[name,rep,model] = combined
        audit_tasks.append(dict(task=name,samples=len(task.df),official_folds=5,representations=reps,
                               substantive_models=['Linear','RF','ET'],reference='Dummy',
                               fold_predictions_verified=len(reps)*len(MODELS)*5))
    data = pd.DataFrame(folds)
    data.to_csv(OUT/'fold_metrics.csv',index=False)
    metrics = ['mae','rmse','r2','roc_auc','balanced_accuracy','accuracy','f1']
    summary = data.groupby(['task','representation','model'])[metrics].agg(['mean','std']).reset_index()
    summary.columns = ['_'.join(filter(None,col)) for col in summary.columns]
    summary.to_csv(OUT/'summary.csv',index=False)
    audit = dict(pass_all=True,tasks=audit_tasks,total_main_fits=len(data),
                 official_validation_sha256=sha(MBV01_VALIDATION_DATA_PATH),
                 checks=['exact official train/test IDs and order','each sample tested once per configuration',
                         'three substantive models in every task','saved targets match official data',
                         'all metrics recomputed from saved predictions','all official result files validate',
                         'classification ROC-AUC uses probabilities'],
                 limitations=['No external SOTA reproduction','No hyperparameter search','Fold SD is not a confidence interval'])
    lines = ['# Topic 1：材料表示与常规模型的对照实验', '',
             '本报告由实际运行结果生成。采用 Matbench v0.1 官方数据和固定 5 折；不使用 Amadeus 原演示案例的自定义划分。', '',
             '## 老师要求核对', '',
             '| 要求 | 完成证据 |','|---|---|',
             '| 至少 5 个 Matbench 任务 | steels、jdft2d、expt_gap、expt_is_metal、glass 全部完成 |',
             '| 官方训练／测试 folds | official_folds.json 与官方接口逐 ID、逐顺序核对；每种配置每个样本恰好测试一次 |',
             '| 至少 3 个常规基线 | 回归 Ridge、RF、ET；分类 Logistic Regression、RF、ET；另加 Dummy |',
             '| 性能好坏原因分析 | 表示消融、固定 RF 的学习曲线、错误样本与误差分组、结构补充对照 |', '',
             f'完整主实验与结构补充共 **{len(data)} 次拟合**；其中 A/B/C 主实验 300 次（225 次正式基线 + 75 次 Dummy）。', '',
             '## 固定实验协议', '',
             'A = 118 维元素比例 + 3 维化学计量（121 维）；B = Magpie 元素属性统计（132 维）；C = A+B（253 维）。',
             'Magpie 特征由组成计算，不依赖任务标签。所有拟合得到的中位数填补和标准化参数仅来自当前训练折。',
             'Ridge alpha=10；Logistic C=1、max_iter=3000；RF/ET 各 100 棵树、max_features=1.0、min_samples_leaf=1；seed=42。',
             '参数在评测前固定，没有使用测试折选参数、选特征或做 early stopping。三种表示使用相同模型设置和相同外层折。',
             '分类概率 >0.5 转为标签用于 accuracy、balanced accuracy、F1；ROC-AUC 直接使用概率。',
             '报告均值 ± 样本标准差（ddof=1），不是置信区间。不同任务 MAE 单位不同，不计算跨任务平均 MAE。', '',
             '## 五个任务与数据', '',
             '| 任务 | 样本数 | 类型 | 目标／单位 |','|---|---:|---|---|']
    for name,meta in metadata.items():
        lines.append(f'| {name} | {meta["n_samples"]} | {meta["task_type"]} | {meta["target"]} / {meta["unit"] or "无量纲"} |')
    lines += ['', '## 固定 C 表示的主结果', '', '此表预先固定 C 表示，不根据测试结果挑选表示。Linear 在回归任务指 Ridge，在分类任务指 Logistic Regression。', '',
              '| 任务 | 指标 | Linear | RF | ET | Dummy |','|---|---|---:|---:|---:|---:|']
    for name,meta in metadata.items():
        metric = 'roc_auc' if meta['task_type']=='classification' else 'mae'
        subset = summary[(summary.task==name)&(summary.representation=='C')].set_index('model')
        cells = [f'{subset.loc[m,metric+"_mean"]:.4f} ± {subset.loc[m,metric+"_std"]:.4f}' for m in MODELS]
        lines.append('| '+ ' | '.join([name,metric+(' ↑' if metric=='roc_auc' else ' ↓')]+cells)+' |')
    lines += ['', '## 表示消融与逐任务解释', '', '下列差异是同模型、同测试折的观察结果。测试结果可用于事后解释，但不能据此声称所选最优配置的无偏泛化性能。', '']
    fig, axes = plt.subplots(2,3,figsize=(14,8),layout='constrained')
    for ax,(name,meta) in zip(axes.flat,metadata.items()):
        metric = 'roc_auc' if meta['task_type']=='classification' else 'mae'
        table = summary[(summary.task==name)&summary.representation.isin(['A','B','C'])].pivot(index='model',columns='representation',values=metric+'_mean').reindex(MODELS)[['A','B','C']]
        ax.imshow(table.values,cmap='YlGnBu' if metric=='roc_auc' else 'YlOrRd',aspect='auto',
                  norm=None if metric=='roc_auc' else LogNorm(vmin=table.values.min(),vmax=table.values.max()))
        ax.set(xticks=range(3),xticklabels=['A','B','C'],yticks=range(4),yticklabels=MODELS,
               title=name.replace('matbench_','')+' / '+metric+(' (higher better)' if metric=='roc_auc' else ' (lower better)'))
        for i in range(4):
            for j in range(3):
                ax.text(j,i,f'{table.iloc[i,j]:.3f}',ha='center',va='center',color='black',bbox={'facecolor':'white','alpha':.7,'edgecolor':'none','pad':2})
        lines += [f'### {name}', '', table.round(4).to_markdown(), '']
        a,b,c = [float(table.loc['RF',r]) for r in ['A','B','C']]
        direction = (c-a) if metric=='roc_auc' else (a-c)
        lines.append(f'固定 RF：A={a:.4f}、B={b:.4f}、C={c:.4f}；C 相对 A 的改善量为 {direction:+.4f}（负数表示变差）。')
        frame = prediction_frames[name,'C','RF'].copy()
        frame['n_elements_bin'] = pd.cut(frame.n_elements,[0,2,4,np.inf],labels=['1–2','3–4','5+'])
        frame['error'] = ((frame.prediction>.5)!=frame.target.astype(bool)).astype(float) if meta['task_type']=='classification' else frame.abs_error
        groups=[]
        for column in ['n_elements_bin','composition_seen_in_train','unseen_element']:
            if column not in frame: continue
            g=frame.groupby(column,observed=True).agg(n=('error','size'),error=('error','mean')).reset_index()
            g.columns=['group','n','error'];g.insert(0,'slice',column);groups.append(g)
        slices = pd.concat(groups,ignore_index=True)
        slices.to_csv(OUT/name/'error_slices_RF_C.csv',index=False)
        lines += ['', '固定 RF/C 的误差分组（分类为错误率；回归为 MAE；按合并测试样本统计，与主表五折等权平均可能略有差别）：', '',slices.round(4).to_markdown(index=False),'']
        errors = frame.sort_values('abs_error',ascending=False).head(10)
        errors.to_csv(OUT/name/'top_errors_RF_C.csv',index=False)
        lines += ['最大错误样本（前 5 个；完整前 10 个另存 CSV）：','',
                  errors[['id','formula','target','prediction','abs_error']].head(5).round(4).to_markdown(index=False),'']
        if meta['task_type']=='regression':
            lines.append(f'目标中位数 {frame.target.median():.4f}，95% 分位数 {frame.target.quantile(.95):.4f}，最大值 {frame.target.max():.4f}；最高 10 个绝对误差占总绝对误差 {errors.abs_error.sum()/frame.abs_error.sum():.1%}。')
        explanations = {
            'matbench_steels':'解释假设：312 个样本限制了复杂关系的估计；组成输入不含热处理和加工历史，不能区分所有相同组成下的强度差异。结合学习曲线及错误分组判断，不能仅凭低分证明具体机理。',
            'matbench_jdft2d':'解释假设：剥离能与层间作用和几何环境有关，组成会丢失这些信息；长尾样本也可能支配 MAE。下方结构描述符对照用于检验简单结构信息的增益；描述符没有覆盖全部局域结构。',
            'matbench_expt_gap':'解释假设：元素属性统计可提供化学规律，但组成无法完整区分晶型和实验条件；零带隙材料与非零带隙材料混合也使单一回归模型更困难。',
            'matbench_expt_is_metal':'解释假设：金属性与元素电子结构相关，属性统计可能比稀疏元素比例更易泛化；相近组成也可能受结构影响，错误样本用于提出下一步核查方向。',
            'matbench_glass':'解释假设：玻璃形成涉及多元素相互作用和制备条件；组成数据未包含全部工艺信息。类别比例不均衡，因此不能只看 accuracy；ROC-AUC 与 balanced accuracy 需要一起解释。'}
        lines += [explanations[name],'']
    axes.flat[-1].axis('off')
    fig.savefig(figures/'representation_heatmaps.png',dpi=180)
    plt.close(fig)
    lines += ['![模型与表示对照](figures/representation_heatmaps.png)', '',
              '回归热力图颜色使用各任务内对数尺度，以同时显示严重失败和正常模型；单元格是原始 MAE，未对分数取对数。不同面板的颜色不能直接比较。', '',
              '## Ridge 严重外推的实际失败原因', '',
              '此诊断在主实验完成后进行，仅解释结果，不用于修改参数或重新选择正式模型。重新拟合两次相同 Ridge，并核对预测与原结果一致。', '']
    evidence=json.loads((OUT/'ridge_failure_diagnostic.json').read_text())
    for row in evidence['results']:
        top=row['contributions'][0]
        lines += [f'**{row["representation"]} 表示，第 {row["fold"]} 折，{row["formula"]}（{row["sample"]}）**：真实带隙 {row["target"]:.4f} eV，预测 {row["prediction"]:.4f} eV。', '',
                  f'最大贡献来自 `{top["feature"]}`：训练特征范围 [{top["train_min"]:.8g}, {top["train_max"]:.8g}]，标准化尺度仅 {top["train_scale"]:.8g}，测试值却达到 {top["test_value"]:.8g}；标准化后为 {top["standardized_value"]:.4g}，线性预测贡献 {top["contribution"]:.4f} eV。', '']
    lines += ['这说明小尺度特征上的分布外输入可让线性模型严重外推。树模型按阈值划分并平均叶节点训练标签，不会以同样方式线性放大该数值。',
              '该失败没有删除、截断或替换。训练折内的近零方差过滤、稳健缩放可作为后续独立方案，但应在内层验证中确定并单独评测，不能用当前测试结果调到最好。', '',
              '## 结构信息补充实验', '',
              '在 jdft2d 上增加密度、每原子体积、晶胞位点数、晶格长度比及最小／最大晶格角，共 6 个特征。未加入目标或目标计算的衍生量；晶胞选择和真空层可能影响这些简单描述符，解释需谨慎。', '']
    st = summary[(summary.task=='matbench_jdft2d')&summary.representation.isin(['C','C_structure'])][['representation','model','mae_mean','mae_std']]
    lines += [st.round(4).to_markdown(index=False),'','## 小数据学习曲线','',
              '预先选 steels 与 expt_gap，固定 RF/C。每个官方训练折内用 seed=42+fold 生成嵌套的 20/40/60/80% 子集，测试折始终不变；100% 复用主实验。没有根据测试结果选择训练比例。','']
    learning=[]
    for name in ['matbench_steels','matbench_expt_gap']:
        for path in sorted((OUT/name/'learning').glob('*/*/*.json')):
            r=json.loads(path.read_text()); f=pd.read_csv(path.with_suffix('.csv'), float_precision='round_trip').rename(columns={'mbid':'id'})
            task=MatbenchTask(name)
            xtest,y=task.get_test_data(r['fold'],include_target=True)
            assert list(f.id)==list(xtest.index)
            np.testing.assert_allclose(f.target,np.asarray(y,dtype=float))
            np.testing.assert_allclose(mean:=scores(y,f.prediction.to_numpy(),False)['mae'],r['metrics']['mae'])
            learning.append(dict(task=name,fold=r['fold'],fraction=r['fraction'],n_train=r['n_train'],mae=mean))
        for _,r in data[(data.task==name)&(data.representation=='C')&(data.model=='RF')].iterrows():
            learning.append(dict(task=name,fold=r.fold,fraction=1.,n_train=r.n_train,mae=r.mae))
    lc=pd.DataFrame(learning)
    assert len(lc)==50,'Learning curves incomplete: need two tasks × five folds × five fractions.'
    lc.to_csv(OUT/'learning_curve.csv',index=False)
    fig,axes=plt.subplots(1,2,figsize=(11,4),layout='constrained')
    for ax,name in zip(axes,['matbench_steels','matbench_expt_gap']):
        g=lc[lc.task==name].groupby('fraction').mae.agg(['mean','std'])
        ax.errorbar(g.index*100,g['mean'],yerr=g['std'],marker='o',capsize=4)
        ax.set(title=name.replace('matbench_','')+' / RF + C',xlabel='Fraction of official training fold (%)',ylabel='MAE ('+metadata[name]['unit']+')')
        ax.grid(alpha=.2)
        lines += [f'**{name}**', '', g.round(4).to_markdown(), '',f'20% → 100% 的平均 MAE 变化：{g.loc[.2,"mean"]:.4f} → {g.loc[1.,"mean"]:.4f}。误差条为折间标准差，折训练集相互重叠，不代表独立重复实验的置信区间。','']
    fig.savefig(figures/'learning_curves.png',dpi=180);plt.close(fig)
    lines += ['![学习曲线](figures/learning_curves.png)','', '## 局限与展示建议', '',
              '- 本轮完成常规模型和表示对照，不声称 SOTA；未运行 CrabNet/MODNet 等加分模型。',
              '- 固定参数控制比较条件，未必给每个模型最佳性能；一次随机种子不能量化全部训练随机性。',
              '- 相同组成可能出现在不同官方折；严格遵守官方划分并报告该现象，不擅自重划数据。',
              '- 表示增加维度不等于增加独立信息；C 的原子统计大多由 A 的元素比例决定，改善主要可解释为归纳偏置和有限样本下的学习便利性。',
              '- 本报告的材料层解释是待验证假设，未通过新增物理实验确认。',
              '- 不将本轮概率 ROC-AUC 与旧版阈值化的分类排行榜分数直接比较。',
              '- 建议 10 分钟展示：研究问题 1 分钟、任务与官方协议 2 分钟、模型与表示 1 分钟、主结果 2 分钟、学习曲线和错误分析 3 分钟、结论及局限 1 分钟。', '',
              '## 文件与复现', '',
              '`summary.csv`：均值与标准差；`fold_metrics.csv`：逐折指标；`learning_curve.csv`：学习曲线；`requirement_audit.json`：完整性审核。',
              '各任务目录保存 `official_folds.json`、`features.json`、`samples.csv`、逐折 CSV 预测与参数 JSON、可由 Matbench 读取的 `official_results/*.json.gz`。',
              '运行命令及安装版本见仓库 `topic1/README.md`、`topic1/requirements.lock.txt`、`environment.json`、`protocol.json`。', '',
              '来源：[Matbench 官方项目](https://github.com/materialsproject/matbench)、[Matbench 论文](https://doi.org/10.1038/s41524-020-00406-3)、[Matminer](https://github.com/hackingmaterials/matminer)。']
    (OUT/'REPORT.md').write_text('\n'.join(lines))
    audit['learning_fits']=40
    audit['diagnostic_refits']=2
    audit['learning_points_verified']=50
    dataset_dir=Path(matminer.datasets.__file__).parent
    registry=json.loads((dataset_dir/'dataset_metadata.json').read_text())
    provenance=[]
    (OUT/'data').mkdir(exist_ok=True)
    for name in TASKS:
        source=dataset_dir/(name+'.json.gz')
        digest=sha(source)
        assert digest==registry[name]['hash']
        shutil.copy2(source,OUT/'data'/source.name)
        provenance.append(dict(task=name,url=registry[name]['url'],sha256=digest,registry_hash_matches=True))
    write_json(OUT/'data_provenance.json',provenance)
    audit['raw_dataset_hashes_verified']=5
    write_json(OUT/'requirement_audit.json',audit)
    print(json.dumps(audit,ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()

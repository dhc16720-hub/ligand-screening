"""Independently audit V3 predictions and selections, then report all results."""
import json
import os
import zipfile
import joblib
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve
from topic1.run import ROOT, OUT as V1, TASKS, sha, write_json, scores
from topic1.advanced import FAMILIES, inner_folds, mean_inner_loss
from topic1.refine import OUT, V2, register
from topic1.refine_models import configs, predict, combine, select

os.environ.setdefault('MPLCONFIGDIR', str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UNITS = {'matbench_steels':'MPa', 'matbench_jdft2d':'meV/atom', 'matbench_expt_gap':'eV',
         'matbench_expt_is_metal':'unitless', 'matbench_glass':'unitless'}


def read(path):
    return pd.read_csv(path, float_precision='round_trip')


def source(name, rep, family):
    return (V2 if family in FAMILIES else OUT)/name/'runs'/rep/family


def audit():
    register()
    preserved = {}
    for version, directory in [('V1', V1), ('V2', V2)]:
        manifest = json.loads((directory/'delivery_manifest.json').read_text())
        for path, checksum in manifest.items():
            assert sha(ROOT/path) == checksum, f'Historical artifact changed: {path}'
        preserved[version] = len(manifest)
    rows, selections, residuals, importance = [], [], [], []
    replayed = 0
    for name in TASKS:
        task = MatbenchTask(name)
        cl = task.metadata.task_type == 'classification'
        info = read(V1/name/'samples.csv').set_index('id')
        for rep in ['C'] + (['C_structure'] if name=='matbench_jdft2d' else []):
            matrix = np.load(V1/name/f'features_{rep}.npy')
            names = json.loads((V1/name/'features.json').read_text())[rep]['names']
            for family in list(configs(cl))+['Selected']:
                official = MatbenchTask.from_file(str(OUT/name/'official_results'/f'{rep}_{family}.json.gz'))
                official.validate()
                ids = []
                for fold in task.folds:
                    train, y = task.get_train_and_val_data(fold)
                    test, target = task.get_test_data(fold, include_target=True)
                    inner, grouped = inner_folds(np.asarray(y), info.loc[train.index,'formula'].to_numpy(), cl, fold)
                    manifest = json.loads((OUT/name/'splits'/f'fold_{fold}.json').read_text())
                    expected = dict(outer_train=list(train.index), outer_test=list(test.index), grouped=grouped,
                                    inner=[dict(train=list(train.index[a]),validation=list(train.index[b])) for a,b in inner])
                    assert manifest == expected
                    assert set(train.index).isdisjoint(test.index)
                    folder = source(name, rep, family)
                    result = json.loads((folder/f'fold_{fold}.json').read_text())
                    frame = read(folder/f'fold_{fold}.csv')
                    assert list(frame.mbid) == list(test.index)
                    np.testing.assert_array_equal(frame.target, np.asarray(target, float))
                    p = frame.prediction.to_numpy()
                    np.testing.assert_array_equal(p, [official.results[f'fold_{fold}']['data'][i] for i in test.index])
                    metrics = scores(target, p, cl)
                    for key, value in metrics.items():
                        np.testing.assert_allclose(value, result['metrics'][key], rtol=1e-12, atol=1e-12)
                    if family != 'Selected':
                        search = result['search']
                        assert [s['parameters'] for s in search] == configs(cl)[family]
                        for s in search:
                            assert len(s['inner_losses']) == 3
                            np.testing.assert_allclose(s['inner_loss_mean'], np.mean(s['inner_losses']))
                        winner = min(search, key=lambda s:s['inner_loss_mean'])
                        assert result['selected_parameters'] == winner['parameters']
                        oof = read(folder/f'inner_oof_{fold}.csv')
                        assert list(oof.mbid) == list(train.index)
                        np.testing.assert_array_equal(oof.target, np.asarray(y,float))
                        for j,(_,b) in enumerate(inner): assert (oof.inner_fold.iloc[b]==j).all()
                        np.testing.assert_allclose(mean_inner_loss(np.asarray(y),oof.prediction.to_numpy(),inner,cl), winner['inner_loss_mean'])
                        model = joblib.load(folder/f'model_{fold}.joblib')
                        actual = predict(model, matrix[task.df.index.get_indexer(test.index)], cl)
                        np.testing.assert_array_equal(p, actual)
                        replayed += 1
                        if family == 'CatBoost':
                            for feature, weight in zip(names, model[-1].feature_importances_):
                                importance.append(dict(task=name,representation=rep,fold=fold,feature=feature,importance=float(weight)))
                    else:
                        pool = FAMILIES+list(configs(cl))
                        oofs, preds = {}, {}
                        for f in pool:
                            other = source(name,rep,f)
                            oof = read(other/f'inner_oof_{fold}.csv')
                            outer = read(other/f'fold_{fold}.csv')
                            assert list(oof.mbid)==list(train.index) and list(outer.mbid)==list(test.index)
                            for j,(_,b) in enumerate(inner): assert (oof.inner_fold.iloc[b]==j).all()
                            oofs[f] = oof.prediction.to_numpy(); preds[f] = outer.prediction.to_numpy()
                        recipe, candidates = select(np.asarray(y),oofs,inner,cl)
                        assert recipe==result['recipe'] and candidates==result['candidate_recipes']
                        np.testing.assert_array_equal(p,combine(preds,recipe))
                        selections.append(dict(task=name,representation=rep,fold=fold,members='+'.join(recipe['members']),operation=recipe['operation'],inner_loss=recipe['inner_loss']))
                        train_formulas=set(info.loc[train.index,'formula'])
                        for i, row in frame.iterrows():
                            sample=info.loc[row.mbid]
                            residuals.append(dict(task=name,representation=rep,fold=fold,mbid=row.mbid,formula=sample.formula,
                                                  n_elements=int(sample.n_elements),formula_seen_in_train=sample.formula in train_formulas,
                                                  target=float(row.target),prediction=float(row.prediction),absolute_error=float(abs(row.target-row.prediction))))
                    if cl: metrics['brier']=float(brier_score_loss(target,p))
                    rows.append(dict(task=name,representation=rep,model=family,fold=fold,seconds=result.get('seconds',0),**metrics))
                    ids.extend(frame.mbid)
                assert len(ids)==len(set(ids))==len(task.df) and set(ids)==set(task.df.index)
        print('AUDIT OK',name,flush=True)
    df=pd.DataFrame(rows);df.to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(selections).to_csv(OUT/'selection_history.csv',index=False)
    pd.DataFrame(residuals).sort_values('absolute_error',ascending=False).to_csv(OUT/'sample_diagnostics.csv',index=False)
    pd.DataFrame(importance).to_csv(OUT/'catboost_feature_importance.csv',index=False)
    result=dict(pass_all=True,official_tasks=TASKS,outer_folds=5,verified_prediction_folds=len(rows),
                replayed_saved_models=replayed,new_inner_fits=replayed*9,new_outer_refits=replayed,total_new_fits=replayed*10,
                preserved_historical_files=preserved,selection_recomputed=True,validation='Official IDs, splits, metrics, exports, model replay and inner-only recipe choice checked')
    write_json(OUT/'requirement_audit.json',result)
    return df,result


def report(df,audit_result):
    old=read(V2/'fold_metrics.csv')
    summary=[];comparison=[]
    for (name,rep,family),part in df.groupby(['task','representation','model']):
        metric='roc_auc' if name in TASKS[3:] else 'mae'
        summary.append(dict(task=name,representation=rep,model=family,metric=metric,unit=UNITS[name],mean=part[metric].mean(),std=part[metric].std(),seconds=part.seconds.sum()))
    summary=pd.DataFrame(summary);summary.to_csv(OUT/'summary.csv',index=False)
    for name in TASKS:
        rep='C_structure' if name=='matbench_jdft2d' else 'C'
        metric='roc_auc' if name in TASKS[3:] else 'mae'
        a=old[(old.task==name)&(old.representation==rep)&(old.model=='Selected')].sort_values('fold')[metric].to_numpy()
        b=df[(df.task==name)&(df.representation==rep)&(df.model=='Selected')].sort_values('fold')[metric].to_numpy()
        improvement=b-a if metric=='roc_auc' else a-b
        comparison.append(dict(task=name,representation=rep,metric=metric,v2_mean=a.mean(),v3_mean=b.mean(),improvement=improvement.mean(),folds_improved=int((improvement>0).sum()),relative_error_reduction_pct=100*improvement.mean()/a.mean() if metric=='mae' else None))
    comp=pd.DataFrame(comparison);comp.to_csv(OUT/'v2_v3_comparison.csv',index=False)
    fig,axes=plt.subplots(1,5,figsize=(15,3.8))
    for ax,row in zip(axes,comparison):
        name,rep,metric=row['task'],row['representation'],row['metric']
        a=old[(old.task==name)&(old.representation==rep)&(old.model=='Selected')].sort_values('fold')[metric].to_numpy()
        b=df[(df.task==name)&(df.representation==rep)&(df.model=='Selected')].sort_values('fold')[metric].to_numpy()
        for x,y in zip(a,b): ax.plot([0,1],[x,y],color='#aab6c4',marker='o',lw=1,ms=4)
        ax.plot([0,1],[a.mean(),b.mean()],color='#087f8c',marker='D',lw=2.5,label='Five-fold mean')
        ax.set_xticks([0,1],['V2','V3']);ax.set_title(name.replace('matbench_',''));ax.set_ylabel('ROC-AUC (higher better)' if metric=='roc_auc' else f'MAE, {UNITS[name]} (lower better)')
        ax.grid(axis='y',alpha=.2)
    axes[0].legend(fontsize=8);fig.suptitle('Frozen inner selection; same official outer folds and representations')
    fig.tight_layout();(OUT/'figures').mkdir(exist_ok=True);fig.savefig(OUT/'figures/v2_v3_paired.png',dpi=180);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(9,4))
    for ax,name in zip(axes,TASKS[3:]):
        for version,directory,color in [('V2',V2,'#a569bd'),('V3',OUT,'#087f8c')]:
            p=pd.concat([read(directory/name/'runs/C/Selected'/f'fold_{f}.csv') for f in range(5)])
            fraction,mean_probability=calibration_curve(p.target,p.prediction,n_bins=10,strategy='quantile')
            ax.plot(mean_probability,fraction,'o-',label=version,color=color)
        ax.plot([0,1],[0,1],'--',color='gray',label='Ideal')
        ax.set_title(name.replace('matbench_',''));ax.set_xlabel('Mean predicted probability');ax.set_ylabel('Observed positive fraction')
        ax.legend();ax.grid(alpha=.15)
    fig.suptitle('Held-out probability reliability (10 quantile bins; descriptive)')
    fig.tight_layout();fig.savefig(OUT/'figures/probability_reliability.png',dpi=180);plt.close(fig)
    lines=['# Topic 1：V3 预测质量优化与独立核验','',
           '本轮新增 CatBoost，并对回归任务进行相同训练预算的 XGBoost 损失函数/目标变换对照。保留五个任务、官方五折和前两版全部结果。模型参数与组合只由外层训练集内部三折决定；不根据外层测试成绩挑选最终模型。', '',
           '**研究边界：**V3 在查看 V1/V2 结果后设计，属于探索性迭代，并非新的盲测。V3 每个 boosting 模型为 512 次迭代，V2 为 128 棵树，因此跨版本变化不能全部归因于算法。五折均值和折间标准差是描述性统计，五折训练集重叠，不能当作五次独立实验。','',
           '## 1. 主结果：预先固定的内部选择流程','',
           'jdft2d 主结果固定使用 C_structure，其余任务固定使用 C；与 V2 相同表示比较，避免把结构信息的收益混入算法收益。每折候选池为 V2 的 Linear/RF/ET/XGB，加上本轮新模型。按内部验证排序，比较最佳单模型、前两名均值、前三名均值；回归额外比较前三名中位数。外层标签仅在预测完成后用于评价。','',
           '|任务|表示|指标|V2|V3|改善折数 / 5|','|---|---|---|---:|---:|---:|']
    for r in comparison: lines.append(f"|{r['task']}|{r['representation']}|{r['metric']} ({UNITS[r['task']]})|{r['v2_mean']:.6f}|{r['v3_mean']:.6f}|{r['folds_improved']}|")
    lines+=['','MAE 越低越好；ROC-AUC 越高越好。未改进的任务和折次同样保留，不以单个模型的测试集最高分替换以上主结果。','','![五折配对结果](figures/v2_v3_paired.png)','','## 2. 新模型完整结果','','下表包含所有新增模型及内部选择流程，未按外层测试成绩删减。std 为五折样本标准差。','','|任务|表示|模型|均值|折间 std|','|---|---|---|---:|---:|']
    for r in summary.itertuples(): lines.append(f'|{r.task}|{r.representation}|{r.model}|{r.mean:.6f}|{r.std:.6f}|')
    lines+=['','## 3. 可解释的科学对照','',
            '- **损失函数：**XGB_Squared 与 XGB_MAE 使用相同的 512 次迭代、三个深度候选和内层划分，比较平方损失与绝对损失。两者分别调参，结果代表两种匹配搜索预算的流程，并非固定同一深度的单变量试验。',
            '- **目标变换：**XGB_Log 使用可逆的 sign(y)·log(1+|y|)，保留零值和负值，不删除或裁剪标签；逆变换后按原始单位 MAE 选择参数。它可能缓和极端值影响，但不保证原始尺度的 MAE 下降。',
            '- **CatBoost：**三个配置在内层选择；回归同时包含 MAE 和 RMSE 配置。因此其结果是整个候选流程的表现，不能单独归因于某一种损失。',
            '- **结构消融：**jdft2d 同时保留 C 与 C_structure，可在相同算法下比较六个全局结构描述符的贡献；这些描述符尚不等同于完整局部原子环境。',
            '- **误差与重要性：**sample_diagnostics.csv 保留全部 Selected 样本的公式、元素数、公式是否在该训练折出现、真值及预测。catboost_feature_importance.csv 提供各折特征重要性，表示模型依赖程度，不构成因果解释。分类 absolute_error 是概率误差，不是 MAE 主指标。','',
            '### 匹配预算下，绝对损失和目标变换相对平方损失的变化','',
            '|任务|表示|Squared MAE|MAE 损失的 MAE|Log 变换的 MAE|','|---|---|---:|---:|---:|']
    for name in TASKS[:3]:
        for rep in ['C']+(['C_structure'] if name=='matbench_jdft2d' else []):
            sub=summary[(summary.task==name)&(summary.representation==rep)].set_index('model')
            lines.append(f"|{name}|{rep}|{sub.loc['XGB_Squared','mean']:.6f}|{sub.loc['XGB_MAE','mean']:.6f}|{sub.loc['XGB_Log','mean']:.6f}|")
    diagnostics=read(OUT/'sample_diagnostics.csv')
    importance=read(OUT/'catboost_feature_importance.csv')
    selections=read(OUT/'selection_history.csv')
    lines+=['','### 误差分布与模型依赖：用结果解释表现','',
            '以下统计使用各样本恰好一次的外层预测。大误差样本用于诊断，未从评估中剔除，也未用于重新选择本轮参数。','']
    for name in TASKS:
        rep='C_structure' if name=='matbench_jdft2d' else 'C'
        part=diagnostics[(diagnostics.task==name)&(diagnostics.representation==rep)]
        ranking=importance[(importance.task==name)&(importance.representation==rep)].groupby('feature').importance.mean().sort_values(ascending=False)
        chosen=selections[(selections.task==name)&(selections.representation==rep)]
        recipes=chosen.groupby(['members','operation']).size().sort_values(ascending=False)
        choices='；'.join(f'{members}（{operation}，{count}/5 折）' for (members,operation),count in recipes.items())
        if name in TASKS[:3]:
            n_top=max(1,int(np.ceil(len(part)*.01)))
            share=part.absolute_error.nlargest(n_top).sum()/part.absolute_error.sum()*100
            worst=part.loc[part.absolute_error.idxmax()]
            lines.append(f"- **{name}：**绝对误差中位数 {part.absolute_error.median():.4f}，90% 分位数 {part.absolute_error.quantile(.9):.4f}；误差最高的约 1% 样本贡献了 {share:.1f}% 的绝对误差总和。最大误差样本 {worst.mbid}（{worst.formula}），真值 {worst.target:.4f}，预测 {worst.prediction:.4f}。长尾统计描述误差集中程度，不能单凭它认定标签错误。")
        else:
            brier=float(np.mean((part.target-part.prediction)**2))
            base=old[(old.task==name)&(old.representation==rep)&(old.model=='Selected')]
            # Brier is compared using pooled predictions, not unweighted fold means.
            previous=pd.concat([read(V2/name/'runs'/rep/'Selected'/f'fold_{f}.csv') for f in range(5)])
            prior=float(np.mean((previous.target-previous.prediction)**2))
            lines.append(f'- **{name}：**全体外层样本 Brier 分数为 {brier:.6f}，V2 为 {prior:.6f}（越低越好）。ROC-AUC 衡量排序，Brier 同时考察概率误差，两者应一起解读，不能仅凭排序得分宣称概率已校准。')
        lines.append(f"  内层选择记录：{choices}。CatBoost 五折平均重要性最高的三个特征为 {', '.join(ranking.head(3).index)}；此处解释的是 CatBoost，最终组合还可能包含其他模型。")
    lines+=['','![分类概率可靠性](figures/probability_reliability.png)','',
            '可靠性图将各折测试概率汇总后按分位数分箱，虚线为理想校准；只用于诊断，未在这些测试标签上拟合校准器。不同版本的分箱边界可能不同，需结合 Brier 分数解读。','',
            '### 与强模型公开预测的对照','',
            '沿用 V2 对固定 Matbench 仓库归档预测的独立重评分，不下载当前排行榜来替换原基线。公开模型的输入表示、调参和计算预算不同；这里只比较官方相同样本与折上的指标，不是公平算力竞赛。MODNet 的 jdft2d 结果利用结构信息；CrabNet 使用组成信息。','',
            '|任务|V3 内层选择主结果|MODNet 公开预测复核|CrabNet 公开预测复核|','|---|---:|---:|---:|']
    refs=read(V2/'reference_fold_metrics.csv')
    for row in comparison:
        values=[]
        for model in ['MODNet','CrabNet']:
            sub=refs[(refs.task==row['task'])&(refs.model==model)]
            values.append(f"{sub[row['metric']].mean():.6f}" if len(sub) else '—')
        lines.append(f"|{row['task']}|{row['v3_mean']:.6f}|{values[0]}|{values[1]}|")
    lines+=['','来源及原始文件哈希见 V2 的 references/provenance.json；本轮没有本地重训 MODNet/CrabNet。','','## 4. 老师要求与交付证据','',
            '|要求|证据|','|---|---|',
            '|至少五个 Matbench 任务|steels、jdft2d、expt_gap、expt_is_metal、glass 全部完成|',
            '|官方训练/测试划分|沿用 Matbench 官方五折；逐项核验样本 ID、顺序及无交叉|',
            '|至少三个常规模型|V1/V2 的 Linear、RF、ET、XGB 完整保留，V3 增加 CatBoost 与回归目标对照|',
            '|分析表现与科学表示|V1/V2 的 A/B/C、结构消融、小样本学习曲线、异常线性模型诊断；V3 增加损失函数与目标变换对照|',
            '|强模型比较（加分项）|V2 已对 MODNet/CrabNet 官方归档预测按原始折重新评分，属于公开预测复核，非本地重训；算力预算不匹配|',
            '|不确定性与稳定性扩展|V2 已完成独立校准集 conformal 区间及多随机种子实验；这些结论不自动转移到 V3 新模型|','',
            f"本轮完成 **{audit_result['total_new_fits']} 次新拟合**，核验 **{audit_result['verified_prediction_folds']} 份外层预测**，逐个重新加载 **{audit_result['replayed_saved_models']} 个保存模型**重算预测；V1/V2 交付清单中的文件 SHA256 均保持不变。",'',
            '上述内容完成了代码与实验侧的要求。组员登记、最终约 10 分钟汇报及 2 分钟问答仍需按课程安排完成，不能把实验完成等同于课程提交完成。','',
            '## 5. 复现与文件','',
            '运行方法见 topic1/V3_README.md。protocol.json 冻结训练前的模型配置、选择规则和源码哈希；每折保存参数搜索日志、内层 OOF、外层预测、最终模型以及官方 Matbench 格式。requirement_audit.json 为独立核验结果。', '',
            '官方技术参考：[CatBoost regression objectives](https://catboost.ai/docs/en/concepts/loss-functions-regression)、[XGBoost parameters](https://xgboost.readthedocs.io/en/stable/parameter.html)。实际安装版本和依赖见 environment.json 与 requirements-v3.lock.txt。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')
    improved=[r for r in comparison if r['improvement']>0]
    declined=[r for r in comparison if r['improvement']<0]
    talk=['# 10 分钟汇报提纲与问答准备','',
          '这是内容提纲，尚不是最终幻灯片。组员分工与课程登记请按实际情况补齐。','',
          '|页|时长|讲述重点|应展示的证据|','|---|---:|---|---|',
          '|1|45 秒|问题：小样本材料性质预测，科学表示与模型选择各有什么作用？|五个任务及目标单位|',
          '|2|75 秒|数据和表示：元素比例、Magpie 统计、拼接表示；二维材料补充结构信息|V1/V2 A/B/C 与结构消融|',
          '|3|90 秒|评价设计：官方外层五折；内层三折负责调参和组合；重复组成内层分组|protocol.json 与每折 splits 文件|',
          '|4|60 秒|基线覆盖：Linear、RF、ET、XGB；Dummy 作为参照|V1/V2 完整均值和折间波动|',
          '|5|90 秒|失败分析：带隙线性模型因近常数特征缩放而异常；稳健预处理改善|V1 diagnosis 与 V2 固定 alpha 控制实验|',
          '|6|90 秒|新迭代：CatBoost 与三种 XGBoost 目标；成功和负结果一起呈现|V3 配对图与完整结果表|',
          '|7|60 秒|与 MODNet/CrabNet 公开预测对照，说明输入与预算差异|V2 reference_fold_metrics.csv、来源哈希|',
          '|8|90 秒|误差长尾、小数据学习曲线、不确定性、限制与下一步|V3 样本诊断；V2 学习曲线与区间覆盖|','',
          '## 本轮可以准确说出的结果','']
    for r in comparison:
        change=f"MAE 相对变化 {-r['relative_error_reduction_pct']:+.2f}%" if r['metric']=='mae' else f"ROC-AUC 绝对变化 {r['improvement']:+.6f}"
        talk.append(f"- {r['task']}：{r['v2_mean']:.6f} → {r['v3_mean']:.6f}，{change}，改善 {r['folds_improved']}/5 折。")
    talk+=['',f'本轮 {len(improved)} 个任务的主指标改善，{len(declined)} 个退步。不能说所有任务都变好了；也不能把测试集上分数最好的单模型替代预先约定的内部选择流程。','',
           '## 常见问答','',
           '**是否使用测试集调参？** 本轮每折的参数及组合由训练折内部选择，外层标签不参与这些选择。但方案设计已参考前两版结果，所以这是探索性改进，不是全新盲测。若要确认可迁移收益，需要额外未触碰的数据。','',
           '**为什么不只选测试集分数最高的模型？** 那会把评价数据变成调参数据。这里报告完整候选结果，同时固定用内层选择流程的成绩作为主结果。','',
           '**新模型的优势是否来自更多算力？** 跨版本无法排除该因素：新 boosting 模型是 512 次迭代，V2 为 128 棵树。三种新 XGBoost 流程的预算匹配，适合比较损失函数和目标变换。','',
           '**为什么钢材可能不提升？** 仅 312 个样本，内层优劣估计会有较大波动；增加候选数还会增加选择的不确定性。当前结果支持“小幅退步”这一观察，不能证明具体机制，后续需更多独立数据或重复嵌套验证。','',
           '**是否重训了 MODNet/CrabNet？** 没有。本项目重评分了官方归档预测并核验样本和折，计算预算未匹配。应该明确标注公开预测复核。','',
           '**特征重要性能证明物理因果吗？** 不能。重要性反映模型依赖，相关描述符可能分摊重要性；应作为后续科学验证的线索。','',
           '**V2 的 90% 区间是否保证 V3 覆盖率？** 不能。V2 区间来自其独立校准实验，不能转用来替新模型背书；V3 若需区间，应为对应流程另设合规校准。','',
           '**下一步怎样提高可信度？** 冻结当前方案后，引入未使用的外部数据作检验；对结构任务增加局部原子环境表示，并在训练前规定公平的搜索预算。不要继续依据同一外层测试成绩反复挑配方。']
    (OUT/'PRESENTATION_NOTES.md').write_text('\n'.join(talk)+'\n')


def package():
    # Incremental, allowlisted delivery: no credentials, environments, or unrelated application data.
    paths=list((ROOT/'topic1').glob('*.py'))+list((ROOT/'topic1').glob('*.md'))+list((ROOT/'topic1').glob('*.txt'))+[ROOT/'topic1/python.sh']
    paths+=list((ROOT/'tests').glob('test_topic1*.py'))
    paths+=[p for p in OUT.rglob('*') if p.is_file() and 'mpl-cache' not in p.parts and p.name!='delivery_manifest.json']
    paths=sorted(set(paths))
    write_json(OUT/'delivery_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in paths})
    paths.append(OUT/'delivery_manifest.json')
    destination=ROOT/'outputs/Topic1_V3_新增实验与模型.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in paths:z.write(p,p.relative_to(ROOT))
    print('PACKAGE',destination,round(destination.stat().st_size/1e6,2),'MB',flush=True)


if __name__=='__main__':
    frame,result=audit()
    report(frame,result)
    package()

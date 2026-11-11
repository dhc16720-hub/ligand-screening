"""Audit nested selection and build an evidence-based V2 research report."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import joblib
from matbench.task import MatbenchTask
from sklearn.metrics import brier_score_loss,log_loss
from sklearn.calibration import calibration_curve
from topic1.run import ROOT,OUT as V1,TASKS,write_json,sha,scores
from topic1.advanced import OUT,FAMILIES,CONFIGS,select_blend,inner_folds,mean_inner_loss,predict,export
from topic1.conformal import conformal_radius
os.environ.setdefault('MPLCONFIGDIR',str(OUT/'mpl-cache'))
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def read_prediction(path):
    return pd.read_csv(path,float_precision='round_trip')


def normalize_float32_exports():
    """Lossless float32-to-float64 CSV expansion, checked against saved models.

    Pandas writes float32 with round-trip float32 precision, not full float64
    decimal expansion. Recover the original floats, then export full precision.
    This changes neither trained models nor their mathematical predictions.
    """
    verified=0
    for name in TASKS:
        task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
        for rep in ['C']+(['C_structure'] if name=='matbench_jdft2d' else []):
            matrix=np.load(V1/name/f'features_{rep}.npy');changed=False
            for fold in task.folds:
                test=task.get_test_data(fold)
                folder=OUT/name/'runs'/rep/'XGB'
                path=folder/f'fold_{fold}.csv';p=read_prediction(path)
                restored=p.prediction.to_numpy().astype(np.float32).astype(float)
                model=joblib.load(folder/f'model_{fold}.joblib')
                actual=np.asarray(predict(model,matrix[task.df.index.get_indexer(test.index)],cl),float)
                np.testing.assert_array_equal(restored,actual)
                if not np.array_equal(p.prediction.to_numpy(),restored):
                    p['prediction']=restored;p.to_csv(path,index=False);changed=True
                selected=OUT/name/'runs'/rep/'Selected'/f'fold_{fold}.csv'
                recipe=json.loads(selected.with_suffix('.json').read_text())['recipe']['members']
                if recipe==['XGB']:
                    q=read_prediction(selected)
                    if not np.array_equal(q.prediction.to_numpy(),restored):
                        q['prediction']=restored;q.to_csv(selected,index=False);changed=True
                verified+=1
            if changed:export(task,rep,['XGB','Selected'])
    return verified


def audit():
    protocol=json.loads((OUT/'protocol.json').read_text())
    assert protocol['source_sha256']==sha(ROOT/'topic1/advanced.py')
    rows=[];frames={};selection=[];inner_fits=0;refits=0
    for name in TASKS:
        task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
        info=pd.read_csv(V1/name/'samples.csv').set_index('id')
        reps={'C':FAMILIES+['Selected','ET_seed7','ET_seed2026'],'A':['ET'],'B':['ET']}
        if name=='matbench_jdft2d':reps['C_structure']=FAMILIES+['Selected','ET_seed7','ET_seed2026']
        for rep,families in reps.items():
            for family in families:
                saved=[]
                official=MatbenchTask.from_file(str(OUT/name/'official_results'/f'{rep}_{family}.json.gz'));official.validate()
                for fold in task.folds:
                    train,ytrain=task.get_train_and_val_data(fold);test,ytest=task.get_test_data(fold,include_target=True)
                    manifest=json.loads((OUT/name/'splits'/f'fold_{fold}.json').read_text())
                    assert manifest['outer_train']==list(train.index) and manifest['outer_test']==list(test.index)
                    inner,grouped=inner_folds(np.asarray(ytrain),info.loc[train.index,'formula'].to_numpy(),cl,fold)
                    assert grouped==manifest['grouped']
                    for j,(a,b) in enumerate(inner):
                        assert manifest['inner'][j]==dict(train=list(train.index[a]),validation=list(train.index[b]))
                        assert set(train.index[a]).isdisjoint(train.index[b]) and set(train.index).isdisjoint(test.index)
                    path=OUT/name/'runs'/rep/family/f'fold_{fold}.json'
                    r=json.loads(path.read_text());p=read_prediction(path.with_suffix('.csv'))
                    assert list(p.mbid)==list(test.index) and r['n_train']==len(train) and r['n_test']==len(test)
                    np.testing.assert_array_equal(p.target,np.asarray(ytest,float))
                    recalculated=scores(ytest,p.prediction.to_numpy(),cl)
                    for k,v in recalculated.items():np.testing.assert_allclose(v,r['metrics'][k],rtol=1e-10,atol=1e-12)
                    np.testing.assert_array_equal(p.prediction,[official.results[f'fold_{fold}']['data'][i] for i in test.index])
                    if family in FAMILIES:
                        traces=r['search'];assert len(traces)==3 and all(len(x['inner_losses'])==3 for x in traces)
                        for i,tr in enumerate(traces):
                            assert tr['parameters']==CONFIGS[family][i]
                            np.testing.assert_allclose(tr['inner_loss_mean'],np.mean(tr['inner_losses']))
                        winner=int(np.argmin([tr['inner_loss_mean'] for tr in traces]))
                        assert r['selected_parameters']==traces[winner]['parameters']
                        oof=read_prediction(path.parent/f'inner_oof_{fold}.csv')
                        assert list(oof.mbid)==list(train.index)
                        np.testing.assert_array_equal(oof.target,np.asarray(ytrain))
                        np.testing.assert_allclose(mean_inner_loss(np.asarray(ytrain),oof.prediction.to_numpy(),inner,cl),r['selected_inner_loss'])
                        inner_fits+=9;refits+=1
                    elif family=='Selected':
                        oofs={f:read_prediction(OUT/name/'runs'/rep/f/f'inner_oof_{fold}.csv').prediction.to_numpy() for f in FAMILIES}
                        recipe,_=select_blend(np.asarray(ytrain),oofs,inner,cl)
                        assert recipe==r['recipe']
                        component=[read_prediction(OUT/name/'runs'/rep/f/f'fold_{fold}.csv').prediction.to_numpy() for f in recipe['members']]
                        np.testing.assert_array_equal(np.mean(component,axis=0),p.prediction.to_numpy())
                        selection.append(dict(task=name,representation=rep,fold=fold,members='+'.join(recipe['members']),inner_loss=recipe['inner_loss']))
                    else:
                        base=json.loads((OUT/name/'runs'/rep/'ET'/f'fold_{fold}.json').read_text())
                        assert r['selected_parameters']==base['selected_parameters']
                        assert r['seed'] in [7,2026];refits+=1
                    if cl:
                        recalculated['brier']=float(brier_score_loss(ytest,p.prediction))
                        recalculated['log_loss']=float(log_loss(ytest,p.prediction,labels=[False,True]))
                    rows.append(dict(task=name,representation=rep,model=family,fold=fold,**recalculated))
                    saved.append(p)
                frame=pd.concat(saved,ignore_index=True)
                assert frame.mbid.is_unique and set(frame.mbid)==set(task.df.index)
                frames[name,rep,family]=frame
        print('Audited nested CV:',name,flush=True)
    # Split-conformal: calibration targets never enter the fixed model fitting.
    for name in TASKS[:3]:
        task=MatbenchTask(name)
        for fold in task.folds:
            r=json.loads((OUT/name/'conformal'/f'fold_{fold}.json').read_text())
            train,_=task.get_train_and_val_data(fold);test,y=task.get_test_data(fold,include_target=True)
            a,b,c=map(set,[r['proper_train_ids'],r['calibration_ids'],r['test_ids']])
            assert a.isdisjoint(b) and (a|b)==set(train.index) and c==set(test.index) and (a|b).isdisjoint(c)
            p=read_prediction(OUT/name/'conformal'/f'fold_{fold}.csv');cal=read_prediction(OUT/name/'conformal'/f'calibration_{fold}.csv')
            assert list(p.mbid)==list(test.index) and list(cal.mbid)==r['calibration_ids']
            np.testing.assert_array_equal(p.target,np.asarray(y,float))
            true_cal=task.df.loc[cal.mbid,task.metadata.target].to_numpy()
            np.testing.assert_allclose(cal.target,true_cal)
            np.testing.assert_allclose(cal.residual,np.abs(cal.target-cal.prediction))
            q=conformal_radius(cal.residual)
            np.testing.assert_allclose(p.lower,p.prediction-q);np.testing.assert_allclose(p.upper,p.prediction+q)
            np.testing.assert_allclose(((p.target>=p.lower)&(p.target<=p.upper)).mean(),r['coverage'])
    # Independently score the stored reference predictions on official labels.
    refs=pd.read_csv(OUT/'reference_fold_metrics.csv')
    for source in json.loads((OUT/'references/provenance.json').read_text()):
        for name in source['tasks']:
            task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
            p=read_prediction(OUT/'references'/source['model']/f'{name}_predictions.csv')
            for fold in task.folds:
                test,y=task.get_test_data(fold,include_target=True);part=p[p.fold==fold]
                assert list(part.mbid)==list(test.index);np.testing.assert_array_equal(part.target,np.asarray(y,float))
                current=scores(y,part.prediction.to_numpy(),cl)
                row=refs[(refs.model==source['model'])&(refs.task==name)&(refs.fold==fold)].iloc[0]
                for k,v in current.items():
                    if pd.notna(row.get(k)):np.testing.assert_allclose(v,row[k],rtol=1e-10,atol=1e-12)
    linear=pd.read_csv(OUT/'linear_ablation_metrics.csv')
    task=MatbenchTask('matbench_expt_gap')
    for fold in task.folds:
        x,y=task.get_test_data(fold,include_target=True)
        p=read_prediction(OUT/task.dataset_name/'linear_ablation'/f'fold_{fold}.csv')
        assert list(p.mbid)==list(x.index)
        np.testing.assert_array_equal(p.target,np.asarray(y,float))
        expected=scores(y,p.prediction.to_numpy(),False)
        row=linear[(linear.setting=='Robust_alpha10')&(linear.fold==fold)].iloc[0]
        for metric,value in expected.items():np.testing.assert_allclose(value,row[metric])
    result=dict(pass_all=True,official_tasks=TASKS,outer_folds=5,inner_folds=3,
                verified_prediction_folds=len(rows),inner_candidate_fits=inner_fits,outer_model_refits=refits,
                conformal_fits=15,linear_ablation_fits=5,total_local_fits=inner_fits+refits+20,external_reference_folds=len(refs),
                trained_baselines=FAMILIES,selection_audit='Every parameter and blend choice reconstructed from inner validation only',
                tests=['official sample identity and order','train/validation/test separation','inner score recomputation',
                       'selected configurations equal inner argmin','blends equal inner-selected member predictions',
                       'seed perturbations keep parameters fixed','official export validation','outer metric recomputation',
                       'conformal calibration separation and quantile','published reference metrics recomputed'])
    assert len(rows)==260 and inner_fits==1530 and refits==230
    return pd.DataFrame(rows),frames,pd.DataFrame(selection),result


def main():
    verified_float32_models=normalize_float32_exports()
    data,frames,selection,verification=audit()
    verification['saved_xgboost_models_replayed']=verified_float32_models
    verification['csv_precision']='XGBoost float32 values expanded losslessly to float64 text; matched saved-model inference.'
    figures=OUT/'figures';figures.mkdir(exist_ok=True)
    data.to_csv(OUT/'fold_metrics.csv',index=False);selection.to_csv(OUT/'selection_history.csv',index=False)
    metrics=['mae','rmse','r2','roc_auc','balanced_accuracy','accuracy','f1','brier','log_loss']
    summary=data.groupby(['task','representation','model'])[metrics].agg(['mean','std']).reset_index()
    summary.columns=['_'.join(filter(None,c)) for c in summary.columns]
    summary.to_csv(OUT/'summary.csv',index=False)
    v1=pd.read_csv(V1/'fold_metrics.csv');refs=pd.read_csv(OUT/'reference_fold_metrics.csv')
    meta={name:dict(MatbenchTask(name,autoload=False).metadata) for name in TASKS}
    primary=lambda n:'roc_auc' if meta[n]['task_type']=='classification' else 'mae'
    def get(name,rep,model,metric=None):
        metric=metric or primary(name)
        row=summary[(summary.task==name)&(summary.representation==rep)&(summary.model==model)].iloc[0]
        return float(row[metric+'_mean']),float(row[metric+'_std'])
    cell=lambda pair:f'{pair[0]:.4f} ± {pair[1]:.4f}'
    report=['# Topic 1 优化版：材料表示、嵌套验证与可靠性分析','',
            '## 研究问题与贡献','',
            '在相同 Matbench 官方测试折上，材料表示、训练折内模型选择和结构信息如何影响泛化？模型的误差和不确定性有哪些边界？','',
            '本版的贡献是完整、可审计的实验设计：四类基线的嵌套调参，内层验证决定单模型或双模型均值融合，等预算表示对照，训练随机性复核，以及公开强模型预测的统一重评分。这里不把组合现有方法宣称为全新算法。','',
            '**研究状态：这是观察第一版结果后的探索性改进，不是全新的盲测。** V2 的参数和融合选择只使用官方训练折内的数据；没有挑选最好的测试折、随机种子或删除失败样本。','',
            '## 老师要求与本版完成情况','',
            '| 要求 | 完成内容 |','|---|---|',
            '| 至少 5 个 Matbench 任务 | steels、jdft2d、expt_gap、expt_is_metal、glass |',
            '| 官方 train/test folds | 每任务全部官方五折；逐 ID 和顺序核验 |',
            '| 至少 3 个常规基线 | Ridge/Logistic、RF、ET、XGBoost 四类；第一版 Dummy 仍作参照 |',
            '| 为什么好／为什么差 | 表示对照、内层选择记录、失败诊断、多种子、结构对照及区间覆盖分析 |',
            '| 强模型比较（加分方向） | 复评 MODNet 五任务、CrabNet 三个回归任务的公开预测；没有在本机重训这些模型 |','',
            f'本机新增 **{verification["total_local_fits"]} 次拟合**：内层候选 {verification["inner_candidate_fits"]} 次、外层模型重拟合／种子复核 {verification["outer_model_refits"]} 次、独立区间实验 15 次、固定正则化缩放对照 5 次。公开参照另复评 40 折，不计为本机训练。','',
            '## 协议：测试集始终留在外层','',
            '1. 外层严格沿用官方五折；内层三折只切分当前外层训练数据。若训练集中存在重复组成，内层按组成分组；分类同时尽量分层。','2. 每个模型使用 3 组预先固定候选参数、相同内层划分。回归以 MAE 最小、分类以概率 ROC-AUC 最大为目标。','3. 线性模型在每个训练子集内做中位数填补、常量列过滤及 10–90% 分位区间的 RobustScaler；树模型仅填补，不需要缩放。','4. RF、ET、XGBoost 均为 128 棵树，seed=42；完整候选配置见 protocol.json。未使用测试集 early stopping 或按测试分数选择参数。','5. Selected 在内层选择最好的单模型，或表现更好的前两模型等权平均。最终仅用全部外层训练数据重拟合；每折的选择与参数完整保存。','6. A/B/C 表示沿用第一版 121/132/253 维；jdft2d 额外使用 259 维 C_structure。特征是无监督、逐样本确定性计算，拟合型处理仅在训练子集执行。','7. 不同任务的 MAE 单位不同，不取跨任务原始 MAE 均值；报告五折均值 ± 样本标准差，不把它当作置信区间。','',
            '## 主结果：固定 C 表示','',
            '| 任务 | 指标 | Linear | RF | ET | XGB | Selected |','|---|---|---:|---:|---:|---:|---:|']
    for name in TASKS:
        label='ROC-AUC ↑' if primary(name)=='roc_auc' else 'MAE ('+meta[name]['unit']+') ↓'
        report.append('| '+' | '.join([name,label]+[cell(get(name,'C',f)) for f in FAMILIES+['Selected']])+' |')
    report+=['','Selected 的家族及融合方式由内层验证决定，不是事后从上表中选测试分数最好的模型。','',selection.to_markdown(index=False),'',
             '## 与第一版的逐折比较','',
             '参照固定为第一版 ET/C；不按任务从第一版挑选测试最好的模型。改善为回归 MAE 降低量／分类 AUC 增加量，正值较好。五折相互依赖，下面只报告描述性差异与胜出折数，不声称统计显著。','']
    comparisons=[]
    fig,axes=plt.subplots(1,5,figsize=(17,3.7),layout='constrained')
    for ax,name in zip(axes,TASKS):
        metric=primary(name)
        old=v1[(v1.task==name)&(v1.representation=='C')&(v1.model=='ET')].sort_values('fold')[metric].to_numpy()
        new=data[(data.task==name)&(data.representation=='C')&(data.model=='Selected')].sort_values('fold')[metric].to_numpy()
        delta=old-new if metric=='mae' else new-old
        comparisons.append(dict(task=name,metric=metric,v1_ET_C=float(old.mean()),v2_Selected_C=float(new.mean()),improvement=float(delta.mean()),improved_folds=int((delta>0).sum())))
        ax.plot([0,1],np.vstack([old,new]),color='#b6c7d1',alpha=.65)
        ax.plot([],[],color='#b6c7d1',label='Individual fold')
        ax.plot([0,1],[old.mean(),new.mean()],marker='o',color='#006c67',linewidth=3,label='Mean')
        ax.set(xticks=[0,1],xticklabels=['V1 ET','V2 Selected'],title=name.replace('matbench_',''),ylabel='ROC-AUC' if metric=='roc_auc' else 'MAE ('+meta[name]['unit']+')')
        if name==TASKS[0]:ax.legend(fontsize=8)
        ax.grid(axis='y',alpha=.15)
    fig.savefig(figures/'v1_v2_paired.png',dpi=170);plt.close(fig)
    comparison=pd.DataFrame(comparisons);comparison.to_csv(OUT/'v1_v2_comparison.csv',index=False)
    report+=[comparison.round(5).to_markdown(index=False),'','![逐折对比](figures/v1_v2_paired.png)','']
    improved=comparison[comparison.improvement>0]
    report+=[f'本轮 Selected/C 相对 V1 ET/C 在 {len(improved)}/5 个任务的平均主指标改善。对下降任务保留完整结果；更复杂的选择流程和更多拟合不会自动带来更低测试误差。','',
             '## 表示对照与结构信息','',
             'V1 是固定参数的 A/B/C 受控实验；V2 对 ET 的三种表示使用相同 3×3 搜索预算，检验各表示经训练内调参后的表现。V2 表格同时包含参数适配效应，不能全归因于表示本身。','',
             '| 任务 | ET / A | ET / B | ET / C |','|---|---:|---:|---:|']
    for name in TASKS:report.append('| '+' | '.join([name]+[cell(get(name,r,'ET')) for r in ['A','B','C']])+' |')
    report+=['','C 中的元素属性统计来自组成及固定元素属性表，提供科学先验和不同归纳偏置，不是新增样本测量。特征数增加既可能帮助，也可能引入冗余。','',
             '**jdft2d：组成与结构对照（MAE，meV/atom）**','',
             '| 模型 | C | C_structure |','|---|---:|---:|']
    for family in FAMILIES+['Selected']:report.append(f'| {family} | {cell(get("matbench_jdft2d","C",family))} | {cell(get("matbench_jdft2d","C_structure",family))} |')
    report+=['','结构表示加入密度、原子体积、位点数、晶格长度比及角度极值，未加入剥离能衍生标签。它们不等于完整局域几何；晶胞选择和真空层也可能影响描述符，所以增益不能被解释为唯一物理机制的证明。','',
             '## 训练随机性：不挑选最好的 seed','',
             '固定各折内层已经选出的 ET 参数，追加 seed=7、2026，与 seed=42 并列。这里评估条件于同一参数的训练随机性，不包含重新调参的随机性。','',
             '| 任务／表示 | seed 42 | seed 7 | seed 2026 |','|---|---:|---:|---:|']
    for name in TASKS:
        for rep in ['C']+(['C_structure'] if name=='matbench_jdft2d' else []):
            report.append('| '+' | '.join([name+'/'+rep]+[cell(get(name,rep,f)) for f in ['ET','ET_seed7','ET_seed2026']])+' |')
    report+=['','## 强模型参照：同一批官方测试样本重新评分','',
             '以下 MODNet/CrabNet 数值来自官方仓库保存的已发表逐样本预测，本机仅核对样本并重新计算指标。其训练预算、特征、软件版本不同，没有据此声称同算力胜出或当前 SOTA。分类直接用保存的概率重算 ROC-AUC，避免旧排行榜阈值化分数的混淆。','',
             '| 任务 | 本机 Selected/C | MODNet v0.1.12 | CrabNet |','|---|---:|---:|---:|']
    for name in TASKS:
        metric=primary(name);cells=[]
        for model in ['MODNet','CrabNet']:
            values=refs[(refs.task==name)&(refs.model==model)][metric].dropna()
            cells.append(cell((values.mean(),values.std())) if len(values) else '未提供该任务预测')
        report.append('| '+' | '.join([name,cell(get(name,'C','Selected'))]+cells)+' |')
    report+=['','jdft2d 的 MODNet 使用结构信息，而 CrabNet 只用组成。上表本机 C 也只用组成；本机 C_structure 的结果另见结构对照，不能把输入信息差异当作模型优劣的唯一证据。','',
             '## 失败分析：稳健缩放解决了什么','']
    old_linear=pd.read_csv(V1/'summary.csv').query("task=='matbench_expt_gap' and representation=='C' and model=='Linear'").iloc[0].mae_mean
    new_linear=get('matbench_expt_gap','C','Linear')[0]
    report+=[f'第一版带隙 Ridge/C 的平均 MAE 为 {old_linear:.4f} eV；优化版稳健缩放＋内层正则化选择后为 {new_linear:.4f} eV。','',
             '第一版 Fe9Co7 样本的 minimum GSmagmom 在训练中近乎恒定，StandardScaler 极小尺度将其放大到约 289 万，导致巨大线性外推。V2 的稳健缩放不以该极小标准差作分母。没有删除这个样本，也没有截断目标或测试预测。','',
             '另增加固定 alpha=10 的五折对照，将缩放影响与参数选择区分开；原始诊断保留在第一版报告。','']
    linear_control=pd.read_csv(OUT/'linear_ablation_metrics.csv')
    control_table=linear_control.groupby('setting').agg(mae_mean=('mae','mean'),mae_std=('mae','std'),rmse_mean=('rmse','mean'))
    report += [control_table.round(5).to_markdown(),'']
    fixed=linear_control[linear_control.setting=='Robust_alpha10'].sort_values('fold').mae.to_numpy()
    tuned=linear_control[linear_control.setting=='Robust_tuned'].sort_values('fold').mae.to_numpy()
    if np.allclose(fixed,tuned):
        report += ['固定 alpha=10 与内层调参的稳健版本在五折中结果一致；本任务五折内层均选择 alpha=10。因此，此处观察到的主要改善来自预处理变化，而不是改变正则化强度。常量列过滤也属于稳健流程，但常量列本身不提供 Ridge 信号。','']
    for name in TASKS:
        frame=frames[name,'C','Selected'].copy();frame['abs_error']=(frame.target-frame.prediction).abs()
        frame=frame.merge(pd.read_csv(V1/name/'samples.csv').rename(columns={'id':'mbid'}),on='mbid',validate='one_to_one')
        worst=frame.sort_values('abs_error',ascending=False).head(10)
        worst.to_csv(OUT/name/'selected_top_errors.csv',index=False)
        report += [f'**{name}：Selected/C 的最大错误样本**','',worst[['mbid','formula','target','prediction','abs_error']].head(3).round(4).to_markdown(index=False),'']
    report+=['组成模型不包含热处理、制备条件或所有晶体几何；这些信息缺失是合理解释假设，不是由本次相关性实验确证的因果结论。大误差样本可指导后续材料核查，但不会从测试结果中移除。','',
             '## 预测区间：单独的 split-conformal 实验','',
             '三个回归任务，每个官方训练折随机分出 20% 作校准，其余 80% 拟合固定 ET（128 棵树，不调参）；校准标签不参与拟合或参数选择。用校准绝对残差的有限样本校正分位数构建名义 90% 区间。','',
             '这组点预测不是主表的调参模型，不能直接把主表误差和本区间拼成同一模型结果。保证依赖样本交换性；材料数据的分布偏移、重复或相关性可能使条件覆盖不可靠。','']
    conf=pd.read_csv(OUT/'conformal_metrics.csv')
    table=conf.groupby('task').agg(coverage_mean=('coverage','mean'),coverage_std=('coverage','std'),width_mean=('mean_width','mean'),point_mae=('mae','mean'))
    report += [table.round(4).to_markdown(),'','区间宽度单位与各自目标相同。较高覆盖率但很宽的区间仍可能缺乏实用价值，应同时阅读覆盖和宽度。','']
    fig,axes=plt.subplots(1,3,figsize=(13,3.5),layout='constrained')
    for ax,name in zip(axes,TASKS[:3]):
        subset=conf[conf.task==name].sort_values('fold');ax.bar(subset.fold,subset.coverage,color='#008e85')
        ax.axhline(.9,color='#c46a36',linestyle='--',label='Nominal 90%');ax.set(ylim=(0,1),xlabel='Official fold',ylabel='Empirical coverage',title=name.replace('matbench_',''));ax.legend(fontsize=8)
    fig.savefig(figures/'interval_coverage.png',dpi=180);plt.close(fig)
    report+=['![区间覆盖](figures/interval_coverage.png)','',
             '## 分类概率质量','',
             'ROC-AUC 衡量排序，不能证明概率可靠。补报 Brier score 与 log loss（均越低越好），可靠性图基于合并的折外预测，属于描述性评估。','']
    calibration=summary[(summary.task.isin(TASKS[3:]))&(summary.representation=='C')&summary.model.isin(FAMILIES+['Selected'])][['task','model','roc_auc_mean','balanced_accuracy_mean','brier_mean','log_loss_mean']]
    report += [calibration.round(4).to_markdown(index=False),'']
    fig,axes=plt.subplots(1,2,figsize=(10,4),layout='constrained')
    for ax,name in zip(axes,TASKS[3:]):
        for family in ['ET','XGB','Selected']:
            frame=frames[name,'C',family];actual,prob=calibration_curve(frame.target,frame.prediction,n_bins=8,strategy='quantile')
            ax.plot(prob,actual,marker='o',label=family)
        ax.plot([0,1],[0,1],'--',color='gray');ax.set(xlim=(0,1),ylim=(0,1),xlabel='Mean predicted probability',ylabel='Observed positive fraction',title=name.replace('matbench_',''));ax.legend()
    fig.savefig(figures/'classification_reliability.png',dpi=180);plt.close(fig)
    report+=['![概率可靠性](figures/classification_reliability.png)','',
             '## 小数据结论与展示主线','',
             '第一版固定 RF/C 的五折嵌套训练量曲线仍作为独立的小数据对照：steels 从 20% 到 100% 训练量，MAE 156.64→94.39 MPa；expt_gap 为 0.5765→0.4282 eV。这些是第一版固定参数结果，没有把 V2 用全部训练折调出的参数用于小子集后冒充纯小样本训练。','',
             '建议展示主线：材料表示问题 → 官方外层与训练内选择 → 四个基线和 Selected → 表示与结构的得失 → 失败诊断及不确定性 → 强模型参照和局限。十分钟内优先讲清实验协议、一个明确改善和一个真实失败。','',
             '## 可复核材料与局限','',
             '- 所有改进／退化、五折结果和种子结果均保留；不以测试结果挑选最好的一次作为主结论。','- 只有五个外层折且训练集重叠；折标准差与胜出折数不是显著性检验。','- V2 是基于 V1 观察的探索性迭代；若进一步提出普适性能主张，需要独立未看过的数据验证。','- 强模型结果是公开预测复评；没有在本机重现 MODNet/CrabNet 的训练，也未匹配其计算预算。','- 有限搜索预算不是对每个模型的全局最优证明；没有宣称 SOTA。','',
             '`protocol.json` 固定设计及代码哈希；`splits/` 保存内外层 ID；`runs/` 保存候选验证分数、胜出参数、内层折外预测、外层预测及 C 模型文件；`official_results/` 可由 Matbench 读取。','',
             '`summary.csv`、`v1_v2_comparison.csv`、`selection_history.csv`、`conformal_metrics.csv`、`reference_fold_metrics.csv` 提供全部表格。复现命令见仓库 `topic1/V2_README.md`。','',
             '来源：[Matbench](https://github.com/materialsproject/matbench)、[MODNet 官方归档](https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa/benchmarks/matbench_v0.1_modnet_v0.1.12)、[CrabNet 官方归档](https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa/benchmarks/matbench_v0.1_CrabNet)。']
    (OUT/'REPORT.md').write_text('\n'.join(report))
    write_json(OUT/'requirement_audit.json',verification)
    print(json.dumps(verification,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

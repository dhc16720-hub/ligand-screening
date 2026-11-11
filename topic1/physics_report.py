"""Audit V4 matched representation experiments, gating, and historical integrity."""
import json
import zipfile
import joblib
import numpy as np
import pandas as pd
from matbench.task import MatbenchTask
from topic1.run import ROOT,OUT as V1,TASKS,sha,write_json,scores
from topic1.physics import OUT,V2,V3,TARGETS,read,register,old_oofs
from topic1.physics_models import configs,gate
from topic1.refine_models import predict,select,combine
from topic1.advanced import inner_folds,mean_inner_loss
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

UNITS={'matbench_steels':'MPa','matbench_jdft2d':'meV/atom','matbench_expt_gap':'eV','matbench_expt_is_metal':'unitless','matbench_glass':'unitless'}


def audit():
    register()
    preserved={}
    for label,directory in [('V1',V1),('V2',V2),('V3',V3)]:
        manifest=json.loads((directory/'delivery_manifest.json').read_text())
        for path,checksum in manifest.items():assert sha(ROOT/path)==checksum,f'Historical change: {path}'
        preserved[label]=len(manifest)
    rows=[];decisions=[];replayed=0;feature_rows=[]
    for name in TASKS:
        task=MatbenchTask(name);cl=task.metadata.task_type=='classification'
        old_rep='C_structure' if name=='matbench_jdft2d' else 'C'
        if name in TARGETS:
            meta=json.loads((OUT/name/'features.json').read_text())
            matrix=np.load(OUT/name/'features_D.npy');previous=np.load(V1/name/f'features_{old_rep}.npy')
            assert sha(OUT/name/'features_D.npy')==meta['matrix_sha256']
            assert sha(V1/name/f'features_{old_rep}.npy')==meta['base_matrix_sha256']
            assert meta['ids']==list(task.df.index) and matrix.shape==(len(task.df),len(meta['names']))
            np.testing.assert_array_equal(matrix[:,:previous.shape[1]],previous)
            assert not np.isinf(matrix).any()
            feature_rows.append(dict(task=name,samples=len(task.df),old_dimension=previous.shape[1],new_dimension=matrix.shape[1],rows_with_missing_extra=meta['rows_with_any_extra_missing']))
            info=read(V1/name/'samples.csv').set_index('id')
        experiments=[('V4','Selected')]+([('D',f) for f in list(configs(cl))+['Selected']] if name in TARGETS else [])
        for rep,family in experiments:
            official=MatbenchTask.from_file(str(OUT/name/'official_results'/f'{rep}_{family}.json.gz'));official.validate()
            collected=[]
            for fold in task.folds:
                train,ytrain=task.get_train_and_val_data(fold);test,target=task.get_test_data(fold,include_target=True)
                folder=OUT/name/'runs'/rep/family;r=json.loads((folder/f'fold_{fold}.json').read_text());p=read(folder/f'fold_{fold}.csv')
                assert list(p.mbid)==list(test.index) and set(train.index).isdisjoint(test.index)
                np.testing.assert_array_equal(p.target,np.asarray(target,float))
                np.testing.assert_array_equal(p.prediction,[official.results[f'fold_{fold}']['data'][i] for i in test.index])
                s=scores(target,p.prediction.to_numpy(),cl)
                for key,val in s.items():np.testing.assert_allclose(val,r['metrics'][key],rtol=1e-12,atol=1e-12)
                if name in TARGETS:
                    y=np.asarray(ytrain);inner,grouped=inner_folds(y,info.loc[train.index,'formula'].to_numpy(),cl,fold)
                    manifest=dict(outer_train=list(train.index),outer_test=list(test.index),grouped=grouped,
                                  inner=[dict(train=list(train.index[a]),validation=list(train.index[b])) for a,b in inner])
                    assert manifest==json.loads((OUT/name/'splits'/f'fold_{fold}.json').read_text())
                    if rep=='D' and family!='Selected':
                        search=r['search'];assert [x['parameters'] for x in search]==configs(cl)[family]
                        old_folder=(V3 if family=='CatBoost' else V2)/name/'runs'/old_rep/family
                        old_r=json.loads((old_folder/f'fold_{fold}.json').read_text())
                        assert [x['parameters'] for x in old_r['search']]==configs(cl)[family]
                        for item in search:
                            assert len(item['inner_losses'])==3
                            np.testing.assert_allclose(item['inner_loss_mean'],np.mean(item['inner_losses']))
                        winner=min(search,key=lambda x:x['inner_loss_mean'])
                        assert r['selected_parameters']==winner['parameters']
                        oof=read(folder/f'inner_oof_{fold}.csv')
                        assert list(oof.mbid)==list(train.index)
                        np.testing.assert_array_equal(oof.target,np.asarray(y,float))
                        for j,(_,b) in enumerate(inner):assert (oof.inner_fold.iloc[b]==j).all()
                        np.testing.assert_allclose(mean_inner_loss(y,oof.prediction.to_numpy(),inner,cl),winner['inner_loss_mean'])
                        model=joblib.load(folder/f'model_{fold}.joblib')
                        actual=predict(model,matrix[task.df.index.get_indexer(test.index)],cl)
                        # Parallel forest addition order can differ at machine epsilon.
                        np.testing.assert_allclose(actual,p.prediction,rtol=1e-12,atol=1e-10)
                        replayed+=1
                    else:
                        oofs={f:read(OUT/name/'runs/D'/f/f'inner_oof_{fold}.csv').prediction.to_numpy() for f in configs(cl)}
                        preds={f:read(OUT/name/'runs/D'/f/f'fold_{fold}.csv').prediction.to_numpy() for f in configs(cl)}
                        recipe,candidates=select(y,oofs,inner,cl)
                        if rep=='D':
                            assert recipe==r['recipe'] and candidates==r['candidate_recipes']
                            np.testing.assert_array_equal(p.prediction,combine(preds,recipe))
                        else:
                            old_pool=old_oofs(name,old_rep,fold,train)
                            old_recipe,_=select(y,old_pool,inner,cl)
                            d=gate(y,combine(old_pool,old_recipe),combine(oofs,recipe),inner,cl)
                            assert d==r['gate'] and recipe==r['new_recipe'] and old_recipe==r['old_recipe']
                            old_prediction=read(V3/name/'runs'/old_rep/'Selected'/f'fold_{fold}.csv').prediction.to_numpy()
                            expected=combine(preds,recipe) if d['use_new'] else old_prediction
                            np.testing.assert_array_equal(p.prediction,expected)
                            decisions.append(dict(task=name,fold=fold,**{k:v for k,v in d.items() if k not in ['old_inner_losses','new_inner_losses','rule']},new_members='+'.join(recipe['members']),new_operation=recipe['operation']))
                else:
                    old_path=V3/name/'runs'/old_rep/'Selected'/f'fold_{fold}.csv'
                    assert r['source_sha256']==sha(old_path)
                    np.testing.assert_array_equal(p.prediction,read(old_path).prediction)
                rows.append(dict(task=name,representation=rep,model=family,fold=fold,**s));collected.extend(p.mbid)
            assert len(collected)==len(set(collected))==len(task.df) and set(collected)==set(task.df.index)
        print('AUDIT OK',name,flush=True)
    frame=pd.DataFrame(rows);frame.to_csv(OUT/'fold_metrics.csv',index=False)
    pd.DataFrame(decisions).to_csv(OUT/'gate_decisions.csv',index=False)
    pd.DataFrame(feature_rows).to_csv(OUT/'feature_coverage.csv',index=False)
    result=dict(pass_all=True,tasks=TASKS,new_representation_tasks=TARGETS,official_folds=5,verified_prediction_folds=len(rows),
                replayed_saved_models=replayed,new_inner_fits=replayed*9,new_outer_refits=replayed,total_new_fits=replayed*10,
                preserved_historical_files=preserved,gate_reconstructed=True,matched_candidate_budgets_verified=True,
                model_replay_tolerance=dict(rtol=1e-12,atol=1e-10),unchanged_tasks_exact=True)
    write_json(OUT/'requirement_audit.json',result)
    return frame,result


def report(frame,audit_result):
    comparisons=[];controls=[];summaries=[]
    for (name,rep,family),part in frame.groupby(['task','representation','model']):
        metric='roc_auc' if name in TASKS[3:] else 'mae'
        summaries.append(dict(task=name,representation=rep,model=family,metric=metric,unit=UNITS[name],mean=part[metric].mean(),std=part[metric].std()))
    summary=pd.DataFrame(summaries);summary.to_csv(OUT/'summary.csv',index=False)
    for name in TASKS:
        rep='C_structure' if name=='matbench_jdft2d' else 'C';metric='roc_auc' if name in TASKS[3:] else 'mae'
        old=np.array([json.loads((V3/name/'runs'/rep/'Selected'/f'fold_{f}.json').read_text())['metrics'][metric] for f in range(5)])
        new=frame[(frame.task==name)&(frame.representation=='V4')].sort_values('fold')[metric].to_numpy()
        gain=new-old if metric=='roc_auc' else old-new
        comparisons.append(dict(task=name,metric=metric,v3=old.mean(),v4=new.mean(),gain=gain.mean(),folds_improved=int((gain>1e-12).sum()),folds_unchanged=int((np.abs(gain)<=1e-12).sum()),new_experiment=name in TARGETS))
        if name in TARGETS:
            for family in ['ET','RF','CatBoost']:
                directory=V3 if family=='CatBoost' else V2
                before=np.array([json.loads((directory/name/'runs'/rep/family/f'fold_{f}.json').read_text())['metrics'][metric] for f in range(5)])
                after=frame[(frame.task==name)&(frame.representation=='D')&(frame.model==family)].sort_values('fold')[metric].to_numpy()
                controls.append(dict(task=name,model=family,metric=metric,old_representation=rep,old_mean=before.mean(),new_mean=after.mean(),gain=float(np.mean(after-before if metric=='roc_auc' else before-after))))
    pd.DataFrame(comparisons).to_csv(OUT/'v3_v4_comparison.csv',index=False)
    pd.DataFrame(controls).to_csv(OUT/'matched_representation_controls.csv',index=False)
    gates=read(OUT/'gate_decisions.csv');coverage=read(OUT/'feature_coverage.csv')
    (OUT/'figures').mkdir(exist_ok=True)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,name in zip(axes,TARGETS):
        for color,family in zip(['#087f8c','#a569bd','#d17c30'],['ET','RF','CatBoost']):
            row=next(r for r in controls if r['task']==name and r['model']==family)
            ax.plot([0,1],[row['old_mean'],row['new_mean']],'o-',color=color,label=family)
        ax.set_xticks([0,1],['Original','Physics D']);ax.set_title(name.replace('matbench_',''))
        ax.set_ylabel('ROC-AUC (higher better)' if name=='matbench_glass' else f'MAE, {UNITS[name]} (lower better)')
        ax.grid(alpha=.15)
    handles,labels=axes[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.5,.94),ncol=3,frameon=False)
    fig.suptitle('Matched model budgets: does physical representation help?');fig.tight_layout(rect=(0,0,1,.87))
    fig.savefig(OUT/'figures/representation_controls.png',dpi=180);plt.close(fig)
    lines=['# Topic 1：V4 物理表示优化','',
           '本版针对钢材、二维材料和玻璃形成三个任务新增物理描述符，并使用相同模型与候选搜索预算对照原表示。带隙与金属性直接沿用 V3，仍保持五个任务的完整官方五折结果。','',
           '**本版性质：**这是查看前三版结果后的探索性迭代，不是全新盲测。新增表示的适用性与切换规则在本轮训练前固定，保存在 protocol.json。继续在同一基准上迭代不能替代未触碰的外部验证集。','',
           '## 1. 主结果','',
           '|任务|指标|V3|V4|改善折数|不变折数|本轮状态|','|---|---|---:|---:|---:|---:|---|']
    for r in comparisons:
        lines.append(f"|{r['task']}|{r['metric']} ({UNITS[r['task']]})|{r['v3']:.6f}|{r['v4']:.6f}|{r['folds_improved']}/5|{r['folds_unchanged']}/5|{'新表示实验' if r['new_experiment'] else 'V3 原样保留'}|")
    lines+=['','MAE 越低越好，ROC-AUC 越高越好。主结果使用预先约定的内部切换规则，不根据外层测试集挑选有利版本。','','## 2. 新增了哪些物理信息','',
            '- **钢材与玻璃：**在原 C 的 253 维组成特征后追加五维：Miedema 金属间化合物、非晶、固溶体形成焓，以及 Yang 的热力学比值 omega 和原子尺寸失配 delta。它们是近似描述符，不是新的测量标签；Miedema 对复杂多元合金的近似也存在局限。',
            '- **二维材料：**在原 C_structure 后追加 CrystalNNFingerprint_ops 的各原子局部环境指纹的均值和标准差。相比密度、原子体积等全局量，它增加了局部配位和几何信息。使用默认近邻与半径约定，未额外求解氧化态；该表示不等同于精确化学键或完整能量计算。',
            '- **缺失值：**不删除样本、不用全数据拟合补全器。缺少元素/二元参数时保留 NaN，最终只在训练折中用中位数补全。额外检查 Yang 的二元混合焓参数，避免缺失值被其分母下限处理掩盖。','',
            '|任务|样本数|旧维数|新维数|新增特征存在缺失的样本数|','|---|---:|---:|---:|---:|']
    for r in coverage.itertuples():lines.append(f'|{r.task}|{r.samples}|{r.old_dimension}|{r.new_dimension}|{r.rows_with_missing_extra}|')
    lines+=['','特征来源：[matminer 合金描述符文档](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.composition.html)、[局部环境指纹文档](https://hackingmaterials.lbl.gov/matminer/matminer.featurizers.site.html)。实际安装版本、特征名、缺失比例、错误记录与输入 ID 均保存在本版输出中。','','## 3. 相同预算下的表示对照','',
            'RF/ET 均为 128 棵树，CatBoost 为 512 次迭代；各模型使用与其原表示版本相同的三个候选、内层三折、随机种子和官方外层折。这能比较“相同模型搜索流程下增加表示”的效果，但不同模型之间仍不是严格相同算力。参数在各表示下分别选择，因此不是固定参数的单变量试验。D 组合只有这三个模型，而 V3 组合的候选池更大；因此完整流程之间的差异不能全部归因于表示，判断表示贡献应优先看以下同模型对照。','',
            '|任务|模型|原表示均值|物理 D 均值|改善量（正为改善）|','|---|---|---:|---:|---:|']
    for r in controls:lines.append(f"|{r['task']}|{r['model']}|{r['old_mean']:.6f}|{r['new_mean']:.6f}|{r['gain']:+.6f}|")
    lines+=['','![表示对照](figures/representation_controls.png)','','## 4. 切换规则及其局限','',
            '先用内层 OOF 在三个 D 模型中选择单模型或等权组合，规则与 V3 相同。再与 V3 流程的内层 OOF 配对比较：仅当三折平均损失改善大于折间改善的标准误（样本标准差 / √3），且至少两折改善时，采用 D 流程；否则保留整个 V3 流程。','',
            '**这是减少偶然切换的启发式规则，不是显著性检验，也不保证外层结果不会变差。**内层 OOF 同时参与参数、组合与切换选择，不是独立的第二层元验证；三折之间训练数据也有重叠。外层官方折仍只用于本轮最终评价。','',
            '|任务|采用 D 的官方外层折数|保留 V3 的折数|','|---|---:|---:|']
    for name in TARGETS:
        count=int(gates[gates.task==name].use_new.sum());lines.append(f'|{name}|{count}|{5-count}|')
    lines+=['','|任务|V3 流程|直接采用 D 组合|V4 切换规则后|','|---|---:|---:|---:|']
    for name in TARGETS:
        r=next(r for r in comparisons if r['task']==name)
        raw=summary[(summary.task==name)&(summary.representation=='D')&(summary.model=='Selected')]['mean'].iloc[0]
        lines.append(f"|{name}|{r['v3']:.6f}|{raw:.6f}|{r['v4']:.6f}|")
    lines+=['','完整逐折理由见 gate_decisions.csv。未经切换规则的 D-only 组合完整展示，防止只呈现被接受的结果。新旧模型即使使用相同随机种子，特征维数变化仍会改变随机特征采样；这些对照不构成某个物理机制的因果证明。','','## 5. 对研究问题的解释','']
    for name in TARGETS:
        c=[r for r in controls if r['task']==name]
        improved=sum(r['gain']>0 for r in c)
        r=next(r for r in comparisons if r['task']==name)
        lines.append(f"- **{name}：**相同预算下 {improved}/3 个单模型的五折均值在使用新增表示后改善；预先规定的最终流程从 {r['v3']:.6f} 变为 {r['v4']:.6f}。表示对某个模型有效并不意味着对所有模型都有效；组合和切换还可能改变最终收益。")
    lines+=['','前三版的学习曲线、异常缩放诊断、多随机种子实验、conformal 区间和公开 MODNet/CrabNet 预测复核继续保留。其中特征重要性不等于因果证据，V2 的区间覆盖率也不能自动用于 V4 新模型。','',
            '## 6. 要求核对与复现','',
            '- 五个任务和官方五折完整保留；新增实验全部使用原始样本，测试集没有删点。',
            '- Linear、RF、ET、XGB 等常规基线保留；本轮额外完成 RF、ET、CatBoost 的匹配表示对照。',
            '- 新旧表示、切换记录、失败/退步结果及方法限制均公开。强模型比较仍明确标注为公开预测复核，而非本地重训。',
            f"- 新增 {audit_result['total_new_fits']} 次拟合，核验 {audit_result['verified_prediction_folds']} 份外层预测，重新加载 {audit_result['replayed_saved_models']} 个模型复核推断；随机森林并行求和仅允许机器精度级误差（rtol=1e-12，atol=1e-10）。",
            '- V1/V2/V3 交付清单哈希全部通过，历史结果没有改写。',
            '- 复现命令见 topic1/V4_README.md。完整包包含四版代码、模型与结果，不含环境目录或凭据。',
            '- 代码与实验完成不等于课程提交完成：正式幻灯片、组员登记及现场汇报仍需按课程安排完成。']
    (OUT/'REPORT.md').write_text('\n'.join(lines)+'\n')


def package():
    code=list((ROOT/'topic1').glob('*.py'))+list((ROOT/'topic1').glob('*.md'))+list((ROOT/'topic1').glob('*.txt'))+[ROOT/'topic1/python.sh']
    code+=list((ROOT/'tests').glob('test_topic1*.py'))
    fresh=[p for p in OUT.rglob('*') if p.is_file() and 'mpl-cache' not in p.parts and p.name!='delivery_manifest.json']
    write_json(OUT/'delivery_manifest.json',{str(p.relative_to(ROOT)):sha(p) for p in sorted(set(code+fresh))})
    paths=code+[p for directory in [V1,V2,V3,OUT] for p in directory.rglob('*') if p.is_file() and 'mpl-cache' not in p.parts and '__pycache__' not in p.parts]
    destination=ROOT/'outputs/Topic1_V4_完整代码模型与结果.zip'
    with zipfile.ZipFile(destination,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(set(paths)):z.write(p,p.relative_to(ROOT))
    print('PACKAGE',destination,round(destination.stat().st_size/1e6,2),'MB',flush=True)


if __name__=='__main__':
    data,result=audit();report(data,result);package()

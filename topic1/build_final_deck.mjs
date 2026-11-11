import fs from 'node:fs/promises';
import {fileURLToPath} from 'node:url';
const ROOT=fileURLToPath(new URL('..',import.meta.url)).replace(/\/$/,'');
const C={bg:'#F5F4EF',text:'#17242B',teal:'#087F8C',muted:'#536970',rust:'#B75B3F'};
function text(slide,value,x,y,w,h,size=28,bold=false,color=C.text){
 const s=slide.shapes.add({geometry:'textbox',position:{left:x,top:y,width:w,height:h},fill:'none',line:{fill:'none',width:0}});
 s.text=value;s.text.style={typeface:'Heiti SC',fontSize:size,bold,color,verticalAlignment:'top',insets:0};return s;
}
function base(p,title,number){const s=p.slides.add({width:1280,height:720});s.background.fill=C.bg;text(s,title,72,52,1136,70,44,true);text(s,String(number),1160,665,50,26,18,false,C.muted);return s;}
function table(s,values,y=180,widths=null,size=25){
 const rows=values.length,cols=values[0].length;
 const t=s.tables.add({rows,columns:cols,left:72,top:y,width:1136,height:rows*62,values,columnWidths:widths||Array(cols).fill(1136/cols)});
 t.borders.assign({style:'solid',fill:'#D0D8D8',width:1});
 for(let i=0;i<rows;i++)for(let j=0;j<cols;j++){
  const cell=t.getCell(i,j);cell.fill=i===0?'#234A56':(i%2===0?'#EAF0EF':'#FFFFFF');
  cell.text.style={typeface:'Heiti SC',fontSize:size,color:i===0?'#FFFFFF':C.text,bold:i===0,alignment:j===0?'left':'center',verticalAlignment:'middle',insets:{left:12,right:12,top:8,bottom:8}};
 }
 return t;
}
function chart(s,type,categories,series,x,y,w,h,{min,max,format='0.00',legend=true}={}){
 return s.charts.add(type,{position:{left:x,top:y,width:w,height:h},categories,series,hasLegend:legend,
  chartFill:C.bg,plotAreaFill:C.bg,legend:{position:'bottom',textStyle:{fontSize:20,typeface:'Heiti SC'}},
  xAxis:{textStyle:{fontSize:20,typeface:'Heiti SC',fill:C.text}},
  yAxis:{minimumScale:min,maximumScale:max,numberFormatCode:format,textStyle:{fontSize:18,typeface:'Heiti SC',fill:C.text},majorGridlines:{fill:'#DDE3E0',width:1}},
  barOptions:{direction:'column',grouping:'clustered'},dataLabels:{showValue:false}});
}
function notes(s,value){s.speakerNotes.text=value;}
const round=(v,m='mae')=>v.toFixed(m==='roc_auc'?4:3);

export async function addSlides({presentation,input}){
 let d=null;if(input.end>=4)d=JSON.parse(await fs.readFile(`${ROOT}/outputs/topic1_final/evidence.json`,'utf8'));
 for(let n=input.start;n<=input.end;n++){
  if(n===1){
   const s=presentation.slides.add({width:1280,height:720});s.background.fill=C.bg;
   text(s,'小样本材料性质预测',90,175,1100,110,60,true);
   text(s,'物理表示与可信评估\nSDSC6020  Topic 1',90,375,1100,100,30,false,C.muted);
   notes(s,'建议 30 秒。介绍研究问题：小样本材料任务中科学表示何时有效，以及内部高分能否迁移到新制备条件。主汇报 10 页约 10 分钟，附录用于 2 分钟讨论。');
  }
  if(n===2){
   const s=base(presentation,'五个任务与官方评价协议',2);
   table(s,[['任务','样本数','输入','预测目标'],['steels','312','组成','屈服强度 MPa'],['jdft2d','636','结构','剥离能 meV/atom'],['expt_gap','4,604','组成','实验带隙 eV'],['expt_is_metal','4,921','组成','是否为金属'],['glass','5,680','组成','玻璃形成能力']],170,[270,150,170,546],25);
   text(s,'外层固定为官方五折，内层三折负责参数和组合选择',76,585,1100,50,27,false,C.teal);
   notes(s,'建议 80 秒。先解释回归和分类任务的目标与量纲。所有外层划分均来自 MatbenchTask；组成重复时仅在训练内部分组，不修改官方外层。必做要求：至少五任务、官方 folds、至少三个常规基线、解释表现。来源：用户提供 Topics.pdf 第 2–4 页；https://doi.org/10.1038/s41524-020-00406-3 。');
  }
  if(n===3){
   const s=base(presentation,'材料表示与受控比较',3);
   table(s,[['表示','科学信息','维数'],['A','元素比例与简单化学计量','121'],['B','Magpie 元素性质统计','132'],['C','A 与 B 的组合','253'],['结构补充','全局几何量及局部配位','259 / 381'],['合金补充','Miedema 形成焓与 Yang 参数','258']],165,[190,760,186],24);
   text(s,'同模型  同候选预算  同官方划分',76,590,1100,45,29,true,C.teal);
   notes(s,'建议 55 秒。A/B/C 是主表示对照。局部环境与合金参数是额外实验；物理含义不意味着必然增加有效信息。全部表示从单个输入独立计算，缺失值只在训练折补全。参考：https://hackingmaterials.lbl.gov/matminer/featurizer_summary.html 。');
  }
  if(n===4){
   const s=base(presentation,'常规基线与最终官方结果',4);
   const rows=[['任务及指标','Linear','RF','ET','XGB','最终流程']];
   for(const r of d.tasks){const short=r.id.replace('matbench_','');rows.push([`${short}\n${r.unit}`,...['Linear','RF','ET','XGB'].map(m=>round(r.baselines[m].mean,r.metric)),round(r.mean,r.metric)]);}
   table(s,rows,165,[256,176,176,176,176,176],22);
   text(s,'MAE 越低越好，ROC-AUC 越高越好。最终流程不在所有任务上胜出。',76,580,1100,75,25,false,C.muted);
   notes(s,'建议 70 秒。基线为 V2 同表示训练内调参结果，jdft2d 用 C_structure，其余用 C；最终流程为冻结 V4。RF/ET 128 树，最终 boosting 部分 512 次迭代，跨模型计算成本不完全匹配。强调钢材最终略逊于 ET，不挑测试分数最好的模型替换主方法。来源：outputs/topic1_v2/summary.csv 与 outputs/topic1_v4/summary.csv；折间标准差见报告。');
  }
  if(n===5){
   const s=base(presentation,'小数据收益与异常外推',5);
   const r=d.learning.matbench_steels;
   chart(s,'line',r.map(x=>`${Math.round(x.fraction*100)}%`),[{name:'钢材 MAE  MPa',values:r.map(x=>x.mean),fill:C.teal}],65,170,650,390,{format:'0',legend:true});
   text(s,'Fe9Co7 异常案例',770,185,430,45,30,true);
   text(s,'真实带隙  0 eV\nV1 Ridge  约 69,528 eV',770,255,430,100,28);
   text(s,'近常数特征的训练标准差约 5.35×10⁻⁷，测试值远超训练范围。',770,380,430,130,26,false,C.rust);
   text(s,'固定 RF 学习曲线。异常样本保留，稳健缩放改善 Ridge。',76,600,1100,50,24,false,C.muted);
   notes(s,'建议 70 秒。左图使用 V1 固定 RF/C，不代表最终流程。钢材训练比例 20% 到 100% 的 MAE 明显下降。右侧 Fe9Co7 的 minimum GSmagmom 训练范围 0 至约 2.25e-5，测试值 1.548，标准化值约 2.89e6。V2 固定 alpha 控制也改善，支持缩放是关键因素。来源：outputs/topic1/learning_curve.csv 与 ridge_failure_diagnostic.json，outputs/topic1_v2/linear_ablation_metrics.csv。');
  }
  if(n===6){
   const s=base(presentation,'物理表示的收益依赖任务',6);
   table(s,[['相同候选池组合','原表示','物理 D'],...d.common_pool.map(r=>[r.label,round(r.C,r.metric),round(r.D,r.metric)])],185,[560,288,288],29);
   text(s,'RF、ET、CatBoost 使用相同配置预算与选择规则',76,500,1100,50,27,false,C.teal);
   text(s,'钢材退步；二维材料和玻璃均值改善。结果不等同于物理因果证明。',76,570,1100,75,26);
   notes(s,'建议 60 秒。这是最终补充的同候选池比较，修正了 V3 与 V4 候选池大小不同的问题。单模型对照也完整保留：钢材 0/3 改善，玻璃 3/3 改善，二维材料 2/3 改善。新旧模型参数分别在内层选择；增加特征维数还会改变随机特征采样。来源：outputs/topic1_final/validation_fold_metrics.csv 中 common_pool_C/D，V4 matched_representation_controls.csv。');
  }
  if(n===7){
   const s=base(presentation,'稳定性与重新校准的区间',7);
   table(s,[['最终流程校准','90% 区间实测覆盖','平均宽度'],...d.conformal.map(r=>[r.label,`${(r.coverage*100).toFixed(1)}%`,r.width.toFixed(2)])],160,[500,318,318],26);
   const rs=d.stability.filter(x=>x.stage==='reselection');
   text(s,'完整重选择的种子均值范围',76,455,1120,45,28,true);
   text(s,rs.map(r=>`${r.label}  ${r.minimum.toFixed(2)} 至 ${r.maximum.toFixed(2)}`).join('\n'),76,515,1120,90,28);
   notes(s,'建议 70 秒。条件重拟合覆盖五任务，完整重选择覆盖 steels 和 jdft2d，新增种子 7/2026。区间实验在每个官方训练折内划分 proper/calibration，整个选择流程仅用 proper 部分。宽度依次为 MPa、meV/atom、eV。交换性等条件未获实证保证，报告实际覆盖而不承诺任何新材料都 90% 覆盖。完整选模种子范围只是诊断，不是置信区间。参考：https://arxiv.org/abs/2107.07511 。');
  }
  if(n===8){
   const s=base(presentation,'外部实验揭示迁移失效',8);const e=d.external;
   text(s,`${e.primary.n.toLocaleString()} 条独立外部记录`,76,160,1080,55,36,true,C.teal);
   table(s,[['评价场景','ROC-AUC','阈值 0.5 表现'],['官方玻璃五折',d.tasks[4].mean.toFixed(4),'原基准评价'],['外部溅射实验',e.primary.roc_auc.toFixed(4),`Balanced accuracy ${e.primary.balanced_accuracy.toFixed(3)}`]],245,[410,250,476],28);
   text(s,'CoVZr 体系内 AUC 约 0.474；FeTiNb 只有负类，AUC 不定义。',76,475,1120,70,27,false,C.rust);
   text(s,'外部标签没有参与训练、阈值选择或校准。工艺与化学体系均发生变化。',76,575,1120,70,25);
   notes(s,'建议 80 秒，重点讲述可信度而非追逐分数。数据来源 Ren 2018 高通量溅射，共 5170 条，精确组成重叠 0 条，预先规定 L1 距离 <=0.02 的近似重叠排除 48 条。冻结 V4 流程在全体 matbench_glass 训练，最终选择 C 的 ET/RF 均值。主 AUC 约 .746，默认阈值 BA .527，概率 Brier .525。总体 AUC 还混合体系间差异。按组成簇的 bootstrap 区间只对这四个体系及固定模型条件成立。来源：https://pmc.ncbi.nlm.nih.gov/articles/PMC5898831/ 与 outputs/topic1_final/external/result.json 。');
  }
  if(n===9){
   const s=base(presentation,'强模型归档预测对照',9);const rows=[['任务','最终流程','MODNet','CrabNet']];
   for(const r of d.tasks){const refs=Object.fromEntries(d.references.filter(x=>x.task===r.id).map(x=>[x.model,x.mean]));rows.push([r.label,round(r.mean,r.metric),refs.MODNet===undefined?'无记录':round(refs.MODNet,r.metric),refs.CrabNet===undefined?'无记录':round(refs.CrabNet,r.metric)]);}
   table(s,rows,165,[410,242,242,242],25);
   text(s,'公开预测重评分，非本地重训。输入和计算预算不完全匹配。',76,580,1120,75,27,false,C.rust);
   notes(s,'建议 55 秒。三个回归任务使用各自单位 MAE，两个分类任务使用 ROC-AUC。我们核验官方测试 ID 并重新评分，处理历史分类评分定义差异。MODNet 在玻璃和带隙仍更强，因此不宣称 SOTA。来源：https://github.com/materialsproject/matbench/tree/07e50bcac791ce7660684d71e59646a1b6b82faa/benchmarks/matbench_v0.1_modnet_v0.1.12 与 matbench_v0.1_CrabNet；本地 references/provenance.json。');
  }
  if(n===10){
   const s=base(presentation,'结论与适用边界',10);
   text(s,'表示效果取决于任务和模型',76,190,1100,60,36,true,C.teal);
   text(s,'物理描述符在玻璃任务有收益，在钢材任务未改善。',76,265,1100,65,29);
   text(s,'可信预测需要外部证据',76,370,1100,60,36,true,C.teal);
   text(s,'官方高分未转化为跨制备条件的可靠分类。\n下一步需要独立数据及与任务相关的实验条件信息。',76,445,1100,115,29);
   notes(s,'建议 30 秒。主汇报 10 页合计约 10 分钟，之后 2 分钟讨论。结论需同时涵盖成功和失败。实验满足五任务、官方折、多个基线及性能分析。多轮基准迭代属于探索性研究，外部验证仅玻璃一项，完整重选择种子检查仅两个小任务。');
  }
  if(n===11){
   const s=base(presentation,'附录 训练与校准分离',11);
   table(s,[['数据部分','允许用途','禁止用途'],['Proper training','内层调参、选组合、决定切换、最终拟合','使用外层测试标签'],['Calibration','计算固定模型的残差分位数','重新选择模型或超参数'],['Official test','最终评分与覆盖率评价','参与本轮训练和校准']],185,[300,480,356],24);
   text(s,'90% 覆盖依赖交换性等条件，实际数据中的相关性与分布变化需要另行检验。',76,505,1120,100,27,false,C.rust);
   notes(s,'讨论备用页。校准没有复用旧 V4 在完整训练折上选出的参数，因为那些参数已经间接使用校准标签。我们在 proper 子集上重跑整个选择流程。分组拆分控制相同组成穿越 proper/calibration，但不自动证明样本交换性。');
  }
  if(n===12){
   const s=base(presentation,'附录 多种子检查的范围',12);
   const rows=[['任务','固定配方后重拟合','重新搜索和选择']];
   for(const task of d.tasks){const a=d.stability.find(r=>r.task===task.id&&r.stage==='conditional');const b=d.stability.find(r=>r.task===task.id&&r.stage==='reselection');rows.push([task.label,`${round(a.minimum,task.metric)} 至 ${round(a.maximum,task.metric)}`,b?`${round(b.minimum,task.metric)} 至 ${round(b.maximum,task.metric)}`:'本轮未覆盖']);}
   table(s,rows,170,[360,388,388],24);
   text(s,'范围包含种子 42、7、2026 的五折均值，不是置信区间。',76,585,1120,55,25,false,C.muted);
   notes(s,'讨论备用页。完整重选择包括参数搜索、组合与 V4 切换；两种检查都固定内层划分。不能把条件重拟合的稳定性等同于选择过程稳定性，也不能用两个附加种子支持统计显著性。');
  }
  if(n===13){
   const s=base(presentation,'附录 复现与交付',13);
   text(s,'五任务官方折与逐样本预测均可复核',76,180,1120,65,32,true);
   text(s,`补充验证 ${d.audit.total_new_fits.toLocaleString()} 次拟合\n${d.audit.validated_outer_prediction_files} 份外层预测核验\n${d.audit.replayed_bundles} 个组合模型重放`,76,280,1120,180,31,false,C.teal);
   text(s,'代码、固定协议、参数日志、模型、来源哈希及报告使用同一证据快照。',76,510,1120,85,27);
   notes(s,'讨论备用页。运行 topic1.check_delivery 检查完整性，可从已校验压缩包只恢复缺失文件，不覆盖已有修改。随后按 FINAL_README.md 复现实验。V1–V4 历史清单哈希保持不变。组员名单与 TA 登记由小组确认，不由实验代码代办。');
  }
 }
 if(input.end>=13){
  for(const idx of [5,6,8,11]){
   const t=presentation.slides.getItem(idx).tables.items[0];
   const labs=idx===5?['钢材 MAE  MPa','二维材料 MAE  meV/atom','玻璃 ROC-AUC']:idx===6?['钢材 MPa','二维材料 meV/atom','带隙 eV']:['钢材 MAE  MPa','二维材料 MAE  meV/atom','带隙 MAE  eV','金属性 ROC-AUC','玻璃 ROC-AUC'];
   labs.forEach((x,i)=>{t.getCell(i+1,0).value=x;});
  }
 }
 await presentation.render();return {total:presentation.slides.items.length};
}

"""Render-ready course report from verified evidence, using bundled python-docx."""
import json
from pathlib import Path
from docx import Document
from docx.shared import Inches,Pt,RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT,WD_CELL_VERTICAL_ALIGNMENT
from docx.oxml import OxmlElement
from docx.oxml.ns import qn

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'outputs/topic1_final'
DELIVERY=OUT/'deliverables'
D=json.loads((OUT/'evidence.json').read_text())
doc=Document();sec=doc.sections[0]
sec.page_width=Inches(8.5);sec.page_height=Inches(11)
sec.top_margin=sec.bottom_margin=Inches(.68);sec.left_margin=sec.right_margin=Inches(.7)
sec.header_distance=sec.footer_distance=Inches(.28)
for name in ['Normal','Title','Subtitle','Heading 1','Heading 2','Heading 3']:
    st=doc.styles[name];st.font.name='Heiti SC';st.font.color.rgb=RGBColor(0,0,0)
    st.element.get_or_add_rPr().rFonts.set(qn('w:eastAsia'),'Heiti SC')
for st in doc.styles:
    for border in list(st.element.iter(qn('w:pBdr'))):border.getparent().remove(border)
    for fonts in st.element.iter(qn('w:rFonts')):
        for key in list(fonts.attrib):
            if 'Theme' in key:del fonts.attrib[key]
        for key in ['ascii','hAnsi','eastAsia','cs']:fonts.set(qn('w:'+key),'Heiti SC')
normal=doc.styles['Normal'];normal.font.size=Pt(11)
normal.paragraph_format.line_spacing=1.17;normal.paragraph_format.space_after=Pt(7)
doc.styles['Title'].font.size=Pt(25)
doc.styles['Subtitle'].font.size=Pt(12)
doc.styles['Heading 1'].font.size=Pt(16);doc.styles['Heading 1'].paragraph_format.space_after=Pt(10)
doc.styles['Heading 2'].font.size=Pt(12);doc.styles['Heading 2'].paragraph_format.space_before=Pt(10)
header=sec.header.paragraphs[0];header.text='SDSC6020  Topic 1  Materials Property Prediction'
header.runs[0].font.size=Pt(8);header.runs[0].font.color.rgb=RGBColor(0,0,0)
footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.RIGHT
field=OxmlElement('w:fldSimple');field.set(qn('w:instr'),'PAGE');footer._p.append(field)


def p(text,style=None):return doc.add_paragraph(text,style)
def h(text):return doc.add_heading(text,level=2)
def page(title):doc.add_page_break();doc.add_heading(title,level=1)
def number(value,metric='mae'):
    return f'{value:.4f}' if metric=='roc_auc' else f'{value:.3f}'


def table(headers,rows,widths=None):
    t=doc.add_table(rows=1,cols=len(headers));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    if widths is None:widths=[7.1/len(headers)]*len(headers)
    for j,w in enumerate(widths):t.columns[j].width=Inches(w)
    for j,value in enumerate(headers):t.rows[0].cells[j].text=str(value)
    for row in rows:
        cells=t.add_row().cells
        for j,value in enumerate(row):cells[j].text=str(value)
    for i,row in enumerate(t.rows):
        trPr=row._tr.get_or_add_trPr();cant=OxmlElement('w:cantSplit');trPr.append(cant)
        if i==0:
            repeat=OxmlElement('w:tblHeader');trPr.append(repeat)
        for j,cell in enumerate(row.cells):
            cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            pr=cell._tc.get_or_add_tcPr();sh=OxmlElement('w:shd');sh.set(qn('w:fill'),'234A56' if i==0 else ('F0F4F5' if i%2==0 else 'FFFFFF'));pr.append(sh)
            margins=OxmlElement('w:tcMar')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:w'),'85');e.set(qn('w:type'),'dxa');margins.append(e)
            pr.append(margins);borders=OxmlElement('w:tcBorders')
            for edge in ['top','left','bottom','right']:
                e=OxmlElement('w:'+edge);e.set(qn('w:val'),'single');e.set(qn('w:sz'),'4');e.set(qn('w:color'),'D9D9D9');borders.append(e)
            pr.append(borders)
            for para in cell.paragraphs:
                para.paragraph_format.space_after=Pt(1);para.paragraph_format.space_before=Pt(1);para.paragraph_format.line_spacing=1.05
                para.alignment=WD_ALIGN_PARAGRAPH.LEFT if j==0 else WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size=Pt(10.5);run.font.bold=i==0;run.font.color.rgb=RGBColor(255,255,255) if i==0 else RGBColor(0,0,0)
    p('')
    return t


def figure(path,caption,width=6.8):
    para=p('');para.alignment=WD_ALIGN_PARAGRAPH.CENTER;para.add_run().add_picture(str(path),width=Inches(width))
    cap=p(caption);cap.paragraph_format.space_after=Pt(8)
    for run in cap.runs:run.font.size=Pt(9)


doc.add_paragraph('小样本材料性质预测',style='Title')
doc.add_paragraph('物理表示与可信评估',style='Subtitle')
p('课程研究报告  Topic 1')
team_path=OUT/'team.json'
if team_path.exists():
    team=json.loads(team_path.read_text())
    if team.get('authors'):p('作者  '+team['authors'])
h('摘要')
p('我们在五个 Matbench 任务的官方五折上，比较元素比例、元素属性统计和物理描述符对材料性质预测的影响。实验包含 Linear、Random Forest、Extra Trees、XGBoost 等常规基线，并在训练折内部进行参数与组合选择。最终流程固定为 V4，新增验证不再改变其官方主结果。')
p('主要发现是表示收益依赖任务和模型。物理描述符在玻璃任务的三个同预算单模型上均带来改善，在钢材任务上均未改善。二维材料的结构信息有帮助，但更多局部环境特征并未通过最终切换规则。独立溅射实验检验进一步表明，官方玻璃任务约 0.955 的 AUC 不能直接迁移为新制备条件下的可靠分类。')
h('最终官方结果')
table(['任务','样本','主指标及单位','五折均值','折间标准差'],[[r['label'],r['n'],r['unit'],number(r['mean'],r['metric']),number(r['std'],r['metric'])] for r in D['tasks']],[1.7,.55,1.55,1.6,1.7])
p('回归使用 MAE，越低越好；分类使用 ROC-AUC，越高越好。标准差描述五个官方折的波动，不是独立重复实验的置信区间。四轮迭代已查看过这些官方测试结果，因此本文将其明确定位为探索性基准研究。')

page('研究问题与课程要求')
p('研究问题是：在材料样本有限时，增加科学描述符是否能稳定改善传统模型，以及内部验证中的优势能否迁移到新数据。我们关注可复核的对照与失败分析，不预设复杂表示一定优于简单基线。')
table(['课程必做项','实施证据'],[
    ['至少五个 Matbench 任务','steels、jdft2d、expt_gap、expt_is_metal、glass'],
    ['官方 train/test folds','逐折调用 MatbenchTask，保存并核对训练与测试 ID'],
    ['至少三个常规基线','Linear、RF、ET、XGBoost；Dummy 作为附加检查'],
    ['解释表现好坏','表示消融、学习曲线、异常缩放案例、跨来源检验']],[2.2,4.9])
p('老师还鼓励与 MODNet、CrabNet 等强模型比较。我们复核了官方归档预测并重新评分，明确保留“公开预测复核”的性质，不将其写成本地重训或相同算力竞赛。[1–3,8–9]')
h('数据与任务差异')
p('钢材屈服强度含 312 个样本，二维材料剥离能含 636 个样本，是小数据分析的重点。实验带隙含 4,604 个样本，金属性含 4,921 个样本，玻璃形成能力含 5,680 个样本。除 jdft2d 提供晶体结构外，其余四个任务只使用化学组成。不同任务的回归量纲不同，不能直接比较 MAE 数值大小。')
p('材料描述符从单个输入独立计算。缺失值保留到训练折内再补全，没有基于目标值删除测试样本。官方外层划分始终保留，组成重复时仅在训练折内进行分组交叉验证。[2–5]')
h('课程事务')
p('课程说明要求每组 3 至 5 人，由一名成员向 TA 提交姓名和学号等信息；汇报安排为 11 月 19 日或 26 日，约 10 分钟报告和 2 分钟讨论。[1] 当前报告不代替小组登记、入选确认和现场汇报。署名与登记状态由组员提供后补入课程材料。')

page('表示与训练协议')
table(['表示','信息内容','维数或范围'],[
 ['A','118 个元素比例，加元素数、组成熵和比例平方和','121'],
 ['B','Magpie 元素性质的统计量','132'],
 ['C','A 与 B 拼接','253'],
 ['C_structure','C 加密度、原子体积、晶胞位点数和几何量','259'],
 ['D 合金','C 加 Miedema 三种形成焓、Yang omega 和 delta','258'],
 ['D 结构','C_structure 加 CrystalNN 局部环境均值和标准差','381']],[1.2,4.7,1.2])
p('Miedema 与 Yang 特征是基于外部元素或二元参数的近似描述符。CrystalNN 指纹使用默认半径和近邻约定，未额外求解氧化态。物理含义不等于物理精确性，也不保证在现有特征之外提供足够的新信息。[4–5]')
h('官方外层与训练内层')
p('外层为官方五折，目标仅用于最终评分。内层为三折：补全器、缩放器和模型全部在相应训练子集拟合。每个模型比较三个候选配置，按回归 MAE 或分类的 1−ROC-AUC 选择。选定后在完整外层训练集上重新拟合，并对外层测试集产生预测。')
p('V3 的候选池包括 V2 基线与新增 boosting 流程。组合只比较最佳单模型、前两名均值、前三名均值；回归额外比较前三名中位数。V4 另外比较 D 表示的 RF、ET、CatBoost 组合，仅在内层平均改善超过其折间标准误且至少两折改善时替换 V3。该规则是启发式，不是显著性检验。')
p('旧版部分模型为 128 棵树，新 boosting 为 512 次迭代，跨版本结果不能单独归因于算法。最终补充实验使用相同 RF、ET、CatBoost 候选池、配置预算和选择规则比较原表示与 D，消除候选池不同造成的混杂。')

page('常规基线与最终流程')
p('以下基线采用 V2 的训练内调参结果，jdft2d 使用 C_structure，其余使用 C。每个单元格为五折均值及标准差；最终列使用冻结的 V4。不同模型的树数和训练成本不完全一致。')
for indices,title in [(range(3),'回归 MAE'),(range(3,5),'分类 ROC AUC')]:
    h(title);rows=[]
    for i in indices:
        r=D['tasks'][i];row=[r['label']+'\n'+r['unit']]
        for model in ['Linear','RF','ET','XGB']:
            b=r['baselines'][model];row.append(number(b['mean'],r['metric'])+'\n± '+number(b['std'],r['metric']))
        row.append(number(r['mean'],r['metric'])+'\n± '+number(r['std'],r['metric']));rows.append(row)
    table(['任务','Linear','RF','ET','XGB','最终流程'],rows,[1.6,1.1,1.1,1.1,1.1,1.1])
p('最终流程并没有在所有任务上胜过最简单的强基线。例如钢材最终 MAE 为 81.909 MPa，而 V2 的 ET 为 81.205 MPa。这个结果完整保留，避免在看过测试分数后将最终流程替换成单个表现最好的模型。')
p('对二维材料，V3 在相同结构表示下将最终 MAE 从 V2 的 40.898 降至 36.168 meV/atom，五折均改善；该收益属于探索性比较。玻璃任务 V4 相对 V3 的 AUC 增量仅约 0.000327，不能据此宣称统计显著提升。')

page('表示消融与小数据效应')
p('使用相同 ET 搜索预算比较 A、B、C。下面结果分别以各任务主指标计量，回归越低越好、分类越高越好。')
table(['任务','A','B','C'],[[r['label']]+[number(r['values'][k],r['metric']) for k in ['A','B','C']] for r in D['representation']],[2.6,1.5,1.5,1.5])
figure(OUT/'figures/learning.png','图 1  固定 RF 基线与 C 表示的训练量曲线。误差线为外层折标准差。该曲线不代表重新调参后的最终流程。')
p('在训练量不足时，增加样本带来的收益往往比调整模型组合更明显。曲线只在两个任务和固定基线上检验这一现象，不能推广为所有材料任务的通用缩放规律。')

page('物理表示的匹配对照')
p('针对钢材和玻璃任务加入五个合金描述符；针对二维材料加入 122 个局部环境统计量。保留全部样本，玻璃任务 28 个样本的部分新增特征缺失，只在相应训练折中补全。')
table(['任务','模型','原表示','物理 D'],[[next(t['label'] for t in D['tasks'] if t['id']==r['task']),r['model'],number(r['old_mean'],r['metric']),number(r['new_mean'],r['metric'])] for r in D['physics_controls']],[2.2,1.2,1.85,1.85])
h('统一候选池后的组合比较')
table(['任务','原表示组合','物理 D 组合'],[[r['label'],number(r['C'],r['metric']),number(r['D'],r['metric'])] for r in D['common_pool']],[3.1,2,2])
p('同一候选池下，物理表示改善了二维材料和玻璃任务的组合均值，钢材仍退步。单模型的收益也不一致：二维材料 ET 改善而 RF 退步。特征维数变化会同时影响随机特征采样，因此这些结果支持“表示流程的效果依任务而异”，不能直接证明某个物理机制的因果作用。')

page('失败案例与解释边界')
h('近常数特征造成线性模型异常外推')
p('在 V1 带隙任务中，Fe9Co7 的真实带隙为 0 eV，C 表示的 Ridge 预测约为 69,528.5 eV。诊断定位到 MagpieData minimum GSmagmom：训练范围约为 0 至 2.25×10⁻⁵，训练标准差仅约 5.35×10⁻⁷，而该测试样本的特征值为 1.548。标准化后约为 2.89×10⁶，导致异常大的线性贡献。')
p('V2 使用稳健缩放，并额外运行固定 alpha 的控制实验，带隙 Ridge MAE 从约 15.85 降至约 0.743 eV。异常点没有被删除，预测也没有被人为裁剪。这说明某些“模型差”的结果来自预处理与分布外特征值的相互作用，而不能直接归因于材料规律复杂。')
h('最终回归误差仍有长尾')
p('V3 的逐样本分析显示，二维材料绝对误差最高的约 1% 样本贡献了约 24.7% 的总绝对误差。例如 Mg2SiO4 样本的目标约 1604.04 meV/atom，最终预测约 65.23。带隙任务中 MnF2 样本目标为 9.9 eV，预测约 0.551。上述样本保留在总分中。')
p('这些现象提示模型在少见材料或极端目标区间可能失效，但大误差本身不能证明标签错误。完整原因需要检查原始测量或计算条件。特征重要性只能说明模型依赖，不能替代这样的科学核验。')
h('对小样本钢材任务的认识')
p('钢材只有 312 个样本。增加合金描述符后，三个匹配单模型的均值均未改善，训练内切换规则也未采用 D。信息冗余、样本量和未提供的实验条件都可能相关，但当前对照不足以区分这些原因，因此本文保留为待验证解释。')

page('冻结流程的稳定性')
p('最终方案冻结后，使用额外随机种子 7 和 2026 进行两种检查。条件重拟合固定原来的参数、组合和切换，只改变模型随机性，覆盖五个任务。完整重选择在 steels 和 jdft2d 上重新执行参数搜索、组合选择和切换，检验小数据任务对模型随机性的敏感程度。内层划分保持固定。')
rows=[]
for r in D['stability']:
    if r['stage']=='conditional':rows.append([r['label']]+[number(r['seed_means'][str(k)] if str(k) in r['seed_means'] else r['seed_means'][k],r['metric']) for k in [42,7,2026]])
table(['条件重拟合','种子 42','种子 7','种子 2026'],rows,[2.6,1.5,1.5,1.5])
rows=[]
for r in D['stability']:
    if r['stage']=='reselection':rows.append([r['label'],number(r['minimum']),number(r['maximum']),str(r['recipe_changes'])+'/10'])
table(['完整重选择','种子均值最小值','种子均值最大值','组合发生变化'],rows,[2.5,1.5,1.5,1.6])
p('以上范围包含原种子 42，都是各自五折均值的范围。两个新增种子只能作为稳定性诊断，不构成总体置信区间。完整重选择只覆盖两个小任务，不能把结论自动扩展到其余三个任务。')
p('另对五个任务的第一个官方折重新执行冻结流程，预测与原 V4 在数值容差内一致。这一检查验证了新验证程序正确复用了原方法，避免将实现改变误认为模型随机性。')

page('重新校准的预测区间')
p('每个官方训练折进一步按组成分组，约 80% 用作 proper training，约 20% 用作校准。所有参数搜索、模型组合及物理表示切换都只使用 proper training。拟合后，用校准残差的有限样本分位数构造名义 90% 对称区间，再在原官方测试折评价。')
table(['任务','实测覆盖率','平均区间宽度','单折覆盖范围'],[[r['label'],f"{r['coverage']:.1%}",f"{r['width']:.3f}",f"{r['fold_min']:.1%} 至 {r['fold_max']:.1%}"] for r in D['conformal']],[2.2,1.4,1.7,1.8])
p('宽度单位依次为 MPa、meV/atom 和 eV。这里评估的是在 proper training 上重新选择并拟合的流程，其训练样本少于官方主结果，点预测分数不能直接当作同一模型的改进或退步。')
figure(OUT/'figures/validation.png','图 2  左侧为最终流程重新校准后的覆盖率，右侧为独立外部玻璃数据的 ROC 曲线。两项实验检验不同问题。')
p('Conformal 覆盖保证依赖交换性等条件。材料组成相关性、分组切分和实际分布变化可能使这些条件不成立，因此本文报告实际覆盖率和宽度，不承诺任意新材料都有 90% 覆盖。[7] 分类概率可靠性另用 Brier 分数和外部阈值表现分析。')

page('独立来源的迁移检验')
ex=D['external'];main=ex['primary']
p('外部数据为 Ren 等人的 glass_ternary_hipt 高通量溅射实验，共 5,170 条记录，包含四个三元化学体系。[6] 模型先仅在完整 matbench_glass 上按冻结 V4 流程选择并拟合，再加载外部数据。外部标签没有用于训练、选阈值或校准。该实验考察跨来源与跨制备条件迁移，不是同条件重复测量。')
p(f"按照训练前固定的规则，完全相同组成重叠为 {ex['n_exact_overlap']} 条；另排除与训练组成的元素分数 L1 距离不超过 0.02 的 {ex['n_nonexact_near_overlap']} 条记录，主分析保留 {main['n']} 条。")
table(['指标','外部主结果'],[['ROC-AUC',f"{main['roc_auc']:.4f}"],['Balanced accuracy',f"{main['balanced_accuracy']:.4f}"],['Accuracy 阈值 0.5',f"{main['accuracy']:.4f}"],['Brier 越低越好',f"{main['brier']:.4f}"],['外部正类比例',f"{main['positive_fraction']:.1%}"]],[3.5,3.6])
table(['体系','样本数','正类比例','ROC-AUC'],[[k,v['n'],f"{v['positive_fraction']:.1%}",'不定义' if v.get('roc_auc') is None else f"{v['roc_auc']:.4f}"] for k,v in ex['per_system'].items()],[2.5,1.1,1.7,1.8])
p('整体 AUC 仍有排序信息，但固定 0.5 阈值的 balanced accuracy 接近 0.5。CoVZr 的体系内 AUC 约 0.474，FeTiNb 全为负类，不能定义 AUC。总体指标会混合不同体系间与体系内的差异，因此不能仅用整体 AUC 判断每个体系都可用。')
p(f"按组成簇进行的条件 bootstrap AUC 95% 区间为 {ex['conditional_cluster_bootstrap_auc_95pct'][0]:.3f} 至 {ex['conditional_cluster_bootstrap_auc_95pct'][1]:.3f}。它只描述给定模型和这四个体系下的采样波动，不能作为所有新化学体系的保证。")

page('强模型对照与结论')
rows=[]
for r in D['tasks']:
    refs={x['model']:x['mean'] for x in D['references'] if x['task']==r['id']}
    rows.append([r['label'],number(r['mean'],r['metric'])]+[number(refs[m],r['metric']) if m in refs else '无该任务记录' for m in ['MODNet','CrabNet']])
table(['任务','最终流程','MODNet 归档','CrabNet 归档'],rows,[2.3,1.6,1.6,1.6])
p('上述强模型预测来自固定 Matbench 仓库归档，已核验官方测试样本并按原标签重新评分。[8–9] 该对照满足加分方向的比较需求，但没有本地重训强模型，输入信息和计算预算也不完全匹配。我们不据此宣称全面超越强模型或达到 SOTA。')
h('结论')
p('第一，组成统计和结构信息的价值依赖任务。以相同模型与预算比较，物理表示改善了玻璃任务的三个模型，而钢材的三个模型均退步。第二，严格的训练内选择有助于避免按测试分数挑选版本，但不能消除多轮查看基准带来的探索性偏差。第三，独立外部检验暴露出明显的跨条件失效，概率和决策阈值尤其不可靠。')
h('仍然存在的限制')
p('外部验证只覆盖玻璃形成能力，其余四个任务尚无独立来源确认。完整重选择的种子检查只覆盖两个小任务。标准误切换是启发式规则，复用同一组内层预测进行参数、组合和切换选择会带来选择乐观性。新增特征未必能补足实验工艺或极端样本的信息。这些限制随结果一起披露。')
p('下一步应在新的数据收集或独立制备条件下冻结验证方案，并引入与任务相关的实验条件变量。继续调整同一官方测试集上的分数，不能替代这一证据。')

page('复现与参考资料')
p(f"最终补充验证共执行 {D['audit']['total_new_fits']} 次新拟合，复核 {D['audit']['validated_outer_prediction_files']} 份外层预测文件，并重放 {D['audit']['replayed_bundles']} 个组合模型。V1–V4 文件清单哈希保持不变。详细协议、完整预测、模型、参数选择记录、校准划分及来源哈希随交付包保存。")
p('复现入口为 topic1/python.sh。先运行 topic1.check_delivery 检查文件，再按 FINAL_README.md 执行验证或重建报告。缺失文件可从已验证压缩包恢复，恢复工具不会覆盖已修改的现有文件。报告和汇报稿使用同一份 evidence.json，避免手工抄写造成数字不一致。')
h('参考资料')
for r in D['sources']:
    para=p(f"[{r['id']}] {r['title']}\n{r['url']}")
    para.paragraph_format.space_after=Pt(7)
    for run in para.runs:run.font.size=Pt(9)
h('交付范围')
p('本交付包含实验代码和结果、课程报告、约 10 分钟汇报稿及问答准备。组员姓名学号、分工、TA 登记状态和实际汇报时间仍须由小组确认；任何缺失信息均未虚构。')

DELIVERY.mkdir(parents=True,exist_ok=True)
path=DELIVERY/'Topic1_课程研究报告.docx';doc.save(path)
print(path)

# Topic 1：从公开 Fork 重新运行

这是公开下载版的首选入口。`topic1/*README.md` 是各轮冻结的历史记录，其中的 ZIP、文稿链接和本机输出不是本次 GitHub 下载的附件。新机器无需取得那些私有文件，也不需要任何 LLM API 密钥；下面的脚本会按依赖顺序生成实验输出。

## 当前版本与验证状态（2026-11-11）

| 范围 | 状态与证据 |
|---|---|
| macOS 原实验 minimum → V2 → V3 → V4 → final | 从独立克隆完成 7,481 次拟合，见下方历史记录 |
| macOS 增量研究 | 1,050 次拟合及全部模型重放；外部诊断不训练 |
| macOS 统一 `check-project` | 实测通过：6,278 个原始文件、最终 86 个模型组合及小数据 1,050 个模型重放，原文件不变 |
| 测试 | 当前 macOS 61 项：原科学测试 18 + 复现回归 14 + 小数据/外部诊断 8 + 项目验收/发布 18 + 证据完整性 3；Linux 原训练提交为 48 项 |
| 发布包独立解压核验 | v1.0.0：新克隆、新依赖环境、54 项测试及总验收通过；[证据](results/topic1-improvements/release_rehearsal.json)。该独立副本更新至 v1.0.1 并解压对应两个 ZIP 后，58 项测试、总验收、故障注入与恢复通过；[v1.0.1 证据](results/topic1-improvements/release_v101_rehearsal.json) |
| Linux minimum | 历史两次通过；本次也完成，见下方全训练证据 |
| Linux 全训练及分阶段验收 | `06d0936` 完成 8,531 次拟合及各阶段审计；原运行误取消后，`ea56c8c` 补验收成功重放 86 + 1,050 个模型并完成外部诊断。[证据与 528 项跨平台比较](results/topic1-improvements/LINUX_VALIDATION.md) |
| Linux 统一 `check-project` | **未完整执行**；原选择性附件缺少 1,510 个历史清单条目，不能冒充完整重放包。当前工作流已改为保存完整 outputs，但新配置尚未跑完整训练流程 |
| Windows/WSL2 | 未实测 |

原始实验版本保持冻结。当前复现入口增加了整体验收和发布工具；文档中的历史测试数量、旧平台记录不覆盖后来新增代码。

## 历史全链路复现记录（2026-11-11）

在独立克隆和全新虚拟环境中，使用本次公开复现入口，已完整跑通 **minimum → V2 → V3 → V4 → final**。五个基准数据集从公开源重新下载，没有复制原实验模型或特征缓存；pip 可复用下载缓存。科学源码未改动，18 项原测试和 4 项回归测试通过。

| 阶段 | 本次实际拟合次数 | 独立审计 |
|---|---:|---|
| minimum | 362（320 主/结构对照 + 40 学习曲线 + 2 诊断） | 五任务、官方五折、三个正式基线与报告通过 |
| V2 | 1,780 | 260 组逐折预测、内层搜索和参考复评分通过 |
| V3 | 900 | 120 组预测、90 个保存模型重放通过 |
| V4 | 450 | 85 组预测、45 个模型重放、匹配预算与切换重建通过 |
| final | 3,989 | 115 份外层预测、86 个模型组合重放、外部验证通过 |
| 合计 | **7,481** | 全部阶段通过 |

macOS Apple Silicon / Python 3.12.14 完成上述全链路。最终 116 份预测文件（含外部验证）与原实验在 rtol/atol=1e-10 内一致，191 项参数/组合决策无变化。V1、V3 逐折指标在该容差内一致；V2、V4、final 的少量 ROC-AUC 因近似并列预测的排序变化而不逐数相同，最大差约 1.32e-5。原始预测与分数未被舍入或修改来强行制造一致。详见 [全链路审计](results/topic1-final/reproduction_full_audit.json) 和 [实测逐折指标](results/topic1-final/reproduction_fold_metrics.csv)。

Ubuntu 24.04.5 x86_64 / Python 3.12.16 也已两次通过从零安装、22 项测试、完整 minimum 和科学审计。[最新成功运行记录](https://github.com/dhc16720-hub/ligand-screening/actions/runs/42982176504)及 [Linux 审计与跨平台比较](results/topic1-final/reproduction_linux_audit.json)可核验。以上是早期 minimum 记录。后续已补齐 Linux V2–V4/final 和小数据训练；最新验收范围及限制见文首与 [Linux 实测说明](results/topic1-improvements/LINUX_VALIDATION.md)。Windows/WSL2 尚未实测。

跨系统不保证数值逐项相同。主要表示下，Linux 与 macOS 的钢材 RF 五折平均 MAE 差约 0.132 MPa，二维材料 RF 差约 0.282 meV/atom，分类基线平均 AUC 差不超过 0.00020；完整逐折差异另存于审计。两平台源代码、协议、样本 ID 和官方划分一致，均独立通过科学审计；部分特征数组哈希不同。底层数值库、浮点计算、求解器和树分裂敏感性各自的贡献尚未完全拆分，不能将流程成功写成跨平台逐数一致。

进一步诊断确认，3 个差异特征数组的最大元素差为 2.84e-14；macOS 使用 Accelerate，Ubuntu 使用 OpenBLAS。跨平台的个别预测差异可能明显大于平均指标差异，例如本次钢材单样本最大差约 70.43 MPa、分类概率最大差 0.15。两次 Ubuntu 运行的树模型指标基本一致，部分 Linear 分类基线仍出现最大逐折 AUC 差约 0.00203。不同因素的贡献尚未完全隔离；固定依赖和随机种子不等于所有硬件上的逐数一致。请按平台对应的结果表核对，不把近似结果当成相同的逐样本预测。

缺失/空清单会拒绝通过；重复运行已完成阶段先核验完整性，不覆盖历史清单。外部数据下载和 SHA-256 校验已实测。入口范围记录见 [reproduction_check.json](results/topic1-final/reproduction_check.json)。

## 环境与安装

已验证 Apple Silicon macOS 的完整流程；Ubuntu 24.04 x86_64 已完成所有训练阶段和保存模型补验收，统一 `check-project` 的限制见文首。需要 Python 3.12、Git、Bash、网络、足够磁盘空间以及可持续运行的 CPU。Windows 可尝试 WSL2 的 Linux 路径，但当前不能标记为已验证。依赖锁包含原生科学库，不保证其他 Python 版本、CPU 架构或旧系统可用。

先安装 Python 3.12（确认 `python3.12 --version`）；若解释器不在 PATH，用实际的 Python 3.12 可执行文件绝对路径替换第一条 Python 命令。下载 ZIP 的用户解压后进入含 `README.md` 的仓库根目录，跳过 git clone。

```bash
git clone https://github.com/dhc16720-hub/ligand-screening.git
cd ligand-screening
python3.12 scripts/reproduce_topic1.py setup
bash topic1/python.sh scripts/reproduce_topic1.py test
```

`setup` 会创建独立 `.venv-topic1`，安装 `requirements-v4.lock.txt`，将官方 Matbench 检出至 `third_party/matbench` 并固定到 `07e50bcac791ce7660684d71e59646a1b6b82faa`，保留 MODNet/CrabNet 归档，检查软件版本、官方划分哈希及 OpenMP 加载。已有不同版本的 Matbench 目录会报错，不会覆盖它。`bash topic1/python.sh` 不要求文件具有可执行位，也会处理 macOS OpenMP 路径。

## 先完整满足老师的最低实验要求

```bash
bash topic1/python.sh scripts/reproduce_topic1.py minimum
```

本阶段真实运行五个完整 Matbench 任务的官方五折；A/B/C（二维材料另有 C_structure）分别比较 Linear、RF、ET 三个正式基线和额外 Dummy。含 320 次主/结构对照拟合、40 次额外学习曲线拟合、2 次失败诊断重拟合。`prepare` 会从官方登记地址下载五个数据集并生成特征，不使用本机原实验缓存。

完成后检查：

- `outputs/topic1/REPORT.md`：指标、表示消融、小样本学习曲线、好/坏案例与局限。
- `outputs/topic1/requirement_audit.json`：必须 `pass_all: true`，五任务各五折、每任务三种正式模型。
- `outputs/topic1/fold_metrics.csv` / `summary.csv` / `learning_curve.csv`：逐折与汇总结果。
- `outputs/topic1/delivery_manifest.json`：仅在科学审计成功后生成的该次运行清单。

单个任务、单折或单元测试通过不代表课程实验完成。报告生成前会独立核对官方 ID、目标、保存预测、指标、五折覆盖及全部学习曲线。

## 继续复现优化版和最终验证

按顺序执行；每步需要上一阶段的完整输出和清单：

```bash
bash topic1/python.sh scripts/reproduce_topic1.py v2
bash topic1/python.sh scripts/reproduce_topic1.py v3
bash topic1/python.sh scripts/reproduce_topic1.py v4
bash topic1/python.sh scripts/reproduce_topic1.py final
bash topic1/python.sh scripts/reproduce_topic1.py check-complete
```

也可从零执行 `bash topic1/python.sh scripts/reproduce_topic1.py all`。完整训练有数千次拟合，需要比最低要求阶段多得多的时间；脚本不虚报进度或跳过失败步骤。中断后重跑相同阶段会复用已完成折；已有清单时先校验，不重写历史清单。若改动代码或参数，请使用新的项目副本开展新实验，不要删除唯一结果或覆盖旧清单。

V2 包含 XGBoost、内层搜索、强模型归档复评分和额外验证；脚本补齐此前漏掉的 V1/V2 清单生成。V3/V4 的报告程序会自己生成清单。final 会先下载并核验独立来源玻璃数据的原始字节，再调用冻结流程；下载器不解析标签，评估器在源数据模型保存后才读取外部表。最终生成 `outputs/topic1_final/requirement_audit.json`、`validation_summary.csv`、`evidence.json` 和图表。

新运行会有自己的耗时、环境信息、模型序列化哈希及清单哈希，不应强求与原实验逐字一致。科学比较应看相同任务、划分、指标和方法；内部模型重放仍按源码规定的容差严格核验。公开 `results/topic1-final/protocol.json` 是原实验来源记录，**不要复制到新运行的 outputs**，否则历史清单哈希会冲突。

### 完整验收（含最终补充实验）

`final` / `all` 完成后会自动生成 `outputs/topic1_final/verification_manifest.json`。已有旧版完整输出的用户先运行一次：

```bash
bash topic1/python.sh scripts/reproduce_topic1.py seal-final
bash topic1/python.sh scripts/reproduce_topic1.py check-complete
```

`seal-final` 只有在科学重放成功后才建立新清单，不覆盖已有清单或历史交付清单。`check-complete` 必须已有该清单；它覆盖 V1–V4，以及 final 的 453 个必需实验/依赖文件，复核 115 份外层预测并重放 86 个模型组合，同时检查外部 Brier 分数。缺文件、文件被修改、清单不完整或科学核验失败均报错。该范围不包含可选 PPT、DOCX 或美化图表，也不代表所有操作系统通过。

科学重放在临时副本中执行，原始预测、模型、报告表和历史审计均保持原样。需额外约 300 MB 临时空间及模型推理时间，不重新训练。首次建清单证明当前产物通过独立核验，不能倒推此前从未被修改；后续清单用于检测变动。只核对哈希不能代替模型重放，因此完整验收会实际执行两者。

## 增量研究：小数据表示对照与外部失败分析

这些研究使用独立 `experiments/` 模块和输出目录，不改写 V1–V4/final 的冻结实验。安装与 minimum 完成后可运行小数据实验；外部诊断需要已经完成 final 并生成最终验证清单：

```bash
bash topic1/python.sh scripts/reproduce_topic1.py small-data
bash topic1/python.sh scripts/reproduce_topic1.py check-small-data
bash topic1/python.sh scripts/reproduce_topic1.py external-diagnostics
```

`small-data` 在五任务官方五折上进行 A/B/C × RF/ET 配对学习曲线：20%、50%、100% 的训练样本，前两档各三次嵌套抽样，共 1,050 次拟合；全量点不重复计数。默认四个进程，每个模型单线程。所有模型和预测保存于 `outputs/topic1_small_data`，需额外磁盘和训练时间。`check-small-data` 检查清单、重放全部 1,050 个模型，并核对训练内预处理与官方测试 ID。汇总先平均折内抽样重复，再计算五折描述性统计，不以重复抽样数量扩大显著性。

`external-diagnostics` 先完整核验原最终实验，再从冻结的外部预测生成分体系混淆矩阵、固定分箱可靠性图和恒定概率参照；输出在 `outputs/topic1_external_diagnostics`。它不训练模型、不改阈值、不校准，是对已查看结果的描述性分析。公开阅读入口为 [课程研究指南](COURSE_GUIDE.md)。新研究的实测平台与结果以其审计和报告为准；旧 Linux 运行记录不代表新研究已在 Linux 通过。

## 整套项目运行与验收

```bash
# 从零依次完成历史实验、新增研究和最终总验收（原 macOS 实测 8,531 次拟合）
bash topic1/python.sh scripts/reproduce_topic1.py project

# 已有完整结果时，只重放与核验，不训练
bash topic1/python.sh scripts/reproduce_topic1.py check-project
```

最终组合中重新拟合的模型数量取决于训练内选择结果；其他平台应读取该次审计中的实际拟合次数，不把原 macOS 的 8,531 次作为固定计数。

`all` 和 `check-complete` 保持原有 V1–V4/final 范围；`project` 和 `check-project` 才包含新增小数据研究与外部诊断。总验收复核完整性、官方数据身份、保存模型预测、汇总表、外部诊断来源和报告；小数据与外部重算在临时目录进行。图表文件校验完整性，数值结论通过模型重放和表格重算核对；不声称进行图像像素相等验证。

验收记录在 `outputs/project_verification/latest.json`。`project` / `check-project` 在预检查和训练之前先写 `running / pass_all: false`，成功后才写 `passed`；异常写 `failed`，强制终止留下 `running`，不会沿用之前的通过状态。记录包含阶段范围、源码清单哈希、环境、耗时及逐文件哈希。它仅代表记录中的源码与产物；发布前再次检查，变动后需重新验收。

### 资源与中断恢复

本机已验收文件合计约 **1.82 GB（十进制）**，其中 1,396 个序列化模型约 **1.55 GB**，小数据目录约 **1.13 GB**。这些是本次产物实测大小，不包含虚拟环境、依赖缓存或打包副本。建议准备至少 10 GB 可用空间；若保留多轮训练、下载包和解压副本，应另加空间。

本次 macOS Apple Silicon 总验收约 **78.2 秒**，只做重放、不训练；运行时间会随 CPU、磁盘和系统不同而变化。尚未完整记录各历史阶段的墙钟耗时，不把各模型耗时相加冒充总耗时。新增研究默认 4 个进程、每个模型单线程；低内存机器可在 `minimum` 完成后直接使用 `bash topic1/python.sh -m experiments.small_data run --workers 1`，再运行公开入口 `small-data` 完成审计和报告。

普通中断后可重跑相同阶段，脚本按已完成的折/检查点恢复。已完成检查点需通过配置和哈希核对；损坏文件会报错，不自动覆盖。若修改训练代码或协议，应在新目录重新实验。失败记录与原始输出应先保留，再定位原因，不通过删除清单绕过验收。

发布包的用途、下载、解压和校验见 [RELEASE.md](RELEASE.md)。

## GitHub 上独立运行 Linux 验证

在自己的 Fork 打开 Actions → **Topic 1 Linux reproduction** → Run workflow，选择 `minimum`、`all` 或 `project`。工作流使用 Ubuntu 24.04、Python 3.12、锁定科学依赖和固定提交的官方 Actions；仓库权限为只读，不需要 API 密钥。先安装和测试，再运行完整指定阶段，最后上传审计、报告、运行日志和数值诊断附件。`all` 是历史实验入口；`project` 还运行新增研究和总验收。提供入口或启动任务不代表验收通过。

新工作流的 Actions 附件保留 90 天；历史运行按当时的 7 天设置保留。附件不是永久归档，重要结果应及时下载。历史运行 #3 的附件为选择性证据，缺少 1,510 个原清单条目，不是完整离线重放包。当前工作流已改为保存全部 outputs，并预留超时后的收尾时间；新配置尚未完成整条运行实测。重放已发布的 macOS 模型请使用同一 Release 的源码、结果包与模型包；仓库中的复现审计 JSON 和逐折指标是长期保留的摘要。运行失败不能以某个前置步骤成功替代最终科学审计。

## 不同检查能证明什么

| 命令 / 文件 | 能证明什么 | 不能证明什么 |
|---|---|---|
| `python3 scripts/reproduce_topic1.py check-public` | 下载的源码、文档及摘要与发布清单一致 | 没有训练任何模型 |
| `... reproduce_topic1.py test` | 单元测试与环境检查通过 | 不等于五任务实验完成 |
| `... reproduce_topic1.py check-outputs` | V1–V4 清单必须存在、非空，所有登记文件哈希正确 | 单独的哈希校验不替代科学审计 |
| `... reproduce_topic1.py check-complete` | V1–V4 + final 完整性、最终科学重放和已保存汇总一致 | 不重训、不保证跨平台逐数相同、不检查可选演示文件 |
| `... reproduce_topic1.py check-project` | 历史实验 + final + 小数据 + 外部诊断统一验收，保留原文件 | 不训练、不保证跨平台逐数相同或新域性能 |
| 各阶段 `requirement_audit.json` | 已生成预测按独立审计规则通过 | 不保证新材料域上的性能 |

历史 `topic1.check_delivery` 在没有清单时会检查零文件，因此公开下载版统一使用上表的新入口；不能把旧脚本的零文件通过作为证据。新入口有回归测试，空清单、缺失文件、修改文件和越界路径均失败。

## 阅读结果与制作汇报

无需训练即可查看 [公开研究摘要](RESEARCH_SUMMARY.md)、[来源与结果表](results/topic1-final/README.md) 和 [更新记录](UPDATES.md)。这些是原实验已完成结果，不能冒充你刚刚重跑的结果。

各阶段 Markdown 报告及图表可用于制作普通 PPT，无需 Codex 或 Spaces。最终 DOCX 构建是可选项，另需 `python-docx` 和中文字体；`build_final_deck.mjs` 依赖原生 artifact session，普通 Python 环境不会生成 PPTX。原生幻灯片和老师原 PDF 未公开。

老师的四项最低实验要求：至少五任务、官方划分、至少三种常规基线、性能好坏分析；主题还要求关注科学表示和小数据。强模型比较是加分方向，本项目使用公开预测重评分，未本地重训这些强模型。课程另要求 3–5 人组队、向 TA 登记姓名学号等、约 10 分钟展示和 2 分钟讨论；课件列出 11 月 19/26 日展示，实际分组时段需小组确认。这些事务不能由代码审计代替。

## 故障处理

- `python3.12` 找不到：先安装 Python 3.12 或使用其绝对路径，不要切换到系统旧 Python。
- 下载失败：确认 GitHub、PyPI、Figshare 可访问，再重跑；校验失败不要跳过哈希检查。
- 缺清单：按阶段顺序运行，不能手写 `pass_all` 或复制原实验清单冒充重跑。
- `Protocol/code changed`：使用新项目目录；已冻结的输出不能混用修改后的训练源码。
- 不要设置 `PYTHONOPTIMIZE` 或使用 `python -O`，源码中的断言承担科学核验。

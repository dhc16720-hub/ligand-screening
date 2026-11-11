# Topic 1：从公开 Fork 重新运行

这是公开下载版的首选入口。`topic1/*README.md` 是各轮冻结的历史记录，其中的 ZIP、文稿链接和本机输出不是本次 GitHub 下载的附件。新机器无需取得那些私有文件，也不需要任何 LLM API 密钥；下面的脚本会按依赖顺序生成实验输出。

## 本次实际验证范围（2026-11-11）

独立克隆公开 Fork，创建全新虚拟环境，按锁定版本安装并通过 `pip check`。五个数据集从公开源重新下载，未复制原模型/数据输出；pip 可复用下载缓存。18 项原测试和 4 项新增回归测试通过。

`minimum` 从准备数据到报告、审计、清单完整跑通：320 + 40 + 2 次拟合；320 行逐折指标与原实验在 rtol/atol=1e-10 内一致，最大 MAE 差约 7.1e-14；分类指标完全一致。缺失清单拒绝通过，重复运行已完成 minimum 仅核验，不覆盖清单。外部数据下载及 SHA-256 校验也已实测。

V2–V4 和最终补充验证是此前已完成的实验；本次修复了这些阶段的依赖入口，但**没有在此干净副本中再训练完全部优化实验**，不能声称其全部完成新机器端到端复现。证据见 [reproduction_check.json](results/topic1-final/reproduction_check.json) 和 [本次最低要求审计](results/topic1-final/reproduction_minimum_audit.json)。

## 环境与安装

本次实际验证平台为 Apple Silicon macOS、Python 3.12。需要 Git、Bash、网络、足够磁盘空间以及可持续运行的 CPU。Linux 可按同一流程安装，但本次未在 Linux 实测；Windows 请使用 WSL2。依赖锁包含原生科学库，不保证其他 Python 版本、CPU 架构或旧系统可用。

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
bash topic1/python.sh scripts/reproduce_topic1.py check-outputs
```

也可从零执行 `bash topic1/python.sh scripts/reproduce_topic1.py all`。完整训练有数千次拟合，需要比最低要求阶段多得多的时间；脚本不虚报进度或跳过失败步骤。中断后重跑相同阶段会复用已完成折；已有清单时先校验，不重写历史清单。若改动代码或参数，请使用新的项目副本开展新实验，不要删除唯一结果或覆盖旧清单。

V2 包含 XGBoost、内层搜索、强模型归档复评分和额外验证；脚本补齐此前漏掉的 V1/V2 清单生成。V3/V4 的报告程序会自己生成清单。final 会先下载并核验独立来源玻璃数据的原始字节，再调用冻结流程；下载器不解析标签，评估器在源数据模型保存后才读取外部表。最终生成 `outputs/topic1_final/requirement_audit.json`、`validation_summary.csv`、`evidence.json` 和图表。

新运行会有自己的耗时、环境信息、模型序列化哈希及清单哈希，不应强求与原实验逐字一致。科学比较应看相同任务、划分、指标和方法；内部模型重放仍按源码规定的容差严格核验。公开 `results/topic1-final/protocol.json` 是原实验来源记录，**不要复制到新运行的 outputs**，否则历史清单哈希会冲突。

## 检查的三种含义

| 命令 / 文件 | 能证明什么 | 不能证明什么 |
|---|---|---|
| `python3 scripts/reproduce_topic1.py check-public` | 下载的源码、文档及摘要与发布清单一致 | 没有训练任何模型 |
| `... reproduce_topic1.py test` | 单元测试与环境检查通过 | 不等于五任务实验完成 |
| `... reproduce_topic1.py check-outputs` | V1–V4 清单必须存在、非空，所有登记文件哈希正确 | 单独的哈希校验不替代科学审计 |
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

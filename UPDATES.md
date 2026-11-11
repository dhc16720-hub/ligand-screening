# 更新记录

## 2026-11-11：v1.0.1 状态记录修正

公开入口在预检查和训练前即使旧成功状态失效；预检查失败、训练失败、中断或子命令零退出但没有完整审计时均不能保留通过状态。新增四项回归测试，当前共 58 项。科学源码、模型与预测未改变，v1.0.0 的真实成功验收仍有效。

## 2026-11-11：统一验收、Linux 全项目入口与可校验发布

- 新增 `project` / `check-project`，覆盖原 V1–V4/final、小数据配对实验与外部诊断；旧 `all` 范围保持明确。
- 总验收实际重放模型和重算诊断，原输出保持不变；运行前清除“上次通过”状态，失败或中断不能冒充本次通过。
- 新增发布工具：只打包已验收文件，逐文件及整包校验，拒绝覆盖已发布版本。
- Linux 工作流新增 `project` 范围、完整增量证据和 90 天附件保留；运行 #3 已启动，结果以运行记录为准。
- 最新测试、资源大小、恢复方法和发布使用说明集中在 REPRODUCING.md 与 RELEASE.md。未修改冻结科学实验与结果，未制作课件或讲稿。


## 2026-11-11 · 优化第 2–3 步：小数据表示配对研究与外部失败分析

新增独立 `experiments/` 模块。在五任务官方五折上固定 RF/ET 各 100 棵树，对比 A/B/C 与 20%/50%/100% 训练量。前两档各三次配对子集，全量点不重复拟合；完成 **1,050 次拟合、1,050 个模型重放**，逐项验证训练内补缺/标准化、样本身份、保存预测和指标。重复抽样先在折内平均，不当作独立测试折。

结果保留正负两面：钢材 ET 在 20% 训练量时 C 比 A 的 MAE 高约 10.694 MPa，五折均退步；玻璃和金属性的小样本档则显示更大的 C−A 收益。详见 [新增研究及解释](results/topic1-improvements/README.md)。首次审计发现 CSV 默认读取器的浮点往返误差，改为 round-trip 读取后仍使用严格目标相等核验；训练函数与参数不变，1,050 组模型/预测哈希全部保留，修复历史随本地完整输出保存。

外部玻璃诊断不重训、不校准、不改阈值。新增分体系混淆矩阵、可靠性图、距离分组和固定概率参照：真实正类 30.63%，预测正类 96.21%；原模型 Brier 0.52474，高于恒定 0.5 的 0.25。分别解释排序、概率与分类阈值，不掩盖外部失败。

新增 [课程入口](COURSE_GUIDE.md)，明确上游 Amadeus 界面与课程扩展的贡献边界。原平台界面不改，旧 V4 主方法和实验输出保持冻结。新命令为 `small-data`、`check-small-data`、`external-diagnostics`。本轮仅 macOS 实测；Linux 完整优化、新研究跨平台验证、Windows/WSL2、完整公开 Release 和通用 PPTX 仍待后续完成。

## 2026-11-11 · 优化第 1 步：最终实验完整验收

新增 `check-complete` 与 `seal-final`，补齐旧 `check-outputs` 仅覆盖 V1–V4 的范围。最终验收显式要求 453 个实验/依赖文件，执行 115 份外层预测核验、86 个模型组合重放及外部 Brier 复算，并与已有审计和汇总表比较。全部重放在临时副本中完成，原实验源码、模型、预测和报告不改写。

`final` / `all` 在新运行结束后自动建立最终验证清单；旧输出通过 `seal-final` 补建，已有清单不覆盖。再次运行已封存的 `final` 时先完整验收，失败即停止，不用重新生成报告掩盖损坏。

原实验目录和此前的独立复现目录均已真实通过新验收；18 项原测试及 14 项复现测试通过（本步新增 10 项）。新入口仅在 macOS 实测，旧 Linux 成功记录不能当作新入口实测。CI 增补最终预测、模型及验证清单的附件收集路径，但本步未重跑 Linux。证据见 [验收记录](results/topic1-final/complete_verification_check.json)。小样本表示实验、外部失败分析和后续交付优化仍是下一步工作，本条不声称这些已完成。

## 2026-11-11 · 补齐完整重训与独立 Linux 验证

在此前独立克隆与新环境的 minimum 结果上，继续完整重训 V2、V3、V4 和最终补充实验。五阶段共 **7,481 次拟合**，全部独立科学审计通过。最终检查 115 份外层预测、重放 86 个模型组合，外部主评估含 5,122 条样本；与原实验比较的 116 份预测均在 1e-10 容差内一致，191 项参数/组合决策无变化。

新增 Ubuntu GitHub Actions 实测：从零安装锁定依赖、22 项测试、完整五任务官方五折 minimum 和报告审计通过。工作流更新为官方 Node 24 组件并保留数值诊断附件。Linux 的完整优化流程与 Windows/WSL2 仍未实测，不作已验证声明。

公开 [全链路审计](results/topic1-final/reproduction_full_audit.json)、[Linux 审计](results/topic1-final/reproduction_linux_audit.json)和 [本次逐折指标](results/topic1-final/reproduction_fold_metrics.csv)。同平台少量 AUC 并列排序差异、跨平台数值差异均如实保留；科学源码及原实验结果未改动。当前安装与验证范围以 [REPRODUCING.md](REPRODUCING.md) 为准，以下旧条目记录当时的验证范围。

## 2026-11-11 · 公开下载版复现检查与修复

首次上传文件齐全不等于从零流程可运行。本次发现并补齐：Matbench 独立源码检出、V1/V2 交付清单生成、外部数据下载及校验、缺失/空清单拒绝通过，以及公开基线/学习曲线和文字分析证据。

新增 [统一复现入口](REPRODUCING.md)、`scripts/reproduce_topic1.py` 和 4 项回归测试；补充 [研究摘要](RESEARCH_SUMMARY.md) 与公开来源清单。冻结科学模块及原实验产物不修改。

独立克隆 + 全新虚拟环境实测通过：锁定依赖安装、官方划分检查、18 + 4 项测试、五任务官方五折和完整最低要求报告（320 主/结构对照 + 40 学习曲线 + 2 诊断拟合）。320 行逐折指标与原结果在 1e-10 容差内一致。详细边界和机器记录见 [reproduction_check.json](results/topic1-final/reproduction_check.json)。本次未重训全部 V2–V4/最终验证，未测试其他操作系统。

以下首轮发布记录保留作历史参考；新下载者统一按 REPRODUCING.md 运行。

## 2026-11-11 · Topic 1 课程研究扩展

本 Fork 基于 [nighty35628/ligand-screening](https://github.com/nighty35628/ligand-screening) 的 Amadeus 项目，新增独立的 `topic1/` 实验模块及测试。上游界面、原有案例和作者归属保留。课程实验使用 Matbench 官方五折，与原 Amadeus 案例的评估协议分别报告。

### 老师要求落实情况

| 要求 | 实现及检查依据 |
|---|---|
| 至少五个 Matbench 任务 | steels、jdft2d、expt_gap、expt_is_metal、glass |
| 使用官方 folds | 五任务各五折；审计逐样本测试 ID、目标与导出结果 |
| 至少三个常规基线 | Ridge/Logistic、Random Forest、Extra Trees；另补 XGBoost、CatBoost，Dummy 仅作参照 |
| 解释性能好坏 | A/B/C 表示消融、学习曲线、物理表示匹配对照、长尾误差、异常外推与跨来源验证 |
| 强模型比较方向 | MODNet / CrabNet 官方归档预测重评分，明确不是本地同预算重训 |

实验要求已有代码和结果证据。小组姓名学号、分工、TA 登记和实际汇报安排仍需小组完成；本仓库不声称完成这些课程事务。

### 代码与验证改进

- V1：元素比例、Magpie、组合及简单结构表示；折内预处理、常规基线、学习曲线与误差诊断。
- V2：内层三折搜索、固定预算与强模型归档比较；区间估计。
- V3：CatBoost、不同 XGBoost 目标；用训练内 OOF 预测选择单模型或组合。
- V4：合金热力学及局域结构描述符；相同预算对照、保守切换。切换规则是启发式，不能解释为显著性检验。
- 最终验证：抽取可复用训练与组合模型；条件重拟合、完整重选择、整个选择流程与校准隔离、共同候选池、独立来源验证及实现等价检查。
- 追溯能力：固定协议、源码/数据哈希、断点续跑、预测与模型重放；补充五个测试文件、锁定依赖和 macOS OpenMP 包装脚本。

最终补充实验完成 3,989 次拟合。完整本地产物审计核验 115 份外层预测、重放 86 个模型组合；五任务第 0 折另作实现等价检查。18 项 Topic 1 单元测试通过。该审计依赖完整输出，公开摘要不等同于完整审计包。

### 冻结主结果

| 任务 | 官方五折均值 | 方向 |
|---|---:|---|
| steels | MAE 81.909 MPa | 越低越好 |
| jdft2d | MAE 36.168 meV/atom | 越低越好 |
| expt_gap | MAE 0.377 eV | 越低越好 |
| expt_is_metal | ROC-AUC 0.9780 | 越高越好 |
| glass | ROC-AUC 0.9554 | 越高越好 |

逐折指标、匹配对照、最终验证、协议和审计摘要见 [results/topic1-final](results/topic1-final/README.md)。

独立来源玻璃数据剔除预设近似重叠后保留 5,122 条：ROC-AUC 0.7460，balanced accuracy 0.5273，Brier 0.5247。跨制备条件的可靠性仍明显不足。三个回归任务名义 90% 区间的实际覆盖为 92.9%、92.5%、89.8%，区间仍较宽。

### 新机器复现入口

从仓库根目录使用 Python 3.12：

```bash
python3.12 -m venv .venv-topic1
.venv-topic1/bin/python -m pip install -r topic1/requirements-v4.lock.txt
bash topic1/python.sh -m unittest tests.test_topic1 tests.test_topic1_advanced tests.test_topic1_refine tests.test_topic1_physics tests.test_topic1_final
```

安装官方 Matbench 依赖需要 Git 和网络，锁定提交为 `07e50bcac791ce7660684d71e59646a1b6b82faa`。本次发布前测试在已有本地科学环境执行，未重新进行全新操作系统安装验证。

完整实验须按 [V1](topic1/README.md) → [V2](topic1/V2_README.md) → [V3](topic1/V3_README.md) → [V4](topic1/V4_README.md) → [最终验证](topic1/FINAL_README.md) 的顺序生成依赖结果；下载数据和训练可能耗时较长。网页上传可能不保留脚本可执行位，因此文档中的 `./topic1/python.sh` 可统一写为 `bash topic1/python.sh`。

`check_delivery` 与 `final_audit` 用于已有完整输出的复核；仅克隆仓库没有输出/清单时，不能以零文件检查作为实验通过证据。报告构建另需 python-docx 与中文字体；原生演示稿构建需相应 artifact session，普通 Python 安装不会提供它。

### 本次上传范围与限制

发布源码、测试、依赖锁、启动脚本和小体积结果摘要。虚拟环境、密钥、模型、原始数据缓存、老师 PDF、私人文稿链接和完整本地 ZIP 不上传。历史文档中的报告、讲稿与压缩包是本地交付物，尚未发布到此 Fork 的 Release。

官方基准经过多轮探索性迭代，不是全新盲测；完整重选择稳定性仅覆盖两个小任务，外部验证仅覆盖玻璃任务。强模型为归档比较，不能声称同算力胜出。课程要求完成不代表这些研究局限已消失。

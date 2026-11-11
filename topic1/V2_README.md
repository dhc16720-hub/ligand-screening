# Topic 1 优化版：复现与交付说明

本版继续满足五个 Matbench 任务、官方外层五折、至少三个常规模型和性能原因分析。原版代码及 `outputs/topic1/` 结果保留；新增产物位于 `outputs/topic1_v2/`。

## 实验设计

- 四个正式基线：Ridge/Logistic Regression、Random Forest、Extra Trees、XGBoost。
- 外层仍是官方五折；只在每个外层训练折里运行三折内层验证。重复组成在内层分组，分类尽量分层。
- 每个模型三个候选配置，每个候选评估三个内层折，按回归 MAE／分类概率 ROC-AUC 选择。候选在 `advanced.py` 与输出 `protocol.json` 中固定。
- 回归正则化 alpha 为 0.1/10/100；分类 C 为其倒数。线性预处理依次为训练中位数填补、常量列过滤、RobustScaler（10–90%）。所有树模型为 128 棵树，种子 42；仅做训练内填补。
- `Selected` 根据同一组内层折外预测，选择最佳家族或前两家族的等权平均。不得用外层成绩决定单模型或融合。
- 主实验固定 C 组合特征；jdft2d 增加 C_structure。A/B 的 ET 使用同等内层搜索预算，结合 C/ET 构成优化版表示比较。
- ET 再用种子 7 和 2026 重拟合，保留原内层选择的参数，不挑选最佳种子。
- 三个回归任务另做固定 ET 的 split-conformal：官方训练折的 80% 用于拟合，20% 用于校准，名义覆盖率 90%；校准标签不用于调参。这不是主表调参模型的区间。
- 带隙任务增加固定 alpha=10 的 RobustScaler 对照，以区分缩放改进与调参影响。
- MODNet 和 CrabNet 来自固定 Matbench 官方提交的已发表预测，本机进行样本核验及重评分；明确不声称重新训练了这两个模型。

本版是看过 V1 结果后的探索性迭代，虽然 V2 所有参数选择均在训练折内，也不应被描述为全新盲测。相同官方协议方便比较，不能消除人工迭代产生的测试集适应风险。

## 环境

使用独立 `.venv-topic1`（Python 3.12）。完整依赖版本在 `requirements-v2.lock.txt`。Matbench 安装自官方提交 `07e50bcac791ce7660684d71e59646a1b6b82faa`，代码在 `third_party/matbench`，不改动官方实现。

```bash
python3.12 -m venv .venv-topic1
.venv-topic1/bin/python -m pip install -r topic1/requirements-v2.lock.txt
git clone https://github.com/materialsproject/matbench.git third_party/matbench
git -C third_party/matbench checkout 07e50bcac791ce7660684d71e59646a1b6b82faa
.venv-topic1/bin/python -m pip install ./third_party/matbench
```

Mac 请使用 `bash topic1/python.sh ...` 调用解释器，它为 XGBoost 配置当前虚拟环境自带的 OpenMP 路径，不依赖全局 Homebrew 配置。

## 运行顺序

从仓库根目录执行。已有完整第一版结果时，直接从优化版命令开始；空环境需先依照 `topic1/README.md` 完成第一版，生成特征与基准结果。

```bash
bash topic1/python.sh -m unittest discover -s tests -p 'test_topic1*.py'
bash topic1/python.sh -m topic1.advanced main
bash topic1/python.sh -m topic1.advanced ablation
bash topic1/python.sh -m topic1.advanced robustness
bash topic1/python.sh -m topic1.conformal
bash topic1/python.sh -m topic1.linear_ablation
bash topic1/python.sh -m topic1.references
bash topic1/python.sh -m topic1.advanced_report
```

`main/ablation/robustness` 支持断点续跑及 `--tasks` 子集选项。修改训练代码或候选配置后应归档旧结果并使用新的输出目录；协议哈希会阻止混合结果。

## 实际工作量

| 部分 | 拟合次数 |
|---|---:|
| 主 C 和 jdft2d C_structure：6 × 4 家族 × 5 外折 ×（9 内层 + 1 重拟合） | 1200 |
| ET 的 A/B 表示对照：5 × 2 × 5 ×（9 内层 + 1 重拟合） | 500 |
| ET 额外两个随机种子：6 × 5 × 2 | 60 |
| 三个回归任务的区间实验：3 × 5 | 15 |
| 带隙固定 alpha 缩放对照 | 5 |
| 总计 | 1780 |

Selected 使用已有预测，不额外拟合；强模型复评 40 折不计入本机训练次数。

## 结果与审计

- `REPORT.md`：中文方法、完整结果、失败分析、强模型比较与局限。
- `requirement_audit.json`：完整性与内外层选择审核。
- `fold_metrics.csv` / `summary.csv`：所有逐折结果与均值、样本标准差。
- `selection_history.csv`：每折内层决定的家族或融合成员。
- 每任务 `splits/`：外层和内层样本 ID；`runs/`：候选搜索记录、内层折外预测、测试预测，C/C_structure 的训练模型文件。
- 每任务 `official_results/`：官方 Matbench 可读取的结果。
- `references/`：强模型公开预测和固定来源；`reference_fold_metrics.csv` 是重新计算后的指标，不照抄旧榜单。
- `conformal_metrics.csv`、`linear_ablation_metrics.csv`：不确定性和失败机制的独立对照。
- 第一版的学习曲线继续作为固定参数小数据实验，不把全量训练调参信息带入小子集实验。

报告不以测试成绩挑选最漂亮的配置；完整保留退化结果。五折标准差不是置信区间，多种子复核也不替代外部验证。分类概率的排序与校准分别评估。

## 分享包

完整交付 ZIP 包含代码、版本锁、原始数据备份、两版结果表、逐折预测、候选搜索记录、图表及 120 个 C/C_structure 模型文件（约 149 MB）。不包含虚拟环境或官方仓库的完整克隆；新机器按上方步骤安装环境。

结果审计会重放 30 个 XGBoost 模型，验证其 float32 预测与保存值一致，并将 CSV 扩展为保留原值的 float64 文本精度。此操作只规范保存格式，不改变模型或预测语义。

默认运行会复用已完成的拟合。若要从头重训 V2，请先将 `outputs/topic1_v2` 改名归档，再依次运行全部 V2 命令。保留 `outputs/topic1`，其中包含本版复用的无监督特征和第一版参照结果。

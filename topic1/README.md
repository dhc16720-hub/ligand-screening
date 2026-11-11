# Topic 1 课程实验

研究问题：在固定常规模型与官方评测划分下，材料表示如何影响预测，尤其在小样本条件下？

## 老师的要求与实现

1. 五个官方任务：matbench_steels、matbench_jdft2d、matbench_expt_gap、matbench_expt_is_metal、matbench_glass。
2. 每项完整执行 Matbench 官方固定五折，直接调用 `MatbenchTask.get_train_and_val_data` 与 `get_test_data`，不用原 Amadeus 的随机／分组外层划分替代。
3. 每个任务、每种表示比较三个正式基线：回归 Ridge、Random Forest、Extra Trees；分类 Logistic Regression、Random Forest、Extra Trees。Dummy 只是额外参照，不计入三种模型。
4. 分析性能好坏：A/B/C 受控表示对比，steels/expt_gap 学习曲线，jdft2d 结构信息补充对照，RF/C 的误差分组和典型错误样本。

## 固定实验设计

| 表示 | 内容 | 维度 |
|---|---|---:|
| A | 全部 118 种元素比例，元素数、组成熵、比例平方和 | 121 |
| B | Matminer Magpie 元素属性统计 | 132 |
| C | A+B | 253 |
| C_structure | 仅 jdft2d：C 加 6 个简单结构描述符 | 259 |

元素属性统计是由组成计算的确定性变换；C 增加的是表示能力，不是独立物理信息。特征字典、维度和缺失率保存到各任务 `features.json`。

Ridge alpha=10；逻辑回归 C=1、最大迭代 3000；RF/ET 都是 100 棵树、max_features=1.0、min_samples_leaf=1。seed=42。不做超参数搜索，参数在查看测试结果前固定。

中位数填补和标准化在每个训练折内拟合；特征提取本身不拟合数据分布，也不使用目标。不删除测试样本、不截断异常标签、不改变官方折、不从测试集筛选模型。

回归主指标 MAE，辅以 RMSE、R²；分类主指标概率 ROC-AUC，辅以 balanced accuracy、accuracy、F1。阈值固定 >0.5。汇总标准差为样本标准差 ddof=1；官方导出的内部 std 可能使用不同自由度，比较时以原始逐折指标为准。

主实验 5 任务 × 3 表示 × 4 模型 × 5 折 = 300 次拟合，其中正式基线为 225 次。jdft2d 的结构补充为 20 次。学习曲线 2 任务 × 4 个额外训练比例 × 5 折 = 40 次，100% 点复用主实验，总计 360 次。

学习曲线固定使用 RF/C；训练比例 20/40/60/80/100%，每折的子集按同一随机序列嵌套，测试集不变。只对两个指定回归任务执行，避免分类小子集缺少类别。

## 本机复现

从仓库根目录运行：

```bash
.venv-topic1/bin/python -m unittest discover -s tests -p test_topic1.py
.venv-topic1/bin/python -m topic1.run prepare
.venv-topic1/bin/python -m topic1.run run
.venv-topic1/bin/python -m topic1.run learning --tasks matbench_steels matbench_expt_gap
.venv-topic1/bin/python -m topic1.diagnose
.venv-topic1/bin/python -m topic1.report
```

所有产物在 `outputs/topic1/`；运行阶段可断点续跑。`protocol.json` 保存代码与官方折文件哈希；如修改参数或训练代码，应将旧结果目录改名归档后开始新实验，不要混合结果。

新机器需要 Python 3.12。建立 `.venv-topic1` 后安装 `requirements.lock.txt`，再安装固定官方 Matbench 源码：

```bash
python3.12 -m venv .venv-topic1
.venv-topic1/bin/python -m pip install -r topic1/requirements.lock.txt
git clone https://github.com/materialsproject/matbench.git third_party/matbench
git -C third_party/matbench checkout 07e50bcac791ce7660684d71e59646a1b6b82faa
.venv-topic1/bin/python -m pip install ./third_party/matbench
```

课程环境与工作台 `.venv-allis` 独立。实验不需要 DeepSeek 或其他 LLM 密钥，不会调用收费模型。

## 审核与结果

`topic1.report` 独立读取已保存预测，重新核对官方 ID、目标值、指标、每样本测试次数和官方结果格式；只有全部期望任务和配置齐全，才会生成完整报告。学习曲线还会核验每折预测与测试 ID。

- `REPORT.md`：中文结果、原因分析与 10 分钟展示安排。
- `requirement_audit.json`：老师要求的可复核检查结果。
- `summary.csv`、`fold_metrics.csv`：汇总与逐折指标。
- `learning_curve.csv`、`figures/`：学习曲线和图表。
- 各任务 `runs/`：逐折测试预测、真实标签、误差、模型参数。
- 各任务 `official_results/`：由官方 `record/validate/to_file` 导出的完整五折结果。
- `environment.json`、`protocol.json`：软件版本、源码提交和协议证据。
- `ridge_failure_diagnostic.json`：带隙 Ridge 的极端外推诊断，额外重拟合 2 次原配置用于核对特征贡献，不改变正式实验。

结果用于课程对照研究。没有独立外部集来评估事后挑选的最好配置；不宣称 SOTA。结构信息、工艺条件等材料解释是待进一步验证的假设。

## 来源

- [Matbench 官方项目](https://github.com/materialsproject/matbench)
- [Matbench 论文](https://doi.org/10.1038/s41524-020-00406-3)
- [Matminer](https://github.com/hackingmaterials/matminer)

Matbench 官方 README 提醒：旧分类排行榜把预测阈值化后计算 ROC-AUC，实际上等价于 balanced accuracy。本项目采用当前源码的概率 ROC-AUC，因此不能与未重算的旧榜单直接比较。

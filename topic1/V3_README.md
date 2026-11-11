# V3：损失函数与 CatBoost 优化

本版为 V1/V2 后续探索性实验。V1/V2 文件与模型不修改。

## 本地复现

在项目根目录执行：

```bash
./topic1/python.sh -m unittest tests.test_topic1 tests.test_topic1_advanced tests.test_topic1_refine
./topic1/python.sh -m topic1.refine
./topic1/python.sh -m topic1.refine_report
```

运行器支持 `--tasks matbench_steels` 等子集，已完成折次可续跑。源码与协议哈希不匹配时停止，修改训练方案应另开输出目录。进程使用四个模型线程；不需要 API 密钥。

## 方法

- 五任务官方五折不变，沿用 V2 训练折内的三折划分和重复组成分组规则。
- 每个新模型三个候选，各三次内层拟合；最优候选再拟合整个外层训练集。
- CatBoost 用于全部任务；三个回归任务增加 XGB_Squared、XGB_MAE、XGB_Log。
- jdft2d 额外重复结构表示；共 90 个最终新模型，810 次内层拟合，合计 900 次。
- 主表示事先固定：jdft2d C_structure，其余 C；jdft2d C 作为补充消融。
- 新旧模型共同参与内层选择，比较最佳单模型、前二均值、前三均值，回归额外比较前三中位数。最终配置不由外层成绩决定。
- 目标变换可逆且接受负值；内层始终按原始单位 MAE 或 ROC-AUC 选择。
- 每个 boosting 模型 512 次迭代；与 V2 128 树不是等算力比较。三种新 XGBoost 流程之间训练预算相同。
- 所有预测转换为 float64 后再保存与评分，避免 float32 CSV 十进制精度差异。

## 交付与依赖

`outputs/Topic1_V3_新增实验与模型.zip` 是增量包，含 V3 全部模型、预测、审计报告和相关代码。复现实验需先解压 V2 完整包到同一目录：V3 复用其原始数据、确定性特征、固定划分及四个旧模型的内层/外层预测。只阅读报告与检查新模型文件不需重新训练。

新环境请按 `requirements-v3.lock.txt` 安装依赖，Matbench 已固定为官方仓库提交 `07e50bcac791ce7660684d71e59646a1b6b82faa`，无需替换本机路径。安装该依赖需要 Git 及网络。`topic1/python.sh` 默认使用项目内 `.venv-topic1`（Python 3.12），macOS 上自动设置 OpenMP 库路径。

`refine_report.py` 在写报告前核验官方 ID/划分、每折指标、参数选择、组合选择、90 个保存模型的预测重放及 V1/V2 历史清单哈希。正式交付不能把公开预测复核写成本地重训，也不能将这次迭代宣称为未见测试集上的盲测或新的 SOTA。

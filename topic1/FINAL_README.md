# Topic 1 最终补充验证与课程交付

本轮冻结 V4 的主方法和官方结果，仅补充验证。不要按新的测试分数改写最终方法。所有旧版清单保持不变。

## 已完成的要求

| 老师要求 | 实施与证据 |
| --- | --- |
| 至少 5 个 Matbench 任务 | steels、jdft2d、expt_gap、expt_is_metal、glass |
| 官方 train/test folds | 五任务均使用官方五折，逐样本测试 ID 与目标核对 |
| 至少 3 个常规基线 | Linear、RF、ET、XGBoost；另有 Dummy |
| 分析为什么好或不好 | A/B/C 表示消融、学习曲线、物理特征匹配对照、Ridge 异常外推、长尾误差、外部迁移检验 |
| 强模型比较方向 | MODNet 和 CrabNet 官方归档预测重评分；不是本地重训，不是相同算力比较 |
| 约 10 分钟汇报 | 10 页主体、3 页问答附录，含讲者备注和时间分配 |

课程事务仍需小组确认：3–5 人名单、学号、分工、TA 登记和实际汇报安排。未虚构署名，未向 TA 发送消息。

## 新增验证

- 条件重拟合：5 任务 × 5 官方折 × 新种子 7/2026，固定原选择的参数与组合。
- 完整重选择：steels、jdft2d 两任务 × 5 折 × 新种子 7/2026，重跑搜索、组合及切换，内层划分保持固定。
- 最终流程区间：三个回归任务，每个官方训练折按组成分组为 proper training/calibration；只在 proper 部分进行整个选择与拟合，用校准残差构造名义 90% 区间。
- 公平表示对照：C（jdft2d 为 C_structure）和 D 使用相同 RF/ET/CatBoost 候选池、配置预算与组合规则。
- 独立来源验证：冻结流程先在全部 matbench_glass 上训练，再评价 Ren 2018 高通量溅射数据。5,170 条中 0 条精确重叠，预设 L1 距离 ≤0.02 的近似重叠剔除 48 条，主结果保留 5,122 条。
- 实现等价检查：五任务官方第 0 折从头训练，预测与 V4 在严格容差内一致。

新增 3,989 次拟合：3,789 次内层候选拟合、200 次最终分量重拟合。审计核验 115 份外层预测并重放 86 个组合模型，18 项单元测试通过。等价检查的 5 个模型另在专用检查中验证。

## 阅读顺序与文件

1. `outputs/topic1_final/deliverables/Topic1_课程研究报告.docx`：12 页研究报告。
2. `outputs/topic1_final/deliverables/presentation.json`：可编辑原生演示文稿的位置，打开后可导出 PPTX。
3. `outputs/topic1_final/deliverables/Topic1_汇报讲稿与问答.md`：排练稿与问答。
4. `outputs/topic1_final/evidence.json`：报告与演示文稿共同的数字来源。
5. `outputs/topic1_final/requirement_audit.json`：机器可复核的实验审计。
6. `outputs/topic1_final/validation_fold_metrics.csv` 与 `validation_summary.csv`：完整补充验证结果。
7. `outputs/topic1_final/protocol.json`、各任务的 `selection.json`、`inner_oof.csv`、预测及模型：固定协议和逐折证据。

`outputs/Topic1_最终实验与报告_完整交付.zip` 包含 V1–V4 和最终补充验证的代码、模型、数据、报告及讲稿，无需逐个叠加历史包。演示文稿为 Spaces 原生文件，压缩包保存其来源信息、内容快照和构建代码；包内没有本地 PPTX。虚拟环境、应用 UI、账号配置、API 凭据及渲染中间文件不打包。

## 本机复核

在项目根目录使用已安装的 `.venv-topic1`，不要用系统 Python 直接加载模型：

```bash
./topic1/python.sh -m topic1.check_delivery
./topic1/python.sh -m unittest tests.test_topic1 tests.test_topic1_advanced tests.test_topic1_refine tests.test_topic1_physics tests.test_topic1_final
./topic1/python.sh -m topic1.final_audit
./topic1/python.sh -m topic1.final_report_data
```

文件缺失时可恢复，已有修改不会被覆盖：

```bash
python3 -m topic1.check_delivery --restore-missing-from outputs/Topic1_最终实验与报告_完整交付.zip
```

`final_audit` 会校验旧版清单、官方测试 ID 和目标、选择记录、OOF 组合、校准隔离、残差分位数、外部组成排除规则、原始数据哈希以及模型预测重放。它不重新搜索超参数。

## 重新运行实验

```bash
./topic1/python.sh -m topic1.final_equivalence
./topic1/python.sh -m topic1.final_validation conditional
./topic1/python.sh -m topic1.final_validation reselection
./topic1/python.sh -m topic1.final_validation conformal
./topic1/python.sh -m topic1.final_validation common_pool
./topic1/python.sh -m topic1.final_external
./topic1/python.sh -m topic1.final_audit
```

已有完成标记的结果会复用；需要真正从零重跑时，在独立解压副本中先移走最终验证的任务结果目录和 external 目录，保留固定协议与原始数据，勿删除唯一交付副本。修改冻结训练源码会触发协议校验失败，方法变更应新建输出版本。完整重跑耗时较长。

新机器使用 Python 3.12，按 `requirements-v4.lock.txt` 重建科学环境。官方 Matbench 固定为 `07e50bcac791ce7660684d71e59646a1b6b82faa`。五个官方原始缓存位于 `outputs/topic1/data/`；离线时复制到所用环境的 `matminer/datasets/`。外部缓存位于 `outputs/topic1_final/datasets/`。macOS 的 `python.sh` 设置所需 OpenMP 路径。不同平台浮点或底层库可能造成小差异，不能因此跳过来源或测试 ID 检查。

## 重建正式材料

报告构建器 `topic1/build_final_report.py` 使用 python-docx；本次通过 Codex 内置文档运行时生成，并用 `render_docx.py` 渲染全部 12 页进行目视检查。中文字体为 Heiti SC。当前 headless 渲染器需要显式提供中文字体搜索路径；字体只用于本机 QA，不随包分发。重新排版后应重新检查分页、中文和表格。

演示文稿构建器 `topic1/build_final_deck.mjs` 导出 `addSlides({presentation,input})`，供原生 artifact session 使用。在新的空文稿移除默认空白页后，按 `{start:1,end:13}` 构建；不要重复往已有完整文稿追加。内容与数字来自 evidence.json，表格和图表可编辑。

## 结论与边界

最终官方均值：钢材 MAE 81.909 MPa、二维材料 36.168 meV/atom、带隙 0.377 eV；金属性 AUC 0.9780、玻璃 AUC 0.9554。物理表示并非普遍有效。

区间实际覆盖为 92.9%、92.5%、89.8%，平均宽度分别为 499.953 MPa、194.372 meV/atom、2.130 eV。覆盖保证依赖交换性等条件，不能承诺任意新材料都满足。

外部玻璃 AUC 0.7460，默认阈值 balanced accuracy 0.5273，Brier 0.5247，说明跨来源与跨制备条件的概率和分类可靠性明显不足。该数据仅覆盖四个体系；FeTiNb 只有负类，AUC 不定义。组成簇 bootstrap 只给出固定模型与这四个体系下的条件波动，不能推广为新化学体系保证。外部结果没有用于重新调参、选阈值或校准。

多轮官方基准迭代属于探索性研究；完整重选择稳定性仅覆盖两项，外部验证仅覆盖一项，强模型为归档比较。以上限制在报告、汇报和结果中一致保留。

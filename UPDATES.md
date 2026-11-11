# 更新记录

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

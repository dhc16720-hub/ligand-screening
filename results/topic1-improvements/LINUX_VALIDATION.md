# Linux 实测与跨平台差异（2026-11-11）

## 已完成范围

[原运行](https://github.com/dhc16720-hub/ligand-screening/actions/runs/43019918648)在 Ubuntu 24.04 x86_64、Python 3.12.16 上从零安装，完成 minimum、V2、V3、V4、final 和小数据研究，共 **8,531 次拟合**。各阶段科学审计通过：最低要求覆盖五个官方任务、官方五折和 Linear/RF/ET 三种正式基线；最终验证重放 86 个模型组合，小数据研究审计全部 1,050 个模型。该运行固定在 `06d0936`，包含当时的 48 项测试。

原工作流在全部训练和小数据报告完成后，因维护时误取消而中止，整体状态是 **cancelled**。外部诊断和统一 `check-project` 当时没有完成，不能将它描述为整条工作流成功。

[补验收运行](https://github.com/dhc16720-hub/ligand-screening/actions/runs/43044885944)固定在 `ea56c8c`，通过官方 Artifact 下载并校验原证据包，不重新训练。它再次重放最终 86 个模型组合与小数据 1,050 个模型，完成外部诊断，并确认所有可用原始证据未改变。补验收整体成功。新旧实验的科学源码未改变；[源码对照](linux_release_source_comparison.json)区分训练提交与发布提交。

**限制：Linux 完整 `check-project` 仍未完成。** 原工作流选择性归档了 4,764 个文件，其中 4,762 个清单条目可逐字校验；原清单还有 1,510 个条目未收进附件，包括部分历史辅助文件和模型。补验收明确报告这些缺失，不补造文件、重写原清单或冒充完整覆盖。macOS 完整 `check-project` 已通过。当前工作流已改为归档整个 `outputs`，并将实验步骤限时设为 330 分钟，为 360 分钟作业上限内的证据上传预留时间；这项新归档配置尚未重新执行整条训练工作流。

## 数值比较

五任务的样本记录和官方划分逐字一致。16 组基础特征中，3 组数组字节不同，最大元素差为 `2.842170943040401e-14`。跨平台不保证逐数一致，微小输入和底层数值差异对树分裂、模型选择等环节的贡献未被完全分离。

V4 正式选择结果的五折平均值如下；详细比较保留原始精度，展示表格仅作四舍五入。

| 任务 | 指标 | macOS | Linux |
|---|---|---:|---:|
| steels | MAE / MPa | 81.9089 | 82.2543 |
| jdft2d | MAE / meV/atom | 36.1677 | 36.1676 |
| expt_gap | MAE / eV | 0.37702 | 0.37287 |
| expt_is_metal | ROC-AUC | 0.97804 | 0.97824 |
| glass | ROC-AUC | 0.95541 | 0.95552 |

总计核对 **528 项数值**，不是所有结果都接近到同一程度。例如最终补充实验中，钢材的某项平均 MAE 差约 1.94 MPa、区间平均宽度差约 17.62 MPa；二维材料某项平均 MAE 差约 1.51 meV/atom。完整结果按阶段、任务和指标分别记录，不混合不同单位求一个总误差。

小数据研究的 90 组配对平均改善值没有发生正负方向翻转。20% 训练数据下，钢材 ET 使用 C 相对 A 的平均 MAE 增加约 10.65 MPa（macOS 为 10.69 MPa）；完整数据下 C 的平均改善约 0.90 MPa（macOS 为 0.87 MPa）。这是描述性复现，不是独立样本显著性检验。

外部玻璃失效结论保留：Linux 主评估 AUC 为 0.74608，Brier 为 0.53501（macOS 为 0.52474），均明显差于固定 0.5 概率的 Brier 0.25；预测正类比例两端均为 96.21%。没有用外部标签改阈值、校准或重新训练。

## 复核入口

- [训练与阶段审计](linux_training_audit.json)
- [补验收范围与结果](linux_replay_audit.json)
- [比较摘要](linux_cross_platform_comparison.json)、[528 项比较明细](linux_cross_platform_metrics.csv)、[分任务最大差异](linux_cross_platform_max_differences.csv)
- [特征差异](linux_feature_comparison.json)

`scripts/compare_platform_evidence.py` 可对同版本 macOS 输出与已合并原始、补验收附件的 Linux 输出执行 `--recovered-evidence` 比较。该选项要求补验收记录成功、原表格哈希正确及样本/划分一致；它不会把证据子集标为完整 `check-project`。Windows/WSL2 仍未实测。

原 Linux 运行的历史 `all` 阶段（含审计）约 2 小时 11 分钟，小数据训练、审计和报告约 17 分钟。这是该次 GitHub 托管环境的墙钟记录，不是对其他机器的速度保证。

# Topic 1 公开结果摘要

原实验摘要逐字复制自冻结产物，来源映射见 `provenance.json`，SHA-256 可核验。`reproduction_*` 文件记录独立克隆后的重新训练和跨平台验证，应与原实验分开阅读。V4 是主方法；final 为补充验证，不用补充测试结果重新挑选主方法。

| 文件 | 内容 |
|---|---|
| baseline_v1_summary.csv / minimum_requirement_audit.json | 三个正式基线、五任务官方五折最低要求证据 |
| baseline_v2_summary.csv / reference_fold_metrics.csv | 优化基线、表示消融与强模型归档重评分 |
| learning_curve.csv | 两任务固定 RF 的小样本学习曲线 |
| reproduction_check.json / reproduction_minimum_audit.json | 本次干净克隆的实际验证范围与五任务审计 |
| reproduction_full_audit.json | macOS 全链路 7,481 次拟合、各阶段独立审计及与原结果比较 |
| reproduction_linux_audit.json | 独立 Ubuntu minimum 运行、平台信息及跨平台差异 |
| reproduction_fold_metrics.csv | 本次实测逐折指标；platform/run_stage 区分平台和阶段，final 的 stage 区分补充实验；不含耗时 |
| source_manifest.json | 公开源码、文档和摘要的完整性清单，不代表训练通过 |
| v4_summary.csv / v4_fold_metrics.csv | V4 各方法汇总与官方逐折指标 |
| matched_representation_controls.csv | 相同模型、搜索预算的物理表示对照 |
| gate_decisions.csv | 内层切换判定 |
| validation_summary.csv / validation_fold_metrics.csv | 种子稳定性、区间及共同候选池验证 |
| external_result.json | 独立来源玻璃数据的指标、重叠排除及分体系结果 |
| protocol.json | 冻结的最终验证方案与来源哈希 |
| requirement_audit.json | 本地完整产物审计结果 |

这里是可浏览的摘要，不包含逐样本预测、完整数据缓存和序列化模型。审计 JSON 记录本地完整环境的检查，不能单凭这些摘要独立重放全部模型。完整实验需按 [更新说明](../../UPDATES.md) 重建输出。历史交付文档中的 outputs 路径、报告及 ZIP 指本地交付物，本次未发布为 GitHub Release。

多轮官方基准迭代属于探索性研究；强模型是归档预测重评分；外部迁移仅验证玻璃任务，且概率可靠性不足。请保留这些限制，不把官方五折均值解释为新材料体系的保证。

公开下载的运行入口见 [REPRODUCING.md](../../REPRODUCING.md)，完整文字分析见 [研究摘要](../../RESEARCH_SUMMARY.md)。

# Topic 1 公开结果摘要

这些文件逐字复制自本地冻结实验产物，来源映射见 `provenance.json`，SHA-256 可核验。V4 是主方法；final 为补充验证，不用补充测试结果重新挑选主方法。

| 文件 | 内容 |
|---|---|
| v4_summary.csv / v4_fold_metrics.csv | V4 各方法汇总与官方逐折指标 |
| matched_representation_controls.csv | 相同模型、搜索预算的物理表示对照 |
| gate_decisions.csv | 内层切换判定 |
| validation_summary.csv / validation_fold_metrics.csv | 种子稳定性、区间及共同候选池验证 |
| external_result.json | 独立来源玻璃数据的指标、重叠排除及分体系结果 |
| protocol.json | 冻结的最终验证方案与来源哈希 |
| requirement_audit.json | 本地完整产物审计结果 |

这里是可浏览的摘要，不包含逐样本预测、完整数据缓存和序列化模型。审计 JSON 记录本地完整环境的检查，不能单凭这些摘要独立重放全部模型。完整实验需按 [更新说明](../../UPDATES.md) 重建输出。历史交付文档中的 outputs 路径、报告及 ZIP 指本地交付物，本次未发布为 GitHub Release。

多轮官方基准迭代属于探索性研究；强模型是归档预测重评分；外部迁移仅验证玻璃任务，且概率可靠性不足。请保留这些限制，不把官方五折均值解释为新材料体系的保证。

# Topic 1 增量研究

本目录为公开摘要与图表。完整模型、逐样本预测及实验历史保存在本地 outputs，下载者可按 REPRODUCING.md 重新生成；本目录不冒充完整模型交付包。

- [小数据配对研究报告](small-data/REPORT.md)：五任务、官方五折、A/B/C × RF/ET × 三档训练量，1,050 次拟合及模型重放。
- [研究结论与解释](INTERPRETATION.md)：正结果、负结果和解释边界。
- [外部失败分析](external/REPORT.md)：分体系混淆矩阵、概率可靠性和固定参照，不重新训练或调阈值。

原 V4 主方法保持冻结。这些是此前已查看基准上的探索性补充，不是新的盲测或同算力 SOTA 竞赛。新增两种模型用于表示对照；课程至少三个正式基线的要求仍由原 minimum/V2 实验覆盖。

本轮在 macOS Apple Silicon 实测。Linux 完整优化实验、Windows/WSL2、新增增量研究的跨平台验证仍未完成。小数据审计发现并修复了 CSV 默认读取器的浮点往返精度问题，保留严格目标相等检查；原训练函数、模型与预测未改变，修复记录见 small-data/audit_reader_revision.json。

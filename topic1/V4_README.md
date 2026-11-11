# V4：物理描述符与保守切换

本版在 V3 之后进行探索性迭代，不修改前三版文件。

## 实验范围

- 钢材和玻璃形成：原 C 加 Miedema 三维形成焓、Yang omega/delta，共 258 维。
- 二维材料：原 C_structure 加 CrystalNNFingerprint_ops 的原子环境均值、标准差。
- 在三个任务上训练 RF、ET、CatBoost。每个模型三个候选 × 内层三折 + 一次外层重拟合；官方五折，共 450 次拟合、45 个最终模型。
- RF/ET 128 棵树，CatBoost 512 次迭代，与旧表示相应模型的配置、搜索候选、内层划分和随机种子一致。
- 内层选择 D 模型组合，再根据配对损失改善是否超过其标准误且至少两折改善来决定是否替换 V3。规则是启发式，不是显著性检验，不能保证测试成绩提升。
- 带隙和金属性不重训，逐条原样保留 V3 预测。
- 全部新增特征只从输入计算，缺失值保留至折内补全，不删除样本。

## 运行

在项目根目录、既有 `.venv-topic1` 环境中：

```bash
./topic1/python.sh -m unittest tests.test_topic1 tests.test_topic1_advanced tests.test_topic1_refine tests.test_topic1_physics
./topic1/python.sh -m topic1.physics
./topic1/python.sh -m topic1.physics_report
```

支持 `--tasks matbench_jdft2d` 等子集。已完成的折次可续跑，源码和协议变化会停止执行；新方案应另设版本。结构特征采用多进程，请通过上述模块命令执行，不要从标准输入脚本启动多进程。原始数据、特征、旧预测和模型由完整交付包提供；新环境需安装 `requirements-v4.lock.txt`，其官方 Matbench 依赖固定到提交 `07e50bcac791ce7660684d71e59646a1b6b82faa`。

Matbench 的原始下载缓存随包保存在 `outputs/topic1/data/`。若新机器无法联网，先把其中五个 `<task>.json.gz` 复制到该 Python 环境的 `matminer/datasets/` 目录，再运行。相关版本与校验值见各版环境和清单。macOS 上 `python.sh` 会设置 sklearn 随附的 OpenMP 库路径。

## 核验与交付

`physics_report.py` 独立核对旧版哈希、新特征矩阵及样本顺序、匹配搜索预算、参数选择、模型重放、D 组合与最终切换、官方导出和未改任务的完全一致性。

`outputs/Topic1_V4_完整代码模型与结果.zip` 包含 V1–V4 实验代码、原始缓存、特征、模型、预测和报告，不需要逐个叠加旧增量包。它不包含虚拟环境、API 凭据、原应用的 UI，也不包含官方 Matbench 的完整仓库历史。安装依赖仍需要相应运行环境。

报告：`outputs/topic1_v4/REPORT.md`；原始逐折结果：`fold_metrics.csv`；表示对照：`matched_representation_controls.csv`；切换理由：`gate_decisions.csv`；协议与校验：`protocol.json`、`requirement_audit.json`、`delivery_manifest.json`。

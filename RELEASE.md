# Topic 1 可核验发布包

发布包用于保存可追溯的课程研究成果。原 Amadeus 平台归属上游；课件、讲稿和组员行政信息由小组另行办理，不在本发布包范围。

## 下载与用途

在本 Fork 的 [Releases](https://github.com/dhc16720-hub/ligand-screening/releases) 选择 `topic1-v1.0.0`：

- GitHub 自动提供的 **Source code (zip)**：该标签对应的完整仓库源码，包括原平台与课程扩展。只需从零复现时，下载这一份即可。
- **topic1-v1.0.0-results.zip**：逐样本预测、指标、特征、实验协议、审计和研究图表；含用于重放的原始实验记录。
- **topic1-v1.0.0-models.zip**：1,396 个已训练模型。与结果包、同版本源码配套使用。
- **release.json / SHA256SUMS.txt**：每个附件的字节数、SHA-256、文件数和已验证平台。

附件中的模型来自已完成的 macOS 实验。下载和验收已有模型不等于在本机重新训练；Linux 的新训练记录另行列明，不与 macOS 模型混合。不要加载不明来源或被修改的模型文件。

## 下载后核验

将两个 ZIP、`release.json` 和 `SHA256SUMS.txt` 放在同一目录。源码解压后，先用 Python 3.12 的标准库校验附件及内部每个文件，无需安装科学依赖：

```bash
python3.12 scripts/package_topic1_release.py verify /absolute/path/to/downloads
```

通过后，将结果包和模型包解压到包含 `README.md` 的同一仓库根目录。两包各有独立的内部清单，完整科学产物清单也保存在各阶段 outputs 中。若不需要已有输出，直接按 REPRODUCING.md 从零运行。

```bash
python3.12 scripts/reproduce_topic1.py setup
bash topic1/python.sh scripts/reproduce_topic1.py check-project
```

模型重放按已验证环境及代码中的容差执行；其他系统不保证与 macOS 逐数相同，失败时保留日志并参照平台验证记录。

## 维护者重新打包

```bash
bash topic1/python.sh scripts/reproduce_topic1.py check-project
bash topic1/python.sh scripts/package_topic1_release.py build \
  --version topic1-v1.0.0 --output outputs/releases/topic1-v1.0.0
```

打包器只收录最近成功总验收登记的文件；源码清单或产物变化则拒绝打包。打包时重新校验实际字节，打包后读取 ZIP 内全部文件复核，并生成外层 SHA-256。已有版本目录和附件不能覆盖；修订时使用新版本号。发布时应把 GitHub 标签固定到与包内源码清单相符的提交。

发布包含已有验证成果，不表示所有操作系统都通过，也不表示外部泛化问题已经解决。平台状态和完整命令见 [REPRODUCING.md](REPRODUCING.md)。

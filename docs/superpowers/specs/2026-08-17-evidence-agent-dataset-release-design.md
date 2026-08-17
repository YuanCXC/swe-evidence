# evidence_agent_dataset_v1 分卷发布设计

## 背景

`data/evidence_agent_dataset_v1` 包含 4 个文件。只有
`repository_runtime.sqlite3` 超过 GitHub Release 的 2 GiB 单文件限制，
其大小约为 62 GiB。两个 Parquet 文件分别约为 449 MiB 和 212 MiB，
虽然超过普通 Git 的 100 MiB 单文件限制，但均可作为单个 Release
附件上传。

本方案只将 SQLite 文件按原始字节分卷。两个 Parquet 文件作为单独的
Release 附件上传。仓库保存数据集原始 `manifest.json`、发布清单、恢复
脚本和使用说明。恢复后必须得到与本地源文件逐字节一致的 SQLite 和
Parquet 文件。

## 发布范围

源目录为 `data/evidence_agent_dataset_v1`，包含：

- `manifest.json`：直接提交到 Git 仓库。
- `policy_evidence.parquet`：作为单个附件上传到 GitHub Release。
- `repository_runtime.sqlite3`：分卷上传到 GitHub Release。
- `tasks.parquet`：作为单个附件上传到 GitHub Release。

工作区中的其他修改、数据和模型不属于本次发布范围。

## SQLite 分卷格式

- 每个完整分卷固定为 1,900 MiB，即 1,992,294,400 字节。
- 最后一个分卷保存剩余字节，可以小于固定大小。
- 分卷不压缩，内容是源文件中对应区间的原始字节。
- 预计生成 34 个分卷。
- 分卷名称包含源文件名、从 1 开始的序号和总卷数，例如：
  `repository_runtime.sqlite3.part-00001-of-00034`。
- 按序号升序直接拼接所有分卷，即可还原源文件。

不使用压缩可以避免压缩流损坏扩大影响范围，也能让中断后的分卷生成、
上传和下载独立重试。

## 分卷清单

仓库中的 `release_manifest.json` 使用版本化 JSON 结构，记录：

- 清单格式版本和 GitHub Release 标签。
- 固定分卷大小。
- 每个源文件的发布模式、相对路径、总字节数和 SHA-256。
- SQLite 每个分卷的名称、序号、字节数和 SHA-256。
- 两个 Parquet 单附件的名称、字节数和 SHA-256。

清单采用稳定排序和 UTF-8 编码，便于审查和重复生成。

## 工具接口

提供一个仅依赖 Python 标准库的命令行工具，支持以下操作：

- `split`：流式读取 SQLite，生成分卷；同时计算全部发布文件的清单。
- `verify-parts`：检查 SQLite 分卷是否齐全，并校验每卷及拼接字节流的 SHA-256。
- `merge`：按清单顺序流式拼接 SQLite，写入临时文件；校验成功后原子替换目标文件。
- `upload`：通过 GitHub CLI 上传 SQLite 分卷和两个完整 Parquet；同名且大小一致的远端附件自动跳过。
- `download`：通过 GitHub CLI 下载清单要求的 SQLite 分卷和两个完整 Parquet。
- `verify-files`：校验下载或恢复后的 3 个数据文件。

`merge` 只合并 SQLite，不处理两个 Parquet。它不会覆盖未经确认的现有
文件。缺卷、重复卷、大小不符、分卷哈希不符或最终文件哈希不符时，
命令以非零状态退出，并保留现有目标文件。

## 发布与恢复流程

发布流程如下：

1. 对小样本执行分卷和合并测试，确认红绿测试循环通过。
2. 对真实 SQLite 生成分卷，并为 3 个数据文件生成 `release_manifest.json`。
3. 运行 `verify-parts`，校验全部真实分卷及完整拼接字节流。
4. 提交原始 `manifest.json`、发布清单、工具、测试和使用说明，并推送发布分支。
5. 创建草稿 GitHub Release，上传 SQLite 分卷和两个完整 Parquet。
6. 使用本地分卷执行一次真实 SQLite 合并，并校验全部文件的 SHA-256。
7. 对照发布清单检查远端附件的大小和 GitHub SHA-256 摘要。
8. 发布 GitHub Release，并创建草稿 Pull Request。

恢复流程如下：

1. 克隆仓库并安装 GitHub CLI 与 Python 3.10 或更高版本。
2. 下载指定 Release 的 SQLite 分卷和两个完整 Parquet。
3. 运行 `merge`，将 SQLite 恢复到 `data/evidence_agent_dataset_v1`。
4. 运行 `verify-files`，校验合并后的 SQLite 和下载的两个 Parquet。

## 测试与验收

自动化测试至少覆盖：

- SQLite 单卷样本和多卷样本可以逐字节恢复。
- 最后一个分卷长度正确。
- 缺少分卷时拒绝合并。
- 分卷内容损坏时拒绝合并。
- 最终文件已存在时默认拒绝覆盖。
- 合并失败时不留下伪装成完整文件的输出。
- 清单序列化结果稳定。

真实数据验收标准如下：

- 清单中 3 个数据文件的字节数与源文件一致。
- SQLite 恰好生成 34 个分卷，两个 Parquet 各对应 1 个完整附件。
- 所有 Release 附件均小于 2 GiB。
- `verify-parts` 对全部分卷返回成功。
- 合并后的 SQLite 及下载的两个 Parquet 的 SHA-256 与源文件一致。
- Git 提交不包含工作区中的其他修改。

## 失败恢复

- SQLite 分卷使用确定性名称；已有且校验通过的分卷可以复用。
- 上传失败后可重新执行 `upload`，已上传的分卷或完整附件不会重复上传。
- 下载失败后可重新执行 `download`，已下载且校验通过的附件可以复用。
- 合并写入同目录临时文件，只有最终校验通过后才替换目标路径。
- 临时分卷和合并验证输出在发布完成后删除，并明确报告清理范围。

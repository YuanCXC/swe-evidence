# evidence_agent_dataset_v1

该数据集通过 GitHub Release 发布。只有约 62 GiB 的
`repository_runtime.sqlite3` 被拆成 100 MiB 分卷；两个 Parquet
文件保持为完整附件。

## 文件布局

- `manifest.json`：数据集原始清单。
- `release_manifest.json`：Release 附件、大小和 SHA-256 清单。
- `policy_evidence.parquet`：单个 Release 附件。
- `repository_runtime.sqlite3.part-*`：SQLite 原始字节分卷。
- `tasks.parquet`：单个 Release 附件。

## 下载与恢复

需要 Python 3.10 或更高版本，以及已经登录的 GitHub CLI。

在仓库根目录运行：

```powershell
python scripts/release_dataset.py download `
  --manifest data/evidence_agent_dataset_v1/release_manifest.json `
  --download-dir data/evidence_agent_dataset_v1

python scripts/release_dataset.py merge `
  --manifest data/evidence_agent_dataset_v1/release_manifest.json `
  --parts-dir data/evidence_agent_dataset_v1 `
  --output-dir data/evidence_agent_dataset_v1

python scripts/release_dataset.py verify-files `
  --manifest data/evidence_agent_dataset_v1/release_manifest.json `
  --files-dir data/evidence_agent_dataset_v1
```

`merge` 按序拼接 SQLite 分卷，先写入临时文件。只有最终字节数和
SHA-256 均与发布清单一致时，工具才会生成
`repository_runtime.sqlite3`。两个 Parquet 不需要合并。

Release 地址：
[evidence-agent-dataset-v1](https://github.com/YuanCXC/swe-evidence/releases/tag/evidence-agent-dataset-v1)。

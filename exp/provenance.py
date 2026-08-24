"""冻结实验配置身份，并保护 JSONL 断点续跑。"""

from __future__ import annotations

import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_path(path: Path | str) -> str:
    """计算文件或目录的稳定 SHA-256。"""

    target = Path(path).resolve()
    if target.is_file():
        return _sha256_file(target)
    digest = hashlib.sha256()
    for file in sorted(item for item in target.rglob("*") if item.is_file()):
        digest.update(file.relative_to(target).as_posix().encode("utf-8"))
        digest.update(_sha256_file(file).encode("ascii"))
    return digest.hexdigest()


def artifact_identity(path: Path | str) -> dict[str, Any]:
    """优先复用同目录冻结 manifest 的文件哈希。"""

    target = Path(path).resolve()
    manifest_path = target.parent / "manifest.json"
    if target.is_file() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        record = (manifest.get("files") or {}).get(target.name)
        if record and int(record["bytes"]) == target.stat().st_size:
            return {
                "path": str(target),
                "bytes": target.stat().st_size,
                "sha256": str(record["sha256"]),
                "identity_source": str(manifest_path),
            }
    return {
        "path": str(target),
        "bytes": target.stat().st_size if target.is_file() else None,
        "sha256": sha256_path(target),
        "identity_source": "computed",
    }


def code_identity(project_root: Path, roots: Sequence[str]) -> dict[str, str]:
    """记录 Git 提交及实际参与运行的 Python 源码内容哈希。"""

    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=project_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    digest = hashlib.sha256()
    for root in roots:
        for file in sorted((project_root / root).rglob("*.py")):
            digest.update(file.relative_to(project_root).as_posix().encode("utf-8"))
            digest.update(file.read_bytes())
    return {"git_commit": commit, "code_sha256": digest.hexdigest()}


def config_hash(config: Mapping[str, Any]) -> str:
    """计算规范化 JSON 配置指纹。"""

    encoded = json.dumps(
        dict(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def manifest_path(output: Path | str) -> Path:
    target = Path(output).resolve()
    return target.with_name(target.name + ".manifest.json")


def ensure_manifest(output: Path | str, config: Mapping[str, Any]) -> dict[str, Any]:
    """创建或核对运行 manifest，禁止不同配置续写同一 JSONL。"""

    output_path = Path(output).resolve()
    sidecar = manifest_path(output_path)
    fingerprint = config_hash(config)
    if sidecar.exists():
        existing = json.loads(sidecar.read_text(encoding="utf-8"))
        if existing.get("run_config_hash") != fingerprint:
            raise ValueError(f"运行配置与已有 manifest 不一致：{sidecar}")
        return existing
    if output_path.exists() and output_path.stat().st_size:
        raise ValueError(f"已有结果缺少运行 manifest，拒绝续写：{output_path}")
    payload = {
        "schema_version": "evidence-agent-run-manifest-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_config_hash": fingerprint,
        "config": dict(config),
    }
    sidecar.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload

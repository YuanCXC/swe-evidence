#!/usr/bin/env python3
"""Publish and restore evidence_agent_dataset_v1 through GitHub Release assets."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import subprocess
from pathlib import Path
from typing import Any, Iterable


DEFAULT_CHUNK_SIZE = 1_900 * 1024 * 1024
GITHUB_RELEASE_ASSET_LIMIT = 2 * 1024 * 1024 * 1024
BUFFER_SIZE = 8 * 1024 * 1024
SQLITE_NAME = "repository_runtime.sqlite3"
SINGLE_ASSET_NAMES = ("policy_evidence.parquet", "tasks.parquet")
DATA_FILE_NAMES = ("policy_evidence.parquet", SQLITE_NAME, "tasks.parquet")


class IntegrityError(RuntimeError):
    """Raised when a release asset does not match its manifest."""


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".partial")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise IntegrityError(f"Unsupported manifest schema: {payload.get('schema_version')}")
    return payload


def _split_sqlite(source: Path, parts_dir: Path, chunk_size: int) -> dict[str, Any]:
    if chunk_size <= 0 or chunk_size >= GITHUB_RELEASE_ASSET_LIMIT:
        raise ValueError("chunk_size must be greater than zero and less than 2 GiB")

    source_size = source.stat().st_size
    total_parts = max(1, math.ceil(source_size / chunk_size))
    number_width = max(5, len(str(total_parts)))
    parts_dir.mkdir(parents=True, exist_ok=True)
    file_digest = hashlib.sha256()
    parts: list[dict[str, Any]] = []

    with source.open("rb") as source_stream:
        for index in range(1, total_parts + 1):
            part_name = (
                f"{source.name}.part-{index:0{number_width}d}"
                f"-of-{total_parts:0{number_width}d}"
            )
            part_path = parts_dir / part_name
            temporary = part_path.with_name(part_path.name + ".partial")
            expected_size = min(chunk_size, source_size - ((index - 1) * chunk_size))
            part_digest = hashlib.sha256()
            written = 0
            try:
                with temporary.open("wb") as part_stream:
                    while written < expected_size:
                        block = source_stream.read(
                            min(BUFFER_SIZE, expected_size - written)
                        )
                        if not block:
                            raise IntegrityError(
                                f"Unexpected end of source file: {source}"
                            )
                        part_stream.write(block)
                        part_digest.update(block)
                        file_digest.update(block)
                        written += len(block)
                os.replace(temporary, part_path)
            finally:
                if temporary.exists():
                    temporary.unlink()

            parts.append(
                {
                    "index": index,
                    "name": part_name,
                    "sha256": part_digest.hexdigest(),
                    "size": written,
                }
            )

    return {
        "mode": "parts",
        "parts": parts,
        "path": source.name,
        "sha256": file_digest.hexdigest(),
        "size": source_size,
    }


def build_release_manifest(
    source_dir: Path,
    parts_dir: Path,
    manifest_path: Path,
    release_tag: str,
    repository: str,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
) -> dict[str, Any]:
    """Split the SQLite file and write a deterministic release manifest."""
    source_dir = Path(source_dir)
    parts_dir = Path(parts_dir)
    manifest_path = Path(manifest_path)
    missing = [name for name in DATA_FILE_NAMES if not (source_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"Missing dataset files: {', '.join(missing)}")

    entries: list[dict[str, Any]] = []
    for name in DATA_FILE_NAMES:
        source = source_dir / name
        if name == SQLITE_NAME:
            entries.append(_split_sqlite(source, parts_dir, chunk_size))
            continue
        size = source.stat().st_size
        if size >= GITHUB_RELEASE_ASSET_LIMIT:
            raise ValueError(f"Single Release asset exceeds 2 GiB: {source}")
        entries.append(
            {
                "asset": name,
                "mode": "single",
                "path": name,
                "sha256": _hash_file(source),
                "size": size,
            }
        )

    manifest = {
        "chunk_size": chunk_size,
        "files": entries,
        "release_tag": release_tag,
        "repository": repository,
        "schema_version": 1,
    }
    _write_json_atomic(manifest_path, manifest)
    return manifest


def _iter_checked_part_blocks(
    entry: dict[str, Any], parts_dir: Path
) -> Iterable[bytes]:
    for part in entry["parts"]:
        path = parts_dir / part["name"]
        if not path.is_file():
            raise IntegrityError(f"Missing part: {path}")
        if path.stat().st_size != part["size"]:
            raise IntegrityError(f"Part size mismatch: {path}")

        digest = hashlib.sha256()
        with path.open("rb") as stream:
            for block in iter(lambda: stream.read(BUFFER_SIZE), b""):
                digest.update(block)
                yield block
        if digest.hexdigest() != part["sha256"]:
            raise IntegrityError(f"Part SHA-256 mismatch: {path}")


def verify_parts(manifest_path: Path, parts_dir: Path) -> list[Path]:
    """Verify each SQLite part and the hash of their concatenated bytes."""
    manifest = _load_manifest(Path(manifest_path))
    parts_dir = Path(parts_dir)
    verified: list[Path] = []
    for entry in manifest["files"]:
        if entry["mode"] != "parts":
            continue
        digest = hashlib.sha256()
        total_size = 0
        for block in _iter_checked_part_blocks(entry, parts_dir):
            digest.update(block)
            total_size += len(block)
        if total_size != entry["size"]:
            raise IntegrityError(f"Combined size mismatch: {entry['path']}")
        if digest.hexdigest() != entry["sha256"]:
            raise IntegrityError(f"Combined SHA-256 mismatch: {entry['path']}")
        verified.extend(parts_dir / part["name"] for part in entry["parts"])
    return verified


def merge_parts(
    manifest_path: Path,
    parts_dir: Path,
    output_dir: Path,
    force: bool = False,
) -> list[Path]:
    """Merge part-based entries and atomically publish verified output files."""
    manifest = _load_manifest(Path(manifest_path))
    parts_dir = Path(parts_dir)
    output_dir = Path(output_dir)
    outputs: list[Path] = []

    for entry in manifest["files"]:
        if entry["mode"] != "parts":
            continue
        output = output_dir / entry["path"]
        if output.exists() and not force:
            raise FileExistsError(f"Refusing to overwrite existing file: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_name(output.name + ".partial")
        if temporary.exists():
            temporary.unlink()

        digest = hashlib.sha256()
        total_size = 0
        try:
            with temporary.open("wb") as stream:
                for block in _iter_checked_part_blocks(entry, parts_dir):
                    stream.write(block)
                    digest.update(block)
                    total_size += len(block)
            if total_size != entry["size"]:
                raise IntegrityError(f"Merged size mismatch: {entry['path']}")
            if digest.hexdigest() != entry["sha256"]:
                raise IntegrityError(f"Merged SHA-256 mismatch: {entry['path']}")
            os.replace(temporary, output)
        finally:
            if temporary.exists():
                temporary.unlink()
        outputs.append(output)
    return outputs


def verify_files(manifest_path: Path, files_dir: Path) -> list[Path]:
    """Verify all restored or downloaded data files against the manifest."""
    manifest = _load_manifest(Path(manifest_path))
    files_dir = Path(files_dir)
    verified: list[Path] = []
    for entry in manifest["files"]:
        path = files_dir / entry["path"]
        if not path.is_file():
            raise IntegrityError(f"Missing data file: {path}")
        if path.stat().st_size != entry["size"]:
            raise IntegrityError(f"Data file size mismatch: {path}")
        if _hash_file(path) != entry["sha256"]:
            raise IntegrityError(f"Data file SHA-256 mismatch: {path}")
        verified.append(path)
    return verified


def collect_upload_assets(
    manifest_path: Path, source_dir: Path, parts_dir: Path
) -> list[Path]:
    """Return ordered local paths for all assets described by the manifest."""
    manifest = _load_manifest(Path(manifest_path))
    source_dir = Path(source_dir)
    parts_dir = Path(parts_dir)
    assets: list[Path] = []
    for entry in manifest["files"]:
        if entry["mode"] == "single":
            assets.append(source_dir / entry["asset"])
        else:
            assets.extend(parts_dir / part["name"] for part in entry["parts"])
    missing = [str(path) for path in assets if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing upload assets: {', '.join(missing)}")
    return assets


def _run_gh(gh_bin: str, arguments: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [gh_bin, *arguments],
        check=True,
        text=True,
        capture_output=True,
    )


def upload_assets(
    manifest_path: Path,
    source_dir: Path,
    parts_dir: Path,
    gh_bin: str = "gh",
) -> tuple[list[str], list[str]]:
    """Upload missing assets and skip remote assets with matching names and sizes."""
    manifest = _load_manifest(Path(manifest_path))
    repository = manifest["repository"]
    release_tag = manifest["release_tag"]
    assets = collect_upload_assets(manifest_path, source_dir, parts_dir)
    view = _run_gh(
        gh_bin,
        ["release", "view", release_tag, "--repo", repository, "--json", "assets"],
    )
    remote_assets = {
        asset["name"]: int(asset["size"])
        for asset in json.loads(view.stdout).get("assets", [])
    }
    uploaded: list[str] = []
    skipped: list[str] = []
    for asset in assets:
        remote_size = remote_assets.get(asset.name)
        if remote_size is not None:
            if remote_size != asset.stat().st_size:
                raise IntegrityError(
                    f"Remote asset has unexpected size: {asset.name}"
                )
            skipped.append(asset.name)
            continue
        _run_gh(
            gh_bin,
            [
                "release",
                "upload",
                release_tag,
                str(asset),
                "--repo",
                repository,
            ],
        )
        uploaded.append(asset.name)
    return uploaded, skipped


def download_assets(
    manifest_path: Path,
    download_dir: Path,
    gh_bin: str = "gh",
) -> list[Path]:
    """Download all manifest assets, preserving already downloaded files."""
    manifest = _load_manifest(Path(manifest_path))
    repository = manifest["repository"]
    release_tag = manifest["release_tag"]
    assets: list[tuple[str, int, str]] = []
    for entry in manifest["files"]:
        if entry["mode"] == "single":
            assets.append((entry["asset"], entry["size"], entry["sha256"]))
        else:
            assets.extend(
                (part["name"], part["size"], part["sha256"])
                for part in entry["parts"]
            )

    download_dir = Path(download_dir)
    download_dir.mkdir(parents=True, exist_ok=True)
    downloaded: list[Path] = []
    for name, expected_size, expected_sha256 in assets:
        destination = download_dir / name
        if destination.is_file():
            if destination.stat().st_size != expected_size:
                raise IntegrityError(
                    f"Existing download has unexpected size: {destination}"
                )
            if _hash_file(destination) != expected_sha256:
                raise IntegrityError(
                    f"Existing download has unexpected SHA-256: {destination}"
                )
            downloaded.append(destination)
            continue
        _run_gh(
            gh_bin,
            [
                "release",
                "download",
                release_tag,
                "--repo",
                repository,
                "--pattern",
                name,
                "--dir",
                str(download_dir),
            ],
        )
        if destination.stat().st_size != expected_size:
            raise IntegrityError(f"Downloaded asset size mismatch: {destination}")
        if _hash_file(destination) != expected_sha256:
            raise IntegrityError(f"Downloaded asset SHA-256 mismatch: {destination}")
        downloaded.append(destination)
    return downloaded


def _path(value: str) -> Path:
    return Path(value).resolve()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    split = subparsers.add_parser("split", help="Split SQLite and create manifest")
    split.add_argument("--source-dir", required=True, type=_path)
    split.add_argument("--parts-dir", required=True, type=_path)
    split.add_argument("--manifest", required=True, type=_path)
    split.add_argument("--release-tag", required=True)
    split.add_argument("--repository", required=True)
    split.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK_SIZE)

    verify_parts_parser = subparsers.add_parser(
        "verify-parts", help="Verify SQLite parts"
    )
    verify_parts_parser.add_argument("--manifest", required=True, type=_path)
    verify_parts_parser.add_argument("--parts-dir", required=True, type=_path)

    merge = subparsers.add_parser("merge", help="Merge SQLite parts")
    merge.add_argument("--manifest", required=True, type=_path)
    merge.add_argument("--parts-dir", required=True, type=_path)
    merge.add_argument("--output-dir", required=True, type=_path)
    merge.add_argument("--force", action="store_true")

    verify_files_parser = subparsers.add_parser(
        "verify-files", help="Verify restored dataset files"
    )
    verify_files_parser.add_argument("--manifest", required=True, type=_path)
    verify_files_parser.add_argument("--files-dir", required=True, type=_path)

    upload = subparsers.add_parser("upload", help="Upload GitHub Release assets")
    upload.add_argument("--manifest", required=True, type=_path)
    upload.add_argument("--source-dir", required=True, type=_path)
    upload.add_argument("--parts-dir", required=True, type=_path)
    upload.add_argument("--gh-bin", default="gh")

    download = subparsers.add_parser("download", help="Download GitHub Release assets")
    download.add_argument("--manifest", required=True, type=_path)
    download.add_argument("--download-dir", required=True, type=_path)
    download.add_argument("--gh-bin", default="gh")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if args.command == "split":
        manifest = build_release_manifest(
            args.source_dir,
            args.parts_dir,
            args.manifest,
            args.release_tag,
            args.repository,
            args.chunk_size,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    elif args.command == "verify-parts":
        for path in verify_parts(args.manifest, args.parts_dir):
            print(path)
    elif args.command == "merge":
        for path in merge_parts(
            args.manifest, args.parts_dir, args.output_dir, args.force
        ):
            print(path)
    elif args.command == "verify-files":
        for path in verify_files(args.manifest, args.files_dir):
            print(path)
    elif args.command == "upload":
        uploaded, skipped = upload_assets(
            args.manifest, args.source_dir, args.parts_dir, args.gh_bin
        )
        print(json.dumps({"uploaded": uploaded, "skipped": skipped}, indent=2))
    elif args.command == "download":
        for path in download_assets(args.manifest, args.download_dir, args.gh_bin):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

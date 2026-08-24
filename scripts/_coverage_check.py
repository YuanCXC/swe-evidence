#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
覆盖率检查：比对 strong_teacher_v1_3_all/<split>/md/*.md 与
result/<split>/*.md，判断哪些源文件已有合法 JSON array 结果。
用于断点续跑：只把缺失文件所在的分片交回子智能体。
"""
import os, json, sys

ROOT = "E:/Code_Personal/Subject/evidence-agent/data/.external_supervision"
SRC_BASE = os.path.join(ROOT, "strong_teacher_v1_3_all")
OUT_BASE = os.path.join(ROOT, "result")

def check_split(split):
    manifest_path = os.path.join(OUT_BASE, split, "_manifest.json")
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    src_dir = manifest["src_dir"]
    out_dir = manifest["out_dir"]
    chunks = manifest["chunks"]

    done = 0
    missing_chunks = []   # (chunk_index, [missing filenames])
    bad = []              # (filename, reason)
    for idx, chunk in enumerate(chunks):
        missing = []
        for fn in chunk:
            p = os.path.join(out_dir, fn)
            if not os.path.exists(p):
                missing.append(fn)
                continue
            try:
                obj = json.loads(open(p, encoding="utf-8").read())
                if not (isinstance(obj, list) and len(obj) >= 1 and "task_id" in obj[0]):
                    raise ValueError("not a [obj] array")
                done += 1
            except Exception as e:
                bad.append((fn, str(e)[:60]))
                missing.append(fn)
        if missing:
            missing_chunks.append((idx, missing))
    total = sum(len(c) for c in chunks)
    print(f"=== {split} ===")
    print(f"  total={total}  done={done}  missing_chunks={len(missing_chunks)}  bad={len(bad)}")
    if bad:
        print("  bad files:", bad[:5])
    return missing_chunks

if __name__ == "__main__":
    splits = sys.argv[1:] or ["benchmark", "train", "validation"]
    for sp in splits:
        check_split(sp)

# -*- coding: utf-8 -*-
"""
将 Stage 1 决策结果追加到原始监督样本末尾，输出到 result/validation/。

用法：
  python append_stage1_result.py --task_id task_xxx --decisions-file /path/to/dec.json

dec.json 结构：
  {
    "fault_location":       {"decision": "...", "reason": "..."},
    "fault_logic":          {"decision": "...", "reason": "..."},
    "dependency_context":   {"decision": "...", "reason": "..."},
    "state_flow":           {"decision": "...", "reason": "..."},
    "behavior_constraint":  {"decision": "...", "reason": "..."},
    "repair_scope":         {"decision": "...", "reason": "..."},
    "validation_constraint":{"decision": "...", "reason": "..."}
  }

输出文件 = 原文全文 + 末尾 "# Result" 段（含 ```json 代码块）。
"""
import argparse
import json
import os

BASE = r"E:\Code_Personal\Subject\evidence-agent\data\.external_supervision"
SRC_DIR = os.path.join(BASE, "stage1_all_v2_10", "validation")
DST_DIR = os.path.join(BASE, "result", "validation")

SLOTS = [
    "fault_location",
    "fault_logic",
    "dependency_context",
    "state_flow",
    "behavior_constraint",
    "repair_scope",
    "validation_constraint",
]
DECISIONS = {"repository_required", "question_satisfied", "not_required", "uncertain"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task_id", required=True)
    ap.add_argument("--decisions-file", required=True)
    args = ap.parse_args()

    with open(args.decisions_file, encoding="utf-8") as f:
        dec = json.load(f)

    # 校验槽位完整性
    missing = set(SLOTS) - set(dec.keys())
    extra = set(dec.keys()) - set(SLOTS)
    assert not missing, f"缺少槽位: {missing}"
    assert not extra, f"多余槽位: {extra}"
    for s in SLOTS:
        v = dec[s]
        assert isinstance(v, dict), f"{s} 不是对象"
        d = v.get("decision")
        assert d in DECISIONS, f"{s} 的 decision 非法: {d}"
        r = v.get("reason", "")
        assert isinstance(r, str) and r.strip(), f"{s} 缺少 reason"

    src = os.path.join(SRC_DIR, args.task_id + ".md")
    assert os.path.exists(src), f"源文件不存在: {src}"
    with open(src, encoding="utf-8") as f:
        content = f.read()

    result = {"task_id": args.task_id, "slots": dec}
    block = "\n\n# Result\n\n```json\n" + json.dumps(result, ensure_ascii=False, indent=2) + "\n```\n"
    out = content.rstrip("\n") + block

    os.makedirs(DST_DIR, exist_ok=True)
    dst = os.path.join(DST_DIR, args.task_id + ".md")
    with open(dst, "w", encoding="utf-8") as f:
        f.write(out)
    print("WROTE", dst)


if __name__ == "__main__":
    main()

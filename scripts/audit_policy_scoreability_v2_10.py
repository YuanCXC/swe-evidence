#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""V2.10 policy scoreability / candidate-overflow 只读发布前审计。"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def open_ro(path: Path) -> sqlite3.Connection:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    conn = sqlite3.connect(path.as_uri() + "?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only=ON")
    return conn


def ratio(n: int, d: int) -> float | None:
    return n / d if d else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("data/.build/unified_swe_v1.sqlite3"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/.build/audit_policy_scoreability_v2_10/report.json"),
    )
    args = parser.parse_args()

    conn = open_ro(args.db)
    try:
        # state -> split/state_type/pool metadata
        states: dict[str, dict[str, Any]] = {}
        overflow_reasons = Counter()
        overflow_by_state_type = Counter()
        overflow_by_split = Counter()
        overflow_contract_violations: list[str] = []

        for row in conn.execute(
            """
            SELECT
                p.state_id,
                p.payload_json,
                c.final_split
            FROM policy_states p
            JOIN canonical_tasks c ON c.task_id=p.task_id
            """
        ):
            payload = json.loads(row["payload_json"])
            pool = payload.get("candidate_pool_stats") or {}
            state_id = str(row["state_id"])
            state_type = str(payload.get("state_type"))
            split = str(row["final_split"])
            overflow = bool(pool.get("candidate_overflow"))
            reasons = list(map(str, pool.get("overflow_reasons") or []))
            states[state_id] = {
                "state_type": state_type,
                "split": split,
                "ranking_loss_mask": bool(payload.get("ranking_loss_mask")),
                "candidate_overflow": overflow,
                "overflow_reasons": reasons,
            }

            if overflow:
                overflow_by_state_type[state_type] += 1
                overflow_by_split[split] += 1
                overflow_reasons.update(reasons)

                online_cap = int(pool.get("online_single_cap") or 0)
                online_count = int(pool.get("online_single_count") or 0)
                injected_count = int(pool.get("injected_required_single_count") or 0)
                pair_cap = int(pool.get("regular_pair_cap") or 0)
                pair_count = int(pool.get("pair_count") or 0)

                expected = (
                    ("required_single" in reasons and online_count + injected_count > online_cap)
                    or ("required_pair" in reasons and pair_count > pair_cap)
                )
                if not expected:
                    overflow_contract_violations.append(state_id)

        action_count = 0
        unscoreable_count = 0
        positive_label_count = 0
        positive_active_count = 0
        positive_masked_unscoreable_count = 0

        unscoreable_by_state_type = Counter()
        unscoreable_by_action_type = Counter()
        unscoreable_by_scope = Counter()
        positive_masked_by_state_type = Counter()
        positive_masked_by_scope = Counter()

        state_positive_label = defaultdict(int)
        state_positive_active = defaultdict(int)
        state_positive_masked_unscoreable = defaultdict(int)

        for row in conn.execute(
            "SELECT state_id,payload_json FROM candidate_actions"
        ):
            action_count += 1
            state_id = str(row["state_id"])
            action = json.loads(row["payload_json"])
            state_type = states[state_id]["state_type"]

            scoreable = bool(action.get("scoreable"))
            label = str(action.get("action_label"))
            loss_mask = bool(action.get("action_loss_mask"))
            action_type = str(action.get("action_type"))
            scope = str(action.get("candidate_scope"))

            if not scoreable:
                unscoreable_count += 1
                unscoreable_by_state_type[state_type] += 1
                unscoreable_by_action_type[action_type] += 1
                unscoreable_by_scope[scope] += 1

            if label == "positive":
                positive_label_count += 1
                state_positive_label[state_id] += 1
                if loss_mask:
                    positive_active_count += 1
                    state_positive_active[state_id] += 1
                if (not loss_mask) and (not scoreable):
                    positive_masked_unscoreable_count += 1
                    state_positive_masked_unscoreable[state_id] += 1
                    positive_masked_by_state_type[state_type] += 1
                    positive_masked_by_scope[scope] += 1

        positive_states = {
            sid for sid, count in state_positive_label.items() if count > 0
        }
        active_positive_states = {
            sid for sid, count in state_positive_active.items() if count > 0
        }
        masked_positive_states = {
            sid for sid, count in state_positive_masked_unscoreable.items() if count > 0
        }
        all_positive_masked_states = sorted(
            positive_states - active_positive_states
        )

        all_positive_masked_by_state_type = Counter(
            states[sid]["state_type"] for sid in all_positive_masked_states
        )
        all_positive_masked_by_split = Counter(
            states[sid]["split"] for sid in all_positive_masked_states
        )

        overflow_states = {
            sid for sid, meta in states.items() if meta["candidate_overflow"]
        }

        report = {
            "mode": "read_only_policy_scoreability_audit",
            "database": str(args.db.resolve()),
            "counts": {
                "state_count": len(states),
                "action_count": action_count,
                "candidate_overflow_state_count": len(overflow_states),
                "unscoreable_action_count": unscoreable_count,
                "positive_label_action_count": positive_label_count,
                "positive_loss_active_action_count": positive_active_count,
                "positive_masked_unscoreable_action_count": positive_masked_unscoreable_count,
                "positive_label_state_count": len(positive_states),
                "positive_loss_active_state_count": len(active_positive_states),
                "positive_masked_unscoreable_state_count": len(masked_positive_states),
                "all_positive_masked_state_count": len(all_positive_masked_states),
                "overflow_and_positive_masked_state_count": len(
                    overflow_states & masked_positive_states
                ),
                "overflow_and_all_positive_masked_state_count": len(
                    overflow_states & set(all_positive_masked_states)
                ),
                "overflow_contract_violation_count": len(
                    overflow_contract_violations
                ),
            },
            "rates": {
                "unscoreable_action_rate": ratio(unscoreable_count, action_count),
                "positive_masked_unscoreable_action_rate": ratio(
                    positive_masked_unscoreable_count, positive_label_count
                ),
                "all_positive_masked_state_rate_among_positive_states": ratio(
                    len(all_positive_masked_states), len(positive_states)
                ),
            },
            "candidate_overflow": {
                "reason_counts": dict(sorted(overflow_reasons.items())),
                "by_state_type": dict(sorted(overflow_by_state_type.items())),
                "by_split": dict(sorted(overflow_by_split.items())),
                "contract_violation_state_ids": overflow_contract_violations[:100],
            },
            "unscoreable_actions": {
                "by_state_type": dict(sorted(unscoreable_by_state_type.items())),
                "by_action_type": dict(sorted(unscoreable_by_action_type.items())),
                "by_candidate_scope": dict(sorted(unscoreable_by_scope.items())),
            },
            "positive_masked_unscoreable": {
                "by_state_type": dict(sorted(positive_masked_by_state_type.items())),
                "by_candidate_scope": dict(sorted(positive_masked_by_scope.items())),
                "all_positive_masked_by_state_type": dict(
                    sorted(all_positive_masked_by_state_type.items())
                ),
                "all_positive_masked_by_split": dict(
                    sorted(all_positive_masked_by_split.items())
                ),
                "all_positive_masked_state_ids": all_positive_masked_states[:100],
            },
        }

        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))

        # release gate:
        # 1) overflow 语义必须自洽；
        # 2) 不允许因 scoreability 导致一个原本有 positive label 的 state
        #    完全失去所有 loss-active positive。
        if overflow_contract_violations or all_positive_masked_states:
            return 2
        return 0
    finally:
        conn.close()


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build a derived multi-stage Decision-Boundary bundle for stage-2 fine-tuning.

Purpose
-------
The frozen Evidence-Agent bundle contains at most one near-complete
``decision_boundary`` state per task.  This script derives *multiple* incomplete
states from the existing audited Teacher obligations / witness groups so that the
policy is trained on trajectories such as::

    {} -> {e1} -> {e1,e2} -> {e1,e2,e3} -> ... -> sufficient -> STOP

The source bundle is never modified.  Train/validation policy states are replaced
with mechanically derived multi-stage boundary states; benchmark rows are kept
bit-for-bit at the row-object level (the output parquet is rewritten, but benchmark
supervision content is not changed).

Important contracts
-------------------
* No new Teacher labels are invented.
* OR-of-AND witness semantics are preserved exactly.
* A derived state is emitted only when K is non-empty and insufficient.
* STOP is negative on every derived state.
* Positive actions are only actions that strictly increase deterministic witness
  progress / completion under the current K.
* Non-witness evidence is used for negatives; witness evidence with zero immediate
  gain is never mislabeled negative.
* model_input_token_count is recomputed using the same renderer/tokenizer contract
  as the stage-1 trainer.
* Positive unscoreable actions are dropped; if the canonical next acquisition is
  unscoreable the state is skipped rather than silently corrupting supervision.
* policy_evidence.parquet is hard-linked (or copied) unchanged.

Typical use (after stage-1 three-epoch training)::

    python3 scripts/build_multistage_boundary_bundle.py \
      --source-bundle data/evidence_agent_dataset_v1 \
      --output-bundle data/evidence_agent_multistage_boundary_v1 \
      --trainer-script scripts/train_evidence_policy.py \
      --model-dir models/evidence_policy_v1_0/best \
      --evidence-cache models/evidence_policy_v1_0/evidence_lookup.sqlite3 \
      --local-files-only

Then fine-tune from stage-1 ``best``; do NOT use ``--resume-from`` because this is
an intentional new training stage, not crash recovery.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import importlib.util
import json
import math
import os
import shutil
import sqlite3
import sys
import tempfile
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

SCRIPT_VERSION = "1.1.2"
EPS = 1e-12


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def stable_json(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def sha256_file(path: Path, chunk_size: int = 8 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def sha1_short(text: str, n: int = 12) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:n]


def atomic_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".tmp.", dir=str(path.parent))
    os.close(fd)
    tmp = Path(name)
    try:
        tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


def load_module(path: Path, name: str):
    """Load a Python file under a stable module name.

    ``dataclasses`` (Python 3.10 included) resolves annotations through
    ``sys.modules[cls.__module__]`` while a decorated class is being defined.
    Therefore the module must be registered *before* ``exec_module``.
    ``module_from_spec`` alone is not sufficient.
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import module: {path}")
    mod = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(name)
    sys.modules[name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:
        if previous is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = previous
        raise
    return mod


# ---------------------------------------------------------------------------
# Witness semantics
# ---------------------------------------------------------------------------


def applicable_obligations(supervision: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Normalize all applicable obligations while retaining mandatory semantics.

    This mirrors frozen V2.10 ``evidence_state_metrics``: progress is averaged
    over every applicable obligation; completion is computed only over mandatory
    obligations.  A mandatory obligation without a witness group makes a
    repository-only completion trajectory undefined and is rejected.
    """
    result: list[dict[str, Any]] = []
    for raw in supervision.get("obligations") or []:
        if not isinstance(raw, Mapping) or raw.get("applicable") is not True:
            continue
        groups: list[set[str]] = []
        for g in raw.get("witness_groups") or []:
            if not isinstance(g, Mapping):
                continue
            ids = {str(x) for x in (g.get("evidence_ids") or []) if str(x)}
            if ids:
                groups.append(ids)
        if raw.get("mandatory") is True and not groups:
            return []
        result.append({
            "obligation_id": str(raw.get("obligation_id") or f"ob{len(result)}"),
            "mandatory": bool(raw.get("mandatory")),
            "groups": groups,
        })
    return result


def witness_ids(obligations: Sequence[Mapping[str, Any]]) -> set[str]:
    return {eid for ob in obligations for group in (ob.get("groups") or []) for eid in group}


def state_metrics(k: Iterable[str], obligations: Sequence[Mapping[str, Any]]) -> tuple[float | None, float | None]:
    """Exact V2.10-style (completion_score, progress_score)."""
    selected = set(map(str, k))
    applicable = list(obligations)
    mandatory = [ob for ob in applicable if ob.get("mandatory") is True]

    completed_ids: set[str] = set()
    progress_values: list[float] = []
    for ob in applicable:
        groups = [set(map(str, g)) for g in (ob.get("groups") or []) if g]
        if any(g <= selected for g in groups):
            completed_ids.add(str(ob.get("obligation_id") or ""))
        progress_values.append(max((len(selected & g) / len(g) for g in groups), default=0.0))

    if mandatory:
        mandatory_ids = {str(ob.get("obligation_id") or "") for ob in mandatory}
        completion = len(mandatory_ids & completed_ids) / len(mandatory_ids)
    else:
        completion = None
    progress = sum(progress_values) / len(progress_values) if progress_values else None
    return completion, progress


def is_sufficient(k: Iterable[str], obligations: Sequence[Mapping[str, Any]]) -> bool:
    completion, _ = state_metrics(k, obligations)
    return completion is not None and completion >= 1.0 - EPS


def minimize_complete_certificate(
    complete_ids: Sequence[str],
    obligations: Sequence[Mapping[str, Any]],
    token_cost: Mapping[str, int],
) -> list[str]:
    """Greedily remove redundant evidence while preserving sufficiency.

    The input comes from the frozen complete state, so this is a conservative
    derivation: no evidence outside that audited complete state is inserted into
    the canonical path.
    """
    current = list(dict.fromkeys(map(str, complete_ids)))
    if not is_sufficient(current, obligations):
        return []
    # Expensive evidence is attempted first; deterministic tie break by id.
    removal_order = sorted(current, key=lambda x: (-int(token_cost.get(x, 0)), x))
    keep = set(current)
    for eid in removal_order:
        trial = keep - {eid}
        if trial and is_sufficient(trial, obligations):
            keep = trial
    return [eid for eid in current if eid in keep]


def acquisition_order(
    certificate: Sequence[str],
    obligations: Sequence[Mapping[str, Any]],
    token_cost: Mapping[str, int],
) -> list[str]:
    """Deterministic single-evidence path from empty K to the certificate.

    At each step select the item with the largest completion gain, then progress
    gain, then lower token cost.  Single-step ordering intentionally yields
    intermediate K states even when a pair action could complete an AND witness.
    """
    remaining = set(map(str, certificate))
    k: list[str] = []
    out: list[str] = []
    while remaining:
        base_c, base_p = state_metrics(k, obligations)
        assert base_c is not None and base_p is not None
        scored: list[tuple[float, float, int, str]] = []
        for eid in remaining:
            c, p = state_metrics([*k, eid], obligations)
            assert c is not None and p is not None
            scored.append((c - base_c, p - base_p, -int(token_cost.get(eid, 0)), eid))
        scored.sort(reverse=True)
        chosen = scored[0][3]
        k.append(chosen)
        out.append(chosen)
        remaining.remove(chosen)
    return out


def select_prefix_lengths(total_items: int, max_boundaries: int) -> list[int]:
    """Select non-empty incomplete prefix lengths, covering early/mid/late."""
    possible = list(range(1, total_items))
    if max_boundaries <= 0 or len(possible) <= max_boundaries:
        return possible
    # Evenly cover the trajectory and always include the first and last boundary.
    raw = [1 + round(i * (total_items - 2) / (max_boundaries - 1)) for i in range(max_boundaries)]
    result = sorted(set(min(total_items - 1, max(1, x)) for x in raw))
    # Rounding can collapse indices. Fill deterministically if needed.
    for x in possible:
        if len(result) >= max_boundaries:
            break
        if x not in result:
            result.append(x)
    return sorted(result)[:max_boundaries]


def stage_name(progress: float) -> str:
    if progress < 0.35:
        return "early"
    if progress < 0.70:
        return "mid"
    if progress < 0.90:
        return "late"
    return "near_complete"


# ---------------------------------------------------------------------------
# Stage-1 evidence cache reader
# ---------------------------------------------------------------------------


class EvidenceLookup:
    def __init__(self, path: Path):
        if not path.is_file():
            raise FileNotFoundError(f"stage-1 evidence cache not found: {path}")
        self.conn = sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA query_only=ON")
        tables = {str(r[0]) for r in self.conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        if "evidence" not in tables:
            raise ValueError(f"{path} does not contain evidence table")

    def close(self) -> None:
        self.conn.close()

    def get_many(self, ids: Sequence[str]) -> dict[str, dict[str, Any]]:
        unique = list(dict.fromkeys(map(str, ids)))
        out: dict[str, dict[str, Any]] = {}
        for off in range(0, len(unique), 700):
            chunk = unique[off:off + 700]
            if not chunk:
                continue
            ph = ",".join("?" for _ in chunk)
            for r in self.conn.execute(f"SELECT * FROM evidence WHERE evidence_id IN ({ph})", chunk):
                out[str(r["evidence_id"])] = {
                    "evidence_id": str(r["evidence_id"]),
                    "path": str(r["path"]),
                    "unit_type": str(r["unit_type"]),
                    "symbol": r["symbol"],
                    "start_line": int(r["start_line"]),
                    "end_line": int(r["end_line"]),
                    "content": str(r["content"]),
                    "rendered_body": str(r["rendered_body"]),
                    "metadata": str(r["metadata"]),
                    "rendered_token_count": int(r["rendered_token_count"]),
                }
        missing = sorted(set(unique) - set(out))
        if missing:
            raise KeyError(f"evidence cache missing {len(missing)} ids, first={missing[:5]}")
        return out

    def token_costs(self, ids: Sequence[str]) -> dict[str, int]:
        return {eid: int(rec["rendered_token_count"]) for eid, rec in self.get_many(ids).items()}


# ---------------------------------------------------------------------------
# Candidate templates and derived actions
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ActionTemplate:
    action_type: str
    evidence_ids: tuple[str, ...]
    candidate_scope: str
    online_rank: int | None
    online_score: float | None


def template_priority(t: ActionTemplate) -> tuple[int, int, str]:
    return (
        0 if t.candidate_scope == "online" else 1,
        t.online_rank if t.online_rank is not None else 2**31 - 1,
        "|".join(t.evidence_ids),
    )


def collect_action_templates(states: Sequence[Mapping[str, Any]]) -> tuple[dict[str, ActionTemplate], dict[tuple[str, str], ActionTemplate]]:
    singles: dict[str, ActionTemplate] = {}
    pairs: dict[tuple[str, str], ActionTemplate] = {}
    for state in states:
        for a in state.get("candidate_actions") or []:
            if not isinstance(a, Mapping):
                continue
            typ = str(a.get("action_type") or "")
            ids = tuple(map(str, a.get("evidence_ids") or []))
            if typ not in {"single", "pair"}:
                continue
            if typ == "single" and len(ids) != 1:
                continue
            if typ == "pair" and len(ids) != 2:
                continue
            scope = str(a.get("candidate_scope") or "offline_injected")
            rank_raw = a.get("online_retrieval_rank")
            score_raw = a.get("online_retrieval_score")
            t = ActionTemplate(
                typ,
                ids,
                scope,
                None if rank_raw is None else int(rank_raw),
                None if score_raw is None else float(score_raw),
            )
            if typ == "single":
                old = singles.get(ids[0])
                if old is None or template_priority(t) < template_priority(old):
                    singles[ids[0]] = t
            else:
                key = tuple(sorted(ids))
                old = pairs.get(key)
                if old is None or template_priority(t) < template_priority(old):
                    pairs[key] = t
    return singles, pairs


def make_action_base(
    *,
    task_id: str,
    state_id: str,
    action_type: str,
    ids: Sequence[str],
    template: ActionTemplate | None,
    label: str,
    completion_gain: float,
    progress_gain: float,
) -> dict[str, Any]:
    ids = list(map(str, ids))
    key = f"{task_id}|{state_id}|{action_type}|{'|'.join(ids)}"
    scope = "stop" if action_type == "stop" else (
        template.candidate_scope if template is not None else "offline_injected"
    )
    action: dict[str, Any] = {
        "action_id": f"msb:{action_type}:{sha1_short(key)}",
        "action_type": action_type,
        "evidence_ids": ids,
        "candidate_scope": scope,
        "completion_gain": float(completion_gain),
        "progress_gain": float(progress_gain),
        "action_label": label,
        "action_loss_mask": True,
        "scoreable": True,
        "rendered_state_body_evidence_ids": [],
        "model_input_token_count": None,
    }
    if template is not None:
        if template.online_rank is not None:
            action["online_retrieval_rank"] = int(template.online_rank)
        if template.online_score is not None:
            action["online_retrieval_score"] = float(template.online_score)
    return action


def candidate_gain(k: Sequence[str], add_ids: Sequence[str], obligations: Sequence[Mapping[str, Any]]) -> tuple[float, float]:
    base_c, base_p = state_metrics(k, obligations)
    new_c, new_p = state_metrics([*k, *add_ids], obligations)
    assert base_c is not None and base_p is not None and new_c is not None and new_p is not None
    return new_c - base_c, new_p - base_p


def action_rank_key(a: Mapping[str, Any]) -> tuple[Any, ...]:
    return (
        -float(a.get("completion_gain") or 0.0),
        -float(a.get("progress_gain") or 0.0),
        0 if a.get("candidate_scope") == "online" else 1,
        int(a.get("online_retrieval_rank")) if a.get("online_retrieval_rank") is not None else 2**31 - 1,
        0 if a.get("action_type") == "single" else 1,
        tuple(map(str, a.get("evidence_ids") or [])),
    )


def choose_state_body_ids(k: Sequence[str], evidence: Mapping[str, Mapping[str, Any]], budget: int) -> list[str]:
    if budget <= 0:
        return []
    chosen: set[str] = set()
    used = 0
    # Prefer most recently acquired evidence body; metadata for every K item is
    # always visible regardless of this body subset.
    for eid in reversed(list(k)):
        cost = max(0, int(evidence[eid].get("rendered_token_count") or 0))
        if used + cost <= budget:
            chosen.add(eid)
            used += cost
    return [eid for eid in k if eid in chosen]


def render_and_filter_actions(
    *,
    trainer: Any,
    tokenizer: Any,
    question_view: str,
    state: dict[str, Any],
    actions: Sequence[dict[str, Any]],
    evidence: Mapping[str, Mapping[str, Any]],
    body_ids: Sequence[str],
    model_max_length: int,
    canonical_next: str,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    canonical_ok = False
    for raw in actions:
        a = copy.deepcopy(raw)
        a["rendered_state_body_evidence_ids"] = list(body_ids)
        text = trainer.render_action_text(
            question_view=question_view,
            state=state,
            action=a,
            evidence=evidence,
        )
        n = int(trainer.token_length_without_warning(tokenizer, text, add_special_tokens=True))
        a["model_input_token_count"] = n
        if n > model_max_length:
            # Never retain an active unscoreable action in the stage-2 bundle.
            continue
        a["scoreable"] = True
        a["action_loss_mask"] = True
        if (
            a.get("action_type") == "single"
            and list(map(str, a.get("evidence_ids") or [])) == [canonical_next]
            and a.get("action_label") == "positive"
        ):
            canonical_ok = True
        out.append(a)
    if not canonical_ok:
        return []
    return out


def build_boundary_state(
    *,
    task_id: str,
    row_input: Mapping[str, Any],
    original_states: Sequence[Mapping[str, Any]],
    obligations: Sequence[Mapping[str, Any]],
    witness_set: set[str],
    k: Sequence[str],
    canonical_next: str,
    prefix_index: int,
    total_prefixes: int,
    singles: Mapping[str, ActionTemplate],
    pairs: Mapping[tuple[str, str], ActionTemplate],
    evidence_lookup: EvidenceLookup,
    trainer: Any,
    policy_builder: Any,
    tokenizer: Any,
    raw_obligations: Sequence[Mapping[str, Any]],
    hard_negative_ids: set[str],
    max_actions: int,
    max_positive_pairs: int,
    state_body_token_budget: int,
    model_max_length: int,
) -> dict[str, Any] | None:
    completion, progress = state_metrics(k, obligations)
    if completion is None or progress is None or not k or completion >= 1.0 - EPS:
        return None
    stage = stage_name(progress)
    state_id = f"{task_id}:msb:{prefix_index:02d}of{total_prefixes:02d}:{stage}"

    # Candidate universe: existing online/offline singles + every witness not yet
    # selected. Pair candidates include existing pairs plus a bounded set of
    # witness pairs. Labels/gains/Pareto are delegated to the *frozen V2.10*
    # label_candidate_actions implementation so stage-2 semantics stay identical
    # to stage-1.
    remaining_witness = sorted(witness_set - set(k))
    candidate_single_ids = list(dict.fromkeys([
        *remaining_witness,
        *[eid for eid, _t in sorted(singles.items(), key=lambda kv: template_priority(kv[1])) if eid not in set(k)],
    ]))

    pair_candidates: list[tuple[str, str]] = []
    for key, _t in sorted(pairs.items(), key=lambda kv: template_priority(kv[1])):
        if not (set(key) & set(k)):
            pair_candidates.append(tuple(sorted(key)))
    witness_pair_scored: list[tuple[float, float, tuple[str, str]]] = []
    for i, left in enumerate(remaining_witness):
        for right in remaining_witness[i + 1:]:
            cg, pg = candidate_gain(k, [left, right], obligations)
            if cg > EPS or pg > EPS:
                witness_pair_scored.append((cg, pg, tuple(sorted((left, right)))))
    witness_pair_scored.sort(key=lambda x: (-x[0], -x[1], x[2]))
    # Feed more than the final retained pair count into Pareto labeling, then prune.
    pair_candidates.extend(pair for _cg, _pg, pair in witness_pair_scored[: max(8, max_positive_pairs * 4)])
    pair_candidates = list(dict.fromkeys(pair_candidates))

    all_candidate_ids = set(candidate_single_ids)
    for pair in pair_candidates:
        all_candidate_ids.update(pair)
    token_cost_map = evidence_lookup.token_costs(sorted(set(k) | all_candidate_ids))

    labeled = policy_builder.label_candidate_actions(
        task_id=task_id,
        state_id=state_id,
        state_evidence_ids=list(k),
        candidate_evidence_ids=candidate_single_ids,
        pair_evidence_ids=pair_candidates,
        obligations=list(raw_obligations),
        token_costs=token_cost_map,
        known_negative_evidence_ids=set(hard_negative_ids),
    )

    # Restore truthful candidate scope/rank metadata from the source actions.
    for a in labeled:
        typ = str(a.get("action_type") or "")
        ids = tuple(map(str, a.get("evidence_ids") or []))
        if typ == "stop":
            continue
        template = singles.get(ids[0]) if typ == "single" and len(ids) == 1 else pairs.get(tuple(sorted(ids)))
        if template is not None:
            a["candidate_scope"] = template.candidate_scope
            a["online_retrieval_rank"] = template.online_rank
            a["online_retrieval_score"] = template.online_score
        elif set(ids) <= witness_set:
            a["candidate_scope"] = "offline_injected"
            a["online_retrieval_rank"] = None
            a["online_retrieval_score"] = None

    positives = [a for a in labeled if a.get("action_loss_mask") is True and a.get("action_label") == "positive"]
    stop_actions = [a for a in labeled if str(a.get("action_type") or "") == "stop"]
    negatives = [a for a in labeled if a.get("action_loss_mask") is True and a.get("action_label") == "negative" and str(a.get("action_type") or "") != "stop"]

    positives.sort(key=action_rank_key)
    negatives.sort(key=lambda a: (
        0 if a.get("candidate_scope") == "online" else 1,
        int(a.get("online_retrieval_rank")) if a.get("online_retrieval_rank") is not None else 2**31 - 1,
        0 if a.get("action_type") == "single" else 1,
        tuple(map(str, a.get("evidence_ids") or [])),
    ))
    retained: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    def keep(a: dict[str, Any]) -> None:
        aid = str(a.get("action_id") or "")
        if aid and aid not in seen_ids:
            retained.append(a)
            seen_ids.add(aid)
    for a in positives:
        keep(a)
    for a in stop_actions:
        keep(a)
    target = max(max_actions, len(retained)) if max_actions > 0 else math.inf
    for a in negatives:
        if len(retained) >= target:
            break
        keep(a)
    if not any(a.get("action_label") == "positive" and a.get("action_loss_mask") is True for a in retained):
        return None

    required_ids = set(k)
    for a in retained:
        required_ids.update(map(str, a.get("evidence_ids") or []))
    evidence = evidence_lookup.get_many(sorted(required_ids))

    base_conf = 1.0
    for s in original_states:
        if str(s.get("state_type") or "") == "decision_boundary" and s.get("confidence") is not None:
            base_conf = float(s.get("confidence"))
            break

    exact_metrics = policy_builder.evidence_state_metrics(set(k), list(raw_obligations))
    state: dict[str, Any] = {
        "state_id": state_id,
        "state_type": "decision_boundary",
        "evidence_ids": list(dict.fromkeys(map(str, k))),
        "label_source": "multistage_gold_prefix",
        "completed_obligation_ids": list(exact_metrics.get("completed_obligation_ids") or []),
        "completion_score": exact_metrics.get("completion_score"),
        "progress_score": exact_metrics.get("progress_score"),
        "step": int(prefix_index),
        "removed_evidence_ids": [],
        "added_evidence_ids": [],
        "confidence": base_conf,
        "stop_label": "negative",
        "stop_loss_mask": True,
        "ranking_loss_mask": True,
        "candidate_actions": [],
    }

    question = trainer.build_question(row_input)
    rendered_info = policy_builder.render_policy_model_inputs(
        question=question,
        state_evidence_ids=list(k),
        candidate_evidence_ids=[list(map(str, a.get("evidence_ids") or [])) for a in retained],
        evidence_by_id=evidence,
        tokenizer=tokenizer,
        model_max_length=model_max_length,
    )
    rendered: list[dict[str, Any]] = []
    for a, info in zip(retained, rendered_info):
        a = copy.deepcopy(a)
        a["model_input_token_count"] = int(info["model_input_token_count"])
        a["rendered_state_body_evidence_ids"] = list(info["rendered_state_body_evidence_ids"])
        a["scoreable"] = bool(a.get("scoreable", True) and info["scoreable"])
        if not a["scoreable"]:
            a["action_loss_mask"] = False
        rendered.append(a)

    rendered.sort(key=lambda a: (
        {"single": 0, "pair": 1, "stop": 2}.get(str(a.get("action_type") or ""), 9),
        int(a.get("online_retrieval_rank")) if a.get("online_retrieval_rank") is not None else 2**31 - 1,
        tuple(map(str, a.get("evidence_ids") or [])),
    ))
    pos = sum(a.get("action_label") == "positive" and a.get("action_loss_mask") is True and a.get("scoreable") is True for a in rendered)
    neg = sum(a.get("action_label") == "negative" and a.get("action_loss_mask") is True and a.get("scoreable") is True for a in rendered)
    if not pos or not neg:
        return None
    stop = next((a for a in rendered if str(a.get("action_type") or "") == "stop"), None)
    if stop is None:
        return None
    state["candidate_actions"] = rendered
    state["stop_label"] = str(stop.get("action_label") or "")
    state["stop_loss_mask"] = bool(stop.get("action_loss_mask"))
    state["ranking_loss_mask"] = bool(pos and neg)
    return state if state["ranking_loss_mask"] else None


# ---------------------------------------------------------------------------
# Bundle rewrite
# ---------------------------------------------------------------------------


def hardlink_or_copy(src: Path, dst: Path) -> str:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with contextlib.suppress(FileNotFoundError):
        dst.unlink()
    try:
        os.link(src, dst)
        return "hardlink"
    except OSError:
        shutil.copy2(src, dst)
        return "copy"


def existing_complete_state(states: Sequence[Mapping[str, Any]], obligations: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    candidates = []
    for s in states:
        ids = list(map(str, s.get("evidence_ids") or []))
        if str(s.get("state_type") or "") == "complete" and is_sufficient(ids, obligations):
            candidates.append(s)
    if not candidates:
        for s in states:
            ids = list(map(str, s.get("evidence_ids") or []))
            if is_sufficient(ids, obligations):
                candidates.append(s)
    if not candidates:
        return None
    candidates.sort(key=lambda s: (len(s.get("evidence_ids") or []), str(s.get("state_id") or "")))
    return candidates[0]


def rewrite_tasks(
    *,
    source_tasks: Path,
    output_tasks: Path,
    splits_to_rebuild: set[str],
    evidence_lookup: EvidenceLookup,
    trainer: Any,
    policy_builder: Any,
    tokenizer: Any,
    max_boundaries_per_task: int,
    max_actions: int,
    max_positive_pairs: int,
    state_body_token_budget: int,
    keep_original_anchors: bool,
    progress_every: int,
) -> dict[str, Any]:
    import pyarrow as pa
    import pyarrow.parquet as pq

    pf = pq.ParquetFile(source_tasks)
    tmp = output_tasks.with_name(output_tasks.name + f".tmp.{os.getpid()}")
    with contextlib.suppress(FileNotFoundError):
        tmp.unlink()
    writer = pq.ParquetWriter(tmp, pf.schema_arrow, compression="zstd", use_dictionary=True, write_statistics=True)

    c = Counter()
    stage_counts = Counter()
    model_max = int(getattr(trainer, "MODEL_MAX_LENGTH", 4096))
    started = time.perf_counter()
    try:
        for rg in range(pf.num_row_groups):
            rows = pf.read_row_group(rg, use_threads=True).to_pylist()
            out_rows: list[dict[str, Any]] = []
            for row in rows:
                c["tasks"] += 1
                split = str(row.get("split") or "")
                if split not in splits_to_rebuild or row.get("experiment_eligible") is not True:
                    out_rows.append(row)
                    continue

                row = copy.deepcopy(row)
                sup = copy.deepcopy(row.get("supervision") or {})
                states = list(sup.get("policy_states") or [])
                obs = applicable_obligations(sup)
                if not obs:
                    c[f"skip_no_mandatory:{split}"] += 1
                    sup["policy_states"] = [] if not keep_original_anchors else [
                        copy.deepcopy(s) for s in states if str(s.get("state_type") or "") in {"initial", "complete"}
                    ]
                    row["supervision"] = sup
                    out_rows.append(row)
                    continue

                complete = existing_complete_state(states, obs)
                if complete is None:
                    c[f"skip_no_complete:{split}"] += 1
                    sup["policy_states"] = [] if not keep_original_anchors else [
                        copy.deepcopy(s) for s in states if str(s.get("state_type") or "") in {"initial", "complete"}
                    ]
                    row["supervision"] = sup
                    out_rows.append(row)
                    continue

                complete_ids = list(map(str, complete.get("evidence_ids") or []))
                all_witness = witness_ids(obs)
                ids_for_cost = sorted(set(complete_ids) | all_witness)
                try:
                    costs = evidence_lookup.token_costs(ids_for_cost)
                except KeyError:
                    c[f"skip_missing_evidence:{split}"] += 1
                    sup["policy_states"] = []
                    row["supervision"] = sup
                    out_rows.append(row)
                    continue

                cert = list(policy_builder._minimum_sufficient_certificate(list(sup.get("obligations") or []), costs))
                if len(cert) < 2:
                    c[f"skip_certificate_lt2:{split}"] += 1
                    sup["policy_states"] = [] if not keep_original_anchors else [
                        copy.deepcopy(s) for s in states if str(s.get("state_type") or "") in {"initial", "complete"}
                    ]
                    row["supervision"] = sup
                    out_rows.append(row)
                    continue

                order = acquisition_order(cert, obs, costs)
                prefix_lengths = select_prefix_lengths(len(order), max_boundaries_per_task)
                singles, pairs = collect_action_templates(states)
                generated: list[dict[str, Any]] = []
                for out_index, prefix_len in enumerate(prefix_lengths, start=1):
                    k = order[:prefix_len]
                    canonical_next = order[prefix_len]
                    state = build_boundary_state(
                        task_id=str(row.get("task_id") or ""),
                        row_input=row.get("input") or {},
                        original_states=states,
                        obligations=obs,
                        witness_set=all_witness,
                        k=k,
                        canonical_next=canonical_next,
                        prefix_index=out_index,
                        total_prefixes=len(prefix_lengths),
                        singles=singles,
                        pairs=pairs,
                        evidence_lookup=evidence_lookup,
                        trainer=trainer,
                        policy_builder=policy_builder,
                        tokenizer=tokenizer,
                        raw_obligations=list(sup.get("obligations") or []),
                        hard_negative_ids=set(map(str, sup.get("hard_negative_evidence_ids") or [])),
                        max_actions=max_actions,
                        max_positive_pairs=max_positive_pairs,
                        state_body_token_budget=state_body_token_budget,
                        model_max_length=model_max,
                    )
                    if state is None:
                        c[f"skip_unscoreable_state:{split}"] += 1
                        continue
                    generated.append(state)
                    _, p = state_metrics(k, obs)
                    assert p is not None
                    stage_counts[f"{split}:{stage_name(p)}"] += 1

                if keep_original_anchors:
                    anchors = [
                        copy.deepcopy(s)
                        for s in states
                        if str(s.get("state_type") or "") in {"initial", "complete"}
                    ]
                else:
                    anchors = []
                sup["policy_states"] = [*generated, *anchors]
                row["supervision"] = sup
                out_rows.append(row)

                c[f"rebuilt_tasks:{split}"] += 1
                c[f"generated_boundaries:{split}"] += len(generated)
                c[f"anchors:{split}"] += len(anchors)
                c[f"certificate_evidence:{split}"] += len(cert)
                if c["tasks"] % progress_every == 0:
                    elapsed = max(1e-9, time.perf_counter() - started)
                    log(
                        f"[multistage] tasks={c['tasks']:,} "
                        f"train_boundary={c['generated_boundaries:train']:,} "
                        f"val_boundary={c['generated_boundaries:validation']:,} "
                        f"rate={c['tasks']/elapsed:.2f} task/s"
                    )

            table = pa.Table.from_pylist(out_rows, schema=pf.schema_arrow)
            writer.write_table(table)
    except BaseException:
        writer.close()
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    else:
        writer.close()
        os.replace(tmp, output_tasks)

    return {
        **dict(c),
        "stage_counts": dict(stage_counts),
        "seconds": round(time.perf_counter() - started, 3),
    }


def audit_output(tasks_path: Path, trainer: Any) -> dict[str, Any]:
    """Use the actual trainer gate plus extra derived-state semantic checks."""
    report = trainer.audit_task_supervision(tasks_path, show_progress=False)
    errors: list[str] = []
    counts = Counter()
    for split in ("train", "validation"):
        for row in trainer.iter_task_rows(tasks_path, split=split, eligible_only=True, shuffle_row_groups=False):
            sup = row.get("supervision") or {}
            obs = applicable_obligations(sup)
            for s in sup.get("policy_states") or []:
                if str(s.get("state_type") or "") != "decision_boundary":
                    continue
                counts[f"{split}:boundary"] += 1
                k = list(map(str, s.get("evidence_ids") or []))
                completion, _ = state_metrics(k, obs)
                if not k:
                    errors.append(f"EMPTY_BOUNDARY:{row.get('task_id')}:{s.get('state_id')}")
                if completion is None or completion >= 1.0 - EPS:
                    errors.append(f"BOUNDARY_NOT_INCOMPLETE:{row.get('task_id')}:{s.get('state_id')}:{completion}")
                if str(s.get("stop_label") or "") != "negative":
                    errors.append(f"BOUNDARY_STOP_NOT_NEGATIVE:{row.get('task_id')}:{s.get('state_id')}")
                active = trainer.active_actions(s)
                pos = [a for a in active if a.get("action_label") == "positive"]
                neg = [a for a in active if a.get("action_label") == "negative"]
                if not pos or not neg:
                    errors.append(f"BOUNDARY_RANKING_INVALID:{row.get('task_id')}:{s.get('state_id')}")
                for a in pos:
                    cg, pg = candidate_gain(k, list(map(str, a.get("evidence_ids") or [])), obs)
                    if str(a.get("action_type") or "") == "stop" or (cg <= EPS and pg <= EPS):
                        errors.append(f"POSITIVE_WITHOUT_GAIN:{row.get('task_id')}:{s.get('state_id')}:{a.get('action_id')}")
    return {
        "status": "PASS" if not errors else "FAIL",
        "error_count": len(errors),
        "errors": errors[:100],
        "derived_counts": dict(counts),
        "trainer_audit": report,
    }


def build_manifest(
    *,
    source_manifest: Mapping[str, Any],
    source_manifest_sha: str,
    output_dir: Path,
    tasks_path: Path,
    evidence_path: Path,
    evidence_mode: str,
    rewrite_report: Mapping[str, Any],
    audit: Mapping[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    manifest = copy.deepcopy(dict(source_manifest))
    manifest["dataset_name"] = "evidence_agent_multistage_boundary_v1"
    manifest["dataset_version"] = "1.0.0"
    manifest["script_version"] = SCRIPT_VERSION
    manifest["status"] = "FROZEN"
    manifest["training_ready"] = audit.get("status") == "PASS"
    manifest["semantic_review_complete"] = True
    manifest["integrity_audit_passed"] = audit.get("status") == "PASS"
    manifest["files_created"] = 3
    contracts = copy.deepcopy(dict(source_manifest.get("contracts") or {}))
    contracts.update({
        "benchmark_for_training_or_tuning": False,
        "policy_evidence_is_self_contained_for_offline_training": True,
        "source_bundle_immutable": True,
        "multistage_boundary_derived_from_existing_teacher_only": True,
        "benchmark_supervision_not_rebuilt": True,
    })
    manifest["contracts"] = contracts
    src_ev = ((source_manifest.get("files") or {}).get("policy_evidence.parquet") or {})
    manifest["files"] = {
        "tasks.parquet": {
            "bytes": tasks_path.stat().st_size,
            "sha256": sha256_file(tasks_path),
        },
        "policy_evidence.parquet": {
            "bytes": evidence_path.stat().st_size,
            "rows": src_ev.get("rows"),
            "sha256": str(src_ev.get("sha256") or sha256_file(evidence_path)),
        },
    }
    manifest["source"] = {
        "source_bundle": str(args.source_bundle),
        "source_manifest_sha256": source_manifest_sha,
        "source_dataset_name": source_manifest.get("dataset_name"),
        "source_dataset_version": source_manifest.get("dataset_version"),
        "stage1_model_dir": str(args.model_dir),
        "trainer_script": str(args.trainer_script),
        "frozen_policy_builder": str(args.policy_builder),
        "evidence_cache": str(args.evidence_cache),
    }
    manifest["multistage_boundary"] = {
        "semantics": "non-empty incomplete K prefixes under mandatory OR-of-AND witness obligations",
        "progress_metric": "mean_obligation_max_witness_fraction",
        "positive_action_rule": "completion_gain>0 OR progress_gain>0",
        "stop_rule": "negative for every derived boundary; STOP positive remains learned from source complete states in stage-1",
        "splits_rebuilt": sorted(args.rebuild_splits),
        "keep_original_anchors": bool(args.keep_original_anchors),
        "max_boundaries_per_task": int(args.max_boundaries_per_task),
        "max_actions_per_state": int(args.max_actions_per_state),
        "max_positive_pairs": int(args.max_positive_pairs),
        "state_body_token_budget": int(args.state_body_token_budget),
        "policy_evidence_mode": evidence_mode,
        "rewrite_report": dict(rewrite_report),
    }
    manifest["final_audit"] = dict(audit)
    # Remove source runtime details that are not files in this offline stage-2 bundle.
    manifest.pop("repository_runtime", None)
    return manifest


# ---------------------------------------------------------------------------
# CLI / self-test
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--source-bundle", type=Path, default=Path("data/evidence_agent_dataset_v1"))
    p.add_argument("--output-bundle", type=Path, default=Path("data/evidence_agent_multistage_boundary_v1"))
    p.add_argument("--trainer-script", type=Path, default=Path("scripts/train_evidence_policy.py"))
    p.add_argument("--policy-builder", type=Path, default=Path("scripts/build_unified_dataset_v2_10.py"), help="frozen V2.10 builder used for exact metrics/labels/rendering")
    p.add_argument("--model-dir", type=Path, required=False, help="stage-1 best checkpoint; tokenizer is loaded from here")
    p.add_argument("--evidence-cache", type=Path, default=Path("models/evidence_policy_v1_0/evidence_lookup.sqlite3"))
    p.add_argument("--rebuild-splits", default="train,validation")
    p.add_argument("--max-boundaries-per-task", type=int, default=6)
    p.add_argument("--max-actions-per-state", type=int, default=16)
    p.add_argument("--max-positive-pairs", type=int, default=4)
    p.add_argument("--state-body-token-budget", type=int, default=1024)
    p.add_argument("--keep-original-anchors", action=argparse.BooleanOptionalAction, default=False,
                   help="also retain original initial/complete states in rebuilt train/validation")
    p.add_argument("--local-files-only", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--progress-every", type=int, default=1000)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--self-test", action="store_true")
    args = p.parse_args()
    args.rebuild_splits = {x.strip() for x in str(args.rebuild_splits).split(",") if x.strip()}
    allowed = {"train", "validation"}
    if not args.rebuild_splits or not args.rebuild_splits <= allowed:
        p.error(f"--rebuild-splits must be subset of {sorted(allowed)}")
    if args.max_boundaries_per_task < 1:
        p.error("--max-boundaries-per-task must be >=1")
    if args.max_actions_per_state < 2:
        p.error("--max-actions-per-state must be >=2")
    if args.max_positive_pairs < 0:
        p.error("--max-positive-pairs must be >=0")
    if args.state_body_token_budget < 0:
        p.error("--state-body-token-budget must be >=0")
    return args


def self_test() -> int:
    obs = [
        {"obligation_id": "o1", "mandatory": True, "groups": [{"a", "b"}, {"x"}]},
        {"obligation_id": "o2", "mandatory": True, "groups": [{"c"}]},
    ]
    c, p = state_metrics([], obs)
    assert c == 0.0 and p == 0.0
    c, p = state_metrics(["a"], obs)
    assert c == 0.0 and abs(p - 0.25) < 1e-9
    c, p = state_metrics(["a", "b"], obs)
    assert abs(c - 0.5) < 1e-9 and abs(p - 0.5) < 1e-9
    assert is_sufficient(["a", "b", "c"], obs)
    assert is_sufficient(["x", "c"], obs)
    costs = {"a": 5, "b": 5, "c": 3, "noise": 100}
    cert = minimize_complete_certificate(["a", "b", "c", "noise"], obs, costs)
    assert cert == ["a", "b", "c"], cert
    order = acquisition_order(cert, obs, costs)
    assert set(order) == {"a", "b", "c"} and len(order) == 3
    # Every canonical prefix is incomplete and the next item strictly increases progress.
    for i in range(1, len(order)):
        k = order[:i]
        assert not is_sufficient(k, obs)
        cg, pg = candidate_gain(k, [order[i]], obs)
        assert cg > EPS or pg > EPS
    assert select_prefix_lengths(8, 4)[0] == 1
    assert select_prefix_lengths(8, 4)[-1] == 7
    assert stage_name(0.1) == "early"
    assert stage_name(0.5) == "mid"
    assert stage_name(0.8) == "late"
    assert stage_name(0.95) == "near_complete"
    print("SELF_TEST_OK")
    return 0


def install_warning_free_tokenizer_contract(*, trainer: Any, policy_builder: Any) -> None:
    """Reuse the stage-1 trainer's warning-free token inspection helpers.

    The frozen V2.10 policy builder predates the trainer v1.3 fix for very long
    SWE issue text.  Its render path can legitimately tokenize a >tokenizer-native
    max-length *temporary* string merely to measure it before applying the explicit
    4096 model-input contract.  Transformers emits a misleading warning in that
    situation even though no over-length tensor is sent to the model.

    Stage-2 keeps the frozen supervision/render semantics, but swaps only the
    length-inspection helpers for the already-audited warning-free trainer helpers.
    """
    if hasattr(policy_builder, "_truncate_question_view"):
        policy_builder._truncate_question_view = trainer.truncate_question_view

    if hasattr(policy_builder, "_model_token_count"):
        def _safe_model_token_count(tokenizer: Any, text: str) -> int:
            return int(
                trainer.token_length_without_warning(
                    tokenizer, text, add_special_tokens=True
                )
            )
        policy_builder._model_token_count = _safe_model_token_count

    if hasattr(policy_builder, "_batch_model_token_counts"):
        def _safe_batch_model_token_counts(tokenizer: Any, texts: Sequence[str]) -> list[int]:
            return [
                int(
                    trainer.token_length_without_warning(
                        tokenizer, text, add_special_tokens=True
                    )
                )
                for text in texts
            ]
        policy_builder._batch_model_token_counts = _safe_batch_model_token_counts


def main() -> int:
    args = parse_args()
    if args.self_test:
        return self_test()
    if args.model_dir is None:
        raise SystemExit("--model-dir is required unless --self-test")

    root = Path.cwd().resolve()
    def resolve(p: Path) -> Path:
        return p.resolve() if p.is_absolute() else (root / p).resolve()

    source_bundle = resolve(args.source_bundle)
    output_bundle = resolve(args.output_bundle)
    trainer_path = resolve(args.trainer_script)
    policy_builder_path = resolve(args.policy_builder)
    model_dir = resolve(args.model_dir)
    evidence_cache_path = resolve(args.evidence_cache)
    args.source_bundle = source_bundle
    args.output_bundle = output_bundle
    args.trainer_script = trainer_path
    args.policy_builder = policy_builder_path
    args.model_dir = model_dir
    args.evidence_cache = evidence_cache_path

    source_manifest_path = source_bundle / "manifest.json"
    source_tasks = source_bundle / "tasks.parquet"
    source_evidence = source_bundle / "policy_evidence.parquet"
    for p in (source_manifest_path, source_tasks, source_evidence, trainer_path, policy_builder_path, evidence_cache_path):
        if not p.is_file():
            raise FileNotFoundError(p)
    if not model_dir.is_dir():
        raise FileNotFoundError(model_dir)

    if output_bundle.exists():
        if not args.overwrite:
            raise FileExistsError(f"output exists; use --overwrite: {output_bundle}")
        shutil.rmtree(output_bundle)
    output_bundle.mkdir(parents=True, exist_ok=False)

    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("training_ready") is not True:
        raise ValueError("source bundle training_ready != true")
    source_manifest_sha = sha256_file(source_manifest_path)

    trainer = load_module(trainer_path, "_stage2_trainer_contract")
    policy_builder = load_module(policy_builder_path, "_frozen_v210_policy_contract")
    policy_required = ["evidence_state_metrics", "_minimum_sufficient_certificate", "label_candidate_actions", "render_policy_model_inputs", "MODEL_MAX_LENGTH"]
    policy_missing = [x for x in policy_required if not hasattr(policy_builder, x)]
    if policy_missing:
        raise RuntimeError(f"frozen V2.10 builder lacks required helpers: {policy_missing}")
    required = [
        "build_question", "truncate_question_view", "render_action_text",
        "token_length_without_warning", "audit_task_supervision", "iter_task_rows",
        "active_actions", "MODEL_MAX_LENGTH",
    ]
    missing = [x for x in required if not hasattr(trainer, x)]
    if missing:
        raise RuntimeError(f"trainer lacks required contract helpers: {missing}")

    from transformers import AutoTokenizer
    log(f"[multistage] loading tokenizer: {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(
        str(model_dir),
        local_files_only=bool(args.local_files_only),
        trust_remote_code=False,
    )
    install_warning_free_tokenizer_contract(
        trainer=trainer, policy_builder=policy_builder
    )
    log("[multistage] warning-free token inspection enabled")

    evidence_lookup = EvidenceLookup(evidence_cache_path)
    try:
        out_tasks = output_bundle / "tasks.parquet"
        log("[multistage] rebuilding train/validation boundary states ...")
        rewrite_report = rewrite_tasks(
            source_tasks=source_tasks,
            output_tasks=out_tasks,
            splits_to_rebuild=set(args.rebuild_splits),
            evidence_lookup=evidence_lookup,
            trainer=trainer,
            policy_builder=policy_builder,
            tokenizer=tokenizer,
            max_boundaries_per_task=args.max_boundaries_per_task,
            max_actions=args.max_actions_per_state,
            max_positive_pairs=args.max_positive_pairs,
            state_body_token_budget=args.state_body_token_budget,
            keep_original_anchors=args.keep_original_anchors,
            progress_every=max(1, args.progress_every),
        )
    finally:
        evidence_lookup.close()

    out_evidence = output_bundle / "policy_evidence.parquet"
    evidence_mode = hardlink_or_copy(source_evidence, out_evidence)
    log(f"[multistage] policy_evidence: {evidence_mode}")

    log("[multistage] auditing derived supervision ...")
    audit = audit_output(out_tasks, trainer)
    if audit.get("status") != "PASS":
        atomic_json(output_bundle / "audit_failed.json", audit)
        raise RuntimeError(f"derived bundle audit failed: {audit.get('errors', [])[:10]}")

    manifest = build_manifest(
        source_manifest=source_manifest,
        source_manifest_sha=source_manifest_sha,
        output_dir=output_bundle,
        tasks_path=out_tasks,
        evidence_path=out_evidence,
        evidence_mode=evidence_mode,
        rewrite_report=rewrite_report,
        audit=audit,
        args=args,
    )
    atomic_json(output_bundle / "manifest.json", manifest)

    result = {
        "status": "PASS",
        "script_version": SCRIPT_VERSION,
        "output_bundle": str(output_bundle),
        "rewrite_report": rewrite_report,
        "audit": {
            "status": audit.get("status"),
            "error_count": audit.get("error_count"),
            "derived_counts": audit.get("derived_counts"),
        },
        "training_ready": manifest.get("training_ready"),
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Compare reference vs two-pass streaming gradients on one fixed state.

Diagnostic only. No optimizer step is executed and no checkpoint is modified.
Runs the same healthy checkpoint/state under BF16 and FP32 and reports whole-model
gradient agreement.
"""

from __future__ import annotations

import math
import random
import sys
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, str(Path("scripts").resolve()))
import train_evidence_policy as T


BUNDLE = Path("data/evidence_agent_final_v1")
CACHE = Path("models/evidence_policy_corrected_v2/evidence_lookup.sqlite3")
CKPT = Path("models/evidence_policy_corrected_v1/recovery")

SEED = 42
DROPOUT_SEED = 123456
GRAD_ACCUM = 8
PRECISIONS = ("bf16", "fp32")


def reset_rng() -> None:
    random.seed(DROPOUT_SEED)
    torch.manual_seed(DROPOUT_SEED)
    torch.cuda.manual_seed_all(DROPOUT_SEED)


def load_model(device: torch.device):
    model = AutoModelForSequenceClassification.from_pretrained(
        CKPT,
        local_files_only=True,
    )
    model.to(device)
    model.train()
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    model.zero_grad(set_to_none=True)
    return model


def capture_grads(model):
    return {
        name: p.grad.detach().float().cpu().clone()
        for name, p in model.named_parameters()
        if p.grad is not None
    }


def compare_grads(ref_grads, model):
    ref_sq = 0.0
    stream_sq = 0.0
    diff_sq = 0.0
    dot = 0.0
    max_abs_diff = 0.0
    max_abs_name = None
    missing = []
    reports = []

    for name, p in model.named_parameters():
        g1 = ref_grads.get(name)
        g2 = p.grad

        if g1 is None and g2 is None:
            continue
        if g1 is None or g2 is None:
            missing.append(name)
            continue

        g2 = g2.detach().float().cpu()
        if tuple(g1.shape) != tuple(g2.shape):
            raise RuntimeError(
                f"gradient shape mismatch: {name}: "
                f"{tuple(g1.shape)} vs {tuple(g2.shape)}"
            )

        d = g2 - g1
        r2 = float(torch.sum(g1 * g1))
        s2 = float(torch.sum(g2 * g2))
        d2 = float(torch.sum(d * d))
        dp = float(torch.sum(g1 * g2))

        ref_sq += r2
        stream_sq += s2
        diff_sq += d2
        dot += dp

        local_max = float(d.abs().max()) if d.numel() else 0.0
        if local_max > max_abs_diff:
            max_abs_diff = local_max
            max_abs_name = name

        ref_norm = math.sqrt(r2)
        reports.append(
            (
                math.sqrt(d2) / (ref_norm + 1e-30),
                local_max,
                name,
                ref_norm,
                math.sqrt(s2),
            )
        )

    ref_norm = math.sqrt(ref_sq)
    stream_norm = math.sqrt(stream_sq)
    diff_norm = math.sqrt(diff_sq)
    relative_l2 = diff_norm / (ref_norm + 1e-30)
    cosine = dot / (ref_norm * stream_norm + 1e-30)

    return {
        "missing": missing,
        "ref_norm": ref_norm,
        "stream_norm": stream_norm,
        "diff_norm": diff_norm,
        "relative_l2": relative_l2,
        "cosine": cosine,
        "max_abs_diff": max_abs_diff,
        "max_abs_name": max_abs_name,
        "reports": sorted(reports, reverse=True),
    }


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this diagnostic")

    bundle = T.load_bundle(BUNDLE, verify_hashes=False)
    fingerprint = T.cache_fingerprint(bundle)

    tokenizer = AutoTokenizer.from_pretrained(
        CKPT,
        use_fast=True,
        local_files_only=True,
    )
    cache = T.EvidenceCache(CACHE, fingerprint)

    builder = T.ExampleBuilder(
        cache=cache,
        tokenizer=tokenizer,
        max_candidates=12,
        pair_negative_quota=3,
        offline_positive_weight=1.0,
        verify_token_counts=False,
        verify_limit=0,
    )

    example = None
    for ex in T.iter_state_examples(
        bundle["tasks_path"],
        split="validation",
        builder=builder,
        epoch=0,
        seed=SEED + 97_531,
        boundary_repeat=1,
        shuffle=False,
        max_states=None,
    ):
        if ex.state_type == "complete" and len(ex.texts) == 12:
            example = ex
            break

    if example is None:
        raise RuntimeError("没有找到 12-candidate Complete state")

    print("task =", example.task_id)
    print("state =", example.state_id)
    print("state_type =", example.state_type)
    print("candidates =", len(example.texts))
    print("stop_index =", example.stop_index)

    device = torch.device("cuda")

    for precision in PRECISIONS:
        print("\n" + "=" * 88)
        print("PRECISION =", precision)
        print("=" * 88)

        print("\n===== REFERENCE BACKWARD =====")
        model = load_model(device)
        reset_rng()

        ref_scores, ref_loss = T.reference_listwise_backward(
            model,
            tokenizer,
            example,
            device=device,
            precision=precision,
            candidate_microbatch=1,
            grad_accum_steps=GRAD_ACCUM,
            use_state_confidence=False,
            scaler=None,
        )

        ref_score_cpu = ref_scores.detach().float().cpu().clone()
        ref_grads = capture_grads(model)

        print("reference loss =", float(ref_loss.detach().float().cpu()))
        print("reference scores =", ref_score_cpu.tolist())
        print("reference gradient tensors =", len(ref_grads))

        del ref_scores, ref_loss, model
        torch.cuda.empty_cache()

        print("\n===== TWO-PASS STREAMING BACKWARD =====")
        model = load_model(device)
        reset_rng()

        stream_scores, stream_loss = T.streaming_listwise_backward(
            model,
            tokenizer,
            example,
            device=device,
            precision=precision,
            candidate_microbatch=1,
            grad_accum_steps=GRAD_ACCUM,
            use_state_confidence=False,
            scaler=None,
            verify_replay=True,
            replay_atol=1e-5,
        )

        stream_score_cpu = stream_scores.detach().float().cpu().clone()

        print("stream loss =", float(stream_loss.detach().float().cpu()))
        print("stream scores =", stream_score_cpu.tolist())
        print(
            "max score abs diff =",
            float((stream_score_cpu - ref_score_cpu).abs().max()),
        )

        result = compare_grads(ref_grads, model)

        print("\n================ GRADIENT COMPARISON ================")
        print("missing gradient tensors =", len(result["missing"]))
        if result["missing"]:
            print("first missing =", result["missing"][:20])
        print("reference global grad norm =", result["ref_norm"])
        print("stream global grad norm    =", result["stream_norm"])
        print("gradient diff L2           =", result["diff_norm"])
        print("relative gradient L2 diff  =", result["relative_l2"])
        print("gradient cosine similarity =", result["cosine"])
        print("max absolute grad diff      =", result["max_abs_diff"])
        print("max diff parameter          =", result["max_abs_name"])

        print("\nTop 10 parameter relative differences:")
        for rel, absmax, name, n1, n2 in result["reports"][:10]:
            print(
                f"{name}\n"
                f"  relative_L2={rel:.9g} "
                f"max_abs={absmax:.9g} "
                f"ref_norm={n1:.9g} "
                f"stream_norm={n2:.9g}"
            )

        del stream_scores, stream_loss, model, ref_grads
        torch.cuda.empty_cache()

    cache.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

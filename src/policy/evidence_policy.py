"""加载训练完成的 Cross-Encoder Evidence Policy 并统一评分。"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any, Mapping, Sequence

from .input_renderer import PolicyInputRenderer


class EvidencePolicy:
    """对给定状态中的 Single、Pair、STOP 输出可比较的标量分数。"""

    def __init__(
        self,
        checkpoint_dir: Path | str,
        *,
        device: str = "auto",
        precision: str = "auto",
        candidate_microbatch: int = 1,
    ) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        checkpoint = Path(checkpoint_dir)
        self.device = torch.device(
            "cuda"
            if device == "auto" and torch.cuda.is_available()
            else device
            if device != "auto"
            else "cpu"
        )
        if precision == "auto":
            if self.device.type == "cuda" and torch.cuda.is_bf16_supported():
                precision = "bf16"
            elif self.device.type == "cuda":
                precision = "fp16"
            else:
                precision = "fp32"
        self.precision = precision
        self.candidate_microbatch = candidate_microbatch
        self.tokenizer = AutoTokenizer.from_pretrained(
            checkpoint,
            use_fast=True,
            local_files_only=True,
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint,
            local_files_only=True,
        ).to(self.device)
        self.model.eval()
        self.renderer = PolicyInputRenderer(self.tokenizer)

    def _autocast(self) -> Any:
        import torch

        if self.device.type != "cuda" or self.precision == "fp32":
            return contextlib.nullcontext()
        dtype = torch.bfloat16 if self.precision == "bf16" else torch.float16
        return torch.autocast(device_type="cuda", dtype=dtype)

    def _score_texts(self, texts: Sequence[str]) -> list[float]:
        import torch

        results = []
        with torch.inference_mode():
            for offset in range(0, len(texts), self.candidate_microbatch):
                chunk = list(texts[offset : offset + self.candidate_microbatch])
                encoded = self.tokenizer(
                    chunk,
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                )
                encoded = {key: value.to(self.device) for key, value in encoded.items()}
                with self._autocast():
                    logits = self.model(**encoded).logits
                if logits.ndim == 2 and logits.shape[-1] == 1:
                    scores = logits[:, 0]
                elif logits.ndim == 1:
                    scores = logits
                else:
                    raise ValueError(
                        f"Evidence Policy 必须为每个动作输出一个分数，实际形状为 {tuple(logits.shape)}"
                    )
                results.extend(float(score) for score in scores.detach().float().cpu())
        return results

    def rank_actions(
        self,
        *,
        task_input: Mapping[str, Any],
        current_evidence: Sequence[Mapping[str, Any]],
        actions: Sequence[Mapping[str, Any]],
    ) -> list[dict[str, Any]]:
        """在同一个 (q, K) 下统一排序全部可评分动作。"""

        rendered = self.renderer.render_actions(
            task_input=task_input,
            current_evidence=current_evidence,
            actions=actions,
        )
        scores = self._score_texts([item["text"] for item in rendered])
        ranked = [
            {
                **item["action"],
                "score": score,
                "model_input_token_count": item["model_input_token_count"],
                "rendered_state_body_evidence_ids": item[
                    "rendered_state_body_evidence_ids"
                ],
            }
            for item, score in zip(rendered, scores)
        ]
        return sorted(ranked, key=lambda action: float(action["score"]), reverse=True)

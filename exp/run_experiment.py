"""统一运行 Evidence Agent、Baseline、语义 Judge 和结果汇总。"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from pathlib import Path
from threading import Lock
from typing import Any, Iterable, Mapping, Sequence

import pyarrow.dataset as ds
from dotenv import load_dotenv
from openai import OpenAI
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from exp.ablations import ABLATION_VARIANTS, build_ablation  # noqa: E402
from exp.api_usage import capture_api_usage, record_api_usage  # noqa: E402
from exp.baselines import (  # noqa: E402
    BM25Baseline,
    DenseBaseline,
    DenseEncoder,
    FixedIterativeBaseline,
    HybridBaseline,
    OneShotBaseline,
    RerankBaseline,
    RerankCaller,
)
from exp.baselines.external import (  # noqa: E402
    AgentlessBaseline,
    AgentlessOutputStore,
    LocAgentBaseline,
    LocAgentOutputStore,
    SweRankBaseline,
    SweRankOutputStore,
)
from exp.ours import build_ours  # noqa: E402
from exp.provenance import (  # noqa: E402
    artifact_identity,
    code_identity,
    ensure_manifest,
    manifest_path,
)
from src.agents import RETRIEVAL_CHANNELS, RetrievalPlanner  # noqa: E402
from src.agents.planner import PLANNER_PROMPT_VERSION  # noqa: E402
from src.data import RuntimeRepository, TaskReader, build_online_issue  # noqa: E402
from src.data.supervision_reader import SupervisionReader  # noqa: E402
from src.evaluation.aggregation import aggregate_rows  # noqa: E402
from src.evaluation.cost_metrics import auc_sufficiency_cost  # noqa: E402
from src.evaluation.interaction_metrics import evaluate_interactions  # noqa: E402
from src.evaluation.localization_metrics import localization_metrics  # noqa: E402
from src.evaluation.retrieval_metrics import retrieval_metrics  # noqa: E402
from src.evaluation.semantic_judge import (  # noqa: E402
    SEMANTIC_JUDGE_PROMPT_VERSION,
    judge_evidence_package,
)
from src.evaluation.semantic_metrics import (  # noqa: E402
    aggregate_semantic_judgments,
)
from src.evaluation.sufficiency_metrics import evaluate_sufficiency  # noqa: E402
from src.evaluation.trajectory_metrics import evaluate_trajectories  # noqa: E402
from src.policy import EvidencePolicy  # noqa: E402
from src.retrieval import RepositoryRAG  # noqa: E402


DEFAULT_TASKS = PROJECT_ROOT / "data/evidence_agent_dataset_v1/tasks.parquet"
DEFAULT_RUNTIME = (
    PROJECT_ROOT / "data/evidence_agent_dataset_v1/repository_runtime.sqlite3"
)
DEFAULT_RESULTS = PROJECT_ROOT / "exp/results"
OURS_METHODS = ("ours",)
ABLATION_METHODS = tuple(ABLATION_VARIANTS)
POLICY_METHODS = (*OURS_METHODS, *ABLATION_METHODS)
DENSE_METHODS = ("dense", "hybrid")
RERANK_METHODS = ("rerank",)
EXTERNAL_METHODS = ("swerank", "agentless", "locagent")
CHAT_API_METHODS = ("one_shot", "fixed_iterative", *POLICY_METHODS)
API_METHODS = (*DENSE_METHODS, *RERANK_METHODS, *CHAT_API_METHODS)
RUN_METHODS = (
    "bm25",
    *DENSE_METHODS,
    *RERANK_METHODS,
    *EXTERNAL_METHODS,
    "one_shot",
    "fixed_iterative",
    *POLICY_METHODS,
)
API_PROFILES = {
    "openai": {
        "base_env": "OPENAI_BASE_URL",
        "key_envs": (
            "OPENAI_API_KEY",
            "OPENAI_API_KEY_2",
            "OPENAI_API_KEY_3",
            "OPENAI_API_KEY_4",
        ),
        "model_env": "LLM_MODEL",
        "non_thinking_body": {"enable_thinking": False},
    },
    "qwen": {
        "base_env": "QWEN_API_URL",
        "key_envs": ("QWEN_API_KEY",),
        "model_env": "QWEN_MODEL",
        "non_thinking_body": {"enable_thinking": False},
    },
    "glm": {
        "base_env": "BIGMOD_API_URL",
        "key_envs": (
            "BIGMOD_API_KEY",
            "BIGMOD_API_KEY_2",
            "BIGMOD_API_KEY_3",
            "BIGMOD_API_KEY_4",
            "BIGMOD_API_KEY_5",
        ),
        "model_env": "BIGMOD_API_MODEL",
        "non_thinking_body": {"thinking": {"type": "disabled"}},
    },
    "deepseek": {
        "base_default": "https://api.deepseek.com",
        "key_envs": ("DEEPSEEK_API_KEY",),
        "model_default": "deepseek-v4-flash",
        "non_thinking_body": {"thinking": {"type": "disabled"}},
    },
    "lin": {
        "base_env": "lin_API_URL",
        "key_envs": (
            "lin_API_KEY",
            "lin_API_KEY_1",
            "lin_API_KEY_2",
            "lin_API_KEY_3",
        ),
        "model_env": "LIN_MODEL",
        "non_thinking_body": {"thinking": {"type": "disabled"}},
    },
    "custom": {
        "key_envs": (),
        "non_thinking_body": {"enable_thinking": False},
    },
}


class OpenAICompatibleCaller:
    """使用 OpenAI-compatible Chat Completions 接口返回纯文本。"""

    def __init__(
        self,
        *,
        model: str,
        api_base: str | None,
        api_keys: Sequence[str],
        timeout: float,
        max_retries: int,
        extra_body: Mapping[str, Any],
    ) -> None:
        self.model = model
        self.clients = [
            OpenAI(
                api_key=api_key,
                base_url=api_base,
                timeout=timeout,
                max_retries=max_retries,
            )
            for api_key in api_keys
        ]
        self.call_count = 0
        self.pool_lock = Lock()
        self.extra_body = dict(extra_body)

    def __call__(self, prompt: str) -> str:
        with self.pool_lock:
            client = self.clients[self.call_count % len(self.clients)]
            self.call_count += 1
        response = client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            extra_body=self.extra_body,
        )
        record_api_usage(response.usage)
        return str(response.choices[0].message.content)


class SerializedPolicy:
    """共享一份 checkpoint，并串行执行 GPU Policy 推理。"""

    def __init__(self, policy: EvidencePolicy) -> None:
        self.policy = policy
        self.policy_lock = Lock()

    def rank_actions(self, **arguments: Any) -> list[dict[str, Any]]:
        with self.policy_lock:
            return self.policy.rank_actions(**arguments)


def read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    """逐行读取 UTF-8 JSONL。"""

    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                yield json.loads(line)


def latest_successful_records(
    paths: Sequence[Path],
) -> dict[tuple[str, str], dict[str, Any]]:
    """按 `(run_name, task_id)` 保留最后一条成功记录。"""

    records: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        for row in read_jsonl(path):
            if row.get("status") == "ok":
                key = (
                    str(row.get("run_name") or row["method"]),
                    str(row["task_id"]),
                )
                records[key] = row
    return records


def completed_task_ids(path: Path) -> set[str]:
    """读取已经成功完成、续跑时应跳过的任务 ID。"""

    return {
        str(row["task_id"]) for row in read_jsonl(path) if row.get("status") == "ok"
    }


def validate_result_manifest(path: Path) -> None:
    """核对显式实验输入中的每条结果都属于同一个运行配置。"""

    sidecar = manifest_path(path)
    if not sidecar.exists():
        raise ValueError(f"正式汇总输入缺少运行 manifest：{path}")
    expected = str(json.loads(sidecar.read_text(encoding="utf-8"))["run_config_hash"])
    mismatched = sum(
        1
        for row in read_jsonl(path)
        if row.get("status") == "ok" and str(row.get("run_config_hash")) != expected
    )
    if mismatched:
        raise ValueError(f"{path} 中有 {mismatched} 条结果不属于其运行 manifest")


def file_label(value: str) -> str:
    """将模型或 checkpoint 名称转换为结果文件名片段。"""

    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def dense_cache_path(args: argparse.Namespace) -> Path:
    """生成与 API Profile 和 Dense 模型绑定的文件向量缓存路径。"""

    model = str(args.active_dense_model)
    return Path(
        args.dense_cache
        or PROJECT_ROOT
        / "exp/cache/dense"
        / f"{args.api_profile}-{file_label(model)}.sqlite3"
    ).resolve()


def default_run_output(args: argparse.Namespace) -> Path:
    """根据方法、模型和预算生成不会混淆配置的默认结果路径。"""

    parts = [args.method]
    if args.method in CHAT_API_METHODS:
        model, _, api_keys = resolve_api_config(
            args,
            model_argument="planner_model",
        )
        args.active_model = model
        args.active_api_pool_size = len(api_keys)
        parts.extend((args.api_profile, file_label(model)))
    if args.method in DENSE_METHODS:
        model, _, api_keys = resolve_api_config(
            args,
            model_argument="dense_model",
            default_model_env="EMBEDDING_MODEL",
        )
        args.active_dense_model = model
        args.active_api_pool_size = len(api_keys)
        parts.extend((args.api_profile, file_label(model)))
    if args.method in RERANK_METHODS:
        model, _, api_keys = resolve_api_config(
            args,
            model_argument="rerank_model",
            default_model_env="RERANK_MODEL",
        )
        args.active_rerank_model = model
        args.active_api_pool_size = len(api_keys)
        parts.extend((args.api_profile, file_label(model)))
    if args.method == "swerank":
        parts.extend(
            (
                file_label(Path(args.swerank_output).stem),
                f"k{args.swerank_top_k}",
            )
        )
    if args.method == "agentless":
        parts.extend(
            (
                file_label(Path(args.agentless_output).stem),
                args.agentless_stage,
            )
        )
    if args.method == "locagent":
        parts.extend(
            (
                file_label(Path(args.locagent_output).stem),
                args.locagent_level,
            )
        )
    if args.checkpoint:
        parts.append(file_label(args.checkpoint.name))
    if args.method not in EXTERNAL_METHODS:
        parts.extend(
            (
                f"u{args.evidence_unit_budget}",
                f"t{args.evidence_token_budget}",
                f"r{args.retrieval_limit}",
            )
        )
    if args.method in ("bm25", *DENSE_METHODS, *RERANK_METHODS):
        parts.append(f"f{args.file_limit}")
    if args.method == "hybrid":
        parts.extend(
            (
                f"cf{args.hybrid_candidate_file_limit}",
                f"k{args.rrf_rank_constant}",
            )
        )
    if args.method == "rerank":
        parts.extend(
            (
                f"cf{args.rerank_candidate_file_limit}",
                f"mc{args.rerank_max_chunks_per_doc}",
                f"o{args.rerank_overlap_tokens}",
            )
        )
    if args.method == "fixed_iterative":
        parts.append(f"s{args.fixed_steps}")
    return DEFAULT_RESULTS / args.split / ("-".join(parts) + ".jsonl")


def append_jsonl(file: Any, row: Mapping[str, Any]) -> None:
    """写入一条结果并立即刷新到磁盘。"""

    file.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
    file.flush()


def task_count(reader: TaskReader, split: str) -> int:
    """计算当前 split 中满足实验资格的任务数。"""

    expression = (ds.field("split") == split) & (
        ds.field("experiment_eligible") == True  # noqa: E712
    )
    return int(reader.dataset.count_rows(filter=expression))


def progress_bar(*, total: int, description: str) -> Any:
    """创建单行、不刷屏的任务进度条。"""

    return tqdm(
        total=total,
        desc=description,
        unit="任务",
        dynamic_ncols=True,
        mininterval=1.0,
        leave=True,
        disable=not sys.stderr.isatty(),
    )


def concurrent_results(
    items: Iterable[Any],
    *,
    total: int,
    concurrency: int,
    description: str,
    worker: Any,
) -> Iterable[tuple[Any, Any, Exception | None]]:
    """以有限在途任务数执行工作，并按完成顺序返回结果。"""

    iterator = iter(items)
    futures: dict[Future[Any], Any] = {}
    failed = 0
    progress = progress_bar(total=total, description=description)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        for _ in range(concurrency):
            try:
                item = next(iterator)
            except StopIteration:
                break
            futures[executor.submit(worker, item)] = item

        while futures:
            done, _ = wait(futures, return_when=FIRST_COMPLETED)
            for future in done:
                item = futures.pop(future)
                try:
                    yield item, future.result(), None
                except Exception as error:  # noqa: BLE001
                    failed += 1
                    progress.set_postfix_str(f"错误={failed}", refresh=False)
                    yield item, None, error
                progress.update(1)
                try:
                    next_item = next(iterator)
                except StopIteration:
                    continue
                futures[executor.submit(worker, next_item)] = next_item
    progress.close()


def resolve_api_config(
    args: argparse.Namespace,
    *,
    model_argument: str,
    default_model_env: str | None = None,
) -> tuple[str, str | None, list[str]]:
    """根据 API Profile 和命令行覆盖项读取模型连接配置。"""

    profile = API_PROFILES[args.api_profile]
    key_envs = tuple(profile["key_envs"])
    model_env = default_model_env or profile.get("model_env")
    base_env = profile.get("base_env")
    selected_key_envs = tuple(args.api_key_env or key_envs)
    model = getattr(args, model_argument) or (
        os.environ.get(model_env) if model_env else None
    ) or profile.get("model_default")
    api_base = args.api_base or (
        os.environ.get(base_env) if base_env else profile.get("base_default")
    )
    api_keys = [
        str(os.environ[name]) for name in selected_key_envs if os.environ.get(name)
    ]
    if not model:
        raise ValueError(f"Profile {args.api_profile} 没有配置模型名")
    if not api_keys:
        raise ValueError(
            f"Profile {args.api_profile} 没有读取到可用的 API Key 环境变量"
        )
    return str(model), api_base, api_keys


def build_planner(args: argparse.Namespace, *, use_structure: bool = True) -> Any:
    """构造 OpenAI-compatible Retrieval Planner。"""

    model, api_base, api_keys = resolve_api_config(
        args,
        model_argument="planner_model",
    )
    args.active_model = model
    args.active_api_pool_size = len(api_keys)
    args.active_api_base = api_base
    channels = (
        RETRIEVAL_CHANNELS
        if use_structure
        else tuple(channel for channel in RETRIEVAL_CHANNELS if channel != "structure")
    )
    caller = OpenAICompatibleCaller(
        model=model,
        api_base=api_base,
        api_keys=api_keys,
        timeout=args.api_timeout,
        max_retries=args.api_max_retries,
        extra_body=API_PROFILES[args.api_profile]["non_thinking_body"],
    )
    return RetrievalPlanner(
        caller,
        retrieval_channels=channels,
        evidence_body_token_budget=args.planner_evidence_body_token_budget,
    )


def build_method(
    args: argparse.Namespace,
    repository: RuntimeRepository,
    *,
    planner: RetrievalPlanner | None,
    policy: Any,
    dense_encoder: DenseEncoder | None,
    rerank_caller: RerankCaller | None,
    external_outputs: Any,
) -> Any:
    """根据 method 名称构造一次实验所需的运行对象。"""

    if args.method == "bm25":
        return BM25Baseline(
            repository,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            file_limit=args.file_limit,
        )
    if args.method == "dense":
        return DenseBaseline(
            repository,
            dense_encoder,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            file_limit=args.file_limit,
        )
    if args.method == "hybrid":
        return HybridBaseline(
            repository,
            dense_encoder,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            file_limit=args.file_limit,
            candidate_file_limit=args.hybrid_candidate_file_limit,
            rank_constant=args.rrf_rank_constant,
        )
    if args.method == "rerank":
        return RerankBaseline(
            repository,
            rerank_caller,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            file_limit=args.file_limit,
            candidate_file_limit=args.rerank_candidate_file_limit,
        )
    if args.method == "swerank":
        return SweRankBaseline(
            repository,
            external_outputs,
            top_k=args.swerank_top_k,
        )
    if args.method == "agentless":
        return AgentlessBaseline(repository, external_outputs)
    if args.method == "locagent":
        return LocAgentBaseline(repository, external_outputs)

    rag = RepositoryRAG(repository)
    if args.method == "one_shot":
        return OneShotBaseline(
            planner,
            rag,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            retrieval_limit=args.retrieval_limit,
        )
    if args.method == "fixed_iterative":
        return FixedIterativeBaseline(
            planner,
            rag,
            max_steps=args.fixed_steps,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            retrieval_limit=args.retrieval_limit,
        )

    if args.method == "ours":
        return build_ours(
            repository,
            planner,
            policy,
            evidence_token_budget=args.evidence_token_budget,
            evidence_unit_budget=args.evidence_unit_budget,
            retrieval_limit=args.retrieval_limit,
            pair_limit=args.pair_limit,
        )

    return build_ablation(
        repository,
        planner,
        policy,
        ABLATION_VARIANTS[args.method],
        evidence_token_budget=args.evidence_token_budget,
        evidence_unit_budget=args.evidence_unit_budget,
        retrieval_limit=args.retrieval_limit,
        pair_limit=args.pair_limit,
    )


def build_shared_resources(
    args: argparse.Namespace,
) -> tuple[
    RetrievalPlanner | None,
    Any,
    DenseEncoder | None,
    RerankCaller | None,
    Any,
]:
    """为并发任务共享 API Pool，并让 Ours 只加载一份 checkpoint。"""

    if args.method == "bm25":
        return None, None, None, None, None
    if args.method in DENSE_METHODS:
        model, api_base, api_keys = resolve_api_config(
            args,
            model_argument="dense_model",
            default_model_env="EMBEDDING_MODEL",
        )
        args.active_dense_model = model
        args.active_api_pool_size = len(api_keys)
        args.active_api_base = api_base
        return (
            None,
            None,
            DenseEncoder(
                model,
                api_base=api_base,
                api_keys=api_keys,
                timeout=args.api_timeout,
                max_retries=args.api_max_retries,
                batch_size=args.dense_batch_size,
                cache_path=dense_cache_path(args),
            ),
            None,
            None,
        )
    if args.method in RERANK_METHODS:
        model, api_base, api_keys = resolve_api_config(
            args,
            model_argument="rerank_model",
            default_model_env="RERANK_MODEL",
        )
        args.active_rerank_model = model
        args.active_api_pool_size = len(api_keys)
        args.active_api_base = api_base
        return (
            None,
            None,
            None,
            RerankCaller(
                model,
                api_base=api_base,
                api_keys=api_keys,
                timeout=args.api_timeout,
                max_retries=args.api_max_retries,
                max_chunks_per_doc=args.rerank_max_chunks_per_doc,
                overlap_tokens=args.rerank_overlap_tokens,
            ),
            None,
        )
    if args.method == "swerank":
        return None, None, None, None, SweRankOutputStore(args.swerank_output)
    if args.method == "agentless":
        return (
            None,
            None,
            None,
            None,
            AgentlessOutputStore(
                args.agentless_output,
                stage=args.agentless_stage,
            ),
        )
    if args.method == "locagent":
        return (
            None,
            None,
            None,
            None,
            LocAgentOutputStore(
                args.locagent_output,
                level=args.locagent_level,
            ),
        )
    use_structure = (
        ABLATION_VARIANTS[args.method].use_structure
        if args.method in ABLATION_METHODS
        else True
    )
    planner = build_planner(args, use_structure=use_structure)
    if args.method not in POLICY_METHODS:
        return planner, None, None, None, None
    if not args.checkpoint:
        raise ValueError("Ours 及其消融需要通过 --checkpoint 指定训练完成的模型")
    policy = EvidencePolicy(
        args.checkpoint,
        device=args.device,
        precision=args.precision,
        candidate_microbatch=args.candidate_microbatch,
    )
    return (
        planner,
        SerializedPolicy(policy) if args.concurrency > 1 else policy,
        None,
        None,
        None,
    )


def build_run_config(
    args: argparse.Namespace,
    *,
    external_path: Path | None,
) -> dict[str, Any]:
    """构造会影响逐任务实验结果的完整冻结配置。"""

    config: dict[str, Any] = {
        "method": args.method,
        "split": args.split,
        "code": code_identity(PROJECT_ROOT, ("src", "exp")),
        "tasks": artifact_identity(args.tasks),
        "runtime": artifact_identity(args.runtime),
        "api": {
            "profile": args.api_profile if args.method in API_METHODS else None,
            "base_url": getattr(args, "active_api_base", None),
            "planner_model": getattr(args, "active_model", None),
            "dense_model": getattr(args, "active_dense_model", None),
            "rerank_model": getattr(args, "active_rerank_model", None),
            "thinking_mode": "disabled",
            "max_retries": args.api_max_retries,
        },
        "prompt_versions": {
            "planner": PLANNER_PROMPT_VERSION,
            "policy_input": "1.0",
        },
        "parameters": {
            "evidence_token_budget": args.evidence_token_budget,
            "evidence_unit_budget": args.evidence_unit_budget,
            "retrieval_limit": args.retrieval_limit,
            "file_limit": args.file_limit,
            "fixed_steps": args.fixed_steps,
            "pair_limit": args.pair_limit,
            "planner_evidence_body_token_budget": (
                args.planner_evidence_body_token_budget
            ),
            "dense_batch_size": args.dense_batch_size,
            "hybrid_candidate_file_limit": args.hybrid_candidate_file_limit,
            "rrf_rank_constant": args.rrf_rank_constant,
            "rerank_candidate_file_limit": args.rerank_candidate_file_limit,
            "rerank_max_chunks_per_doc": args.rerank_max_chunks_per_doc,
            "rerank_overlap_tokens": args.rerank_overlap_tokens,
            "swerank_top_k": args.swerank_top_k,
            "agentless_stage": args.agentless_stage,
            "locagent_level": args.locagent_level,
        },
    }
    if args.checkpoint:
        config["checkpoint"] = artifact_identity(args.checkpoint)
    if external_path:
        config["external_output"] = artifact_identity(external_path)
    return config


def run_experiment(args: argparse.Namespace) -> None:
    """运行一个方法，并将每个任务的轨迹即时写入 JSONL。"""

    required_external_outputs = {
        "swerank": (args.swerank_output, "--swerank-output"),
        "agentless": (args.agentless_output, "--agentless-output"),
        "locagent": (args.locagent_output, "--locagent-output"),
    }
    external_path = None
    if args.method in EXTERNAL_METHODS:
        external_path, argument = required_external_outputs[args.method]
        if external_path is None:
            raise ValueError(f"{args.method} 需要通过 {argument} 指定官方输出")
        if args.split != "benchmark":
            raise ValueError(f"官方 {args.method} 对比只允许在 benchmark split 上运行")
    output = args.output or default_run_output(args)
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    run_name = output.stem
    reader = TaskReader(args.tasks)
    (
        planner,
        policy,
        dense_encoder,
        rerank_caller,
        external_outputs,
    ) = build_shared_resources(args)
    run_manifest = ensure_manifest(
        output,
        build_run_config(args, external_path=external_path),
    )
    run_config_hash = str(run_manifest["run_config_hash"])
    code_record = run_manifest["config"]["code"]
    completed = completed_task_ids(output)
    if args.method in EXTERNAL_METHODS:
        selected_tasks = [
            task
            for task in reader.iter_tasks(split=args.split, experiment_only=True)
            if external_outputs.contains(task)
        ]
        selected_ids = {str(task["task_id"]) for task in selected_tasks}
        completed &= selected_ids
        total = len(selected_tasks)
        tasks = (
            task
            for task in selected_tasks
            if str(task["task_id"]) not in completed
        )
    else:
        total = task_count(reader, args.split)
        tasks = (
            task
            for task in reader.iter_tasks(split=args.split, experiment_only=True)
            if str(task["task_id"]) not in completed
        )
    pending_total = max(0, total - len(completed))
    if pending_total == 0:
        print(f"运行完成：{args.method}，已存在 {total} 条成功结果，文件：{output}")
        return

    def run_task(task: Mapping[str, Any]) -> dict[str, Any]:
        with capture_api_usage() as api_usage:
            with RuntimeRepository(args.runtime) as repository:
                method = build_method(
                    args,
                    repository,
                    planner=planner,
                    policy=policy,
                    dense_encoder=dense_encoder,
                    rerank_caller=rerank_caller,
                    external_outputs=external_outputs,
                )
                result = method.run(task)
        if api_usage["api_calls"]:
            result.update(api_usage)
        return result

    succeeded = 0
    failed = 0
    with output.open("a", encoding="utf-8") as file:
        results = concurrent_results(
            tasks,
            total=pending_total,
            concurrency=args.concurrency,
            description=f"运行 {args.method}",
            worker=run_task,
        )
        for task, result, error in results:
            task_id = str(task["task_id"])
            if error is None:
                result.setdefault(
                    "planner_calls",
                    len(result.get("retrieval_rounds") or []),
                )
                append_jsonl(
                    file,
                    {
                        "status": "ok",
                        "run_name": run_name,
                        "method": args.method,
                        "split": args.split,
                        "run_config_hash": run_config_hash,
                        "git_commit": str(code_record["git_commit"]),
                        "code_sha256": str(code_record["code_sha256"]),
                        "api_profile": args.api_profile
                        if args.method in API_METHODS
                        else None,
                        "planner_model": getattr(args, "active_model", None),
                        "dense_model": getattr(args, "active_dense_model", None)
                        if args.method in DENSE_METHODS
                        else None,
                        "rerank_model": getattr(args, "active_rerank_model", None)
                        if args.method in RERANK_METHODS
                        else None,
                        "external_output": str(external_path.resolve())
                        if args.method in EXTERNAL_METHODS
                        else None,
                        "curve_id": str(args.swerank_output.resolve())
                        if args.method == "swerank"
                        else run_name,
                        "dense_cache": str(dense_cache_path(args))
                        if args.method in DENSE_METHODS
                        else None,
                        "api_pool_size": getattr(
                            args,
                            "active_api_pool_size",
                            0,
                        ),
                        "concurrency": args.concurrency,
                        "thinking_mode": "disabled"
                        if args.method in CHAT_API_METHODS
                        else None,
                        "checkpoint": str(args.checkpoint.resolve())
                        if args.checkpoint
                        else None,
                        "problem_statement": str(
                            task["input"]["problem_statement"]
                        ),
                        "online_issue": build_online_issue(task["input"]),
                        **result,
                    },
                )
                succeeded += 1
            else:
                append_jsonl(
                    file,
                    {
                        "status": "error",
                        "run_name": run_name,
                        "method": args.method,
                        "split": args.split,
                        "run_config_hash": run_config_hash,
                        "task_id": task_id,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                failed += 1

    print(
        f"运行结束：{args.method}，新增成功 {succeeded}，失败 {failed}，"
        f"已跳过 {len(completed)}，结果：{output}"
    )


def references_from_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """从离线监督行提取确定性评价和语义 Judge 所需字段。"""

    supervision = row["supervision"]
    return {
        "obligations": supervision.get("obligations") or [],
        "evidence_labels": supervision.get("evidence_labels") or [],
        "modified_files": supervision.get("modified_files") or [],
        "gold_patch": supervision.get("gold_patch") or "",
        "test_patch": supervision.get("test_patch") or "",
    }


def load_references(
    tasks_path: Path | str,
    task_ids: set[str],
    splits: set[str],
) -> dict[str, dict[str, Any]]:
    """仅在离线评价阶段读取目标任务的 Gold 与 obligations。"""

    reader = SupervisionReader(tasks_path)
    references: dict[str, dict[str, Any]] = {}
    for split in sorted(splits):
        for row in reader.iter_supervision(split=split, experiment_only=True):
            task_id = str(row["task_id"])
            if task_id in task_ids:
                references[task_id] = references_from_row(row)
    return references


def run_judge(args: argparse.Namespace) -> None:
    """对已经冻结的 Evidence Package 执行离线语义 Judge。"""

    input_path = Path(args.input).resolve()
    model, api_base, api_keys = resolve_api_config(
        args,
        model_argument="judge_model",
    )
    output = args.output or input_path.with_name(
        f"{input_path.stem}.{args.api_profile}-{file_label(model)}.judgments.jsonl"
    )
    output = Path(output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    args.active_api_base = api_base
    judge_manifest = ensure_manifest(
        output,
        {
            "mode": "semantic_judge",
            "code": code_identity(PROJECT_ROOT, ("src", "exp")),
            "input": artifact_identity(input_path),
            "tasks": artifact_identity(args.tasks),
            "api": {
                "profile": args.api_profile,
                "base_url": api_base,
                "model": model,
                "thinking_mode": "disabled",
                "max_retries": args.api_max_retries,
            },
            "prompt_version": SEMANTIC_JUDGE_PROMPT_VERSION,
        },
    )
    raw = latest_successful_records([input_path])
    completed = completed_task_ids(output)
    pending = [row for row in raw.values() if str(row["task_id"]) not in completed]
    references = load_references(
        args.tasks,
        {str(row["task_id"]) for row in pending},
        {str(row["split"]) for row in pending},
    )
    caller = OpenAICompatibleCaller(
        model=model,
        api_base=api_base,
        api_keys=api_keys,
        timeout=args.api_timeout,
        max_retries=args.api_max_retries,
        extra_body=API_PROFILES[args.api_profile]["non_thinking_body"],
    )

    def judge_task(row: Mapping[str, Any]) -> dict[str, Any]:
        task_id = str(row["task_id"])
        reference = references[task_id]
        with capture_api_usage() as api_usage:
            judgment = judge_evidence_package(
                caller,
                issue=str(row.get("online_issue") or row["problem_statement"]),
                evidence_package=row["evidence_package"],
                gold_patch=str(reference["gold_patch"]),
                test_patch=str(reference["test_patch"]),
            )
        judgment.update(api_usage)
        return judgment

    succeeded = 0
    failed = 0
    with output.open("a", encoding="utf-8") as file:
        results = concurrent_results(
            pending,
            total=len(pending),
            concurrency=args.concurrency,
            description="运行语义 Judge",
            worker=judge_task,
        )
        for row, judgment, error in results:
            task_id = str(row["task_id"])
            if error is None:
                append_jsonl(
                    file,
                    {
                        "status": "ok",
                        "run_name": str(row.get("run_name") or row["method"]),
                        "method": str(row["method"]),
                        "split": str(row["split"]),
                        "api_profile": args.api_profile,
                        "judge_model": model,
                        "api_pool_size": len(api_keys),
                        "concurrency": args.concurrency,
                        "thinking_mode": "disabled",
                        "run_config_hash": str(judge_manifest["run_config_hash"]),
                        "task_id": task_id,
                        "case_id": f"{row['method']}:{task_id}",
                        **judgment,
                    },
                )
                succeeded += 1
            else:
                append_jsonl(
                    file,
                    {
                        "status": "error",
                        "run_name": str(row.get("run_name") or row["method"]),
                        "method": str(row["method"]),
                        "split": str(row["split"]),
                        "run_config_hash": str(judge_manifest["run_config_hash"]),
                        "task_id": task_id,
                        "error": f"{type(error).__name__}: {error}",
                    },
                )
                failed += 1

    print(
        f"Judge 结束：新增成功 {succeeded}，失败 {failed}，"
        f"已跳过 {len(completed)}，结果：{output}"
    )


def trajectory_with_sufficiency(
    record: Mapping[str, Any],
    obligations: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """沿轨迹回放 Evidence IDs，并标记每一步之后是否充分。"""

    selected: set[str] = set()
    steps = []
    for step in record["steps"]:
        selected.update(map(str, step.get("added_evidence_ids") or []))
        annotated = dict(step)
        annotated["sufficient_after"] = bool(
            evaluate_sufficiency(selected, obligations)["sufficient"]
        )
        steps.append(annotated)
    return {**record, "steps": steps}


def task_metrics(
    record: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    """计算一个方法输出的确定性、定位、交互和成本指标。"""

    evidence_ids = list(map(str, record["final_evidence_ids"]))
    obligations = reference["obligations"]
    sufficiency = evaluate_sufficiency(evidence_ids, obligations)
    interactions = evaluate_interactions(evidence_ids, obligations)
    gold_ids = {
        str(item["evidence_id"])
        for item in reference["evidence_labels"]
        if item.get("relevance") == "positive"
    }
    localization = localization_metrics(
        record["evidence_package"],
        gold_evidence_ids=gold_ids,
        gold_files=set(map(str, reference["modified_files"])),
    )
    ranked_retrieved = []
    seen_retrieved: set[str] = set()
    for retrieval_round in record["retrieval_rounds"]:
        for evidence_id in map(
            str,
            retrieval_round.get("candidate_evidence_ids") or [],
        ):
            if evidence_id not in seen_retrieved:
                ranked_retrieved.append(evidence_id)
                seen_retrieved.add(evidence_id)
    retrieval = retrieval_metrics(
        ranked_retrieved,
        {
            str(item["evidence_id"]): float(item.get("confidence") or 1.0)
            for item in reference["evidence_labels"]
            if item.get("relevance") == "positive"
        },
    )
    metrics = {
        "run_name": str(record.get("run_name") or record["method"]),
        "method": str(record["method"]),
        "split": str(record["split"]),
        "sufficient": float(sufficiency["sufficient"]),
        "critical_requirement_coverage": float(
            sufficiency["critical_requirement_coverage"]
        ),
        "obligation_coverage": float(sufficiency["obligation_coverage"]),
        "witness_group_coverage": float(sufficiency["witness_group_coverage"]),
        "complementary_group_coverage": float(
            sufficiency["complementary_group_coverage"]
        ),
        "evidence_units": float(len(evidence_ids)),
        "evidence_tokens": float(record["final_evidence_tokens"]),
        "curve_id": str(record.get("curve_id") or record.get("run_name")),
        "rank_cutoff": float(record["rank_cutoff"])
        if record.get("rank_cutoff") is not None
        else None,
        **localization,
        **interactions,
        **{f"retrieval_{name}": value for name, value in retrieval.items()},
    }
    if bool(
        record.get(
            "trajectory_observed",
            record.get("execution_cost_observed", True),
        )
    ):
        metrics.update(
            {
                "steps": float(len(record["steps"])),
                "tool_calls": float(
                    sum(
                        int(step.get("tool_calls") or 0) for step in record["steps"]
                    )
                ),
                "planner_calls": float(record.get("planner_calls") or 0),
            }
        )
    for metric_name in (
        "api_prompt_tokens",
        "api_completion_tokens",
        "api_total_tokens",
        "api_calls",
        "external_agent_iterations",
        "external_tool_calls",
        "external_elapsed_seconds",
    ):
        if record.get(metric_name) is not None:
            metrics[metric_name] = float(record[metric_name])
    if record.get("external_mapping_rate") is not None:
        metrics.update(
            {
                "external_mapping_rate": float(record["external_mapping_rate"]),
                "external_output_count": float(record["external_output_count"]),
                "mapped_external_output_count": float(
                    record["mapped_external_output_count"]
                ),
                "snapshot_verification_rate": float(
                    bool(record.get("snapshot_verified"))
                ),
                "execution_cost_observation_rate": float(
                    bool(record.get("execution_cost_observed"))
                ),
            }
        )
    metrics.update(
        {
            f"{dimension}_coverage": float(score)
            for dimension, score in sufficiency["coverage_by_dimension"].items()
        }
    )
    trajectory = trajectory_with_sufficiency(record, obligations)
    return metrics, trajectory


def run_aggregate(args: argparse.Namespace) -> None:
    """汇总目录中的原始结果和可选语义 Judge 结果。"""

    if args.inputs:
        raw_paths = [Path(path).resolve() for path in args.inputs]
        input_dir = raw_paths[0].parent
        for path in raw_paths:
            validate_result_manifest(path)
    else:
        input_dir = Path(args.input_dir).resolve()
        raw_paths = sorted(
            path
            for path in input_dir.glob("*.jsonl")
            if not path.name.endswith(".judgments.jsonl")
        )
    if not raw_paths:
        raise ValueError("没有指定可汇总的实验结果 JSONL")
    raw = latest_successful_records(raw_paths)
    if not raw:
        raise ValueError("指定的实验结果中没有成功任务")
    run_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in raw.values():
        run_groups[
            (
                str(row.get("run_name") or row["method"]),
                str(row["method"]),
                str(row["split"]),
            )
        ].append(row)
    common_task_ids: dict[str, set[str]] = {}
    for split in {key[2] for key in run_groups}:
        task_sets = [
            {str(row["task_id"]) for row in rows}
            for key, rows in run_groups.items()
            if key[2] == split
        ]
        common_task_ids[split] = set.intersection(*task_sets)
    if args.expected_task_count is not None:
        mismatches = {
            split: len(task_ids)
            for split, task_ids in common_task_ids.items()
            if len(task_ids) != args.expected_task_count
        }
        if mismatches:
            raise ValueError(
                f"共同任务数量与 --expected-task-count 不一致：{mismatches}"
            )
    comparable_rows = [
        row
        for row in raw.values()
        if str(row["task_id"]) in common_task_ids[str(row["split"])]
    ]
    external_rows = [
        row for row in comparable_rows if str(row["method"]) in EXTERNAL_METHODS
    ]
    if args.require_external_snapshot_verified:
        unverified = [
            str(row["task_id"])
            for row in external_rows
            if not bool(row.get("snapshot_verified"))
        ]
        if unverified:
            raise ValueError(
                f"外部方法存在 {len(unverified)} 个未验证快照任务，拒绝生成主表"
            )
    if args.min_external_mapping_rate is not None:
        low_mapping = [
            str(row["task_id"])
            for row in external_rows
            if float(row.get("external_mapping_rate") or 0.0)
            < args.min_external_mapping_rate
        ]
        if low_mapping:
            raise ValueError(
                f"外部方法存在 {len(low_mapping)} 个任务的映射率低于 "
                f"{args.min_external_mapping_rate:.3f}"
            )
    task_ids = {str(row["task_id"]) for row in comparable_rows}
    splits = {str(row["split"]) for row in raw.values()}
    references = load_references(args.tasks, task_ids, splits)

    metric_rows = []
    trajectory_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in comparable_rows:
        metrics, trajectory = task_metrics(row, references[str(row["task_id"])])
        metric_rows.append(metrics)
        if bool(
            row.get(
                "trajectory_observed",
                row.get("execution_cost_observed", True),
            )
        ):
            trajectory_groups[
                (
                    str(row.get("run_name") or row["method"]),
                    str(row["method"]),
                    str(row["split"]),
                )
            ].append(trajectory)

    summaries = {
        (str(row["run_name"]), str(row["method"]), str(row["split"])): row
        for row in aggregate_rows(
            metric_rows,
            group_by=("run_name", "method", "split"),
        )
    }
    curve_ids = {
        (
            str(row["run_name"]),
            str(row["method"]),
            str(row["split"]),
        ): str(row["curve_id"])
        for row in metric_rows
    }
    for key, summary in summaries.items():
        summary["curve_id"] = curve_ids[key]
        summary["mean_evidence_units"] = float(summary["evidence_units"])
        summary["mean_evidence_tokens"] = float(summary["evidence_tokens"])
        if "steps" in summary:
            summary["mean_acquisition_steps"] = float(summary["steps"])
        if "tool_calls" in summary:
            summary["mean_tool_calls"] = float(summary["tool_calls"])
        if "planner_calls" in summary:
            summary["mean_planner_calls"] = float(summary["planner_calls"])
        for metric_name in (
            "api_prompt_tokens",
            "api_completion_tokens",
            "api_total_tokens",
            "api_calls",
            "external_agent_iterations",
            "external_tool_calls",
            "external_elapsed_seconds",
        ):
            if metric_name in summary:
                summary[f"mean_{metric_name}"] = float(summary[metric_name])
    for key, trajectories in trajectory_groups.items():
        summaries[key].update(evaluate_trajectories(trajectories))
    for key, summary in summaries.items():
        summary["available_task_count"] = len(run_groups[key])
        summary["common_task_count"] = len(common_task_ids[key[2]])

    curve_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries.values():
        curve_groups[
            (
                str(summary["method"]),
                str(summary["split"]),
                str(summary["curve_id"]),
            )
        ].append(summary)
    for curve in curve_groups.values():
        points = sorted(
            (
                float(summary["evidence_tokens"]),
                float(summary["sufficient"]),
            )
            for summary in curve
        )
        if len(points) < 2:
            continue
        auc = auc_sufficiency_cost(points)
        curve_points = [
            {"mean_evidence_tokens": cost, "sufficiency_rate": score}
            for cost, score in points
        ]
        for summary in curve:
            summary["sufficiency_token_auc"] = auc
            summary["sufficiency_token_curve"] = curve_points

    judgment_paths = (
        [Path(path).resolve() for path in args.judgments]
        if args.judgments
        else sorted(input_dir.glob("*.judgments.jsonl"))
        if not args.inputs
        else []
    )
    if args.judgments:
        for path in judgment_paths:
            validate_result_manifest(path)
    judgments = latest_successful_records(judgment_paths)
    judgment_groups: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in judgments.values():
        if str(row["task_id"]) not in common_task_ids.get(str(row["split"]), set()):
            continue
        judgment_groups[
            (
                str(row.get("run_name") or row["method"]),
                str(row["method"]),
                str(row["split"]),
            )
        ].append(row)
    for key, rows in judgment_groups.items():
        if key in summaries:
            summaries[key].update(aggregate_semantic_judgments(rows))
    semantic_score = "reference_grounded_semantic_sufficiency_rate"
    for curve in curve_groups.values():
        if len(curve) < 2 or any(semantic_score not in summary for summary in curve):
            continue
        points = sorted(
            (
                float(summary["evidence_tokens"]),
                float(summary[semantic_score]),
            )
            for summary in curve
        )
        auc = auc_sufficiency_cost(points)
        curve_points = [
            {
                "mean_evidence_tokens": cost,
                "semantic_sufficiency_rate": score,
            }
            for cost, score in points
        ]
        for summary in curve:
            summary["semantic_sufficiency_token_auc"] = auc
            summary["semantic_sufficiency_token_curve"] = curve_points

    output = Path(args.output or input_dir / "summary.json").resolve()
    output.write_text(
        json.dumps(
            [summaries[key] for key in sorted(summaries)],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(
        f"汇总完成：读取 {len(raw)} 个任务结果，"
        f"按共同任务交集汇总 {len(summaries)} 个运行，文件：{output}"
    )


def add_api_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--api-profile",
        choices=tuple(API_PROFILES),
        default="openai",
        help="选择 .env 中的一组 OpenAI-compatible API 配置",
    )
    parser.add_argument(
        "--api-base",
        help="覆盖 Profile 对应的 API Base URL",
    )
    parser.add_argument(
        "--api-key-env",
        action="append",
        help="覆盖 Profile 对应的 API Key 环境变量名",
    )
    parser.add_argument("--api-timeout", type=float, default=300.0)
    parser.add_argument(
        "--api-max-retries",
        type=int,
        default=3,
        help="429、5xx 和网络错误的最大重试次数",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evidence Agent 统一实验入口")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="运行 Ours、Baseline 或消融")
    run.add_argument("--method", choices=RUN_METHODS, required=True)
    run.add_argument("--split", default="validation")
    run.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    run.add_argument("--runtime", type=Path, default=DEFAULT_RUNTIME)
    run.add_argument("--output", type=Path)
    run.add_argument("--planner-model", help="覆盖 Profile 对应的 Planner 模型名")
    run.add_argument("--checkpoint", type=Path)
    run.add_argument("--device", default="auto")
    run.add_argument("--precision", default="auto")
    run.add_argument("--candidate-microbatch", type=int, default=1)
    run.add_argument("--evidence-token-budget", type=int, default=32_768)
    run.add_argument("--evidence-unit-budget", type=int, default=64)
    run.add_argument("--retrieval-limit", type=int, default=64)
    run.add_argument("--file-limit", type=int, default=32)
    run.add_argument("--fixed-steps", type=int, default=10)
    run.add_argument("--pair-limit", type=int, default=8)
    run.add_argument(
        "--planner-evidence-body-token-budget",
        type=int,
        default=8192,
        help="Planner 可见的当前 Evidence 正文预算；元数据始终全量保留",
    )
    run.add_argument(
        "--dense-model",
        help="覆盖 .env 中的 EMBEDDING_MODEL",
    )
    run.add_argument("--dense-batch-size", type=int, default=16)
    run.add_argument("--dense-cache", type=Path)
    run.add_argument(
        "--rerank-model",
        help="覆盖 .env 中的 RERANK_MODEL",
    )
    run.add_argument("--hybrid-candidate-file-limit", type=int, default=128)
    run.add_argument("--rrf-rank-constant", type=int, default=60)
    run.add_argument("--rerank-candidate-file-limit", type=int, default=64)
    run.add_argument("--rerank-max-chunks-per-doc", type=int, default=8)
    run.add_argument("--rerank-overlap-tokens", type=int, default=64)
    run.add_argument("--swerank-output", type=Path)
    run.add_argument(
        "--swerank-top-k",
        type=int,
        choices=(1, 3, 5, 10, 20, 100),
        default=100,
        help="SweRank 官方函数排名截断点",
    )
    run.add_argument("--agentless-output", type=Path)
    run.add_argument(
        "--agentless-stage",
        choices=("file", "related", "edit"),
        default="edit",
        help="使用 Agentless 官方文件、关联位置或编辑位置输出",
    )
    run.add_argument("--locagent-output", type=Path)
    run.add_argument(
        "--locagent-level",
        choices=("file", "module", "function"),
        default="function",
        help="使用 LocAgent 官方文件、模块或函数层定位输出",
    )
    run.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="同时运行的任务数；API 方法会在当前 Profile 的 Key 之间轮转",
    )
    add_api_arguments(run)
    run.set_defaults(handler=run_experiment)

    judge = commands.add_parser("judge", help="运行离线语义 Judge")
    judge.add_argument("--input", type=Path, required=True)
    judge.add_argument("--output", type=Path)
    judge.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    judge.add_argument("--judge-model", help="覆盖 Profile 对应的 Judge 模型名")
    judge.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="同时执行的语义评审任务数；API Key 在请求之间轮转",
    )
    add_api_arguments(judge)
    judge.set_defaults(handler=run_judge)

    aggregate = commands.add_parser("aggregate", help="汇总实验指标")
    aggregate_source = aggregate.add_mutually_exclusive_group(required=True)
    aggregate_source.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        help="正式实验显式指定需要比较的原始结果 JSONL",
    )
    aggregate_source.add_argument(
        "--input-dir",
        type=Path,
        help="兼容模式：扫描目录中的全部原始结果 JSONL",
    )
    aggregate.add_argument(
        "--judgments",
        type=Path,
        nargs="*",
        help="显式指定与 --inputs 对应的语义 Judge JSONL",
    )
    aggregate.add_argument(
        "--expected-task-count",
        type=int,
        help="要求每个 split 的共同任务交集严格等于该数量",
    )
    aggregate.add_argument(
        "--require-external-snapshot-verified",
        action="store_true",
        help="拒绝包含未核对 base_commit 的外部方法结果",
    )
    aggregate.add_argument(
        "--min-external-mapping-rate",
        type=float,
        help="拒绝任何单任务映射率低于该阈值的外部方法结果",
    )
    aggregate.add_argument("--output", type=Path)
    aggregate.add_argument("--tasks", type=Path, default=DEFAULT_TASKS)
    aggregate.set_defaults(handler=run_aggregate)
    return parser


def main() -> int:
    load_dotenv(PROJECT_ROOT / ".env", override=False)
    args = build_parser().parse_args()
    if hasattr(args, "api_max_retries") and args.api_max_retries < 0:
        raise ValueError("--api-max-retries 不能小于 0")
    if (
        hasattr(args, "planner_evidence_body_token_budget")
        and args.planner_evidence_body_token_budget < 0
    ):
        raise ValueError("--planner-evidence-body-token-budget 不能小于 0")
    if (
        getattr(args, "min_external_mapping_rate", None) is not None
        and not 0.0 <= args.min_external_mapping_rate <= 1.0
    ):
        raise ValueError("--min-external-mapping-rate 必须位于 [0, 1]")
    if (
        getattr(args, "expected_task_count", None) is not None
        and args.expected_task_count <= 0
    ):
        raise ValueError("--expected-task-count 必须大于 0")
    args.handler(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

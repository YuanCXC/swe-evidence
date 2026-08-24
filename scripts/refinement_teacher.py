# -*- coding: utf-8 -*-
"""
Supervision Refinement Teacher v1.8.2
（DeepSeek 监督修正教师）

建议位置：
    scripts/refinement_teacher.py

本文件属于 Data Processing（数据处理）。

它负责：
    Issue / Question（问题描述）
    + Original Supervision（原始监督）
    + Candidate Evidence Pool（候选证据池）
    + Optional Gold Reference（可选离线 Gold 参考）
        -> DeepSeek
        -> Structured Refinement Proposal（结构化修正建议）

它不负责：
    - 最终 Certificate（证书）计算；
    - Evidence ID 真实性校验；
    - 修改 Parquet；
    - 修改原 supervision；
    - Semantic Evaluation（语义评价）。

最终是否合法，由 scripts/refinement_core.py 决定。

为什么不能复用 src/evaluation/judge.py：
    Semantic Judge（语义评审）是在评价答案；
    Refinement Teacher（修正教师）是在产生/修正答案。

两者必须职责隔离，避免“自己出答案、自己判答案”。
"""

from __future__ import annotations

import hashlib
import json
import os
import queue
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEFAULT_REFINEMENT_MODEL = "deepseek-v4-flash"
DEFAULT_REASONING_EFFORT = "high"
DEFAULT_THINKING_TYPE = "disabled"

# ---------------------------------------------------------------------
# SiliconFlow（硅基流动）主 Teacher 配置。
#
# .env 中：
#   OPENAI_BASE_URL=https://api.siliconflow.cn/v1
#   OPENAI_API_KEY=
#   OPENAI_API_KEY_2=
#   OPENAI_API_KEY_3=
#   OPENAI_API_KEY_4=
#   LLM_MODEL=Qwen/Qwen3-8B
#
# 每个 API Key 最大并发 4。
# 4 个 Key 全部存在时，总并发上限 = 16。
# ---------------------------------------------------------------------

SILICONFLOW_API_KEY_ENV_NAMES = (
    "OPENAI_API_KEY",
    "OPENAI_API_KEY_2",
    "OPENAI_API_KEY_3",
    "OPENAI_API_KEY_4",
)

DEFAULT_SILICONFLOW_BASE_URL = (
    "https://api.siliconflow.cn/v1"
)

DEFAULT_SILICONFLOW_MODEL = (
    "Qwen/Qwen3-8B"
)

DEFAULT_SILICONFLOW_PER_KEY_CONCURRENCY = 4


# ---------------------------------------------------------------------
# BigModel / 智谱 GLM Primary Teacher（主教师）配置。
#
# 项目根目录 .env：
#
#   BIGMOD_API_KEY=
#   BIGMOD_API_URL=https://open.bigmodel.cn/api/paas/v4/
#   BIGMOD_API_MODEL=GLM-4.7-Flash
#
# 官方模型代码使用小写 glm-4.7-flash。
# runner 会对 BIGMOD_API_MODEL 做 strip().lower()，
# 因此用户写 GLM-4.7-Flash 也可以。
#
# 当前只做 Primary Teacher A/B：
#   Qwen3-8B vs GLM-4.7-Flash
#
# DeepSeek Strong Reviewer（强审）保持完全相同。
# ---------------------------------------------------------------------

DEFAULT_BIGMODEL_BASE_URL = (
    "https://open.bigmodel.cn/api/paas/v4/"
)

DEFAULT_BIGMODEL_MODEL = (
    "glm-4.7-flash"
)

DEFAULT_BIGMODEL_CONCURRENCY = 1

BIGMODEL_API_KEY_ENV = "BIGMOD_API_KEY"


REFINEMENT_SYSTEM_PROMPT = r"""你是 Fixed Decision Teacher（固定决策教师）。

你的任务不是生成完整监督答案，而是对 7 个固定语义槽位做有限决策。
程序负责 Evidence ID（证据ID）、Obligation ID（证据要求ID）、assessment（总体判断）、STOP（停止判断）和 Certificate（证书）。

必须恰好输出 7 个 slot：
1. fault_location（故障位置/新增功能的现有集成位置）
2. fault_logic（修复前故障机制）
3. dependency_context（依赖/调用/配置上下文）
4. state_flow（状态生产、传播、缓存、失效、更新）
5. behavior_constraint（问题描述要求的预期行为）
6. repair_scope（修复影响范围、导出点、注册点）
7. validation_constraint（验证、兼容、边界约束）

每个 slot 的 decision 只能是：
- keep_original：只有 [ORIGINAL SLOT AVAILABILITY] 明确 present=true 时才允许；表示原 slot 的必要性与 Witness 结构保持不变。
- use_candidates：该 slot 必要，且必须由 repository Evidence 支撑。
- question_satisfied：该 slot 必要，但 Issue/Question 已经明确给出，不需要重复检索。
- not_required：达到修复上下文充分性不需要该 slot。
- missing_pre_fix_context：真正必要、且修复前本来应存在的上下文不在 Candidate Pool。
- uncertain：无法可靠判断。

Candidate Number（候选序号）是唯一 Evidence 选择接口。
Candidate header 例如：
[CANDIDATE 7] id=ev_xxx | path=pkg/a.py | symbol=foo | ...
你只能输出数字 7。不要输出 Evidence ID、path、symbol。

OR-of-AND（组间或、组内与）：
[[2,5]] = Candidate 2 AND Candidate 5
[[2],[5,9]] = Candidate 2 OR (Candidate 5 AND Candidate 9)
只有真正的替代方案才拆成多个 group。

Original Supervision 是待审假设，不是默认正确，也不是默认错误。
先检查 [ORIGINAL SLOT AVAILABILITY]：
- present=true 才允许 keep_original；
- present=false 时禁止 keep_original。

如果 present=true 且原 slot 已经足够正确，使用 keep_original。
不要因为另一个 Candidate 更详细、描述更漂亮、上下文更多就改变 slot。

程序会把 keep_original 展开为 Original Slot 的真实有效结构，因此 keep_original 与重新选择完全相同 Original Witness 的 use_candidates 可以在 Slot Consensus（槽位共识）中视为等价。

Question-satisfied（问题已满足）：
Issue 已明确 expected behavior、validation rule、compatibility rule、明确输入输出合同时，优先考虑 question_satisfied。
但实现位置、状态来源、依赖调用关系不能仅因 Issue 提到 symbol 就视为 question_satisfied。

Feature Addition（新增功能）是高风险规则：
如果 Issue 明确说 Add function X / Introduce API X / Implement X / The function added in this PR (X)，则 X 默认是 Future/Post-fix Symbol（未来/修复后符号）。
pre-fix 中不存在 X 是正常现象，绝不能因此判 missing_pre_fix_context，也不能请求未来 X 的函数正文或未来测试。
新增功能应审查已有 pre-fix 的 analogous implementation（类似实现）、integration point（集成位置）、import/export convention（导入导出约定）、registration convention（注册约定）、dependency/state context（依赖/状态上下文）。
如果 Issue 已明确未来功能行为，behavior_constraint 应优先 question_satisfied。

每种 decision 的字段合同：
- keep_original / question_satisfied / not_required / uncertain：witness_groups=[]，missing_context=null。
- use_candidates：witness_groups 必须非空，missing_context=null。
- missing_pre_fix_context：witness_groups=[]，missing_context 必须是 {path_hint, symbol_hint, keywords, reason}。

禁止输出：
assessment、stop_assessment、Evidence ID、source_obligation_id、obligation_id、Certificate、Certificate Candidate Number。

只输出一个 JSON object，不要 Markdown，不要代码块，不要额外字段：
{
  "confidence": 0.0,
  "slots": {
    "fault_location": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."},
    "fault_logic": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."},
    "dependency_context": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."},
    "state_flow": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."},
    "behavior_constraint": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."},
    "repair_scope": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."},
    "validation_constraint": {"decision":"...","witness_groups":[],"missing_context":null,"reason":"..."}
  }
}

use_candidates 示例：
{"decision":"use_candidates","witness_groups":[[2,5],[9]],"missing_context":null,"reason":"2+5 共同足够，或 9 单独足够。"}

missing_pre_fix_context 示例：
{"decision":"missing_pre_fix_context","witness_groups":[],"missing_context":{"path_hint":"修复前真实存在的源码路径线索","symbol_hint":"修复前真实存在的 symbol 线索","keywords":["state","producer"],"reason":"为什么当前候选缺少真正必要的修复前上下文"},"reason":"当前 Candidate Pool 缺少必要 pre-fix context。"}

输出前逐 slot 检查：
1. 原 slot 正确 -> keep_original。
2. Issue 本身已满足必要语义 -> question_satisfied。
3. repository Evidence 必须支撑 -> use_candidates。
4. 不属于充分性必要条件 -> not_required。
5. 真正必要的 pre-fix 上下文缺失 -> missing_pre_fix_context。
6. 无法可靠判断 -> uncertain。
最后确认恰好 7 个 slot，且没有 Evidence ID、assessment、STOP、Certificate。""".strip()


# ============================================================================
# v1.9.2 Two-Stage Teacher Prompts（两阶段教师提示）
# ============================================================================

REQUIREMENT_DECISION_SYSTEM_PROMPT = r"""你是 Requirement Decision Teacher（证据需求决策教师）。

你当前只做 Stage 1：
判断“下游修复模型为了达到修复上下文充分性，
每个固定语义槽位究竟需要哪一类信息”。

================================================================
一、Stage 1 与 Candidate Pool 完全解耦
================================================================

当前阶段故意看不到 Candidate Evidence Pool（候选证据池）。

原因：
“某个语义是否需要仓库证据”
与
“当前候选池里有没有这类证据”
是两个不同问题。

Stage 1 只判断必要性。
Candidate 可用性全部交给 Stage 2。

因此你绝对不能：
- 选择 Candidate Number；
- 根据“候选里刚好有某段代码”就提高 repository_required；
- 判断 Candidate Pool 是否 insufficient；
- 输出 Evidence ID。

================================================================
二、固定 7 个 slot
================================================================

必须恰好判断：

1. fault_location
2. fault_logic
3. dependency_context
4. state_flow
5. behavior_constraint
6. repair_scope
7. validation_constraint

================================================================
三、decision 只能是 4 种
================================================================

1. repository_required

该语义对修复上下文充分性是必要的，
而且仅凭 Issue / Question 不能知道，
必须查看 pre-fix repository（修复前仓库）才能可靠获得。

2. question_satisfied

该语义是必要的，
但 Issue / Question 已经明确给出，
不应该要求 Retriever（检索器）重复检索同一个事实。

3. not_required

该语义不是达到当前任务“修复上下文充分性”的必要条件。

“可能有帮助”不等于 required。

4. uncertain

无法可靠判断。

宁可 uncertain，也不要猜。

================================================================
四、behavior_constraint / validation_constraint 的硬规则
================================================================

这是当前协议的重点。

如果 Issue 已经明确写出以下任一内容：

- Expected behavior
- Describe the solution you'd like
- 应返回什么类型 / 值
- 应支持什么输入
- 不应抛什么异常
- 默认行为是什么
- 兼容性要求是什么
- 明确的复现条件和期望结果
- 明确的 validation / boundary condition

则相应的：

    behavior_constraint
或
    validation_constraint

优先判断为：

    question_satisfied

只有当“真正的行为约束本身”必须从代码中推断、
Issue 并没有说清楚时，
才可以 repository_required。

不要因为“实现这个行为还需要看代码”，
就把 behavior_constraint 本身判为 repository_required。

实现代码属于：
    fault_location
    fault_logic
    state_flow
    dependency_context
    repair_scope

行为事实本身如果 Issue 已经说清楚，
仍然是 question_satisfied。

================================================================
五、state_flow 的边界
================================================================

只有以下信息对原因分析 / 补丁规划确实必要时，
state_flow 才 repository_required：

- cache identity / invalidation
- mutable state producer / consumer
- data lifecycle
- binding / propagation
- state update ordering
- value 在多个阶段之间的传播

以下情况不要机械判 state_flow：

- 只是参数顺序变化
- 只是新增 CLI option
- 只是 import/export
- 只是函数签名扩展
- 只是文案 / description 格式
- 没有实际状态传播问题的 feature addition

================================================================
六、dependency_context 的边界
================================================================

只有调用链、依赖组件、配置来源或跨模块依赖
对理解故障或规划修复不可缺少时才 repository_required。

不要因为存在“另一个相关函数”就自动要求 dependency_context。

================================================================
七、repair_scope 的边界
================================================================

repair_scope 关注：
为了形成有根据的补丁规划，
需要知道哪些现有模块 / API / integration point 会受影响。

如果 Issue 已经明确指出唯一修改位置，
且没有跨模块影响，
repair_scope 可以 question_satisfied。

如果修改范围仍需从仓库结构推断，
才 repository_required。

================================================================
八、Feature Addition（新增功能）
================================================================

如果 Issue 明确：

    Add function X
    Introduce API X
    Implement X

则未来 X 在 pre-fix 中不存在是正常的。

Stage 1 不判断 Candidate 缺失，
因此绝不能因为未来 X 不存在而输出 uncertain。

只判断：
为了规划新增功能，是否需要查看已有 pre-fix：
- analogous implementation（类似实现）
- integration point（集成位置）
- export / registration convention
- dependency / state convention

================================================================
九、禁止输出
================================================================

禁止：

- Candidate Number
- Evidence ID
- witness_groups
- missing_context
- assessment
- stop_assessment
- Certificate
- source_obligation_id

================================================================
十、输出 Schema
================================================================

只输出 JSON object：

{
  "confidence": 0.0,
  "slots": {
    "fault_location": {
      "decision": "repository_required | question_satisfied | not_required | uncertain",
      "reason": "简洁说明"
    },
    "fault_logic": {
      "decision": "...",
      "reason": "..."
    },
    "dependency_context": {
      "decision": "...",
      "reason": "..."
    },
    "state_flow": {
      "decision": "...",
      "reason": "..."
    },
    "behavior_constraint": {
      "decision": "...",
      "reason": "..."
    },
    "repair_scope": {
      "decision": "...",
      "reason": "..."
    },
    "validation_constraint": {
      "decision": "...",
      "reason": "..."
    }
  }
}

最后检查：
- 是否恰好 7 个 slot；
- behavior / validation 已由 Issue 明确时，是否正确 question_satisfied；
- 是否把“实现需要代码”误写成“行为事实需要代码”；
- 是否错误输出 Candidate / Evidence / witness_groups。""".strip()


WITNESS_SELECTION_SYSTEM_PROMPT = r"""
你是 Targeted Witness Selector（定向支撑证据选择器）。

你当前只做 Stage 2：
对“一个已经确定 repository_required 的语义槽位”
从当前 pre-fix Candidate Pool 中选择最小、充分的 Witness（支撑证据）。

你不再判断这个 slot 是否 required。
Stage 1 已经决定：
    repository_required

你只判断：
    当前 Candidate Pool 中哪些 Candidate 能支撑该 slot。

status 只能是：

1. select
   当前候选池存在充分的 pre-fix Witness。

2. insufficient
   该 slot 需要修复前仓库证据，
   但当前 Candidate Pool 缺少真正必要、修复前本来应该存在的上下文。

3. uncertain
   无法可靠选择 Witness。

Candidate Number 是唯一 Evidence 选择接口。
禁止输出 Evidence ID / path / symbol。

OR-of-AND（组间或、组内与）：
    [[2, 5]]
        = Candidate 2 AND Candidate 5

    [[2], [5, 9]]
        = Candidate 2 OR (Candidate 5 AND Candidate 9)

同一 AND-group 内必须缺一不可。
不同 group 只有在每组单独都足够时才构成 OR 替代。

Minimality（最小性）：
只选择真正用于支撑当前 TARGET SLOT 的 Evidence。
不要因为“有帮助”就把额外上下文塞入 Witness。

Feature Addition（新增功能）：
未来新增函数 / API / 测试在 pre-fix 中不存在是正常的。
绝不能把“未来 X 不存在”作为 insufficient 的理由。
只能选择或请求修复前真实存在的：
- analogous implementation
- integration / registration / export point
- dependency / state context

status=select：
    witness_groups 必须非空。

status=insufficient / uncertain：
    witness_groups 必须为空。

只输出 JSON object：

{
  "confidence": 0.0,
  "status": "select | insufficient | uncertain",
  "witness_groups": [[1], [2, 5]],
  "reason": "简洁说明"
}

不要输出任何其它字段。
""".strip()


STRONG_REVIEW_SYSTEM_PROMPT = r"""
你是 Strong Supervision Reviewer（强监督复核器）。

你不是第二个 Teacher。
你不重新生成新的 supervision proposal。

你的任务是：
审核 Primary Teacher 已生成并经过 Deterministic Verifier
机械规范化后的 Primary Canonical Proposal（主教师规范提案）。

审核依据：
- Issue / Question；
- Original Obligations；
- Candidate Evidence Pool；
- Primary Canonical Proposal；
- Primary Verifier Summary。

你的目标不是“和 Primary 一致”，而是判断它是否真的语义可靠。

============================================================
一、审核原则
============================================================

1. 不因为措辞不同而 reject。
2. 不因为“还可以加更多上下文”就 reject。
3. 不得为了配合 Primary 自动 approve。
4. 无法可靠判断时使用 uncertain。
5. 不能把 Deterministic Verifier 的结构合法，当成语义正确。

6. Primary Canonical Proposal 是“Primary 实际输出了什么”的唯一事实来源。

   当你批评：
       satisfied_by_question
       witness_groups
       obligation type
       certificate

   时，必须先读取 Primary Canonical Proposal 中该字段当前真实值。

   禁止：
       Primary 实际 satisfied_by_question=true，
       但你的 issue reason 声称它是 false。

   也禁止根据 rationale（解释文字）
   猜测结构字段当前值。

7. Feature Addition（新增功能）审核时，
   不得凭空假设“未来新增函数必须位于某个新文件”。

   pre-fix 中允许使用：
       analogous implementation（已有类似实现）
       integration point（集成位置）
       public API export（公开接口导出）
       existing convention（已有约定）

   只有 Candidate 正文本身不能支撑所声称的 analogous/integration 语义，
   才应因为 Witness 语义错误而 reject。

============================================================
二、六项强制检查
============================================================

A. obligation_set_correct
   （证据要求集合正确）

检查：
- 是否保留真正必要的 Original Obligation；
- 是否新增真正必要的 obligation；
- 是否存在非必要 obligation；
- 是否遗漏影响故障定位、机制理解、依赖状态或修复范围的必要语义。

------------------------------------------------------------

B. question_satisfied_correct
   （问题描述已满足判断正确）

重点检查：

如果 Issue / Question 已明确给出：
- expected behavior（预期行为）；
- 输入输出要求；
- validation constraint（验证约束）；
- 明确禁止行为；

对应必要 obligation 如果存在，通常应：

    satisfied_by_question = true
    witness_groups = []

不要强迫 Retriever 再检索重复仓库证据。

反过来：
Issue 没明确给出，就不能错误标为 true。

还要检查：
Primary 是否完全遗漏了一个“必要但已由 Question 满足”的 obligation。

------------------------------------------------------------

C. witness_semantics_correct
   （支撑证据语义正确）

检查 retrieval_required obligation 的 Witness：

- Evidence 是否真的支持该 obligation；
- 是否只是同文件相关但并不能证明；
- 是否把 test/doc/邻近函数当成强支撑；
- 是否把弱相关 Evidence 当充分 Evidence；
- Feature Addition 是否用了真正有效的 analogous / integration Evidence。

------------------------------------------------------------

D. and_or_correct
   （AND/OR 结构正确）

一个 witness_group 内：
    A + B = A AND B

多个 witness_group：
    group1 OR group2

检查是否存在：
- 自然语言说 A+B 共同必需，结构却写 [A] OR [B]；
- A 或 B 单独足够，结构却写 [A,B]。

------------------------------------------------------------

E. certificate_consistent
   （证书语义一致）

Deterministic Verifier 已从 Primary witness graph
计算 verified minimal certificate。

你不检查 Teacher 是否“抄对数组”，而检查：

这个 witness graph 在语义上是否真的形成
Minimal Sufficient Evidence（最小充分证据）。

如果图本身错，即使 certificate 数学上自洽，也必须 reject。

------------------------------------------------------------

F. feature_addition_handled_correctly
   （新增功能处理正确）

对于 Feature Addition：

pre-fix 中不存在未来新函数是正常的。

应检查 Primary 是否使用真正有价值的：
- analogous implementation；
- integration point；
- caller / registration；
- API export；
- existing convention；
- dependency/state context。

如果 Primary 把无关函数误当 analogous implementation，必须 reject。

============================================================
三、Review Decision
============================================================

review_decision 只能是：

    approve
    reject
    uncertain

approve：
    六项 checks 必须全部 true；
    issues 必须为 []。

reject：
    至少一项 checks=false；
    issues 必须非空；
    不要输出新的 supervision proposal。

uncertain：
    issues 必须非空；
    说明为什么无法可靠判断。

============================================================
四、输出 Schema
============================================================

只输出一个 JSON object。
不要 Markdown。
不要代码块。
不要额外字段。

{
  "review_decision": "approve | reject | uncertain",
  "confidence": 0.0,
  "summary": "简洁说明总体判断",
  "checks": {
    "obligation_set_correct": true,
    "question_satisfied_correct": true,
    "witness_semantics_correct": true,
    "and_or_correct": true,
    "certificate_consistent": true,
    "feature_addition_handled_correctly": true
  },
  "issues": [
    {
      "code": "简短机器可读代码",
      "obligation_type": "fault_location | fault_logic | dependency_context | state_flow | behavior_constraint | repair_scope | validation_constraint；若不对应具体 obligation，请输出真正 JSON null（不是字符串 \"null\"）",
      "reason": "具体、可审计的语义问题"
    }
  ]
}
""".strip()


STRONG_REVIEW_CHECK_KEYS = (
    "obligation_set_correct",
    "question_satisfied_correct",
    "witness_semantics_correct",
    "and_or_correct",
    "certificate_consistent",
    "feature_addition_handled_correctly",
)


def parse_strong_review_decision(
    text: str,
) -> dict[str, Any]:
    """
    解析 Strong Reviewer（强审）输出。

    approve 只有在六项检查全部 true 且 issues=[] 时才合法。
    reject 至少需要一个 false check 和一个 issue。
    uncertain 必须给出 issue。

    reject / uncertain 都不会被程序自动改写成训练监督。
    """

    if not isinstance(text, str) or not text.strip():
        raise ValueError("Strong Reviewer 输出为空")

    try:
        payload = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(
            "Strong Reviewer 输出不是合法 JSON object"
        ) from exc

    if not isinstance(payload, dict):
        raise ValueError(
            "Strong Reviewer 顶层必须是 JSON object"
        )

    allowed_top_keys = {
        "review_decision",
        "confidence",
        "summary",
        "checks",
        "issues",
    }

    extra_keys = set(payload) - allowed_top_keys
    if extra_keys:
        raise ValueError(
            "Strong Reviewer 存在未允许字段："
            f"{sorted(extra_keys)}"
        )

    decision = str(
        payload.get("review_decision") or ""
    ).strip()

    if decision not in {
        "approve",
        "reject",
        "uncertain",
    }:
        raise ValueError(
            f"非法 review_decision={decision!r}"
        )

    confidence_raw = payload.get("confidence")
    if (
        isinstance(confidence_raw, bool)
        or not isinstance(
            confidence_raw,
            (int, float),
        )
    ):
        raise ValueError(
            "Strong Reviewer confidence 必须是数字"
        )

    confidence = float(confidence_raw)
    if not 0.0 <= confidence <= 1.0:
        raise ValueError(
            "Strong Reviewer confidence 必须在 [0,1]"
        )

    summary = str(
        payload.get("summary") or ""
    ).strip()

    if not summary:
        raise ValueError(
            "Strong Reviewer summary 不能为空"
        )

    raw_checks = payload.get("checks")
    if not isinstance(raw_checks, dict):
        raise ValueError(
            "Strong Reviewer checks 必须是 object"
        )

    if set(raw_checks) != set(STRONG_REVIEW_CHECK_KEYS):
        raise ValueError(
            "Strong Reviewer checks 字段必须严格为："
            f"{list(STRONG_REVIEW_CHECK_KEYS)}"
        )

    checks: dict[str, bool] = {}

    for key in STRONG_REVIEW_CHECK_KEYS:
        value = raw_checks.get(key)
        if not isinstance(value, bool):
            raise ValueError(
                f"checks.{key} 必须是 boolean"
            )
        checks[key] = value

    raw_issues = payload.get("issues")
    if not isinstance(raw_issues, list):
        raise ValueError(
            "Strong Reviewer issues 必须是 list"
        )

    allowed_issue_keys = {
        "code",
        "obligation_type",
        "reason",
    }

    allowed_obligation_types = {
        "fault_location",
        "fault_logic",
        "dependency_context",
        "state_flow",
        "behavior_constraint",
        "repair_scope",
        "validation_constraint",
    }

    issues: list[dict[str, Any]] = []

    for index, raw_issue in enumerate(raw_issues):
        if not isinstance(raw_issue, dict):
            raise ValueError(
                f"issues[{index}] 必须是 object"
            )

        extra_issue_keys = (
            set(raw_issue)
            - allowed_issue_keys
        )

        if extra_issue_keys:
            raise ValueError(
                f"issues[{index}] 存在未允许字段："
                f"{sorted(extra_issue_keys)}"
            )

        code = str(
            raw_issue.get("code") or ""
        ).strip()

        reason = str(
            raw_issue.get("reason") or ""
        ).strip()

        obligation_type = (
            raw_issue.get(
                "obligation_type"
            )
        )

        # ----------------------------------------------------------
        # 无语义风险格式规范化：
        #
        # "null" / "None" / "" 在这里都只表示：
        #     “这个 review issue 不对应具体 obligation type。”
        #
        # 因此统一成 None。
        # 其它未知非空字符串仍严格报错。
        # ----------------------------------------------------------

        if obligation_type is not None:
            obligation_type = str(
                obligation_type
            ).strip()

            if obligation_type.lower() in {
                "",
                "null",
                "none",
            }:
                obligation_type = None

        if obligation_type is not None:
            if (
                obligation_type
                not in allowed_obligation_types
            ):
                raise ValueError(
                    f"issues[{index}].obligation_type 非法："
                    f"{obligation_type!r}"
                )

        if not code:
            raise ValueError(
                f"issues[{index}].code 不能为空"
            )

        if not reason:
            raise ValueError(
                f"issues[{index}].reason 不能为空"
            )

        issues.append(
            {
                "code": code,
                "obligation_type": obligation_type,
                "reason": reason,
            }
        )

    all_checks_true = all(checks.values())
    any_check_false = any(
        not value
        for value in checks.values()
    )

    if decision == "approve":
        if not all_checks_true:
            raise ValueError(
                "review_decision=approve 时六项 checks 必须全部为 true"
            )
        if issues:
            raise ValueError(
                "review_decision=approve 时 issues 必须为空"
            )

    elif decision == "reject":
        if not any_check_false:
            raise ValueError(
                "review_decision=reject 时至少一项 checks 必须为 false"
            )
        if not issues:
            raise ValueError(
                "review_decision=reject 时 issues 必须非空"
            )

    elif decision == "uncertain":
        if not issues:
            raise ValueError(
                "review_decision=uncertain 时 issues 必须非空"
            )

    return {
        "review_decision": decision,
        "confidence": confidence,
        "summary": summary,
        "checks": checks,
        "issues": issues,
    }


@dataclass(frozen=True)
class RefinementTeacherConfig:
    model: str = DEFAULT_REFINEMENT_MODEL
    base_url: str = DEFAULT_DEEPSEEK_BASE_URL
    api_key_env: str = "DEEPSEEK_API_KEY"
    thinking_type: str = DEFAULT_THINKING_TYPE
    reasoning_effort: str = DEFAULT_REASONING_EFFORT
    max_tokens: int = 3000
    timeout_seconds: float = 180.0
    max_retries: int = 3

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError("model 不能为空")
        if not self.base_url.strip():
            raise ValueError("base_url 不能为空")
        if self.base_url.rstrip("/").endswith("/chat/completions"):
            raise ValueError("base_url 应为 API 根地址，不能包含 /chat/completions")
        if self.thinking_type not in {"enabled", "disabled"}:
            raise ValueError("thinking_type 只能 enabled/disabled")
        if self.max_tokens < 256:
            raise ValueError("max_tokens 必须 >= 256")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds 必须 > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries 不能为负数")


@dataclass(frozen=True)
class RefinementTeacherCallMetadata:
    provider: str
    model: str
    base_url: str
    thinking_type: str
    reasoning_effort_requested: str
    reasoning_effort_effective: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_seconds: float
    offline_gold_reference_used: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "base_url": self.base_url,
            "thinking_type": self.thinking_type,
            "reasoning_effort_requested": self.reasoning_effort_requested,
            "reasoning_effort_effective": self.reasoning_effort_effective,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "latency_seconds": self.latency_seconds,
            "offline_gold_reference_used": self.offline_gold_reference_used,
        }


@dataclass(frozen=True)
class RefinementTeacherCallResult:
    output_text: str
    metadata: RefinementTeacherCallMetadata


def load_project_dotenv() -> dict[str, Any]:
    """
    只加载项目根目录 .env。

    override=False：
        shell 环境变量优先，不让 .env 覆盖显式设置。
    """
    dotenv_path = PROJECT_ROOT / ".env"
    exists = dotenv_path.is_file()
    loaded = False

    if exists:
        try:
            from dotenv import load_dotenv
        except ImportError as exc:
            raise RuntimeError(
                "项目存在 .env，但未安装 python-dotenv。"
                "请执行：python -m pip install -U python-dotenv"
            ) from exc

        loaded = bool(load_dotenv(dotenv_path=dotenv_path, override=False))

    return {
        "dotenv_path": str(dotenv_path),
        "dotenv_exists": exists,
        "dotenv_loaded": loaded,
    }


def resolve_api_key(
    preferred_env: str = "DEEPSEEK_API_KEY",
) -> tuple[str, str]:
    """
    读取 DeepSeek API Key。

    这里故意不再 fallback 到 OPENAI_API_KEY。

    原因：
        当前项目的 OPENAI_API_KEY* 全部属于 SiliconFlow。
        如果 DeepSeek Key 缺失却错误 fallback 到 OPENAI_API_KEY，
        会导致“拿 SiliconFlow Key 请求 DeepSeek Endpoint”的隐蔽错误。
    """
    candidates = tuple(
        dict.fromkeys(
            [
                preferred_env,
                "DEEPSEEK_API_KEY",
            ]
        )
    )

    for name in candidates:
        value = os.getenv(name)

        if (
            isinstance(value, str)
            and value.strip()
        ):
            return (
                value.strip(),
                name,
            )

    raise RuntimeError(
        "没有找到 DeepSeek API Key。"
        "请在项目根目录 .env 设置 DEEPSEEK_API_KEY。"
    )


def resolve_siliconflow_api_keys(
    env_names: Sequence[str] = (
        SILICONFLOW_API_KEY_ENV_NAMES
    ),
) -> list[tuple[str, str]]:
    """
    读取所有非空 SiliconFlow API Key。

    返回：
        [
          ("OPENAI_API_KEY", "<secret>"),
          ("OPENAI_API_KEY_2", "<secret>"),
          ...
        ]

    重要：
        后续 report 只记录“Key 数量”，不会写 Key 内容或 hash。
    """
    resolved: list[
        tuple[str, str]
    ] = []

    for name in env_names:
        value = os.getenv(
            str(name)
        )

        if (
            isinstance(value, str)
            and value.strip()
        ):
            resolved.append(
                (
                    str(name),
                    value.strip(),
                )
            )

    if not resolved:
        raise RuntimeError(
            "没有找到 SiliconFlow API Key。"
            "请至少设置 OPENAI_API_KEY / "
            "OPENAI_API_KEY_2 / OPENAI_API_KEY_3 / OPENAI_API_KEY_4 中一个。"
        )

    return resolved


def api_key_fingerprint(api_key: str) -> str:
    """只记录不可逆短指纹，不泄漏 API Key。"""
    digest = hashlib.sha256(api_key.encode("utf-8")).hexdigest()[:12]
    return f"sha256:{digest}"


def resolve_bigmodel_api_keys(
    *,
    max_suffix: int = 32,
) -> list[tuple[str, str]]:
    """
    自动发现 BigModel Multi-Key（多密钥）。

    支持：
        BIGMOD_API_KEY
        BIGMOD_API_KEY_2
        BIGMOD_API_KEY_3
        ...
        BIGMOD_API_KEY_32

    返回：
        [
            (api_key, env_name),
            ...
        ]

    规则：
        1. 只读取 BIGMOD_API_KEY*；
        2. 忽略不存在或空字符串；
        3. 按主 Key、_2、_3...稳定排序；
        4. 相同 Key 即使被重复填入多个变量，也只保留一次；
        5. 不 fallback 到 OPENAI_API_KEY* / DEEPSEEK_API_KEY。

    安全：
        API Key 本身不会写入 report。
    """

    if max_suffix < 2:
        raise ValueError(
            "max_suffix 必须 >= 2"
        )

    env_names = [
        BIGMODEL_API_KEY_ENV,
        *[
            f"{BIGMODEL_API_KEY_ENV}_{index}"
            for index in range(
                2,
                max_suffix + 1,
            )
        ],
    ]

    discovered: list[
        tuple[str, str]
    ] = []

    seen_keys: set[str] = set()

    for env_name in env_names:
        value = os.getenv(
            env_name
        )

        if (
            not isinstance(
                value,
                str,
            )
            or not value.strip()
        ):
            continue

        api_key = value.strip()

        # 防止用户不小心把同一 Key 重复写进多个变量，
        # 否则会错误放大同一 Key 的并发。
        if api_key in seen_keys:
            continue

        seen_keys.add(
            api_key
        )

        discovered.append(
            (
                api_key,
                env_name,
            )
        )

    if not discovered:
        raise RuntimeError(
            "没有找到 BigModel API Key。"
            "请在项目根目录 .env 设置 "
            "BIGMOD_API_KEY 或 BIGMOD_API_KEY_2..."
        )

    return discovered


@dataclass(frozen=True)
class BigModelTeacherConfig:
    """
    BigModel / GLM Primary Teacher 配置。

    为了和 Qwen3-8B 做公平 A/B：
        thinking_type = disabled
        max_tokens = 3000

    也就是说，只改变 Primary Model（主模型），
    不同时改变“是否思考”和输出预算。
    """

    model: str = DEFAULT_BIGMODEL_MODEL
    base_url: str = DEFAULT_BIGMODEL_BASE_URL
    # 每个 BigModel API Key 的最大并发槽位。
    #
    # 当前默认 1：
    #     5 个 Key -> 总并发 5
    #
    # 不建议单 Key 再提高到 4；
    # 之前单 Key 高并发已经观察到 429。
    per_key_concurrency: int = (
        DEFAULT_BIGMODEL_CONCURRENCY
    )

    thinking_type: str = DEFAULT_THINKING_TYPE

    # 监督标签生成要求可重复。
    # BigModel 默认开启 sampling（随机采样），因此显式关闭。
    do_sample: bool = False

    max_tokens: int = 3000
    timeout_seconds: float = 180.0
    max_retries: int = 6
    retry_max_sleep_seconds: float = 20.0

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError(
                "BigModel model 不能为空"
            )

        if not self.base_url.strip():
            raise ValueError(
                "BigModel base_url 不能为空"
            )

        if (
            self.base_url.rstrip("/")
            .endswith(
                "/chat/completions"
            )
        ):
            raise ValueError(
                "BigModel base_url 应为 API 根地址，"
                "不能直接包含 /chat/completions"
            )

        if self.per_key_concurrency < 1:
            raise ValueError(
                "BigModel per_key_concurrency 必须 >= 1"
            )

        if self.thinking_type != "disabled":
            raise ValueError(
                "当前监督协议要求 GLM 使用非思考模式："
                'thinking={"type":"disabled"}'
            )

        if self.do_sample is not False:
            raise ValueError(
                "监督标签生成要求可重复：BigModel do_sample 必须为 False"
            )

        if self.max_tokens < 256:
            raise ValueError(
                "BigModel max_tokens 必须 >= 256"
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "BigModel timeout_seconds 必须 > 0"
            )

        if self.max_retries < 0:
            raise ValueError(
                "BigModel max_retries 不能为负数"
            )


def validate_bigmodel_environment(
    config: BigModelTeacherConfig,
) -> dict[str, Any]:
    """
    检查 BigModel 环境。

    Report（报告）只记录：
        Key 是否存在
        Key 来源环境变量名

    不记录 Key 本身，也不记录 hash。
    """

    config.validate()

    dotenv = (
        load_project_dotenv()
    )

    key_records = (
        resolve_bigmodel_api_keys()
    )

    key_sources = [
        env_name
        for _api_key, env_name in key_records
    ]

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "缺少 openai Python SDK。"
            "请执行：python -m pip install -U openai"
        ) from exc

    return {
        "provider": "bigmodel",
        "model": config.model,
        "base_url": config.base_url,
        "api_key_present": True,

        # 只记录变量名和数量，不记录 Key 内容。
        "available_api_key_count": len(
            key_records
        ),
        "api_key_source_envs": (
            key_sources
        ),
        "per_key_concurrency": (
            config.per_key_concurrency
        ),
        "effective_max_concurrency": (
            len(
                key_records
            )
            * config.per_key_concurrency
        ),
        "thinking_type": (
            config.thinking_type
        ),
        "do_sample": (
            config.do_sample
        ),
        "generation_determinism": (
            "sampling_disabled"
            if config.do_sample is False
            else "sampling_enabled"
        ),
        "openai_sdk_present": True,
        "openai_sdk_version": (
            getattr(
                openai,
                "__version__",
                "unknown",
            )
        ),
        "dotenv": dotenv,
    }


@dataclass(frozen=True)
class SiliconFlowTeacherConfig:
    """
    SiliconFlow + Qwen3-8B 主 Teacher 配置。

    注意：
        Qwen3-8B 调用不传 reasoning_effort；
        只显式设置：

            "thinking": {"type": "disabled"}

        避免把 DeepSeek 专用调用参数混到 SiliconFlow Provider。
    """

    model: str = (
        DEFAULT_SILICONFLOW_MODEL
    )

    base_url: str = (
        DEFAULT_SILICONFLOW_BASE_URL
    )

    per_key_concurrency: int = (
        DEFAULT_SILICONFLOW_PER_KEY_CONCURRENCY
    )

    thinking_type: str = (
        DEFAULT_THINKING_TYPE
    )

    max_tokens: int = 3000
    timeout_seconds: float = 180.0

    # 显式重试由本类控制。
    max_retries: int = 4

    # 429 / 5xx 退避上限。
    retry_max_sleep_seconds: float = 20.0

    def validate(self) -> None:
        if not self.model.strip():
            raise ValueError(
                "SiliconFlow model 不能为空"
            )

        if not self.base_url.strip():
            raise ValueError(
                "SiliconFlow base_url 不能为空"
            )

        if (
            self.base_url.rstrip("/")
            .endswith(
                "/chat/completions"
            )
        ):
            raise ValueError(
                "SiliconFlow base_url 应为 /v1 根地址，"
                "不能直接包含 /chat/completions"
            )

        if self.per_key_concurrency < 1:
            raise ValueError(
                "per_key_concurrency 必须 >= 1"
            )

        if self.thinking_type != "disabled":
            raise ValueError(
                "当前监督修正协议要求 SiliconFlow/Qwen3 "
                '使用非思考模式：thinking={"type":"disabled"}'
            )

        if self.max_tokens < 256:
            raise ValueError(
                "max_tokens 必须 >= 256"
            )

        if self.timeout_seconds <= 0:
            raise ValueError(
                "timeout_seconds 必须 > 0"
            )

        if self.max_retries < 0:
            raise ValueError(
                "max_retries 不能为负数"
            )


def validate_siliconflow_environment(
    config: SiliconFlowTeacherConfig,
) -> dict[str, Any]:
    """
    检查 SiliconFlow 环境。

    不输出 API Key，也不输出 API Key hash。
    """
    config.validate()

    dotenv = (
        load_project_dotenv()
    )

    keys = (
        resolve_siliconflow_api_keys()
    )

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "缺少 openai Python SDK。"
            "请执行：python -m pip install -U openai"
        ) from exc

    return {
        "provider": (
            "siliconflow"
        ),
        "model": config.model,
        "base_url": (
            config.base_url
        ),
        "available_api_key_count": (
            len(keys)
        ),
        "per_key_concurrency": (
            config.per_key_concurrency
        ),
        "effective_max_concurrency": (
            len(keys)
            * config.per_key_concurrency
        ),
        "thinking_type": (
            config.thinking_type
        ),
        "openai_sdk_present": True,
        "openai_sdk_version": (
            getattr(
                openai,
                "__version__",
                "unknown",
            )
        ),
        "dotenv": dotenv,
    }


def validate_refinement_environment(
    config: RefinementTeacherConfig,
) -> dict[str, Any]:
    config.validate()
    dotenv = load_project_dotenv()
    api_key, source = resolve_api_key(config.api_key_env)

    try:
        import openai
    except ImportError as exc:
        raise RuntimeError(
            "缺少 openai Python SDK。"
            "请执行：python -m pip install -U openai"
        ) from exc

    return {
        "provider": "deepseek",
        "model": config.model,
        "base_url": config.base_url,
        "api_key_present": True,
        "api_key_source_env": source,
        "openai_sdk_present": True,
        "openai_sdk_version": getattr(openai, "__version__", "unknown"),
        "thinking_type": config.thinking_type,
        "reasoning_effort_effective": config.reasoning_effort,
        "dotenv": dotenv,
    }


def _json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
    )


def compact_original_obligations(
    obligations: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """
    将 Original Obligations（原始证据要求）压缩为 Teacher 真正需要的字段。

    明确删除：
        annotation_ids
        construction_method
        confidence
        provenance
        其它训练/审计元数据

    保留：
        obligation_id
        type
        description
        applicable
        mandatory
        witness_groups 的 Evidence ID 逻辑

    这样不会改变监督语义，但能显著减少 Prompt Token（提示词词元）。
    """
    result: list[dict[str, Any]] = []

    for obligation in obligations:
        compact_groups: list[list[str]] = []

        for group in (
            obligation.get("witness_groups")
            or []
        ):
            ids = [
                str(evidence_id)
                for evidence_id in (
                    group.get("evidence_ids")
                    or []
                )
            ]

            if ids:
                compact_groups.append(ids)

        result.append({
            "obligation_id": str(
                obligation.get("obligation_id")
                or ""
            ),
            "type": str(
                obligation.get("type")
                or ""
            ),
            "description": str(
                obligation.get("description")
                or ""
            ),
            "applicable": bool(
                obligation.get("applicable")
            ),
            "mandatory": bool(
                obligation.get("mandatory")
            ),
            # 外层 list = OR；内层 list = AND。
            "witness_groups": compact_groups,
        })

    return result


def extract_gold_change_hints(
    gold_patch: str | None,
    test_patch: str | None,
) -> dict[str, Any] | None:
    """
    从完整 Gold Patch / Test Patch 中确定性提取轻量 Change Hints（修改提示）。

    不把 added/removed code 正文送给 Teacher。

    只保留：
        changed_files
        test_files
        hunk_headers

    目的：
        保留“真实修改集中在哪里”的离线监督价值，
        同时避免完整 diff 大量消耗 Prompt Token。

    注意：
        这是 Data Processing（数据处理）辅助信息，
        不是在线 Agent 可见 Evidence。
    """

    def parse_patch(
        patch_text: str,
    ) -> tuple[
        set[str],
        list[str],
    ]:
        files: set[str] = set()
        hunks: list[str] = []

        current_file: str | None = None

        for raw_line in patch_text.splitlines():
            line = raw_line.rstrip()

            if line.startswith("diff --git "):
                # 格式通常：
                # diff --git a/path/to/file.py b/path/to/file.py
                parts = line.split()

                if len(parts) >= 4:
                    candidate = parts[3]

                    if candidate.startswith("b/"):
                        candidate = candidate[2:]

                    current_file = candidate

                    if candidate:
                        files.add(candidate)

                continue

            if line.startswith("+++ "):
                candidate = line[4:].strip()

                if candidate == "/dev/null":
                    continue

                if candidate.startswith("b/"):
                    candidate = candidate[2:]

                if candidate:
                    current_file = candidate
                    files.add(candidate)

                continue

            if line.startswith("@@"):
                # 只保留 hunk header，不保留 diff 正文。
                # 示例：
                # @@ -120,5 +120,8 @@ def convert_encodings(...)
                header = line

                # 防止异常超长函数签名污染 prompt。
                if len(header) > 300:
                    header = header[:300]

                if current_file:
                    hunks.append(
                        f"{current_file}: {header}"
                    )
                else:
                    hunks.append(header)

        # 稳定去重，保持原出现顺序。
        hunks = list(dict.fromkeys(hunks))

        return files, hunks

    gold_text = (
        gold_patch
        if isinstance(gold_patch, str)
        else ""
    )

    test_text = (
        test_patch
        if isinstance(test_patch, str)
        else ""
    )

    if not gold_text.strip() and not test_text.strip():
        return None

    changed_files, gold_hunks = parse_patch(
        gold_text
    )

    test_files, test_hunks = parse_patch(
        test_text
    )

    # 有些 test_patch 也会改非 tests/ 文件，因此分开记录来源，
    # 不尝试通过文件名猜测“是不是测试”。
    return {
        "changed_files": sorted(
            changed_files
        ),
        "test_patch_files": sorted(
            test_files
        ),
        "hunk_headers": (
            gold_hunks[:32]
        ),
        "test_hunk_headers": (
            test_hunks[:32]
        ),
    }


def _candidate_block(
    index: int,
    evidence: Mapping[str, Any],
) -> str:
    """
    Compact Candidate Rendering（紧凑候选渲染）。

    Teacher 真正需要：
        Evidence ID
        path
        symbol
        line range
        online rank（如果有）
        pre-fix 代码正文

    v1.8.1 有意不再把：
        certificate
        original_witness
        boundary

    写到每条 Candidate header。

    原因：
        这些是旧监督身份，不是 Evidence 语义质量；
        在每条候选上反复强调会诱发 Anchoring（锚定偏置）。

    Original Trajectory / Original Obligations 区域仍保留旧监督，
    因此 Teacher 仍然知道原 Certificate 是哪些 ID，
    只是不能把“旧身份”当作候选质量提示。

    不再把完整 candidate_stats JSON 放进 prompt。
    """
    stats = (
        evidence.get("candidate_stats")
        or {}
    )

    fields = [
        f"id={evidence.get('evidence_id')}",
        f"path={evidence.get('path')}",
    ]

    symbol = evidence.get("symbol")
    if symbol:
        fields.append(
            f"symbol={symbol}"
        )

    start_line = evidence.get("start_line")
    end_line = evidence.get("end_line")

    if (
        start_line is not None
        or end_line is not None
    ):
        fields.append(
            f"lines={start_line}-{end_line}"
        )

    rank = stats.get(
        "min_online_rank"
    )

    if rank is not None:
        fields.append(
            f"rank={rank}"
        )

    # v1.8.1：
    # Candidate header 不展示旧监督身份 flags。
    # Original Trajectory / Obligations 已经提供原监督关系。
    content = str(
        evidence.get("content")
        or ""
    )

    return (
        f"[CANDIDATE {index}] "
        + " | ".join(fields)
        + "\n"
        + content
    )


def build_refinement_user_prompt(
    *,
    task_id: str,
    question: str,
    original_obligations: Sequence[
        Mapping[str, Any]
    ],
    original_boundary_evidence_ids: Sequence[str],
    original_certificate_evidence_ids: Sequence[str],
    candidate_evidence: Sequence[
        Mapping[str, Any]
    ],
    gold_patch: str | None = None,
    test_patch: str | None = None,
) -> str:
    """
    构造 v1.2 Compact User Prompt（紧凑任务提示词）。

    与 v1.1 相比：
        - Original Obligations 只保留必要字段；
        - Candidate metadata 大幅压缩；
        - 完整 Gold/Test Patch 不再进入 prompt；
        - 只传确定性 Gold Change Hints（修改位置提示）。

    Issue / Question 暂时不裁剪：
        因为 satisfied_by_question（问题已满足）判断依赖完整问题语义。
        这是有意保守，而不是遗漏优化。
    """
    compact_obligations = (
        compact_original_obligations(
            original_obligations
        )
    )

    gold_hints = (
        extract_gold_change_hints(
            gold_patch,
            test_patch,
        )
    )

    fixed_slot_types = (
        "fault_location",
        "fault_logic",
        "dependency_context",
        "state_flow",
        "behavior_constraint",
        "repair_scope",
        "validation_constraint",
    )

    original_by_type = {
        str(
            obligation.get(
                "type"
            )
            or ""
        ): obligation
        for obligation in (
            compact_obligations
        )
        if str(
            obligation.get(
                "type"
            )
            or ""
        )
    }

    slot_availability = {
        slot_type: {
            "present": (
                slot_type
                in original_by_type
            ),
            "mandatory": (
                bool(
                    original_by_type[
                        slot_type
                    ].get(
                        "mandatory",
                        False,
                    )
                )
                if slot_type
                in original_by_type
                else False
            ),
            "original_witness_group_count": (
                len(
                    original_by_type[
                        slot_type
                    ].get(
                        "witness_groups"
                    )
                    or []
                )
                if slot_type
                in original_by_type
                else 0
            ),
            "keep_original_allowed": (
                slot_type
                in original_by_type
            ),
        }
        for slot_type in (
            fixed_slot_types
        )
    }

    blocks = [
        "[TASK]",
        task_id,
        "",
        "[ISSUE / QUESTION]",
        question.strip(),
        "",
        "[ORIGINAL TRAJECTORY]",
        "boundary="
        + json.dumps(
            list(
                original_boundary_evidence_ids
            ),
            ensure_ascii=False,
        ),
        "complete="
        + json.dumps(
            list(
                original_certificate_evidence_ids
            ),
            ensure_ascii=False,
        ),
        "",
        "[ORIGINAL OBLIGATIONS - COMPACT]",
        _json(
            compact_obligations
        ),
        "",
        "[ORIGINAL SLOT AVAILABILITY]",
        _json(
            slot_availability
        ),
        "",
        "[CANDIDATE EVIDENCE POOL]",
    ]

    for index, evidence in enumerate(
        candidate_evidence,
        start=1,
    ):
        blocks.extend(
            [
                "",
                _candidate_block(
                    index,
                    evidence,
                ),
            ]
        )

    if gold_hints is not None:
        blocks.extend(
            [
                "",
                "[GOLD CHANGE HINTS - OFFLINE ONLY, NOT EVIDENCE]",
                _json(
                    gold_hints
                ),
            ]
        )

    blocks.extend(
        [
            "",
            "[OUTPUT]",
            (
                "只输出符合 system prompt 的 Fixed Decision JSON object；"
                "只能引用 Candidate Number，不得输出 Evidence ID。"
            ),
        ]
    )

    return "\n".join(
        blocks
    )



class DeepSeekRefinementTeacher:
    """DeepSeek Chat Completions adapter（适配器）。"""

    def __init__(
        self,
        config: RefinementTeacherConfig,
        *,
        client: Any | None = None,
    ) -> None:
        config.validate()

        if config.thinking_type != "disabled":
            raise ValueError(
                "当前监督修正协议要求 DeepSeek 使用非思考模式："
                'thinking={"type":"disabled"}'
            )

        self.config = config

        load_project_dotenv()
        api_key, source = resolve_api_key(config.api_key_env)
        self.api_key_source = source

        if client is None:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError("缺少 openai Python SDK") from exc

            self.client = OpenAI(
                api_key=api_key,
                base_url=config.base_url,
                timeout=config.timeout_seconds,
                max_retries=config.max_retries,
            )
        else:
            self.client = client

    def call(
        self,
        *,
        user_prompt: str,
        offline_gold_reference_used: bool,
        system_prompt: str | None = None,
    ) -> RefinementTeacherCallResult:
        """
        调用一次 DeepSeek。

        reasoning_content（思维正文）不保存；
        只保存最终 JSON 与 token/latency metadata。
        """
        started = time.perf_counter()

        response = self.client.chat.completions.create(
            model=self.config.model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        system_prompt
                        or REFINEMENT_SYSTEM_PROMPT
                    ),
                },
                {
                    "role": "user",
                    "content": user_prompt,
                },
            ],
            reasoning_effort=self.config.reasoning_effort,
            extra_body={
                "thinking": {
                    "type": self.config.thinking_type,
                }
            },
            response_format={"type": "json_object"},
            max_tokens=self.config.max_tokens,
            stream=False,
        )

        latency = time.perf_counter() - started

        choices = getattr(response, "choices", None)
        if not choices:
            raise RuntimeError("DeepSeek 返回空 choices")

        output_text = getattr(choices[0].message, "content", None)
        if not isinstance(output_text, str) or not output_text.strip():
            raise RuntimeError("DeepSeek 没有返回非空 JSON content")

        usage = getattr(response, "usage", None)
        prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
        completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)
        total_tokens = int(
            getattr(
                usage,
                "total_tokens",
                prompt_tokens + completion_tokens,
            )
            or (prompt_tokens + completion_tokens)
        )

        model = str(getattr(response, "model", None) or self.config.model)

        return RefinementTeacherCallResult(
            output_text=output_text,
            metadata=RefinementTeacherCallMetadata(
                provider="deepseek",
                model=model,
                base_url=self.config.base_url,
                thinking_type=self.config.thinking_type,
                reasoning_effort_requested=self.config.reasoning_effort,
                reasoning_effort_effective=self.config.reasoning_effort,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                latency_seconds=latency,
                offline_gold_reference_used=offline_gold_reference_used,
            ),
        )


class DeepSeekRefinementTeacherPool:
    """
    DeepSeek Strong Teacher Pool（强教师并发池）。

    用户当前 DeepSeek Key 的服务端并发上限很高，
    但程序默认不会直接开到 500。

    Runner 可以从较保守的并发（例如 8/16/32）开始，
    再根据实际 429 / TPM 情况调整。

    每个 Slot（槽位）拥有独立 OpenAI Client，
    避免多线程共享同一个 Client 对象。
    """

    def __init__(
        self,
        config: RefinementTeacherConfig,
        *,
        concurrency: int,
        teachers: Sequence[
            DeepSeekRefinementTeacher
        ] | None = None,
    ) -> None:
        config.validate()

        if config.thinking_type != "disabled":
            raise ValueError(
                "DeepSeek Strong Teacher 必须使用 thinking disabled"
            )

        if concurrency < 1:
            raise ValueError(
                "DeepSeek concurrency 必须 >= 1"
            )

        self.config = config

        if teachers is None:
            slots = [
                DeepSeekRefinementTeacher(
                    config
                )
                for _ in range(
                    concurrency
                )
            ]
        else:
            slots = list(
                teachers
            )

            if not slots:
                raise ValueError(
                    "teachers 不能为空"
                )

        self._available: queue.Queue[
            tuple[
                int,
                DeepSeekRefinementTeacher,
            ]
        ] = queue.Queue()

        for index, teacher in enumerate(
            slots
        ):
            self._available.put(
                (
                    index,
                    teacher,
                )
            )

        self.slot_count = len(
            slots
        )

    def call(
        self,
        *,
        user_prompt: str,
        offline_gold_reference_used: bool,
        system_prompt: str | None = None,
    ) -> RefinementTeacherCallResult:
        (
            index,
            teacher,
        ) = self._available.get()

        try:
            return teacher.call(
                user_prompt=(
                    user_prompt
                ),
                offline_gold_reference_used=(
                    offline_gold_reference_used
                ),
                system_prompt=(
                    system_prompt
                ),
            )

        finally:
            self._available.put(
                (
                    index,
                    teacher,
                )
            )


class BigModelRefinementTeacherPool:
    """
    BigModel / GLM OpenAI-compatible Primary Teacher Pool。

    当前用户只配置一个 BIGMOD_API_KEY，
    所以通过 concurrency 创建多个独立 Client Slot。

    请求合同：
        model = glm-4.7-flash
        thinking.type = disabled
        response_format = json_object
        stream = false

    429 / 5xx：
        显式指数退避 + jitter（随机抖动）。
    """

    def __init__(
        self,
        config: BigModelTeacherConfig,
        *,
        clients: Sequence[Any] | None = None,
    ) -> None:
        config.validate()
        self.config = config

        load_project_dotenv()

        if clients is not None:
            if not clients:
                raise ValueError(
                    "clients 不能为空"
                )
            self._slots = list(
                clients
            )

        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 openai Python SDK"
                ) from exc

            key_records = (
                resolve_bigmodel_api_keys()
            )

            self._slots = []

            # ------------------------------------------------------
            # 每个 Key 默认只创建 1 个 Client Slot。
            #
            # 例如：
            #     5 keys × per_key_concurrency=1
            #         -> 5 total slots
            #
            # 这样增加吞吐靠“增加独立 Key”，
            # 而不是把单 Key 并发压高导致 429。
            # ------------------------------------------------------

            for (
                api_key,
                _source_env,
            ) in key_records:
                for _ in range(
                    config.per_key_concurrency
                ):
                    self._slots.append(
                        OpenAI(
                            api_key=(
                                api_key
                            ),
                            base_url=(
                                config.base_url
                            ),
                            timeout=(
                                config.timeout_seconds
                            ),
                            max_retries=0,
                        )
                    )

        self._available: queue.Queue[
            tuple[int, Any]
        ] = queue.Queue()

        for index, client in enumerate(
            self._slots
        ):
            self._available.put(
                (
                    index,
                    client,
                )
            )

        self.slot_count = len(
            self._slots
        )

    @staticmethod
    def _status_code(
        exc: BaseException,
    ) -> int | None:
        value = getattr(
            exc,
            "status_code",
            None,
        )

        if isinstance(value, int):
            return value

        response = getattr(
            exc,
            "response",
            None,
        )

        value = getattr(
            response,
            "status_code",
            None,
        )

        if isinstance(value, int):
            return value

        return None

    def _should_retry(
        self,
        exc: BaseException,
    ) -> bool:
        status = self._status_code(
            exc
        )

        if status is None:
            return True

        return (
            status == 429
            or status >= 500
        )

    def call(
        self,
        *,
        user_prompt: str,
        offline_gold_reference_used: bool,
        system_prompt: str | None = None,
    ) -> RefinementTeacherCallResult:
        """
        调用 GLM Primary Teacher。

        thinking 必须 disabled，
        与 Qwen baseline 保持实验变量一致。
        """

        (
            slot_index,
            client,
        ) = self._available.get()

        try:
            last_error: BaseException | None = None

            for attempt in range(
                self.config.max_retries
                + 1
            ):
                started = (
                    time.perf_counter()
                )

                try:
                    response = (
                        client
                        .chat
                        .completions
                        .create(
                            model=(
                                self.config.model
                            ),
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        system_prompt
                                        or REFINEMENT_SYSTEM_PROMPT
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        user_prompt
                                    ),
                                },
                            ],

                            # GLM-4.7 系列默认开启 thinking。
                            # 本实验显式关闭，保证与 Qwen A/B 公平。
                            extra_body={
                                "thinking": {
                                    "type": (
                                        self.config
                                        .thinking_type
                                    )
                                },

                                # 监督生成禁用随机采样。
                                # do_sample=false 时服务端忽略 temperature/top_p。
                                "do_sample": (
                                    self.config
                                    .do_sample
                                ),
                            },

                            response_format={
                                "type": (
                                    "json_object"
                                )
                            },
                            max_tokens=(
                                self.config
                                .max_tokens
                            ),
                            stream=False,
                        )
                    )

                    latency = (
                        time.perf_counter()
                        - started
                    )

                    choices = getattr(
                        response,
                        "choices",
                        None,
                    )

                    if not choices:
                        raise RuntimeError(
                            "BigModel 返回空 choices"
                        )

                    output_text = getattr(
                        choices[0].message,
                        "content",
                        None,
                    )

                    if (
                        not isinstance(
                            output_text,
                            str,
                        )
                        or not output_text.strip()
                    ):
                        raise RuntimeError(
                            "BigModel 没有返回非空 JSON content"
                        )

                    usage = getattr(
                        response,
                        "usage",
                        None,
                    )

                    prompt_tokens = int(
                        getattr(
                            usage,
                            "prompt_tokens",
                            0,
                        )
                        or 0
                    )

                    completion_tokens = int(
                        getattr(
                            usage,
                            "completion_tokens",
                            0,
                        )
                        or 0
                    )

                    total_tokens = int(
                        getattr(
                            usage,
                            "total_tokens",
                            (
                                prompt_tokens
                                + completion_tokens
                            ),
                        )
                        or (
                            prompt_tokens
                            + completion_tokens
                        )
                    )

                    model = str(
                        getattr(
                            response,
                            "model",
                            None,
                        )
                        or self.config.model
                    )

                    return (
                        RefinementTeacherCallResult(
                            output_text=(
                                output_text
                            ),
                            metadata=(
                                RefinementTeacherCallMetadata(
                                    provider="bigmodel",
                                    model=model,
                                    base_url=(
                                        self.config
                                        .base_url
                                    ),
                                    thinking_type=(
                                        self.config
                                        .thinking_type
                                    ),
                                    reasoning_effort_requested=(
                                        "not_applicable"
                                    ),
                                    reasoning_effort_effective=(
                                        "not_applicable"
                                    ),
                                    prompt_tokens=(
                                        prompt_tokens
                                    ),
                                    completion_tokens=(
                                        completion_tokens
                                    ),
                                    total_tokens=(
                                        total_tokens
                                    ),
                                    latency_seconds=(
                                        latency
                                    ),
                                    offline_gold_reference_used=(
                                        offline_gold_reference_used
                                    ),
                                )
                            ),
                        )
                    )

                except BaseException as exc:
                    last_error = exc

                    if (
                        attempt
                        >= self.config.max_retries
                        or not self._should_retry(
                            exc
                        )
                    ):
                        raise

                    sleep_seconds = min(
                        self.config
                        .retry_max_sleep_seconds,
                        (
                            2.0 ** attempt
                        )
                        + random.uniform(
                            0.0,
                            0.75,
                        ),
                    )

                    time.sleep(
                        sleep_seconds
                    )

            assert last_error is not None
            raise last_error

        finally:
            self._available.put(
                (
                    slot_index,
                    client,
                )
            )


class SiliconFlowRefinementTeacherPool:
    """
    SiliconFlow Multi-Key Teacher Pool（多密钥教师池）。

    并发合同：
        每个 API Key 创建 per_key_concurrency 个独立 Client Slot（客户端槽位）。

    例如：
        4 个 Key
        × 每 Key 4 个 Slot
        = 最多 16 个并发请求。

    为什么不用“一个 Client + Semaphore”：
        独立 Client Slot 可以避免多线程共享同一个 Client 对象时
        出现未明确保证的线程安全问题。

    429/5xx：
        使用显式指数退避 + jitter（随机抖动）。

    注意：
        Key 内容绝不进入日志/report。
    """

    def __init__(
        self,
        config: SiliconFlowTeacherConfig,
        *,
        clients: Sequence[Any] | None = None,
    ) -> None:
        config.validate()

        self.config = config

        load_project_dotenv()

        if clients is not None:
            if not clients:
                raise ValueError(
                    "clients 不能为空"
                )

            # Fake Client / 单元测试路径。
            self._slots = list(
                clients
            )

        else:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "缺少 openai Python SDK"
                ) from exc

            key_pairs = (
                resolve_siliconflow_api_keys()
            )

            slots: list[Any] = []

            for (
                _env_name,
                api_key,
            ) in key_pairs:
                for _ in range(
                    config.per_key_concurrency
                ):
                    slots.append(
                        OpenAI(
                            api_key=api_key,
                            base_url=(
                                config.base_url
                            ),
                            timeout=(
                                config.timeout_seconds
                            ),

                            # SDK 内部不再自动重试，
                            # 避免与我们的显式重试叠加。
                            max_retries=0,
                        )
                    )

            self._slots = slots

        self._available: queue.Queue[
            tuple[int, Any]
        ] = queue.Queue()

        for index, client in enumerate(
            self._slots
        ):
            self._available.put(
                (
                    index,
                    client,
                )
            )

        # 只做计数，不记录 Key。
        self.slot_count = len(
            self._slots
        )

    @staticmethod
    def _status_code(
        exc: BaseException,
    ) -> int | None:
        value = getattr(
            exc,
            "status_code",
            None,
        )

        if isinstance(
            value,
            int,
        ):
            return value

        response = getattr(
            exc,
            "response",
            None,
        )

        value = getattr(
            response,
            "status_code",
            None,
        )

        if isinstance(
            value,
            int,
        ):
            return value

        return None

    def _should_retry(
        self,
        exc: BaseException,
    ) -> bool:
        status = self._status_code(
            exc
        )

        if status is None:
            # 网络瞬时异常通常没有 HTTP status。
            return True

        return (
            status == 429
            or status >= 500
        )

    def call(
        self,
        *,
        user_prompt: str,
        offline_gold_reference_used: bool,
        system_prompt: str | None = None,
    ) -> RefinementTeacherCallResult:
        """
        从 Pool 获取一个 Client Slot，完成后归还。

        所有 SiliconFlow/Qwen3 调用强制：
            thinking.type = disabled
        """
        (
            slot_index,
            client,
        ) = self._available.get()

        try:
            last_error: BaseException | None = None

            for attempt in range(
                self.config.max_retries
                + 1
            ):
                started = (
                    time.perf_counter()
                )

                try:
                    response = (
                        client
                        .chat
                        .completions
                        .create(
                            model=(
                                self.config.model
                            ),
                            messages=[
                                {
                                    "role": "system",
                                    "content": (
                                        system_prompt
                                        or REFINEMENT_SYSTEM_PROMPT
                                    ),
                                },
                                {
                                    "role": "user",
                                    "content": (
                                        user_prompt
                                    ),
                                },
                            ],

                            # 用户明确要求：
                            # Qwen3 必须非思考模式。
                            extra_body={
                                "thinking": {
                                    "type": "disabled"
                                }
                            },

                            response_format={
                                "type": (
                                    "json_object"
                                )
                            },
                            max_tokens=(
                                self.config
                                .max_tokens
                            ),
                            stream=False,
                        )
                    )

                    latency = (
                        time.perf_counter()
                        - started
                    )

                    choices = getattr(
                        response,
                        "choices",
                        None,
                    )

                    if not choices:
                        raise RuntimeError(
                            "SiliconFlow 返回空 choices"
                        )

                    output_text = (
                        getattr(
                            choices[0].message,
                            "content",
                            None,
                        )
                    )

                    if (
                        not isinstance(
                            output_text,
                            str,
                        )
                        or not output_text.strip()
                    ):
                        raise RuntimeError(
                            "SiliconFlow 没有返回非空 JSON content"
                        )

                    usage = getattr(
                        response,
                        "usage",
                        None,
                    )

                    prompt_tokens = int(
                        getattr(
                            usage,
                            "prompt_tokens",
                            0,
                        )
                        or 0
                    )

                    completion_tokens = int(
                        getattr(
                            usage,
                            "completion_tokens",
                            0,
                        )
                        or 0
                    )

                    total_tokens = int(
                        getattr(
                            usage,
                            "total_tokens",
                            (
                                prompt_tokens
                                + completion_tokens
                            ),
                        )
                        or (
                            prompt_tokens
                            + completion_tokens
                        )
                    )

                    model = str(
                        getattr(
                            response,
                            "model",
                            None,
                        )
                        or self.config.model
                    )

                    return (
                        RefinementTeacherCallResult(
                            output_text=(
                                output_text
                            ),
                            metadata=(
                                RefinementTeacherCallMetadata(
                                    provider=(
                                        "siliconflow"
                                    ),
                                    model=model,
                                    base_url=(
                                        self.config
                                        .base_url
                                    ),
                                    thinking_type=(
                                        "disabled"
                                    ),

                                    # SiliconFlow/Qwen3 路径不使用
                                    # reasoning_effort。
                                    reasoning_effort_requested=(
                                        "not_applicable"
                                    ),
                                    reasoning_effort_effective=(
                                        "not_applicable"
                                    ),
                                    prompt_tokens=(
                                        prompt_tokens
                                    ),
                                    completion_tokens=(
                                        completion_tokens
                                    ),
                                    total_tokens=(
                                        total_tokens
                                    ),
                                    latency_seconds=(
                                        latency
                                    ),
                                    offline_gold_reference_used=(
                                        offline_gold_reference_used
                                    ),
                                )
                            ),
                        )
                    )

                except BaseException as exc:
                    last_error = exc

                    if (
                        attempt
                        >= self.config.max_retries
                        or not self._should_retry(
                            exc
                        )
                    ):
                        raise

                    # 1, 2, 4, 8... 秒指数退避，
                    # 再加入 0~0.75 秒随机抖动。
                    sleep_seconds = min(
                        self.config
                        .retry_max_sleep_seconds,
                        (
                            2.0**attempt
                            + random.uniform(
                                0.0,
                                0.75,
                            )
                        ),
                    )

                    time.sleep(
                        sleep_seconds
                    )

            assert (
                last_error is not None
            )

            raise last_error

        finally:
            self._available.put(
                (
                    slot_index,
                    client,
                )
            )


def bigmodel_teacher_protocol_metadata(
    config: BigModelTeacherConfig,
) -> dict[str, Any]:
    """BigModel Primary Teacher 协议元数据。"""

    config.validate()

    return {
        "provider": "bigmodel",
        "api_format": (
            "openai_chat_completions"
        ),
        "endpoint": (
            "/chat/completions"
        ),
        "model": config.model,
        "base_url": (
            config.base_url
        ),
        "thinking": {
            "type": (
                config.thinking_type
            )
        },
        "do_sample": (
            config.do_sample
        ),
        "sampling_policy": (
            "disabled_for_reproducible_supervision"
        ),
        "reasoning_effort": (
            "not_applicable"
        ),
        "response_format": {
            "type": "json_object"
        },
        "max_tokens": (
            config.max_tokens
        ),
        "per_key_concurrency": (
            config.per_key_concurrency
        ),
        "purpose": (
            "two-stage requirement and targeted-witness teacher for offline supervision refinement"
        ),
        "protocol_version": (
            "supervision-refinement-v1.9.2.1"
        ),
        "allowed_evidence_source": (
            "pre-fix Candidate Numbers only; program resolves Evidence IDs"
        ),
        "ab_role": (
            "alternative primary teacher for Qwen3-8B comparison"
        ),
    }


def siliconflow_teacher_protocol_metadata(
    config: SiliconFlowTeacherConfig,
) -> dict[str, Any]:
    config.validate()

    return {
        "provider": (
            "siliconflow"
        ),
        "api_format": (
            "openai_chat_completions"
        ),
        "endpoint": (
            "/chat/completions"
        ),
        "model": config.model,
        "base_url": (
            config.base_url
        ),
        "thinking": {
            "type": "disabled"
        },
        "reasoning_effort": (
            "not_applicable"
        ),
        "response_format": {
            "type": "json_object"
        },
        "max_tokens": (
            config.max_tokens
        ),
        "purpose": (
            "primary offline supervision refinement teacher"
        ),
        "protocol_version": (
            "supervision-refinement-v1.9.2.1"
        ),
        "allowed_evidence_source": (
            "pre-fix candidate Evidence IDs only"
        ),
    }


def refinement_teacher_protocol_metadata(
    config: RefinementTeacherConfig,
) -> dict[str, Any]:
    """写入 report 的数据处理协议元数据。"""
    config.validate()
    return {
        "provider": "deepseek",
        "api_format": "openai_chat_completions",
        "endpoint": "/chat/completions",
        "model": config.model,
        "base_url": config.base_url,
        "thinking": {"type": config.thinking_type},
        "reasoning_effort": config.reasoning_effort,
        "response_format": {"type": "json_object"},
        "max_tokens": config.max_tokens,
        "stream": False,
        "purpose": "offline supervision refinement; not semantic evaluation",
        "protocol_version": "supervision-refinement-v1.9.2.1",
        "prompt_strategy": (
            "compact rules + explicit JSON examples + compact task payload"
        ),
        "gold_reference_strategy": (
            "deterministic change hints only; full patch text excluded from prompt"
        ),
        "allowed_evidence_source": "pre-fix candidate Evidence IDs only",
    }


def _self_check() -> None:
    """Fake Client 自检，不发送真实 API。"""

    class FakeUsage:
        prompt_tokens = 100
        completion_tokens = 20
        total_tokens = 120

    class FakeMessage:
        content = json.dumps({
            "assessment": "keep",
            "stop_assessment": "original_stop_correct",
            "confidence": 0.9,
            "rationale": "ok",
            "missing_candidate_requests": [],
            "refined_obligations": [],
            "proposed_certificate_evidence_ids": ["ev1"],
        })

    class FakeChoice:
        message = FakeMessage()

    class FakeResponse:
        choices = [FakeChoice()]
        usage = FakeUsage()
        model = "deepseek-v4-flash"

    class FakeCompletions:
        def create(self, **kwargs: Any) -> FakeResponse:
            self.last_kwargs = kwargs
            return FakeResponse()

    class FakeChat:
        def __init__(self) -> None:
            self.completions = FakeCompletions()

    class FakeClient:
        def __init__(self) -> None:
            self.chat = FakeChat()

    old = os.environ.get("DEEPSEEK_API_KEY")
    os.environ["DEEPSEEK_API_KEY"] = "test-key"

    try:
        config = RefinementTeacherConfig()
        fake = FakeClient()
        teacher = DeepSeekRefinementTeacher(config, client=fake)
        result = teacher.call(
            user_prompt="test",
            offline_gold_reference_used=True,
        )

        assert result.metadata.prompt_tokens == 100
        assert result.metadata.offline_gold_reference_used is True
        assert fake.chat.completions.last_kwargs["response_format"]["type"] == "json_object"

        # --------------------------------------------------------------
        # SiliconFlow Pool Fake Test：
        # 验证 thinking disabled + 不传 reasoning_effort。
        # --------------------------------------------------------------

        pool = SiliconFlowRefinementTeacherPool(
            SiliconFlowTeacherConfig(
                per_key_concurrency=1,
            ),
            clients=[
                fake,
            ],
        )

        pool_result = pool.call(
            user_prompt="test",
            offline_gold_reference_used=True,
        )

        pool_kwargs = (
            fake
            .chat
            .completions
            .last_kwargs
        )

        assert (
            pool_result
            .metadata
            .provider
            == "siliconflow"
        )

        assert (
            pool_kwargs[
                "extra_body"
            ][
                "thinking"
            ][
                "type"
            ]
            == "disabled"
        )

        assert (
            "reasoning_effort"
            not in pool_kwargs
        )
    finally:
        if old is None:
            os.environ.pop("DEEPSEEK_API_KEY", None)
        else:
            os.environ["DEEPSEEK_API_KEY"] = old


if __name__ == "__main__":
    _self_check()
    print("scripts/refinement_teacher.py self-check: passed")

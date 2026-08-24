# Evidence Agent

> **Gather, Combine, or Skip：面向软件修复上下文充分性的状态感知、轨迹感知与证据交互感知 Evidence Acquisition Framework**

版本日期：2026-08-22

---

## 0. 项目摘要

Evidence Agent 是一个面向软件工程缺陷定位与修复上下文获取的研究项目。项目以：

```text
SWE Issue / Problem Statement
+
Pre-fix Repository Snapshot
```

作为在线输入，通过多通道 RAG、canonical repository structure、状态条件化 Evidence Policy 和多轮 Agentic orchestration，逐步构造结构化 Evidence Package。

项目当前研究的核心不是“直接生成补丁”，而是：

> **Agent 能否在不查看 Gold Patch、不执行补丁生成和测试反馈闭环的条件下，从修复前仓库中逐步获得足以支持合理修复决策的关键证据，并在证据充分时正确停止。**

最终输出为：

```text
Evidence Package
├── Selected Evidence Units
├── Relevant Files / Symbols
├── Structural Relations
├── Acquisition Trace
├── Covered Obligations
├── Remaining Deficits
└── Sufficiency / STOP Decision
```

本项目明确不把 patch generation、patch application、test execution 或 test-feedback repair loop 作为当前实验组成部分。

---

# 1. 研究问题

传统代码检索通常学习：

\[
Score(q,u)
\]

其中：

- \(q\)：Issue；
- \(u\)：候选代码证据。

这种静态相关性排序无法回答：

```text
已经知道 K 以后，下一条 Evidence 还有多少新增价值？
```

也无法自然处理：

```text
A 单独不足，B 单独不足，但 A+B 足够；
A 和 B 可以表达相同信息，因此只需读取一个；
当前 Evidence 已经充分，继续检索只增加成本与噪声。
```

因此，本项目将问题定义为状态条件化的序列证据获取：

\[
S_t=(q,K_t)
\]

策略在每一轮选择：

\[
A_t\in\{[u],[u,v],STOP\}
\]

并学习：

\[
s_A=f_\theta(q,K_t,A)
\]

目标不是单纯提高某一次 Top-K 召回，而是形成完整轨迹：

\[
K_0=\emptyset
\rightarrow K_1
\rightarrow \cdots
\rightarrow K_T
\]

使最终 Evidence Package 达到预定义的修复上下文充分性，同时控制 Evidence 数量、token、轮数与冗余。

---

# 2. 项目边界

## 2.1 Online Agent 可以看到什么

在线系统只允许访问：

```text
Issue / Problem Statement
Pre-fix Repository
Current Evidence State K
Online Retriever / Structure Candidate
Budget / Acquisition Trace
```

## 2.2 Offline supervision / evaluator 可以看到什么

离线监督与最终评价允许访问：

```text
Gold Patch
Test Patch
Gold-derived obligations
Witness groups
Teacher-derived supervision
Evaluation labels
```

必须满足：

```text
Online Agent visible fields
               ⟂
Gold / Teacher supervision-only fields
```

Gold Patch / Test Patch 的作用是提供参考修复约束，而不是直接告诉 Agent 下一步读取什么。

## 2.3 当前明确不做

```text
Patch generation
Patch application
Test execution
Test-feedback repair loop
Post-fix code retrieval
Gold-aware online routing
```

因此论文结果应表述为：

> Evidence Acquisition / Context Sufficiency performance

而不是：

> Automated Program Repair success rate

---

# 3. 整体系统架构

```text
                 SWE Issue
                    │
                    ▼
            Problem Analysis
                    │
                    ▼
       ┌────────────────────────┐
       │ Online Candidate Space │
       │                        │
       │ Path / Content FTS     │
       │ BM25 / Symbol          │
       │ Canonical Structure    │
       └───────────┬────────────┘
                   │
                   ▼
        Candidate Actions A_t
          [u] / [u,v] / STOP
                   │
             q + K_t + A_t
                   │
                   ▼
     Cross-Encoder Evidence Policy
                   │
             argmax action
          ┌────────┼────────┐
          │        │        │
        [u]      [u,v]     STOP
          │        │        │
          └── update K ─────┘
                   │
              next round
                   │
                   ▼
           Evidence Package
```

系统中只有一套核心可训练 Evidence Policy。Retriever、结构扩展、候选生成和 Pair 构造均保持 deterministic / non-trainable，以便把“候选是否可达”和“策略是否会正确选择”分开研究。

---

# 4. Structured Evidence Sufficiency

## 4.1 七类 repair evidence obligations

项目不将“命中修改文件/修改行”直接等同于“上下文充分”。Strong Teacher / derived supervision 将修复所需信息划分为七类：

1. `fault_location`
2. `fault_logic`
3. `dependency_context`
4. `state_flow`
5. `behavior_constraint`
6. `repair_scope`
7. `validation_constraint`

因此：

```text
Gold location hit
≠
Repair-context sufficiency
```

一个 Evidence Package 即使命中了修改位置，如果缺少依赖关系、状态传播、行为约束或修改范围信息，也可能仍然不充分。

## 4.2 OR-of-AND witness semantics

每个 obligation 可以有一个或多个 witness group。

例如：

```text
[[2, 5]]
```

表示：

\[
e_2\land e_5
\]

即 2 和 5 必须同时存在。

例如：

```text
[[2], [5, 9]]
```

表示：

\[
e_2\lor(e_5\land e_9)
\]

因此同一 schema 能显式表达：

```text
Complementarity：必须组合
Substitutability：存在替代路径
```

这一语义是 Evidence Sufficiency、Pair action 与后续 trajectory evaluation 的共同基础。

---

# 5. Repository Evidence Space

## 5.1 Frozen V2.10 base release

基础冻结版本：

```text
dataset_name    = unified_swe_dataset_v2_10
dataset_version = 2.10.0
script_version  = 0.2.10
format          = parquet
```

任务规模：

| Split | Tasks |
|---|---:|
| Train | 18,347 |
| Validation | 223 |
| Benchmark | 2,294 |
| **Total** | **20,864** |

Repository corpus：

| 指标 | 数量 |
|---|---:|
| File Versions | 1,027,752 |
| Evidence Units | 25,496,300 |

Repository 被转化为稳定可寻址 Evidence Units，而不是让模型直接读取完整仓库。

## 5.2 Frozen V2.10 policy statistics

V2.10 基础 builder 的 policy state/action 统计为：

```text
Policy states = 42,284
Initial       = 20,864
Boundary      = 556
Complete      = 20,864
```

Candidate actions：

```text
Total  = 3,094,993
Single = 2,721,201
Pair   =   331,508
STOP   =    42,284
```

这里的统计属于“基础冻结 release”，不能与后续 Teacher-integrated / derived training bundle 的活跃训练状态数量混为一谈。

---

# 6. Strong Teacher 与派生监督

## 6.1 Teacher freeze

当前机械冻结 Teacher：

```text
data/strong_teacher_mechanical_v1_0.parquet
```

统计：

```text
Teacher rows = 20,588
Included     = 20,501
Excluded     =     87
```

Teacher 是 sidecar / derived supervision，不回写和覆盖冻结 V2.10 base release。

## 6.2 Teacher 的角色

Teacher 用于解决 deterministic rules 和公共 supervision 无法稳定解决的语义歧义，例如：

```text
某段代码究竟承担 fault_logic 还是 dependency_context；
哪些 Evidence 必须联合才能满足某个 obligation；
哪些不同 witness path 可以互相替代。
```

Teacher 不属于最终部署模型，也不能在 online rollout 中出现。

## 6.3 Experiment eligibility

Teacher-integrated policy audit 最终发现 5 个任务存在：

```text
POLICY_POSITIVE_UNSCOREABLE
```

这些任务保留 provenance，但全局：

```text
experiment_eligible = false
```

因此：

```text
Total tasks      = 20,864
Experiment valid = 20,859
Excluded         =      5
```

这 5 个任务不进入正式训练/评价指标。

---

# 7. Final Experiment Bundle

最终面向训练与 rollout 的自包含数据包为：

```text
data/evidence_agent_dataset_v1/
├── tasks.parquet
├── policy_evidence.parquet
├── repository_runtime.sqlite3
└── manifest.json
```

四个文件职责明确：

## `tasks.parquet`

包含：

```text
merged split
Teacher-integrated supervision
policy states/actions
experiment_eligible
provenance
```

## `policy_evidence.parquet`

包含所有最终任务中实际引用的稳定 Evidence 内容，用于：

```text
training
offline validation
offline benchmark
```

当前行数：

```text
998,682
```

## `repository_runtime.sqlite3`

保存完整 online runtime repository / FTS 结构，用于真正从 `K=∅` 开始的 Agent rollout。

## `manifest.json`

保存：

```text
hashes
counts
versions
training_ready
provenance
```

训练/验证只需要：

```text
tasks.parquet
policy_evidence.parquet
manifest.json
```

在线 rollout 额外需要：

```text
repository_runtime.sqlite3
```

---

# 8. Retriever 与 Canonical Structure

## 8.1 Candidate generation

Retriever 负责：

> 正确 Evidence 是否能够进入当前候选池？

而 Policy 负责：

> 候选已经可达以后，当前状态应该选择哪个动作？

因此不应使用 Policy ranking 指标替代 Retriever coverage，也不能用 Retriever recall 代表 Agent 最终能力。

## 8.2 Canonical structure

结构扩展只允许读取：

```text
Current K
+
Canonical pre-fix repository structure
```

禁止读取：

```text
Gold witness
Teacher answer
Gold Patch
Test Patch
```

当前主要 1-hop relation：

```text
parent
child
previous
next
```

V2.10 clean audit 中：

```text
lexical-only boundary coverage  ≈ 45.68%
+ canonical 1-hop               ≈ 55.04%
```

这说明结构扩展确实能够提高 state-conditioned candidate reachability，但该指标仍然只是 Retriever / dataset audit，不是训练后的 Policy performance。

---

# 9. Evidence Policy Model

## 9.1 Cross-Encoder formulation

核心模型：

> Cross-Encoder Evidence Policy Ranker

评分函数：

\[
s_A=f_\theta(q,K,A)
\]

输入：

```text
Issue q
Current Evidence K
Candidate Action A
```

动作空间：

```text
[u]       Single Evidence
[u,v]     Pair Evidence
STOP      Stop acquisition
```

## 9.2 Unified action space

Single、Pair、STOP 共用同一 backbone 和同一 scoring space。

项目不训练：

```text
独立 STOP classifier
独立 Pair selector
第二套 Retriever neural model
```

因此模型必须统一比较：

```text
下一条 Evidence 的价值
vs
两条联合 Evidence 的价值
vs
继续探索已经没有必要
```

## 9.3 Backbone 与 token contract

当前 baseline initialization 使用：

```text
BAAI/bge-reranker-v2-m3
revision = 953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

输入契约：

```text
model_max_length    = 4096
question_max_tokens = 2048
```

Evidence 内容不允许静默截断。如果完整 `(q,K,A)` 超过可评分上限，则：

```text
scoreable = false
action_loss_mask = false
```

相关记录仍保留用于审计。

## 9.4 Training objective

当前采用 multi-positive listwise softmax：

\[
L=\log\sum_{a\in A}e^{s_a}
-\log\sum_{a\in A^+}e^{s_a}
\]

一个 state 可以存在多个 positive action，因为序列证据获取并不要求下一步唯一。

训练状态 active 条件包括：

```text
ranking_loss_mask = true
```

训练动作 active 条件包括：

```text
action_loss_mask = true
scoreable = true
有合法 positive / negative supervision
```

---

# 10. Stage 1：基础 Policy Training

Stage 1 使用 Teacher-integrated final experiment bundle 中的 Initial / Boundary / Complete 状态进行基础训练。

Train active states：

```text
Initial   = 14,964
Boundary  =  6,509
Complete  = 14,964
Total     = 36,437
```

Validation active states：

```text
Initial   = 196
Boundary  = 109
Complete  = 196
Total     = 501
```

Stage 1 共训练 3 epochs。

## 10.1 Aggregate result

Stage 1 最佳 overall validation MRR：

```text
best_validation_mrr ≈ 0.8841
```

Epoch 3 overall validation：

```text
Hit@1 ≈ 0.7745
MRR   ≈ 0.8835
```

如果只看 aggregate metric，模型似乎已经具有不错的 action-ranking 能力。

## 10.2 分状态分析暴露核心瓶颈

Epoch 3 validation：

| State | Hit@1 | MRR | STOP Accuracy |
|---|---:|---:|---:|
| Initial | 94.39% | 0.9719 | 99.49% |
| Complete | 88.27% | 0.9360 | 88.27% |
| Boundary | **27.52%** | **0.6300** | **29.36%** |

关键发现：

> **Aggregate ranking 指标会被 Initial / Complete 这些相对容易的状态显著抬高，从而掩盖真正决定 Agent 多步获取能力的 Intermediate Boundary failure。**

这意味着：

```text
模型会开始
+
模型会在完整状态 STOP

并不等价于

模型会在中间轨迹中持续选择正确 Evidence
```

特别是 Boundary STOP accuracy 约 29.36%，说明 premature STOP 是必须重点研究的问题。

---

# 11. 原始 Decision Boundary 的结构性问题

进一步检查 frozen policy builder 后发现，原始 Boundary 并不是自然 rollout 中所有 Intermediate states 的代表。

其核心构造逻辑接近：

```text
minimum sufficient certificate
→ 删除一条 Evidence
→ 选择 closest-to-complete state
```

因此原始 Boundary 高度偏向：

```text
near-complete
```

而真实 sequential Agent 应经历：

```text
K={}
↓
K={e1}
↓
K={e1,e2}
↓
K={e1,e2,e3}
↓
...
↓
Complete
```

即：

```text
Early
Mid
Late
Near-complete
```

都属于实际重要状态。

由此形成第二个核心实验发现：

> **Near-complete-only Decision Boundary supervision 不能充分表示真实 Evidence Acquisition trajectory 的状态分布。**

这不是单纯“多训练几个相同样本”可以解决的问题，因此 `boundary-repeat` 只能作为 oversampling diagnostic，而不能替代新的 K-state reconstruction。

---

# 12. Multi-stage Decision Boundary Reconstruction

项目基于现有 Teacher / witness / corpus，在不重新采集数据、不修改 frozen V2.10 的前提下构造：

```text
data/evidence_agent_multistage_boundary_v1/
```

该 bundle 专门服务 Stage 2 Boundary fine-tuning。

## 12.1 Train

总 Boundary states：

```text
10,907
```

分布：

| Stage | Count | Ratio |
|---|---:|---:|
| Early | 1,105 | 10.1% |
| Mid | 5,173 | 47.4% |
| Late | 3,901 | 35.8% |
| Near-complete | 728 | 6.7% |

## 12.2 Validation

总 Boundary states：

```text
200
```

分布：

| Stage | Count |
|---|---:|
| Early | 22 |
| Mid | 95 |
| Late | 69 |
| Near-complete | 14 |

Audit：

```text
PASS
error_count = 0
training_ready = true
```

## 12.3 Stage definition

基于当前 evidence-progress：

```text
early          progress < 0.35
mid            progress < 0.70
late           progress < 0.90
near_complete  progress >= 0.90
```

Multi-stage reconstruction 的目标不是“人为把每个任务复制 6 次”，而是从可满足证书形成真实可区分的 Prefix states。

由于不同任务 minimum sufficient certificate 长度不同，很多任务天然只能产生 1–2 个合法 intermediate prefixes，因此最终平均 boundary/task 小于 max cap 是正常现象。

---

# 13. Stage 2：Multi-stage Boundary Fine-tuning

Stage 2 从 Stage 1 best checkpoint 初始化：

```text
models/evidence_policy_v1_0/best
```

主要配置：

```text
epochs              = 2
learning_rate       = 5e-6
weight_decay        = 0.01
warmup_ratio        = 0.06
max_candidates      = 12
pair_negative_quota = 3
grad_accum_steps    = 8
boundary_repeat     = 1
```

Stage 2 train/validation supervision 是 Boundary-only：

```text
Train      = 10,907 multi-stage Boundary states
Validation =    200 multi-stage Boundary states
```

当前状态：

> **Stage 2 已进入 Epoch 2/2，训练仍在进行中；最终 validation、best checkpoint 和最终 benchmark 结果尚未产生。**

因此不能把训练 loss 或训练集 Hit@1 写成最终泛化结论。

Stage 2 的目标是：

```text
让模型专门适应 Early / Mid / Late / Near-complete 中间状态
```

而不是重新学习基础 Initial / Complete。

这也意味着必须额外执行 retention evaluation，检查 Stage 2 是否造成 Initial / Complete catastrophic forgetting。

---

# 14. 公平模型比较协议

Stage 1 与 Stage 2 不能直接比较各自训练时使用的不同 validation set。

正式比较必须使用同一批：

```text
200 Multi-stage Validation states
```

比较：

```text
Stage1 best
vs
Stage2 best
```

核心指标：

```text
Hit@1
MRR
STOP Accuracy
Premature STOP
by-stage Early / Mid / Late / Near-complete metrics
Single / Pair / STOP top-action distribution
```

只有在同分布、同候选集、同 evaluator 上比较，才能把差异归因于 Multi-stage fine-tuning。

---

# 15. Retention Evaluation

Stage 2 是 Boundary 专项训练，因此可能提升 intermediate decision 的同时损伤 Initial / Complete。

Stage 2 best 产生后必须回测原始 501-state validation：

```text
Initial  = 196
Boundary = 109
Complete = 196
```

重点关注：

```text
Initial Hit@1 / MRR
Complete Hit@1 / MRR
Complete STOP Accuracy
```

最终模型选择不能只看 Boundary gain，还必须满足：

\[
Boundary\ Improvement
+
No\ Material\ Initial/Complete\ Regression
\]

如果 easy states 明显退化，再考虑 replay / anchor mixing，例如：

```text
60–70% Boundary
15–20% Initial
15–20% Complete
```

但只有 retention 实验确实发现退化后才需要做，不能预先增加不必要复杂度。

---

# 16. Frozen Benchmark

Benchmark 始终不能参与：

```text
training
early stopping
checkpoint selection
hyperparameter tuning
threshold selection
```

当前 benchmark 保持未用于模型选择。

正式顺序必须是：

```text
Stage 2 complete
↓
Same-200 validation comparison
↓
Original-validation retention
↓
Final model frozen
↓
Benchmark evaluation
```

只有冻结最终模型后才能正式打开 benchmark 结果。

---

# 17. Agent Online Rollout

State-level ranking 仍然不是项目最终目标。

真正系统价值需要从：

\[
K_0=\emptyset
\]

开始执行：

```text
Round 1:
Retriever(q, K0)
→ Policy
→ A1
→ K1

Round 2:
Retriever(q, K1)
+ Canonical Structure(K1)
→ Policy
→ A2
→ K2

...

Round T:
Policy → STOP
```

最终输出 Evidence Package。

## 17.1 为什么 rollout 是必要的

即使 Boundary state-level Hit@1 很高，也不能保证 trajectory 成功。

原因包括：

```text
某一步错误选择会改变后续 K；
某一次 premature STOP 会直接终止整个 trajectory；
错误证据会引入新的噪声候选；
局部排名误差可以累积。
```

因此必须报告完整 rollout 指标。

## 17.2 核心 trajectory metrics

推荐至少包括：

```text
Sufficiency Success Rate
Trajectory Success Rate
Premature STOP Rate
Average Completion Steps
Median Completion Steps
Average Evidence Count
Average Evidence Tokens
Final Obligation Coverage
Over-Collection Rate
Redundant Evidence Rate
Hard-budget Termination Rate
```

最终论文最重要的证据链应是：

\[
State\text{-}level\ Improvement
+
No\ Catastrophic\ Forgetting
+
Trajectory\text{-}level\ Sufficiency\ Improvement
\]

---

# 18. Evaluation Framework

项目评价必须分成四层，不能混在一起。

## 18.1 Retriever / Reachability

回答：

> 正确 Evidence 是否进入在线候选池？

指标：

```text
Online Positive Coverage
Recall@K
File Universe Miss
Unit Miss
Pair Realizability
Channel Hit
Canonical Structure Increment
```

## 18.2 Policy / State-level Ranking

回答：

> 正确动作已经可达时，模型能否选对？

指标：

```text
Hit@1
MRR
NDCG
Single accuracy
Pair accuracy
STOP accuracy
Premature STOP
by-state-type metrics
by-boundary-stage metrics
```

## 18.3 Deterministic Sufficiency

基于 obligation / witness：

```text
Obligation Coverage
Witness Coverage
Minimum Certificate Completion
Critical Deficit Count
```

这是主要可重复的上下文充分性评价轨道。

## 18.4 Reference-Grounded Semantic Sufficiency

可使用：

```text
Issue
+
Agent Evidence Package
+
Gold Patch / Test Patch
```

由 evaluator 判断 Evidence 是否覆盖理解和支持该参考修复所需的关键语义。

该 semantic Judge 不能替代 deterministic metrics，也不能作为在线 Agent 的 oracle。

建议自动校准：

```text
关键证据删除
无关证据注入
跨任务证据错配
重复 Judge / 多 Judge 一致性
```

项目不依赖多人专家人工审查作为实验前提。

---

# 19. 核心创新结构

论文主贡献建议维持三项，避免创新点过度碎片化。

## Contribution 1：Reference-Grounded Structured Evidence Sufficiency

把软件修复检索目标从：

```text
相关代码 / Gold location recall
```

提升为：

```text
结构化修复信息需求是否被 Evidence Package 充分覆盖
```

核心包括：

```text
7-slot obligations
OR-of-AND witness semantics
reference-grounded evaluator boundary
```

## Contribution 2：Trajectory-Aware State-Conditioned Evidence Policy

策略学习：

\[
f_\theta(q,K,A)
\]

而不是：

\[
f(q,u)
\]

并通过：

```text
Initial / Intermediate / Complete state semantics
Early / Mid / Late / Near-complete Multi-stage Boundary
Sufficiency-aware STOP
```

学习真实 sequential acquisition 中的决策。

## Contribution 3：Complementarity- and Substitutability-Aware Evidence Acquisition

利用：

```text
OR-of-AND witness
Single / Pair / STOP unified action space
state-dependent marginal value
```

显式建模：

```text
complement
substitute
redundant
```

避免把所有 Evidence 当作互相独立的相关文档。

### Multi-stage Boundary 是否单列为第四贡献？

建议当前放在 Contribution 2 内。

如果最终实验满足：

```text
same-200 Stage1 → Stage2 显著提升
+
Initial/Complete retention 稳定
+
rollout premature STOP / sufficiency 明显改善
```

则可以在论文最终版本中将：

> Multi-stage Decision Boundary Reconstruction

升级成独立第四贡献。

---

# 20. Agent / Multi-Agent 层的定位

Multi-Agent 不应成为“为了创新而增加多个 LLM”的包装。

推荐职责：

```text
Problem Analysis
Retrieval
Structure Exploration
Evidence Policy
Coordinator / Budget / Trace
```

核心训练模型仍然只有 Evidence Policy。

可选系统增强：

> State-Adaptive Repository Exploration

根据当前 `q + K` 动态选择：

```text
global lexical retrieval
path / symbol retrieval
canonical structure expansion
caller/callee exploration
local neighborhood exploration
STOP
```

如果未实现，该点只应写成后续系统增强，不能写成已完成论文结果。

---

# 21. 核心消融与对比实验

考虑单人项目的工程约束，优先级建议如下。

## A1. Original Boundary vs Multi-stage Boundary

```text
near-complete-only supervision
vs
Early + Mid + Late + Near-complete
```

回答：

> 多阶段中间状态监督是否必要？

## A2. Stage 1 best vs Stage 2 best

在同一 200 Multi-stage Validation 上比较。

这是最直接、最关键的实验。

## A3. Stateless vs State-aware

```text
Score(q,u)
vs
Score(q,K,u)
```

回答：

> Current Evidence State 是否真正改变候选价值？

## A4. Single-only vs Single + Pair

回答：

> 显式 Pair action 是否改善互补 witness 的完成率？

## A5. Fixed-step / No learned STOP vs Learned STOP

回答：

> 自适应停止是否减少 premature / over-collection？

## A6. Lexical-only vs Canonical Structure

回答：

> Repository structure 是否提高后续 evidence reachability 和 trajectory sufficiency？

## A7. Static one-shot RAG vs Iterative Evidence Agent

比较：

```text
one-shot Top-K context
vs
iterative q+K conditioned acquisition
```

这是面向最终系统价值的重要 baseline。

---

# 22. 当前开发状态

截至 2026-08-22：

| 模块 | 状态 |
|---|---|
| Research scope / project boundary | ✅ 已冻结 |
| Unified task identity / split | ✅ 已完成 |
| Pre-fix repository corpus | ✅ 已完成 |
| Evidence Unit extraction | ✅ 已完成 |
| Frozen V2.10 base dataset | ✅ 已完成 |
| Strong Teacher freeze | ✅ 已完成 |
| 7-slot obligations | ✅ 已完成 |
| OR-of-AND witnesses | ✅ 已完成 |
| Teacher-integrated derived data | ✅ 已完成 |
| Final 4-file experiment bundle | ✅ 已完成 |
| Leakage / integrity audit | ✅ 已完成 |
| Stage 1 Cross-Encoder training | ✅ 已完成 |
| Stage 1 validation diagnosis | ✅ 已完成 |
| Multi-stage Boundary reconstruction | ✅ 已完成 |
| Multi-stage bundle audit | ✅ PASS |
| Stage 2 Boundary fine-tuning | 🔄 进行中（Epoch 2/2） |
| Same-200 Stage1 vs Stage2 evaluation | ⏳ 待完成 |
| Original validation retention | ⏳ 待完成 |
| Frozen benchmark | ⏳ 尚未正式使用 |
| Full Agent rollout | ⏳ 待完成 |
| Core ablations | ⏳ 待完成 |
| Final paper tables / analysis | ⏳ 待完成 |

---

# 23. 后续严格执行顺序

推荐不再继续扩张研究模块，按以下顺序闭环：

```text
1. 完成 Stage 2 Epoch 2
2. 选择 Stage 2 best
3. Stage1 best vs Stage2 best：同一 200 Multi-stage Validation
4. Stage2 best 回测原始 501 Validation，检查 Initial / Complete retention
5. 冻结最终 Policy checkpoint
6. 正式 Benchmark
7. 实现 K={} → STOP 的完整 Agent rollout
8. 计算 sufficiency / premature STOP / cost / over-collection
9. 完成核心消融
10. 汇总论文表格、曲线、案例分析
```

在完成第 3、4 步以前，不应因为训练集 loss 很低就宣布 Multi-stage 方法已经成功。

在完成第 7、8 步以前，不应因为 state-level MRR 很高就宣布 Agent trajectory 已经成功。

---

# 24. 论文结果解释原则

必须避免以下错误表述。

### 错误 1

```text
Overall MRR 高
→ Agent 已会收集完整 Evidence
```

不成立。Stage 1 已证明 aggregate metric 会掩盖 Boundary failure。

### 错误 2

```text
Training Boundary Hit@1 高
→ Multi-stage 泛化好
```

不成立。必须看独立 200-state validation。

### 错误 3

```text
Boundary state-level accuracy 高
→ rollout 一定成功
```

不成立。trajectory 中错误会累积，premature STOP 具有终止性。

### 错误 4

```text
Gold Patch 用于 evaluator
→ online leakage
```

不成立，只要 Gold Patch 严格在 Agent 完成后进入 offline evaluator，并且不影响候选、Policy 或 routing。

### 错误 5

```text
项目没有生成 patch
→ 无法评价证据是否正确
```

不成立。项目使用 deterministic obligation/witness coverage + reference-grounded semantic evaluation 两条轨道评估上下文充分性。

---

# 25. 复现与事实优先级

项目文档中的事实优先级：

```text
Frozen / final Manifest
        ↓
Current builder / trainer / audit code
        ↓
Generated audit reports
        ↓
Training logs / checkpoint metadata
        ↓
Research design documents
```

设计文档与实际实现冲突时，应以代码、manifest 和 audit 为准。

特别注意：

```text
Frozen V2.10 base release
Teacher-integrated final bundle
Multi-stage derived bundle
```

是三个不同层级，统计数据不可直接互换。

---

# 26. 推荐论文主线

最终论文不应写成：

> “我们设计了一个新的代码 reranker。”

更完整的逻辑应是：

```text
软件修复需要多类信息
↓
Gold location 不等于 repair-context sufficiency
↓
定义 7-slot Structured Evidence Obligations
↓
用 OR-of-AND witness 表达替代与互补
↓
把检索改写成 sequential evidence acquisition
↓
用 q + K + A 的统一 Evidence Policy 决策 Single / Pair / STOP
↓
Stage 1 发现 aggregate metric 掩盖 Intermediate failure
↓
进一步发现 near-complete-only Boundary 与真实 trajectory 不匹配
↓
提出 Multi-stage Decision Boundary Reconstruction
↓
训练 Early / Mid / Late / Near-complete states
↓
同分布验证 + retention
↓
从 K={} 完整 rollout
↓
评价 sufficiency、premature STOP、cost 与 over-collection
```

这条链条同时包含：

```text
Problem formulation
Representation
Policy learning
Training distribution correction
Evaluation methodology
Agent-level verification
```

---

# 27. FAQ

## Q1：这是 RAG 项目吗？

是，但不是普通 one-shot RAG。

```text
RAG = candidate reachability
Policy = state-conditioned next-action selection
Agent = iterative acquisition trajectory
```

## Q2：为什么不直接让大模型读完整仓库？

因为完整仓库 token 成本高、噪声大、难以审计，也无法清楚研究哪条 Evidence 产生了增量价值。

## Q3：为什么需要 Pair action？

因为 OR-of-AND witness 中存在“单条不足、联合充分”的真实结构。

## Q4：为什么 STOP 不能只是固定 Top-K？

固定 Top-K 无法根据不同 Issue 的证据需求自适应停止，也无法研究 premature STOP 与 over-collection。

## Q5：为什么 Multi-stage Boundary 很重要？

真实 Agent 大多数决策发生在“已有部分证据但尚未充分”的 Intermediate states。只训练 near-complete 状态会造成状态分布失配。

## Q6：Stage 2 训练 loss 很低是否代表已经成功？

不是。训练 loss 只能说明优化过程拟合当前训练分布，必须等待同一 200-state validation 和最终 rollout。

## Q7：为什么还要回测 Initial / Complete？

因为 Stage 2 是 Boundary-only fine-tuning，可能产生 catastrophic forgetting。

## Q8：Benchmark 为什么不能现在直接看？

因为 benchmark 不能参与模型选择或超参数调节，否则失去严格的最终泛化评价意义。

## Q9：项目最终会输出 patch 吗？

当前不会。最终输出 Evidence Package。

## Q10：Agent 层是否需要很多额外创新？

不需要。核心研究价值应集中在 evidence sufficiency、state/trajectory-aware policy 和 evidence interaction。Agent runtime 只需要可靠地完成多轮执行与评价。

---

# 28. Summary

Evidence Agent 的研究目标可以压缩为：

> **在严格 pre-fix、无 Gold 在线泄漏的条件下，将软件修复上下文获取建模为一个结构化、状态条件化的序列证据充分性问题；利用统一 Cross-Encoder Policy 在 Single、Pair 和 STOP 动作之间决策，并通过 Multi-stage Boundary supervision 与完整 rollout 验证 Agent 能否从零逐步获得充分而不过量的修复证据。**

当前最重要的工作不再是增加新模块，而是完成三个实验闭环：

```text
1. Same-state distribution improvement
2. No catastrophic forgetting
3. Full trajectory sufficiency improvement
```

只有这三部分同时成立，项目才可以从一个强 reranking / retrieval 系统完整升级为：

> **Agentic Software Repair Evidence Acquisition / Context Sufficiency Framework**

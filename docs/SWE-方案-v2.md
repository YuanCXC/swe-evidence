# Gather, Combine, or Skip：最新执行方案 v2

## 一、项目最终定位

项目继续围绕三个闭合问题展开：

1. 当前证据集合 (K) 对固定修复器 (G) 有多大修复价值；
2. 候选证据与已有证据之间是互补、替代、冗余、独立还是冲突；
3. 候选证据的条件边际收益是否值得其获取成本，以及何时停止探索。

最终三项贡献保持为：

* **修复价值导向的证据获取任务**；
* **Repair Value-of-Information Policy**；
* **证据互补与替代感知建模**。

该结构与创新点文件中“行为价值、信息价值、证据交互”的收敛方向一致。 原执行方案中的 patch 映射、质量证书、反事实删除和固定修复器评测继续保留，但作为数据构造与验证机制，而不是平行创新。

核心定义不变：

[
V_G(K)=P_G(\text{repair succeeds}\mid q,K)
]

[
\Delta_G(u\mid K)=V_G(K\cup{u})-V_G(K)
]

[
Q_G(u\mid K)=\Delta_G(u\mid K)-\lambda Cost(u)
]

证据交互值：

[
I_G(u,v\mid K)
==============

V_G(K+u+v)-V_G(K+u)-V_G(K+v)+V_G(K)
]

---

# 二、v2 方案的主要调整

旧方案倾向于从 SWE-bench 原始数据开始，自行构造几乎全部静态证据。v2 调整为：

```text
现成高质量上下文与轨迹数据
        ↓
统一实例、代码单元和 provenance
        ↓
静态交互弱监督与模型预训练
        ↓
只对高价值子集运行反事实修复实验
        ↓
生成真正独有的 VOI、交互和 STOP 标签
```

具体变化如下。

### 1. 不再重复生产已有标签

ContextBench 已提供 1,136 个任务的人工 gold context，覆盖 66 个仓库和 8 种语言；字段直接包含 `base_commit`、文件、行范围和代码内容，并有 500 个实例的 Verified 子集。

SWE-Explore 已提供 848 个任务的 line-level core/optional context、读取步骤和成功轨迹 provenance，覆盖 203 个仓库和 10 种语言。

因此，不再从 patch 单独推导全部“相关上下文”，而是将：

* ContextBench 作为人工 gold context；
* SWE-Explore 作为 trajectory-grounded context；
* patch mapping 作为 edit-oriented context；
* 行为反事实结果作为 causal repair value。

四种监督严格分开。

### 2. CORE-Bench 降为可选预训练源

CORE-Bench 提供三个层级的代码检索任务，包含超过 18 万个查询和约 10.6 万个 broader-context relevance labels，适合 retriever 预训练。

但它不是项目主数据源。只有完成下载、字段和许可证 smoke test 后才接入；不能让其阻塞 ContextBench、SWE-Explore 和 SWE-bench 主线。

### 3. 行为反事实成为主要新增数据资产

公开数据仍未提供在同一冻结修复器下的：

```text
K
K + u
K + v
K + u + v
```

对应的统一执行结果。

因此，项目真正需要自行产生的是：

```text
V_G(K)
Δ_G(u | K)
I_G(u,v | K)
Sub_G(u→v | K)
Cost-aware STOP
```

---

# 三、数据源分工

| 数据源              | 在项目中的角色                              |             可见性 |  是否作为行为真值 |
| ---------------- | ------------------------------------ | --------------: | --------: |
| SWE-bench        | issue、repo、base commit、patch、测试和执行环境 |              混合 | 是，限测试执行结果 |
| ContextBench     | 人工 gold context、span 映射校准            |      label-only |         否 |
| SWE-Explore      | core/optional context、读取步骤和轨迹来源      |      label-only |         否 |
| SWE-bench BM25   | 固定检索基线与候选池基线                         |        baseline |         否 |
| SWE-bench Oracle | 检索上界与诊断候选池                           | diagnostic-only |         否 |
| Agentless        | 预处理仓库结构、层级定位结果和 baseline             |              辅助 |         否 |
| LocAgent         | 依赖图生成与图搜索 baseline                   |              辅助 |         否 |
| CORE-Bench       | retriever 外部预训练                      |      train-only |         否 |
| 自建行为数据           | 固定修复器的反事实执行结果                        |      label-only |     **是** |

SWE-bench 官方数据包含 `repo`、`base_commit`、`problem_statement`、patch、test patch、FAIL_TO_PASS 和 PASS_TO_PASS 等字段；官方也提供 Oracle 和 BM25 预处理检索版本。

Agentless 已发布完整 Lite、Verified 运行结果及每个 SWE-bench 问题的预处理仓库结构。 LocAgent 提供将代码库转换为有向异构图的实现，可直接用于构造 contains、calls 和依赖边。

---

# 四、最关键的数据划分方案

ContextBench 和 SWE-Explore 都包含 SWE-bench Verified 实例。如果直接用其标签训练，再在相同实例上做最终评测，会产生任务级监督泄漏。

因此必须先统一实例，再冻结划分。

## 1. 建立 Master Instance Registry

统一主键：

```text
canonical_instance_id
repo
base_commit
original_instance_id
issue/pr identity
normalized_problem_hash
patch_target_signature
source_datasets
```

映射优先级：

```text
original_instance_id exact match
→ repo + base_commit
→ issue/pr URL
→ normalized problem statement
→ patch target signature
```

同一任务在不同数据源中的记录必须合并为同一个 `task_group_id`。

## 2. 冻结四个集合

### Static Train

来源：

* SWE-bench 官方 train；
* ContextBench 中不属于最终评测池的实例；
* SWE-Explore 中不属于最终评测池的实例；
* 可选 CORE-Bench train 数据。

用途：

```text
retriever 训练
静态 interaction warm-start
weak STOP
evidence role
```

### Dev

从可执行且映射稳定的实例中冻结一批，用于：

```text
λ 成本权重
τ STOP 阈值
δc / δs / δn 关系阈值
候选池大小
模型选择
```

### Test-Retrieval

用于独立评估上下文获取：

* ContextBench 未用于训练的人工 gold context；
* SWE-Explore 未用于训练的 line-level core context。

ContextBench 当前公开数据使用单一 split，因此必须由项目自行划分，不能把 Hugging Face 的 `train` 名称理解为模型训练集。

### Test-Behavior

从 SWE-bench Verified 中冻结可执行实例，用于：

```text
fixed-generator repair success
VOI
interaction
adaptive STOP
success–cost Pareto
```

这些实例对应的 ContextBench、SWE-Explore 标签不能进入训练，只能用于最终诊断。

## 3. 必须同时报告两种泛化

```text
Task-unseen：
测试任务从未进入训练

Repo-unseen：
测试仓库从未进入训练
```

若行为测试难以做到 repo-unseen，则行为主结果报告 task-unseen，并增加独立的 repo-unseen retrieval track。

---

# 五、统一数据目录

```text
data/
├── raw/
│   ├── swebench/
│   ├── contextbench/
│   ├── swe_explore/
│   ├── corebench/
│   ├── agentless/
│   └── locagent/
│
├── registry/
│   ├── source_manifest.jsonl
│   ├── instance_aliases.jsonl
│   ├── master_instances.jsonl
│   ├── overlap_report.json
│   └── master_split_v2.json
│
├── cache/
│   ├── repo_cache/
│   ├── graph_indexes/
│   ├── generator_runs/
│   ├── evaluation_runs/
│   └── llm/
│
├── processed/
│   ├── normalized/
│   ├── patches/
│   ├── visible_units/
│   ├── mappings/
│   ├── certificates/
│   ├── external_labels/
│   ├── structural_graphs/
│   ├── interaction_candidates/
│   ├── interaction_labels/
│   ├── static_samples/
│   ├── behavior_packets/
│   ├── behavior_outcomes/
│   ├── value_labels/
│   ├── behavior_samples/
│   ├── human_annotations/
│   └── reports/
│
└── releases/
    ├── metadata_only/
    └── reproducibility_manifests/
```

SWE-Explore 的数据集许可证标记为 CC BY-NC-ND 4.0，因此不能默认公开发布转换后的派生数据。 对外发布时优先发布：

```text
instance ID
unit ID
关系标签
哈希
构造脚本
```

而不重新分发原始文本。

---

# 六、统一数据对象

## 1. Evidence Unit

```json
{
  "unit_id": "...",
  "canonical_instance_id": "...",
  "repo": "owner/repo",
  "base_commit": "...",
  "unit_type": "function",
  "file_path": "src/a.py",
  "symbol": "Class.method",
  "start_line": 30,
  "end_line": 57,
  "raw_content": "...",
  "content_hash": "...",
  "token_count": 280,
  "line_count": 28,
  "provenance": "base_commit",
  "visibility": "visible"
}
```

## 2. Supervision Annotation

```json
{
  "canonical_instance_id": "...",
  "unit_id": "...",
  "label_type": "gold_context",
  "label_value": 1,
  "label_source": "contextbench",
  "confidence": 1.0,
  "supervision_tier": "human_context",
  "visibility": "label_only"
}
```

`label_source` 可取：

```text
patch_mapping
contextbench
swe_explore_core
swe_explore_optional
structural_rule
llm_relation
human_relation
behavior_counterfactual
```

## 3. Interaction Edge

```json
{
  "unit_u": "...",
  "unit_v": "...",
  "relation": "complement",
  "direction": "symmetric",
  "confidence": 0.84,
  "label_sources": [
    "structural_rule",
    "trajectory_consensus"
  ],
  "behavior_verified": false
}
```

## 4. Behavior Packet

```json
{
  "packet_id": "...",
  "canonical_instance_id": "...",
  "base_state_id": "K2",
  "selected_unit_ids": [],
  "intervention": {
    "type": "pair_addition",
    "u": "...",
    "v": "..."
  },
  "input_hash": "...",
  "generator_config_hash": "..."
}
```

---

# 七、最新脚本顺序

保留当前已有的 02–04 脚本，新增统一数据接入层。

## A. 数据盘点与隔离

```text
00_fetch_external_datasets.py
01_build_source_manifest.py

02_normalize_swebench.py
03_parse_gold_patches.py
04_extract_patch_supervision_llm.py

05_build_master_instance_registry.py
06_audit_overlap_and_freeze_splits.py
07_validate_source_licenses.py
```

`source_manifest` 必须保存：

```text
source URL/reference
download date
revision/commit
file hash
dataset license
schema version
row count
```

## B. 导入已有处理标签

```text
08_import_contextbench_labels.py
09_import_swe_explore_labels.py
10_import_retrieval_assets.py
11_import_agentless_structures.py
```

其中 `10_import_retrieval_assets.py` 负责：

```text
SWE-bench BM25
SWE-bench Oracle
CORE-Bench（通过 smoke test 后）
```

Oracle 始终标为：

```text
visibility = diagnostic_only
```

## C. 修复前仓库和代码单元

```text
12_index_base_repositories.py
13_build_visible_units.py
14_map_external_spans_to_units.py
15_map_patch_to_visible_units.py
16_build_mapping_certificates.py
```

构造粒度：

```text
file
class
function/method
branch
statement block
callsite
existing test
configuration section
documentation section
semantic summary
```

第一版只强制实现：

```text
file
function/method/class
branch/block
existing test
```

所有 unit 必须重新从 `base_commit` 验证。即使 ContextBench 已提供代码内容，也不能跳过 commit-level checksum 检查。

## D. 图和交互弱监督

```text
17_build_structural_graphs.py
18_build_interaction_candidates.py
19_build_rule_based_interaction_labels.py
20_adjudicate_interactions_llm.py
21_build_static_nips_samples.py
```

结构图优先构造：

```text
contains
calls
called_by
imports
defines
reads_state
writes_state
raises
handles
test_covers
```

语义关系：

```text
complement
substitute
redundant
independent
conflict
uncertain
```

---

# 八、如何利用现成数据构造交互弱标签

## 1. 互补弱标签

候选同时满足以下部分条件：

```text
都属于 ContextBench gold context
但承担不同 evidence role

都属于 SWE-Explore core regions
且在成功轨迹中被读取

存在 caller–callee、producer–consumer、
state-write–state-read 或 exception–handler 关系

单条证据只覆盖部分 patch-derived role
组合后覆盖完整 role closure
```

例如：

```text
buggy function
+
caller/test constraint
→ complement candidate
```

注意：

> 共同出现在成功轨迹中只是互补候选生成依据，不是互补行为真值。

## 2. 替代弱标签

高精度规则：

```text
statement ↔ containing branch
branch ↔ containing function
原始代码 ↔ 对应语义摘要
高度重叠的 line regions
同一 assertion 的代码与测试表达
同一 symbol 的多个重复窗口
```

还需满足：

```text
evidence role 相同
或表达相同约束
```

不能仅凭 embedding 相似度判定替代。

## 3. 独立和 hard negative

重点构造：

```text
高文本相似但不同 symbol
同文件但无依赖关系
同一通用 API 的其他调用点
相同 evidence role 但针对其他行为
SWE-Explore optional context 中的模型特有读取
```

## 4. LLM 的位置

LLM 只用于裁决：

```text
结构规则冲突
role 不明确
跨文件语义关系
代码和测试是否表达同一约束
```

输出必须包含原文 quote，并通过程序检查。LLM 标签始终为弱监督。

---

# 九、静态 MVP

行为实验成本较高，因此先证明 interaction-aware acquisition 在静态监督上成立。

## 1. 模型对比

至少训练：

```text
A. Independent Scorer
   s(q,u)

B. State-Aware Scorer
   s(q,K,u)

C. Interaction-Aware VOI Proxy
   s(q,K,u,relations(u,K))
```

## 2. 静态边际价值代理

在没有行为标签时定义：

[
\widetilde{\Delta}(u\mid K)
===========================

\text{新增 gold-role coverage}
+
\alpha\text{ complement proxy}
------------------------------

## \beta\text{ substitute proxy}

\lambda Cost(u)
]

其中新增 coverage 可来自：

```text
ContextBench 未覆盖 gold span
SWE-Explore 未覆盖 core region
patch-derived 未覆盖 evidence role
```

## 3. 弱 STOP

满足以下条件时标记 weak STOP：

```text
高置信 required roles 已覆盖
剩余候选主要为替代或冗余
剩余静态净收益 ≤ 0
```

它不能作为最终 STOP 真值，只用于 warm-start。

## 4. MVP 指标

```text
Gold Context Recall@B
Line Coverage@B
Context Precision
Redundant Evidence Rate
Substitute Duplication Rate
Complementary Pair Recovery
Typed Role Closure
Cost-adjusted NDCG
Weak STOP Accuracy
```

验收条件：

```text
Interaction-aware scorer
必须优于 independent scorer，

且提升不能只来自读取更多 token。
```

---

# 十、行为校准子集

静态 MVP 通过后，才启动固定修复器反事实实验。

## 1. 实例筛选

优先选择：

```text
SWE-bench Verified
环境可构建
FAIL_TO_PASS 可稳定复现
base commit 完整
mapping certificate 为 strong/partial
gold context 与 patch mapping 可对齐
存在至少一个互补或替代候选
patch 规模适中
```

建议三阶段扩展：

```text
Pilot：
少量实例，验证实验协议和信号

Core：
覆盖 train/dev/test 的主要行为数据

Expansion：
根据不确定性和关系缺口主动增加实例
```

不应一开始对全部实例运行大量组合。

## 2. 每个实例的候选对

优先保留：

```text
3 个高概率 complement pair
2 个高概率 substitute pair
2 个 independent pair
1–2 个随机对照 pair
```

## 3. 最小反事实设计

每个 pair 构造：

```text
K
K + u
K + v
K + u + v
```

关键证据再增加：

```text
K_full - u
K_full - v
K_full - {u,v}
```

这比对所有证据子集穷举更可控。

## 4. 主动实验设计

第一轮执行后，根据以下条件补充 packet：

```text
预测交互强但行为结果不确定
模型与弱标签分歧
置信区间过宽
替代关系方向不明确
STOP 边界附近
```

这样把修复器调用集中在信息量最大的样本上。

---

# 十一、固定修复器协议

固定修复器 (G) 必须冻结：

```text
模型名称和精确版本
系统 prompt
用户 prompt 模板
工具权限
最大输入和输出 token
temperature
采样次数
seed
超时
最大工具调用数
patch 格式
测试环境版本
```

输入只能包括：

```text
problem statement
pre-fix evidence packet
允许的工具定义
```

禁止包括：

```text
gold patch
test patch
修复后代码
ContextBench gold 标识
SWE-Explore core 标识
patch-derived fix intent
行为标签
```

每次运行保存：

```text
完整 prompt
模型响应
tool trajectory
generated patch
token usage
latency
异常
generator_config_hash
```

---

# 十二、行为价值估计

`resolved` 仍是 (V_G) 的主要结果：

[
V_G(K)=E[\mathbb{I}(\text{resolved})]
]

但单次成功/失败噪声较大，因此同时保存辅助结果：

```text
patch_applied
test_collection_success
FAIL_TO_PASS pass ratio
PASS_TO_PASS preservation ratio
resolved
```

## 1. 重复运行

```text
训练 packet：
至少一次，重点 packet 可追加

dev/test 核心 packet：
使用多个配对 seed

最终案例：
增加重复次数
```

配对 packet 尽量使用相同 seed 集合，以减少比较方差。

## 2. 平滑估计

不要直接把一次 0/1 当作精确概率。使用 Beta-Binomial 或经验贝叶斯平滑得到：

```text
posterior mean V_G(K)
credible interval
```

再计算：

[
\Delta_G(u\mid K)
]

[
I_G(u,v\mid K)
]

同时保存原始计数，确保结果可审计。

## 3. 关系标签

阈值由 dev 冻结：

```text
I > δc
→ complement

Δ(v|K) > δv
且 Δ(v|K+u) < ε
→ u substitutes v

I < -δn
→ conflict

其余
→ independent/uncertain
```

替代性保留方向：

```text
u substitutes v
不必等价于
v substitutes u
```

---

# 十三、行为模型训练

## Stage 1：Retriever Warm-start

训练来源：

```text
patch mapping
ContextBench gold context
SWE-Explore core regions
可选 CORE-Bench
```

采用：

```text
BM25
dense retrieval
hybrid retrieval
file-to-symbol hierarchical retrieval
```

## Stage 2：Static Interaction Warm-start

训练目标：

```text
set-valued action ranking
role coverage
complement/substitute classification
weak marginal value
weak STOP
```

## Stage 3：Behavior-Calibrated Fine-tuning

主要损失：

[
L =
L_{\Delta}
+\lambda_v L_V
+\lambda_a L_{\text{action}}
+\lambda_i L_{\text{interaction}}
+\lambda_s L_{\text{stop}}
]

其中：

```text
LΔ：
条件边际价值回归或 pairwise ranking

LV：
状态修复价值估计

Laction：
set-valued acceptable action loss

Linteraction：
行为 complement/substitute 标签

Lstop：
是否仍存在正净收益
```

行为标签较稀疏时，优先采用 pairwise ranking：

```text
Q(u|K) > Q(v|K)
```

而不是强行回归不稳定的小数值。

## Stage 4：STOP Calibration

只在 dev 上确定：

```text
状态价值阈值 τ
成本权重 λ
最小净收益阈值
temperature scaling
关系分类阈值
```

---

# 十四、模型结构

主模型由四部分组成：

```text
Issue Encoder
Evidence Encoder
State Set/Graph Encoder
Candidate–State Interaction Encoder
```

状态表示：

[
h_K=SetEncoder({h_{u_1},...,h_{u_t}})
]

候选与已选证据交互：

[
e_{u,i}=Interaction(h_u,h_{u_i})
]

[
h_{u,K}=Aggregate({e_{u,i}})
]

最终输出：

```text
V̂_G(K)
Δ̂_G(u|K)
Q̂_G(u|K)
P(complement)
P(substitute)
P(conflict)
P(STOP)
```

第一版优先使用：

```text
Set Transformer
+
显式结构特征
```

图 Transformer 作为增强或消融，不应成为 MVP 的工程阻塞点。

---

# 十五、端到端推理

```python
K = []

while cost(K) < budget:
    C = retriever.search(issue=q, state=K)

    state_value = policy.predict_state_value(q, K)

    scored = []
    for u in C:
        delta = policy.predict_marginal_value(q, K, u)
        net_value = delta - lambda_cost * evidence_cost(u)
        scored.append((u, net_value))

    best_u, best_q = max(scored, key=lambda x: x[1])

    if state_value >= tau and best_q <= 0:
        break

    K.append(best_u)

patch = fixed_generator.generate(q, K)
return K, patch
```

必须设置硬预算，避免 STOP 失效导致无限探索。

---

# 十六、最终评测分为三条轨道

## Track A：Retrieval and Context Acquisition

使用未参与训练的 ContextBench 和 SWE-Explore 标签。

指标：

```text
File/Symbol/Line Recall
Context Precision
Ranked Coverage@B
Redundancy
Evidence Drop
Core/Optional Coverage
```

ContextBench 原生关注 file、symbol、span 和 edit-location 粒度的上下文获取；SWE-Explore则在固定行预算下评估 coverage、ranking 和 efficiency。

## Track B：Behavior-Calibrated VOI

指标：

```text
State Value MAE/Brier
Marginal Value Spearman
Positive-gain Precision@k
Negative-gain Avoidance
Interaction-score correlation
Complement F1
Substitute F1
```

## Track C：End-to-End Repair–Cost

主图：

```text
横轴：
retrieved tokens / lines / tool calls / latency

纵轴：
fixed-generator resolved rate
```

主指标：

[
AUC_{\text{Repair-Cost}}
]

同时报告：

```text
resolved rate
mean evidence tokens
tool calls
premature STOP
late STOP overhead
substitute duplication
negative-marginal evidence rate
```

---

# 十七、基线

## Retrieval

```text
BM25
Dense Retriever
Hybrid Retriever
Cross-encoder Reranker
Agentless hierarchical localization
LocAgent graph localization
```

## Acquisition

```text
Fixed Top-1
Fixed Top-3
Fixed Top-5
Fixed token budget
Fixed number of rounds
Stateless scorer
State-aware scorer without interaction
```

## 方法消融

```text
Full model
- complement features
- substitute features
- structural graph
- behavior fine-tuning
- cost term
- state value head
- adaptive STOP
- external processed data
```

最重要对比仍然是：

```text
independent relevance scoring
vs
context-dependent conditional marginal value
```

---

# 十八、统计分析

所有主结果按实例配对。

报告：

```text
mean / median
95% bootstrap confidence interval
paired improvement
per-repository macro average
```

交互实验还需报告：

```text
I_G 的分布
正交互与负交互比例
不同证据角色组合的交互强度
弱标签与行为标签的一致率
```

如果二元 `resolved` 信号过稀疏：

```text
主结论仍使用 resolved
辅助训练使用测试通过比例
显著性分析采用配对成功差异
```

不能以辅助指标替换最终修复成功率。

---

# 十九、人工审计

人工标注不必覆盖所有样本，重点覆盖争议区域。

标注对象：

```text
证据角色
patch-to-unit 映射
complement/substitute/independent
证据包充分性
可删除证据
STOP 合理性
```

重点抽样：

```text
规则与 LLM 分歧
弱标签与行为标签分歧
高相关但低 VOI
预测 complement 但行为无增益
预测 substitute 但组合仍有增益
错误 STOP
```

报告：

```text
双人一致率
Cohen’s κ 或 Krippendorff’s α
各监督来源 precision
certificate 与人工正确率关系
```

---

# 二十、泄漏与复现审计

新增统一脚本：

```text
33_audit_leakage_and_reproducibility.py
```

检查：

```text
相同任务是否跨 split
同一 PR 的别名是否跨 split
gold context 是否进入 test 输入
patch added line 是否进入 visible unit
Oracle context 是否进入主评测
SWE-Explore core 标识是否进入 test 输入
LLM semanticizer 是否看到 patch
generator 是否看到 label-only 字段
dev/test behavior outcome 是否进入训练
仓库内容是否匹配 base_commit
```

目标：

```json
{
  "cross_split_task_leaks": 0,
  "gold_context_input_leaks": 0,
  "patch_line_leaks": 0,
  "oracle_input_leaks": 0,
  "behavior_label_leaks": 0,
  "commit_mismatches": 0,
  "status": "passed"
}
```

---

# 二十一、阶段验收门

## Gate 0：数据可用性

必须完成：

```text
所有来源可下载
revision 和哈希已冻结
许可证已记录
实例别名可映射
```

CORE-Bench 未通过下载和 schema 检查时直接跳过，不阻塞主线。

## Gate 1：统一代码单元

目标：

```text
base commit 验证通过
外部 span 可映射到 unit
patch mapping 可审计
无修复后代码泄漏
```

## Gate 2：静态 MVP

必须观察到：

```text
interaction-aware > independent scorer
substitute duplication 下降
complement recovery 提升
成本没有显著恶化
```

未通过时，不启动大规模行为实验。

## Gate 3：行为 Pilot

必须确认：

```text
不同 packet 的结果存在可测差异
至少部分 pair 呈正交互
至少部分 pair 呈边际收益衰减
STOP 边界可校准
```

如果行为差异过于稀疏，应增加重复运行或重新选择实例，而不是扩大低质量样本。

## Gate 4：论文核心

最终需支持：

```text
1. 相关性不能替代修复价值；
2. 候选价值依赖当前证据状态；
3. 部分证据需要联合获取；
4. 部分相关证据可以相互替代；
5. 条件边际价值改善 success–cost Pareto；
6. adaptive STOP 优于固定 Top-k。
```

---

# 二十二、最终脚本清单

```text
00_fetch_external_datasets.py
01_build_source_manifest.py
02_normalize_swebench.py
03_parse_gold_patches.py
04_extract_patch_supervision_llm.py
05_build_master_instance_registry.py
06_audit_overlap_and_freeze_splits.py
07_validate_source_licenses.py
08_import_contextbench_labels.py
09_import_swe_explore_labels.py
10_import_retrieval_assets.py
11_import_agentless_structures.py
12_index_base_repositories.py
13_build_visible_units.py
14_map_external_spans_to_units.py
15_map_patch_to_visible_units.py
16_build_mapping_certificates.py
17_build_structural_graphs.py
18_build_interaction_candidates.py
19_build_rule_based_interaction_labels.py
20_adjudicate_interactions_llm.py
21_build_static_nips_samples.py
22_train_retriever.py
23_train_static_interaction_policy.py
24_evaluate_static_mvp.py
25_select_behavior_subset.py
26_build_adaptive_counterfactual_packets.py
27_run_fixed_patch_generator.py
28_evaluate_generated_patches.py
29_compute_smoothed_value_labels.py
30_build_behavior_nips_samples.py
31_train_behavior_calibrated_policy.py
32_calibrate_stop.py
33_audit_leakage_and_reproducibility.py
34_run_evidence_agent.py
35_evaluate_retrieval_track.py
36_evaluate_behavior_track.py
37_evaluate_end_to_end.py
38_evaluate_failure_modes.py
39_generate_paper_tables.py
```

---

# 二十三、最终实施优先级

## 第一优先级：建立可信数据底座

```text
实例统一
重叠审计
split 冻结
ContextBench/SWE-Explore 导入
visible units
span mapping
```

## 第二优先级：完成静态交互 MVP

```text
结构图
interaction candidates
规则弱标签
独立 scorer
interaction-aware scorer
静态指标
```

## 第三优先级：产生独有行为数据

```text
固定修复器
反事实 packets
真实执行结果
V / Δ / I / substitute labels
```

## 第四优先级：VOI 和 STOP

```text
behavior fine-tuning
cost-aware selection
adaptive stopping
success–cost Pareto
```

## 第五优先级：扩展和论文验证

```text
第二修复器
repo-unseen
多语言
人工审计
失败模式
完整消融
```

---

# 二十四、v2 方案的最终原则

```text
ContextBench
不等于修复价值，
它提供人工上下文监督。

SWE-Explore
不等于因果证据，
它提供成功轨迹监督。

Patch mapping
不等于充分性，
它提供编辑位置监督。

LLM relation label
不等于行为真值，
它提供语义弱监督。

只有固定修复器的反事实执行
用于定义 V、Δ、I 和 STOP。
```

最终的数据闭环是：

```text
SWE-bench issue + base commit
           ↓
统一 Evidence Units
           ↓
人工 context + trajectory + patch weak labels
           ↓
Interaction-aware static policy
           ↓
主动选择反事实 packets
           ↓
Frozen generator + SWE-bench execution
           ↓
Behavior-calibrated VOI labels
           ↓
Cost-aware acquisition + adaptive STOP
           ↓
Repair–Cost Pareto evaluation
```



# 一、先确定最终要实现什么

系统输入为：

```text
问题描述 q
+ 修复前仓库 base_commit
+ 当前已读取证据集合 K
+ 剩余候选证据 C
```

系统循环选择下一条证据：

```text
文件
函数/方法
分支
语句块
调用点
测试
配置
文档
或 STOP
```

最终目标不是“检索到 patch 修改位置”，而是：

[
\max_\pi V_G(K_T)-\lambda Cost(K_T)
]

其中：

```text
G：冻结的下游修复器
V_G(K)：修复器在证据包 K 下成功修复的概率
Cost(K)：token、代码行、工具调用、时间和费用
```

核心预测量为：

[
\Delta_G(u\mid K)=V_G(K\cup{u})-V_G(K)
]

系统应优先选择净价值最大的证据：

[
Q_G(u\mid K)=\Delta_G(u\mid K)-\lambda Cost(u)
]

停止条件为：

[
\hat V_G(K)\ge\tau
\quad\land\quad
\max_{u\in C}\hat Q_G(u\mid K)\le\epsilon_q
]

这决定了整个工程顺序：**先建立可信证据，再训练检索器，再训练状态感知策略，最后用真实修复行为校准。**

---

# 二、从 0 开始的整体流水线

完整数据链应固定为：

```text
数据源规格
    ↓
下载与审计
    ↓
Master Instance Registry
    ↓
冻结 train/dev/test
    ↓
base_commit 仓库快照
    ↓
Evidence Units
    ↓
Patch / 外部 span 映射
    ↓
Evidence Interaction Graph
    ↓
静态训练样本
    ↓
Retriever
    ↓
静态 Interaction-aware Policy
    ↓
受控行为 Evidence Packets
    ↓
Frozen Generator + Test Harness
    ↓
V / Delta / I / Substitute / STOP 标签
    ↓
行为校准模型
    ↓
Repair–Cost 端到端评测
```

推荐把工程划分成 15 个可独立验收的阶段。

---

# 三、阶段 0：冻结研究契约

在写数据处理代码前，先创建：

```text
configs/
├── project.yaml
├── sources.yaml
├── evidence.yaml
├── generator.yaml
├── training.yaml
└── evaluation.yaml
```

`project.yaml` 至少固定：

```yaml
schema_version: "1.0"
random_seed: 42

visible_revision: base_commit

cost:
  token_weight: 1.0
  line_weight: 0.0
  tool_call_weight: 0.0

hard_limits:
  max_evidence_tokens: 12000
  max_units: 12
  max_rounds: 10
  max_tool_calls: 10
```

必须明确三类零泄漏要求：

1. 同一 issue、PR 或等价任务不能跨 train、dev、test。
2. 修复后代码和 patch added lines 不能进入 evidence。
3. gold context、core/optional 标签、行为结果不能进入 agent 输入。

同时定义证据可见性：

```text
visible：
base_commit 中存在的内容

label_only：
gold patch、test patch、gold context 标识、行为 outcome

forbidden：
修复后代码、patch-derived fix intent、人工答案
```

---

# 四、阶段 1：建立工程骨架

建议目录：

```text
evidence-agent/
├── configs/
├── data/
│   ├── raw/
│   ├── registry/
│   ├── cache/
│   ├── processed/
│   └── releases/
├── src/
│   └── evidence_agent/
│       ├── sources/
│       ├── registry/
│       ├── repositories/
│       ├── evidence/
│       ├── mappings/
│       ├── graphs/
│       ├── samples/
│       ├── retrieval/
│       ├── policy/
│       ├── behavior/
│       ├── evaluation/
│       └── common/
├── scripts/
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── leakage/
│   └── reproducibility/
├── pyproject.toml
└── README.md
```

建议将命令行脚本保持为薄封装，核心实现全部放进 `src/evidence_agent/`。否则后期很难测试和复用。

所有产物统一附加：

```json
{
  "schema_version": "1.0",
  "generator_script": "...",
  "script_version": "...",
  "source_revision": "...",
  "source_hashes": {},
  "configuration_hash": "...",
  "created_at": "..."
}
```

---

# 五、阶段 2：数据源下载与可信审计

第一版只接入三条主线：

```text
SWE-bench
ContextBench
SWE-Explore
```

CORE-Bench 是增强项，不得阻塞主线。SWE-Gym、SWE-smith、Open-SWE-Traces 暂时只作为预训练候选。

建议脚本：

```text
00_fetch_external_datasets.py
01_validate_source_schemas.py
02_build_source_manifest.py
03_audit_sources.py
```

主要产物：

```text
data/registry/
├── source_specs.lock.json
├── source_manifest.jsonl
├── source_schema_report.json
├── source_license_report.json
└── source_audit_report.json
```

`source_specs.lock.json` 记录：

```json
{
  "source_name": "swebench",
  "source_type": "huggingface",
  "repository": "...",
  "revision": "固定 revision 或 commit",
  "required_files": [],
  "optional": false
}
```

`source_manifest.jsonl` 每个本地文件一条：

```json
{
  "source_name": "contextbench",
  "relative_path": "raw/contextbench/data.jsonl",
  "size": 123456,
  "sha256": "...",
  "revision": "...",
  "download_status": "ok"
}
```

当前你实际使用的 `00_fetch_external_datasets.py` 根据命令行帮助仅支持：

```text
--project-root
--skip-github
--skip-huggingface
```

因此正确基础命令是：

```powershell
python scripts/00_fetch_external_datasets.py `
  --project-root .
```

不要再传当前代码中不存在的：

```text
--hf-endpoint
--skip-viewer
```

镜像、代理和 viewer 回退应由配置层或下载适配器实现；在代码没有声明参数前，不应凭空向命令行添加参数。

### 下载适配器接口

每个来源实现统一接口：

```python
class SourceAdapter:
    def fetch(self, spec, output_dir): ...
    def validate_schema(self, local_files): ...
    def enumerate_records(self): ...
    def provenance(self): ...
```

失败处理必须区分：

```text
required source 下载失败 → 流水线失败
optional source 下载失败 → 记录并跳过
schema 不匹配 → 禁止进入后续阶段
hash 改变 → 要求重新冻结 lock
```

### Gate 0

必须达到：

```text
revision 已冻结
文件哈希已保存
schema 已验证
来源许可证已记录
主数据源可以重复获取
```

---

# 六、阶段 3：统一实例与冻结划分

不同数据源可能指向同一 issue 或 PR。必须先统一身份，再划分数据。

建议脚本：

```text
04_build_master_registry.py
05_detect_instance_overlap.py
06_freeze_splits.py
```

`master_instances.jsonl`：

```json
{
  "canonical_instance_id": "repo::issue-or-hash",
  "repo": "owner/repo",
  "base_commit": "...",
  "issue_url": "...",
  "pr_url": "...",
  "problem_statement": "...",
  "problem_statement_hash": "...",
  "patch_target_signature": "...",
  "source_records": [],
  "original_instance_ids": [],
  "task_group_id": "..."
}
```

匹配优先级：

```text
原始 instance_id 精确匹配
→ issue/pr URL
→ repo + base_commit
→ repo + PR number
→ normalized problem hash
→ patch target signature
```

划分单位必须是 `task_group_id`，不能按单条 source record 随机切分。

建议输出：

```text
instance_aliases.jsonl
master_instances.jsonl
overlap_report.json
frozen_splits.json
```

泄漏测试：

```python
assert train_task_groups.isdisjoint(dev_task_groups)
assert train_task_groups.isdisjoint(test_task_groups)
assert dev_task_groups.isdisjoint(test_task_groups)
```

---

# 七、阶段 4：构建修复前仓库底座

建议脚本：

```text
07_materialize_repositories.py
08_verify_base_commits.py
```

对每个实例：

1. 获取仓库。
2. 验证 `base_commit` 存在。
3. 建立只读工作树或缓存。
4. 使用 `git show <base_commit>:<file>` 读取证据。
5. 保存内容哈希。

仓库缓存建议：

```text
data/cache/repo_cache/
└── owner__repo/
    ├── bare.git/
    └── worktrees/
        └── <base_commit>/
```

任何 evidence content 都必须能由以下命令重新生成：

```bash
git show <base_commit>:<file_path>
```

核心测试：

```text
unit 内容与 git show 一致
base_commit 不含 patch added lines
仓库工作区没有提前应用 gold patch
相同 repo + commit 复用同一缓存
```

### Gate 1 的第一部分

```text
所有选中实例的 base_commit 可读取
仓库内容哈希稳定
修复后内容泄漏为 0
```

---

# 八、阶段 5：Evidence Unit 抽取

第一版支持：

```text
file
class
function/method
branch
statement block
callsite
existing test
configuration section
documentation section
```

MVP 建议先限制为 Python，并优先实现：

```text
file
function/method
branch
callsite
existing test
```

建议脚本：

```text
09_extract_evidence_units.py
10_validate_evidence_units.py
```

Unit schema：

```json
{
  "unit_id": "...",
  "canonical_instance_id": "...",
  "repo": "owner/repo",
  "base_commit": "...",
  "unit_type": "function",
  "file_path": "src/a.py",
  "symbol": "Class.method",
  "language": "python",
  "start_line": 30,
  "end_line": 57,
  "raw_content": "...",
  "normalized_content_hash": "...",
  "parent_unit_id": "...",
  "token_count": 280,
  "line_count": 28,
  "visibility": "visible",
  "provenance": "base_commit"
}
```

`unit_id` 必须确定性生成，例如：

```text
sha256(
  canonical_instance_id
  + base_commit
  + file_path
  + unit_type
  + start_line
  + end_line
  + normalized_content_hash
)
```

解析顺序：

```text
Python：ast + tokenize
其他语言：Tree-sitter
解析失败：ctags
再次失败：line-window fallback
```

不要在这一阶段生成全仓库 LLM 摘要。

---

# 九、阶段 6：Patch 和外部标签映射

建议脚本：

```text
11_map_patch_to_units.py
12_align_external_spans.py
13_validate_mappings.py
```

## Patch-to-Unit

只能使用 patch 的 old side 定位：

```text
old_path 精确匹配
→ deleted line 精确匹配
→ old hunk range 与 AST span 重叠
→ section header 与 symbol 匹配
→ normalized token match
→ 父 unit 回退
```

不能把 patch 新增内容写入训练输入。

输出允许多值：

```json
{
  "role": "buggy_logic",
  "canonical_unit_id": "statement-unit-id",
  "acceptable_unit_ids": [
    "statement-unit-id",
    "branch-unit-id",
    "function-unit-id"
  ],
  "mapping_method": "deleted_line_exact",
  "confidence": 1.0
}
```

## 外部 Span-to-Unit

对 ContextBench、SWE-Explore 等外部行级标签：

[
Overlap(u,s)=
\frac{|Lines(u)\cap Lines(s)|}{|Lines(s)|}
]

映射证书：

```text
1.0        → exact-contained
≥ 0.8      → strong-overlap
≥ 0.5      → partial-overlap
< 0.5      → weak/unmapped
```

嵌套的 statement、branch、function 同时覆盖时，保存为 equivalence group，不要强制只保留最小 unit。

### Gate 1 完整验收

```text
base_commit 可读取
patch old side 可映射
外部 span 可映射
unit 内容哈希正确
patch added lines 泄漏为 0
```

---

# 十、阶段 7：Evidence Interaction Graph

建议脚本：

```text
14_build_structural_graph.py
15_generate_interaction_candidates.py
```

确定性边：

```text
contains
defines
imports
calls
called_by
reads
writes
raises
handles
tests
overlaps
duplicates
```

候选语义边：

```text
supports_candidate
complements_candidate
substitutes_candidate
conflicts_candidate
```

行为校准后再增加：

```text
behavior_complements
behavior_substitutes
behavior_conflicts
behavior_redundant
```

互补候选规则：

```text
buggy logic ↔ caller
buggy logic ↔ callee
buggy logic ↔ related test
state writer ↔ state reader
exception source ↔ handler
producer ↔ consumer
configuration definition ↔ configuration use
return value ↔ downstream assertion
```

替代候选规则：

```text
statement ↔ containing branch
branch ↔ containing function
同一 symbol 的重叠窗口
标准化内容重复的 unit
函数体 ↔ 相关局部分支
调用方约束 ↔ 表达同一约束的测试
```

第一版不需要 Graph Transformer。将图转换成候选特征即可：

```text
shortest_path
edge_types
same_file
same_symbol
span_overlap
common_neighbors
dependency_direction
```

---

# 十一、阶段 8：最小化 LLM 弱标注

程序可以确定的任务禁止调用大模型：

```text
patch 解析
AST 识别
span 映射
调用关系
包含关系
重复检测
候选枚举
成本计算
STOP 公式计算
```

只有以下情况进入语义裁决：

```text
结构规则无法确定
至少一个外部来源支持相关性
训练确实需要该关系
当前预测不确定度较高
```

建议脚本：

```text
16_select_semantic_ambiguities.py
17_label_semantic_relations.py
18_validate_semantic_labels.py
```

只标优先级最高的 5%～10%：

[
Priority=
Uncertainty
\times PotentialValue
\times SourceDisagreement
]

LLM 输出必须有原文引用：

```json
{
  "relation": "complement",
  "shared_information": "...",
  "unique_information_u": "...",
  "unique_information_v": "...",
  "joint_information": "...",
  "quotes": [
    {"unit_id": "u", "quote": "..."},
    {"unit_id": "v", "quote": "..."}
  ],
  "confidence": 0.82
}
```

自动检查 quote 是否真实存在，并检查是否引用了 patch added line。

---

# 十二、阶段 9：构建静态训练样本

建议脚本：

```text
19_build_static_acquisition_samples.py
20_validate_training_samples.py
```

状态序列可以从以下来源构造：

```text
外部成功轨迹
gold/core context 的覆盖顺序
结构图上的合理探索路径
Retriever 排名产生的模拟状态
```

训练样本：

```json
{
  "sample_id": "instance::t2",
  "canonical_instance_id": "...",
  "question": "...",
  "state": {
    "selected_unit_ids": [],
    "token_cost": 0,
    "line_cost": 0,
    "tool_call_cost": 0
  },
  "candidates": [
    "unit-a",
    "unit-b",
    "__STOP__"
  ],
  "labels": {
    "acceptable_actions": ["unit-a"],
    "state_value": null,
    "state_value_mask": true,
    "weak_marginal_values": {},
    "interaction_labels": {},
    "stop": false
  },
  "metadata": {
    "supervision_tier": "external_or_static_weak"
  }
}
```

静态数据不能伪装成真实 `V_G`。没有行为执行结果时：

```text
state_value = null
behavior mask = true
```

---

# 十三、阶段 10：训练两级 Retriever

第一层：

```text
issue → file
```

第二层：

```text
issue + 当前状态 → function/branch/test
```

基线依次实现：

```text
BM25
Dense Retriever
BM25 + Dense Hybrid
Cross-Encoder Reranker
```

建议脚本：

```text
21_build_retrieval_index.py
22_train_dense_retriever.py
23_train_reranker.py
24_evaluate_retriever.py
```

训练标签来自：

```text
SWE-bench patch old-side locations
ContextBench gold spans
SWE-Explore core regions
CORE-Bench qrels（可选）
```

必须先验证候选池召回率。如果 gold/core evidence 根本没有进入候选池，后续 policy 不可能学会选择。

Track A 指标：

```text
File Recall@k
Symbol Recall@k
Span Recall@预算
Core Region Coverage@预算
Context Precision
Ranked Coverage
Evidence Token Cost
```

---

# 十四、阶段 11：训练静态 Interaction-aware Policy

第一版模型采用：

```text
Evidence Encoder
+ DeepSets State Encoder
+ Candidate-State Interaction
+ 多任务输出头
```

编码：

[
h_u=Enc(q,u)
]

状态：

[
h_K=SetEncoder({h_{u_1},...,h_{u_t}})
]

候选与已选证据逐项交互：

[
e_{u,i}=Interaction(h_u,h_{u_i})
]

最终输出：

```text
弱条件边际价值
action ranking
complement probability
substitute probability
conflict probability
evidence role
deficit type
```

静态代理价值：

[
\widetilde{\Delta}(u\mid K)=
NewCoverage
+\alpha CompProxy
-\beta Redundancy
-\lambda Cost
]

对比模型必须同时保留：

```text
Independent Scorer：s(q,u)
State-Aware Scorer：s(q,K,u)
Interaction-Aware Policy
```

否则无法证明提升来自状态和交互建模。

### Gate 2

启动昂贵行为实验前，至少观察到：

```text
State-aware 优于 Independent
Interaction-aware 优于 State-aware
Substitute Duplication Rate 下降
Complement Candidate Recovery 提升
提升不是通过读取更多 token 获得
```

未通过时，应修正候选生成、状态编码或弱标签，而不是直接扩大行为调用。

---

# 十五、阶段 12：冻结修复器与行为实验环境

建议脚本：

```text
25_build_behavior_packets.py
26_run_frozen_generator.py
27_execute_candidate_patches.py
28_collect_behavior_outcomes.py
```

必须冻结：

```text
模型精确版本
system prompt
user prompt
temperature
seed
输入输出 token 上限
工具权限
最大工具调用数
patch 格式
超时
容器镜像
测试 harness 版本
```

冻结修复器只能看到：

```text
issue
+ 指定 evidence packet
```

不能看到：

```text
gold patch
test patch
修复后代码
gold/core 标记
patch-derived fix intent
行为 outcome
```

Evidence packet 建议使用明确边界：

```text
[ISSUE]
...

[EVIDENCE 1]
unit_id:
file:
symbol:
lines:
content:

[EVIDENCE 2]
...
```

应禁止修复器自行浏览整个仓库，否则无法测量单个证据包的价值。

---

# 十六、阶段 13：生成行为价值标签

对基础状态 `K` 和候选对 `(u,v)`，运行：

```text
K
K + u
K + v
K + u + v
```

得到：

[
V_G(K)
]

[
\Delta_G(u\mid K)=V_G(K+u)-V_G(K)
]

[
I_G(u,v\mid K)=
V_G(K+u+v)-V_G(K+u)-V_G(K+v)+V_G(K)
]

替代性保留方向：

[
Sub_G(u\rightarrow v\mid K)=
\Delta_G(v\mid K)-\Delta_G(v\mid K+u)
]

替代判定不能只看差值，还应要求：

```text
v 在原状态下确实有价值
加入 u 后 v 的价值接近 0
```

否则“两条证据都无价值”也会被错误标为替代。

## 第一轮 Pilot

严格按方案先做：

```text
20 个实例
每个实例 4 个候选对
每个候选对 4 个 packet
1 个 seed
```

共：

```text
20 × 4 × 4 = 320 次修复运行
```

候选对组成：

```text
2 个 complement candidate
1 个 substitute candidate
1 个 independent control
```

保存：

```text
patch 是否成功解析
patch 是否应用成功
FAIL_TO_PASS 通过比例
PASS_TO_PASS 通过比例
resolved
运行时间
失败类型
```

修复器有随机性时，在 dev/test 使用配对 seed，并用 Beta-Binomial 平滑：

[
\hat V_G(K)=
\frac{s+\alpha}{n+\alpha+\beta}
]

### Gate 3

需要确认：

```text
存在正 Delta
存在正 I
存在加入已有证据后 Delta 衰减
不同 packet 的修复结果并非完全相同
测试执行具有可接受稳定性
```

如果所有 packet 结果几乎相同，通常意味着：

```text
修复器绕过了证据限制
证据粒度不合适
测试过弱
候选对没有信息差
或修复器能力与任务难度不匹配
```

---

# 十七、阶段 14：行为校准与 STOP

用真实行为标签训练：

```text
V_G(K)
Delta_G(u | K)
I_G(u,v | K)
Sub_G(u → v | K)
```

行为样本较少时，优先做 pairwise ranking：

[
\Delta(u_i\mid K)>\Delta(u_j\mid K)
]

损失：

[
L_{pair}
========

-\log\sigma(
\hat Q(u_i\mid K)-\hat Q(u_j\mid K)
)
]

总损失：

[
L=
L_\Delta
+\lambda_vL_V
+\lambda_aL_{set-action}
+\lambda_iL_{interaction}
+\lambda_dL_{deficit}
]

最后只在 dev 集确定：

```text
τ：最低状态价值
λ：成本权重
εq：最低净收益
temperature scaling
互补/替代阈值
```

STOP 不应训练成一个与价值函数割裂的普通二分类器。其决策直接由状态价值和剩余边际价值导出。

---

# 十八、最终推理 Agent

```python
selected_units = []

while cost(selected_units) < hard_budget:
    candidates = retriever.search(
        issue=issue,
        state=selected_units,
    )

    state_value = policy.predict_state_value(
        issue=issue,
        state=selected_units,
    )

    scored = []
    for unit in candidates:
        delta = policy.predict_marginal_value(
            issue=issue,
            state=selected_units,
            candidate=unit,
        )

        net_value = delta - cost_weight * unit.cost
        scored.append((unit, net_value))

    if not scored:
        break

    best_unit, best_value = max(
        scored,
        key=lambda item: item[1],
    )

    if state_value >= tau and best_value <= epsilon_q:
        break

    selected_units.append(best_unit)

patch = frozen_generator.generate(
    issue=issue,
    evidence=selected_units,
)
```

无论 STOP 是否正常，都必须有硬限制：

```text
最大 token
最大 evidence 数量
最大探索轮数
最大工具调用
最大 wall time
```

---

# 十九、最终评测

## Track A：上下文获取

```text
File Recall@k
Symbol Recall@k
Span Recall@Budget
Core Region Coverage@Budget
Context Precision
Evidence Token Cost
```

## Track B：价值和交互预测

```text
State Value Brier Score
State Value ECE
Marginal Value Spearman
Pairwise Accuracy
Positive-Gain Precision@k
Negative-Gain Avoidance
Behavioral Complement F1
Behavioral Substitute F1
Interaction-Score Correlation
```

## Track C：端到端 Repair–Cost

横轴：

```text
evidence tokens
代码行数
tool calls
wall time
费用
```

纵轴：

```text
冻结修复器 resolved rate
```

主指标：

[
AUC_{Repair-Cost}
]

辅助指标：

```text
Resolved Rate
Mean Evidence Tokens
Median Tool Calls
Premature STOP Rate
Late STOP Overhead
Never STOP Rate
Substitute Duplication Rate
Redundant Evidence Rate
Negative-Marginal Selection Rate
```

对比实验：

```text
Top-1 / Top-3 / Top-5
固定 token budget
固定探索轮数
Independent Scorer
State-Aware Scorer
Interaction-Aware VOI Policy
```

关键消融：

```text
移除结构图
移除 complement 特征
移除 substitute 特征
移除行为校准
移除状态价值
移除成本项
移除 adaptive STOP
```

---

# 二十、必须编写的测试

至少实现以下自动化测试：

```text
test_source_hash_matches_lock
test_source_schema_is_valid
test_same_task_group_not_cross_split
test_base_commit_is_readable
test_unit_content_matches_git_show
test_patch_added_lines_not_visible
test_gold_labels_not_in_model_input
test_external_span_mapping_is_reproducible
test_behavior_packet_contains_only_selected_units
test_generator_config_is_frozen
test_behavior_label_formula
test_same_config_produces_same_manifest
```

最重要的是泄漏测试。建议让每个数据发布任务先执行：

```bash
pytest tests/leakage -q
```

任何一项失败都禁止训练或评测。

---

# 二十一、推荐脚本执行顺序

```text
00_fetch_external_datasets.py
01_validate_source_schemas.py
02_build_source_manifest.py
03_audit_sources.py
04_build_master_registry.py
05_detect_instance_overlap.py
06_freeze_splits.py
07_materialize_repositories.py
08_verify_base_commits.py
09_extract_evidence_units.py
10_validate_evidence_units.py
11_map_patch_to_units.py
12_align_external_spans.py
13_validate_mappings.py
14_build_structural_graph.py
15_generate_interaction_candidates.py
16_select_semantic_ambiguities.py
17_label_semantic_relations.py
18_validate_semantic_labels.py
19_build_static_acquisition_samples.py
20_validate_training_samples.py
21_build_retrieval_index.py
22_train_dense_retriever.py
23_train_reranker.py
24_evaluate_retriever.py
25_build_behavior_packets.py
26_run_frozen_generator.py
27_execute_candidate_patches.py
28_collect_behavior_outcomes.py
29_compute_behavior_labels.py
30_train_behavior_policy.py
31_calibrate_stop.py
32_evaluate_end_to_end.py
33_build_release.py
```

这些是推荐的职责划分，不代表现有仓库已经全部存在。

---

# 二十二、实际开发时的最小可行版本

不要第一轮就做全语言、全数据、图神经网络和大规模行为实验。合理的 MVP 是：

```text
语言：仅 Python
数据：SWE-bench + ContextBench + SWE-Explore
证据：file/function/branch/callsite/test
检索：BM25 + Dense Hybrid
状态编码：DeepSets
交互：结构特征 + MLP
LLM 标注：仅高不确定候选
行为实验：20 个实例、320 次运行
评测：Track A + Pilot Track B
```

MVP 的正确成功标准不是 resolved rate 很高，而是验证以下因果链：

```text
候选证据可被稳定抽取
→ 外部标签可映射
→ 状态改变候选价值
→ 存在互补和替代现象
→ 策略减少重复证据
→ 相同预算下修复结果改善
```

只有 Gate 0～3 全部通过，才扩展到更多语言、更多修复器、更大行为数据和正式 Repair–Cost 主实验。这样实现顺序与两份来源中的理论贡献、数据策略和实验主张是一致的。

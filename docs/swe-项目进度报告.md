# 软件测试修复实验：项目进度报告、后续安排与完整创新点

> 当前版本定位：**Software Repair Evidence Acquisition / Context Sufficiency**
> 核心目标：研究 Agent 能否在**不查看 gold patch**、不执行补丁生成与测试反馈闭环的前提下，从修复前仓库中逐步收集足以支持正确修复的结构化证据，并在证据充分时正确停止。

---

## 1. 项目总体定位

本项目当前不再以“自动生成补丁并执行测试”作为核心研究目标，而是聚焦于：

**SWE Issue + 修复前仓库 → Agentic Evidence Acquisition → Structured Sufficient Evidence Package**

核心研究问题是：

> 给定一个软件缺陷 Issue 和对应的修复前代码仓库，Agent 是否能够通过多步证据获取，逐渐构造出足以支持合理修复决策的证据集合，并在证据已经充分时停止继续检索？

因此，本项目的研究重点是：

- 证据收集；
- 上下文充分性；
- 中间状态下的下一步证据决策；
- 证据互补关系；
- 自适应停止；
- 完整 Agent 轨迹评估。

而不包含：

- Patch generation；
- Patch application；
- Test execution；
- Test-feedback repair loop。

---

## 2. 当前项目总体进度

目前项目已经完成数据、监督信号、Policy 建模和第一阶段训练，并完成 Multi-stage Boundary 数据重构。

当前处于：

> **Stage 2 Multi-stage Boundary 专项训练阶段**

整体状态如下：

| 模块 | 当前状态 | 说明 |
|---|---|---|
| 研究问题定义 | 已完成 | Evidence Collection / Context Sufficiency |
| SWE 数据统一 | 已完成 | 共 20,864 tasks |
| Repository Evidence Corpus | 已完成 | 约 2,550 万 Evidence Units |
| Strong Teacher | 已完成 | 20,588 个任务获得 Teacher 结果 |
| 7-slot obligation schema | 已完成 | 已冻结 |
| OR-of-AND witness semantics | 已完成 | 已进入 supervision |
| Policy state/action 构造 | 已完成 | single / pair / STOP |
| 最终训练 Bundle | 已完成 | training_ready=true |
| Stage 1 基础训练 | 已完成 | 3 epochs |
| Stage 1 Validation | 已完成 | 已发现 Boundary 瓶颈 |
| Multi-stage Boundary 重构 | 已完成 | Audit PASS，0 errors |
| Stage 2 Boundary Fine-tuning | 进行中 | 最新 recovery：state 8000 |
| Stage 2 Validation | 待完成 | 200 个 multi-stage states |
| 原 Validation Retention | 待完成 | 检查 Initial / Complete 是否退化 |
| Benchmark | 尚未使用 | 保持严格隔离 |
| Agent Trajectory Rollout | 待完成 | 最重要的最终实验 |
| Ablation | 待完成 | 支撑论文创新 |
| 最终论文结果整理 | 待完成 | 等最终实验完成 |

---

## 3. 数据体系

### 3.1 基础任务数量

最终统一任务总数：

\[
20,864
\]

其中：

- Train：18,347
- Validation：223
- Benchmark：2,294

最终 experiment eligible：

\[
20,859
\]

排除任务：

\[
5
\]

排除原因为：

```text
POLICY_POSITIVE_UNSCOREABLE
```

这 5 个任务仍然保留在数据中作为 provenance，只是不进入正式实验指标。

这样可以同时保证：

1. 数据来源完整；
2. 训练和评估不会受到无法合法评分样本的污染。

---

## 4. Repository Evidence Corpus

项目没有让模型直接把整个 repository 当作一个黑盒输入，而是将修复前仓库拆分为稳定的 Evidence Units。

当前语料规模约为：

\[
25,496,300
\]

个 Evidence Units。

对应约：

\[
1,027,752
\]

个文件版本。

每个 Evidence Unit 包含稳定的信息，例如：

- Evidence ID；
- 文件路径；
- 代码内容；
- 结构位置；
- token cost；
- repo snapshot；
- base_commit 对应关系。

这一步将：

```text
Repository
```

转化为：

```text
Queryable Evidence Space
```

从而为 Agent 的动作空间提供稳定基础。

最终实际进入 Policy 训练和离线评估的证据子集为：

\[
998,682
\]

行。

需要区分：

```text
全量 Repository Corpus
≈ 25.5M Evidence Units
```

与：

```text
Policy Training Evidence
≈ 998.7K Evidence Units
```

---

## 5. Strong Teacher

Strong Teacher 是项目中最关键的监督来源之一。

当前 Teacher 冻结结果：

- Teacher rows：20,588
- Included：20,501
- Excluded：87

Teacher 不直接输出：

> “正确文件是哪一个”或者“正确代码行是哪一行”。

而是将修复所需要的信息拆解为 7 类 obligation：

1. `fault_location`
2. `fault_logic`
3. `dependency_context`
4. `state_flow`
5. `behavior_constraint`
6. `repair_scope`
7. `validation_constraint`

这比传统的：

```text
gold file
gold line
gold context
```

更加结构化。

Teacher 真正回答的是：

> 一个能够合理支持正确修复的 Evidence Package 应该包含哪些信息角色？

这就是本项目中“Context Sufficiency”的核心定义。

---

## 6. OR-of-AND Witness 语义

本项目不是简单地建立：

```text
obligation → evidence
```

的一一映射，而是采用：

\[
\text{OR-of-AND witnesses}
\]

例如：

```text
[[2,5]]
```

表示：

\[
e_2 \land e_5
\]

即证据 2 和证据 5 必须同时存在，才能满足对应 obligation。

例如：

```text
[[2], [5,9]]
```

表示：

\[
e_2 \lor (e_5 \land e_9)
\]

即：

- 证据 2 单独可以满足该 obligation；
- 或者证据 5 与证据 9 联合也可以满足该 obligation。

这个设计能够显式表示两类真实关系。

### 6.1 替代关系

例如：

```text
完整函数
```

或者：

```text
相关测试 + 调用方
```

可能都能提供等价约束。

### 6.2 互补关系

例如：

```text
buggy logic
+
caller constraint
```

两条证据必须共同出现才能形成完整判断。

因此，本项目研究的并不是：

\[
Independent\ Evidence\ Relevance
\]

而是：

\[
Structured\ Evidence\ Sufficiency
\]

---

## 7. Evidence Policy 建模

当前模型是：

> **Cross-Encoder Evidence Policy Ranker**

模型评分：

\[
s_A=f_\theta(q,K,A)
\]

其中：

- \(q\)：Issue；
- \(K\)：当前已经收集到的 Evidence State；
- \(A\)：候选下一步动作。

动作空间为：

\[
A_t \in \{[u],[u,v],STOP\}
\]

其中：

- `[u]`：读取一条 Evidence Unit；
- `[u,v]`：联合读取两条证据；
- `STOP`：停止继续收集。

Agent 状态定义为：

\[
S_t=(q,K_t)
\]

因此，模型学习的不是：

\[
Score(q,u)
\]

而是：

\[
Score(q,K,u)
\]

这意味着：

> 同一条 Evidence 在不同当前证据状态下，其价值可以完全不同。

这正是 state-dependent policy 与普通 Retriever 的根本区别。

---

## 8. Multi-positive Listwise Training

当前训练目标不是普通二分类，而是 multi-positive listwise ranking：

\[
L=
\log\sum_{a\in A}e^{s_a}
-
\log\sum_{a\in A^+}e^{s_a}
\]

一个状态下允许同时存在多个正确动作。

这是合理的，因为在真实证据收集中：

> 下一步不一定只有唯一正确 Evidence。

只要某个候选动作能够有效提高当前证据集合的充分性或进度，它就可以成为 positive action。

---

## 9. Stage 1 训练结果

Stage 1 使用三类状态：

- Initial
- Decision Boundary
- Complete

共训练 3 epochs。

Epoch 3 Validation 总体指标：

```text
Overall Hit@1 ≈ 77.45%
Overall MRR   ≈ 0.8835
```

如果只看 aggregate metric，模型似乎已经表现不错。

但分状态分析后发现了明显结构性问题。

### 9.1 Initial

```text
Hit@1 ≈ 94.39%
MRR ≈ 0.9719
STOP accuracy ≈ 99.49%
```

### 9.2 Complete

```text
Hit@1 ≈ 88.27%
MRR ≈ 0.9360
STOP accuracy ≈ 88.27%
```

### 9.3 Decision Boundary

```text
Hit@1 ≈ 27.52%
MRR ≈ 0.6300
STOP accuracy ≈ 29.36%
```

由此得到第一个重要实验发现：

> Aggregate metric 会严重掩盖 Intermediate Evidence Decision 的失败。

模型可以很擅长：

- 完全没有证据时决定第一步；
- 证据已经齐全时决定 STOP；

但在真实 Agent 最常出现的中间状态：

> 已经有部分证据，但还不充分，

模型表现非常弱。

---

## 10. 原始 Boundary 构造存在的问题

进一步检查原始 Policy builder 后发现：

原始 V2.10 Boundary 基本采用：

> 从 minimum sufficient certificate 中删除一条 Evidence，然后选取最接近 Complete 的状态。

因此，一个任务最多只形成一个 near-complete Boundary。

这种训练分布与真实 Agent trajectory 不一致。

真实 Agent 应该经历：

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

因此，Intermediate state 不应该只由 near-complete 状态代表。

这导致第二个重要实验发现：

> Near-complete-only supervision 不能充分表示真实 Agent Evidence Acquisition trajectory。

---

## 11. Multi-stage Decision Boundary Reconstruction

针对上述问题，项目进一步构造了 Multi-stage Boundary。

Boundary 被划分为：

- Early
- Mid
- Late
- Near-complete

当前构建结果：

### Train

总计：

\[
10,907
\]

个新的 Boundary states。

分布：

```text
early          1,105   ≈10.1%
mid            5,173   ≈47.4%
late           3,901   ≈35.8%
near_complete    728   ≈ 6.7%
```

### Validation

总计：

\[
200
\]

个 multi-stage Boundary states。

分布：

```text
early          22
mid            95
late           69
near_complete  14
```

Audit 结果：

```text
PASS
error_count = 0
```

相比原始 Boundary，这个分布更加符合真实 sequential Agent 的状态结构。

---

## 12. Stage 2 当前训练状态

Stage 2 从 Stage 1 的：

```text
models/evidence_policy_v1_0/best
```

初始化。

训练配置：

```text
epochs = 2
learning_rate = 5e-6
grad_accum_steps = 8
max_candidates = 12
boundary_repeat = 1
```

当前最新有效 recovery：

```text
Epoch 1
state = 8,000 / 10,907
global_optimizer_step = 1,000
```

截至 recovery，累计训练指标：

\[
Hit@1=
\frac{6804}{8000}
\approx85.05\%
\]

\[
MRR\approx0.9167
\]

\[
STOPAccuracy=
\frac{7557}{8000}
\approx94.46\%
\]

平均 loss：

\[
0.2987
\]

需要强调：

> 这些属于训练集累计指标，不是最终 validation result。

因此当前只能说明：

> Multi-stage Boundary supervision 是可学习的，模型已经明显适应新的 Boundary distribution。

还不能得出：

> “Boundary 泛化达到 85%”

这样的结论。

真正的泛化效果必须看 200 个 Stage 2 validation states。

---

## 13. 当前工程问题

Stage 2 训练中出现过 native crash。

最新一次异常发生在约：

```text
state ≈ 8163
```

系统线程状态显示：

```text
pt_autograd_0 → do_coredump
其他线程 → do_exit
```

说明这不是普通 Python traceback，而更可能属于：

- PyTorch native crash；
- CUDA native crash；
- 底层 C/C++ runtime crash。

同时服务器配置为：

```text
ulimit -c = unlimited
```

并且：

```text
core_pattern → apport
```

导致 native crash 后触发 core dump，进程在退出阶段长时间阻塞，GPU 显存无法释放。

目前旧进程已经彻底清理。

好消息是：

```text
state = 8000
optimizer step = 1000
```

的 recovery checkpoint 已完整保存。

因此训练成果没有丢失。

后续需要增加：

- per-process core dump suppression；
- Python `faulthandler`；
- native crash diagnostics。

这个问题属于训练工程稳定性问题，不属于研究方法本身失败。

---

# 14. 项目完整创新点

## 创新一：将代码检索重新定义为 Evidence Sufficiency Acquisition

传统软件修复检索通常是：

\[
q\rightarrow TopK(Code)
\]

目标是找“最相关代码”。

本项目重新定义为：

\[
(q,K_t)\rightarrow A_{t+1}
\]

其中当前 Evidence State \(K_t\) 会直接影响下一步动作。

最终目标是：

\[
K_0=\emptyset
\rightarrow
K_1
\rightarrow
...
\rightarrow
K_T
\]

使：

\[
Sufficiency(K_T)=1
\]

并尽可能降低：

- Evidence 数量；
- token cost；
- trajectory steps；
- redundant evidence。

因此研究对象从：

```text
Static Retrieval
```

升级为：

```text
State-dependent Sequential Evidence Acquisition
```

---

## 创新二：Structured Repair Evidence Obligations

项目没有把修复上下文简化成：

```text
gold file
gold line
```

而是采用 7 类结构化信息需求：

1. fault_location
2. fault_logic
3. dependency_context
4. state_flow
5. behavior_constraint
6. repair_scope
7. validation_constraint

因此：

> “命中补丁位置”

不再等价于：

> “已经获得足够的修复上下文”。

这是从 localization correctness 向 repair-context sufficiency 的转变。

---

## 创新三：OR-of-AND Witness Sufficiency

项目显式建模：

\[
e_a\lor(e_b\land e_c)
\]

从而同时表示：

- Alternative Evidence；
- Complementary Evidence。

传统 flat relevance label 通常默认每条证据相互独立。

本项目则把：

\[
Evidence\ Sufficiency
\]

建模为一种结构化逻辑覆盖关系。

---

## 创新四：State-dependent Evidence Policy

传统 Retriever 学习：

\[
Score(q,u)
\]

而本项目学习：

\[
Score(q,K,u)
\]

同一条 Evidence：

- 在 Early state 可能非常关键；
- 在 Late state 可能已经完全冗余。

因此，Evidence 的价值取决于 Agent 当前已经知道什么。

这是 Agentic RAG 与静态检索之间的重要区别。

---

## 创新五：Single / Pair / STOP 统一动作空间

动作空间统一为：

\[
\{[u],[u,v],STOP\}
\]

使模型能够同时学习：

1. 单条 Evidence 的价值；
2. 两条 Evidence 的互补价值；
3. 当前是否已经应该停止。

尤其 Pair Action 可以建模：

> 两条证据单独不够，但联合之后能够满足 witness。

---

## 创新六：Sufficiency-aware STOP

STOP 不是独立的启发式阈值，而是直接由 Evidence Sufficiency 决定。

当：

\[
Sufficiency(K)<1
\]

时：

```text
STOP = negative
```

当：

\[
Sufficiency(K)=1
\]

时：

```text
STOP = positive
```

从而可以自然评价：

\[
PrematureSTOPRate
\]

这对真实 Agent 非常重要。

---

## 创新七：Multi-stage Decision Boundary Reconstruction

原始训练只包含一个 near-complete Boundary。

本项目进一步构造：

```text
Early
Mid
Late
Near-complete
```

从而更接近真实 Evidence Acquisition trajectory。

核心状态为：

\[
K_t\neq\emptyset
\land
Sufficiency(K_t)<1
\]

模型必须学会：

> 当前证据还没齐，下一步应该继续获取什么。

这一创新不是简单的数据扩充，而是针对原始训练分布无法覆盖真实 Agent 中间状态的问题进行结构性修正。

---

## 创新八：Evidence-level 到 Trajectory-level Evaluation

传统 Retrieval 通常只测：

```text
Recall@K
MRR
Hit@K
```

本项目最终还要从：

\[
K_0=\emptyset
\]

开始完整 rollout。

Agent 连续执行：

\[
A_1,A_2,\ldots,A_T
\]

直到：

```text
STOP
```

最终评价：

- 是否达到 sufficiency；
- 是否提前停止；
- 用了多少步；
- 收集了多少 Evidence；
- 是否过度收集；
- 是否成功完成整条 trajectory。

最终核心指标包括：

\[
SufficiencySuccessRate
\]

\[
TrajectorySuccessRate
\]

\[
PrematureSTOPRate
\]

\[
AverageCompletionSteps
\]

\[
AverageEvidenceCount
\]

\[
ContextCost
\]

\[
OverCollectionRate
\]

\[
FinalObligationCoverage
\]

这一层使项目不再只是一个 ranking model，而成为真正的 Evidence Acquisition Agent。

---

# 15. 后续实验安排

## Phase A：完成 Stage 2

从最新 recovery：

```text
state = 8000
opt_step = 1000
```

继续训练。

完成：

```text
Epoch 1
Epoch 2
```

得到：

```text
models/evidence_policy_multistage_ft_v1/best
```

Stage 2 `best` 应按 200 个 multi-stage Boundary validation states 的 MRR 选择。

---

## Phase B：同一 Multi-stage Validation 上做公平前后对比

必须避免：

```text
Stage1 old Boundary 109
vs
Stage2 new Boundary 200
```

这种非同分布比较。

正确做法：

```text
Stage1 best
→ new 200 Boundary validation

Stage2 best
→ same new 200 Boundary validation
```

比较：

- Loss
- Hit@1
- MRR
- STOP Accuracy

这样才能证明：

> Multi-stage Fine-tuning 是否真正提高了 Intermediate Evidence Decision。

---

## Phase C：检查灾难性遗忘

Stage 2 主要训练 Boundary。

因此需要评估：

```text
Stage2 best
→ 原始 validation 501 states
```

分别报告：

- Initial；
- Original Boundary；
- Complete。

尤其关注 Complete。

因为 Stage 2 大量训练：

```text
STOP = negative
```

可能导致模型出现：

> 证据已经充分，却仍然继续收集。

理想情况：

```text
Boundary ↑↑
Initial ≈
Complete ≈
```

如果出现：

```text
Boundary ↑↑
Complete ↓↓↓
```

则需要再做 replay / anchor fine-tuning：

```text
Multi-stage Boundary
+
少量 Initial
+
少量 Complete
```

而不是推翻 Stage 2。

---

## Phase D：最终 Benchmark

Benchmark：

```text
tasks = 2294
eligible = 2292
```

目前 Benchmark 没有参与训练和调参，这是正确的。

最终模型确定后，Benchmark 只做正式评估，不再反向调参。

需要报告两层结果。

### State-level Policy Ranking

例如：

- Hit@1
- MRR
- STOP Accuracy

### Trajectory-level Agent Acquisition

这是最终更重要的部分。

---

## Phase E：Agent Trajectory Rollout

每个任务从：

\[
K_0=\emptyset
\]

开始。

Agent 流程：

```text
Retriever
↓
Candidate Generator
↓
Policy Ranker
↓
Action
↓
Update K
↓
Next State
```

直到：

```text
STOP
```

或者达到最大步数。

最终根据 Teacher obligation / witness 判断 Evidence Package 是否充分。

主要指标：

1. Sufficiency Success Rate
2. Trajectory Success Rate
3. Premature STOP Rate
4. Average Steps to Sufficiency
5. Average Evidence Count
6. Token / Context Cost
7. Over-collection Rate
8. Final Obligation Coverage

这一实验将回答：

> Agent 是否真的能够从零开始独立完成证据收集？

---

# 16. 消融实验安排

考虑到项目由单人完成，消融实验不宜无限扩张。

建议优先做以下几组。

## Ablation 1：Original Boundary vs Multi-stage Boundary

比较：

```text
Original near-complete Boundary
vs
Multi-stage Boundary
```

验证 Multi-stage Reconstruction 的必要性。

---

## Ablation 2：Stage 1 vs Stage 2

在同一个 200-state Multi-stage Validation 上比较：

```text
Stage1 best
vs
Stage2 best
```

这是最直接的微调效果实验。

---

## Ablation 3：No Pair vs Single + Pair

比较：

```text
Single only
vs
Single + Pair
```

验证 Evidence complementarity 的作用。

---

## Ablation 4：Fixed-step / No STOP vs Learned STOP

比较：

```text
No STOP / fixed steps
vs
Sufficiency-aware learned STOP
```

验证自适应停止。

---

## Ablation 5：Flat Relevance vs State-dependent Policy

如果工程成本允许，再比较：

```text
Score(q,u)
vs
Score(q,K,u)
```

用于验证 Current Evidence State 是否真正必要。

---

# 17. 最终论文逻辑链

最终论文不应该写成：

> 我们设计了一个新的代码 Retriever。

更合理的完整逻辑是：

```text
真实软件修复依赖多种信息
↓
单一 gold location 不等于修复上下文充分
↓
定义 Structured Evidence Sufficiency
↓
使用 7 类 obligation 描述信息需求
↓
使用 OR-of-AND witness 表达替代和互补
↓
将仓库探索建模为 sequential evidence acquisition
↓
模型根据 Issue + Current Evidence + Candidate Action 决策
↓
支持 single / pair / STOP
↓
发现 aggregate metric 掩盖 Intermediate state 失败
↓
发现 near-complete-only Boundary 不代表真实 trajectory
↓
提出 Multi-stage Decision Boundary Reconstruction
↓
训练 Early / Mid / Late / Near-complete 状态
↓
从 K={} 进行完整 rollout
↓
评价 Agent 是否能够以有限成本达到 Evidence Sufficiency
```

这形成了完整闭环：

\[
Task
+
Representation
+
Policy
+
Training
+
Evaluation
\]

---

# 18. 当前已经得到的重要实验发现

## 发现一：Aggregate Metric 会掩盖 Intermediate State 的失败

Stage 1：

```text
Overall MRR ≈ 0.884
```

但：

```text
Boundary Hit@1 ≈ 27.5%
Boundary STOP Accuracy ≈ 29.4%
```

说明：

> 如果只看总体指标，会误以为模型已经能够进行 Evidence Acquisition；但真正最关键的 Intermediate Evidence Decision 仍然很弱。

---

## 发现二：Near-complete-only Boundary 不足以代表真实 Agent

原始 Policy builder 主要产生 near-complete Boundary。

而真实 Agent 的大部分轨迹处于：

- Early；
- Mid；
- Late；

而不是始终接近 Complete。

因此需要 Multi-stage Reconstruction。

---

## 发现三：Multi-stage Boundary Supervision 是可学习的

截至 Stage 2 recovery：

```text
Train Hit@1 ≈ 85.05%
Train MRR ≈ 0.9167
Train STOP Accuracy ≈ 94.46%
Mean Loss ≈ 0.2987
```

虽然这些还不是最终 validation 结果，但至少证明：

> 新构造的 multi-stage Policy supervision 不是随机噪声，模型能够明显学习其中的决策结构。

---

# 19. 最终论文建议压缩成四项主要贡献

最终摘要和 Introduction 中建议不要列过多碎片化创新。

最适合压缩成以下四项。

## Contribution 1：新任务定义

提出 **Software Repair Evidence Acquisition / Context Sufficiency**，将软件修复上下文构建从静态相关性检索重新定义为状态依赖的序列证据获取问题。

---

## Contribution 2：Structured Evidence Sufficiency Representation

提出由：

- 7 类 repair evidence obligations；
- OR-of-AND witnesses；

共同构成的结构化充分性表示，从而显式表达证据的：

- 替代性；
- 互补性；
- 逻辑充分性。

---

## Contribution 3：State-dependent Evidence Policy

提出 Evidence Policy Ranker，根据：

```text
Issue
+
Current Evidence State
+
Candidate Action
```

统一选择：

```text
single
pair
STOP
```

并通过 multi-positive listwise supervision 学习 Evidence Acquisition Policy。

---

## Contribution 4：Multi-stage Boundary + Trajectory Evaluation

提出 Multi-stage Decision Boundary Reconstruction，将单一 near-complete Boundary 扩展为：

```text
Early
Mid
Late
Near-complete
```

并进一步从 state-level ranking 扩展到完整 trajectory-level sufficiency evaluation。

---

# 20. 后续执行顺序

从现在开始建议冻结大的研究方向。

后续严格按以下顺序执行：

```text
1. 完善 native crash 诊断
2. 从 state=8000 recovery 恢复
3. 完成 Stage 2 两个 epoch
4. 选择 Stage2 best
5. Stage1 best vs Stage2 best：
   在同一 200 Multi-stage Validation 上公平比较
6. Stage2 best：
   回测原始 501 Validation，检查 Initial / Complete retention
7. 确定最终 Policy Model
8. 正式 Benchmark
9. 实现完整 Agent Trajectory Rollout
10. 完成核心 Ablation
11. 汇总最终论文表格、曲线和分析
```

最终必须形成三条证据链：

\[
State\text{-}level\ Improvement
\]

+

\[
No\ Catastrophic\ Forgetting
\]

+

\[
Trajectory\text{-}level\ Sufficiency\ Improvement
\]

如果这三部分都能够成立，那么该项目就不再只是：

> 一个 Code Reranker 或 Retriever 实验，

而是一个完整的：

> **Agentic Software Repair Evidence Acquisition / Context Sufficiency Framework**

---

## 21. 当前阶段结论

当前项目最核心的数据、监督、Policy 建模和 Stage 1 基线已经完成。

现阶段最重要的不是继续增加新的方法模块，而是把已有创新通过实验闭环证明完整。

当前 Stage 2 的训练趋势说明：

> Multi-stage Boundary 方向具有明显可学习性。

接下来真正决定论文强度的，是：

1. 同一 200-state Multi-stage Validation 上 Stage 1 → Stage 2 的真实提升；
2. 原始 Initial / Complete 是否保持稳定；
3. 最终 Benchmark；
4. 从 \(K=\emptyset\) 开始的完整 Agent Trajectory Rollout；
5. 核心消融实验。

最终论文的核心价值，应集中在：

> **结构化定义“修复证据是否充分”，并让 Agent 学会在不同证据状态下持续获取必要信息，直到证据足够，而不是单纯检索与 Issue 最相关的代码。**

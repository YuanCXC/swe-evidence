# Evidence Agent

> 面向修复保障度的证据获取：在有限预算下，软件修复系统应该收集哪些修复前仓库证据、哪些证据需要组合使用、哪些证据可以相互替代，以及何时停止继续读取仓库。

本项目**不生成补丁、不修改代码、不运行测试**，也不使用真实 patch 成功率作为标签或指标。项目只负责从修复前仓库（`repo + base_commit`）获取证据，并判断证据集合是否在信息层面足以支持后续形成正确修复。仓库中已有的测试代码和测试断言可以作为证据读取，但不能执行。

---

## 一、研究背景与动机

传统代码检索通常隐含假设：每条证据的价值可以独立相加：

```
Value(K) = Σ Value(u_i)
```

但实际修复证据往往存在明显交互：

- **互补**：单独看 buggy logic 不足以修复，单独看 caller constraint 也不足以修复，两者一起才能确定正确修改；
- **替代**：完整函数、局部分支、相关测试断言、函数语义摘要可能表达相同信息，全部读取只会增加冗余；
- **冗余**：当前证据集合已经覆盖某条候选的信息，继续读取没有新增价值。

因此需要显式建模：

```
Evidence Value = Individual Value + Complementarity − Substitutability/Redundancy
```

## 二、三项核心贡献

### 贡献一：Sufficiency-Calibrated Repair Evidence Acquisition

将仓库探索从「位置相关性和覆盖率优化」提升为「成本约束下的证据充分性优化」。定义当前证据状态的充分性价值（修复保障度）：

```
S(K) = P(C(K)=1 | q, K)
```

其中 `C(K)=1` 表示证据包已覆盖故障位置、故障机理、预期行为、相关约束和影响边界，且不存在关键缺口。任务目标升级为：

```
max_π  S(K_T) − λ · Cost(K_T)
```

### 贡献二：Evidence Sufficiency Value-of-Information Policy

估计当前证据包的充分性价值 `Ŝ(K_t)` 与候选证据的条件边际价值：

```
Δ(u | K_t) = S(K_t ∪ {u}) − S(K_t)
Q(u | K_t) = Δ(u | K_t) − λ · Cost(u)
```

同一证据在不同状态下可以有不同价值。自适应停止由剩余信息价值决定：

```
max_{u ∈ C_t} Q(u | K_t) ≤ 0   且   Ŝ(K_t) ≥ τ
```

即剩余候选证据对充分性的预期增益已经不足以抵消读取成本。

### 贡献三：Complementarity- and Substitutability-Aware Evidence Acquisition

显式建模修复证据之间的互补、替代、冗余和条件依赖关系。定义交互价值：

```
I(u, v | K) = S(K ∪ {u,v}) − S(K ∪ {u}) − S(K ∪ {v}) + S(K)
```

- `I > 0`：组合增益大于单项增益之和，支持互补；
- `I ≈ 0`：贡献基本可加，可能独立或单侧冗余；
- `I < 0`：存在价值重叠或边际收益递减，支持替代或冗余。

## 三、数据来源

本项目只使用以下来源，SWE-bench 是唯一任务基准：

| 来源 | 角色 | 是否创建新任务 | 对齐要求 |
|------|------|----------------|----------|
| SWE-bench | 唯一任务基准、问题、commit、patch 和测试 | 是 | 不适用 |
| ContextBench | Gold context 和细粒度强证据 | 否 | 必须可靠映射到 SWE-bench `instance_id` |
| SWE-Explore | 多成功轨迹共识区域、可选区域和读取顺序 | 否 | 必须精确命中 SWE-bench `instance_id` |
| repository snapshot | 修复前完整仓库语料 | 否 | 必须匹配 SWE-bench `repo + base_commit` |

无法与 SWE-bench 可靠对齐的来源记录不下载、不合并、不发布。

**关键的防泄漏规则**：Gold patch、test patch、ContextBench gold context、SWE-Explore trajectory 等只能作为离线标签，不能暴露给在线检索模型。所有模型可见代码都必须来自修复前 `base_commit` 对应的快照。

## 四、最终发布结构

完整数据集由一条命令生成，最终发布 5 个文件：

```
data/unified_swe_dataset_v1/
├── train.parquet              # 18,347 个训练任务
├── validation.parquet         # 223 个验证任务
├── benchmark.parquet          # 2,294 个评测任务
├── repository_corpus.parquet  # 1,027,752 个唯一文件版本 / 25,496,300 个 Evidence Unit
└── manifest.json              # Schema、来源、哈希、统计和审计元数据
```

当前已发布的实测规模（来自 `manifest.json`）：

| 项目 | 数量 |
|------|-----:|
| SWE-bench train | 19,008 |
| SWE-bench dev | 225 |
| SWE-bench test | 2,294 |
| 删除的无证书 train/dev 任务 | 663 |
| 最终发布 train | 18,347 |
| 最终发布 validation | 223 |
| 最终发布 benchmark | 2,294 |
| 最终发布总任务 | 20,864 |
| 唯一文件版本 | 1,027,752 |
| Evidence Unit | 25,496,300 |
| 仓库快照 | 18,527 |
| snapshot-file 成员关系 | 32,092,093 |
| policy state | 42,284 |
| 候选动作（single+pair+stop） | 2,880,701 |

### 4.1 统一任务 Schema

三个任务文件（train / validation / benchmark）共用同一个物理 Schema，一行代表一个去重任务：

```json
{
  "schema_version": "1.0",
  "task_id": "task_django_12345",
  "task_group_id": "group_django_issue_12345",
  "snapshot_id": "snapshot_django_abc123",
  "input": {},
  "provenance": [],
  "supervision": {},
  "trajectories": [],
  "evaluation": null,
  "split_info": {},
  "quality": {}
}
```

**模型输入与离线监督严格分区**：

- `input`：模型和 Retriever 可以访问的唯一任务输入区，包含 `repo`、`base_commit`、`problem_statement`、`retrieval_scope` 等。禁止写入 Gold patch、test patch、Gold 文件/函数/span、obligation、witness、成功轨迹。
- `supervision`：保存离线训练标签和评分答案，包含 `evidence_labels`、`obligations`、`witness_groups`、`policy_states`、`gold_patch`、`test_patch` 等，不属于模型输入。

### 4.2 义务、Witness Group 与评分

每个任务只生成确实适用且能够获得证据支持的义务。固定 7 类义务：

| 类型 | 含义 |
|------|------|
| `fault_location` | 故障发生的位置 |
| `fault_logic` | 错误机制或错误逻辑 |
| `dependency_context` | 调用、继承、导入等依赖约束 |
| `state_flow` | 状态、参数或数据如何流动 |
| `behavior_constraint` | 问题要求的正确行为 |
| `repair_scope` | 修复可能影响的范围 |
| `validation_constraint` | 测试、边界条件和回归约束 |

采用「方案 C」的语义：义务是「必须弄清楚的问题」，不是某个固定文件或 span。一个义务可以有多个 witness group，同一 group 内按 AND 解释，不同 group 间按 OR 解释。`A + B` 形成互补关系，`[A, B]` 与 `C` 形成可替代路径。

评分采用两个原始分量：

- **完成度** `C(K)`：已完成的 mandatory obligations 数量 / mandatory obligations 总数；
- **进度** `P(K)`：所有 applicable obligations 的 witness 进度平均值（避免把互补证据组的第一个证据错误标成零价值）。

动作增益：

```
completion_gain = C(K ∪ A) − C(K)
progress_gain   = P(K ∪ A) − P(K)
```

### 4.3 状态与候选动作

每个任务自适应生成 2～3 个 `policy_state`：

| `state_type` | 构造方式 | 训练目的 |
|--------------|----------|----------|
| `initial` | 固定 `K = empty` | 从零开始选择第一批证据；STOP 为负例 |
| `decision_boundary` | 最接近充分但仍不充分的 Gold 状态 | 学习继续补证，避免提前 STOP |
| `complete` | `K` 固定为规范化最小充分证书 | 学习证据充分后选择 STOP，拒绝冗余动作 |

动作空间包含三种：

- **单证据动作** `[u]`：获取一条 Evidence Unit；
- **双证据动作** `[u, v]`：同时获取两条证据，用于学习互补/替代/冗余关系；
- **STOP**：与证据动作竞争的特殊动作，由同一个 Cross-Encoder 统一评分。

**在线候选与离线标签候选分离**：

- 在线候选只允许使用真实运行时可见的信息（BM25、路径、符号、结构扩展），固定四个等权 RRF 通道：`bm25_content`、`path_name`、`symbol`、`structure`，通道深度 64，RRF 常数 64，融合保留 64。
- 离线标签候选用于确定训练真值和补足正例，可以使用 patch 映射、ContextBench、SWE-Explore、witness group、规则验证教师标签。

### 4.4 双证据交互与多标签关系

双证据动作显式记录相对于当前状态 `K` 的二阶交互量：

```
I_C(u, v | K) = C(K ∪ {u,v}) − C(K ∪ {u}) − C(K ∪ {v}) + C(K)
I_P(u, v | K) = P(K ∪ {u,v}) − P(K ∪ {u}) − P(K ∪ {v}) + P(K)
```

关系标签按 obligation 展开，支持 5 个类别（多标签，可同时为真）：

| 关系 | 判定原则 |
|------|----------|
| `complement` | 位于同一 AND group，任一单证据无法完成该义务 |
| `substitute` | 位于同一义务的不同 group，任一 group 都能独立满足义务 |
| `redundant` | 内容相同、区间包含、高度重叠或相同语义的重复表示 |
| `independent` | 分别推进不同义务，且不存在明显交互 |
| `conflict` | 版本、行为或约束指向矛盾结论 |

`unknown` 不作为第 6 个输出类别，只通过目标值 null 和损失掩码表达。

### 4.5 严格 STOP

STOP 使用已确认的严格规则：

```
全部 mandatory obligations 已完成 -> STOP 可接受
任一 mandatory obligation 未完成   -> STOP 不可接受
义务定义或覆盖关系不可靠            -> STOP unknown
```

正常终止不设置额外分数阈值或 STOP 二分类器，固定由统一动作排序头的 argmax 决定。为防止 STOP 校准失败导致无界取证，设置两项 train-only 数据驱动的硬预算：

```
evidence_token_budget     = 32768   # 累计获取 Evidence Unit 的渲染成本上限
selected_evidence_unit_cap = 64      # 唯一 Evidence Unit 数量上限
```

### 4.6 repository_corpus.parquet

每行表示一个**唯一文件版本**，唯一键为 `repo + path + blob_oid`：同一仓库、同一路径、同一内容只保存一次，多个代码快照通过 `snapshot_ids` 共享相同文件版本，文件正文只保存一次。

```
file_version_id    全局唯一文件版本 ID
repo / path / blob_oid / content_sha256
snapshot_ids       使用此文件版本的全部快照 ID
evidence_units     文件内的可检索结构（file / class / function / method / code_block / doc_section）
imports            文件的静态依赖声明
extraction         解析器、版本和状态
```

Evidence Unit 的正文通过行号范围恢复：

```python
unit_content = file_content_lines[start_line - 1:end_line]
```

任何任务只能访问 `snapshot_ids` 包含自身 `snapshot_id`、且 `attributes.searchable=true` 的文件版本和 Evidence Unit。

## 五、受约束教师标注

教师标注只补足语义缺口（义务角色、pair 关系），不替代全量规则监督。

- **采用范围**：仅在 ContextBench / SWE-Explore 证据无法确定语义角色、缺少完整义务图、pair 缺少最小 witness group 等情况下调用，不对全部任务做无差别教师调用。
- **教师只输出**：从固定 7 种义务类型中选择、引用真实存在的 `evidence_id`、使用 AND group 和 group 间 OR、输出 `unknown`、给出置信度和依据。
- **教师不得**：创建新代码正文、直接给出 `C`/`P`/动作增益/STOP、直接输出关系对象、把未提供的仓库文件加入 witness、覆盖确定性冲突标签。
- **单次标注与程序验证**：每个教师包只进行 1 次语义标注，不做多次采样或多数投票；通过全部程序约束的结果记录为 `teacher_verified`，未通过的不得进入训练监督。
- **最终冻结规模**：1,800 个有效教师标签（train 1,400 + validation 400），教师模型为 `deepseek-v4-flash`，Prompt 版本 `unified-swe-teacher-v4`。benchmark 禁止 teacher-only 标签作为 Gold。

## 六、唯一训练模型：Cross-Encoder Evidence Policy Ranker

最终只训练**一个**面向软件仓库问题定位的、状态条件化且支持证据组合的 Cross-Encoder Evidence Policy Ranker。

核心函数：

```
s_A = f_θ(q, K, A)
```

- `q`：SWE-bench 问题描述；
- `K`：当前已获取的有界证据集合；
- `A`：单证据 `[u]`、双证据 `[u, v]` 或 `STOP`；
- `s_A`：动作效用排序分数，分数越高表示当前越值得执行。

「只训练一个模型」的含义：只产生一套参数和一个最终检查点；单证据、双证据和 STOP 共用同一个 Backbone 与动作排序头；不训练独立 Retriever、pair 选择器、STOP 分类器或另一个 Evidence Policy；教师模型只参与离线标注，不属于训练产物。

### 6.1 输入渲染

```
[QUESTION]
problem_statement

[CURRENT EVIDENCE]
K 中按固定顺序渲染的有界 Evidence Unit

[CANDIDATE ACTION]
单个 Evidence Unit、两个 Evidence Unit，或 [STOP]
```

冻结长度约束（基于真实数据审计确定）：

```
model_max_length          = 4096   # 完整 (q, K, A) 输入硬上限
question_max_tokens       = 2048   # 问题视图上限
scoreable_unit_max_tokens = 1024    # 单个候选 Evidence Unit 渲染上限
```

Tokenizer 冻结为 `BAAI/bge-reranker-v2-m3`（revision `953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e`）。渲染器先保留候选动作 `A` 的完整正文和 `K` 的全部证据元数据，再把剩余预算动态分配给 `K` 的正文。DataLoader 必须令 Tokenizer 的自动截断失效，任何依赖 `truncation=true` 静默修剪候选正文的实现都属于发布阻断错误。

### 6.2 训练方式

训练样本按状态组织，主损失采用支持多正例的 listwise ranking loss：

```
L_total = L_rank + λ_relation · L_relation
```

- `L_rank`：只读取 `ranking_loss_mask=true` 的状态和其中 `action_loss_mask=true` 的动作；
- `L_relation`：带掩码的多标签 Binary Cross-Entropy，只在拥有可靠义务级关系标签的双证据动作及 `relation_loss_masks.<class>=true` 的类别上计算。

训练按同一个检查点逐步加入更难样本，不在各阶段分别训练不同模型：

```
单证据初始状态预热
-> 状态感知单证据排序
-> 高置信度双证据动作
-> STOP 与关系联合校准
```

## 七、快速开始

### 7.1 环境要求

- Python 3.11+
- Git（用于 `git cat-file`、`git ls-tree` 等 bare 仓库快照操作）
- 教师标注阶段需要 `TEACHER_API_KEY` 或 `DEEPSEEK_API_KEY`（仅 teacher 阶段需要，确定性阶段无需）

### 7.2 构建数据集

唯一构建入口为 [scripts/build_unified_dataset.py](file:///e:/Code_Personal/Subject/evidence-agent/scripts/build_unified_dataset.py)，不使用外部配置文件。

```powershell
# 实验版（JSONL，逐行可审计）
python scripts/build_unified_dataset.py --format jsonl

# 正式发布版（Parquet，通过硬门禁后原子发布）
python scripts/build_unified_dataset.py --format parquet --release
```

常用参数：

| 参数 | 说明 |
|------|------|
| `--format {jsonl,parquet}` | 输出格式，默认 `jsonl` |
| `--release` | 通过硬门禁后原子发布正式版 |
| `--clean-state` | 复核正式文件哈希后删除 SQLite 状态 |
| `--audit-only` | 只运行已完成阶段的审计，不发布 |
| `--through-phase PHASE` | 执行到指定内部阶段后停止 |
| `--self-test` | 运行脚本内置契约与单元测试 |
| `--workers N` | corpus 物化进程数（Windows 无窗口长跑默认 1） |
| `-v / -vv` | 增加日志详细度 |

### 7.3 构建阶段

单次命令按以下内部阶段执行（支持断点续跑）：

```
sources -> normalize -> identity -> split -> snapshots -> corpus
-> supervision -> teacher -> policy -> write -> audit -> publish
```

- **唯一构建状态库**：`data/.build/unified_swe_v1.sqlite3`，承担规范化、关联、断点和审计状态，不属于最终发布数据。
- **临时文件和原子发布**：先写入临时目录 `data/unified_swe_dataset_v1.tmp/`，5 个文件全部通过发布硬门槛后才原子替换为正式目录。失败时旧版本保持不变，下次从 SQLite 最近的兼容阶段继续。
- **状态清理**：构建成功后默认保留 SQLite，便于复现和增量更新；显式 `--clean-state` 只能在正式目录完成哈希复核后执行。

### 7.4 教师标注配置

教师 API 通过环境变量传入，密钥不得写入脚本、SQLite、manifest 或发布文件：

```powershell
# 按 TEACHER_API_KEY、DEEPSEEK_API_KEY 优先级读取
$env:TEACHER_API_KEY = "sk-..."
# 可选：兼容代理
$env:TEACHER_BASE_URL = "https://your-proxy.example.com"
```

`.env` 中的通用 `LLM_MODEL` 不得覆盖冻结模型 `deepseek-v4-flash`。教师阶段环境变量全部缺失时，确定性阶段继续运行，正式发布在教师阶段硬失败，不生成降级数据集。

## 八、仓库目录结构

```
evidence-agent/
├── scripts/
│   ├── build_unified_dataset.py          # 唯一构建入口（当前主入口）
│   ├── 00_audit_local_datasets.py        # 早期：本地数据集审计
│   ├── 01_normalize_local_sources.py     # 早期：原始数据规范化
│   ├── 02_build_master_registry.py       # 早期：Master Instance Registry
│   ├── 03_freeze_splits.py               # 早期：数据集划分
│   ├── 04_prepare_git_snapshots.py       # 早期：Git 快照准备
│   ├── 04b_repair_git_snapshot_failures.py
│   ├── 05_extract_evidence_anchors.py     # 早期：证据 anchor 抽取
│   ├── 06_extract_pre_fix_evidence_units.py  # 早期：修复前 Evidence Unit
│   ├── 07_build_certification_labels.py   # 早期：认证标签
│   ├── 08_build_release_dataset.py        # 早期：MVP 发布
│   └── 11_build_full_repository_corpus.py # 早期：完整仓库语料构建器
├── data/
│   ├── raw/                               # 冻结原始来源（SWE-bench / ContextBench / SWE-Explore）
│   ├── cache/                             # 外部仓库 bare clone 缓存（gitignore）
│   ├── .build/                            # 构建状态 SQLite 和日志
│   ├── processed/                         # 早期中间产物
│   ├── registry/                          # 早期注册表
│   ├── splits/                            # 早期划分
│   ├── release/                           # 早期 MVP 发布（已冻结为 debug dataset）
│   └── unified_swe_dataset_v1/            # 最终发布（5 个文件）
├── docs/superpowers/
│   ├── plans/                             # 实现计划
│   └── specs/                             # 设计规格
├── findings.md                            # 研究发现
├── progress.md                            # 进度跟踪
├── task_plan.md                           # 任务计划
├── 执行方案v4.md
├── 项目创新点_修订版.md
└── 项目进展.md                            # 阶段总结与后续执行路线
```

> 注：`scripts/01_*.py`～`scripts/11_*.py` 是早期分阶段脚本，对应 MVP（候选级证据数据集）。当前正式路径统一收敛到 `scripts/build_unified_dataset.py` 单脚本，不再新增分阶段脚本或外部配置文件。早期 MVP 已冻结为 debug dataset，不能用于正式主实验。

## 九、评测体系

benchmark 支持内部完整版（`gold_visibility=evaluator_only`）和对外脱敏版（`gold_visibility=private`）两种 flavor。所有证据类评测必须分别报告两种候选模式：

- **`oracle_candidate_ranking`**：向候选池注入缺失 Gold，只评估 Cross-Encoder 对已给定候选的排序、组合与 STOP 能力；
- **`end_to_end_online`**：候选只能由问题、当前证据和 pre-fix 仓库生成，不允许 Gold 注入，用于评估规则召回与 Cross-Encoder 组成的完整系统。

主要指标维度：

| 类别 | 指标 |
|------|------|
| 检索 | File Recall@k、Symbol Recall@k、Span Recall@B、MRR、NDCG、Supporting Unit Recall、Core Context Recall |
| 证据 | Evidence Precision/Recall/F1、Mandatory Obligation Coverage、Structural Path Accuracy、Alternative Evidence Recall |
| 证书 | Certificate Exact Match、Certificate Family Recall、Inclusion-Minimal Accuracy、Minimum-Cost Regret、Premature Certification Rate、False EXPAND Rate、ABSTAIN Accuracy |
| 成本 | Evidence tokens、代码行数、Evidence Unit 数量、检索步数、tool calls、GPU latency、LLM/API cost |
| 下游行为 | Resolved Rate、FAIL_TO_PASS、PASS_TO_PASS、Repair-Cost curve、AUC-Repair-Cost |

实验章节只保留对比实验和消融实验两条主线，不增加补丁生成、测试执行或真实修复成功率评测。

## 十、关键设计原则

1. 每个问题只能检索自己的 `snapshot_id`；不能只根据 repo 过滤，同一仓库不同 commit 的代码不能混用。
2. Gold patch、test patch、trajectory、gold context 只能用于离线标签，不能进入在线 Agent 状态。
3. FullRepo 评测时禁止 force-positive，未召回就是 Retriever 失败。
4. 不要把成功轨迹中的所有读取自动视为必要证据（`trajectory 读取过 ≠ 证据必需`）。
5. 未被 Gold 选中的候选只表示未知，只有满足明确反证规则的单元才能标为 hard negative。
6. 同一任务可能存在多组充分证书，正例必须是集合值 `acceptable_next_unit_ids`，不能只有一个 `positive_unit_id`。
7. 不要把百万级 pair 数当成百万独立任务，必须同时报告独立任务数、状态数、动作数和候选数。
8. 教师标签只能进入 weak supervision / semantic audit / unresolved adjudication，不能采用「LLM 生成 gold → 同一个 LLM 判断 gold → 作为真实标签」的闭环。

完整的设计原则与发布硬门槛见 [docs/superpowers/specs/2026-07-30-unified-swe-release-schema-design.md](file:///e:/Code_Personal/Subject/evidence-agent/docs/superpowers/specs/2026-07-30-unified-swe-release-schema-design.md)。

## 十一、文档索引

| 文档 | 内容 |
|------|------|
| [项目进展.md](file:///e:/Code_Personal/Subject/evidence-agent/项目进展.md) | 阶段总结、已完成工作、质量审计与后续执行路线 |
| [项目创新点_修订版.md](file:///e:/Code_Personal/Subject/evidence-agent/项目创新点_修订版.md) | 三项核心贡献的完整论述与公式 |
| [执行方案v4.md](file:///e:/Code_Personal/Subject/evidence-agent/执行方案v4.md) | 执行方案 |
| [findings.md](file:///e:/Code_Personal/Subject/evidence-agent/findings.md) | 研究发现 |
| [progress.md](file:///e:/Code_Personal/Subject/evidence-agent/progress.md) | 进度跟踪 |
| [task_plan.md](file:///e:/Code_Personal/Subject/evidence-agent/task_plan.md) | 任务计划 |
| [docs/superpowers/specs/2026-07-30-unified-swe-release-schema-design.md](file:///e:/Code_Personal/Subject/evidence-agent/docs/superpowers/specs/2026-07-30-unified-swe-release-schema-design.md) | Unified SWE Dataset 发布结构与任务 Schema 设计 |
| [docs/superpowers/specs/2026-07-30-certievidence-data-v2-design.md](file:///e:/Code_Personal/Subject/evidence-agent/docs/superpowers/specs/2026-07-30-certievidence-data-v2-design.md) | CertiEvidence 数据 v2 设计 |

## 十二、许可证与数据来源

数据来源许可证：

| 来源 | 许可证 |
|------|--------|
| SWE-bench | MIT |
| ContextBench | Apache-2.0 |
| SWE-Explore | MIT |

相关上游项目：

- SWE-bench：https://github.com/princeton-nlp/SWE-bench
- ContextBench
- SWE-Explore：https://github.com/Qiushao-E/SWE-Explore-Bench

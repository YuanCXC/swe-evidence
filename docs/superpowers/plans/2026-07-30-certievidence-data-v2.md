# CertiEvidence 数据流水线 v2 实现计划

> **面向 AI 代理的工作者：** 必需子技能：使用 superpowers:subagent-driven-development（推荐）或 superpowers:executing-plans 逐任务实现此计划。步骤使用复选框（`- [ ]`）语法来跟踪进度。

**目标：** 在不覆盖当前 MVP 的前提下，实现多源 SWE 数据的统一注册、完整仓库语料、分层监督、质量审计和 `release_v2` 发布。

**架构：** 新流水线放在 `scripts/v2`，中间产物写入 `data/v2`，正式产物写入 `data/release_v2`。所有来源通过显式 manifest 和独立 adapter 接入，统一映射到稳定的任务、快照和 Evidence Unit 身份。

**技术栈：** Python 3.13、标准库 `unittest`、SQLite、PyArrow、Git CLI、JSONL、Parquet。

---

## 实现前约束

- 当前工作区包含用户尚未提交的 v1 脚本和数据变更。
- 实现前使用 `using-git-worktrees` 创建隔离工作树。
- 不复制、移动或删除当前 `data/processed` 和 `data/release`。
- 每个任务严格执行红灯、绿灯、重构、完整回归测试、原子 commit。

## 文件结构

### 创建

```text
configs/data_sources_v2.json
scripts/v2/__init__.py
scripts/v2/models.py
scripts/v2/source_manifest.py
scripts/v2/identity.py
scripts/v2/splits.py
scripts/v2/snapshots.py
scripts/v2/repository_corpus.py
scripts/v2/supervision.py
scripts/v2/trajectories.py
scripts/v2/labels.py
scripts/v2/audit.py
scripts/v2/release.py
scripts/v2_pipeline.py
scripts/v2/adapters/__init__.py
scripts/v2/adapters/contextbench.py
scripts/v2/adapters/swebench.py
scripts/v2/adapters/swe_explore.py
scripts/v2/adapters/nebius.py
scripts/v2/adapters/swe_gym.py
scripts/v2/adapters/swe_rebench.py
scripts/v2/adapters/swe_smith.py
tests/v2/__init__.py
tests/v2/test_source_manifest.py
tests/v2/test_contextbench_adapter.py
tests/v2/test_swebench_adapter.py
tests/v2/test_identity.py
tests/v2/test_splits.py
tests/v2/test_repository_corpus.py
tests/v2/test_supervision.py
tests/v2/test_trajectories.py
tests/v2/test_labels.py
tests/v2/test_audit.py
tests/v2/test_release.py
```

### 复用但暂不修改

```text
scripts/04_prepare_git_snapshots.py
scripts/04b_repair_git_snapshot_failures.py
scripts/11_build_full_repository_corpus.py
```

先用回归测试锁定这些脚本的关键行为，再把可复用逻辑提取到 `scripts/v2`。v1 CLI 保持可运行。

## 任务 1：数据源 manifest 与文件角色

**文件：**
- 创建：`configs/data_sources_v2.json`
- 创建：`scripts/v2/source_manifest.py`
- 创建：`tests/v2/test_source_manifest.py`

- [ ] **步骤 1：编写失败测试**

```python
import unittest
from pathlib import Path

from scripts.v2.source_manifest import load_source_manifest


class SourceManifestTest(unittest.TestCase):
    def test_contextbench_has_exactly_one_task_table(self):
        manifest = load_source_manifest(
            Path("configs/data_sources_v2.json")
        )
        files = manifest.sources["contextbench"].files
        task_tables = [item for item in files if item.role == "task_table"]
        self.assertEqual([item.path for item in task_tables], [
            "data/raw/contextbench/full.parquet"
        ])

    def test_eval_sources_are_not_trainable(self):
        manifest = load_source_manifest(
            Path("configs/data_sources_v2.json")
        )
        for source in manifest.sources.values():
            if source.role == "eval_only":
                self.assertFalse(source.trainable)
```

- [ ] **步骤 2：运行测试并确认红灯**

运行：

```powershell
python -m unittest tests.v2.test_source_manifest -v
```

预期：`ModuleNotFoundError: No module named 'scripts.v2'`。

- [ ] **步骤 3：实现最小 manifest 模型和加载器**

`source_manifest.py` 提供：

```python
@dataclass(frozen=True)
class SourceFile:
    path: str
    role: str
    required: bool


@dataclass(frozen=True)
class SourceSpec:
    name: str
    role: str
    trainable: bool
    trust_tier: str
    files: list[SourceFile]


@dataclass(frozen=True)
class SourceManifest:
    schema_version: str
    sources: dict[str, SourceSpec]

```

实现 `load_source_manifest(path: Path) -> SourceManifest`：读取 JSON，拒绝未知
`schema_version`，逐项构造 `SourceFile`、`SourceSpec` 和 `SourceManifest`，并在
required 文件不存在、角色重复或必填来源缺失时抛出 `ManifestValidationError`。
配置必须显式列出 ContextBench 的主表、overlay、variant 和 metadata 文件，并列出
每个新增数据源的本地路径、远程 revision、license 和用途。

- [ ] **步骤 4：运行测试确认绿灯**

```powershell
python -m unittest tests.v2.test_source_manifest -v
```

预期：2 个测试全部 `OK`。

- [ ] **步骤 5：提交**

```powershell
git add configs/data_sources_v2.json scripts/v2 tests/v2/test_source_manifest.py
git commit -m "feat(数据源): 添加 v2 数据源清单"
```

## 任务 2：ContextBench 单任务表与 overlay

**文件：**
- 创建：`scripts/v2/models.py`
- 创建：`scripts/v2/adapters/contextbench.py`
- 创建：`tests/v2/test_contextbench_adapter.py`

- [ ] **步骤 1：编写失败测试**

测试使用临时目录生成 3 条主记录、2 条 verified overlay 和 2 条 CSV metadata：

```python
class ContextBenchAdapterTest(unittest.TestCase):
    def test_overlay_does_not_create_duplicate_tasks(self):
        result = normalize_contextbench(self.fixture_paths)
        self.assertEqual(len(result.tasks), 3)
        self.assertEqual(len(result.overlays), 4)
        self.assertEqual(result.tasks[0].source_name, "contextbench")

    def test_verified_context_has_highest_precedence(self):
        result = normalize_contextbench(self.fixture_paths)
        task = next(item for item in result.tasks if item.source_instance_id == "cb-1")
        variants = result.overlays_by_instance["cb-1"]
        preferred = choose_preferred_context(variants)
        self.assertEqual(preferred.variant_name, "verified_train")

    def test_orphan_overlay_is_quarantined(self):
        result = normalize_contextbench(self.fixture_paths_with_orphan)
        self.assertEqual(result.quarantined[0].reason, "missing_primary_task")
```

- [ ] **步骤 2：运行测试并确认红灯**

```powershell
python -m unittest tests.v2.test_contextbench_adapter -v
```

预期：导入 `normalize_contextbench` 失败。

- [ ] **步骤 3：实现 adapter**

核心 API：

```python
def normalize_contextbench(paths: ContextBenchPaths) -> AdapterResult:
    primary = read_parquet(paths.full)
    tasks = [normalize_primary(row) for row in primary]
    known_ids = {task.source_instance_id for task in tasks}
    overlays = []
    quarantined = []
    for variant_name, path in paths.overlay_files():
        for row in read_records(path):
            overlay = normalize_overlay(variant_name, row)
            if overlay.source_instance_id not in known_ids:
                quarantined.append(
                    QuarantinedRecord(
                        source_name="contextbench",
                        source_record_id=overlay.source_record_id,
                        reason="missing_primary_task",
                    )
                )
                continue
            overlays.append(overlay)
    return AdapterResult(
        tasks=tasks,
        overlays=overlays,
        quarantined=quarantined,
    )
```

优先级固定为：

```python
CONTEXT_VARIANT_PRIORITY = {
    "verified_train": 400,
    "verified_test": 400,
    "verified": 300,
    "train": 200,
    "test": 200,
    "full": 100,
}
```

- [ ] **步骤 4：运行 adapter 测试和全量测试**

```powershell
python -m unittest tests.v2.test_contextbench_adapter -v
python -m unittest discover -s tests/v2 -v
```

预期：全部 `OK`，没有重复任务。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/models.py scripts/v2/adapters/contextbench.py tests/v2/test_contextbench_adapter.py
git commit -m "fix(ContextBench): 消除多文件重复任务"
```

## 任务 3：SWE-bench、Oracle 与 BM25 对齐

**文件：**
- 创建：`scripts/v2/adapters/swebench.py`
- 创建：`tests/v2/test_swebench_adapter.py`

- [ ] **步骤 1：编写失败测试**

覆盖原始任务、Oracle、3 个 BM25 budget、缺失和冲突：

```python
class SweBenchAdapterTest(unittest.TestCase):
    def test_retrieval_variants_attach_to_one_task(self):
        task, overlays = normalize_swebench_bundle(self.bundle)
        self.assertEqual(task.source_instance_id, "repo__name-42")
        self.assertEqual(
            {item.variant_name for item in overlays},
            {"oracle", "bm25_13k", "bm25_27k", "bm25_40k"},
        )

    def test_mismatched_base_commit_is_quarantined(self):
        result = normalize_swebench(self.bundle_with_commit_conflict)
        self.assertEqual(result.quarantined[0].reason, "base_commit_conflict")
```

- [ ] **步骤 2：运行并确认红灯**

```powershell
python -m unittest tests.v2.test_swebench_adapter -v
```

预期：缺少 `normalize_swebench_bundle`。

- [ ] **步骤 3：实现对齐**

严格使用：

```text
instance_id
+ normalized repo
+ normalized base_commit
```

Oracle 和 BM25 的 `text` 原样保存为离线 overlay，同时解析其文件块作为可追溯 annotation。解析失败时保留原始文本并记录 `parse_status=failed`，不得丢弃任务。

- [ ] **步骤 4：验证**

```powershell
python -m unittest tests.v2.test_swebench_adapter -v
python scripts/v2_pipeline.py normalize --source swebench --dry-run
```

预期：本地完整数据 dry-run 报告原始任务 21,527 条，retrieval overlay 覆盖率写入报告。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/adapters/swebench.py tests/v2/test_swebench_adapter.py
git commit -m "feat(SWE-bench): 对齐 Oracle 与 BM25 上下文"
```

## 任务 4：统一身份和冲突隔离

**文件：**
- 创建：`scripts/v2/identity.py`
- 创建：`tests/v2/test_identity.py`

- [ ] **步骤 1：编写失败测试**

```python
class IdentityResolverTest(unittest.TestCase):
    def test_same_snapshot_does_not_imply_same_task(self):
        left = record(issue_url="https://github.com/o/r/issues/1")
        right = record(issue_url="https://github.com/o/r/issues/2")
        result = resolve_identities([left, right])
        self.assertEqual(len(result.master_instances), 2)
        self.assertEqual(
            result.master_instances[0].snapshot_key,
            result.master_instances[1].snapshot_key,
        )

    def test_explicit_original_id_merges_sources(self):
        result = resolve_identities([
            record(source_name="swebench", source_instance_id="x"),
            record(source_name="contextbench", original_inst_id="x"),
        ])
        self.assertEqual(len(result.master_instances), 1)

    def test_conflicting_strong_keys_are_quarantined(self):
        result = resolve_identities(self.conflicting_records)
        self.assertEqual(result.conflicts[0].severity, "high")
```

- [ ] **步骤 2：运行并确认红灯**

```powershell
python -m unittest tests.v2.test_identity -v
```

- [ ] **步骤 3：实现确定性 resolver**

公开函数为 `normalize_repo(value: str) -> str`、
`normalize_commit(value: str) -> str`、
`build_source_record_id(record: SourceRecord) -> str`、
`build_canonical_instance_id(records: Sequence[SourceRecord]) -> str`、
`build_task_group_id(records: Sequence[SourceRecord]) -> str`、
`build_snapshot_id(repo: str, resolved_commit: str) -> str` 和
`resolve_identities(records: Iterable[SourceRecord]) -> IdentityResult`。

匹配规则严格按照设计文档排序，输出 `match_method`、`match_confidence` 和冲突记录。

- [ ] **步骤 4：验证确定性**

```powershell
python -m unittest tests.v2.test_identity -v
python scripts/v2_pipeline.py registry --input-order normal --dry-run
python scripts/v2_pipeline.py registry --input-order reversed --dry-run
```

预期：两次 dry-run 的 registry SHA-256 相同。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/identity.py tests/v2/test_identity.py
git commit -m "feat(注册表): 添加确定性身份解析"
```

## 任务 5：防泄漏 split

**文件：**
- 创建：`scripts/v2/splits.py`
- 创建：`tests/v2/test_splits.py`

- [ ] **步骤 1：编写失败测试**

```python
class SplitAssignmentTest(unittest.TestCase):
    def test_official_evaluation_split_wins(self):
        assignments = freeze_splits(self.records, seed=20260730)
        self.assertEqual(assignments["verified-derived"].split, "eval_only")

    def test_task_group_never_crosses_split(self):
        assignments = freeze_splits(self.records, seed=20260730)
        by_group = defaultdict(set)
        for item in assignments.values():
            by_group[item.task_group_id].add(item.split)
        self.assertTrue(all(len(values) == 1 for values in by_group.values()))

    def test_split_is_deterministic(self):
        self.assertEqual(
            freeze_splits(self.records, seed=20260730),
            freeze_splits(reversed(self.records), seed=20260730),
        )
```

- [ ] **步骤 2：运行并确认红灯**

```powershell
python -m unittest tests.v2.test_splits -v
```

- [ ] **步骤 3：实现 split 优先级**

```python
SPLIT_PRIORITY = {
    "eval_only": 500,
    "test_sufficiency": 400,
    "test_retrieval": 300,
    "dev": 200,
    "train": 100,
    "train_weak": 50,
}
```

先传播官方 eval 标记，再按 `task_group_id` 分组，最后对新增训练源做确定性 repo-disjoint 划分。

- [ ] **步骤 4：验证**

```powershell
python -m unittest tests.v2.test_splits -v
python scripts/v2_pipeline.py split --dry-run
```

预期：`task_group_leakage=0`、`eval_repo_leakage=0`。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/splits.py tests/v2/test_splits.py
git commit -m "feat(切分): 添加任务组级防泄漏策略"
```

## 任务 6：快照与完整仓库语料迁移

**文件：**
- 创建：`scripts/v2/snapshots.py`
- 创建：`scripts/v2/repository_corpus.py`
- 创建：`tests/v2/test_repository_corpus.py`
- 修改：`scripts/11_build_full_repository_corpus.py`

- [ ] **步骤 1：先为现有关键行为写回归测试**

测试：

- inventory 使用 `GIT_NO_LAZY_FETCH=1`。
- `snapshot_id` 唯一绑定 repo/commit。
- 空 inventory 状态禁止 extract。
- 相同 blob 只产生一份 `content_unit`。
- `snapshot_unit` 和结构边不跨 snapshot。

- [ ] **步骤 2：运行测试并确认缺少 v2 API**

```powershell
python -m unittest tests.v2.test_repository_corpus -v
```

- [ ] **步骤 3：提取可复用逻辑**

`repository_corpus.py` 提供
`inventory_snapshots(config: CorpusConfig) -> InventoryReport`、
`extract_content_units(config: CorpusConfig) -> ExtractionReport`、
`export_repository_corpus(config: CorpusConfig) -> ExportReport` 和
`validate_corpus(path: Path) -> CorpusValidation`。

现有 `11_build_full_repository_corpus.py` 改为参数解析和上述 API 的兼容包装器，不改变 v1 默认输出路径。

- [ ] **步骤 4：运行单测和小型 Git fixture 集成测试**

```powershell
python -m unittest tests.v2.test_repository_corpus -v
python scripts/v2_pipeline.py corpus --fixture tests/fixtures/git --phase all
```

预期：fixture corpus 验证通过，重复 blob 物理去重，跨 snapshot 引用为 0。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/snapshots.py scripts/v2/repository_corpus.py scripts/11_build_full_repository_corpus.py tests/v2/test_repository_corpus.py
git commit -m "refactor(语料): 提取 v2 完整仓库构建模块"
```

## 任务 7：监督和轨迹映射

**文件：**
- 创建：`scripts/v2/supervision.py`
- 创建：`scripts/v2/trajectories.py`
- 创建：`scripts/v2/adapters/swe_explore.py`
- 创建：`scripts/v2/adapters/nebius.py`
- 创建：`tests/v2/test_supervision.py`
- 创建：`tests/v2/test_trajectories.py`

- [ ] **步骤 1：编写映射优先级测试**

覆盖精确 span、symbol、内容哈希、重命名和模糊匹配。断言模糊匹配不能生成 `strong_support`。

- [ ] **步骤 2：编写轨迹事件测试**

```python
class TrajectoryNormalizationTest(unittest.TestCase):
    def test_success_and_failure_are_preserved(self):
        events = normalize_nebius_trajectory(self.raw_trajectory)
        self.assertEqual(events.outcome, "resolved")
        self.assertTrue(any(item.action == "read" for item in events.events))

    def test_failed_read_is_not_semantic_negative(self):
        annotation = event_to_annotation(self.failed_read)
        self.assertEqual(annotation.label, "behavior_negative")
        self.assertNotEqual(annotation.label, "semantic_negative")
```

- [ ] **步骤 3：实现映射器和事件规范化**

统一事件类型：

```text
search
read
navigate
edit
test
submit
stop
error
```

每个事件保存 `step_index`、`tool_name`、`file_path`、`line_range`、`logical_unit_ids`、`outcome` 和 provenance。

- [ ] **步骤 4：验证**

```powershell
python -m unittest tests.v2.test_supervision tests.v2.test_trajectories -v
python scripts/v2_pipeline.py map-supervision --dry-run
```

预期：报告分别展示各来源文件级、span 级和事件级映射率。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/supervision.py scripts/v2/trajectories.py scripts/v2/adapters tests/v2/test_supervision.py tests/v2/test_trajectories.py
git commit -m "feat(监督): 映射多源证据与 Agent 轨迹"
```

## 任务 8：标签、证据图和训练样本

**文件：**
- 创建：`scripts/v2/labels.py`
- 创建：`tests/v2/test_labels.py`

- [ ] **步骤 1：编写失败测试**

覆盖：

- 多来源标签不被压平成单值。
- 未选中候选不自动成为负例。
- 失败轨迹只提供行为反例。
- certificate 覆盖所有 mandatory obligations。
- 保留多组等价 certificate。

- [ ] **步骤 2：运行并确认红灯**

```powershell
python -m unittest tests.v2.test_labels -v
```

- [ ] **步骤 3：实现标签模型**

```python
class SupportLevel(StrEnum):
    STRONG = "strong_support"
    SUPPORT = "support"
    WEAK = "weak_support"
    UNRESOLVED = "unresolved"
    CONTRADICTED = "contradicted"
```

实现 `build_reference_evidence`、`build_evidence_graph`、`build_obligations`、
`enumerate_certificate_families`、`build_retrieval_pairs` 和
`build_policy_states`。每个函数接收显式的输入表路径和输出路径，返回包含行数、
跳过数、冲突数及输出哈希的 `BuildReport`；禁止通过模块级变量隐式读取数据。

- [ ] **步骤 4：验证防塌缩统计**

```powershell
python -m unittest tests.v2.test_labels -v
python scripts/v2_pipeline.py build-labels --fixture tests/fixtures/labels
```

预期：fixture 中正例比例、单单元证书比例和多证书数量与断言一致。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/labels.py tests/v2/test_labels.py
git commit -m "feat(标签): 构建分层证据与多组证书"
```

## 任务 9：质量审计

**文件：**
- 创建：`scripts/v2/audit.py`
- 创建：`tests/v2/test_audit.py`

- [ ] **步骤 1：编写每个硬失败条件的独立测试**

至少覆盖：

```text
task_group 跨 split
在线字段泄漏
标签悬空引用
跨 snapshot 候选
高严重度冲突进入 release
manifest 哈希不一致
```

- [ ] **步骤 2：运行并确认红灯**

```powershell
python -m unittest tests.v2.test_audit -v
```

- [ ] **步骤 3：实现审计器**

实现 `audit_registry`、`audit_splits`、`audit_snapshot_isolation`、
`audit_visibility`、`audit_label_references` 和 `audit_distribution`，统一返回
`AuditCheck`。实现 `build_audit_sample`，默认随机种子为 `20260730`，按四个来源层
各抽取 100 条并写出稳定排序的 JSONL。

命令返回码：

```text
0 = passed
1 = passed_with_warnings
2 = failed
```

- [ ] **步骤 4：验证错误码和报告**

```powershell
python -m unittest tests.v2.test_audit -v
python scripts/v2_pipeline.py audit --fixture tests/fixtures/valid_release
```

预期：有效 fixture 返回 0，泄漏 fixture 返回 2。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/audit.py tests/v2/test_audit.py
git commit -m "feat(审计): 添加数据泄漏与质量门禁"
```

## 任务 10：三文件发布和可复现构建

**文件：**
- 创建：`scripts/v2/release.py`
- 创建：`scripts/v2_pipeline.py`
- 创建：`tests/v2/test_release.py`

- [ ] **步骤 1：编写失败测试**

```python
class ReleaseBuilderTest(unittest.TestCase):
    def test_release_contains_exactly_three_files(self):
        build_release(self.inputs, self.output)
        self.assertEqual(
            {path.name for path in self.output.iterdir()},
            {"benchmark.jsonl", "repository_corpus.parquet", "manifest.json"},
        )

    def test_online_input_has_no_forbidden_fields(self):
        build_release(self.inputs, self.output)
        for row in read_jsonl(self.output / "benchmark.jsonl"):
            self.assertFalse(FORBIDDEN_ONLINE_FIELDS & row["input"].keys())

    def test_second_build_is_byte_identical(self):
        build_release(self.inputs, self.first)
        build_release(self.inputs, self.second)
        self.assertEqual(tree_hash(self.first), tree_hash(self.second))
```

- [ ] **步骤 2：运行并确认红灯**

```powershell
python -m unittest tests.v2.test_release -v
```

- [ ] **步骤 3：实现发布器和统一 CLI**

CLI 子命令固定为：

```text
audit-sources
normalize
registry
split
snapshots
corpus
map-supervision
build-labels
build-training
audit
release
all
```

发布器使用在线字段允许列表，构建到临时目录，通过审计后原子移动到 `data/release_v2`。

- [ ] **步骤 4：运行全量单测和 fixture 端到端测试**

```powershell
python -m unittest discover -s tests/v2 -v
python scripts/v2_pipeline.py all --fixture tests/fixtures/end_to_end
python scripts/v2_pipeline.py audit --release data/release_v2
```

预期：全部单测通过；fixture 仅生成 3 个发布文件；审计返回 0。

- [ ] **步骤 5：提交**

```powershell
git add scripts/v2/release.py scripts/v2_pipeline.py tests/v2/test_release.py
git commit -m "feat(发布): 生成可复现的三文件数据集"
```

## 任务 11：真实数据分阶段迁移

**文件：**
- 生成：`data/v2/reports/*.json`
- 生成：`data/v2/registry/*`
- 生成：`data/v2/splits/*`
- 生成：`data/v2/corpus/*`
- 生成：`data/v2/labels/*`
- 生成：`data/release_v2/*`

- [ ] **步骤 1：冻结 v1 并保存哈希**

```powershell
python scripts/v2_pipeline.py audit-sources --freeze-v1
```

预期：只生成 `data/v2/reports/v1_inventory.json` 和哈希清单。

- [ ] **步骤 2：只接入现有数据与 Oracle/BM25**

```powershell
python scripts/v2_pipeline.py normalize --sources swebench contextbench swe_explore
python scripts/v2_pipeline.py registry
python scripts/v2_pipeline.py split
```

验收：SWE-bench 为 21,527 条，ContextBench 主任务为 1,136 条，ContextBench 重复任务为 0。

- [ ] **步骤 3：构建完整仓库语料**

```powershell
python scripts/v2_pipeline.py snapshots
python scripts/v2_pipeline.py corpus --phase inventory
python scripts/v2_pipeline.py corpus --phase extract
python scripts/v2_pipeline.py corpus --phase export
```

验收：snapshot、membership、extraction 和失败统计通过审计。

- [ ] **步骤 4：依次接入轨迹和扩展任务**

按顺序接入：

```text
Nebius trajectories
SWE-Gym
SWE-rebench V2 Python 子集
SWE-smith 弱监督
```

每接入一个来源都重新运行 source、identity、split 和 mapping 审计。任一来源失败时，不影响前一个已通过版本。

- [ ] **步骤 5：构建正式 v2 并记录版本**

```powershell
python scripts/v2_pipeline.py build-labels
python scripts/v2_pipeline.py build-training
python scripts/v2_pipeline.py audit
python scripts/v2_pipeline.py release --version 2.0.0
```

验收：3 个发布文件生成，所有硬门禁通过，manifest 记录所有来源 revision 和文件哈希。

- [ ] **步骤 6：提交代码和小型报告，不提交大数据**

```powershell
git add configs scripts tests docs data/v2/reports
git commit -m "feat(数据集): 完成 CertiEvidence v2 流水线"
```

大数据文件通过独立数据发布渠道或 Git LFS 管理，不直接加入普通 Git 历史。

## 完成定义

- `python -m unittest discover -s tests/v2 -v` 全部通过。
- ContextBench 仅有 1,136 个主任务。
- SWE-bench 21,527 条任务与 Oracle/BM25 对齐报告可审计。
- `task_group_id` 和评测仓库泄漏均为 0。
- `repository_corpus.parquet` 通过 snapshot 隔离和引用完整性检查。
- 在线输入泄漏数为 0。
- 标签悬空引用数为 0。
- 同一输入和版本重复构建得到相同文件哈希。
- `data/release_v2` 只包含 3 个核心文件。

# 发现与决策

## 需求
- 重新设计项目的数据处理流水线。
- 扩大训练规模，同时保留 ContextBench 的高质量证据标签价值。
- 最终产出 `benchmark.jsonl`、`repository_corpus.parquet` 和 `manifest.json`。
- 方案必须覆盖去重、防泄漏、可复现、质量审计和迁移顺序。

## 本地项目发现
- 原始 SWE-bench 共 21,527 条任务。
- ContextBench 实际唯一实例为 1,136 条，但当前目录中的 7 个文件被全部扫描，规范化阶段读入 3,136 条记录。
- SWE-Explore 有 848 条，其中 451 条可与 SWE-bench 对齐，397 条缺少当前通用解析器需要的核心字段。
- 当前 Master Registry 有 23,060 条，满足核心字段要求的可用实例有 22,662 条。
- 当前候选级 release 有 21,982 条任务，但属于 Gold 对齐后的 debug corpus。
- 当前自动认证率为 99.986%，单单元证书比例为 83.864%，不适合作为正式充分性 Gold。
- 完整仓库语料 `repository_corpus.parquet` 尚未生成。
- 当前 108 个仓库缓存和冻结的 `BAAI/bge-reranker-v2-m3` Tokenizer 已存在。
- 当前没有 `scripts/build_unified_dataset.py`，文档规定的五个最终文件均未生成。
- `data/release/` 中的三文件产物是旧 CertiEvidence MVP，不能冒充 Unified SWE Dataset。
- DeepSeek、OpenAI-compatible 和通用教师环境变量均未配置；15,000 个有效教师包因此暂时无法真实生成。
- 当前可用关键依赖为 PyArrow 21.0.0、Transformers 5.5.4、HTTPX 0.28.1 和 rank-bm25 0.2.2；未安装 tree-sitter，结构抽取必须使用标准库 AST 和确定性窗口降级，或在脚本内明确下载依赖后再冻结版本。
- 对 1.06 GB `evidence_units.jsonl` 的身份审计确认：它包含 295,377 个候选级 Evidence Unit，来源是 65,711 个 `candidate_file`，旧 Manifest 明确将其范围标为 `aligned_candidate_corpus`，不是完整 repository corpus。
- 当前 Evidence Unit 每行重复保存正文并绑定单个 `canonical_instance_id + snapshot_id`；最终规范要求 `repository_corpus` 每行保存一个 `repo + path + blob_oid` 唯一文件版本，并在行内嵌套不重复正文的 Evidence Unit，因此当前物理粒度不符合发布要求。
- 当前文件记录缺少 `blob_oid`、`snapshot_ids`、完整文件正文、imports、attributes 和 extraction struct；当前 Evidence Unit 缺少 `qualified_name`、`parent_evidence_id`、冻结 Tokenizer 的 `token_count`、`rendered_token_count` 与 `scoreable`。
- 295,377 个当前单元全部标记为 `offline_supervision`；锚点来源为 155,817 个 Gold patch、11,974 个 test patch 和 10,927 个 ContextBench 外部上下文信号，上游报告明确禁止它们进入在线输入。
- 冻结 Tokenizer 对全部当前正文的实测下界为：mean 718.72、p50 411、p90 1,673、p95 3,260、p99 4,965、max 57,884；仅正文就有 50,198 个单元（16.9946%）超过 1,024 Token，必须继续切分或设为不可评分。
- 当前 unit type 中有 104,716 行（35.45%）不属于最终枚举，主要是 `line_window`、`async_function` 和 `interface`；同时完全缺少规范强制的文件级 Evidence Unit。
- 当前 295,377 个 `evidence_unit_id` 虽然物理唯一，但按 `repo + path + file_sha256 + lines + content_sha256` 只有 286,172 个逻辑单元，至少 9,205 行是跨任务重复身份，不能直接沿用当前 ID。
- 冻结真实来源已由新脚本重新规范化：SWE-bench 21,527 个唯一任务映射为 19,008 train、225 validation、2,294 benchmark；ContextBench 严格对齐 351 个，SWE-Explore 严格对齐 451 个，两者并集 519、交集 283，overlay 没有新增任务。
- 规范化后共有 18,527 个 `repo + base_commit` 唯一快照；任务 ID、任务组均无重复，且没有任务组跨物理 split。
- 唯一 SQLite 构建状态当前约 633 MB，它不是发布文件；最终仍只发布主设计规定的 4 个数据文件和 `manifest.json`。

## 外部数据研究发现
- SWE-bench 官方 Oracle/BM25 版本覆盖 21,527 条任务，可补充 Oracle 文件和 BM25 上下文。
- Nebius `SWE-bench-extra` 约有 6.4K 条任务，`SWE-agent-trajectories` 有 80,036 条轨迹，其中 13,389 条成功、66,647 条失败。
- SWE-rebench V2 有约 32.1K 条真实、可执行任务，覆盖 20 种语言。
- SWE-Gym 有 2,438 条真实 Python 任务，并提供可执行环境。
- SWE-smith 的官方任务集约有 50K 条任务，另有 5,017 条专家轨迹；任务主要为合成故障。
- SweLoc 的构建流程基于 SWE-bench，提供 Issue 到修改函数的定位监督和仓库内 hard negatives。
- MULocBench 有 1,100 条定位任务，覆盖代码、测试、配置、文档、资产和外部依赖。

## 技术决策
| 决策 | 理由 |
|------|------|
| 数据分为强监督、中等监督、轨迹监督、可执行行为监督、弱监督 5 层 | 不把不同可信度的数据强行合并成单一 Gold |
| Oracle/BM25 与现有 SWE-bench 直接按 `instance_id + repo + base_commit` 对齐 | 接入成本最低，覆盖现有全部任务 |
| 轨迹保留成功和失败两类 | 失败轨迹可提供无效读取、循环探索和错误 STOP 的反例 |
| 使用 `task_group_id` 和 `snapshot_id` 双层隔离 | 前者防止同一 Issue/PR 跨 split，后者防止不同版本代码混检索 |
| 所有标签映射到稳定的 `logical_unit_id` | 解除标签与某次抽取运行生成 ID 的耦合 |
| 正式 release 不发布构建状态库和 Gold 原始资产 | 降低泄漏风险和发布复杂度 |

## 遇到的问题
| 问题 | 解决方案 |
|------|---------|
| ContextBench 同一实例存在多个文件版本 | 主表只产生一次任务，其他版本按身份覆盖或附加标签 |
| 不同数据源可能包含相同 Issue/PR | 建立 source alias、PR URL、repo/commit、patch hash 多级匹配 |
| Agent 轨迹不是人工 Gold | 标记为行为弱监督，保留模型、框架、成功状态和许可证 |
| Oracle 来源于 Gold patch | 只作为离线监督和评测上界，禁止进入在线输入 |
| 任务记录数量与训练 pair 数容易混淆 | manifest 同时报告独立任务数、状态数、pair 数和候选数 |
| 主设计要求 15,000 个教师包，但环境没有 API 配置 | 构建器允许在教师阶段前断点续跑；缺少配置时禁止正式发布，不用规则标签伪造教师标签 |

## 资源
- SWE-bench 官方数据说明：https://www.swebench.com/SWE-bench/guides/datasets/
- SWE-bench BM25 13K：https://huggingface.co/datasets/princeton-nlp/SWE-bench_bm25_13K
- Nebius SWE-bench-extra：https://huggingface.co/datasets/nebius/SWE-bench-extra
- Nebius Agent 轨迹：https://huggingface.co/datasets/nebius/SWE-agent-trajectories
- SWE-rebench V2：https://huggingface.co/collections/nebius/swe-rebench-v2
- SWE-Gym：https://github.com/SWE-Gym/SWE-Gym
- SWE-smith：https://huggingface.co/datasets/SWE-bench/SWE-smith
- SweRank/SweLoc：https://github.com/SalesforceAIResearch/SweRank
- MULocBench：https://arxiv.org/abs/2509.25242

## 视觉/浏览器发现
- 未使用视觉材料。

---
*每执行 2 次查看、浏览器或搜索操作后更新此文件。*

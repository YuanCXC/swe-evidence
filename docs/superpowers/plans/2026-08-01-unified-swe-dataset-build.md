# Unified SWE Dataset 单脚本实现计划

> **面向 AI 代理的工作者：** 使用 `executing-plans` 在当前工作区内联执行；用户明确禁止创建分支。所有生产实现和契约自测都必须位于同一个脚本中。

**目标：** 实现 `scripts/build_unified_dataset.py`，从 `data/raw/` 和 `data/cache/repos/` 一次生成符合主设计文档的 JSONL 实验版或五文件 Parquet 正式版。

**架构：** 单脚本内按来源、身份、快照、语料、监督、教师、状态动作、发布审计划分函数区；所有可恢复状态写入 `data/.build/unified_swe_v1.sqlite3`。发布先写同格式临时目录，全部硬门禁通过后原子替换正式目录。教师配置缺失时允许完成教师阶段之前的可恢复工作，但禁止生成正式 release。

**技术栈：** Python 3、PyArrow 21.0.0、Transformers 5.5.4、SQLite、Git plumbing、HTTPX 0.28.1、rank-bm25 0.2.2。

**规范来源：** `docs/superpowers/specs/2026-07-30-unified-swe-release-schema-design.md`。规范与本计划冲突时，以规范为准。

---

## 文件职责

- 创建：`scripts/build_unified_dataset.py`——唯一构建入口、全部阶段实现、CLI、内置契约测试和发布审计。
- 修改：`task_plan.md`、`findings.md`、`progress.md`——仅记录执行状态，不参与构建。
- 不创建：额外 Python 模块、配置文件或独立测试文件。

## 任务 1：CLI、常量与契约自测框架

- [x] 先在唯一脚本中加入 `unittest.TestCase`，断言五个文件名、三个 split 数量、RRF 参数、Token 上限、证据预算、教师包数量及稳定 ID 行为。
- [x] 运行 `python scripts/build_unified_dataset.py --self-test`，确认因生产符号尚未定义而失败。
- [x] 实现版本化常量、CLI、稳定 JSON/哈希/ID 和 `--self-test` 分派，使第一组测试通过。
- [x] 运行 `python scripts/build_unified_dataset.py --self-test -v`，结果为 7/7 通过，退出码为 0。

## 任务 2：来源校验、规范化、身份与冻结切分

- [x] 先加入最小 Parquet/JSONL fixture 测试，覆盖 SWE-bench 唯一任务、ContextBench overlay 去重、SWE-Explore 精确 ID 对齐、Gold 不进入 `input` 和固定 split；无证书删除由任务 4 的真实监督映射测试覆盖。
- [x] 运行自测并确认新增测试因加载函数缺失而失败。
- [x] 实现来源文件发现、固定来源元数据、缺失下载、SHA-256、SQLite 表初始化、任务合并和切分写入。
- [x] 在真实来源上运行 `--through-phase split --audit-only`，验证原始数量为 19,008/225/2,294，overlay 不新增任务。

## 任务 3：Git 快照、唯一文件版本与 Evidence Unit

- [ ] 先加入临时 bare Git 仓库测试，覆盖 `repo + path + blob_oid` 去重、snapshot-path 唯一成员关系、正文哈希、二进制/vendor/generated 控制、文件级 unit 不可评分和超长单元继续分块。
- [ ] 运行自测并确认新增测试失败。
- [ ] 实现仓库缓存定位、commit/tree/blob 校验、流式 Git inventory、Python AST 与确定性窗口降级、import 声明提取、冻结 Tokenizer 计数和 SQLite 断点。
- [ ] 在真实缓存上运行 `--through-phase corpus --audit-only`，验证 21,527 个 snapshot 可解析、引用与正文哈希一致，缺失快照为 0。

## 任务 4：确定性监督、义务与 Witness Group

- [ ] 先加入合成 patch、ContextBench Gold 和 SWE-Explore 轨迹 fixture，覆盖旧侧 patch 映射、包含区间合并、整文件读取不展开、AND/OR witness、mandatory 条件和 674 个删除任务冻结。
- [ ] 运行自测并确认新增测试失败。
- [ ] 实现 patch/测试旧侧解析、overlay 映射、证据对齐、义务、witness、label provenance 和轨迹前缀。
- [ ] 在真实数据运行 `--through-phase supervision --audit-only`，验证最终任务数量、删除原因分布和 Evidence Unit 引用完整率。

## 任务 5：在线候选、状态、动作与数值标签

- [ ] 先加入合成候选图测试，覆盖四通道等权 RRF、通道缺失不归一化、最终排序、在线/离线隔离、pair 配额、STOP、C/P、交互值、Pareto、unknown mask 和 32,768/64 预算。
- [ ] 运行自测并确认新增测试失败。
- [ ] 实现代码感知词法召回、路径/符号通道、结构扩展、RRF、状态构造、pair、程序标签和模型输入渲染计数。
- [ ] 重放所有发布状态的在线候选，要求候选 ID、来源、融合名次和分数完全一致。

## 任务 6：15,000 个受约束教师包

- [ ] 先加入本地假 HTTP 教师响应测试，覆盖 JSON Schema、引用白名单、snapshot 一致性、技术重试、确定性冲突优先、缓存幂等和拒绝后同层替补。
- [ ] 运行自测并确认新增测试失败。
- [ ] 实现 DeepSeek/OpenAI-compatible 教师客户端；密钥、地址和模型只从环境变量读取，原始响应只存 SQLite。
- [ ] 真实运行必须接受 12,000 个 train 主包、1,784 个 validation 包和 1,216 个 train 稀有关系包；缺少配置或有效包不足时退出非零并禁止发布。

## 任务 7：统一 Schema、流式写出与发布硬门禁

- [ ] 先加入小型端到端 fixture 测试，覆盖 JSONL/Parquet 逻辑等价、共同 Schema、Manifest 哈希、临时目录、失败保留旧 release 和原子替换。
- [ ] 运行自测并确认新增测试失败。
- [ ] 实现 PyArrow 显式嵌套 Schema、分批 writer、Manifest 统计、全部第 11 节硬门禁、`--clean-state` 哈希前置检查。
- [ ] 运行 `python scripts/build_unified_dataset.py --self-test -v`，要求全部测试通过。
- [ ] 运行 `python scripts/build_unified_dataset.py --format jsonl` 生成可审计实验版并审计。
- [ ] 配置教师环境变量后运行 `python scripts/build_unified_dataset.py --format parquet --release`；只有 18,336/223/2,294 个任务及五个文件全部通过硬门禁才发布。

## 提交边界

每次提交只暂存 `scripts/build_unified_dataset.py` 和本计划明确修改的文档；不暂存现有 `data/processed/`、`data/release/` 或其他脚本修改。提交信息使用中文 Conventional Commits。

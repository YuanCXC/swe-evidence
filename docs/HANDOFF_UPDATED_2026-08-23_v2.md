# HANDOFF --- 给下一个聊天实例（2026-08-15 最新状态）

> 本文件是在旧 `HANDOFF` 基础上更新后的当前状态快照。\
> **若本文件与旧聊天 / 旧 HANDOFF
> 冲突，以本文件和用户最近明确要求为准。**
>
> 当前阶段已经从"设计 / 调试 Teacher 流程"推进到：
>
> **Strong-Teacher 大批量数据已跑完 →
> 现在重点做结果完整性审计、题答对齐检查、Witness / AND-OR
> 语义质量抽查，并准备最终冻结监督数据。**

------------------------------------------------------------------------

# 0. 当前最重要状态（先读这个）

## 0.1 当前项目阶段

当前不再以旧的 `External Supervision Bridge` 两阶段人工导出流程为主。

最新真实路线已经变成：

``` text
Frozen V2.10 task/question package
        ↓
Strong-Teacher Markdown（1 task / request）
        ↓
Qwen / LIN DeepSeek API
        ↓
Structured Strong-Teacher JSON answer
        ↓
local schema + task_id + Candidate binding validation
        ↓
data/upstream/external_supervision/result/{split}/...
        ↓
100% mechanical audit
        ↓
risk-ranked semantic audit
        ↓
final supervision freeze
```

用户最新明确状态：

``` text
数据已经跑完了。
```

当前最高优先级：

``` text
1. 检查答案文件有没有放错地方；
2. 检查题目与答案 task_id / 文件映射是否错位；
3. 检查 Candidate Number / Schema / OR-of-AND 是否非法；
4. 快速识别“文件名和 task_id 都对，但答案内容其实回答了另一题”的串题；
5. 对高风险 Witness / fault_logic / state_flow / repair_scope 做人工抽查；
6. 决定最终哪些监督可以接受。
```

## 0.2 当前一句话"合格答案"标准

用户最后确认的一句话标准：

> **题答对应正确，核心 Witness 与真实执行/修复语义一致，AND/OR
> 关系正确；允许少量相关冗余或轻微槽位判断偏差，但不能有错误因果、错误证据或无关内容。**

质量原则：

``` text
宁可接受少量相关冗余，
也不能接受错误因果、错误 AND/OR 或无关证据。
```

## 0.3 当前质量等级

统一使用：

``` text
PASS
核心 Witness 正确；
执行路径正确；
AND/OR 正确；
没有实质语义错误。

SOFT PASS
核心监督正确；
可能多 1–2 个相关但可删 Candidate；
repository_need / slot applicability 稍微保守；
reason 有轻微冗余；
但没有改变因果或 Witness 核心语义。
→ 可以接受。

REVIEW
Witness 是否必要、执行路径、修复范围、AND/OR 有明显疑点；
需要人工再看。

FAIL
错误 Evidence；
错误执行路径；
错误根因；
错误 AND / OR；
无关 Evidence；
答案明显属于另一题。
→ 不应直接进入高质量监督。
```

------------------------------------------------------------------------

# 1. 项目总体目标

项目目录 / 上下文：

``` text
evidence-agent/
软件测试修复实验
```

总体研究目标：

> 针对 SWE 类软件修复任务，自动收集达到"预定义修复上下文充分性标准"的
> Evidence
> Package（证据包），支持下游修复模型进行故障定位、原因分析和补丁规划。

在线主架构：

``` text
SWE Task + pre-fix repository
        ↓
NIPS-style Agentic RAG
        ↓
Structured Sufficient Evidence Package
```

实验范围严格限定：

``` text
Evidence Collection / Context Sufficiency
（证据收集 / 上下文充分性）
```

明确排除：

``` text
Patch Generation
Patch Application
Test Execution
Test Feedback Loop
```

允许的项目 / 论文表述：

``` text
达到预定义修复上下文充分性标准
支持下游修复模型定位、原因分析、补丁规划
```

不要写：

``` text
保证修复成功
保证生成正确补丁
```

------------------------------------------------------------------------

# 2. 项目结构与冻结边界

用户明确要求：

``` text
数据处理 / supervision refinement / training tooling
→ scripts/

实验核心
→ src/
```

当前约束：

``` text
src/
├── retrieval/
├── agents/
└── evaluation/

scripts/
├── dataset build
├── supervision refinement
├── Strong-Teacher tooling
├── audit
├── training
└── CLI
```

必须保持：

``` text
src/evaluation/judge.py
= Semantic Judge only
```

不要把 supervision refinement 逻辑塞进去。

## 2.1 冻结数据集

Builder：

``` text
scripts/build_unified_dataset_v2_10.py
```

数据：

``` text
data/upstream/unified_swe_dataset_v2_10/
```

版本：

``` text
dataset version = 2.10.0
script version  = 0.2.10
```

构建 DB：

``` text
data/.build/unified_swe_v1.sqlite3
≈ 60 GB
```

V2.10 已冻结。

当前 Strong-Teacher refinement：

``` text
只写 sidecar / external supervision result
绝不能直接覆盖 V2.10
```

## 2.2 核心锁定文件

除非发现明确 bug，不要重新打开这些语义：

``` text
scripts/refinement_core.py
= v1.7

scripts/refinement_candidate_builder.py
= v1.5.2
```

旧 External Bridge / v1.9.2.x Two-Stage
代码可作为历史参考，但已经不是当前主执行路径。

------------------------------------------------------------------------

# 3. 冻结数据统计

任务：

``` text
train       18,347
validation     223
benchmark    2,294
----------------
total       20,864
```

Corpus：

``` text
file versions   1,027,752
Evidence Units 25,496,300
```

V2.10 Policy：

``` text
actions       3,094,993
singles       2,721,201
pairs           331,508
STOP             42,284

states           42,284
tasks            20,864
invalid               0
```

Boundary：

``` text
556
```

不要为了"增加边界样本"再造 10k--30k boundary。

`overflow_state_count`：

``` text
15,839
```

含义：

``` text
candidate cap / injection retention overflow
```

不是 tokenizer 4096 overflow。

------------------------------------------------------------------------

# 4. 下游模型

当前目标模型：

``` text
Cross-Encoder Evidence Policy Ranker
（交叉编码器证据策略排序模型）
```

评分：

``` text
s_A = f_theta(q, K, A)
```

其中：

``` text
q = issue/problem
K = 当前 Evidence
A = action
```

Action：

``` text
[u]
[u, v]
STOP
```

Candidate retrieval / structure expansion / pair generation：

``` text
deterministic
non-trainable
```

Baseline：

``` text
BAAI/bge-reranker-v2-m3
revision:
953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e
```

约：

``` text
568M
XLM-R
24 layers
hidden 1024
max 4096
question max 2048
full fine-tuning
```

当前本地机器不负责正式训练。

------------------------------------------------------------------------

# 5. 当前 Strong-Teacher 协议

当前已经不再使用旧 Stage 1 / Stage 2 外部桥接作为主监督格式。

现在每个 Strong-Teacher task 一次性返回完整 7-slot 判断。

固定 canonical dimensions：

``` text
fault_location
fault_logic
dependency_context
state_flow
behavior_constraint
repair_scope
validation_constraint
```

## 5.1 每个 slot Schema

每个 slot：

``` json
{
  "applicability": "required | not_required | uncertain",
  "question_coverage": "sufficient | partial | none | uncertain | not_applicable",
  "repository_need": "required | helpful | not_needed | uncertain | not_applicable",
  "candidate_pool_status": "sufficient | insufficient | uncertain | not_needed",
  "sufficient_witness_groups": [],
  "supporting_candidates": [],
  "reason": "简体中文理由"
}
```

顶层：

``` json
{
  "task_id": "...",
  "overall_assessment": "...",
  "slots": {
    "...7 fixed slots..."
  },
  "additional_findings": [
    {
      "description": "...",
      "candidate_numbers": [],
      "reason": "..."
    }
  ],
  "uncertainties": []
}
```

最终：

``` text
纯 JSON array
无 Markdown fence
无 prefix/suffix
```

## 5.2 OR-of-AND

必须保持：

``` text
[[2, 5]]
= Candidate 2 AND Candidate 5

[[2], [5, 9]]
= Candidate 2
  OR
  Candidate 5 AND Candidate 9
```

程序不能自动猜测语义 AND/OR。

## 5.3 Minimal Sufficient Witness 核心

Strong Teacher Prompt 已经明确要求：

``` text
Singleton Scan
Complementary Combination Search
Deletion / Minimality Test
Alternative Search
Superset Elimination
Gold Independence Test
Execution Relevance
Conservative Empty Set
```

最重要：

``` text
如果 [A] 已经 sufficient，
不能再输出 [A,B]。

fault_logic / state_flow 中，
Candidate 必须真的位于 Issue 对应执行路径。
```

## 5.4 Question-Sufficiency Gate

如果 Issue 自己已经说明：

``` text
行为要求
验证条件
参数名
明确修复内容
```

则不应该仅因为 Candidate Pool 有相关代码就强行：

``` text
repository_need=required
```

但用户已放宽标准：

``` text
Question 已 sufficient，
仍选了 1–2 个相关仓库 Evidence，
如果没有错误因果，可以 SOFT PASS。
```

因此不要过度苛刻。

------------------------------------------------------------------------

# 6. 当前 Strong-Teacher 数据目录

用户当前真实输入目录，必须原样使用：

``` text
data\.external_supervision\strong_teacher_v1_3_all
```

不要擅自改回：

``` text
strong_teacher_v143
```

等其它历史目录。

结果目录：

``` text
data\.external_supervision\result
```

典型结构：

``` text
strong_teacher_v1_3_all/
├── train/
│   └── md/
├── validation/
│   └── md/
└── benchmark/
    └── md/

result/
├── train/
├── validation/
└── benchmark/
```

## 6.1 历史 dry-run 数字

曾经一次 dry-run：

``` text
scanned              = 20,588
skipped_nonempty     = 88
zero_byte_retries    = 2,130
selected_for_request = 20,500
```

注意：

``` text
20,588
比 frozen total 20,864
少 276
```

当时结论：

``` text
runner 可以继续；
后续增加 MD 可以 resume；
但最终冻结前必须审计：
哪些 frozen tasks 没进入 Strong-Teacher question tree。
```

现在用户说"数据跑完"，因此最终 audit 应同时关注：

``` text
input question tree 自己是否完整
result tree 是否完整
```

仅检查 result vs input 还不能发现"input 本来就少 276"的问题。

------------------------------------------------------------------------

# 7. 当前 API / Provider

## 7.1 Qwen

`.env` 历史设置：

``` env
QWEN_API_KEY=...
QWEN_API_URL=https://dashscope.aliyuncs.com/compatible-mode/v1
QWEN_MODEL=qwen3.7-max-2026-05-17
```

曾确认：

``` text
05-17 snapshot
only-thinking
不支持 context cache
```

Qwen 可用于 benchmark / 补充任务，但质量不能无条件信任。

## 7.2 LIN / TokenRhythm DeepSeek

当前 `.env`：

``` env
lin_API_KEY=...
lin_API_KEY_1=...
lin_API_KEY_2=...
lin_API_KEY_3=...
...
lin_API_URL=https://tokenrhythm.studio/v1/chat/completions
LIN_MODEL=deepseek-v4-flash-0731
```

当前 runner 已支持：

``` text
自动扫描 lin_API_KEY
自动扫描 lin_API_KEY_N
N 可以任意、不连续
兼容 LIN_API_KEY_N
去重相同 key
忽略空 key
```

例如：

``` env
# lin_API_KEY 已过期
lin_API_KEY_3=...
lin_API_KEY_7=...
lin_API_KEY_20=...
```

也能正常运行。

## 7.3 TokenRhythm URL 处理

用户配置可以是：

``` text
https://tokenrhythm.studio/v1/chat/completions
```

runner 会规范化成 OpenAI SDK base：

``` text
https://tokenrhythm.studio/v1
```

也兼容误写 Markdown URL 的情况。

## 7.4 reasoning_effort

LIN 当前默认：

``` text
reasoning_effort=max
```

通过：

``` python
extra_body={"reasoning_effort": "max"}
```

注意：

``` text
TokenRhythm 公共 API 文档没有明确写 reasoning_effort。
```

所以 runner 设计原则是：

``` text
不要静默降级。
route 真不支持时应明确失败。
```

也可：

``` text
--lin-reasoning-effort none
```

来省略该字段。

------------------------------------------------------------------------

# 8. 当前推荐 Runner：v1.5

当前优先使用：

``` text
scripts/run_strong_teacher_multi_api_v1_5.py
```

对应最新生成文件：

``` text
/mnt/data/run_strong_teacher_multi_api_v1_5.py
```

## 8.1 非覆盖语义

必须保留：

``` text
result 文件 > 0 byte
→ SKIP
→ 绝不覆盖

result 文件 == 0 byte
→ 可重新生成

文件不存在
→ 正常生成
```

写入前还会 race-time re-check，避免另一个进程先生成成功结果后被覆盖。

## 8.2 一个 task 一个请求

默认：

``` text
1 Markdown task
=
1 API request
```

不要自动合并多个 task 进同一个 request。

## 8.3 严格每个 key 单并发

v1.4 起已经改成 Key Lease（Key 租约）：

``` text
一个 key 同一时刻最多 1 个 API request
```

例如：

``` text
4 keys
--concurrency 20
```

实际最多：

``` text
4 个同时 API 请求
每个 key <= 1
```

其它 worker 等空闲 key。

默认 LIN：

``` text
concurrency = 自动发现有效 key 数量
```

因此通常不需要显式写 `--concurrency`。

## 8.4 503/504 / 429 策略

v1.5 专门处理：

``` text
503 SERVICE_BUSY
504 Gateway Timeout
429 RateLimit
```

默认：

``` text
max_retries = 5
```

退避：

``` text
429:
5s → 15s → 30s → 60s

503 / 504:
10s → 30s → 60s → 120s

500 / 502:
5s → 15s → 30s → 60s

network:
3s → 10s → 20s → 40s

validation:
1s → 2s → 4s → 8s
```

## 8.5 Provider Global Cooldown

默认：

``` text
busy_cooldown_threshold = 2
busy_cooldown_window    = 10s
busy_cooldown_seconds   = 30s
```

含义：

``` text
10 秒内出现 2 次 503/504
        ↓
判定共享 backend busy
        ↓
整个 LIN provider cooldown 30 秒
        ↓
自动继续
```

关键实现：

``` text
等待 retry / cooldown 前
先 release key
```

不能拿着 key 睡觉。

## 8.6 quota/balance

如果错误明确属于：

``` text
quota exhausted
balance insufficient
额度不足
余额不足
```

则：

``` text
只 disable 当前 key
其它 key 继续
```

503 / 504 不 disable key。

------------------------------------------------------------------------

# 9. uncertainties 机械规范化

历史遇到：

``` text
ValueError: uncertainties 必须是字符串 list
```

v1.3 起允许安全机械修复：

``` text
null
→ []

""
→ []

"一句话"
→ ["一句话"]

["A", "B"]
→ 原样
```

仍然拒绝：

``` text
{"reason": "..."}
["A", {"reason":"..."}]
123
```

因为这需要猜模型语义。

原则：

``` text
能 100% 无歧义修复
→ 本地 normalize

需要猜模型意思
→ reject / retry
```

------------------------------------------------------------------------

# 10. Strong-Teacher 质量校准经验

## 10.1 Qiskit hard stress

任务：

``` text
task_0004c3905bec5145332e53ba
```

关键：

``` text
Issue 调 cf.synth() 默认路径
Candidate 8 在默认路径
Candidate 24 是 qregs / non-default registerless=False 相关
```

错误答案曾使用：

``` text
[[8,24]]
```

并声称 Candidate 24 参与默认根因。

结论：

``` text
FAIL
```

重要规则：

``` text
Execution-Path Grounding
（执行路径落地）
```

必须守住。

## 10.2 Twisted CFReactor

某 Qwen 答案：

``` text
repair_scope = [[1],[9]]
```

把 reactor implementation 和 CI workflow 都说成可独立完整修复。

结论：

``` text
FAIL / strong REVIEW
```

原因：

``` text
两个修复面并不是真正可互换 OR。
```

## 10.3 Docker compose pull progress

任务：

``` text
task_00052d60...
```

优秀例子：

``` text
fault_logic [[22,4]]
```

Project.pull 并行路径：

``` text
silent=True
```

Service.pull：

``` text
redirect devnull
```

因果链成立。

结论：

``` text
PASS / SOFT PASS
```

## 10.4 qtconsole help truncation

某答案认为：

``` text
pager not triggered
→ console buffer truncates
```

但 Issue 语义是：

``` text
已经分页
只是 page 不够
```

结论：

``` text
FAIL / REVIEW
```

说明：

``` text
术语相关不等于因果正确。
```

## 10.5 pandas categorical scatter

核心 Candidate singleton 正确。

有轻微：

``` text
repository_need=helpful
candidate_pool_status=sufficient
groups=[]
```

一致性小问题。

结论：

``` text
SOFT PASS
```

------------------------------------------------------------------------

# 11. 最近人工抽查 Strong-Teacher 样本

用户上传了三组同名题答：

``` text
604 task_087c...
605 task_088043...
606 task_0883a...
```

人工结论：

``` text
604 PASS
605 SOFT PASS
606 SOFT PASS

可接受率 3/3
严重错误 0/3
```

### 604 Logging docs typo

Issue 已明确：

``` text
project_ids
→ projects
```

Candidate 3 只是额外确认。

答案没有强行 repository required。

结论：

``` text
PASS
```

### 605 Airflow scheduler stuck

核心：

``` text
Candidate 14 = producer send
Candidate 6  = heartbeat consumer recv
```

`[[6,14]]` 作为 fault_logic / state_flow AND 是有执行路径依据的。

但 Issue 本身已经有很强 pipe-full diagnosis，因此：

``` text
repository_need=required
```

略保守。

结论：

``` text
SOFT PASS
```

### 606 pip it's/its typos

Candidates 1/2/3 对应三处独立 typo。

完整修复范围：

``` text
[[1,2,3]]
```

AND 合理。

但 Issue 只说 "some typos"，隐藏 Gold 后不能绝对证明仓库中只有三处，所以
pool status 有轻微过度确定。

结论：

``` text
SOFT PASS
```

------------------------------------------------------------------------

# 12. Qwen 质量结论

之前抽了 5 个 Qwen 样本：

``` text
2/5 可直接接受
3/5 需要 review / fail
```

主要风险：

``` text
1. 多成员 fault_logic/state_flow AND
2. repair_scope 多 OR
3. 默认执行路径与非默认 branch 混淆
4. Gold 影响过大
5. 同一 Candidate 覆盖很多 slots
```

因此：

``` text
Qwen 可以作为便宜 Teacher / 补充 Teacher，
但不能 20k 全量盲信。
```

------------------------------------------------------------------------

# 13. 成本重新估算

用户最新实测 DeepSeek 平均：

``` text
input  ≈ 15k tokens / task
output ≈ 25k tokens / task
        （包含 thinking/reasoning）
```

用户之前给的价格：

``` text
cache miss input = ¥1 / 1M tokens
cache hit input  = ¥0.02 / 1M tokens
output           = ¥2 / 1M tokens
```

无缓存单任务：

``` text
input:
15,000 / 1M × ¥1
= ¥0.015

output:
25,000 / 1M × ¥2
= ¥0.050

total:
≈ ¥0.065 / task
```

规模：

``` text
1,000  → ¥65
5,000  → ¥325
10,000 → ¥650
20,500 → ¥1,332.50
20,588 → ¥1,338.22
20,864 → ¥1,356.16
```

成本结构：

``` text
input  ≈ 23%
output ≈ 77%
```

因此未来若再优化：

``` text
优先压 reasoning/output
>
prompt cache
>
单纯缩输入
```

------------------------------------------------------------------------

# 14. 当前审计脚本

已经生成：

``` text
scripts/audit_strong_teacher_results_v1_0.py
```

最新文件：

``` text
/mnt/data/audit_strong_teacher_results_v1_0.py
```

SHA256：

``` text
b2e31375d186daf9f44cbb7db1278958882458135729202d6712834af0c75692
```

## 14.1 建议运行命令

PowerShell：

``` powershell
python scripts/audit_strong_teacher_results_v1_0.py `
  --input-root "data\.external_supervision\strong_teacher_v1_3_all" `
  --result-root "data\.external_supervision\result"
```

默认报告：

``` text
data\.external_supervision\.audit\strong_teacher_audit\
```

输出：

``` text
audit_summary.json
audit_issues.csv
per_answer_status.csv
semantic_review_queue.csv
random_low_risk_sample.csv
```

------------------------------------------------------------------------

# 15. Audit v1.0 能 100% 检查什么

## 15.1 HARD_ERROR

应尽量全部清零：

``` text
题目有，答案没有
答案有，题目没有
答案放错 train / validation / benchmark
同 split 同名答案重复
0 byte
题目 task_id 解析失败

filename task_id
!= question task_id

question task_id
!= answer JSON task_id

答案 task_id 实际属于另一题

JSON 顶层错误
7 canonical slots 缺失 / 多出
非法 enum
Candidate Number 越界
additional_findings 非法 Candidate
uncertainties 格式异常
空 AND group
AND 内重复 Candidate
重复 OR group
明显非最小 superset group
repository_need 与 Witness 明显不一致
```

目标：

``` json
"issue_severity_counts": {
  "HARD_ERROR": 0
}
```

## 15.2 文件放错 split

会抓：

``` text
RESULT_MISPLACED_SPLIT
ORPHAN_RESULT_WRONG_SPLIT
```

例如：

``` text
question:
train/md/A.md

answer:
result/benchmark/A.md
```

直接报错。

## 15.3 题答 task_id 错位

检查三层：

``` text
filename
↓
question internal task_id
↓
answer JSON task_id
```

例如：

``` text
filename = task_A
question = task_A
answer   = task_B
```

报：

``` text
ANSWER_WRONG_TASK_ID
```

如果 `task_B` 在 question tree 中存在，还会告诉：

``` text
task_B 真正属于哪个题目。
```

------------------------------------------------------------------------

# 16. Audit v1.0 的 semantic risk flags

注意：

``` text
RISK_FLAG ≠ 答案错误
```

只是优先人工检查。

当前主要 flags：

``` text
CAUSAL_MULTI_AND
fault_logic / state_flow 多成员 AND
→ 检查真实执行路径 + deletion test

REPAIR_MULTI_OR
repair_scope 多 OR
→ 检查每个 alternative 是否真能独立完成修复范围

QUESTION_SUFFICIENT_BUT_REPO_REQUIRED
Issue 已 sufficient 却仍 repository required
→ Question-Sufficiency Gate 风险

WITNESS_SATURATION
同一 Candidate 用于很多 required slots

SAME_SINGLETON_SATURATION
同一 singleton 横跨很多 slots

UNCERTAINTY_CAUSAL_CONTRADICTION
uncertainties 说根因未知，
但 fault_logic/state_flow 又声称 sufficient

POOL_STATUS_CROSS_SLOT_TENSION
fault_logic uncertain/insufficient
但 repair_scope sufficient

MANY_SUPPORTING
supporting_candidates 很多

REPAIR_GOLD_PATH_DIVERGENCE
repair_scope Witness path 与 Gold changed_files 完全不重合
```

最后一项：

``` text
只作为 offline review trigger
绝不能自动 FAIL
```

Gold 不能作为 final Witness，也不能单独证明 Candidate sufficient。

------------------------------------------------------------------------

# 17. Audit v1.0 已知缺口：文件名/task_id 都对，但答案内容串题

这是用户最新特别关心的风险。

典型情况：

``` text
filename = task_A
question task_id = task_A
answer task_id = task_A

但是 overall_assessment / reason 实际在回答 task_B。
```

如果 Candidate Number 又恰好合法：

``` text
纯 Schema 校验可能过关。
```

因此：

``` text
audit v1.0 不能 100% 抓这种 semantic cross-task contamination。
```

## 17.1 下一版 Audit 建议补的检查

建议继续开发：

``` text
audit_strong_teacher_results_v1_1.py
```

新增：

``` text
CONTENT_ALIGNMENT_RISK
FOREIGN_ENTITY_CONTAMINATION
WITNESS_REASON_GROUNDING_RISK
CROSS_TASK_CONTENT_SUSPECT
```

## 17.2 Content Alignment 检查思路

从答案：

``` text
overall_assessment
slot.reason
additional_findings
uncertainties
```

抽取高特异性实体：

``` text
path
symbol
function
class
API
identifier
反引号内容
```

与当前题目：

``` text
Issue
Candidate Pool
Candidate path / symbol
```

比对。

例如当前题是：

``` text
NumPy ediff1d
```

答案却出现：

``` text
DagFileProcessorManager
CeleryExecutor
heartbeat
multiprocessing.connection
```

应高风险：

``` text
FOREIGN_ENTITY_CONTAMINATION
```

## 17.3 Witness Grounding 检查

不能只检查：

``` text
Candidate 7 是否存在
```

还应该检查：

``` text
答案 reason 说 Candidate 7 证明 X
        ↓
当前 Candidate 7 的真实内容
        ↓
是否真的和 X 对应
```

例如：

``` text
reason：
Candidate 7 展示 C0 颜色判断
```

但 Candidate 7 实际是：

``` text
install_data(...)
```

应：

``` text
WITNESS_REASON_GROUNDING_RISK
```

## 17.4 最可靠的后续方案

可用 cheap semantic critic（廉价语义审查模型）处理风险样本。

不要重发完整 25k prompt。

只给：

``` text
TASK_ID

[ISSUE]
Issue

[SELECTED EVIDENCE]
只放答案引用的 Candidate

[ANSWER]
Strong-Teacher JSON
```

让 critic 只判断：

``` text
1. 是否明显回答当前 Issue
2. 是否有另一任务实体
3. Witness 与 reason 是否对应
4. 是否有明显错误因果
5. AND/OR 是否明显错误
```

输出短 JSON：

``` json
{
  "alignment": "pass|review|fail",
  "foreign_content": false,
  "witness_grounding": "pass|review|fail",
  "reason": "..."
}
```

------------------------------------------------------------------------

# 18. 最终推荐审计流程

## Step A --- 100% mechanical audit

全量跑：

``` text
question tree
↔
result tree
↔
filename
↔
task_id
↔
Candidate Number
↔
Schema
↔
OR-of-AND canonicalization
```

目标：

``` text
HARD_ERROR = 0
```

## Step B --- 检查 Strong-Teacher question tree 完整性

注意：

``` text
result vs input 全对
≠
frozen 20,864 全覆盖
```

因为历史 input 只有：

``` text
20,588
```

因此最终还要做：

``` text
Frozen V2.10 task manifest
vs
strong_teacher_v1_3_all task ids
```

找出：

``` text
missing 276
```

到底在哪些 split / task。

## Step C --- semantic review queue

默认先看：

``` text
semantic_review_queue.csv
top 300
```

人工只检查高价值问题：

``` text
1. Witness 真的是当前题证据吗？
2. fault_logic/state_flow 是否在真实执行路径？
3. AND 删除任一成员后是否真的不充分？
4. OR 每个 alternative 是否真能独立 sufficient？
5. 是否把“相关”说成“因果已证实”？
```

不要要求所有 slot 绝对完美。

## Step D --- random low-risk sample

必须随机看一批：

``` text
random_low_risk_sample.csv
默认 200
```

原因：

``` text
heuristic 可能漏掉“格式很正常但语义错”的答案。
```

建议：

``` text
high risk ≈ 300
+
low risk random ≈ 200
=
人工约 500 条
```

比全看 20k 高效很多。

## Step E --- Content Alignment / Witness Grounding

如果准备最终冻结数据：

``` text
优先实现 audit v1.1
```

至少把：

``` text
文件名 / task_id 正确但正文串题
```

这种风险再筛一次。

## Step F --- usage log reconciliation

Runner usage log：

``` text
data/upstream/external_supervision/.run_logs/
strong_teacher_multi_api_usage.jsonl
```

最终建议检查：

``` text
question 数
vs
nonempty result 数
vs
SUCCESS task_id 数
vs
FAILED task_id 数
vs
duplicate SUCCESS
```

可抓：

``` text
API SUCCESS 但文件没落盘
文件有结果但 usage 没成功记录
同 task 被多个 provider 重复请求
```

audit v1.0 当前还没集成这一项。

------------------------------------------------------------------------

# 19. Benchmark 规则

虽然已经生成 benchmark Strong-Teacher 结果，但要保持：

``` text
benchmark 不用于 Prompt / protocol 调参
```

可以：

``` text
做质量审计
做最终分析
做冻结后的评估
```

不要：

``` text
看 benchmark Teacher 结果后
再回头改 Prompt / witness protocol
```

否则 benchmark leakage（基准集泄漏）。

------------------------------------------------------------------------

# 20. Current preferred commands

## LIN 默认运行

``` powershell
python scripts/run_strong_teacher_multi_api_v1_5.py `
  --provider lin `
  --input-root "data\.external_supervision\strong_teacher_v1_3_all" `
  --result-root "data\.external_supervision\result"
```

通常不需要写：

``` text
--concurrency
```

默认：

``` text
concurrency = active key count
```

且严格一 key 一并发。

## Qwen benchmark-only

如果以后需要补 benchmark：

``` powershell
python scripts/run_strong_teacher_multi_api_v1_5.py `
  --provider qwen `
  --input-root "data\.external_supervision\strong_teacher_v1_3_all" `
  --result-root "data\.external_supervision\result" `
  --splits benchmark
```

若同时跑 LIN：

``` text
不要让 LIN 与 Qwen 同时抢同一 split
```

虽然不会覆盖非空文件，但会浪费 token。

## Audit

``` powershell
python scripts/audit_strong_teacher_results_v1_0.py `
  --input-root "data\.external_supervision\strong_teacher_v1_3_all" `
  --result-root "data\.external_supervision\result"
```

------------------------------------------------------------------------

# 21. 旧 External Supervision Bridge 的地位

旧 HANDOFF 中写过：

``` text
external_supervision_bridge.py
Stage1
Stage2
WorkBuddy / 网页模型
```

这属于历史路线。

现在不要默认恢复为：

``` text
Stage 1 candidate-blind
→ Stage 2 targeted witness
```

当前主数据已经通过新 Strong-Teacher 完整 7-slot schema 跑完。

旧 bridge：

``` text
可以保留作为历史实验 / fallback
```

但不是当前 P0。

------------------------------------------------------------------------

# 22. 不要恢复的旧方案

不要自动恢复：

``` text
GLM 单次自由生成完整 supervision graph
2-of-3 self-consistency = semantic truth
confidence <0.85 hard gate
DeepSeek 全量二次审查
自动语义修补 AND/OR
future symbol 缺失 = insufficient
Core Accepted = semantic correctness
继续大规模重复跑已经完成的数据
```

------------------------------------------------------------------------

# 23. Core Accepted / Complete 的解释

必须保持：

``` text
Structural / Deterministic Acceptance
≠
Semantic Correctness
```

`Complete`：

``` text
Deterministic Certificate Complete
Proxy Label
```

不是：

``` text
Semantic Truth
```

------------------------------------------------------------------------

# 24. 防止新聊天犯错

不要重新问：

``` text
refinement 放 src 还是 scripts？
```

答案：

``` text
scripts/
```

不要问：

``` text
能不能改 V2.10？
```

答案：

``` text
不能覆盖；
sidecar only。
```

不要问：

``` text
当前 input root 是哪个？
```

答案：

``` text
data\.external_supervision\strong_teacher_v1_3_all
```

不要问：

``` text
result root？
```

答案：

``` text
data\.external_supervision\result
```

不要问：

``` text
一个 key 是否允许多并发？
```

当前 runner：

``` text
严格每个 key 单并发。
```

不要把：

``` text
503/504
```

误判成 key 失效。

它们通常表示：

``` text
backend busy
```

不要把相关冗余自动判 FAIL。

当前接受标准允许：

``` text
少量相关冗余
轻微 slot 分类偏差
```

真正必须拦的是：

``` text
错题答案
错误 Candidate
错误执行路径
错误根因
错误 AND/OR
无关 Evidence
```

------------------------------------------------------------------------

# 25. 下一聊天建议立即做什么

用户换聊天后，最可能继续的是最终审计。

## P0

如果用户提供：

``` text
audit_summary.json
audit_issues.csv
```

直接分析，不要重新问背景。

重点：

``` text
HARD_ERROR 是否为 0
missing result
misplaced split
wrong task_id
illegal Candidate
duplicate result
nonminimal OR-of-AND
```

## P0.1

如果用户问：

> 文件名/task_id 都对，但答案内容串题怎么办？

直接延续：

``` text
audit v1.1
```

实现：

``` text
Content Alignment
Foreign Entity Contamination
Witness-Reason Grounding
Cross-task contamination
```

不要只靠 task_id。

## P0.2

如果机械审计通过：

``` text
先审 semantic_review_queue top 300
再审 random low-risk 200
```

按：

``` text
PASS / SOFT PASS / REVIEW / FAIL
```

给人工标签。

## P1

最终冻结前补：

``` text
Frozen V2.10 20,864
vs
Strong-Teacher question tree
```

完整性对账。

历史上 question tree：

``` text
20,588
```

少：

``` text
276
```

必须最终解释 / 补齐 / 明确排除。

## P2

再补：

``` text
usage JSONL
vs
result tree
```

对账。

## P3

若 semantic audit 显示大量"内容串题"或 Witness grounding 错误：

``` text
不要直接全量重跑 Teacher。
```

先：

``` text
风险筛选
→ 只重跑 FAIL / REVIEW subset
```

这样最省成本。

------------------------------------------------------------------------

# 26. 最终连续性原则

当前目标不是：

``` text
让所有 Teacher 答案看起来完美
```

而是：

``` text
保证最终进入训练的监督：
题答对应正确，
核心 Witness 正确，
执行路径正确，
AND/OR 正确，
允许轻度相关冗余，
但阻断错误因果、错误 Evidence、错误结构和串题污染。
```

当前项目下一阶段关键词：

``` text
Data Integrity Audit（数据完整性审计）
Content Alignment（题答内容对齐）
Witness Grounding（证据落地）
Semantic Risk Review（语义风险复核）
Final Supervision Freeze（最终监督冻结）
```

------------------------------------------------------------------------

# 27. 当前关键文件速查

``` text
Frozen dataset:
data/upstream/unified_swe_dataset_v2_10/

Strong-Teacher questions:
data/upstream/external_supervision/strong_teacher_v1_3_all/

Strong-Teacher answers:
data/upstream/external_supervision/result/

Runner:
scripts/run_strong_teacher_multi_api_v1_5.py

Audit:
scripts/audit_strong_teacher_results_v1_0.py

Usage log:
data/upstream/external_supervision/.run_logs/strong_teacher_multi_api_usage.jsonl

Locked:
scripts/refinement_core.py                  v1.7
scripts/refinement_candidate_builder.py    v1.5.2
scripts/build_unified_dataset_v2_10.py
```

当前 runner 关键能力：

``` text
auto lin_API_KEY_N
strict one-key-one-request
resume
nonempty never overwrite
0B regenerate
atomic write
schema/task_id/Candidate validation
uncertainties mechanical normalization
429 backoff
503/504 backend busy cooldown
quota key disable
```

当前 audit 关键能力：

``` text
file/split alignment
question-answer task_id alignment
schema validation
Candidate binding
OR-of-AND mechanical checks
semantic risk queue
random low-risk sample
```

当前 audit 还缺：

``` text
content alignment
witness-reason semantic grounding
usage-log reconciliation
frozen-20,864 vs question-tree reconciliation
```

这些是下一个聊天最值得继续补的地方。

------------------------------------------------------------------------

# 28. UPDATE 2026-08-23（Stage1 Policy / Multi-stage Boundary / Stage2 Fine-tuning）

## 28.1 Strong-Teacher 阶段结束后的路线变化

原 HANDOFF 主要记录 Strong-Teacher supervision
freeze、数据审计和最终监督冻结阶段。

之后项目进入 Evidence Policy training 阶段：

``` text
Frozen V2.10 task package
        ↓
Strong-Teacher supervision freeze
        ↓
Evidence Policy dataset construction
        ↓
Stage1 Evidence Policy training
        ↓
发现 Decision Boundary supervision distribution problem
        ↓
Multi-stage Boundary reconstruction
        ↓
Stage2 trajectory-aware fine-tuning
```

## 28.2 Stage1 Evidence Policy Training 已完成

模型：

``` text
BAAI/bge-reranker-v2-m3
```

输出：

``` text
models/evidence_policy_v1_0
```

训练：

``` text
epoch = 3
```

最佳验证：

``` text
best_validation_mrr:
0.8840652029274783
```

Validation：

``` text
Hit@1:
0.7744

MRR:
0.8841

STOP accuracy:
0.7984
```

关键发现：

整体指标较高，但 decision_boundary:

``` text
Hit@1:
0.2752

STOP accuracy:
0.2936
```

说明模型对 Initial / Complete 学习较好，但 Boundary decision learning
明显不足。

## 28.3 Multi-stage Boundary Reconstruction

原因：

原始 Boundary 存在 near-complete bias。

它更接近：

``` text
接近完成时是否停止
```

而不是：

``` text
真实 evidence acquisition trajectory 中如何逐步选择 evidence。
```

因此构建：

``` text
Early
Mid
Late
Near-complete
```

多阶段 Boundary。

生成：

``` text
data/evidence_agent_multistage_boundary_v1
```

统计：

``` text
train boundary:
10907

validation boundary:
200
```

## 28.4 Stage2 Boundary Fine-tuning

训练目录：

``` text
models/evidence_policy_multistage_ft_v1
```

参数：

``` text
epochs:
2

learning_rate:
5e-6

weight_decay:
0.01

warmup_ratio:
0.06

max_candidates:
12

pair_negative_quota:
3

grad_accum_steps:
8
```

目标：

验证：

``` text
Boundary improvement
+
Initial retention
+
Complete retention
```

而不是只追求 aggregate MRR。

## 28.5 故障记录

CUDA native crash：

``` text
CUDA error:
unspecified launch failure
```

之后进程进入：

``` text
D state / do_exit
```

原因：

core dump / apport 阻塞退出。

处理：

``` text
kill -9
```

并通过 recovery checkpoint 恢复：

``` text
state=8000
```

## 28.6 Evidence Cache 问题

SQLite evidence cache:

``` text
≈4.6GB
```

恢复训练时：

``` sql
COUNT(*) FROM evidence
```

导致大表扫描。

后续修复方向：

使用 metadata / cached row count，避免启动阶段扫描完整 SQLite。

## 28.7 当前论文主线

三项贡献保持：

1.  Structured Evidence Sufficiency

2.  Trajectory-Aware State-Conditioned Evidence Policy

3.  Evidence Interaction-Aware Acquisition

其中：

Multi-stage Decision Boundary Reconstruction

属于第二项核心方法。

论文逻辑：

``` text
Aggregate metric 看似优秀
↓
Boundary evaluation 暴露失败
↓
发现 trajectory mismatch
↓
提出 Multi-stage Boundary
↓
验证 sequential evidence acquisition improvement
```

## 28.8 下一步实验

必须完成：

### Stage1 vs Stage2

固定：

``` text
validation 200 boundary states
```

比较：

``` text
Hit@1
MRR
STOP accuracy
loss
```

### Retention

检查：

``` text
Initial
Complete
```

防止 catastrophic forgetting。

### Frozen Benchmark

保持 benchmark 不训练。

### Rollout

真实：

``` text
K=∅
→ collect evidence
→ STOP
```

指标：

``` text
sufficiency success
premature STOP
evidence efficiency
```

## 28.9 当前不要做

不要：

-   增加 Agent 模块作为主要创新；
-   修改 Frozen V2.10；
-   重新大规模生成 Strong-Teacher；
-   单纯扩大 Boundary 数量。

当前核心：

验证 Multi-stage Boundary 是否真正改善 trajectory-level evidence
acquisition。

---

# 28. UPDATE 2026-08-23（Policy Training 阶段迁移）

> 本节是在 2026-08-15 HANDOFF 基础上的状态迁移更新。
>
> 不覆盖历史阶段，只记录 Strong-Teacher 冻结之后的新进展。

---

## 28.1 项目阶段变化

2026-08-15 HANDOFF 时：

```text
Strong-Teacher 数据已生成
        ↓
机械审计
        ↓
语义风险抽查
        ↓
最终监督冻结
```

该阶段已经完成。

之后进入：

```text
Frozen V2.10 supervision
        ↓
Evidence Policy Training
        ↓
Stage1 Policy evaluation
        ↓
发现 Decision Boundary supervision mismatch
        ↓
Multi-stage Boundary Reconstruction
        ↓
Stage2 Policy Fine-tuning
```

---

## 28.2 Stage1 Evidence Policy Training（已完成）

模型：

```text
BAAI/bge-reranker-v2-m3
```

输出：

```text
models/evidence_policy_v1_0
```

训练：

```text
epoch = 3
```

最佳验证：

```text
best_validation_mrr:
0.8840652029274783
```

Validation：

```text
Hit@1:
0.7744

MRR:
0.8841

STOP accuracy:
0.7984
```

关键发现：

整体指标较好，但拆分：

```text
decision_boundary:

Hit@1:
0.2752

STOP accuracy:
0.2936
```

说明：

模型能够学习 Initial / Complete 状态，
但中间 Evidence Acquisition Boundary 学习不足。

---

## 28.3 Boundary 问题与 Multi-stage Reconstruction

原因：

原始 Boundary 样本存在：

```text
near-complete bias
```

不能代表：

```text
Early
↓
Mid
↓
Late
↓
Near-complete
↓
Complete
```

真实证据获取轨迹。

因此生成：

```text
data/evidence_agent_multistage_boundary_v1
```

统计：

```text
train boundary:
10907

validation boundary:
200
```

阶段分布：

Train：

```text
early:
1105

mid:
5173

late:
3901

near_complete:
728
```

Validation：

```text
early:
22

mid:
95

late:
69

near_complete:
14
```

---

## 28.4 Stage2 Boundary Fine-tuning（进行中）

训练目录：

```text
models/evidence_policy_multistage_ft_v1
```

参数：

```text
epochs:
2

learning_rate:
5e-6

weight_decay:
0.01

warmup_ratio:
0.06

max_candidates:
12

pair_negative_quota:
3

grad_accum_steps:
8
```

目标：

不是单纯提升 aggregate MRR。

重点：

```text
Boundary improvement

+

Initial retention

+

Complete retention
```

---

## 28.5 训练故障记录

### CUDA native crash

发生：

```text
Stage2 training
state≈8163
```

错误：

```text
CUDA error:
unspecified launch failure
```

之后：

```text
D state
do_exit
```

原因：

native crash 后 core dump/apport 阻塞退出。

处理：

```text
kill -9
```

恢复：

```text
recovery checkpoint
state=8000
```

之后成功继续训练。

---

## 28.6 Evidence Cache 问题

问题：

SQLite evidence cache：

```text
≈4.6GB
```

恢复训练阶段：

```sql
COUNT(*) FROM evidence
```

导致大表扫描。

后续原则：

使用已有 metadata / cached row count。

避免恢复训练阶段扫描完整 SQLite。

---

## 28.7 当前论文主线更新

核心贡献保持：

### Contribution 1

Structured Evidence Sufficiency

### Contribution 2

Trajectory-Aware State-Conditioned Evidence Policy

其中包含：

```text
Multi-stage Decision Boundary Reconstruction
```

### Contribution 3

Evidence Interaction-Aware Acquisition

论文逻辑：

```text
Aggregate metric 看似优秀
        ↓
Boundary evaluation 暴露失败
        ↓
发现 supervision trajectory mismatch
        ↓
提出 Multi-stage Boundary
        ↓
验证 sequential evidence acquisition improvement
```

---

## 28.8 当前实验计划

### Stage1 vs Stage2

固定：

```text
validation 200 boundary states
```

比较：

```text
Hit@1

MRR

STOP accuracy

loss
```

---

### Retention

检查：

```text
Initial

Complete
```

防止 catastrophic forgetting。

---

### Frozen Benchmark

保持：

```text
benchmark 不参与训练
```

---

### Rollout

真实：

```text
K=∅

collect evidence

until STOP
```

指标：

```text
sufficiency success

premature STOP

evidence efficiency
```

---

## 28.9 当前策略

暂不：

```text
增加 Agent 模块创新

修改 Frozen V2.10

重新生成 Strong-Teacher

单纯扩大 Boundary 数量
```

当前核心：

验证：

```text
Multi-stage Boundary
是否真正改善 trajectory-level evidence acquisition。
```

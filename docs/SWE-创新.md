可以。调整后的贡献结构应从：

```text
任务与基准
+ Value-of-Information 方法
+ 充分性与最小性的外部验证
```

改成：

```text
任务与基准
+ Value-of-Information 方法
+ 证据互补与替代建模
```

其中，“跨修复器、反事实删除、真实 patch 成功率”等内容仍然保留为**必要实验和有效性验证**，但不再写成独立创新贡献。这样主线更集中，也能避免第三项贡献看起来只是实验设计。你提供的最新评审也明确建议将项目压缩到行为价值、信息价值与证据交互，而不是继续堆叠平行贡献。

# 一、更新后的核心研究问题

项目最终研究的问题可以定义为：

> 在有限预算下，软件修复 agent 应该收集哪些证据、哪些证据需要组合使用、哪些证据可以相互替代，以及何时停止继续读取仓库？

传统代码检索通常隐含假设：

[
\text{Value}(K)
=

\sum_{u\in K}\text{Value}(u)
]

即每条证据的价值可以独立相加。

但实际修复证据往往具有明显交互：

```text
单独看 buggy logic 可能不足以修复
+
单独看 caller constraint 也可能不足以修复
=
两者一起才能确定正确修改
```

另一方面，不同证据也可能表达相同信息：

```text
完整函数
局部分支
相关测试断言
函数语义摘要
```

它们可能互相替代，同时全部读取只会增加冗余。

因此需要显式建模：

[
\text{Evidence Value}
=

\text{Individual Value}
+
\text{Complementarity}
----------------------

\text{Substitutability/Redundancy}
]

---

# 二、创新一：Generator-Calibrated Repair Evidence Acquisition

第一项创新保持不变，但表述应更精确。

> 提出面向固定修复器校准的修复证据获取任务，使 agent 在修复前仓库中以有限成本收集能够支持下游修复的证据，而不是仅优化 changed-location recall 或上下文相关性。

定义当前证据状态的修复价值：

[
V_G(K)
=

P_G(
\text{repair succeeds}
\mid q,K
)
]

其中：

* (q)：bug report；
* (K)：当前证据集合；
* (G)：固定的下游修复器；
* (V_G(K))：证据集合对该修复器的价值。

Nips 样式仍保持：

[
(q,K_t,C_t)\rightarrow u_{t+1}
]

但任务目标由“命中 gold unit”升级为：

[
\max_\pi
V_G(K_T)
-\lambda\operatorname{Cost}(K_T)
]

其中成本可以包括：

```text
读取 token 数
代码行数
tool calls
交互步数
上下文占用
推理费用
```

这项贡献的重点是：

```text
Context relevance
→ Repair-oriented evidence value
```

而不是单纯增加新的 evidence ontology。

---

# 三、创新二：Repair Value-of-Information Policy

第二项创新继续作为主要算法贡献。

## 1. 状态价值

模型估计当前证据包的修复价值：

[
\hat V_G(K_t)
]

它回答：

> 当前收集到的证据，对下游修复器有多大帮助？

## 2. 候选边际价值

候选证据 (u) 在当前状态下的增益：

[
\Delta_G(u\mid K_t)
=

## V_G(K_t\cup{u})

V_G(K_t)
]

这与普通相关性排序不同。

同一证据在不同状态下可以有不同价值：

```text
还没有目标函数时：
函数定义价值很高

已经读取函数后：
同一个函数摘要价值很低

已经知道 buggy logic 但不知道边界条件时：
相关测试或 caller constraint 价值很高
```

## 3. 成本敏感价值

[
Q_G(u\mid K_t)
=

\Delta_G(u\mid K_t)
-\lambda\operatorname{Cost}(u)
]

策略选择：

[
u_{t+1}
=

\arg\max_{u\in C_t}
Q_G(u\mid K_t)
]

## 4. 自适应停止

停止不再只是独立的 `STOP` 分类，而由剩余信息价值决定：

[
\max_{u\in C_t}
Q_G(u\mid K_t)
\le 0
]

含义是：

> 剩余候选证据的预期修复收益已经不足以抵消读取成本。

可以再加入当前价值阈值：

[
\hat V_G(K_t)\ge\tau
\quad\land\quad
\max_{u\in C_t}Q_G(u\mid K_t)\le0
]

因此：

* `utility` 是成本调整后的边际价值；
* `contribution` 是 (\Delta_G)；
* `deficit` 是当前价值不足的解释；
* `STOP` 是剩余证据不再具有正净收益。

这使原来的多个 prediction head 被统一在一个 Value-of-Information 框架内。

---

# 四、创新三：Complementarity- and Substitutability-Aware Evidence Acquisition

这是替换“外部行为验证”后的新核心贡献。

建议正式命名为：

## Evidence Interaction-Aware Acquisition

中文：

> **证据交互感知的修复证据获取**

核心观点是：

> 软件修复证据不是独立排列的代码片段，而是存在互补、替代、冗余和条件依赖关系的结构化证据集合。

---

## 1. 证据互补性

两条证据单独价值有限，但组合后价值明显提高。

定义交互价值：

[
I_G(u,v\mid K)
=

## V_G(K\cup{u,v})

## V_G(K\cup{u})

V_G(K\cup{v})
+
V_G(K)
]

当：

[
I_G(u,v\mid K)>0
]

说明 (u) 和 (v) 具有互补性。

典型软件修复场景：

### Buggy logic 与 constraint

```text
证据 A：
函数在输入为空时直接返回缓存结果

证据 B：
调用方要求空输入必须产生新的空对象
```

单独看 A 只能知道实现逻辑；
单独看 B 只能知道外部要求；
A+B 才能判断现有返回行为违反约束。

### Target symbol 与 dependency context

```text
证据 A：
目标函数调用 normalize()

证据 B：
normalize() 对特殊类型会保留原始格式
```

只有组合后才能理解为什么目标函数输出错误。

### Error path 与 state update

```text
证据 A：
异常从 parser 传播到 handler

证据 B：
handler 在异常前已经修改共享状态
```

组合后才能发现 rollback 或状态一致性问题。

---

## 2. 证据替代性

两条证据分别都能表达相同或近似相同的信息，读取其中一条后，另一条的边际价值显著降低。

若：

[
\Delta_G(v\mid K)>0
]

但：

[
\Delta_G(v\mid K\cup{u})\approx0
]

则 (u) 可以替代 (v)。

典型例子：

```text
完整函数体
vs
函数中对应 buggy branch

调用方源码
vs
已有测试中对调用方行为的断言

函数原始代码
vs
经过验证的语义摘要

同一约束在 docstring 和 test assertion 中的重复表达
```

替代性不是简单的文本相似。

两条文本可能词面差异很大，但修复信息相同：

```text
代码：
if value is None:
    return []

测试：
assert parse(None) == []
```

它们都可能表达“空输入返回空列表”这一约束。

---

## 3. 证据冗余

替代性与冗余相关，但不完全相同。

* **替代性**：两条证据都可以独立支持同一修复判断；
* **冗余性**：在当前 (K_t) 下，新证据没有提供新增信息。

定义：

[
R_G(u\mid K)
=

## \Delta_G(u\mid\varnothing)

\Delta_G(u\mid K)
]

当 (R_G) 很高时，说明候选原本可能有价值，但当前证据集合已经覆盖了它的信息。

这使 agent 能够避免：

```text
重复读取同一函数的多个粒度
连续获取多个相似调用点
反复读取表达同一约束的测试和文档
```

---

# 五、从单证据评分升级为集合价值建模

普通 reranker 对候选独立打分：

[
s(u\mid q)
]

状态感知 reranker：

[
s(u\mid q,K)
]

新方案还需要建模证据之间的交互：

[
s(u\mid q,K,\mathcal I(u,K))
]

其中 (\mathcal I(u,K)) 表示候选与已有证据的关系。

可以将候选价值写成：

[
Q(u\mid K)
=

\Delta_{\text{individual}}(u)
+
\alpha,\operatorname{Comp}(u,K)
-------------------------------

## \beta,\operatorname{Sub}(u,K)

\lambda,\operatorname{Cost}(u)
]

其中：

* (\operatorname{Comp}(u,K))：候选与现有证据的互补程度；
* (\operatorname{Sub}(u,K))：候选被已有证据替代的程度；
* `Cost`：获取和使用该证据的成本。

不过从统一性来看，更推荐直接预测条件边际价值：

[
\Delta_G(u\mid K)
]

因为互补和替代已经包含在条件边际价值中：

```text
与 K 互补
→ Δ 上升

被 K 替代
→ Δ 下降到接近 0

与 K 冲突或产生噪声
→ Δ 可能为负
```

显式的 complement/substitute head 可以作为辅助监督和解释模块。

---

# 六、Evidence Interaction Graph

为了适配 Nips 的序列选择，同时表达证据关系，可以为每个实例建立：

## Evidence Interaction Graph

节点是 evidence units：

```text
file
function
class
statement
branch
callsite
test
configuration
documentation
semantic summary
```

边类型包括：

```text
complements
substitutes
supports
depends_on
contains
duplicates
conflicts_with
```

示例：

```text
buggy_statement
      │ complements
      ▼
caller_constraint

function_unit
      │ substitutes
      ▼
branch_unit

test_assertion
      │ supports
      ▼
expected_behavior

function_summary
      │ duplicates
      ▼
function_body
```

图中的边不必全部作为强 gold，可以分为：

```text
deterministic structural edges
LLM-derived semantic edges
human-verified interaction edges
behavior-calibrated interaction edges
```

---

# 七、互补与替代标签的构造方式

## 1. 结构规则弱监督

可自动生成高精度弱标签。

### 互补候选

```text
changed symbol ↔ caller
buggy logic ↔ related condition
buggy logic ↔ existing test
field update ↔ field definition
exception site ↔ exception handler
target symbol ↔ called dependency
```

### 替代候选

```text
statement unit ↔ containing branch
branch unit ↔ containing function
raw code ↔ semantic summary
同一 symbol 的重叠窗口
表达同一 assertion 的多个测试
```

这些只是候选标签，不应直接视为行为真值。

## 2. LLM 交互判断

输入两条证据和当前状态，让 LLM 判断：

```json
{
  "relation": "complement | substitute | independent | conflict",
  "shared_information": "...",
  "unique_information_u": "...",
  "unique_information_v": "...",
  "joint_information": "...",
  "source_quotes": [],
  "confidence": 0.84
}
```

仍然要求 quote grounding 和程序验证。

## 3. 人工审计

在人工子集中标注：

```text
两条证据是否需要组合？
只看其中一条是否足够？
第二条是否增加修复信息？
两条是否只是不同粒度的重复？
```

## 4. 行为结果

外部行为验证不再作为独立创新，但仍可作为交互标签的验证方式。

比较：

[
V_G(K\cup{u,v})
]

与：

[
V_G(K\cup{u}),\quad
V_G(K\cup{v})
]

从而计算 (I_G(u,v\mid K))。

---

# 八、Nips 样本中的交互标签

更新后的训练样本可以是：

```json
{
  "qid": "swebench::train::xxx",
  "t": 2,
  "question": "...",
  "state": {
    "K_t": "...",
    "selected_unit_ids": [
      "unit_location",
      "unit_symbol"
    ]
  },
  "candidates": {
    "C_t": [
      "unit_buggy_logic",
      "unit_constraint",
      "unit_function_summary",
      "__STOP__"
    ]
  },
  "labels": {
    "action_label": {
      "acceptable_unit_ids": [
        "unit_buggy_logic"
      ]
    },
    "value_labels": {
      "unit_buggy_logic": 0.21,
      "unit_constraint": 0.08,
      "unit_function_summary": 0.01
    },
    "interaction_labels": {
      "unit_buggy_logic": {
        "complements_selected": [
          "unit_symbol"
        ],
        "substitutes_selected": [],
        "interaction_score": 0.17
      },
      "unit_function_summary": {
        "complements_selected": [],
        "substitutes_selected": [
          "unit_symbol"
        ],
        "interaction_score": -0.09
      }
    },
    "stop_label": {
      "should_stop": false
    }
  }
}
```

当 buggy logic 加入后，下一步 constraint 的价值可能提升：

[
\Delta_G(
u_{\text{constraint}}
\mid
K+{u_{\text{buggy}}}
)

>

\Delta_G(
u_{\text{constraint}}
\mid
K
)
]

这正是互补证据的序列效应。

---

# 九、模型结构的更新

模型可以保留统一的 Value-of-Information 主干，并增加交互建模。

## 编码

分别编码：

```text
bug report q
current evidence set K_t
candidate u
candidate–state interaction
```

状态不能只简单拼接所有证据，建议采用 set encoder 或 graph encoder：

[
h_K
=

\operatorname{SetEncoder}
(
{h_{u_1},\ldots,h_{u_t}}
)
]

候选与每个已选证据进行交互：

[
e_{u,i}
=

\operatorname{Interaction}
(h_u,h_{u_i})
]

聚合：

[
h_{u,K}
=

\operatorname{Aggregate}
(
{e_{u,i}}_{i=1}^{t}
)
]

最终预测：

[
\hat\Delta_G(u\mid K)
=

f(h_q,h_K,h_u,h_{u,K})
]

辅助输出：

```text
complementarity score
substitutability score
evidence role
deficit explanation
```

---

# 十、训练目标更新

总损失可以写成：

[
L
=

L_{\text{VOI}}
+
\lambda_aL_{\text{action}}
+
\lambda_iL_{\text{interaction}}
+
\lambda_sL_{\text{stop}}
+
\lambda_dL_{\text{deficit}}
]

## Value Loss

[
L_{\text{VOI}}
=

\left(
\hat\Delta_G(u\mid K)
---------------------

\Delta_G(u\mid K)
\right)^2
]

数据不足时也可以使用 pairwise ranking：

[
\Delta(u_i\mid K)>\Delta(u_j\mid K)
]

则：

[
L_{\text{pair}}
=

-\log\sigma(
Q(u_i\mid K)-Q(u_j\mid K)
)
]

## Interaction Loss

分类形式：

```text
complement
substitute
independent
conflict
```

或者回归交互值：

[
L_{\text{interaction}}
=

(\hat I_G-I_G)^2
]

## Set-valued Action Loss

[
L_{\text{set-action}}
=

-\log
\frac{
\sum_{u\in A_t}\exp Q(u\mid K_t)
}{
\sum_{u\in C_t}\exp Q(u\mid K_t)
}
]

---

# 十一、第三项创新能够回答的新问题

加入互补与替代后，项目不再只回答：

```text
下一条证据是什么？
```

而可以回答：

```text
为什么这条证据在当前状态下有价值？
它补充了已有证据的什么缺口？
它是否需要和另一条证据组合？
它是否已经被现有证据替代？
为什么一个相关候选仍然不值得读取？
```

这也能够解释一些典型现象：

## 相关但低价值

某函数与 issue 高度相关，但其内容已被已有 branch unit 覆盖，因此：

[
\text{Relevance high},\quad
\Delta_G\approx0
]

## 单独低价值但组合高价值

某 constraint 看起来与 bug 文本相似度低，但与 buggy logic 组合后：

[
I_G(u,v\mid K)\gg0
]

## 更多上下文反而更差

新增重复或冲突证据导致：

[
\Delta_G(u\mid K)<0
]

这能从机制上解释为什么固定 Top-k 或固定获取轮次可能发生过度探索。

---

# 十二、更新后的三项正式贡献

## 贡献一：面向修复价值的证据获取任务

> 我们提出 generator-calibrated repair evidence acquisition，将仓库探索从位置相关性和覆盖率优化提升为成本约束下的修复证据价值优化，并构建包含多值弱监督、证据成本和质量证书的 benchmark。

重点是：

```text
repair-oriented value
budgeted acquisition
pre-fix repository
leakage-safe supervision
```

## 贡献二：Repair Value-of-Information Policy

> 我们提出 Repair Value-of-Information Policy，估计当前证据状态价值和候选证据的条件边际收益，在考虑获取成本后选择下一证据，并在剩余候选不再具有正净收益时自适应停止。

核心公式：

[
V_G(K)
]

[
\Delta_G(u\mid K)
=

V_G(K\cup{u})-V_G(K)
]

[
Q_G(u\mid K)
=

\Delta_G(u\mid K)-\lambda Cost(u)
]

## 贡献三：证据互补与替代感知建模

> 我们显式建模修复证据之间的互补、替代和冗余关系，使策略能够识别需要组合使用的跨位置证据，同时避免重复读取能够相互替代的代码、测试或语义上下文。

包括：

```text
evidence interaction graph
context-dependent marginal value
complementarity prediction
substitutability prediction
interaction-aware acquisition
set-valued equivalent evidence
```

---

# 十三、更新后的论文主张

推荐摘要级英文表述：

> Existing repository explorers largely score evidence units independently, overlooking that repair evidence can be complementary, substitutable, or redundant. We formulate generator-calibrated repair evidence acquisition and introduce a repair value-of-information policy that estimates context-dependent marginal evidence value. By modeling evidence interactions, the policy combines individually insufficient but jointly useful evidence, avoids substitutable context, and stops when further repository exploration is no longer cost-effective.

中文：

> 现有仓库探索方法通常独立评估代码证据，忽略了修复证据之间可能存在互补、替代和冗余关系。我们提出面向固定修复器校准的修复证据获取任务，并设计修复信息价值策略，以估计候选证据在当前状态下的条件边际价值。通过建模证据交互，策略能够组合单独不足但联合有用的证据，避免重复读取可替代上下文，并在继续探索不再具有成本收益时停止。

---

# 十四、外部行为验证的新位置

删除“充分性和最小性的外部行为验证”作为创新点，不代表删除这些实验。

它们应放入：

```text
Experimental validation
Construct validity
Evaluation protocol
```

用于验证前两项和第三项创新：

```text
VOI 预测是否对应真实修复增益
互补关系是否产生超加性收益
替代关系是否意味着第二条证据边际收益接近零
STOP 是否优于固定 Top-k/固定轮次
agent 是否位于更优的 success–cost Pareto 前沿
```

论文贡献强调“提出了什么任务和方法”；实验负责证明“这些定义和方法是否成立”。

---

# 十五、更新后的标题建议

最推荐：

## **Beyond Independent Context: Repair Value of Information with Complementary and Substitutable Evidence**

更简洁：

## **Learning Which Evidence Works Together for Software Repair**

强调获取与停止：

## **Gather, Combine, or Skip: Interaction-Aware Evidence Acquisition for Software Repair**

强调方法：

## **Interaction-Aware Repair Value of Information for Adaptive Repository Exploration**

其中最适合当前创新结构的是：

> **Gather, Combine, or Skip: Interaction-Aware Evidence Acquisition for Software Repair**

它对应三个核心决策：

```text
Gather  → 获取高边际价值证据
Combine → 识别互补证据组合
Skip    → 跳过替代或冗余证据，并适时停止
```





---

## 一、修正后的项目定义

项目是一个结合以下四部分的软件修复证据收集系统：

```text
SWE 修复任务与仓库环境
        ↓
RAG 候选证据检索
        ↓
NIPS 序贯证据价值判断
        ↓
Agent 执行搜索、读取、组合和停止
        ↓
输出充分且可用于修复的证据包
```

四个模块的职责分别是：

```text
SWE：定义要收集哪些软件工程证据
RAG：从仓库中找到候选证据
NIPS：判断候选证据是否值得继续获取
Agent：实际执行检索、读取、组合、跳过和停止
```

系统不进入补丁生成阶段。

---

# 二、最终完整流程

## 1. 输入软件缺陷任务

系统输入包括：

```text
缺陷描述
目标代码仓库
修复前提交 base_commit
可选的错误日志
可选的异常栈
可选的问题讨论
```

例如：

```text
问题描述：
parse_value 在输入为空时返回共享对象，
调用方修改结果后会影响下一次调用。
```

系统只能读取修复前仓库。

不能读取：

```text
正确补丁
修复后的代码
补丁新增内容
人工直接给出的最终修改方案
```

---

## 2. 构建修复前证据库

系统将仓库内容切分为不同粒度的证据单元：

```text
仓库级信息
目录和模块
代码文件
类
函数和方法
代码分支
关键语句
变量定义
调用位置
被调用函数
异常处理逻辑
测试文件
配置文件
文档和注释
```

每个证据单元记录：

```text
证据编号
仓库
提交版本
文件路径
符号名称
开始行和结束行
证据类型
代码内容
内容哈希
读取成本
结构关系
```

例如：

```json
{
  "evidence_id": "ev-001",
  "file_path": "src/parser.py",
  "evidence_type": "function",
  "symbol": "parse_value",
  "start_line": 35,
  "end_line": 82,
  "token_cost": 436
}
```

这一步形成可供 RAG 和 Agent 使用的 SWE 证据底座。

---

## 3. 构建证据关系图

证据不能只按文本相似度处理，还需要建立软件结构关系。

关系包括：

```text
文件包含函数
类包含方法
函数调用函数
变量被读取
变量被写入
函数抛出异常
代码处理异常
文件导入模块
测试引用函数
配置影响模块
代码块共享状态
符号定义与使用
```

例如：

```text
test_parse_empty
    ──引用──>
parse_value
    ──调用──>
get_cached_empty
    ──读取──>
EMPTY_RESULT
```

证据关系图用于帮助系统发现：

```text
直接相关证据
依赖证据
调用方约束
行为说明
互补证据
重复证据
替代证据
```

---

## 4. RAG 召回初始候选证据

RAG 根据缺陷描述进行分层召回。

第一级是文件级检索：

```text
缺陷描述
→ 相关文件
```

例如：

```text
src/parser.py
src/cache.py
tests/test_parser.py
src/utils.py
```

第二级是符号和代码块级检索：

```text
缺陷描述 + 已收集证据
→ 函数、方法、分支、调用方、依赖、测试
```

检索方法可以组合：

```text
关键词检索
向量检索
混合检索
交叉编码重排序
代码图邻居扩展
符号索引检索
```

RAG 的职责只是提供候选集合：

[
C_t={e_1,e_2,\ldots,e_n}
]

RAG 不直接决定哪些证据最终进入证据包。

---

# 三、NIPS 在项目中的作用

NIPS 负责序贯信息获取决策。

每轮状态可以表示为：

[
S_t=(q,K_t,C_t,B_t)
]

其中：

* (q)：缺陷描述；
* (K_t)：当前已经收集的证据；
* (C_t)：当前候选证据；
* (B_t)：剩余读取预算。

NIPS 策略需要判断：

```text
下一条应该收集什么证据
哪些证据应该组合
哪些证据已经被替代
哪些证据是重复的
当前还缺少什么信息
现有证据是否已经充分
是否应该停止
```

项目中的动作可定义为：

```text
GATHER：读取一条证据
EXPAND：沿调用或依赖关系扩展
COMBINE：组合互补证据
SKIP：跳过低价值或重复证据
REPLACE：用更精确的证据替换粗粒度证据
STOP：证据充分后停止
```

---

# 四、Agent 的实际工作循环

Agent 是执行层。

每一轮执行以下流程：

```text
1. 分析当前缺陷和已有证据
2. 调用 RAG 获取候选
3. 使用 NIPS 策略评价候选
4. 选择一个动作
5. 调用工具读取证据
6. 更新当前证据集合
7. 判断还缺少哪些证据角色
8. 重新检索或停止
```

Agent 可调用的工具包括：

```text
搜索文件
搜索符号
读取函数
读取代码范围
查找定义
查找引用
查找调用方
查找被调用函数
读取异常处理逻辑
读取测试描述
读取配置
展开代码图邻居
比较两个证据是否重复
生成文件或函数摘要
```

完整循环是：

```text
问题描述
    ↓
检索候选文件
    ↓
读取最可能相关的文件或函数
    ↓
分析当前证据缺口
    ↓
检索调用方、依赖、状态和行为约束
    ↓
组合互补证据
    ↓
跳过重复证据
    ↓
判断证据充分性
    ↓
停止并输出证据包
```

---

# 五、“证据足够”的定义

由于项目不生成补丁、不执行测试，因此不能把“测试通过”作为充分性标准。

本项目必须把“足够”定义成一个可计算的证据完备性条件。

建议将充分性拆成六个维度。

## 1. 故障定位充分性

证据是否能够定位：

```text
相关文件
相关类或函数
可疑代码块
可能的修改区域
```

例如只找到 `parser.py` 不够，还应尽量定位到：

```text
parse_value 函数
空输入处理分支
返回共享对象的语句
```

---

## 2. 故障原因充分性

证据是否能够解释：

```text
错误为什么发生
错误数据从哪里产生
错误状态如何传播
哪个条件触发错误
错误违反了什么约束
```

例如：

```text
空输入分支返回全局共享列表
调用方会修改返回结果
因此修改会污染后续调用
```

这才构成完整的故障原因链。

---

## 3. 修复约束充分性

证据是否说明修复时必须满足的条件：

```text
期望行为
接口约束
返回值约束
状态约束
异常约束
兼容性约束
调用方依赖
```

例如：

```text
每次空输入调用都应返回独立对象
返回类型必须保持为列表
不能改变非空输入的处理逻辑
调用方允许修改返回值
```

这些约束决定修复方案不能随意生成。

---

## 4. 依赖上下文充分性

证据是否覆盖修复位置涉及的必要依赖：

```text
被调用函数
调用方
共享变量
缓存
配置
继承关系
接口实现
异常传播路径
```

系统不能只收集目标函数，而忽略决定其行为的外部状态。

---

## 5. 证据互补充分性

最终证据包不能只有多个相似片段，而应覆盖不同信息角色。

典型角色包括：

```text
定位证据
原因证据
行为约束证据
依赖证据
调用方证据
边界条件证据
修复范围证据
```

例如：

```text
目标函数
+
共享缓存定义
+
调用方修改行为
+
接口预期说明
```

这四类证据互相补充，才形成可修复证据链。

---

## 6. 证据非冗余性

充分不等于越多越好。

系统还应保证：

```text
没有大量重复代码
没有同一信息的多个低质量副本
没有与缺陷无关的大文件
没有被精确证据替代的粗粒度内容
```

例如已经读取目标函数后，通常没有必要再保留完整的两千行文件，除非文件级上下文确实提供额外约束。

---

# 六、充分性评分模型

可以定义一个证据充分性分数：

[
S(K)=
w_lS_l+
w_cS_c+
w_rS_r+
w_dS_d+
w_iS_i-
w_nS_n
]

其中：

* (S_l)：定位充分性；
* (S_c)：原因解释充分性；
* (S_r)：修复约束充分性；
* (S_d)：依赖覆盖充分性；
* (S_i)：证据互补性；
* (S_n)：冗余和噪声。

当满足以下条件时停止：

[
S(K_t)\ge \tau
]

并且：

[
\max_{e\in C_t}\Delta S(e\mid K_t)\le \epsilon
]

意思是：

```text
当前证据包已经达到充分性阈值
并且
继续读取其他证据带来的提升已经很小
```

这就是本项目不依赖测试反馈的停止条件。

---

# 七、“能修复”应该怎样严谨表达

这里需要区分两个概念。

## 项目可以保证的

项目可以保证输出证据包具备：

```text
明确的缺陷位置
完整的原因链
必要的调用和依赖上下文
明确的行为与修复约束
充分的证据角色覆盖
可追溯的代码来源
有限的冗余和噪声
```

也就是：

> 证据在信息层面足以支持一个具备正常代码修复能力的模型生成修复方案。

## 项目不能直接证明的

由于不生成补丁，也不运行测试，项目本身无法严格证明：

```text
某个下游模型一定能生成正确补丁
生成的补丁一定可以运行
生成的补丁一定通过全部测试
```

因此，论文或方案中最好不要写：

> 保证一定修复成功。

更严谨的表述是：

> 保证证据达到预定义的修复充分性标准。

或者：

> 输出能够支持下游修复模型完成缺陷定位、原因分析和补丁生成的充分证据集。

这是理论和工程上都更准确的表述。

---

# 八、如何训练充分性模型

既然没有测试反馈，训练监督应来自已有数据中的修复证据，而不是运行修复器。

可以使用的监督包括：

```text
正确补丁的旧代码位置
人工标注的相关上下文
成功修复轨迹中的读取记录
修改文件和修改函数
问题描述与代码位置对应关系
调用关系和依赖关系
人工证据角色标注
```

训练样本可以构造成：

```text
缺陷描述 q
当前证据集合 K
候选证据 e
候选是否补充缺失信息
候选属于哪种证据角色
加入候选后是否达到充分
```

正样本例如：

```text
当前已有目标函数
但缺少调用方约束

候选：
调用方对返回对象进行了修改

标签：
高价值、互补、应收集
```

负样本例如：

```text
当前已经有完整目标函数

候选：
同一个函数的摘要

标签：
重复、低价值、应跳过
```

停止标签可以根据证据角色覆盖生成：

```text
修改区域已覆盖
原因链已覆盖
行为约束已覆盖
必要依赖已覆盖
没有明显信息缺口
→ STOP
```

---

# 九、修正后的离线流程

```text
外部 SWE 数据集
    ↓
数据下载与版本冻结
    ↓
字段统一和实例去重
    ↓
训练集、验证集、测试集划分
    ↓
准备修复前 Git 快照
    ↓
抽取补丁旧位置、人工上下文和轨迹锚点
    ↓
构建修复前证据单元
    ↓
建立代码结构与证据关系图
    ↓
训练分层 RAG 检索器
    ↓
构造证据序列与证据缺口标签
    ↓
构造互补、替代、重复和停止标签
    ↓
训练 NIPS 证据价值模型
    ↓
训练 Agent 序贯收集策略
    ↓
评估证据充分性、精确性和成本
```

这里不存在：

```text
补丁生成
补丁应用
测试运行
测试结果反馈
修复成功率奖励
```

---

# 十、修正后的在线流程

```text
输入缺陷描述和修复前仓库
    ↓
RAG 召回相关文件
    ↓
NIPS 选择最有价值文件
    ↓
Agent 读取文件或函数
    ↓
识别当前证据缺口
    ↓
RAG 动态召回调用方、依赖、状态和约束
    ↓
NIPS 判断互补、替代与冗余
    ↓
Agent 继续收集或跳过
    ↓
充分性模型评估当前证据包
    ↓
达到充分性标准后停止
    ↓
输出结构化修复证据包
```

---

# 十一、最终输出内容

系统最终输出的不是补丁，而是证据包，例如：

```json
{
  "instance_id": "task-001",
  "sufficiency_score": 0.91,
  "stop_reason": "核心证据角色已覆盖，剩余候选增益不足",
  "fault_location": {
    "file": "src/parser.py",
    "symbol": "parse_value",
    "lines": [45, 49]
  },
  "root_cause": [
    "空输入分支返回全局共享列表",
    "调用方允许修改返回列表",
    "修改会污染后续调用"
  ],
  "repair_constraints": [
    "每次调用必须返回独立对象",
    "返回值类型必须保持为列表",
    "不能改变非空输入行为"
  ],
  "evidence": [
    {
      "role": "故障定位",
      "file": "src/parser.py",
      "symbol": "parse_value"
    },
    {
      "role": "状态来源",
      "file": "src/cache.py",
      "symbol": "EMPTY_RESULT"
    },
    {
      "role": "调用方约束",
      "file": "src/consumer.py",
      "symbol": "process_result"
    }
  ],
  "missing_information": [],
  "redundant_candidates_skipped": 7,
  "evidence_tokens": 1680
}
```

---

# 十二、最终准确表述

建议将项目定义固定为：

> **本项目面向软件缺陷修复场景，构建一个融合 RAG、Agent、NIPS 序贯信息获取和 SWE 证据建模的自动证据收集框架。系统在修复前仓库中动态检索、选择和组合代码证据，识别互补、替代与冗余关系，并依据故障定位、原因解释、修复约束和依赖覆盖等维度判断证据充分性，最终输出可供下游修复模型直接使用的结构化充分证据包。**

一句话版本：

> **我们不负责修复代码，我们负责把修复代码所需要的证据找全、找准，并在证据已经足够时停止。**

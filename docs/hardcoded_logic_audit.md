# BankReg-TrustRAG 硬编码逻辑审计

## 审计目标

本次审计区分两类逻辑：

1. 用关键词、`qa_type` 或固定模板模拟用户意图理解的上层决策；
2. 日期、数值、Excel 坐标、单位、证据和安全校验等必要的确定性逻辑。

新 Agentic 路径由 `QueryPlan.sub_questions`、`retrieval_tasks`、`operations`、依赖关系和 `answer_requirements.required_outputs` 驱动。Legacy 路径暂不删除，由 `BANKREG_AGENTIC_PLANNER_ENABLED=false` 保留为快速回退方案。

## 审计结果

| 位置 | 原逻辑/风险 | 处理结论 | 替代方案或保留原因 |
|---|---|---|---|
| `bankreg_trustrag/query.py` | 通过“最高、计算、总结、判断”等词和 `qa_type` 推断 intent/requirements | Legacy 保留，不再作为新路径主决策 | `QueryPlanner` 直接读取完整问题并输出结构化 `QueryPlan`；Feature Flag 关闭或选择题兼容路径才使用旧解析器 |
| `bankreg_trustrag/agent.py` 的 `build_agent_workflow()` | 旧工作流仍包含 requirements 辅助规则及历史 `qa_type` 提示 | Legacy 保留并隔离 | 新路径使用 `BoundedAgentExecutor`，逐条执行 Planner 生成的检索任务和计算 DAG |
| `bankreg_trustrag/reasoning.py` | 旧回答包含季度、差值、单表查询等固定分支和 Python 回答模板 | 已退出正常 Agentic 路径；仅由 Feature Flag 关闭、终止状态或模型故障兼容路径使用 | 正常路径使用 `AnswerGenerator`，输入完整 QueryPlan、全部 RetrievalResult/Evidence、CalculationResult 与来源账本 |
| `bankreg_trustrag/generation.py` | 旧 Prompt 强制选择题、跨文件题使用固定段落与说明顺序 | 已删除固定表达要求 | 兼容路径的模型也作为 Evidence-Grounded Answer Agent，自由组织回答；Python 草稿只在模型不可用时兜底 |
| `bankreg_trustrag/service.py` | 历史 ask 流程混合问题分类、检索路由和固定生成 | 已封装为 `_ask_legacy()` | `ask()` 通过 Feature Flag 调用 `_ask_agentic()`；Planner 失败时按配置优雅回退 |
| `bankreg_trustrag/retrieval/index.py` | 存在日期、季度、数字及表格列识别正则 | 保留 | 这些是底层 Metadata/Table 检索与数值归一化，不负责决定用户真实意图 |
| `bankreg_trustrag/retrieval_tools.py` | 用季度格式判断多候选是否缺少维度 | 保留 | 属于检索后二阶段候选唯一性校验；与 Planner 语义澄清共同决定是否提问用户 |
| `bankreg_trustrag/calculator.py` | 运算类型使用有限集合 | 保留 | 运算是确定性 Tool contract，不是用户 intent 枚举；所有金融运算使用 `Decimal` 并保存 trace |
| `bankreg_trustrag/verification.py` | 用正则核验数字、日期、单位、文号；曾根据布尔结果/差值裁决“一致、不一致、完全相等” | 保留事实核验，删除业务语义裁决 | 数字只能来自 Evidence 或 CalculationResult；Verifier 不再推导差值，也不再判断“基本一致、明显不同”等措辞 |
| `bankreg_trustrag/ingestion/parsers.py` | 日期、章节、季度、单位、Excel 单元格解析规则 | 保留 | 属于文档结构化和数据标准化，不参与问题语义路由 |
| `scripts/evaluate.py` | 从答案解析选项和单元格 | 保留 | 只用于评测指标计算，不参与线上问答决策 |
| Choice Agent | 最大/最小值、选项标签、逐项检索采用确定性代码 | 保留 | 这是可靠性机制：选项证据隔离、确定性数值比较和证据不足转人工，不应交给 LLM 猜测 |

## 已新增的上层替代机制

- `query_plan.py`：严格 Pydantic Schema、结果绑定、检索/计算依赖校验。
- `llm_client.py`：统一超时、重试、JSON Structured Output 和错误恢复。
- `query_planner.py`：大模型从完整自然语言生成多任务 QueryPlan，不接收 Gold 答案或题号标签。
- `retrieval_tools.py`：对现有 HybridIndex 的薄包装，按每项任务的来源与语义约束独立检索。
- `calculator.py`：统一 Decimal 计算结果和可追溯公式。
- `agentic_executor.py`：显式有界状态机，执行检索 DAG、计算 DAG、二阶段澄清和回答重试。
- `answer_generator.py`：Evidence-Grounded Answer Agent 接收完整 QueryPlan、全部证据、CalculationResult、来源账本和核验反馈，自由组织最终自然语言；LLM API/结构化输出失败时才使用轻量 fallback formatter。
- `completeness.py`：生成前检查 required outputs，生成后检查每项回答绑定。

## 当前保留边界

- Legacy 路径未删除，确保既有测试、接口与紧急回滚能力不受影响。
- 选择题继续使用现有 Choice Agent 产生隔离证据和确定性比较结果；当 LLM 可用时，最终措辞仍交给 Evidence-Grounded Answer Agent。
- 新路径没有根据具体测试题、question id、Gold 答案或文件名写死结果。

## 后续清理条件

只有在 Agentic 路径完成线上观测、改写鲁棒性和跨文件回归后，才考虑删除 Legacy 上层关键词路由与固定模板。在此之前，新增能力应优先扩展 QueryPlan Tool contract，不应继续向旧 `qa_type` 分支添加问题特判。

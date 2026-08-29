# BankReg-TrustRAG

面向银行监管制度与统计报表的本地可信 RAG 问答系统。系统保留“答案—证据—原始文件位置”的完整链路，支持监管制度检索、Excel 表格取数、跨文件判断、证据校验和不足证据拒答。

## 当前能力

- 支持 DOC、DOCX、PDF、XLS、XLSX 文档解析。
- 支持 BM25、BGE 向量检索、元数据过滤、结构化表格检索、RRF 融合和 BGE-Reranker。
- 表格证据保留指标、期间、单位、Sheet、行列上下文、原始值和单元格地址。
- 支持五类问题：监管事实、条款与阈值、业务流程、统计表查询、跨文件判断。
- 对数字、日期、规范性用语和证据覆盖进行校验；证据不足时澄清或拒答。
- 跨文件判断采用“监管规则/指标解释 → 统计表数值 → 确定性比较”的流程。
- 前端采用“监管证据台账”风格，显示信任分、决策、追踪编号和证据链。
- 选择题由 Choice Agent 先识别意图，再对 A/B/C/D 选项分别检索和逐项核验；证据不足或选项得分接近时转入人工确认，不自动猜测。
- 可选接入本地或获授权的 OpenAI-compatible 大模型：检索到的最小证据集会进入上下文，模型回答必须经过 Claim、数字、日期和规范性用语核验；模型不可用或核验失败时回退到可审计的确定性答案。
- 可通过 Feature Flag 启用 Agentic 查询规划：LLM 把完整问题拆成带结果绑定的检索任务和计算 DAG，现有 HybridIndex 与表格取值工具逐任务执行，最终回答再做完整性和事实核验。

## Agentic 查询规划

新路径采用显式有界状态机，暂不引入 LangGraph：

```text
完整用户问题
  → LLM Query Planner（Structured QueryPlan）
  → 按来源/指标/时间约束逐任务检索
  → Decimal Calculator 执行求和、差值、同比、比较等操作
  → 确定性检查每个 AnswerRequirement.required_outputs
  → Grounded LLM Answer Generator
  → 数字、日期、单位、引用和 Calculation Trace 核验
```

该路径不重写 BM25、BGE、RRF、Reranker、Metadata Filter 或 Excel Cell 检索。执行 Trace 会记录规划、检索、计算、完整性、核验和分阶段耗时；前端只显示“拆解任务、检索证据、执行计算、核验完成”等高层状态，不展示模型内部推理。

## 目录说明

```text
bankreg_trustrag/       后端、检索、推理、校验和存储代码
frontend/               前端页面
scripts/                语料导入、评测和服务烟测脚本
tests/                  pytest 测试
03-*                    原始竞赛数据集
BankReg-TrustRAG_项目书.docx  项目设计依据
.env.example            环境变量模板
server.py               FastAPI 应用入口
```

压缩包默认不包含 `.venv`、`artifacts/models`、`artifacts/bge_vectors`、SQLite 数据库和生成后的大体量证据 JSONL。这些文件可以根据原始数据重新生成，避免把本机环境和模型缓存带入部署包。

## Windows 安装

在项目根目录执行：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

也可以直接使用项目已有环境：

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

## 配置 BGE

复制配置模板：

```powershell
Copy-Item .env.example .env
```

正式模式要求本地模型已经存在：

```env
BANKREG_BGE_MODE=required
BANKREG_BGE_LOCAL_FILES_ONLY=1
```

### 接入大模型生成

默认 `BANKREG_LLM_PROVIDER=none`，不调用任何大模型，也不需要 API Key。若使用本地 Ollama、vLLM 或其他 OpenAI-compatible 服务，请配置：

```env
BANKREG_LLM_PROVIDER=openai_compatible
BANKREG_LLM_MODEL=你的模型名
BANKREG_LLM_BASE_URL=http://127.0.0.1:11434/v1
BANKREG_LLM_API_KEY=
# 建议为查询规划单独配置轻量、非推理模型；URL/Key 留空时复用上面配置
BANKREG_PLANNER_MODEL=你的轻量规划模型名
BANKREG_PLANNER_BASE_URL=
BANKREG_PLANNER_API_KEY=
BANKREG_AGENTIC_PLANNER_ENABLED=true
BANKREG_AGENTIC_PLANNER_FAILURE_MODE=legacy
```

系统只把选中的证据文本、来源标题、页码/单元格位置和确定性计算事实发送到该地址，不发送本地文件路径。使用外部地址前，必须确认数据授权和合规要求。

开发环境暂时没有模型时，可以改为：

```env
BANKREG_BGE_MODE=auto
```

此时系统会明确记录降级到字符 n-gram 检索，而不会伪装成 BGE 已启用。竞赛数据只在本地处理，不上传到第三方服务。

## 初始化知识库

如果压缩包中没有预生成的 `artifacts`，先运行：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_corpus.py
```

脚本会解析 `03-*` 数据集，生成文档、文本证据、表格证据和 SQLite 索引。首次使用 BGE 时，还会在 `artifacts/bge_vectors/` 建立本地向量索引。

## 启动服务

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000/
```

主要接口：

- `GET /health`：健康状态和知识库规模
- `POST /api/qa`：问答接口
- `GET /api/documents`：浏览和筛选知识库目录
- `GET /api/documents/{doc_id}/source`：安全打开已入库的原始文件
- `GET /api/evidence/{evidence_id}`：查看原始证据
- `GET /api/history`：查看问答审计记录
- `POST /api/documents/ingest`：重新导入语料

选择题可以直接把题干和 A/B/C/D 选项粘贴到同一个问题框中，也可以通过 API 单独传入 `choices` 数组。返回的 `query_plan.agent` 会记录意图识别、逐项检索、候选证据和人工确认状态。

示例请求：

```powershell
Invoke-RestMethod http://127.0.0.1:8000/api/qa `
  -Method Post `
  -ContentType 'application/json' `
  -Body (@{question='2025年商业银行主要监管指标情况表中的不良贷款率是多少？'} | ConvertTo-Json)
```

## 自动测试

执行完整测试：

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

执行 FastAPI 服务烟测：

```powershell
.\.venv\Scripts\python.exe scripts\test_service.py
```

烟测结果保存到 `artifacts/service_smoke_report.json`。测试结果必须以本机最新一次完整运行输出为准，不在文档中写死历史通过数。

向其他授权使用者交付项目时，请优先发送不含数据和模型的复现包，并参考 [REPRODUCE.md](REPRODUCE.md)。

## 评测

```powershell
.\.venv\Scripts\python.exe scripts\build_eval_jsonl.py `
  ".\03-金融大模型与智能体赛道-南京银行-面向银行业监管制度与统计报表的可信RAG问答\QA数据.xlsx" `
  artifacts\eval.jsonl

.\.venv\Scripts\python.exe scripts\validate_eval_jsonl.py artifacts\eval.jsonl
.\.venv\Scripts\python.exe scripts\evaluate.py artifacts\eval.jsonl --limit 20
.\.venv\Scripts\python.exe scripts\ablation.py artifacts\eval.jsonl --limit 20
```

评测目标阈值是验收目标，必须以实际运行结果为准，不能把未运行的指标标记为通过。

## 常见问题

### 端口 8000 已被占用

查看占用进程：

```powershell
netstat -ano | Select-String ':8000'
```

也可以换端口启动：

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8001
```

### 查询速度较慢

首次启动需要加载本地 BGE 模型，首次查询还可能建立向量索引。模型和向量索引生成后，后续请求会更快。正式模式建议使用 CPU 内存允许范围内的批大小，并保留 `BANKREG_BGE_LOCAL_FILES_ONLY=1`。

### 数据不存在或年份不匹配

系统不会用相似年份替代请求年份。请检查文件是否位于 `03-*` 数据集目录，并重新执行语料导入；对于知识库没有的年份，系统应返回澄清或拒答。

## 数据安全

原始竞赛数据和生成的证据索引仅限授权环境本地使用。不要把数据集、SQLite、证据 JSONL、模型输入或 API 密钥上传到公共仓库或第三方解析/向量服务。

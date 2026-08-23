# BankReg-TrustRAG 复现说明

本发布包只包含项目源码和复现脚本，不包含竞赛原始数据、解析后的证据索引、SQLite 数据库、本地 BGE 模型、向量索引、`.env` 或 Python 虚拟环境。请仅在已获得数据授权的环境中复现。

## 1. 环境要求

- Windows 10/11、Linux 或 macOS
- Python 3.9+（推荐 3.12）
- 可选：Docker Desktop
- 可选：本地 BGE 模型。无模型时可使用 `auto` 模式，系统会明确降级，不会伪装为 BGE 检索。

## 2. 准备授权数据

将获得授权的原始竞赛数据目录放到项目根目录。默认目录名为：

```text
03-金融大模型与智能体赛道-南京银行-面向银行业监管制度与统计报表的可信RAG问答
```

如果使用其他目录名，请在 `.env` 中设置 `BANKREG_DATA_DIR`。不要将原始数据、生成的证据 JSONL、SQLite 数据库或本地模型提交到公开仓库。

## 3. 本地 Python 复现

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

无本地 BGE 模型时，将 `.env` 中的 `BANKREG_BGE_MODE` 改为 `auto`；有模型时，保持 `required` 并把模型放到 `BANKREG_BGE_CACHE_DIR` 指定目录。

大模型生成功能默认关闭。如果使用本地或已获授权的 OpenAI-compatible 服务，可在 `.env` 中设置 `BANKREG_LLM_PROVIDER`、`BANKREG_LLM_MODEL` 和 `BANKREG_LLM_BASE_URL`；API Key 仅在服务确实需要时填写。模型回答会经过现有核验器，核验失败不会直接展示。

导入授权数据并建立知识库：

```powershell
.\.venv\Scripts\python.exe scripts\ingest_corpus.py
```

启动服务：

```powershell
.\.venv\Scripts\python.exe -m uvicorn server:app --host 127.0.0.1 --port 8000
```

打开 `http://127.0.0.1:8000/`。首次 BGE 查询可能需要建立本地向量索引。

## 4. 验证

```powershell
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe scripts\test_service.py
```

## 5. Docker 复现

创建 `data` 目录，并将授权数据放入其中。然后：

```powershell
docker compose build
docker compose run --rm bankreg-trustrag python scripts/ingest_corpus.py
docker compose up
```

服务只绑定到本机 `127.0.0.1:8000`。Docker 模式下默认使用 `auto` BGE 策略；如需真实 BGE 检索，请把已授权的本地模型挂载到容器，并按 `.env.example` 设置模型路径与 `required` 模式。

## 6. 已知限制

- 完整评测集已提供，但正式全量 306 题指标仍需要在目标机器上完成运行与验收。
- 文档关系当前只安全识别重复文件和明确附件，不会猜测修订或废止关系。
- 原始文件可安全打开；PDF 页、Word 段落和 Excel 单元格的应用内精确跳转仍待增强。

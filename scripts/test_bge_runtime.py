import json
from pathlib import Path

from sentence_transformers import SentenceTransformer

MODEL_PATH = (
    r"E:\ProgramFiles\code\mycode\BankReg-TrustRAG"
    r"\artifacts\models\models--BAAI--bge-small-zh-v1.5"
    r"\snapshots\7999e1d3359715c523056ef9478215996d62a620"
)

TEXT_FILE = Path(
    r"E:\ProgramFiles\code\mycode\BankReg-TrustRAG"
    r"\artifacts\text_evidence.jsonl"
)

print("1. 加载模型")

model = SentenceTransformer(
    MODEL_PATH,
    device="cpu",
)

print("2. 模型加载成功")

texts = []

with TEXT_FILE.open("r", encoding="utf-8") as f:
    for line in f:
        if not line.strip():
            continue

        item = json.loads(line)
        content = str(item.get("content") or "").strip()

        if content:
            texts.append(content)

print("3. 文本数量:", len(texts))

print("4. 开始批量编码")

vectors = model.encode(
    texts,
    batch_size=32,
    normalize_embeddings=True,
    show_progress_bar=True,
)

print("5. 全部编码成功")
print(vectors.shape)
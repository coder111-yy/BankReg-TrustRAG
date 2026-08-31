import os

# 必须放在 transformers / sentence_transformers import 之前
os.environ["HF_ENABLE_PARALLEL_LOADING"] = "false"
os.environ["HF_PARALLEL_LOADING_WORKERS"] = "1"

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

from sentence_transformers import CrossEncoder
MODEL_PATH = (
    r"E:\ProgramFiles\code\mycode\BankReg-TrustRAG"
    r"\artifacts\models\models--BAAI--bge-reranker-base"
    r"\snapshots\2cfc18c9415c912f9d8155881c133215df768a70"
)

print("1. 开始加载 reranker")

model = CrossEncoder(
    MODEL_PATH,
    device="cpu",
)

print("2. reranker 加载成功")

pairs = [
    (
        "在截至当期-账面余额口径下哪一项最高？",
        "资金运用余额为281574亿元。"
    ),
    (
        "在截至当期-账面余额口径下哪一项最高？",
        "年化综合收益率为3.22%。"
    ),
    (
        "在截至当期-账面余额口径下哪一项最高？",
        "银行存款为21558亿元。"
    ),
]

print("3. 开始 rerank")

scores = model.predict(
    pairs,
    batch_size=8,
    show_progress_bar=False,
)

print("4. rerank成功")
print(scores)
from __future__ import annotations

import numpy as np
import pytest

from bankreg_trustrag.retrieval.bge import BGEConfig, PersistentVectorIndex
from bankreg_trustrag.retrieval.index import HybridIndex


class FakeBGE:
    config = BGEConfig(mode="required", embedding_model="fake-bge", reranker_model="fake-reranker")
    enabled = True
    status = {"mode": "required", "embedding_loaded": True, "reranker_loaded": True, "error": None}

    def encode(self, texts):
        vectors = []
        for text in texts:
            value = 1.0 if "客户资金" in text or "不得" in text else 0.0
            vectors.append([value, 1.0 - value])
        result = np.asarray(vectors, dtype="float32")
        norms = np.linalg.norm(result, axis=1, keepdims=True)
        return result / np.maximum(norms, 1e-12)

    def similarity(self, query, texts):
        q = self.encode([query])[0]
        return [float(vector @ q) for vector in self.encode(texts)]

    def rerank(self, query, texts):
        return [1.0 if "不得" in text else 0.1 for text in texts]


def test_persistent_vector_index_records_model_and_corpus(tmp_path):
    pipeline = FakeBGE()
    items = [
        {"evidence_id": "e1", "content": "商业银行不得挪用客户资金"},
        {"evidence_id": "e2", "content": "一般业务办理流程"},
    ]
    index = PersistentVectorIndex(pipeline, tmp_path, "text")
    assert index.build_or_load(items, lambda item: item["content"])
    assert index.manifest["model"] == "fake-bge"
    assert index.search("不得挪用客户资金", 1)[0][0] == "e1"


def test_hybrid_index_exposes_bge_scores_and_reranker():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "监管办法", "file_name": "a.docx", "status": "effective"}],
        [
            {"evidence_id": "e1", "doc_id": "d1", "content": "商业银行不得挪用客户资金"},
            {"evidence_id": "e2", "doc_id": "d1", "content": "一般业务办理流程"},
        ],
        [],
        semantic=FakeBGE(),
        vector_dir=None,
    )
    hits = index.hybrid_search("客户资金不得挪用", "regulatory_fact", 2)
    assert hits[0].evidence_id == "e1"
    assert hits[0].rerank_score > 0


def test_hybrid_search_can_disable_dense_and_rerank_routes():
    index = HybridIndex(
        [{"doc_id": "d1", "title": "监管办法", "file_name": "a.docx", "status": "effective"}],
        [{"evidence_id": "e1", "doc_id": "d1", "content": "商业银行不得挪用客户资金"}],
        [],
        semantic=FakeBGE(),
    )

    hits = index.hybrid_search(
        "客户资金不得挪用",
        "regulatory_fact",
        2,
        rerank=False,
        dense=False,
    )

    assert hits
    assert all(hit.dense_score == 0 for hit in hits)
    assert all(hit.rerank_score == 0 for hit in hits)
    assert index.runtime_status["bge_vector_used"] is False
    assert index.runtime_status["bge_reranker_used"] is False


def test_reranker_uses_explicit_sigmoid_calibration():
    class Model:
        def __init__(self):
            self.activation = None

        def predict(self, pairs, **kwargs):
            self.activation = kwargs.get("activation_fn")
            assert self.activation is not None
            return np.asarray([0.5, 1 / (1 + np.exp(-2))], dtype="float32")

    from bankreg_trustrag.retrieval.bge import BGEPipeline

    pipeline = BGEPipeline(BGEConfig(mode="required"))
    model = Model()
    pipeline._reranker = model
    scores = pipeline.rerank("q", ["a", "b"])

    assert scores is not None
    assert scores[0] == pytest.approx(0.5, abs=1e-5)
    assert scores[1] == pytest.approx(1 / (1 + np.exp(-2)), abs=1e-5)

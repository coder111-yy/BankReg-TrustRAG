"""Opt-in tests against the real locally cached BGE models.

These tests are skipped in ordinary unit-test runs so CI does not require
multi-gigabyte model files. Set BANKREG_RUN_REAL_BGE=1 to execute them.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from bankreg_trustrag.config import Settings
from bankreg_trustrag.retrieval.bge import BGEConfig, BGEPipeline


def _real_bge_config() -> BGEConfig:
    settings = Settings.from_env(Path.cwd())
    return BGEConfig(
        mode="required",
        embedding_model=settings.bge_embedding_model,
        reranker_model=settings.bge_reranker_model,
        cache_dir=settings.bge_cache_dir,
        device=settings.bge_device,
        batch_size=2,
        max_length=settings.bge_max_length,
        local_files_only=True,
    )


@pytest.mark.skipif(
    os.getenv("BANKREG_RUN_REAL_BGE") != "1",
    reason="set BANKREG_RUN_REAL_BGE=1 to run local model integration tests",
)
def test_real_local_bge_embedding_and_reranker_load():
    pipeline = BGEPipeline(_real_bge_config())
    vectors = pipeline.encode(["商业银行监管制度", "银行统计报表"])
    scores = pipeline.rerank("商业银行监管制度", ["商业银行应遵守监管制度", "天气预报"])

    assert vectors is not None
    assert vectors.shape[0] == 2
    assert vectors.shape[1] > 0
    assert scores is not None
    assert len(scores) == 2
    assert all(0.0 <= score <= 1.0 for score in scores)
    assert pipeline.status["embedding_loaded"] is True
    assert pipeline.status["reranker_loaded"] is True


@pytest.mark.skipif(
    os.getenv("BANKREG_RUN_REAL_BGE") != "1",
    reason="set BANKREG_RUN_REAL_BGE=1 to run local model integration tests",
)
def test_real_local_bge_retrieval_reports_actual_routes(tmp_path):
    from bankreg_trustrag.retrieval.index import HybridIndex

    pipeline = BGEPipeline(_real_bge_config())
    index = HybridIndex(
        [{"doc_id": "d1", "title": "监管办法", "file_name": "rule.docx", "status": "effective"}],
        [
            {"evidence_id": "e1", "doc_id": "d1", "content": "商业银行应遵守监管制度"},
            {"evidence_id": "e2", "doc_id": "d1", "content": "与银行无关的天气说明"},
        ],
        [],
        semantic=pipeline,
        vector_dir=tmp_path,
    )
    hits = index.hybrid_search("银行监管制度", "regulatory_fact", top_k=1)

    assert hits
    assert index.runtime_status["bge_vector_used"] is True
    assert index.runtime_status["bge_reranker_used"] is True
    assert index.runtime_status["embedding_loaded"] is True
    assert index.runtime_status["reranker_loaded"] is True

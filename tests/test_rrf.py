from bankreg_trustrag.retrieval.index import Hit, HybridIndex


def _hit(evidence_id: str, **scores: float) -> Hit:
    item = {"evidence_id": evidence_id}
    return Hit("text", item, **scores)


def test_rrf_fuses_all_four_routes_on_one_rank_scale():
    hits = [
        _hit("lexical", lexical_score=10.0),
        _hit("dense", dense_score=0.9),
        _hit("metadata", metadata_score=1.0),
        _hit("table", table_score=1.0),
    ]

    HybridIndex._rrf(hits, k=1)

    scores = {hit.evidence_id: hit.fused_score for hit in hits}
    assert scores["table"] > scores["metadata"]
    assert scores["lexical"] > scores["metadata"]
    assert scores["lexical"] == scores["dense"]


def test_rrf_ignores_routes_with_no_score():
    hit = _hit("e1", lexical_score=2.0)

    HybridIndex._rrf([hit], k=60)

    assert hit.fused_score == 1.0 / 61.0

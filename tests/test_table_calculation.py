from types import SimpleNamespace

from bankreg_trustrag.retrieval.index import Hit
from bankreg_trustrag.reasoning import table_answer
from bankreg_trustrag.service import TrustRAGService
from bankreg_trustrag.verification import verify_claims


def _table_hit(evidence_id, column, value):
    return Hit(
        "table",
        {
            "evidence_id": evidence_id,
            "doc_id": "report",
            "source_title": "2023年12月全国各地区原保险保费收入情况表",
            "period": f"{column} / {value}",
            "indicator": "全国合计",
            "column_header": f"{column} / {value}",
            "value_text": str(value),
            "unit": "亿元",
            "cell_address": evidence_id.rsplit(":", 1)[-1],
            "context": f"全国合计 | {column} / {value} | {value}",
        },
        table_score=5.0,
    )


def test_table_change_question_calculates_difference_from_two_cells():
    hits = [
        _table_hit("cell:report:C4", "合计", 45167.98),
        _table_hit("cell:report:H4", "健康险", 51246.71),
    ]
    question = "根据《2023年12月全国各地区原保险保费收入情况表》，‘全国合计’从‘合计’到‘健康险’的数值变化为多少？"

    draft = table_answer(question, None, hits)
    verification = verify_claims(draft.answer, question, hits, draft.claims, draft.operations)

    assert draft.operations[0]["type"] == "table_calculation"
    assert draft.operations[0]["difference"] == 6078.73
    assert "6078.73" in draft.answer
    assert draft.operations[0]["operand_evidence_ids"] == ["cell:report:C4", "cell:report:H4"]
    assert verification.passed


def test_repeating_complete_question_does_not_use_old_turn_as_retrieval_query():
    question = "2023年12月全国各地区原保险保费收入情况表中全国合计合计口径的数值是多少？"
    hit = _table_hit("cell:report:C4", "合计", 45167.98)

    class RecordingIndex:
        model_status = {"mode": "disabled"}
        doc_by_id = {"report": {"title": "全国各地区原保险保费收入情况表", "file_name": "report.xlsx", "status": "effective"}}

        def __init__(self):
            self.queries = []

        def hybrid_search(self, query, *args, **kwargs):
            self.queries.append(query)
            return [hit]

    index = RecordingIndex()
    service = TrustRAGService.__new__(TrustRAGService)
    service.settings = SimpleNamespace(min_trust=0.58, top_k=8)
    service.store = SimpleNamespace(save_qa=lambda *args: None)
    service.index = index
    service.semantic = SimpleNamespace(enabled=False)

    first = service.ask(question)
    second = service.ask(question, conversation_context=[{"role": "user", "content": question}, {"role": "assistant", "content": first.answer}])

    assert first.answer == second.answer
    assert all("历史问题" not in query for query in index.queries)

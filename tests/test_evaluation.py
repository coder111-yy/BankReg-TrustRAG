from scripts.build_full_eval import BUSINESS_PROCESS_IDS, CLAUSE_THRESHOLD_IDS, REFUSAL_CASES, build_rows
from scripts.evaluate import evidence_set_matches, expected_cell, matches_expected_evidence, reciprocal_rank


def test_full_evaluation_adds_all_routes_and_refusal_cases():
    source = [
        {"id": next(iter(CLAUSE_THRESHOLD_IDS)), "qa_type": "regulatory_fact", "tags": []},
        {"id": next(iter(BUSINESS_PROCESS_IDS)), "qa_type": "cross_file_judgment", "tags": []},
    ]

    rows = build_rows(source)

    assert rows[0]["qa_type"] == "clause_threshold"
    assert rows[1]["qa_type"] == "business_process"
    assert {row["expected_behavior"] for row in rows[-len(REFUSAL_CASES):]} == {"refuse", "clarify"}


def test_evidence_matching_requires_expected_cell_when_available():
    row = {"evidence": "单元格：C5", "file_label": "report.xlsx", "source_title": "报告"}
    evidence = [{"cell_address": "D5", "source_file_name": "report.xlsx"}, {"cell_address": "C5", "source_file_name": "report.xlsx"}]

    assert expected_cell(row["evidence"]) == "C5"
    assert not matches_expected_evidence(row, evidence[0])
    assert matches_expected_evidence(row, evidence[1])
    assert reciprocal_rank(row, evidence) == 0.5


def test_evidence_set_matching_accepts_two_claims_from_two_paragraphs():
    row = {"evidence": "目录包括机构设立和机构变更；法人机构筹建审批属于机构设立类行政许可事项。"}
    evidence = [
        {"content": "目录包括机构设立和机构变更。"},
        {"content": "法人机构筹建审批属于机构设立类行政许可事项。"},
    ]

    assert evidence_set_matches(row, evidence)

---
name: bankreg-rag-evaluation
description: Evaluate BankReg-TrustRAG against the competition goals. Use when creating QA datasets, scoring regulatory facts or table extraction, measuring evidence citation and hallucination/refusal behavior, running ablations, analyzing failures, or producing reproducible evaluation reports.
---

# BankReg-TrustRAG Evaluation

## Mission

Build a reproducible evaluation harness that measures whether the system is accurate, traceable, numerically reliable, and willing to refuse when evidence is insufficient. Never fabricate evaluation results.

## Official target thresholds to treat as acceptance goals

- Regulatory fact question accuracy: **>= 85%**.
- Table value question accuracy: **>= 80%**.
- Evidence citation hit rate: **>= 90%**.
- Error rate for key numbers, dates, institution names, and document numbers: **<= 5%**.
- Refusal/clarification rate on out-of-knowledge-base or insufficient-evidence questions: **>= 80%**.
- System capability: ingest at least **200** regulation/attachment files; support Word, PDF, Excel; return evidence at clause, paragraph, and table-cell granularity.

Treat these as target gates, not as achieved results until measured.

## Required QA categories

Every evaluation set should label one of:

1. `regulatory_fact`
2. `clause_threshold`
3. `business_process`
4. `table_lookup`
5. `cross_file_judgment`

Include easy, medium, and hard items and retain per-category metrics.

## Evaluation dataset contract

Recommended JSONL fields:

```json
{
  "id": "qa_0001",
  "question": "...",
  "answer": "...",
  "evidence": ["..."],
  "source_title": "...",
  "source_url": "...",
  "local_path": "...",
  "difficulty": "medium",
  "qa_type": "clause_threshold",
  "tags": ["capital", "percentage"],
  "expected_behavior": "answer"
}
```

For refusal tests set `expected_behavior` to `refuse` or `clarify` and explain why evidence is intentionally insufficient.

## Core metrics

Measure at minimum:

### Answer quality
- Exact/normalized accuracy for facts and deterministic values.
- Per-category accuracy.
- Optional semantic correctness only when exact matching is inappropriate; define rubric explicitly.

### Retrieval
- Recall@K.
- MRR.
- NDCG@K when graded relevance is available.
- Evidence citation hit rate.
- Minimal-evidence coverage: whether all answer claims have supporting evidence.

### Table accuracy
- Indicator match.
- Period match.
- Unit match.
- Exact value match after normalized numeric parsing.
- Correct calculation output for derived questions.

### Trust/hallucination
- Key-fact error rate for number/date/institution/document number.
- Normative-strength error rate.
- Unsupported-claim rate.
- Correct refusal/clarification rate on insufficient evidence.
- False-refusal rate on answerable questions.

### System
- Mean latency.
- P50/P95 latency.
- Error/timeout rate.
- Cache-hit latency if cache is part of the implementation.

## Strict scoring principles

- Use deterministic scoring wherever possible.
- Normalize whitespace, punctuation, full-width/half-width forms, percentages, and dates before exact comparisons.
- Do not let an LLM judge numeric equality that can be checked programmatically.
- Separate retrieval failure from generation failure from verification failure.
- Save row-level results so every aggregate can be reproduced.
- Report confidence intervals or repeated-run variance for stochastic generation when practical.

## Required ablation study

Compare equivalent versions of the system:

- **A Baseline RAG**: dense/vector retrieval + LLM.
- **B Hybrid RAG**: BM25 + dense retrieval + LLM.
- **C Structure RAG**: hybrid + metadata + structured table retrieval + reranking.
- **D TrustRAG**: Structure RAG + multi-hop + claim verification + dynamic retry/clarify/refusal.

Do not cherry-pick different QA subsets between variants. Keep corpus, QA set, LLM settings, and randomness controls consistent.

## Failure taxonomy

Label each failed row with one primary cause and optional secondary causes:

- `parse_failure`
- `metadata_failure`
- `version_failure`
- `lexical_retrieval_failure`
- `dense_retrieval_failure`
- `table_retrieval_failure`
- `rerank_failure`
- `multihop_failure`
- `calculation_failure`
- `generation_unsupported_claim`
- `numeric_error`
- `entity_error`
- `citation_error`
- `normative_strength_error`
- `verification_miss`
- `incorrect_refusal`
- `should_have_refused`
- `timeout_or_system_error`

## Workflow

When asked to evaluate:

1. Freeze corpus/version and record a dataset hash.
2. Validate JSONL schema with `scripts/validate_eval_jsonl.py`.
3. Run a small smoke set first.
4. Run all required QA categories.
5. Save raw answer, evidence IDs, retrieval scores, verification flags, decision, and latency per row.
6. Compute aggregates by QA type and difficulty.
7. Compare against official target gates.
8. Run ablations when architecture changes affect retrieval/reasoning/trust.
9. Produce a failure-analysis table with representative cases.
10. State unambiguously which thresholds passed/failed; never replace missing runs with estimated numbers.

Read `references/evaluation-protocol.md` for experimental controls and `references/metrics.md` for formulas/normalization rules.

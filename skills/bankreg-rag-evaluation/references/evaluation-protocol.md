# Evaluation Protocol

## Dataset split

Maintain a stable held-out evaluation set. Do not tune prompts/retrieval weights directly on the final held-out set. If the dataset is small, keep a development subset for iteration and reserve a final subset for reporting.

## Coverage

The final set should contain all five QA categories and include:

- exact-document / exact-clause questions;
- thresholds, amounts, percentages, deadlines;
- exceptions and normative-strength wording;
- table indicator + period + unit questions;
- derived table calculations;
- current-version and historical-version questions;
- cross-file multi-hop questions;
- deliberately unanswerable/out-of-KB questions.

## Reproducibility record

For each run save:

- git commit hash;
- corpus hash / manifest version;
- eval dataset hash;
- embedding model/version;
- reranker model/version;
- LLM model/version or endpoint identifier;
- prompts/config version;
- top-k and fusion parameters;
- random seed / temperature;
- start/end timestamp;
- dependency lock or environment snapshot.

## Row-level result schema

Recommended fields:

```json
{
  "id": "qa_0001",
  "qa_type": "...",
  "expected_behavior": "answer",
  "expected_answer": "...",
  "actual_answer": "...",
  "expected_evidence": [],
  "retrieved_evidence": [],
  "cited_evidence": [],
  "retrieval_hit": true,
  "citation_hit": true,
  "answer_correct": true,
  "key_fact_error": false,
  "refusal_correct": null,
  "verification": {},
  "failure_labels": [],
  "latency_ms": 1234
}
```

## Comparison discipline

When comparing A/B/C/D:

- same corpus;
- same questions;
- same answer model and generation parameters where possible;
- same evaluation normalization;
- change only the intended architecture components;
- keep raw outputs for audit.

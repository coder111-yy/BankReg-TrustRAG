# Core Data Contracts

These are recommended contracts. Reuse equivalent existing models if present.

## Document metadata

```json
{
  "doc_id": "NFRA_001",
  "title": "...",
  "authority": "...",
  "document_no": "...",
  "publish_date": "YYYY-MM-DD",
  "effective_date": "YYYY-MM-DD",
  "expire_date": null,
  "document_type": "regulation",
  "topic": ["capital_management"],
  "version": "...",
  "status": "effective",
  "source_url": "...",
  "local_path": "...",
  "sha256": "..."
}
```

## Text evidence

```json
{
  "evidence_id": "text:NFRA_001:p16:a22",
  "doc_id": "NFRA_001",
  "page": 16,
  "chapter": "第三章",
  "article_no": "第二十二条",
  "paragraph_no": 1,
  "content": "...",
  "source_url": "..."
}
```

## Table evidence

```json
{
  "evidence_id": "cell:STAT_001:Sheet1:D8",
  "doc_id": "STAT_001",
  "sheet_name": "Sheet1",
  "table_name": "主要监管指标",
  "indicator": "不良贷款率",
  "period": "2025-03",
  "value": 1.21,
  "unit": "%",
  "row_header": "不良贷款率",
  "column_header": "2025年3月",
  "cell_address": "D8",
  "source_url": "..."
}
```

## Parsed query

```json
{
  "original_query": "...",
  "qa_type": "cross_file_judgment",
  "entities": {
    "authority": null,
    "document_title": null,
    "article_no": null,
    "indicator": "...",
    "period": "...",
    "institution_type": "..."
  },
  "requires_table": true,
  "requires_multi_hop": true,
  "rewritten_queries": []
}
```

## QA response

```json
{
  "answer": "...",
  "qa_type": "...",
  "evidence": [],
  "verification": {
    "numeric_ok": true,
    "date_ok": true,
    "entity_ok": true,
    "document_no_ok": true,
    "normative_strength_ok": true,
    "citation_ok": true,
    "version_ok": true
  },
  "trust": {
    "score": 0.94,
    "decision": "answer",
    "reasons": []
  },
  "latency_ms": 0,
  "trace_id": "..."
}
```

## Audit principle

Audit records should contain reproducible structured facts and tool outputs, not hidden chain-of-thought. Store route decisions, evidence IDs/scores, calculation inputs/outputs, verification flags, and refusal/retry reasons.

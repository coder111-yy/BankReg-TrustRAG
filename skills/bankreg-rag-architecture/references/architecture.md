# End-to-End Architecture Contract

## Primary flow

```text
User/UI
  -> Query Understanding + Router
  -> [BM25 | Dense Vector | Metadata Filter | Structured Table Retriever]
  -> Rank Fusion (RRF)
  -> Reranker
  -> Minimal Sufficient Evidence Selector
  -> [Clause Reasoning | Table Tool Reasoning | Cross-file Multi-hop Reasoning]
  -> Grounded Draft
  -> Claim-Level Verification
  -> Trust Decision
  -> Answer + Traceable Evidence OR Retry/Clarify/Refuse
```

## Five QA routes

### 1. Regulatory fact
Prefer text retrieval with semantic + lexical fusion. Return direct evidence and source metadata.

### 2. Clause / threshold
Favor lexical exact match, metadata filtering, clause-number-aware parsing, and current/historical version resolution. Verify every number/percentage and normative verb.

### 3. Business process
Retrieve all required ordered steps and exceptions. Avoid compressing away conditional branches.

### 4. Statistical table value
Route to structured table retrieval first. Return indicator, period, unit, value, sheet, row/column context, and cell address where feasible. Use deterministic calculation tools.

### 5. Cross-file scenario judgment
Decompose into explicit hops. Typical path:

```text
rule definition -> regulatory threshold -> indicator mapping -> period-specific table value -> deterministic comparison -> grounded conclusion
```

Persist the evidence ID for each hop.

## Minimal sufficient evidence

Evidence selection should optimize for:

- direct support of answer claims;
- current/temporally correct version;
- authoritative source;
- minimum redundancy;
- enough surrounding context to preserve exceptions, conditions, table headers, units, and periods.

## Version handling

For each document, preserve when available:

- `publish_date`
- `effective_date`
- `expire_date`
- `version`
- `status` (`effective`, `superseded`, `expired`, `unknown`)
- `supersedes_doc_id`

For “current” questions, prefer effective documents and surface ambiguity when effective status cannot be established. For historical questions, filter by the asked date.

## Trust-score principle

Do not let one similarity score decide whether the system should answer. Trust should combine retrieval relevance, evidence coverage, claim verification, source authority, and version validity. Keep component scores visible to developers for debugging.

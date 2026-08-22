---
name: bankreg-rag-architecture
description: Build, modify, or review the BankReg-TrustRAG project for bank regulatory rules and statistical reports. Use when work touches document ingestion, Word/PDF/Excel parsing, query routing, hybrid retrieval, RRF/reranking, table retrieval, multi-hop reasoning, evidence tracing, claim-level verification, confidence/refusal, APIs, schemas, or system architecture.
---

# BankReg-TrustRAG Architecture

## Mission

Implement a trustworthy RAG system for banking regulatory rules, policy documents, statistical reports, and business-process documents. The system must answer only from traceable evidence and must prefer deterministic parsing, lookup, comparison, and calculation over free-form LLM inference whenever possible.

## Non-negotiable product goals

The finished system should support:

1. Regulatory fact questions.
2. Clause / threshold questions.
3. Business-process questions.
4. Statistical table value questions.
5. Cross-file scenario judgment questions.

Every supported answer should preserve an evidence chain down to the smallest useful source unit: clause/paragraph for text and cell/row/header context for tables.

## Required architecture

Follow this logical flow unless the existing repository already implements an equivalent design:

1. **Ingestion and structure-aware parsing**
   - Word/PDF -> document metadata, heading hierarchy, chapter, article/clause number, paragraph, page/position.
   - Excel -> workbook, sheet, table region, row header, column header, indicator, period, value, unit, cell address.
   - Preserve `doc_id`, title, issuing authority, document number when available, publish/effective/expiry dates when available, file type, source URL, local path, version/status, and SHA-256/manifest metadata.
   - Do not flatten structured tables into anonymous text as the only representation.

2. **Knowledge/index layer**
   - Text clause/paragraph store.
   - Structured regulatory table store.
   - Document metadata/version store.
   - Vector index for semantic retrieval.
   - Optional relation graph for document revisions, citations, indicator-to-rule links, and report relationships.

3. **Query understanding and routing**
   - Classify into one of the five QA types above.
   - Extract entities such as document title, authority, business topic, clause number, indicator, period, threshold, institution type, and requested time validity.
   - Rewrite/decompose only when it improves retrieval; retain the original query for auditing.

4. **Four-route retrieval**
   - BM25/lexical retrieval for exact terms, clauses, document numbers, dates, percentages, and regulatory language.
   - Dense/vector semantic retrieval for paraphrases and related concepts.
   - Metadata filtering for authority, topic, document/version, date, file name, business domain, and validity status.
   - Structured table retrieval for indicators, periods, units, headers, and cells.
   - Fuse text candidates using RRF or an equivalent rank-fusion method.
   - Rerank high-value candidates with a reranker such as BGE Reranker when configured.
   - Select a *minimal sufficient evidence set* instead of passing excessive context to the LLM.

5. **Reasoning paths**
   - Clause reasoning: exact rule location + version validity.
   - Table reasoning: retrieve exact cells/rows; calculate with DuckDB/Pandas/Python or SQL, not mental arithmetic by the LLM.
   - Cross-file multi-hop reasoning: rule/definition -> metric/threshold -> report/table value -> deterministic comparison -> grounded explanation.

6. **Grounded generation**
   - Generate only from selected evidence.
   - Preserve normative strength exactly: e.g. “应当 / 可以 / 不得 / 原则上” must not be strengthened or weakened.
   - Never invent missing document numbers, percentages, dates, institutions, URLs, clauses, or table values.

7. **Claim-level verification**
   - Split the draft answer into atomic claims.
   - Verify at minimum: numbers, percentages, dates, institution/entity names, document numbers, clause references, normative-strength wording, and citations.
   - Detect evidence conflicts and version conflicts.
   - If verification fails, retry retrieval once with an explicit failure reason; otherwise clarify or refuse.

8. **Trust decision**
   - Compute a confidence/trust score from retrieval relevance, evidence sufficiency, verification status, source authority, and version validity.
   - High confidence -> answer with evidence.
   - Medium/ambiguous -> ask for clarification or retry retrieval.
   - Low/insufficient evidence -> refuse rather than guess.

## Data handling rule

Treat contest-provided “揭榜挂帅” data as non-shareable outside the permitted environment. Do **not** upload contest data or confidential/authorized files to third-party cloud parsers, public APIs, paste sites, external vector services, or telemetry systems unless the user has explicit permission from the organizer/data owner. Prefer local parsing for contest data. Cloud document skills may be used only on public/synthetic samples or when explicit permission is confirmed.

## Implementation guidance

Prefer these components when they fit the existing repo; do not rewrite working infrastructure merely to match this list:

- Backend/API: FastAPI.
- Workflow orchestration: LangGraph.
- Relational metadata: MySQL or PostgreSQL via SQLAlchemy.
- Dense retrieval: FAISS or Qdrant.
- Lexical retrieval: BM25 / Elasticsearch-compatible approach.
- Embedding: BGE-M3 or the model already configured by the project.
- Reranking: BGE Reranker or equivalent.
- Table operations: openpyxl + Pandas + DuckDB.
- Cache: Redis or database cache if already available.

## Suggested module boundaries

Use or adapt this layout:

```text
backend/
  api/
  agents/
  ingestion/
    word_parser.py
    pdf_parser.py
    excel_parser.py
    manifest.py
  retrieval/
    bm25.py
    vector.py
    metadata.py
    table.py
    fusion.py
    reranker.py
    evidence_selector.py
  reasoning/
    clause_reasoner.py
    table_reasoner.py
    multihop_reasoner.py
  verification/
    claim_splitter.py
    numeric_check.py
    entity_check.py
    citation_check.py
    version_check.py
    trust_score.py
  models/
  services/
  db/
  schemas/
tests/
scripts/
```

Read `references/architecture.md` for the end-to-end contract, `references/data-contracts.md` for core schemas, and `references/implementation-roadmap.md` when planning implementation order.

## API contract expectations

At minimum, design equivalents of:

- `POST /api/qa` — ask a question and return answer + evidence + trust/verification information.
- `POST /api/documents/ingest` — ingest permitted files.
- `GET /api/documents/{doc_id}` — metadata/version details.
- `GET /api/evidence/{evidence_id}` — original evidence location/details.
- `POST /api/evaluate` or offline evaluation command — run evaluation dataset.
- `GET /api/history` — optional question history/audit view.

Do not expose chain-of-thought. Store and expose structured audit fields instead: route, retrieved evidence IDs, scores, deterministic operations, verification flags, retry/refusal reason, and latency.

## Development workflow

When asked to build a feature:

1. Inspect the existing repo and avoid duplicating already-implemented components.
2. Identify which architecture layer the change belongs to.
3. Define input/output contracts first.
4. Add or update tests before/with implementation.
5. Implement the smallest vertical slice that can be demonstrated end-to-end.
6. Run relevant tests and report actual results; never claim unrun tests passed.
7. Record known limitations and follow-up work.

## Definition of done

A feature is done only when:

- It has deterministic tests for critical parsing/retrieval/calculation behavior.
- It preserves source metadata needed for traceability.
- Failure cases do not silently fall back to hallucinated answers.
- Numeric/table operations are reproducible.
- It does not bypass data-handling restrictions.
- It integrates with the existing end-to-end QA path or has a clear integration test.

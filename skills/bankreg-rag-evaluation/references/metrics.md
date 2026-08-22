# Metrics and Normalization

## Retrieval Recall@K

A row is a hit when at least one required evidence item is contained in the top-K retrieved evidence set. For multi-hop questions, additionally report *all-hop recall*: every required hop must be found.

## MRR

Use reciprocal rank of the first required relevant evidence item. Report separately for text and table questions when useful.

## NDCG@K

Use only when relevance grades are defined consistently. Do not invent relevance grades after seeing system output.

## Evidence citation hit rate

Count an answer as a citation hit only when its returned/cited evidence contains the expected source evidence at the required granularity or an explicitly accepted equivalent.

## Table normalization

Normalize before comparison:

- `%`, `百分比`, percentage numeric form;
- commas/thousands separators;
- Chinese/Arabic date representations;
- whitespace and full-width punctuation;
- unit scaling (e.g. 元/万元) only when the expected unit conversion rule is explicit.

Record the normalized operands used for every derived calculation.

## Key-fact error rate

Track key facts individually. A wrong number, date, institution, or document number is an error even if the surrounding prose is semantically similar.

## Refusal metrics

Report both:

- correct refusal/clarification rate on intentionally unanswerable rows;
- false-refusal rate on answerable rows.

A system can meet the first metric by refusing too often, so both are necessary for meaningful analysis.

## Latency

Use end-to-end request latency for user-visible QA. Also keep stage timings (routing, retrieval, rerank, generation, verification) when available so bottlenecks are diagnosable.

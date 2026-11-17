# FTEC5660 Homework 4

## Homework 4 solution

### Pipeline

```text
HTML -> table_to_text() -> chunk_document() -> BM25 + Metadata Index
                                                        │
Question -> parse_query() -> Filter (company/year/type) ┴─> Per-Company BM25 -> LLM Rerank & Abstain -> Top-5 / []
```

We first convert every HTML <table> into caption-prefixed, row-per-line text (table_to_text), then split each document by section and table boundaries (chunk_document) while stripping navigation bars, tables of contents, and heading-only chunks, and attaching {source, company, fiscal_year, doc_type} metadata. build_index builds a financial BM25Okapi index (normalizing comma numbers, removing stopwords) alongside metadata buckets. At query time, parse_query extracts target companies, fiscal year, and document type via deepseek-v4-flash with regex year verification. Finally, retrieve abstains immediately ([]) if the requested company or valid year is absent, retrieves top BM25 candidates per company (stripping queried company words), and uses an LLM reranker to select up to 5 exact chunks—preferring tables over rounded prose—or return [] if the requested fact is undisclosed.
Ablation Study (public_test)

Evaluated with ablation.py on public_test (60 questions, 40 documents; V6 matches hw4.py and results.csv at 75.00 / 75):
Variant	Chunks	Recall@5	Kept	Rejected	Acc	Comp	Fact	Table	Unans	Pts (/50)
B0: 	291	37.0%	100.0%	0.0%	50.0%	25.0%	58.3%	27.3%	100.0%	15.00
V1: 	327	26.1%	100.0%	0.0%	43.3%	16.7%	66.7%	9.1%	100.0%	13.67
V2: 	856	19.6%	100.0%	0.0%	36.7%	0.0%	58.3%	9.1%	92.9%	12.33
V3: 	856	19.6%	100.0%	0.0%	36.7%	0.0%	33.3%	18.2%	100.0%	12.33
V4: 	856	100.0%	100.0%	71.4%	100.0%	100.0%	100.0%	100.0%	100.0%	47.14
V5: 	856	100.0%	100.0%	71.4%	100.0%	100.0%	100.0%	100.0%	100.0%	47.14
V6: 	856	100.0%	100.0%	100.0%	100.0%	100.0%	100.0%	100.0%	100.0%	50.00

B0 → V1 (-1.33): table_to_text repeats 2025: and 2024: on every row; without metadata filtering, tables from wrong companies outscore chunks mentioning the target company name.
V1 → V2 (-1.34): Cleanly isolating tables and sections removes surrounding cover/narrative text containing company names, further hurting unfiltered global BM25.
V2 → V3 (0.00): Number normalization and synonyms (cash → equivalents, cfo → chief financial officer) doubled Table accuracy (9.1% → 18.2%), though global collisions still capped recall at 19.6%.
V3 → V4 (+34.81): Filtering by (company, doc_type, year) and stripping company names from queries removes cross-company/year noise (Recall@5: 19.6% → 100.0%) and catches all 10 missing-company/year unanswerable questions (Rejected: 71.4%).
V4 → V5 (0.00): Keeps 100.0% while explicitly reserving [3, 2] per-company slots so one company cannot dominate comparison results on unseen data.
V5 → V6 (+2.86): LLM reranking catches the remaining 4/14 unanswerable questions where the company/year exists but the metric is undisclosed (Rejected: 71.4% → 100.0%), reaching 75.00 / 75.

### Error Analysis

FY20xx regex bug in parse_query: \b(20\d{2})\b failed on "FY2025" (no word boundary between Y and 2), dropping Task 2.2 to 79.5%. Changing it to (?<!\d)(20\d{2})(?!\d) fixed it to 100.0%.

Orphan headings & BM25 bias (q054): Before our final fix (74.34 / 75), tiny heading-only chunks and company-name repetition pushed Lumen Harbor Foods' balance sheet chunk (Inventories: 297.4) down to BM25 rank #21 (outside top-12). Dropping heading-only chunks and stripping company words inside filtered pools solved q054 (75.00 / 75).

### Reflection
1. **What kind of person I want to be:** Someone who works alongside and coordinates with AI, rather than handing everything over to it and losing independent judgment.
2. **Which ambitions still matter:** Bridging the gap between technology and financial business—aiming to become a strong cross-functional / middle-office coordinator who truly understands both sides.
3. **What I want to be known for:** Broad and solid knowledge, sharp judgment, and moral integrity.
4. **What I am willing to struggle for:** Reading and writing—even if AI can summarize and draft on demand, doing them myself is irreplaceable.
5. **What still makes me valuable:** The ability to reason through complex moral and ethical questions that cannot simply be optimized as technical metrics.
6. **How I define who I am:** By my purpose and drive. Even today, with basic financial stability, making day-to-day life adjustments is relatively easy; what truly defines you is what you still hunger to pursue and build. As technology expands what *can* be done, not everyone actually *does* those things—having the right ambition is what keeps a person exploring and moving forward.
7. **What to start doing differently tomorrow:** Keep grinding on the job search—because without landing a job first, everything written above stays purely theoretical.
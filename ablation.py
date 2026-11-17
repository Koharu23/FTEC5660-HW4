"""Task 5: Ablation study on public_test (B0 -> V1 -> V2 -> V3 -> V4 -> V5 -> V6)."""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

import hw4


def eval_variant(
    name: str,
    corpus: dict[str, str],
    questions: dict[str, str],
    ground_truth: dict[str, Any],
    chunks: list[Document],
    retriever_fn,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    retrieved = {qid: retriever_fn(q) for qid, q in questions.items()}

    def run_gen(qid: str) -> str:
        return hw4.generate(questions[qid], retrieved[qid])

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(run_gen, questions)))

    answerable = [qid for qid, g in ground_truth.items() if g["evidence"]]
    unanswerable = [qid for qid, g in ground_truth.items() if not g["evidence"]]

    non_empty = sum(bool(retrieved[qid]) for qid in answerable) / len(answerable)
    recall = sum(hw4.retrieval_hit(retrieved[qid], ground_truth[qid]) for qid in answerable) / len(answerable)
    kept = non_empty
    rejected = sum(not retrieved[qid] for qid in unanswerable) / len(unanswerable)
    acc = sum(hw4.answer_correct(answers[qid], ground_truth[qid]) for qid in ground_truth) / len(ground_truth)

    by_cat: dict[str, float] = {}
    for cat in ("comparison", "fact", "table", "unanswerable"):
        qids = [qid for qid, g in ground_truth.items() if g["category"] == cat]
        by_cat[cat] = sum(hw4.answer_correct(answers[qid], ground_truth[qid]) for qid in qids) / len(qids)

    pts_21 = hw4.POINTS["2.1"] * non_empty
    pts_23 = hw4.POINTS["2.3"] * hw4.clip((recall - hw4.RECALL_ZERO) / (hw4.RECALL_FULL - hw4.RECALL_ZERO))
    pts_3 = hw4.POINTS["3"] * hw4.clip(kept + rejected - 1)
    pts_4 = hw4.POINTS["4"] * acc
    elapsed = time.perf_counter() - t0

    return {
        "name": name,
        "chunks": len(chunks),
        "recall": recall,
        "kept": kept,
        "rejected": rejected,
        "acc": acc,
        "comparison": by_cat["comparison"],
        "fact": by_cat["fact"],
        "table": by_cat["table"],
        "unanswerable": by_cat["unanswerable"],
        "retrieval_gen_pts": pts_21 + pts_23 + pts_3 + pts_4,
        "time_s": elapsed,
    }


def make_naive_chunks(corpus: dict[str, str], use_table_to_text: bool) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks: list[Document] = []
    for fname in sorted(corpus):
        html = corpus[fname]
        soup = BeautifulSoup(html, "html.parser")
        meta = hw4._extract_doc_meta(fname, soup)
        if use_table_to_text:
            for tbl in soup.find_all("table"):
                t_txt = hw4.table_to_text(str(tbl))
                pre = soup.new_tag("pre")
                pre.string = "\n" + t_txt + "\n"
                tbl.replace_with(pre)
        text = soup.get_text("\n", strip=True)
        for piece in splitter.split_text(text):
            chunks.append(Document(page_content=piece[: hw4.MAX_CHUNK_CHARS], metadata=dict(meta)))
    return chunks


def build_ws_bm25(chunks: list[Document]) -> BM25Okapi:
    return BM25Okapi([c.page_content.lower().split() for c in chunks])


def main() -> None:
    folder = Path("public_test")
    corpus = hw4.load_corpus(folder / "corpus")
    questions = hw4.load_json(folder / "questions.json")
    ground_truth = hw4.load_json(folder / "ground_truth.json")

    # 预先缓存 parse_query 结果，供 V4/V5/V6 共用
    for q in questions.values():
        hw4.parse_query(q)

    results = []

    # B0: Naive RAG
    c_b0 = make_naive_chunks(corpus, use_table_to_text=False)
    bm25_b0 = build_ws_bm25(c_b0)
    results.append(eval_variant(
        "B0: Naive RAG (get_text + RecSplitter + WS BM25)",
        corpus, questions, ground_truth, c_b0,
        lambda q: [c_b0[i] for i in sorted(range(len(c_b0)), key=lambda i: bm25_b0.get_scores(q.lower().split())[i], reverse=True)[:5]],
    ))

    # V1: + table_to_text
    c_v1 = make_naive_chunks(corpus, use_table_to_text=True)
    bm25_v1 = build_ws_bm25(c_v1)
    results.append(eval_variant(
        "V1: + table_to_text",
        corpus, questions, ground_truth, c_v1,
        lambda q: [c_v1[i] for i in sorted(range(len(c_v1)), key=lambda i: bm25_v1.get_scores(q.lower().split())[i], reverse=True)[:5]],
    ))

    # V2: + section-aware chunking & headers (hw4.chunk_document)
    c_v2 = [c for f in sorted(corpus) for c in hw4.chunk_document(f, corpus[f])]
    bm25_v2 = build_ws_bm25(c_v2)
    results.append(eval_variant(
        "V2: + section-aware chunking & headers",
        corpus, questions, ground_truth, c_v2,
        lambda q: [c_v2[i] for i in sorted(range(len(c_v2)), key=lambda i: bm25_v2.get_scores(q.lower().split())[i], reverse=True)[:5]],
    ))

    # V3: + financial tokenizer & query synonyms
    idx_v3 = hw4.build_index(c_v2)
    results.append(eval_variant(
        "V3: + financial tokenizer & synonyms",
        corpus, questions, ground_truth, c_v2,
        lambda q: [c_v2[i] for i in hw4._bm25_rank_subset(list(range(len(c_v2))), idx_v3, hw4._bm25_tokenize(q, expand=True), 5)],
    ))

    # V4: + metadata filtering & company/year abstention (pooled top-5)
    def ret_v4(q: str) -> list[Document]:
        parsed = hw4.parse_query(q)
        companies = parsed.get("companies") or []
        req_year = parsed.get("fiscal_year")
        doc_type = parsed.get("doc_type") or "annual_report"
        comp_keys = []
        for comp in companies:
            ck = hw4.company_key(comp)
            if ck not in idx_v3["by_company"]:
                return []
            if ck not in comp_keys:
                comp_keys.append(ck)
        qt = hw4._bm25_tokenize(q, expand=True)
        cw = {w for ck in comp_keys for w in ck.split()}
        fqt = [t for t in qt if t not in cw] or qt
        if comp_keys:
            pool: list[int] = []
            for ck in comp_keys:
                avail = idx_v3["years_by_comp_type"].get((ck, doc_type), set())
                tys = hw4._select_target_years(avail, req_year, doc_type)
                if not tys:
                    return []
                for ty in tys:
                    pool.extend(idx_v3["by_key"].get((ck, doc_type, ty), []))
            if not pool:
                return []
            return [c_v2[i] for i in hw4._bm25_rank_subset(pool, idx_v3, fqt, 5)]
        return [c_v2[i] for i in hw4._bm25_rank_subset(list(range(len(c_v2))), idx_v3, qt, 5)]

    results.append(eval_variant(
        "V4: + metadata filter & company/year abstention",
        corpus, questions, ground_truth, c_v2, ret_v4,
    ))

    # V5: + per-company retrieval for comparisons
    def ret_v5(q: str) -> list[Document]:
        parsed = hw4.parse_query(q)
        companies = parsed.get("companies") or []
        req_year = parsed.get("fiscal_year")
        doc_type = parsed.get("doc_type") or "annual_report"
        comp_keys = []
        for comp in companies:
            ck = hw4.company_key(comp)
            if ck not in idx_v3["by_company"]:
                return []
            if ck not in comp_keys:
                comp_keys.append(ck)
        qt = hw4._bm25_tokenize(q, expand=True)
        cw = {w for ck in comp_keys for w in ck.split()}
        fqt = [t for t in qt if t not in cw] or qt
        if comp_keys:
            picked: list[int] = []
            budgets = [5] if len(comp_keys) == 1 else ([3, 2] if len(comp_keys) == 2 else [2] * len(comp_keys))
            for ck, k in zip(comp_keys, budgets):
                avail = idx_v3["years_by_comp_type"].get((ck, doc_type), set())
                tys = hw4._select_target_years(avail, req_year, doc_type)
                if not tys:
                    return []
                pool: list[int] = []
                for ty in tys:
                    pool.extend(idx_v3["by_key"].get((ck, doc_type, ty), []))
                if not pool:
                    return []
                picked.extend(hw4._bm25_rank_subset(pool, idx_v3, fqt, k))
            return [c_v2[i] for i in picked[:5]]
        return [c_v2[i] for i in hw4._bm25_rank_subset(list(range(len(c_v2))), idx_v3, qt, 5)]

    results.append(eval_variant(
        "V5: + per-company retrieval for comparisons",
        corpus, questions, ground_truth, c_v2, ret_v5,
    ))

    # V6: + LLM reranking & undisclosed-fact abstention (= full hw4.py)
    results.append(eval_variant(
        "V6: + LLM rerank & undisclosed abstention (hw4.py)",
        corpus, questions, ground_truth, c_v2,
        lambda q: hw4.retrieve(idx_v3, q),
    ))

    print("\n=== ABLATION SUMMARY ===")
    header = "| Variant | Chunks | Recall@5 | Kept | Rejected | Acc | Comp | Fact | Table | Unans | Pts(2.1+2.3+3+4 /50) |"
    sep = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    print(header)
    print(sep)
    for r in results:
        print(
            f"| {r['name']} | {r['chunks']} | {r['recall']:.1%} | {r['kept']:.1%} | {r['rejected']:.1%} | "
            f"{r['acc']:.1%} | {r['comparison']:.1%} | {r['fact']:.1%} | {r['table']:.1%} | "
            f"{r['unanswerable']:.1%} | {r['retrieval_gen_pts']:.2f} |"
        )
    print("\nUsage:", hw4.USAGE.summary(), "| Generator:", hw4.GENERATOR_USAGE.summary())


if __name__ == "__main__":
    main()
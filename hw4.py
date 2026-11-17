"""FTEC5660 Homework 4: the retrieval half of a RAG assistant over company reports.

Fill in the five functions marked ### YOUR CODE HERE. You may add imports, constants and helper
functions above them, but do not change the provided code below them (the models, the generator,
the loaders, the runner and the scoring).

    python3 hw4.py --data-folder public_test
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import threading
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.documents import Document
from langchain_deepseek import ChatDeepSeek

load_dotenv()


# =============================================================================
# Your code: add imports, constants and helper functions here.
from bs4 import BeautifulSoup

def _clean_cell_text(tag) -> str:
    return re.sub(r"\s+", " ", tag.get_text(" ", strip=True)).strip()
def _extract_doc_meta(file_name: str, soup: BeautifulSoup) -> dict[str, Any]:
    title_text = _clean_cell_text(soup.title) if soup.title else ""
    h1 = soup.find("h1")
    h1_text = _clean_cell_text(h1) if h1 else ""
    h2 = soup.find("h2")
    h2_text = _clean_cell_text(h2) if h2 else ""
    broker_div = soup.find("div", class_="broker")
    top_text = " ".join([title_text, h1_text, h2_text, _clean_cell_text(broker_div) if broker_div else ""])

    if broker_div or "equity research" in top_text.lower() or "results review" in top_text.lower():
        doc_type = "research_report"
    elif "earnings conference call" in top_text.lower() or "earnings call" in top_text.lower():
        doc_type = "earnings_call"
    else:
        doc_type = "annual_report"

    if doc_type in ("earnings_call", "research_report"):
        company = re.sub(r"\s*\([^)]*\)\s*$", "", h1_text).strip()
    else:
        company = h1_text.strip()

    fiscal_year: int | None = None
    if doc_type == "annual_report":
        m = re.search(r"Annual Report\s+(\d{4})", top_text, re.I)
        if not m:
            body_head = _clean_cell_text(soup.body)[:800] if soup.body else ""
            m = re.search(r"(?:Annual Report|31 December)\s+(\d{4})", body_head, re.I)
        if m:
            fiscal_year = int(m.group(1))
    elif doc_type == "earnings_call":
        m = re.search(r"Full Year\s+(\d{4})", top_text, re.I)
        fiscal_year = int(m.group(1)) if m else 2025
    else:
        m = re.search(r"FY\s*(\d{4})\s+results review", top_text, re.I)
        fiscal_year = int(m.group(1)) if m else 2025

    return {
        "source": file_name,
        "company": company,
        "fiscal_year": fiscal_year if fiscal_year is not None else 2025,
        "doc_type": doc_type,
    }


def _iter_blocks(node):
    for child in node.children:
        if not getattr(child, "name", None):
            txt = re.sub(r"\s+", " ", str(child)).strip()
            if txt:
                yield ("raw", txt)
            continue
        name = child.name.lower()
        if name in ("nav", "style", "script", "head"):
            continue
        if name == "table":
            yield ("table", child)
        elif name in ("h1", "h2", "h3", "h4"):
            yield ("heading", child)
        elif name in ("ul", "ol"):
            yield ("list", child)
        elif name == "p":
            yield ("text", child)
        elif name == "div":
            if child.find(["h1", "h2", "h3", "h4", "p", "ul", "ol", "table", "div"]) is not None:
                yield from _iter_blocks(child)
            else:
                yield ("text", child)
        else:
            yield from _iter_blocks(child)


def _split_long_text(text: str, limit: int) -> list[str]:
    if len(text) <= limit:
        return [text]
    parts: list[str] = []
    # 优先按换行切分，再按句子/空格切分，保证每块不超过 limit
    lines = text.splitlines()
    cur = ""
    for line in lines:
        if len(line) > limit:
            if cur:
                parts.append(cur)
                cur = ""
            words = line.split(" ")
            wcur = ""
            for w in words:
                if len(w) > limit:
                    if wcur:
                        parts.append(wcur)
                        wcur = ""
                    for i in range(0, len(w), limit):
                        parts.append(w[i : i + limit])
                else:
                    cand = f"{wcur} {w}".strip() if wcur else w
                    if len(cand) <= limit:
                        wcur = cand
                    else:
                        parts.append(wcur)
                        wcur = w
            if wcur:
                parts.append(wcur)
        else:
            cand = f"{cur}\n{line}" if cur else line
            if len(cand) <= limit:
                cur = cand
            else:
                parts.append(cur)
                cur = line
    if cur:
        parts.append(cur)
    return parts

_PARSE_CACHE: dict[str, dict[str, Any]] = {}
_PARSE_LLM = None

_PARSE_PROMPT = """Extract structured metadata from the financial question below. Return ONLY valid JSON with three keys: "companies", "fiscal_year", "doc_type".

Rules:
1. "companies": a list of strings containing EVERY target company the question asks about, as named in the question.
   - Include companies even if they might not exist in a database (e.g., if comparing two companies, include both).
   - Do NOT include brokerage firms / equity research firms (e.g., BuyFin Capital, Meridian Equity Research) or audit firms.
2. "fiscal_year": an integer (e.g., 2024) for the fiscal year explicitly named in the question ("FY2024" -> 2024, "31 December 2024" -> 2024), or null if the question does not name any year.
   - Use the year named in the question itself (e.g., "FY2023 revenue" -> 2023; "FY2026 revenue forecast" -> 2026).
3. "doc_type": one of "annual_report", "earnings_call", "research_report":
   - "earnings_call" if the question asks about an earnings call / conference call / Q&A / management call remarks or guidance mentioned on a call.
   - "research_report" if the question asks about a broker's rating, 12-month target price, share price target, forecast/estimate, peer valuation, or results review.
   - "annual_report" otherwise.

Examples:
Question: "What FY2026 revenue does Meridian Equity Research forecast for Sablewood Retail, in USD millions?"
JSON: {"companies": ["Sablewood Retail"], "fiscal_year": 2026, "doc_type": "research_report"}

Question: "Which company held more cash and cash equivalents at 31 December 2024: Northgate Robotics or Northgate Payments?"
JSON: {"companies": ["Northgate Robotics", "Northgate Payments"], "fiscal_year": 2024, "doc_type": "annual_report"}

Question: "In which city is Halcyon Freight headquartered?"
JSON: {"companies": ["Halcyon Freight"], "fiscal_year": null, "doc_type": "annual_report"}

Question: "What 12-month target price does BuyFin Capital set for Kestrel Semiconductor's shares in its FY2025 results review, in USD?"
JSON: {"companies": ["Kestrel Semiconductor"], "fiscal_year": 2025, "doc_type": "research_report"}

Question: {question}
JSON:"""

from rank_bm25 import BM25Okapi

_STOP_WORDS = {
    "what", "was", "were", "which", "who", "whom", "whose", "where", "when", "how",
    "did", "does", "do", "is", "are", "the", "a", "an", "of", "in", "at", "for",
    "to", "and", "or", "by", "from", "with", "on", "as", "its", "their", "his",
    "her", "that", "this", "it", "be", "been", "had", "has", "have", "held",
    "more", "higher", "most", "much", "many", "usd", "millions", "million",
    "percent", "company", "end", "fiscal", "year", "fy", "december", "31",
}

_SYNONYMS = {
    "cash": ["equivalents"],
    "cfo": ["chief", "financial", "officer"],
    "ceo": ["chief", "executive", "officer"],
    "auditor": ["independent", "audit", "firm"],
    "audited": ["independent", "auditor", "audit"],
    "headquartered": ["headquarters"],
    "headcount": ["employees"],
    "employees": ["headcount", "function"],
    "repurchase": ["share", "repurchase", "programme", "buyback"],
    "buyback": ["share", "repurchase", "programme"],
    "guidance": ["outlook", "guided", "guide"],
    "guide": ["guidance", "outlook"],
    "debt": ["borrowings"],
    "borrowings": ["debt", "short-term", "long-term"],
    "receivable": ["accounts", "receivable", "trade"],
    "receivables": ["accounts", "receivable", "trade"],
    "target": ["12-month", "target", "price"],
    "rating": ["rating", "buy", "hold", "reduce"],
}

_RERANK_LLM = None

_RERANK_PROMPT = """You are a financial RAG reranker and answerability checker.
Given a user question and numbered candidate passages (each tagged with its company, fiscal_year, and doc_type), do two things:
1. Check if the candidate passages ACTUALLY contain the information needed to answer the question for the specified company/companies and fiscal year.
   - If the question asks for a specific metric, event, or fact that is NOT disclosed or not present in ANY candidate passage (for example: lobbying expenses, carbon offset purchases, customer churn rate, revenue from largest customer when not mentioned), set "answerable" to false.
   - If the question compares two companies, BOTH companies must have their relevant information present in the candidates; otherwise set "answerable" to false.
2. If answerable is true, select the 1 to 5 BEST passage indices (most useful first) needed to answer the question accurately.
   - Prefer exact figures in financial tables over rounded figures in letters or earnings calls.
   - For comparisons between two companies, include the best passage(s) for EACH company.

Return ONLY valid JSON: {"answerable": true/false, "selected": [indices]}

Question: {question}

Candidates:
{candidates}

JSON:"""


def _bm25_tokenize(text: str, expand: bool = False) -> list[str]:
    norm = unicodedata.normalize("NFKC", text).lower()
    # 去掉数字中的千分位逗号，保留小数点
    norm = re.sub(r"(?<=\d),(?=\d)", "", norm)
    tokens = re.findall(r"[a-z0-9]+(?:\.[0-9]+)?", norm)
    kept = [t for t in tokens if t not in _STOP_WORDS and len(t) > 1 or t.isdigit()]
    if expand:
        extra: list[str] = []
        for t in kept:
            extra.extend(_SYNONYMS.get(t, []))
        kept.extend(extra)
    return kept


def _select_target_years(avail_years: set[int], req_year: int | None, doc_type: str) -> list[int]:
    if not avail_years:
        return []
    if req_year is None:
        return sorted(avail_years, reverse=True)
    if doc_type == "annual_report":
        # 某年的年报同时包含当年与上一年（例如 2024 年报有 2024 和 2023，2025 年报有 2025 和 2024）
        if req_year in avail_years:
            # 优先当年年报；若有次年年报也可作为补充（但当年年报放前面更稳，避免人事变更混淆）
            return [req_year]
        if (req_year + 1) in avail_years:
            return [req_year + 1]
        return []
    elif doc_type == "research_report":
        # 研报通常包含 FY2024A, FY2025A, FY2026E, FY2027E
        max_y = max(avail_years)
        if max_y - 1 <= req_year <= max_y + 2:
            return sorted(avail_years, reverse=True)
        return []
    else:  # earnings_call
        # Q4 FY2025 电话会包含 FY2025 业绩和 FY2026 指引
        max_y = max(avail_years)
        if max_y <= req_year <= max_y + 1:
            return sorted(avail_years, reverse=True)
        return []


def _bm25_rank_subset(subset_indices: list[int], index: dict[str, Any], query_tokens: list[str], top_k: int) -> list[int]:
    if not subset_indices or not query_tokens:
        return subset_indices[:top_k]
    bm25: BM25Okapi = index["bm25"]
    all_scores = bm25.get_scores(query_tokens)
    ranked = sorted(subset_indices, key=lambda idx: all_scores[idx], reverse=True)
    return ranked[:top_k]
# =============================================================================


def table_to_text(table_html: str) -> str:
    """Task 1.1: turn one HTML <table> into text that keeps every value with its row and its column.

    Put each value on a line (at most 400 characters) together with its row label and its column
    header, e.g. "Revenue | 2025: 2,710.4 | 2024: 2,629.1", one table row per line, keep the text of
    every cell unchanged, and include the caption (it holds the unit) in the output. Do not call the
    LLM here.
    """
    ### YOUR CODE HERE
    soup = BeautifulSoup(table_html, "html.parser")
    table = soup.find("table") or soup

    lines: list[str] = []
    cap = table.find("caption")
    if cap:
        cap_text = _clean_cell_text(cap)
        if cap_text:
            lines.append(cap_text)

    # 1. 提取表头（支持 thead 或首行 th，支持 colspan）
    headers: list[str] = []
    thead = table.find("thead")
    header_trs = thead.find_all("tr") if thead else []
    body_trs = []

    if header_trs:
        # 若有多行表头，取最后一行或拼接；通常这里只有一行
        for cell in header_trs[-1].find_all(["th", "td"]):
            text = _clean_cell_text(cell)
            colspan = int(cell.get("colspan", 1) or 1)
            headers.extend([text] * colspan)
        tbody = table.find("tbody")
        body_trs = tbody.find_all("tr") if tbody else [
            tr for tr in table.find_all("tr") if tr not in header_trs
        ]
    else:
        all_trs = table.find_all("tr")
        if all_trs and all_trs[0].find("th"):
            for cell in all_trs[0].find_all(["th", "td"]):
                text = _clean_cell_text(cell)
                colspan = int(cell.get("colspan", 1) or 1)
                headers.extend([text] * colspan)
            body_trs = all_trs[1:]
        else:
            body_trs = all_trs

    # 2. 逐行提取数据
    for tr in body_trs:
        raw_cells = tr.find_all(["td", "th"])
        if not raw_cells:
            continue
        cells: list[str] = []
        for cell in raw_cells:
            text = _clean_cell_text(cell)
            colspan = int(cell.get("colspan", 1) or 1)
            cells.extend([text] * colspan)

        if not any(cells):
            continue

        # 只有一列的分组小标题行（例如资产负债表里的 "Current assets"）
        non_empty = [c for c in cells if c]
        if len(raw_cells) == 1 or (len(non_empty) == 1 and not cells[1:]):
            lines.append(non_empty[0])
            continue

        row_label = cells[0]
        pairs: list[str] = []
        for i in range(1, len(cells)):
            val = cells[i]
            if not val:
                continue
            col = headers[i] if i < len(headers) and headers[i] else f"col_{i}"
            pairs.append(f"{col}: {val}")

        if not pairs:
            if row_label:
                lines.append(row_label)
            continue

        full_line = f"{row_label} | " + " | ".join(pairs) if row_label else " | ".join(pairs)
        if len(full_line) <= MAX_LINE_CHARS:
            lines.append(full_line)
        else:
            # 防止极宽表格超过 400 字符被评分器丢弃，按 400 字符内拆分但每行保留 row_label
            cur = row_label
            for p in pairs:
                candidate = f"{cur} | {p}" if cur else p
                if len(candidate) <= MAX_LINE_CHARS:
                    cur = candidate
                else:
                    if cur:
                        lines.append(cur)
                    cur = f"{row_label} | {p}" if row_label else p
            if cur:
                lines.append(cur)

    return "\n".join(lines)


def chunk_document(file_name: str, html: str) -> list[Document]:
    """Task 1.2: split one document into chunks of at most MAX_CHUNK_CHARS characters.

    Every chunk needs the metadata "source" (= file_name), "company", "fiscal_year" (an int) and
    "doc_type" (one of DOC_TYPES). Do not call the LLM here.
    """
    soup = BeautifulSoup(html, "html.parser")
    meta = _extract_doc_meta(file_name, soup)
    root = soup.body or soup

    chunks: list[Document] = []
    section = ""
    buf: list[str] = []
    skip_next_toc_list = False
    target_text_limit = MAX_CHUNK_CHARS - 120

    def make_doc(text: str) -> Document:
        return Document(page_content=text[:MAX_CHUNK_CHARS], metadata=dict(meta))

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        header = f"[{section}]\n" if section else ""
        avail = MAX_CHUNK_CHARS - len(header)
        cur = ""
        for item in buf:
            for piece in _split_long_text(item, avail):
                cand = f"{cur}\n{piece}" if cur else piece
                if len(cand) <= avail:
                    cur = cand
                else:
                    if cur:
                        chunks.append(make_doc(header + cur))
                    cur = piece
        if cur:
            chunks.append(make_doc(header + cur))
        buf = []

    for kind, node in _iter_blocks(root):
        if kind == "heading":
            txt = _clean_cell_text(node)
            if not txt:
                continue
            if txt.lower() == "contents":
                flush_buf()
                skip_next_toc_list = True
                continue
            skip_next_toc_list = False
            flush_buf()
            section = txt
        elif kind == "list":
            if skip_next_toc_list:
                skip_next_toc_list = False
                continue
            items = []
            for li in node.find_all("li", recursive=False):
                t = _clean_cell_text(li)
                if t:
                    items.append(f"- {t}")
            if not items:
                t = _clean_cell_text(node)
                if t:
                    items.append(t)
            if items:
                buf.append("\n".join(items))
        elif kind == "table":
            skip_next_toc_list = False
            flush_buf()
            t_text = table_to_text(str(node)).strip()
            if not t_text:
                continue
            header = f"[{section}]\n" if section else ""
            if len(header) + len(t_text) <= MAX_CHUNK_CHARS:
                chunks.append(make_doc(header + t_text))
            else:
                t_lines = t_text.splitlines()
                cap_line = t_lines[0] if t_lines else ""
                prefix = header + (cap_line + "\n" if cap_line else "")
                avail = MAX_CHUNK_CHARS - len(prefix)
                if avail < 200:
                    prefix = (cap_line + "\n") if cap_line else ""
                    avail = MAX_CHUNK_CHARS - len(prefix)
                cur_lines: list[str] = []
                cur_len = 0
                body_lines = t_lines[1:] if cap_line else t_lines
                for line in body_lines:
                    for piece in _split_long_text(line, avail):
                        add_len = len(piece) + (1 if cur_lines else 0)
                        if cur_len + add_len <= avail:
                            cur_lines.append(piece)
                            cur_len += add_len
                        else:
                            if cur_lines:
                                chunks.append(make_doc(prefix + "\n".join(cur_lines)))
                            cur_lines = [piece]
                            cur_len = len(piece)
                if cur_lines:
                    chunks.append(make_doc(prefix + "\n".join(cur_lines)))
        else:
            skip_next_toc_list = False
            txt = node if kind == "raw" else _clean_cell_text(node)
            if txt:
                buf.append(txt)
                if sum(len(x) + 1 for x in buf) >= target_text_limit:
                    last = buf[-1]
                    flush_buf()
                    if len(last) <= 300:
                        buf.append(last)

    flush_buf()
    return chunks


def parse_query(question: str) -> dict:
    """Task 2.2: {"companies": [names], "fiscal_year": int or None, "doc_type": one of DOC_TYPES}.

    The rules for each field are under Task 2.2 in the handout. You may use make_llm() here.
    """
    global _PARSE_LLM
    if question in _PARSE_CACHE:
        cached = _PARSE_CACHE[question]
        return {"companies": list(cached["companies"]), "fiscal_year": cached["fiscal_year"], "doc_type": cached["doc_type"]}

    if _PARSE_LLM is None:
        _PARSE_LLM = make_llm(temperature=0.0, thinking=False, max_tokens=200)

    raw = str(_PARSE_LLM.invoke(_PARSE_PROMPT.replace("{question}", question)).content).strip()
    m = re.search(r"\{.*\}", raw, re.S)
    data: dict[str, Any] = {}
    if m:
        try:
            data = json.loads(m.group(0))
        except Exception:
            data = {}

    companies = data.get("companies") or []
    if isinstance(companies, str):
        companies = [companies]
    companies = [str(c).strip() for c in companies if str(c).strip()]

    # 校验 fiscal_year：确保确实是问题中提到的年份
    fy = as_year(data.get("fiscal_year"))
    years_in_q = [int(y) for y in re.findall(r"(?<!\d)(20\d{2})(?!\d)", question)]
    if not years_in_q:
        fy = None
    elif fy not in years_in_q:
        fy = years_in_q[-1]

    doc_type = data.get("doc_type")
    if doc_type not in DOC_TYPES:
        ql = question.lower()
        if any(k in ql for k in ("earnings call", "conference call")):
            doc_type = "earnings_call"
        elif any(k in ql for k in ("target price", "rating", "forecast", "results review", "estimate")):
            doc_type = "research_report"
        else:
            doc_type = "annual_report"

    result = {"companies": companies, "fiscal_year": fy, "doc_type": doc_type}
    _PARSE_CACHE[question] = result
    return {"companies": list(companies), "fiscal_year": fy, "doc_type": doc_type}


def build_index(chunks: list[Document]) -> Any:
    """Task 2: index the chunks of the whole corpus (the output of chunk_document), once per run.

    Return anything retrieve() needs: a BM25 or TF-IDF index, the chunks grouped by metadata, ...
    """
    tokenized = []
    by_company: dict[str, list[int]] = {}
    by_key: dict[tuple[str, str, int], list[int]] = {}
    years_by_comp_type: dict[tuple[str, str], set[int]] = {}

    for i, doc in enumerate(chunks):
        meta = doc.metadata or {}
        comp_name = str(meta.get("company", ""))
        ckey = company_key(comp_name)
        dtype = str(meta.get("doc_type", ""))
        fy = as_year(meta.get("fiscal_year"))

        # 把公司名和标题信息适度纳入词法索引
        tokenized.append(_bm25_tokenize(doc.page_content, expand=False))

        by_company.setdefault(ckey, []).append(i)
        if fy is not None:
            by_key.setdefault((ckey, dtype, fy), []).append(i)
            years_by_comp_type.setdefault((ckey, dtype), set()).add(fy)

    bm25 = BM25Okapi(tokenized) if tokenized else None
    return {
        "chunks": chunks,
        "bm25": bm25,
        "by_company": by_company,
        "by_key": by_key,
        "years_by_comp_type": years_by_comp_type,
    }


def retrieve(index: Any, question: str) -> list[Document]:
    """Tasks 2 and 3: at most MAX_RETRIEVED of the chunks given to build_index, best first.

    Return [] when the corpus cannot answer the question: the generator then answers NOT_FOUND.
    """
    ### YOUR CODE HERE
    global _RERANK_LLM
    if not index or not index.get("chunks"):
        return []

    chunks: list[Document] = index["chunks"]
    parsed = parse_query(question)
    companies: list[str] = parsed.get("companies") or []
    req_year: int | None = parsed.get("fiscal_year")
    doc_type: str = parsed.get("doc_type") or "annual_report"

    # 第一道防线：如果有目标公司不在语料库中，直接拒答
    comp_keys: list[str] = []
    for c in companies:
        ck = company_key(c)
        if ck not in index["by_company"]:
            return []
        if ck not in comp_keys:
            comp_keys.append(ck)

    q_tokens = _bm25_tokenize(question, expand=True)
    comp_word_set = {w for ck in comp_keys for w in ck.split()}
    filtered_q_tokens = [t for t in q_tokens if t not in comp_word_set] or q_tokens

    # 收集每个目标公司的候选 chunk 下标
    candidate_indices: list[int] = []
    if comp_keys:
        per_comp_k = 12 if len(comp_keys) == 1 else 6
        for ck in comp_keys:
            avail_years = index["years_by_comp_type"].get((ck, doc_type), set())
            target_years = _select_target_years(avail_years, req_year, doc_type)
            if not target_years:
                # 若所问年份在该文档类型下完全不存在，直接拒答
                return []
            pool: list[int] = []
            for ty in target_years:
                pool.extend(index["by_key"].get((ck, doc_type, ty), []))
            if not pool:
                return []
            candidate_indices.extend(_bm25_rank_subset(pool, index, filtered_q_tokens, per_comp_k))
    else:
        # 极少数未解析出公司名的情况：全库 BM25
        candidate_indices = _bm25_rank_subset(list(range(len(chunks))), index, q_tokens, 12)

    # 去重并保持顺序
    seen: set[int] = set()
    unique_cands: list[int] = []
    for idx in candidate_indices:
        if idx not in seen:
            seen.add(idx)
            unique_cands.append(idx)

    if not unique_cands:
        return []

    # 第二道防线 + LLM 精排：确认真的披露了该事实，并选出最精准的 <= 5 个 chunk
    if _RERANK_LLM is None:
        _RERANK_LLM = make_llm(temperature=0.0, thinking=False, max_tokens=200)

    cand_lines = []
    for pos, idx in enumerate(unique_cands):
        c = chunks[idx]
        m = c.metadata or {}
        cand_lines.append(
            f"[{pos}] company: {m.get('company')} | fiscal_year: {m.get('fiscal_year')} | doc_type: {m.get('doc_type')}\n{c.page_content}"
        )
    prompt = _RERANK_PROMPT.replace("{question}", question).replace("{candidates}", "\n\n".join(cand_lines))

    try:
        raw = str(_RERANK_LLM.invoke(prompt).content).strip()
        m = re.search(r"\{.*\}", raw, re.S)
        if m:
            res = json.loads(m.group(0))
            if not res.get("answerable", False):
                return []
            sel = [int(i) for i in (res.get("selected") or []) if isinstance(i, int) or str(i).isdigit()]
            picked = []
            used = set()
            for pos in sel:
                if 0 <= pos < len(unique_cands) and pos not in used:
                    used.add(pos)
                    picked.append(chunks[unique_cands[pos]])
                if len(picked) >= MAX_RETRIEVED:
                    break
            if picked:
                return picked
            return []
    except Exception:
        pass

    # 若精排 JSON 解析异常，降级返回 BM25 前几位（对比题保证每家至少 2 个）
    if len(comp_keys) == 2:
        picked = []
        for ck in comp_keys:
            comp_cands = [chunks[i] for i in unique_cands if company_key(chunks[i].metadata.get("company", "")) == ck]
            picked.extend(comp_cands[:2])
        return picked[:MAX_RETRIEVED]
    return [chunks[i] for i in unique_cands[:MAX_RETRIEVED]]


# =============================================================================
# Provided code below. Do not change it.
# =============================================================================

LLM_MODEL = "deepseek-v4-flash"
DOC_TYPES = ("annual_report", "earnings_call", "research_report")

MAX_CHUNK_CHARS = 1500  # characters per chunk
MAX_RETRIEVED = 5       # chunks per question
TIME_LIMIT_SECONDS = 30 * 60

# USD per 1M tokens, only for the cost estimate that the runner prints.
PRICE_LLM_INPUT, PRICE_LLM_OUTPUT = 0.15, 0.60

# Points (Section 2 of the handout).
POINTS = {"1.1": 8, "1.2": 7, "2.1": 5, "2.2": 10, "2.3": 15, "3": 10, "4": 20}
RECALL_ZERO, RECALL_FULL = 0.50, 0.90  # Task 2.3: 0 points at or below RECALL_ZERO, full points from RECALL_FULL
MAX_LINE_CHARS = 400                   # Task 1.1
MIN_PROBES_FOUND = 0.9                 # Task 1.2


class UsageTracker(BaseCallbackHandler):
    """Counts LLM tokens, for the cost estimate. There is no budget."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.llm_calls = self.llm_input_tokens = self.llm_output_tokens = 0

    def on_llm_end(self, response, **kwargs) -> None:
        tokens_in = tokens_out = 0
        for generations in response.generations:
            for generation in generations:
                usage = getattr(getattr(generation, "message", None), "usage_metadata", None) or {}
                tokens_in += usage.get("input_tokens", 0)
                tokens_out += usage.get("output_tokens", 0)
        with self._lock:
            self.llm_calls += 1
            self.llm_input_tokens += tokens_in
            self.llm_output_tokens += tokens_out

    def cost(self) -> float:
        return (self.llm_input_tokens * PRICE_LLM_INPUT + self.llm_output_tokens * PRICE_LLM_OUTPUT) / 1e6

    def summary(self) -> dict:
        return {"llm_calls": self.llm_calls, "llm_input_tokens": self.llm_input_tokens,
                "llm_output_tokens": self.llm_output_tokens, "cost_usd": round(self.cost(), 4)}


USAGE = UsageTracker()            # your calls
GENERATOR_USAGE = UsageTracker()  # the generator's calls


def _chat(usage: UsageTracker, temperature: float, thinking: bool, max_tokens: int | None) -> ChatDeepSeek:
    return ChatDeepSeek(
        model=LLM_MODEL,
        api_key=os.environ["DEEPSEEK_API_KEY"],
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=3,
        # Thinking mode ignores `temperature` and its tokens count as output.
        extra_body={"thinking": {"type": "enabled" if thinking else "disabled"}},
        callbacks=[usage],
    )


def make_llm(temperature: float = 0.0, thinking: bool = False, max_tokens: int | None = None) -> ChatDeepSeek:
    """deepseek-v4-flash. Use it for every LLM call you make (only in parse_query, build_index and retrieve)."""
    return _chat(USAGE, temperature, thinking, max_tokens)


# ----------------------------------------------------------------------------- the generator

GENERATOR_PROMPT = """You answer questions about company reports, using only the context below.

Context:
{context}

Question: {question}

Rules:
- Use only the context. The header of each passage says which document it comes from.
- Check that the passage you use is about the company and the fiscal year that the question asks about.
- If the context does not contain the answer, reply exactly NOT_FOUND.
- For a number, reply with that one number only, in the unit the question asks for, without units, words or other numbers (for example: 1913.4).
- For a name (a person, a company, a city, an audit firm or a rating), reply with the name only.

Answer:"""

_generator: ChatDeepSeek | None = None


def format_context(chunks: list[Document]) -> str:
    """The retrieved chunks as the generator sees them: a header from the metadata, then the text."""
    parts = []
    for i, chunk in enumerate(chunks, 1):
        meta = chunk.metadata or {}
        header = " | ".join(f"{key}: {str(meta[key])[:100]}"
                            for key in ("source", "company", "fiscal_year", "doc_type") if meta.get(key) not in (None, ""))
        parts.append(f"[{i}] {header}\n{chunk.page_content}")
    return "\n\n".join(parts)


def generate(question: str, chunks: list[Document]) -> str:
    """The fixed generator: deepseek-v4-flash answers from the retrieved chunks only."""
    global _generator
    if not chunks:
        return "NOT_FOUND"
    if _generator is None:
        _generator = _chat(GENERATOR_USAGE, 0.0, False, 64)
    prompt = GENERATOR_PROMPT.format(context=format_context(chunks), question=question)
    return str(_generator.invoke(prompt).content).strip()


# ----------------------------------------------------------------------------- loaders

def load_corpus(folder: str | Path) -> dict[str, str]:
    """{file name: raw HTML} for every .html file in the folder."""
    return {p.name: p.read_text(encoding="utf-8") for p in sorted(Path(folder).glob("*.html"))}


def load_json(path: str | Path) -> Any:
    """The parsed file, or None if it does not exist (the private set has no answers)."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ----------------------------------------------------------------------------- scoring

# One number: optional minus sign, optional thousands separators, optional decimals.
NUMBER_RE = re.compile(r"(?<![\d.,])-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?")
TITLES = {"mr", "mrs", "ms", "dr", "prof", "professor", "sir"}
LEGAL_SUFFIXES = {"inc", "corp", "corporation", "ltd", "limited", "plc", "llc", "co"}


def parse_number(text: str) -> float | None:
    """The single number in the text, or None if there is not exactly one."""
    found = NUMBER_RE.findall(text.replace("\u2212", "-"))
    if len(found) != 1:
        return None
    return float(found[0].replace(",", ""))


def normalize(text: str) -> str:
    """Lower case, no punctuation, no titles, padded with spaces for whole-word matching."""
    text = unicodedata.normalize("NFKC", str(text)).lower().replace("_", " ")
    words = [w for w in re.sub(r"[^a-z0-9]+", " ", text).split() if w not in TITLES]
    return " " + " ".join(words) + " "


def company_key(name: Any) -> str:
    """A company name without case, punctuation or legal suffix: 'Northgate Robotics Inc.' -> 'northgate robotics'."""
    return " ".join(w for w in normalize(name).split() if w not in LEGAL_SUFFIXES)


def squash(text: Any) -> str:
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(text))).strip().casefold()


def as_year(value: Any) -> int | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.strip().isdigit():
        return int(value.strip())
    return None


def answer_correct(answer: str, gold: dict) -> bool:
    answer = str(answer or "")
    said_not_found = " not found " in normalize(answer)
    if gold["type"] == "not_found":
        return said_not_found
    if said_not_found:
        return False
    if gold["type"] == "number":
        value = parse_number(answer)
        if value is None:
            return False
        return abs(value - gold["answer"]) <= max(0.01 * abs(gold["answer"]), 0.05)
    names = [gold["answer"], *gold.get("aliases", [])]
    if not any(normalize(name) in normalize(answer) for name in names):
        return False
    return not any(normalize(other) in normalize(answer) for other in gold.get("reject", []))


def table_score(output: Any, table: dict) -> float:
    """Task 1.1, one table: the share of cells found on a line with their row and column labels and
    without the label of another row, halved if the caption is missing."""
    if not isinstance(output, str) or not table["cells"]:
        return 0.0
    lines = [squash(line) for line in output.splitlines()]
    lines = [line for line in lines if line and len(line) <= MAX_LINE_CHARS]
    labels = {squash(row) for row, _, _ in table["cells"]}

    def kept(row: str, col: str, value: str) -> bool:
        row, col, value = squash(row), squash(col), squash(value)
        others = [o for o in labels if o not in row]  # e.g. "total liabilities" may be in "total liabilities and equity"
        return any(row in line and col in line and value in line and not any(o in line for o in others)
                   for line in lines)

    share = sum(kept(*cell) for cell in table["cells"]) / len(table["cells"])
    return share if squash(table["caption"]) in squash(output) else share / 2


def document_problems(chunks: Any, error: str | None, meta: dict, file_name: str) -> list[str]:
    """Task 1.2, one document: why its chunks fail the checks (an empty list means they pass)."""
    if error:
        return [f"error: {error}"]
    if not isinstance(chunks, list) or not chunks:
        return ["no chunks"]
    if not all(isinstance(c, Document) for c in chunks):
        return ["not a list of Document"]
    problems = []
    longest = max(len(c.page_content) for c in chunks)
    if longest > MAX_CHUNK_CHARS:
        problems.append(f"a chunk has {longest} characters")
    for key, ok in (("source", lambda v: v == file_name),
                    ("company", lambda v: company_key(v) == company_key(meta["company"])),
                    ("fiscal_year", lambda v: as_year(v) == meta["fiscal_year"]),
                    ("doc_type", lambda v: v == meta["doc_type"])):
        wrong = [c.metadata.get(key) for c in chunks if not ok(c.metadata.get(key))]
        if wrong:
            problems.append(f"{key} {wrong[0]!r} (expected {meta.get(key, file_name)!r}) on {len(wrong)} chunk(s)")
    text = squash("\n".join(c.page_content for c in chunks))
    missing = [p for p in meta["probes"] if squash(p) not in text]
    if len(missing) > (1 - MIN_PROBES_FOUND) * len(meta["probes"]):
        problems.append(f"text missing from the chunks: {missing[:3]}")
    return problems


def parse_score(parsed: Any, gold: dict) -> float:
    """Task 2.2, one question: the share of the three fields that are right."""
    if not isinstance(parsed, dict):
        return 0.0
    expected = gold["query"]
    companies = parsed.get("companies")
    if isinstance(companies, str):
        companies = [companies]
    right = 0
    if isinstance(companies, (list, tuple)) and {company_key(c) for c in companies} == {company_key(c) for c in expected["companies"]}:
        right += 1
    if "fiscal_year" in parsed and as_year(parsed["fiscal_year"]) == expected["fiscal_year"] and \
            (parsed["fiscal_year"] is None) == (expected["fiscal_year"] is None):
        right += 1
    if parsed.get("doc_type") == expected["doc_type"]:
        right += 1
    return right / 3


def contains(chunk_text: str, printed: str) -> bool:
    """The fact as printed in the document is in the chunk (a number only as a whole number)."""
    text, printed = squash(chunk_text), squash(printed)
    if re.fullmatch(r"[\d,.%-]+", printed):
        return re.search(rf"(?<![\d.,]){re.escape(printed)}(?![\d]|[.,]\d)", text) is not None
    return printed in text


def retrieval_hit(chunks: list[Document], gold: dict) -> bool:
    """For every fact the question needs, a retrieved chunk of a document that states it contains it."""
    return bool(gold["evidence"]) and all(
        any(c.metadata.get("source") == file and contains(c.page_content, printed)
            for c in chunks for file, printed in group)
        for group in gold["evidence"])


def clip(x: float) -> float:
    return max(0.0, min(1.0, x))


# ----------------------------------------------------------------------------- runner

def main() -> None:
    parser = argparse.ArgumentParser(description="FTEC5660 HW4: retrieval for RAG over company reports")
    parser.add_argument("--data-folder", default="public_test",
                        help="folder with corpus/, questions.json and (optionally) the answer files")
    parser.add_argument("--output", default="results.csv")
    args = parser.parse_args()

    folder = Path(args.data_folder)
    corpus = load_corpus(folder / "corpus")
    questions = load_json(folder / "questions.json")
    ground_truth = load_json(folder / "ground_truth.json")
    docs_meta = load_json(folder / "docs.json")
    tables = load_json(folder / "tables.json")
    print(f"{len(corpus)} documents, {len(questions)} questions")
    start = time.perf_counter()
    points: dict[str, float] = {}

    # Task 1.1: tables (no model calls allowed)
    if tables:
        before = USAGE.llm_calls
        scores = []
        for table in tables:
            try:
                scores.append(table_score(table_to_text(table["html"]), table))
            except Exception as e:
                print(f"table_to_text failed on a table of {table['source']}: {e!r}")
                scores.append(0.0)
        if USAGE.llm_calls != before:
            print("WARNING: table_to_text called the LLM, which is not allowed (Task 1.1 = 0).")
            scores = [0.0]
        points["1.1"] = POINTS["1.1"] * sum(scores) / len(scores)
        print(f"\nTask 1.1 table_to_text: {sum(scores) / len(scores):.1%} of the table cells kept with their labels")

    # Task 1.2: chunks (no model calls allowed)
    before = USAGE.llm_calls
    chunks_by_file: dict[str, list] = {}
    errors: dict[str, str] = {}
    for file_name, html in corpus.items():
        try:
            chunks_by_file[file_name] = list(chunk_document(file_name, html) or [])
        except Exception as e:
            traceback.print_exc()
            chunks_by_file[file_name], errors[file_name] = [], repr(e)
    called_models = USAGE.llm_calls != before
    if called_models:
        print("WARNING: chunk_document called the LLM, which is not allowed (Task 1.2 = 0).")
    if docs_meta:
        problems = {f: document_problems(chunks_by_file.get(f), errors.get(f), m, f) for f, m in docs_meta.items()}
        passed = sum(not p for p in problems.values()) / len(problems)
        points["1.2"] = 0.0 if called_models else POINTS["1.2"] * passed
        print(f"Task 1.2 chunk_document: {passed:.1%} of the documents pass")
        for f, p in [(f, p) for f, p in problems.items() if p][:5]:
            print(f"  {f}: {'; '.join(p)}")

    # the chunks that retrieve() may return
    all_chunks = [c for f in sorted(chunks_by_file) for c in chunks_by_file[f]
                  if isinstance(c, Document) and len(c.page_content) <= MAX_CHUNK_CHARS]
    by_text: dict[str, list[Document]] = {}
    for c in all_chunks:
        by_text.setdefault(c.page_content, []).append(c)
    print(f"{len(all_chunks)} chunks to index")

    # Task 2.2: query parsing
    parsed: dict[str, Any] = {}
    for qid, question in questions.items():
        try:
            parsed[qid] = parse_query(question)
        except Exception as e:
            parsed[qid] = f"error: {e!r}"

    # Tasks 2.1, 2.3 and 3: retrieval
    index = build_index(all_chunks)
    index_seconds = time.perf_counter() - start
    print(f"Index built in {index_seconds:.1f}s | {USAGE.summary()}")
    retrieved: dict[str, list[Document]] = {}
    valid: dict[str, bool] = {}
    for qid, question in questions.items():
        try:
            result = retrieve(index, question)
            ok = isinstance(result, (list, tuple)) and len(result) <= MAX_RETRIEVED
            kept = []
            for chunk in list(result or []) if isinstance(result, (list, tuple)) else []:
                matches = by_text.get(getattr(chunk, "page_content", None), [])
                if not matches:
                    ok = False
                    continue
                source = (getattr(chunk, "metadata", None) or {}).get("source")
                kept.append(next((m for m in matches if m.metadata.get("source") == source), matches[0]))
            retrieved[qid], valid[qid] = kept[:MAX_RETRIEVED], ok
        except Exception:
            traceback.print_exc()
            retrieved[qid], valid[qid] = [], False
    retrieval_seconds = time.perf_counter() - start - index_seconds

    # Task 4: the fixed generator
    def answer(qid: str) -> str:
        try:
            return generate(questions[qid], retrieved[qid])
        except Exception as e:
            print(f"Generator failed on {qid}: {e!r}")
            return ""

    with ThreadPoolExecutor(max_workers=8) as pool:
        answers = dict(zip(questions, pool.map(answer, questions)))
    total_seconds = time.perf_counter() - start

    rows = []
    for qid, question in questions.items():
        sources = sorted({c.metadata.get("source", "") for c in retrieved[qid]})
        row = {"id": qid, "category": "", "question": question, "parsed": json.dumps(parsed[qid], default=str),
               "parse_score": "", "retrieved": ";".join(sources), "valid": valid[qid], "retrieval_hit": "",
               "answer": answers[qid], "correct": "", "expected": ""}
        if ground_truth and qid in ground_truth:
            gold = ground_truth[qid]
            row["category"] = gold["category"]
            row["parse_score"] = round(parse_score(parsed[qid], gold), 3)
            row["retrieval_hit"] = retrieval_hit(retrieved[qid], gold) if gold["evidence"] else ""
            row["correct"] = answer_correct(answers[qid], gold)
            if not row["correct"]:
                row["expected"] = gold["answer"]
        rows.append(row)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output}")

    if ground_truth:
        graded = [r for r in rows if r["id"] in ground_truth]
        answerable = [r for r in graded if ground_truth[r["id"]]["evidence"]]
        unanswerable = [r for r in graded if not ground_truth[r["id"]]["evidence"]]
        parse_mean = sum(r["parse_score"] for r in graded) / len(graded)
        non_empty = sum(r["valid"] and bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        recall = sum(r["retrieval_hit"] for r in answerable) / max(len(answerable), 1)
        kept = sum(bool(r["retrieved"]) for r in answerable) / max(len(answerable), 1)
        rejected = sum(not r["retrieved"] for r in unanswerable) / max(len(unanswerable), 1)
        accuracy = sum(r["correct"] for r in graded) / len(graded)
        points["2.1"] = POINTS["2.1"] * non_empty
        points["2.2"] = POINTS["2.2"] * parse_mean
        points["2.3"] = POINTS["2.3"] * clip((recall - RECALL_ZERO) / (RECALL_FULL - RECALL_ZERO))
        points["3"] = POINTS["3"] * clip(kept + rejected - 1)
        points["4"] = POINTS["4"] * accuracy
        print(f"Task 2.1 valid, non-empty retrieval on answerable questions: {non_empty:.1%}")
        print(f"Task 2.2 parse_query fields right: {parse_mean:.1%}")
        print(f"Task 2.3 retrieval recall@{MAX_RETRIEVED} on answerable questions: {recall:.1%}")
        print(f"Task 3   answerable questions with chunks: {kept:.1%} | unanswerable questions with []: {rejected:.1%}")
        print(f"Task 4   answer accuracy: {accuracy:.1%}")
        by_category: dict[str, list[tuple]] = {}
        for r in graded:
            by_category.setdefault(r["category"], []).append((r["correct"], r["retrieval_hit"]))
        print("\nBy category       accuracy   recall")
        for category, results in sorted(by_category.items()):
            hits = [h for _, h in results if h != ""]
            recall_text = f"{sum(hits) / len(hits):.1%}" if hits else "-"
            print(f"  {category:<14}{sum(c for c, _ in results) / len(results):>9.1%}{recall_text:>9}")

    if points:
        print("\nEstimated points (the grade uses the private set):")
        for task in POINTS:
            if task in points:
                print(f"  Task {task:<4}{points[task]:6.2f} / {POINTS[task]}")
        print(f"  Total    {sum(points.values()):6.2f} / {sum(POINTS[t] for t in points)}")

    print(f"\nYour usage: {USAGE.summary()}")
    print(f"Generator usage: {GENERATOR_USAGE.summary()}")
    print(f"Time: {total_seconds:.1f}s (index {index_seconds:.1f}s, retrieval {retrieval_seconds:.1f}s)")
    if total_seconds > TIME_LIMIT_SECONDS:
        print(f"WARNING: the run took longer than the {TIME_LIMIT_SECONDS // 60}-minute limit.")


if __name__ == "__main__":
    main()

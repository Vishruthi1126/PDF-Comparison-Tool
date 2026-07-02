"""
compare.py  —  Fast PDF diff engine  (optimised edition)
---------------------------------------------------------

Pipeline per page:
  1. Parallel text extraction using ThreadPoolExecutor
  2. Normalise + hash → skip identical pages instantly
  3. Word-level diff (autojunk=True for 5-10x speedup on large pages)
  4. Pages with > WORD_DIFF_LIMIT changes → one summary row
  5. changes list capped at MAX_CHANGES_ENTRIES to avoid memory bloat
"""

import hashlib
import difflib
import time
import logging
import io
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

import fitz  # PyMuPDF

log = logging.getLogger(__name__)

# ── Tunable thresholds ────────────────────────────────────────────────────────
WORD_DIFF_LIMIT    = 1000   # pages with more changed words get a summary row
MAX_CHANGES_ENTRIES = 30_000  # cap inline diff entries to keep UI fast
MAX_WORKERS = min(8, (multiprocessing.cpu_count() or 2) * 2)
WHITESPACE_NORM    = True

# ── Colours ───────────────────────────────────────────────────────────────────
COLOR_INSERT = (0.18, 0.66, 0.22)
COLOR_MODIFY = (0.90, 0.72, 0.00)
COLOR_DELETE = (0.86, 0.20, 0.20)

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    return " ".join(text.split()) if WHITESPACE_NORM else text

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "")).strip().lower()

def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="replace")).hexdigest()

def _extract_document_title(pdf_bytes: bytes) -> str:
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        title = doc.metadata.get("title", "")
        if title and title.strip():
            return title.strip()
        if len(doc) == 0:
            return ""
        page = doc[0]
        info = page.get_text("dict")
        spans = []
        for block in info.get("blocks", []):
            if block.get("type") != 0:
                continue
            for line in block.get("lines", []):
                for span in line.get("spans", []):
                    text = span.get("text", "").strip()
                    if not text or len(text) > 120:
                        continue
                    if not any(ch.isalpha() for ch in text):
                        continue
                    spans.append({
                        "text": text,
                        "size": span.get("size", 0),
                        "bbox": span.get("bbox", [0, 0, 0, 0]),
                        "font": span.get("font", ""),
                        "flags": span.get("flags", 0),
                    })
        if not spans:
            return ""
        max_size = max(s["size"] for s in spans)
        page_top = page.rect.height * 0.25
        scored = []
        for span in spans:
            score = span["size"] / max_size if max_size else 0
            if span["bbox"][1] < page_top:
                score += 0.2
            if "bold" in span["font"].lower() or (span["flags"] & 2):
                score += 0.2
            scored.append((score, span))
        scored.sort(key=lambda item: item[0], reverse=True)
        best = scored[0][1]
        same_line = [s["text"] for _, s in scored if abs(s["bbox"][1] - best["bbox"][1]) < 4]
        title_line = " ".join(dict.fromkeys(same_line)) or best["text"]
        return re.sub(r"\s+", " ", title_line).strip()

def _page_heading_candidates(page):
    info = page.get_text("dict")
    if not info:
        return []
    spans = []
    for block in info.get("blocks", []):
        if block.get("type") != 0:
            continue
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span.get("text", "").strip()
                if not text or len(text) > 120:
                    continue
                if not any(ch.isalpha() for ch in text):
                    continue
                spans.append({
                    "text": text,
                    "size": span.get("size", 0),
                    "bbox": span.get("bbox", [0, 0, 0, 0]),
                    "font": span.get("font", ""),
                    "flags": span.get("flags", 0),
                })
    if not spans:
        return []
    max_size = max(s["size"] for s in spans)
    page_top = page.rect.height * 0.35
    candidates = []
    for span in spans:
        bold = "bold" in span["font"].lower() or (span["flags"] & 2)
        size_ratio = span["size"] / max_size if max_size else 0
        score = size_ratio
        if bold:
            score += 0.3
        if span["bbox"][1] < page_top:
            score += 0.1
        if re.search(r"\b(section|chapter|part|appendix|article|subtitle)\b", span["text"], re.I):
            score += 0.3
        if score >= 0.75 or bold:
            candidates.append((score, span))
    if not candidates:
        return []
    candidates.sort(key=lambda item: item[0], reverse=True)
    return [c[1]["text"] for c in candidates[:2]]


def _extract_sections(pdf_bytes: bytes):
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = len(doc)
        headings = []
        for page_idx in range(1, total):
            page = doc[page_idx]
            candidates = _page_heading_candidates(page)
            if not candidates:
                continue
            heading_text = candidates[0]
            normal = _normalize_text(heading_text)
            if not normal:
                continue
            if headings and normal == headings[-1]["name"]:
                continue
            headings.append({
                "raw_name": heading_text,
                "name": normal,
                "start": page_idx,
            })
        if not headings:
            return [{"raw_name": "Entire document", "name": "entire document", "start": 0, "end": total - 1}]
        if headings[0]["start"] > 0:
            headings.insert(0, {"raw_name": "Front matter", "name": "front matter", "start": 0})
        for idx in range(len(headings) - 1):
            headings[idx]["end"] = headings[idx + 1]["start"] - 1
        headings[-1]["end"] = total - 1
        return headings


def _fit_change_trend(points):
    """Fit a linear trend line to page change counts."""
    if len(points) < 2:
        return None
    xs = [float(x) for x, _ in points]
    ys = [float(y) for _, y in points]
    n = len(xs)
    sx = sum(xs)
    sy = sum(ys)
    sxx = sum(x * x for x in xs)
    sxy = sum(x * y for x, y in zip(xs, ys))
    denom = n * sxx - sx * sx
    if denom == 0:
        return None
    slope = (n * sxy - sx * sy) / denom
    intercept = (sy - slope * sx) / n
    return {"slope": slope, "intercept": intercept}


def _extract_page_text(pdf_bytes: bytes, page_idx: int) -> str:
    """Open a fresh doc handle (thread-safe) and extract one page's text."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        return doc[page_idx].get_text()
    finally:
        doc.close()


def _extract_page_words(pdf_bytes: bytes, page_idx: int):
    """Return (words, rects) for one page using a fresh doc handle."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    try:
        raw = doc[page_idx].get_text("words", sort=True)
        words = [w[4] for w in raw]
        rects = [fitz.Rect(w[0], w[1], w[2], w[3]) for w in raw]
        return words, rects
    finally:
        doc.close()


# ── Per-page diff (called from thread pool) ───────────────────────────────────

def _diff_page(page_idx, pdf1_bytes, pdf2_bytes, n1, n2):
    """
    Compute diff for one changed page.
    Returns (page_idx, report_rows, changes_fragment, hilites, summary_delta).
    """
    page_label = f"Page {page_idx + 1}"
    report_rows = []
    changes_fragment = []
    hilites = []
    summary_delta = {"inserted": 0, "modified": 0, "deleted": 0}

    words1, _      = _extract_page_words(pdf1_bytes, page_idx) if page_idx < n1 else ([], [])
    words2, rects2 = _extract_page_words(pdf2_bytes, page_idx) if page_idx < n2 else ([], [])

    tok1 = [w.lower() for w in words1]
    tok2 = [w.lower() for w in words2]

    # autojunk=True: use difflib's heuristic — huge speedup on long lists
    # First arg is isjunk (None = no filter); autojunk=True enables the speedup
    matcher = difflib.SequenceMatcher(None, tok1, tok2, autojunk=True)
    opcodes = matcher.get_opcodes()

    page_changes = 0

    for tag, i1, i2, j1, j2 in opcodes:
        if tag == "equal":
            for k in range(j1, j2):
                changes_fragment.append(("equal", words2[k]))
            continue

        page_changes += (i2 - i1) + (j2 - j1)

        if page_changes > WORD_DIFF_LIMIT:
            report_rows.append({
                "page":        page_label,
                "pdf1_text":   f"[{len(words1)} words]",
                "pdf2_text":   f"[{len(words2)} words]",
                "change_type": "Modified",
            })
            summary_delta["modified"] += 1
            if rects2:
                all_x0 = min(r.x0 for r in rects2)
                all_y0 = min(r.y0 for r in rects2)
                all_x1 = max(r.x1 for r in rects2)
                all_y1 = max(r.y1 for r in rects2)
                import fitz as _fitz
                hilites.append((_fitz.Rect(all_x0, all_y0, all_x1, all_y1), COLOR_MODIFY))
            changes_fragment.append((
                "modify",
                f"[Page {page_idx+1} heavily modified — {len(words1)} → {len(words2)} words]"
            ))
            break

        if tag == "delete":
            text = " ".join(words1[i1:i2])
            changes_fragment.append(("delete", text))
            report_rows.append({
                "page":        page_label,
                "pdf1_text":   text,
                "pdf2_text":   "—",
                "change_type": "Deleted",
            })
            summary_delta["deleted"] += 1

        elif tag == "insert":
            for k in range(j1, j2):
                changes_fragment.append(("insert", words2[k]))
                hilites.append((rects2[k], COLOR_INSERT))
            report_rows.append({
                "page":        page_label,
                "pdf1_text":   "—",
                "pdf2_text":   " ".join(words2[j1:j2]),
                "change_type": "Inserted",
            })
            summary_delta["inserted"] += 1

        elif tag == "replace":
            text_del = " ".join(words1[i1:i2])
            changes_fragment.append(("delete", text_del))
            for k in range(j1, j2):
                changes_fragment.append(("modify", words2[k]))
                hilites.append((rects2[k], COLOR_MODIFY))
            report_rows.append({
                "page":        page_label,
                "pdf1_text":   text_del,
                "pdf2_text":   " ".join(words2[j1:j2]),
                "change_type": "Modified",
            })
            summary_delta["modified"] += 1

    return page_idx, report_rows, changes_fragment, hilites, summary_delta, page_changes


# ── Main entry point ──────────────────────────────────────────────────────────

def compare_documents(pdf1_bytes, pdf2_bytes, progress_cb=None):
    """
    Compare two PDFs page-by-page using parallel text extraction + diff.

    Returns:
        report_rows, changes, summary, highlights_per_page
    """
    t_total = time.time()

    # Open docs just to get page counts
    with fitz.open(stream=pdf1_bytes, filetype="pdf") as d1:
        n1 = len(d1)
    with fitz.open(stream=pdf2_bytes, filetype="pdf") as d2:
        n2 = len(d2)

    total = max(n1, n2)

    # ── Phase 1: Parallel text extraction for hashing ─────────────────────────
    log.info("Phase 1: extracting text from %d pages (workers=%d)", total, MAX_WORKERS)
    t1 = time.time()

    texts1 = [""] * total
    texts2 = [""] * total

    def fetch_text(pdf_bytes, page_idx, n):
        if page_idx >= n:
            return page_idx, ""
        return page_idx, _extract_page_text(pdf_bytes, page_idx)

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futs1 = {pool.submit(fetch_text, pdf1_bytes, i, n1): ("1", i) for i in range(total)}
        futs2 = {pool.submit(fetch_text, pdf2_bytes, i, n2): ("2", i) for i in range(total)}
        for f in as_completed({**futs1, **futs2}):
            which, idx = (futs1 if f in futs1 else futs2)[f]
            _, txt = f.result()
            if which == "1":
                texts1[idx] = txt
            else:
                texts2[idx] = txt

    log.info("Phase 1 done in %.2fs", time.time() - t1)

    sections1 = _extract_sections(pdf1_bytes)
    sections2 = _extract_sections(pdf2_bytes)
    title1 = _extract_document_title(pdf1_bytes)
    title2 = _extract_document_title(pdf2_bytes)

    report_rows = []
    changes = []
    highlights_per_page = {}
    summary = {
        "inserted":      0,
        "modified":      0,
        "deleted":       0,
        "skipped_pages": 0,
        "total_pages":   total,
        "trend_slope":   None,
        "trend_intercept": None,
        "trend_direction": "flat",
        "section_count": max(len(sections1), len(sections2)),
        "title_match":   _normalize_text(title1) == _normalize_text(title2),
        "title1":        title1,
        "title2":        title2,
    }
    page_changes_per_page = [0] * total

    sections2_map = {section["name"]: section for section in sections2}
    section_pairs = []
    paired_names = set()

    for section1 in sections1:
        if section1["name"] in sections2_map:
            section_pairs.append((section1, sections2_map[section1["name"]]))
            paired_names.add(section1["name"])
        else:
            section_pairs.append((section1, None))

    for section2 in sections2:
        if section2["name"] not in paired_names:
            section_pairs.append((None, section2))

    if not any(s1 and s2 for s1, s2 in section_pairs) and sections1 and sections2:
        section_pairs = []
        common = min(len(sections1), len(sections2))
        for idx in range(common):
            section_pairs.append((sections1[idx], sections2[idx]))
        for section1 in sections1[common:]:
            section_pairs.append((section1, None))
        for section2 in sections2[common:]:
            section_pairs.append((None, section2))

    page_tasks = []
    for section1, section2 in section_pairs:
        if section1 is None:
            start1 = end1 = None
        else:
            start1 = section1["start"]
            end1 = section1["end"]
        if section2 is None:
            start2 = end2 = None
        else:
            start2 = section2["start"]
            end2 = section2["end"]

        len1 = (end1 - start1 + 1) if start1 is not None else 0
        len2 = (end2 - start2 + 1) if start2 is not None else 0
        for offset in range(max(len1, len2)):
            page1 = start1 + offset if offset < len1 else None
            page2 = start2 + offset if offset < len2 else None
            if page1 is not None and page2 is not None:
                if _hash(_normalize_text(texts1[page1])) != _hash(_normalize_text(texts2[page2])):
                    page_tasks.append((page1, page2))

    page_results = {}
    if page_tasks:
        t3 = time.time()
        with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
            futs = {
                pool.submit(_diff_page, page1, pdf1_bytes, pdf2_bytes, n1, n2): (page1, page2)
                for page1, page2 in page_tasks
            }
            done = 0
            for f in as_completed(futs):
                page_results[futs[f]] = f.result()
                done += 1
                if progress_cb:
                    progress_cb(done, len(page_tasks))
        log.info("Phase 3 diff done in %.2fs", time.time() - t3)
    else:
        log.info("Phase 3 diff done in %.2fs", 0.0)

    for section1, section2 in section_pairs:
        section_name = section1["raw_name"] if section1 is not None else section2["raw_name"]
        report_rows.append({
            "page":        f"Section: {section_name}",
            "pdf1_text":   "",
            "pdf2_text":   "",
            "change_type": "Section",
        })
        changes.append(("equal", f"--- Section: {section_name} ---"))

        if section1 is None:
            start1 = end1 = None
        else:
            start1 = section1["start"]
            end1 = section1["end"]
        if section2 is None:
            start2 = end2 = None
        else:
            start2 = section2["start"]
            end2 = section2["end"]

        len1 = (end1 - start1 + 1) if start1 is not None else 0
        len2 = (end2 - start2 + 1) if start2 is not None else 0
        for offset in range(max(len1, len2)):
            page1 = start1 + offset if offset < len1 else None
            page2 = start2 + offset if offset < len2 else None
            if page1 is not None and page2 is not None:
                if (page1, page2) in page_results:
                    _, rows, cfrag, hilites, sdelta, page_changes = page_results[(page1, page2)]
                    report_rows.extend(rows)
                    if len(changes) < MAX_CHANGES_ENTRIES:
                        remaining = MAX_CHANGES_ENTRIES - len(changes)
                        changes.extend(cfrag[:remaining])
                        if len(cfrag) > remaining:
                            changes.append(("equal", f"[… {len(cfrag)-remaining} more tokens truncated]"))
                    if hilites:
                        highlights_per_page[page2] = hilites
                    for k, v in sdelta.items():
                        summary[k] += v
                    page_changes_per_page[page2] = page_changes
                else:
                    summary["skipped_pages"] += 1
            elif page1 is not None:
                words1, _ = _extract_page_words(pdf1_bytes, page1)
                text = " ".join(words1)
                report_rows.append({
                    "page":        f"Page {page1 + 1}",
                    "pdf1_text":   text[:500] + ("…" if len(text) > 500 else ""),
                    "pdf2_text":   "— (page not in PDF 2)",
                    "change_type": "Deleted",
                })
                summary["deleted"] += 1
                page_changes_per_page[page1] = len(words1)
            elif page2 is not None:
                words2, rects2 = _extract_page_words(pdf2_bytes, page2)
                text = " ".join(words2)
                report_rows.append({
                    "page":        f"Page {page2 + 1}",
                    "pdf1_text":   "— (page not in PDF 1)",
                    "pdf2_text":   text[:500] + ("…" if len(text) > 500 else ""),
                    "change_type": "Inserted",
                })
                summary["inserted"] += 1
                page_changes_per_page[page2] = len(words2)
                highlights_per_page[page2] = [(rects2[k], COLOR_INSERT) for k in range(len(rects2))]

    regression_line = _fit_change_trend([(i + 1, page_changes_per_page[i]) for i in range(total)])
    if regression_line is not None:
        slope = regression_line["slope"]
        summary["trend_slope"] = slope
        summary["trend_intercept"] = regression_line["intercept"]
        if slope > 0:
            summary["trend_direction"] = "rising"
        elif slope < 0:
            summary["trend_direction"] = "falling"
        else:
            summary["trend_direction"] = "flat"

    log.info(
        "TOTAL %.2fs | pages=%d skipped=%d ins=%d mod=%d del=%d slope=%.4f workers=%d sections=%d title_match=%s",
        time.time() - t_total, total, summary["skipped_pages"],
        summary["inserted"], summary["modified"], summary["deleted"],
        summary["trend_slope"] if summary["trend_slope"] is not None else 0.0,
        MAX_WORKERS, summary["section_count"], summary["title_match"],
    )

    return report_rows, changes, summary, highlights_per_page


# ── Plain-text report ─────────────────────────────────────────────────────────

def build_report_txt(report_rows, pdf1_name, pdf2_name, summary):
    lines = [
        "=" * 80,
        "PDF COMPARISON REPORT",
        "=" * 80,
        f"PDF 1 (original)  : {pdf1_name}",
        f"PDF 2 (modified)  : {pdf2_name}",
        f"Title match        : {'Yes' if summary.get('title_match') else 'No'}",
        f"Sections           : {summary.get('section_count', '?')}",
        "-" * 80,
        f"Total pages       : {summary.get('total_pages', '?')}",
        f"Identical (skip)  : {summary.get('skipped_pages', 0)}",
        f"Inserted          : {summary['inserted']}",
        f"Modified          : {summary['modified']}",
        f"Deleted           : {summary['deleted']}",
        "=" * 80,
        "",
        f"{'PAGE':<10}{'PDF1 TEXT':<45}{'PDF2 TEXT':<45}{'CHANGE TYPE':<12}",
        "-" * 112,
    ]
    for row in report_rows:
        p1 = str(row["pdf1_text"])[:43]
        p2 = str(row["pdf2_text"])[:43]
        lines.append(f"{row['page']:<10}{p1:<45}{p2:<45}{row['change_type']:<12}")
    if summary.get("trend_slope") is not None:
        direction = "rising" if summary["trend_slope"] > 0 else "falling" if summary["trend_slope"] < 0 else "flat"
        lines += ["", "Change trend      : {} (slope={:.3f})".format(direction, summary["trend_slope"])]
    lines += ["", "=" * 80]
    return "\n".join(lines).encode("utf-8")

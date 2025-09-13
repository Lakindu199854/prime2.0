#!/usr/bin/env python3
import argparse, sys, re, json, difflib, csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
from lxml import etree

# =========================================================
# ===============  SHARED REGEXES & HELPERS  ==============
# =========================================================

# ---- list-start detectors (for both extraction and tree) ----
DECIMAL_RE         = re.compile(r"^\s*(\d+)\.\s+(.*)")
DECIMAL_START_ONLY = re.compile(r"^\s*\d+\.\s+")
ALPHA_RE           = re.compile(r"^\s*([a-z])\)\s+(.*)")
ALPHA_START_ONLY   = re.compile(r"^\s*[a-z]\)\s+", re.I)

RULE_RE            = re.compile(r"[_\-]{5,}\s*$")   # underline/HR lines
PAGE_NUM_RE        = re.compile(r"^\s*(\d{1,4})\s*$")
FOOTNOTE_LINE_RE   = re.compile(r"^\s*(\d+)\s+(.+)")

def looks_like_new_list_start(text: str) -> bool:
    t = (text or "").strip()
    return bool(DECIMAL_START_ONLY.match(t) or ALPHA_START_ONLY.match(t))

def font_base(font_name: str) -> str:
    f = (font_name or "").lower()
    for pat in ("-bold", "bold", "-bd", " bd", "-it", "italic", "oblique"):
        f = f.replace(pat, "")
    return f.strip()

def require_pymupdf():
    if fitz is None:
        print("ERROR: PyMuPDF (fitz) is not installed. Install with: pip install PyMuPDF", file=sys.stderr)
        sys.exit(2)

def is_bold_font(font_name: str) -> bool:
    f = (font_name or "").lower()
    return ("bold" in f) or f.endswith("-bd") or "-bd" in f or "bd" == f[-2:]

def is_italic_font(font_name: str) -> bool:
    f = (font_name or "").lower()
    return ("italic" in f) or ("oblique" in f) or f.endswith("-it") or "-it" in f

def most_common(items):
    if not items: return ""
    from collections import Counter
    return Counter(items).most_common(1)[0][0]

def median(nums):
    if not nums: return 0.0
    xs = sorted(nums)
    n = len(xs)
    mid = n // 2
    if n % 2: return xs[mid]
    return 0.5*(xs[mid-1] + xs[mid])

def allcaps_ratio(s):
    letters = [ch for ch in s if ch.isalpha()]
    if not letters: return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps / len(letters)

def normalize_text(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n(?=[a-zäöüß])", "", s)
    s = s.replace("\n", " ")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"\s+", " ", s).strip()
    return s

# --- helpers for inline superscript/subscript detection ---

def group_spans_into_lines(spans: List[Dict[str, Any]], y_tol: float = 2.0) -> List[List[Dict[str, Any]]]:
    """
    Group spans into visual text lines by their top y using a small tolerance.
    Assumes spans belong to a single block and are roughly top-to-bottom.
    """
    if not spans:
        return []
    # sort by y then x for stable grouping
    ss = sorted(spans, key=lambda s: (float(s.get("y", 0.0)), float(s.get("x", 0.0))))
    lines: List[List[Dict[str, Any]]] = []
    cur: List[Dict[str, Any]] = []
    cur_y: Optional[float] = None

    for s in ss:
        y = float(s.get("y", 0.0))
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur.append(s)
            # smoothed running baseline to absorb tiny jitter
            cur_y = y if cur_y is None else (cur_y + y) / 2.0
        else:
            lines.append(cur)
            cur = [s]
            cur_y = y

    if cur:
        lines.append(cur)
    return lines


def detect_sup_sub_for_block(spans: List[Dict[str, Any]]) -> List[Optional[str]]:
    """
    For each span: return 'sup', 'sub', or None based on unicode chars,
    size reduction, and vertical baseline shift within its visual line.
    """
    if not spans:
        return []

    SUPERSCRIPT_CHARS = {"¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹","⁰"}
    SUBSCRIPT_CHARS   = {"₁","₂","₃","₄","₅","₆","₇","₈","₉","₀"}

    # default font size if spans are missing size info
    all_sizes = [float(s.get("size") or s.get("h") or 0.0) for s in spans]
    default_size = median([x for x in all_sizes if x > 0]) or 10.0

    lines = group_spans_into_lines(spans, y_tol=2.0)
    idx_map = {id(s): i for i, s in enumerate(spans)}
    out: List[Optional[str]] = [None] * len(spans)

    soft_delta_ratio = 0.10  # baseline shift threshold relative to line size
    small_font_ratio = 0.90  # how much smaller counts as "reduced"

    for line in lines:
        # estimate a baseline and typical size for the line
        baselines, sizes = [], []
        for s in line:
            sz = float(s.get("size", 0.0)) or default_size
            y  = float(s.get("y", 0.0))
            ref = min(sz, default_size)
            baselines.append(y + ref * 0.8)  # rough baseline from top y
            sizes.append(sz)

        line_baseline = median(baselines)
        line_size = median([z for z in sizes if z > 0]) or default_size

        for s, sz, base in zip(line, sizes, baselines):
            i = idx_map[id(s)]
            txt = (s.get("text") or "").strip()
            delta = base - line_baseline
            tag: Optional[str] = None

            # Rule 1: explicit unicode script characters
            if txt in SUPERSCRIPT_CHARS:
                tag = "sup"
            elif txt in SUBSCRIPT_CHARS:
                tag = "sub"
            else:
                # Rule 2: reduced size + vertical shift
                if sz < small_font_ratio * line_size:
                    if delta < -soft_delta_ratio * line_size:
                        tag = "sup"
                    elif delta >  soft_delta_ratio * line_size:
                        tag = "sub"

            out[i] = tag

    return out


def stringify_runs(runs: List[Dict[str,Any]]) -> str:
    """Compact JSON for CSV cell: keep text + style flags + size. u: 0=normal,1=sup,2=sub"""
    try:
        light = []
        for r in runs:
            u = 1 if r.get("sup") == "sup" else (2 if r.get("sup") == "sub" else 0)
            light.append({
                "t": r.get("text",""),
                "b": 1 if r.get("bold") else 0,
                "i": 1 if r.get("italic") else 0,
                "u": u,
                "s": round(float(r.get("size", 0.0)), 2)
            })
        return json.dumps(light, ensure_ascii=False)
    except Exception:
        return ""

# =========================================================
# ====================  EXTRACTION PART  ==================
# =========================================================

def detect_pdf_start_page_by_footer(doc, target_printed: int, bottom_frac: float = 0.15) -> Optional[int]:
    """Return 0-based PDF page index whose bottom area contains printed page `target_printed`."""
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            h = float(page.rect.height)
            cutoff = h * (1.0 - bottom_frac)
            for b in (page.get_text("blocks") or []):
                if not (len(b) >= 5 and isinstance(b[4], str)):
                    continue
                _, y0, _, _, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
                if y0 >= cutoff:
                    t = normalize_text(txt)
                    if t == str(target_printed) or re.fullmatch(rf"\D*{target_printed}\D*", t):
                        return i
    except Exception:
        pass
    return None

def read_text_blocks(page):
    out = []
    try:
        for b in page.get_text("blocks") or []:
            if len(b) >= 5 and isinstance(b[4], str):
                x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
                if txt:
                    out.append({"x": x0, "y": y0, "w": x1-x0, "h": y1-y0, "text": txt})
    except Exception:
        pass
    return out

def read_rawdict(page):
    try:
        return page.get_text("rawdict") or {"blocks": []}
    except Exception:
        return {"blocks": []}

def iter_raw_spans(raw):
    for b in raw.get("blocks", []):
        if b.get("type", 0) == 0:
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    bbox = sp.get("bbox", [0,0,0,0])
                    yield {
                        "x": float(bbox[0]), "y": float(bbox[1]), "w": float(bbox[2]-bbox[0]), "h": float(bbox[3]-bbox[1]),
                        "cx": float((bbox[0]+bbox[2])/2), "cy": float((bbox[1]+bbox[3])/2),
                        "font": sp.get("font", ""), "size": float(sp.get("size", 0.0)),
                        "text": sp.get("text", ""), "flags": int(sp.get("flags", 0))
                    }
        elif b.get("type") == 1:
            bbox = b.get("bbox", [0,0,0,0])
            yield {"image": True, "x": float(bbox[0]), "y": float(bbox[1]), "w": float(bbox[2]-bbox[0]), "h": float(bbox[3]-bbox[1])}

def spans_in_block(spans, blk):
    x0, y0 = blk["x"], blk["y"]
    x1, y1 = x0 + blk["w"], y0 + blk["h"]
    out = []
    for sp in spans:
        if "image" in sp:
            continue
        cx, cy = sp["cx"], sp["cy"]
        if x0-1 <= cx <= x1+1 and y0-1 <= cy <= y1+1:
            out.append(sp)
    return out

def detect_body_size(blocks):
    sizes = [b.get("size", 0.0) for b in blocks if len((b.get("text") or "")) > 40 and b.get("size",0)>0]
    if not sizes:
        sizes = [b.get("size", 0.0) for b in blocks if b.get("size",0)>0]
    xs = sorted(sizes); n = len(xs)
    if not n: return 11.0
    mid = n // 2
    return xs[mid] if n % 2 else 0.5*(xs[mid-1] + xs[mid])

def score_heading(b, page_w, body_size, thresh_regex_bonus=True):
    import re
    score = 0.0
    size = b.get("size", 0.0)
    font = b.get("font", "")
    text = b.get("text", "")
    centered = abs(((b["x"] + b["w"]/2) - (page_w/2))) < 20
    gap_above = b.get("_gap_above", 0.0)
    if size >= 1.30 * body_size: score += 2.0
    if "Bold" in font or "Bd" in font or "bold" in font: score += 1.0
    if centered: score += 1.0
    if gap_above >= 12: score += 1.0
    if allcaps_ratio(text) >= 0.6: score += 0.5
    if thresh_regex_bonus and re.search(r"^(§\s*\d+|Einleitung|Inhaltsverzeichnis)$", text): score += 2.0
    if len(text) > 150: score -= 1.0
    return score

def assign_heading_levels(blocks):
    sizes = sorted({b["size"] for b in blocks if b.get("kind")=="heading" and b.get("size",0)>0}, reverse=True)
    tier = {sz: i+1 for i, sz in enumerate(sizes)}
    for b in blocks:
        if b.get("kind") == "heading":
            b["level"] = tier.get(b["size"], 1)

def similarity(a, b):
    a = (a or "").strip().lower()
    b = (b or "").strip().lower()
    return difflib.SequenceMatcher(a=a, b=b).ratio()

def find_anchor(blocks, start_page, start_heading, anchor_tolerance):
    if not start_heading:
        for i, b in enumerate(blocks):
            if b["page_index"]+1 >= start_page and b.get("kind")=="heading":
                return i, b.get("level", 1), 1.0
        return -1, None, 0.0

    best_i = -1
    best_sim = 0.0
    for i, b in enumerate(blocks):
        if b["page_index"]+1 < start_page: 
            continue
        if b.get("kind") != "heading":
            continue
        sim = similarity(start_heading, b["text"])
        if sim > best_sim:
            best_sim, best_i = sim, i
    if best_i >= 0 and best_sim >= anchor_tolerance:
        return best_i, blocks[best_i].get("level", 1), best_sim
    return -1, None, best_sim

def page_sort_key(b):
    return (b["page_index"], b["y"], b["x"])

def merge_flow_blocks(blocks, gap_px=6.0, size_tol=0.6, x_slop=40.0):
    """Merge adjacent text-body blocks that visually form one paragraph/list item."""
    if not blocks:
        return blocks
    blocks = sorted(blocks, key=page_sort_key)
    out = []; i = 0
    while i < len(blocks):
        cur = blocks[i]; i += 1
        if cur.get("kind") != "text" or cur.get("role") != "body":
            out.append(cur); continue
        while i < len(blocks):
            nxt = blocks[i]
            if not (nxt.get("kind") == "text" and nxt.get("role") == "body" and
                    nxt["page_index"] == cur["page_index"]):
                break
            if looks_like_new_list_start(nxt.get("text","")):
                break
            gap = max(0.0, nxt["y"] - (cur["y"] + cur["h"]))
            same_column = (abs(nxt["x"] - cur["x"]) <= x_slop) or (nxt["x"] >= cur["x"])
            if gap > gap_px or not same_column:
                break
            fs_cur, fs_nxt = float(cur.get("size") or 0.0), float(nxt.get("size") or 0.0)
            if fs_cur and fs_nxt and abs(fs_cur - fs_nxt) > size_tol:
                break
            if font_base(cur.get("font","")) and font_base(nxt.get("font","")) and \
               font_base(cur.get("font","")) != font_base(nxt.get("font","")):
                break
            i += 1
            t1 = (cur.get("text","") or "").rstrip()
            t2 = (nxt.get("text","") or "").lstrip()
            if t1.endswith("-") and (t2[:1].islower() or t2[:1].isdigit()):
                joined = t1[:-1] + t2
            else:
                joined = (t1 + " " + t2).strip()
            cur["text"] = normalize_text(joined)
            r1 = cur.get("runs") or []
            r2 = nxt.get("runs") or []
            cur["runs"] = (r1 + r2) if (r1 or r2) else []
            x0 = min(cur["x"], nxt["x"]); y0 = min(cur["y"], nxt["y"])
            x1 = max(cur["x"] + cur["w"], nxt["x"] + nxt["w"])
            y1 = max(cur["y"] + cur["h"], nxt["y"] + nxt["h"])
            cur.update({"x": x0, "y": y0, "w": x1 - x0, "h": y1 - y0})
        out.append(cur)
    compute_gap_above(out)
    return out

def compute_gap_above(sorted_blocks):
    prev = None
    for b in sorted_blocks:
        if prev is None or prev["page_index"] != b["page_index"]:
            b["_gap_above"] = 999.0
        else:
            b["_gap_above"] = max(0.0, b["y"] - (prev["y"] + prev["h"]))
        prev = b

def extract_start_from_metadata(meta_path: Path):
    """Return (start_printed:int|None, start_heading:str|None) from metadata XML."""
    start_printed = None; start_heading = None
    try:
        tree = etree.parse(str(meta_path))
        vb = tree.find(".//verkblatt")
        if vb is not None and vb.get("seite"):
            try:
                start_printed = int(vb.get("seite"))
            except Exception:
                start_printed = None
        h = tree.find(".//rumpf//einzelvorschrift/ueberschrift")
        if h is not None and (h.text or "").strip():
            start_heading = h.text.strip()
        else:
            h2 = tree.find(".//kopf/erstfassung/normueberschrift")
            if h2 is not None and (h2.text or "").strip():
                start_heading = h2.text.strip()
    except Exception:
        pass
    return start_printed, start_heading

def mark_headers_footers(
    blocks: List[Dict[str, Any]],
    page_heights: Dict[int, float],
    header_frac: float = 0.10,
    footer_frac: float = 0.12,
    min_ratio: float = 0.6,
):
    """Mark repeating text near the top as headers and near the bottom as footers."""
    if not blocks:
        return
    def norm_key(t: str) -> str:
        t = (t or "").strip()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\b\d+\b", "<NUM>", t)
        return t.lower()
    page_count = max(1, len({b["page_index"] for b in blocks}))
    top_counts: Dict[str, int] = {}
    bot_counts: Dict[str, int] = {}
    for b in blocks:
        h = float(page_heights.get(b["page_index"], 842.0))
        top_y = h * header_frac
        bot_y = h * (1.0 - footer_frac)
        raw = (b.get("text", "") or "").strip()
        if not raw:
            continue
        key = norm_key(raw)
        if b["y"] < top_y:
            top_counts[key] = top_counts.get(key, 0) + 1
        if (b["y"] + b["h"]) > bot_y:
            bot_counts[key] = bot_counts.get(key, 0) + 1
    top_common = {k for k, c in top_counts.items() if (c / page_count) >= min_ratio}
    bot_common = {k for k, c in bot_counts.items() if (c / page_count) >= min_ratio}
    for b in blocks:
        h = float(page_heights.get(b["page_index"], 842.0))
        top_y = h * header_frac
        bot_y = h * (1.0 - footer_frac)
        raw = (b.get("text", "") or "").strip()
        key = norm_key(raw)
        is_pure_page_num = bool(re.fullmatch(r"\d{1,4}", raw))
        at_top = b["y"] < top_y
        at_bottom = (b["y"] + b["h"]) > bot_y
        if at_top and key in top_common and not is_pure_page_num:
            b["role"] = "header"
        elif at_bottom and (key in bot_common or is_pure_page_num):
            b["role"] = "footer"
        else:
            b.setdefault("role", "body")

def summarize(blocks, tables, images, start_page, stop_page_incl, printed_start=None, printed_first=None):
    in_range_blocks = [b for b in blocks if start_page-1 <= b["page_index"] <= stop_page_incl]
    heads = [b for b in in_range_blocks if b.get("kind")=="heading"]
    return {
        "pdf_pages": f"{start_page}..{stop_page_incl+1}",
        "printed_start": printed_start,
        "printed_first": printed_first,
        "blocks": len(in_range_blocks),
        "headings": len(heads),
        "tables": len([t for t in tables if start_page-1 <= t['page_index'] <= stop_page_incl]),
        "images": len([im for im in images if start_page-1 <= im['page_index'] <= stop_page_incl]),
        "h1_candidates": len([b for b in heads if b.get('level')==1])
    }

# =========================================================
# ======= BUILD INTERMEDIATE (returns XML root + meta) ====
# =========================================================

def build_intermediate(
    pdf_path: Path, metadata_path: Optional[Path], start_page_cli: Optional[int],
    start_heading_cli: Optional[str], lang: str, heading_threshold: float,
    anchor_tolerance: float, max_pages: Optional[int] = None, no_tables: bool = False,
    debug_csv: Optional[Path] = None, debug_jsonl: Optional[Path] = None,
    start_printed: Optional[int] = None, printed_first_cli: Optional[int] = None,
    infer_printed: bool = True, emit_spans: bool = False
) -> Tuple[etree._Element, Dict[str, Any]]:
    """Build the intermediate XML in memory and return (root, summary)."""
    require_pymupdf()
    doc = fitz.open(pdf_path.as_posix())

    # metadata-derived start
    meta_start_printed = None; meta_start_heading = None
    if metadata_path and metadata_path.exists():
        meta_start_printed, meta_start_heading = extract_start_from_metadata(metadata_path)

    start_page = start_page_cli
    start_heading = start_heading_cli or meta_start_heading
    printed_first = printed_first_cli

    # If not given a PDF page, derive strictly from metadata printed page by scanning footer
    start_printed_effective = (start_printed if start_printed is not None else meta_start_printed)
    if start_page is None and start_printed_effective is not None:
        idx = detect_pdf_start_page_by_footer(doc, start_printed_effective)
        if idx is not None:
            start_page = idx + 1
        else:
            if printed_first is None and infer_printed:
                m = re.search(r"_(\d{2,4})_(\d{2,4})", pdf_path.name)
                if m:
                    try:
                        printed_first = int(m.group(1))
                    except Exception:
                        printed_first = None
            if printed_first is not None:
                start_page = (start_printed_effective - printed_first + 1)
                if start_page < 1 or start_page > doc.page_count:
                    print(f"WARNING: Computed start-page {start_page} out of PDF bounds [1..{doc.page_count}].", file=sys.stderr)
            else:
                print("WARNING: Could not determine the printed-first page; pass --printed-first or disable inference.", file=sys.stderr)

    if start_page is None:
        start_page = 1

    all_blocks: List[Dict[str, Any]] = []
    all_tables: List[Dict[str, Any]] = []
    all_images: List[Dict[str, Any]] = []
    page_heights: Dict[int, float] = {}

    last_page_index = doc.page_count - 1
    if max_pages is not None:
        last_page_index = min(last_page_index, (start_page-1) + max_pages - 1)

    for i in range(start_page-1, last_page_index+1):
        page = doc.load_page(i)
        w, h = page.rect.width, page.rect.height
        page_heights[i] = float(h)

        blk_list = read_text_blocks(page)
        raw = read_rawdict(page)
        spans = list(iter_raw_spans(raw))

        for blk in blk_list:
            ss = spans_in_block(spans, blk)
            runs = []
            if ss:
                tmp_runs = []
                for s in ss:
                    r = {
                        "text": s.get("text",""),
                        "font": s.get("font",""),
                        "size": float(s.get("size",0.0) or 0.0),
                        "x": float(s.get("x",0.0)), "y": float(s.get("y",0.0)),
                        "w": float(s.get("w",0.0)), "h": float(s.get("h",0.0)),
                    }
                    r["bold"] = is_bold_font(r["font"])
                    r["italic"] = is_italic_font(r["font"])
                    tmp_runs.append(r)
                tmp_runs.sort(key=lambda r: (r["y"], r["x"]))
                sup_tags = detect_sup_sub_for_block(tmp_runs)
                for r, tag in zip(tmp_runs, sup_tags):
                    r["sup"] = tag
                runs = tmp_runs
                blk["font"] = most_common([r["font"] for r in runs if r.get("font")])
                blk["size"] = median([r["size"] for r in runs if r.get("size")])
            else:
                blk["font"] = ""
                blk["size"] = 0.0

            blk["runs"] = runs
            blk["text"] = normalize_text(blk.get("text", ""))

            if not blk["text"]:
                if runs:
                    try:
                        joined = " ".join((r.get("text", "") or "").strip()
                                          for r in sorted(runs, key=lambda r: (float(r.get("y",0.0)), float(r.get("x",0.0)))))
                        blk["text"] = normalize_text(joined)
                    except Exception:
                        pass
                if not blk["text"]:
                    try:
                        rect = fitz.Rect(blk["x"], blk["y"], blk["x"] + blk["w"], blk["y"] + blk["h"])
                        fallback = page.get_textbox(rect) or ""
                        blk["text"] = normalize_text(fallback)
                    except Exception:
                        pass

            blk["page_w"], blk["page_h"] = w, h
            blk["page_index"] = i
            all_blocks.append(blk)

        for sp in spans:
            if sp.get("image"):
                img = dict(sp); img["page_index"] = i
                all_images.append(img)

        if not no_tables:
            try:
                if hasattr(page, "find_tables"):
                    tf = page.find_tables()
                    tables = getattr(tf, "tables", []) if tf else []
                    for t in tables:
                        bbox = getattr(t, "bbox", None) or getattr(t, "rect", None)
                        if bbox:
                            x0, y0, x1, y1 = bbox
                            tbl = {"x": float(x0), "y": float(y0), "w": float(x1-x0), "h": float(y1-y0),
                                   "rows": getattr(t, "row_count", getattr(t, "nrows", 0)),
                                   "cols": getattr(t, "col_count", getattr(t, "ncols", 0)),
                                   "page_index": i}
                            try:
                                grid = t.extract(); tbl["grid"] = grid
                            except Exception:
                                pass
                            all_tables.append(tbl)
            except Exception:
                pass

    # Order and compute spacing / classify headings
    all_blocks.sort(key=page_sort_key)
    compute_gap_above(all_blocks)

    body_size = detect_body_size(all_blocks)
    for b in all_blocks:
        score = score_heading(b, b["page_w"], body_size)
        b["_score"] = score
        b["kind"] = "heading" if score >= heading_threshold else "text"
        b["level"] = 0
        b.setdefault("role", "body")
    assign_heading_levels(all_blocks)

    # mark headers/footers
    mark_headers_footers(all_blocks, page_heights)

    # Anchor range
    anchor_idx, anchor_level, sim = find_anchor(all_blocks, start_page, start_heading, anchor_tolerance)
    if anchor_idx < 0 and start_heading:
        print(f"WARNING: Anchor heading not found (best similarity={sim:.2f}). Starting at page {start_page}.", file=sys.stderr)

    stop_page_incl = last_page_index
    bottom_clip_page = None; bottom_clip_y = None
    if anchor_idx >= 0 and anchor_level is not None:
        for j in range(anchor_idx + 1, len(all_blocks)):
            if all_blocks[j]["kind"] == "heading" and all_blocks[j].get("level", 99) <= anchor_level:
                txtj = all_blocks[j].get("text", "")
                if re.match(r"^§\s*\d+", txtj):
                    continue
                stop_page_incl = all_blocks[j]["page_index"]
                bottom_clip_page = all_blocks[j]["page_index"]
                bottom_clip_y = all_blocks[j]["y"]
                break

    top_clip_page = None; top_clip_y = None
    if anchor_idx >= 0:
        anchor_block = all_blocks[anchor_idx]
        top_clip_page = anchor_block["page_index"]
        top_clip_y = anchor_block["y"]

    sel_blocks: List[Dict[str, Any]] = []
    sel_tables: List[Dict[str, Any]] = []
    sel_images: List[Dict[str, Any]] = []

    for b in all_blocks:
        pi = b["page_index"]
        if pi < start_page-1 or pi > stop_page_incl:
            continue
        if top_clip_page is not None and pi == top_clip_page and top_clip_y is not None and b["y"] < top_clip_y - 0.5 and b.get("role") != "header":
            continue
        if bottom_clip_page is not None and pi == bottom_clip_page and bottom_clip_y is not None and b["y"] >= bottom_clip_y - 0.5:
            continue
        sel_blocks.append(b)

    for t in all_tables:
        pi = t["page_index"]
        if pi < start_page-1 or pi > stop_page_incl:
            continue
        if top_clip_page is not None and pi == top_clip_page and top_clip_y is not None and t["y"] < top_clip_y - 0.5:
            continue
        if bottom_clip_page is not None and pi == bottom_clip_page and bottom_clip_y is not None and t["y"] >= bottom_clip_y - 0.5:
            continue
        sel_tables.append(t)

    for im in all_images:
        pi = im["page_index"]
        if pi < start_page-1 or pi > stop_page_incl:
            continue
        if top_clip_page is not None and pi == top_clip_page and top_clip_y is not None and im["y"] < top_clip_y - 0.5:
            continue
        if bottom_clip_page is not None and pi == bottom_clip_page and bottom_clip_y is not None and im["y"] >= bottom_clip_y - 0.5:
            continue
        sel_images.append(im)

    # merge wrapped lines
    sel_blocks = merge_flow_blocks(sel_blocks, gap_px=6.0, size_tol=0.6, x_slop=40.0)

    # Debug CSV/JSONL
    if debug_csv:
        with open(debug_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f, delimiter=";")
            wcsv.writerow(["page","x","y","w","h","font","size","score","kind","level","role","text","runs_json"])
            for b in sel_blocks:
                wcsv.writerow([
                    b["page_index"]+1,
                    f"{b['x']:.2f}", f"{b['y']:.2f}", f"{b['w']:.2f}", f"{b['h']:.2f}",
                    b.get("font",""),
                    f"{b.get('size',0):.1f}",
                    f"{b.get('_score',0):.2f}",
                    b.get("kind","text"),
                    b.get("level",0),
                    b.get("role","body"),
                    (b.get("text","")[:200]).replace("\n"," "),
                    stringify_runs(b.get("runs", []))
                ])
    if debug_jsonl:
        with open(debug_jsonl, "w", encoding="utf-8") as f:
            for b in sel_blocks:
                f.write(json.dumps(b, ensure_ascii=False) + "\n")

    # ---------- Build INTERMEDIATE XML (in memory) ----------
    root = etree.Element("doc")
    root.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
    meta = etree.SubElement(root, "meta")
    etree.SubElement(meta, "source").text = pdf_path.name
    start_el = etree.SubElement(meta, "start")
    start_el.set("page", str(start_page))
    if start_heading:
        start_el.set("heading", start_heading)
    summary_obj = summarize(sel_blocks, sel_tables, sel_images, start_page, stop_page_incl,
                            printed_start=start_printed_effective, printed_first=printed_first)
    etree.SubElement(meta, "summary").text = json.dumps(summary_obj, ensure_ascii=False)

    pages: Dict[int, etree._Element] = {}
    for i in range(start_page-1, stop_page_incl+1):
        page = fitz.open(pdf_path.as_posix()).load_page(i) if fitz else None
        width = f"{page.rect.width:.2f}" if page else "0"
        height = f"{page.rect.height:.2f}" if page else "0"
        page_el = etree.SubElement(root, "page", index=str(i+1), width=width, height=height)
        pages[i] = page_el

    for b in sel_blocks:
        page_el = pages.get(b["page_index"])
        if page_el is None:
            continue
        attrs = {
            "kind": b.get("kind","text"),
            "x": f"{b['x']:.2f}", "y": f"{b['y']:.2f}",
            "w": f"{b['w']:.2f}", "h": f"{b['h']:.2f}"
        }
        if b.get("font"):
            attrs["font"] = b["font"]
        if b.get("size",0) > 0:
            attrs["size"] = f"{b['size']:.1f}"
        if b.get("kind")=="heading":
            attrs["level"] = str(b.get("level", 1))
        if b.get("role"):
            attrs["role"] = b.get("role")
        el = etree.SubElement(page_el, "block", **attrs)
        bt = b.get("text", "") or ""
        if (not bt) and b.get("runs"):
            try:
                bt = normalize_text(" ".join((r.get("text", "") or "").strip()
                                             for r in sorted(b.get("runs"), key=lambda r: (float(r.get("y",0.0)), float(r.get("x",0.0))))))
            except Exception:
                bt = ""
        el.text = bt

        if emit_spans and b.get("runs"):
            for r in b["runs"]:
                span_el = etree.SubElement(el, "span",
                    bold="1" if r.get("bold") else "0",
                    italic="1" if r.get("italic") else "0",
                    sup=("1" if r.get("sup")=="sup" else ("-1" if r.get("sup")=="sub" else "0")),
                    size=f"{float(r.get('size',0.0)):.2f}",
                    x=f"{float(r.get('x',0.0)):.2f}",
                    y=f"{float(r.get('y',0.0)):.2f}",
                    w=f"{float(r.get('w',0.0)):.2f}",
                    h=f"{float(r.get('h',0.0)):.2f}",
                    font=r.get("font","")
                )
                span_el.text = r.get("text","")

    for t in sel_tables:
        page_el = pages.get(t["page_index"])
        if page_el is None:
            continue
        tbl_attrs = {"x": f"{t['x']:.2f}", "y": f"{t['y']:.2f}",
                     "w": f"{t['w']:.2f}", "h": f"{t['h']:.2f}",
                     "rows": str(t.get("rows", 0)), "cols": str(t.get("cols", 0))}
        tbl_el = etree.SubElement(page_el, "table", **tbl_attrs)
        grid = t.get("grid")
        if grid and isinstance(grid, list):
            for row in grid:
                tr = etree.SubElement(tbl_el, "tr")
                for cell in row:
                    td = etree.SubElement(tr, "td")
                    if cell is None:
                        continue
                    td.text = normalize_text(str(cell))

    for im in sel_images:
        page_el = pages.get(im["page_index"])
        if page_el is None:
            continue
        etree.SubElement(page_el, "image", x=f"{im['x']:.2f}", y=f"{im['y']:.2f}", w=f"{im['w']:.2f}", h=f"{im['h']:.2f}")

    return root, summary_obj

# =========================================================
# ===============  TREE-BUILDER (from root)  ==============
# =========================================================

def parse_summary_meta(intermediate_root) -> Tuple[Optional[int], Optional[int]]:
    s_el = intermediate_root.find("./meta/summary")
    if s_el is None or not (s_el.text or "").strip():
        return None, None
    try:
        j = json.loads(s_el.text)
        return (int(j.get("printed_start")) if j.get("printed_start") else None,
                int(j.get("printed_first")) if j.get("printed_first") else None)
    except Exception:
        return None, None

def get_page_blocks(page_el):
    for b in page_el.findall("./block"):
        yield b

def block_text(b: etree._Element) -> str:
    return (b.text or "").strip()

def is_footer_block(b: etree._Element) -> bool:
    return (b.get("role") == "footer")

def is_header_block(b: etree._Element) -> bool:
    return (b.get("role") == "header")

def is_rule_block(b: etree._Element) -> bool:
    return bool(RULE_RE.search(block_text(b)))

def extract_sup_markers(b: etree._Element) -> List[str]:
    out: List[str] = []
    for sp in b.findall("./span"):
        if sp.get("sup") in ("1", "sup"):
            t = (sp.text or "").strip()
            if re.fullmatch(r"[0-9]", t):
                out.append(t)
    return out

def detect_bottom_footnotes(page_el: etree._Element, page_h: float) -> Dict[str, Dict[str,str]]:
    notes: Dict[str, Dict[str,str]] = {}
    try:
        h_frac = 0.82
        for b in page_el.findall("./block"):
            y = float(b.get("y", "0") or 0.0)
            h = float(b.get("h", "0") or 0.0)
            txt = block_text(b)
            if not txt:
                continue
            in_bottom_band = ((y + h) >= page_h * h_frac)
            m = FOOTNOTE_LINE_RE.match(txt)
            if (b.get("role") == "footer" and m) or (in_bottom_band and m):
                marker = m.group(1)
                notes[marker] = {
                    "text": m.group(2).strip(),
                    "x": b.get("x","0"), "y": b.get("y","0"),
                    "w": b.get("w","0"), "h": b.get("h","0")
                }
    except Exception:
        pass
    return notes

def extract_footer_page_number(blocks: List[etree._Element]) -> Optional[int]:
    for b in blocks:
        if is_footer_block(b):
            m = PAGE_NUM_RE.match(block_text(b))
            if m:
                try:
                    return int(m.group(1))
                except Exception:
                    return None
    return None

def ensure_page_node(doc_pages_el: etree._Element, printed_index: int,
                     width: Optional[str], height: Optional[str]) -> etree._Element:
    existing = doc_pages_el.find(f"./page[@index='{printed_index}']")
    if existing is not None:
        return existing
    page_el = etree.SubElement(doc_pages_el, "page", index=str(printed_index))
    if width:  page_el.set("width", width)
    if height: page_el.set("height", height)
    etree.SubElement(page_el, "body")
    etree.SubElement(page_el, "footer")
    return page_el

def add_separator(body_el: etree._Element, src_block: etree._Element):
    etree.SubElement(
        body_el, "separator",
        kind="rule",
        x=src_block.get("x","0"), y=src_block.get("y","0"),
        w=src_block.get("w","0"), h=src_block.get("h","0")
    )

def copy_block_coords_and_style(dst: etree._Element, src_block: etree._Element):
    for attr in ("x","y","w","h","font","size","role"):
        val = src_block.get(attr)
        if val is not None:
            dst.set(attr, val)

def copy_inline_spans(dst_parent: etree._Element, src_block: etree._Element):
    spans = src_block.findall("./span")
    if not spans:
        return
    spans_have_text = any((sp.text or "").strip() for sp in spans)
    if spans_have_text:
        dst_parent.text = None
    for sp in spans:
        out = etree.SubElement(
            dst_parent, "span",
            bold=sp.get("bold","0"),
            italic=sp.get("italic","0"),
            sup=sp.get("sup","0"),
            size=sp.get("size","0"),
            x=sp.get("x","0"), y=sp.get("y","0"),
            w=sp.get("w","0"), h=sp.get("h","0"),
            font=(sp.get("font") or "")
        )
        out.text = sp.text or ""

def bbox_union(items: List[etree._Element]) -> Optional[Tuple[float,float,float,float]]:
    xs = []
    for it in items:
        try:
            x = float(it.get("x","0")); y = float(it.get("y","0"))
            w = float(it.get("w","0")); h = float(it.get("h","0"))
            xs.append((x,y,w,h))
        except Exception:
            pass
    if not xs:
        return None
    x0 = min(x for x,_,_,_ in xs)
    y0 = min(y for _,y,_,_ in xs)
    x1 = max(x+w for x,_,w,_ in xs)
    y1 = max(y+h for _,y,_,h in xs)
    return (x0,y0,x1-x0,y1-y0)

class ListState:
    def __init__(self):
        self.decimal_open = False
        self.decimal_start: Optional[int] = None
        self.alpha_open = False
        self.alpha_start: Optional[int] = None
    def close_alpha(self):
        self.alpha_open = False
        self.alpha_start = None
    def close_decimal(self):
        self.close_alpha()
        self.decimal_open = False
        self.decimal_start = None

def build_tree_from_intermediate(intermediate_root: etree._Element, out_xml: Path):
    printed_start, printed_first = parse_summary_meta(intermediate_root)

    out_root = etree.Element("document")
    out_meta = etree.SubElement(out_root, "meta")
    etree.SubElement(out_meta, "source").text = (intermediate_root.findtext("./meta/source") or "")
    out_pages = etree.SubElement(out_root, "pages")

    for page_el in intermediate_root.findall("./page"):
        pdf_index = int(page_el.get("index"))
        page_w = page_el.get("width")
        page_h = page_el.get("height")
        pdf_h = float(page_h or "0")

        footer_blocks = [b for b in get_page_blocks(page_el) if is_footer_block(b)]
        printed = extract_footer_page_number(footer_blocks)
        if printed is None and printed_first is not None:
            printed = printed_first + (pdf_index - 1)
        if printed is None:
            printed = pdf_index

        out_page = ensure_page_node(out_pages, printed, page_w, page_h)
        body = out_page.find("./body")
        footer = out_page.find("./footer")

        page_notes = detect_bottom_footnotes(page_el, pdf_h)
        heading_markers_on_page: List[str] = []

        state = ListState()
        current_decimal_list_el = None
        current_alpha_list_el = None

        for b in [x for x in get_page_blocks(page_el) if not is_header_block(x)]:
            kind = b.get("kind"); role = b.get("role")
            txt = block_text(b)
            if not txt:
                continue
            if role == "footer" and PAGE_NUM_RE.match(txt):
                continue

            if is_rule_block(b):
                add_separator(body, b); continue

            if kind == "heading":
                heading_markers_on_page.extend(extract_sup_markers(b))
                h_el = etree.SubElement(body, "heading", level=b.get("level", "1"))
                copy_block_coords_and_style(h_el, b)
                h_el.text = txt
                copy_inline_spans(h_el, b)
                state.close_decimal()
                current_decimal_list_el = None
                current_alpha_list_el = None
                continue

            m_dec = DECIMAL_RE.match(txt)
            m_alp = ALPHA_RE.match(txt)

            if m_dec:
                num = int(m_dec.group(1))
                content = m_dec.group(2).strip()
                if not state.decimal_open:
                    current_decimal_list_el = etree.SubElement(body, "list", type="ordered", marker="decimal")
                    state.decimal_open = True
                    state.decimal_start = num
                    current_decimal_list_el.set("start", str(num))
                else:
                    if state.decimal_start is None:
                        current_decimal_list_el.set("start", str(num))
                        state.decimal_start = num
                it = etree.SubElement(current_decimal_list_el, "item")
                copy_block_coords_and_style(it, b)
                p = etree.SubElement(it, "paragraph")
                copy_block_coords_and_style(p, b)
                p.text = content
                state.close_alpha()
                current_alpha_list_el = None
                continue

            if m_alp:
                alpha = m_alp.group(1)
                content = m_alp.group(2).strip()
                parent = None
                if state.decimal_open and current_decimal_list_el is not None:
                    items = current_decimal_list_el.findall("./item")
                    if items:
                        parent = items[-1]
                if not state.alpha_open or current_alpha_list_el is None:
                    list_parent = parent if parent is not None else body
                    current_alpha_list_el = etree.SubElement(list_parent, "list", type="ordered", marker="lower-alpha")
                    start = ord(alpha) - ord('a') + 1
                    current_alpha_list_el.set("start", str(start))
                    state.alpha_open = True
                    state.alpha_start = start
                it = etree.SubElement(current_alpha_list_el, "item")
                copy_block_coords_and_style(it, b)
                p = etree.SubElement(it, "paragraph")
                copy_block_coords_and_style(p, b)
                p.text = content
                continue

            para = etree.SubElement(body, "paragraph")
            copy_block_coords_and_style(para, b)
            para.text = txt
            copy_inline_spans(para, b)

        pn = etree.SubElement(footer, "page-number")
        pn.text = str(printed)

        if page_notes:
            for mk in heading_markers_on_page:
                if mk in page_notes:
                    info = page_notes[mk]
                    fn = etree.SubElement(body, "footnote", marker=mk,
                        x=info.get("x","0"), y=info.get("y","0"),
                        w=info.get("w","0"), h=info.get("h","0"))
                    fn.text = info.get("text","")

        for lst in body.findall(".//list[@type='ordered']"):
            st = lst.get("start")
            if st and st.isdigit() and int(st) > 1:
                lst.set("continue", "true")
            items = lst.findall("./item")
            bb = bbox_union(items)
            if bb:
                x,y,w,h = bb
                lst.set("x", f"{x:.2f}"); lst.set("y", f"{y:.2f}")
                lst.set("w", f"{w:.2f}"); lst.set("h", f"{h:.2f}")

    etree.ElementTree(out_root).write(str(out_xml), encoding="utf-8", xml_declaration=True, pretty_print=True)

# =========================================================
# =========================== CLI =========================
# =========================================================

def main():
    ap = argparse.ArgumentParser(
        description="Extract from PDF and build tree-structured XML (single script)."
    )
    ap.add_argument("pdf", type=Path, help="Input PDF")
    ap.add_argument("--metadata", type=Path, default=None, help="Metadata XML (optional; auto-reads start page/heading)")
    ap.add_argument("--out-tree", type=Path, default=Path("out_tree.xml"), help="Output path for tree XML")
    ap.add_argument("--save-intermediate", type=Path, default=None, help="If set, also write intermediate XML here")

    # Start position: either PDF page or printed page (auto-inferred first printed)
    ap.add_argument("--start-page", type=int, default=None, help="Start page number (PDF, 1-based)")
    ap.add_argument("--start-printed", type=int, default=None, help="Start printed page (journal page, e.g., 229)")
    ap.add_argument("--printed-first", type=int, default=None, help="Printed page number of PDF's first page (e.g., 213)")
    ap.add_argument("--no-infer-printed", action="store_true", help="Disable first-printed inference from filename *_213_257*")

    ap.add_argument("--start-heading", type=str, default=None, help="Override anchor heading (otherwise from metadata)")
    ap.add_argument("--lang", type=str, default="de", help="xml:lang, default 'de'")
    ap.add_argument("--heading-threshold", type=float, default=3.0, help="Score ≥ threshold ⇒ heading (default 3.0)")
    ap.add_argument("--anchor-tolerance", type=float, default=0.75, help="Fuzzy match similarity 0..1 (default 0.75)")
    ap.add_argument("--max-pages", type=int, default=None, help="Process at most N pages starting at start page")
    ap.add_argument("--no-tables", action="store_true", help="Disable table detection")
    ap.add_argument("--debug-csv", type=Path, default=None, help="Write blocks preview CSV (semicolon-separated)")
    ap.add_argument("--debug-jsonl", type=Path, default=None, help="Write blocks as JSON Lines")
    ap.add_argument("--emit-spans", action="store_true", help="Emit <span> with bold/italic/sup/sub in intermediate XML")
    args = ap.parse_args()

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    infer_printed = not args.no_infer_printed

    # 1) Build intermediate in-memory
    intermediate_root, summary = build_intermediate(
        pdf_path=args.pdf,
        metadata_path=args.metadata,
        start_page_cli=args.start_page,
        start_heading_cli=args.start_heading,
        lang=args.lang,
        heading_threshold=args.heading_threshold,
        anchor_tolerance=args.anchor_tolerance,
        max_pages=args.max_pages,
        no_tables=args.no_tables,
        debug_csv=args.debug_csv,
        debug_jsonl=args.debug_jsonl,
        start_printed=args.start_printed,
        printed_first_cli=args.printed_first,
        infer_printed=infer_printed,
        emit_spans=args.emit_spans
    )

    # (optional) write intermediate to disk
    if args.save_intermediate:
        etree.ElementTree(intermediate_root).write(
            str(args.save_intermediate), encoding="utf-8", xml_declaration=True, pretty_print=True
        )

    # 2) Build tree XML from the in-memory intermediate
    build_tree_from_intermediate(intermediate_root, args.out_tree)

    # final summary to stdout (same style as before, add target path)
    print(json.dumps({"wrote_tree": str(args.out_tree), **summary}, ensure_ascii=False))

if __name__ == "__main__":
    main()

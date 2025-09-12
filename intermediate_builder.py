
#!/usr/bin/env python3
import argparse, sys, re, json, difflib, csv
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None
from lxml import etree



def detect_pdf_start_page_by_footer(doc, target_printed: int, bottom_frac: float = 0.15) -> Optional[int]:
    """
    Return 0-based PDF page index whose bottom area contains the printed page number `target_printed`.
    Scans all pages; works even when the filename doesn't encode printed-first.
    """
    try:
        for i in range(doc.page_count):
            page = doc.load_page(i)
            h = float(page.rect.height)
            cutoff = h * (1.0 - bottom_frac)

            for b in (page.get_text("blocks") or []):
                if not (len(b) >= 5 and isinstance(b[4], str)):
                    continue
                x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
                if y0 >= cutoff:
                    t = normalize_text(txt)
                    # exact match or surrounded by non-digits
                    if t == str(target_printed) or re.fullmatch(rf"\D*{target_printed}\D*", t):
                        return i
    except Exception:
        pass
    return None


# ---------------------- Utilities for styles ----------------------

# ---- list-start detectors (don't merge across these) ----
DECIMAL_RE = re.compile(r"^\s*\d+\.\s+")
ALPHA_RE   = re.compile(r"^\s*[a-z]\)\s+", re.I)

def looks_like_new_list_start(text: str) -> bool:
    t = (text or "").strip()
    return bool(DECIMAL_RE.match(t) or ALPHA_RE.match(t))

def font_base(font_name: str) -> str:
    """Normalize font family a bit so 'ArialMT' vs 'Arial-BoldMT' still match."""
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
            # keep a smoothed current y to be robust to tiny jitter
            cur_y = y if cur_y is None else (cur_y + y) / 2.0
        else:
            lines.append(cur)
            cur = [s]
            cur_y = y

    if cur:
        lines.append(cur)
    return lines

def detect_sup_sub_for_block(spans: List[Dict[str, Any]]) -> List[Optional[str]]:
    if not spans:
        return []

    SUPERSCRIPT_CHARS = {"¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹","⁰"}
    SUBSCRIPT_CHARS   = {"₁","₂","₃","₄","₅","₆","₇","₈","₉","₀"}

    def median(xs):
        xs = sorted(xs)
        n = len(xs)
        if n == 0:
            return 0.0
        m = n // 2
        return xs[m] if n % 2 else 0.5 * (xs[m-1] + xs[m])

    # default font size if spans are missing size info
    all_sizes = [float(s.get("size") or s.get("h") or 0.0) for s in spans]
    default_size = median([x for x in all_sizes if x > 0]) or 10.0

    lines = group_spans_into_lines(spans, y_tol=2.0)
    idx_map = {id(s): i for i, s in enumerate(spans)}
    out: List[Optional[str]] = [None] * len(spans)

    strict_delta_ratio = 0.35
    soft_delta_ratio   = 0.10
    small_font_ratio   = 0.9   # how much smaller counts as "reduced"

    for line in lines:
        baselines, sizes = [], []
        for s in line:
            sz = float(s.get("size", 0.0)) or default_size
            y  = float(s.get("y", 0.0))
            ref = min(sz, default_size)
            baselines.append(y + ref * 0.8)
            sizes.append(sz)

        line_baseline = median(baselines)
        line_size = median([z for z in sizes if z > 0]) or default_size

        for s, sz, base in zip(line, sizes, baselines):
            i = idx_map[id(s)]
            txt = (s.get("text") or "").strip()
            delta = base - line_baseline
            tag: Optional[str] = None

            # Rule 1: Unicode superscript/subscript characters
            if txt in SUPERSCRIPT_CHARS:
                tag = "sup"
            elif txt in SUBSCRIPT_CHARS:
                tag = "sub"
            else:
                # Rule 2: Compare size + vertical shift
                if sz < small_font_ratio * line_size:
                    # If shifted up significantly → superscript
                    if delta < -soft_delta_ratio * line_size:
                        tag = "sup"
                    # If shifted down significantly → subscript
                    elif delta > soft_delta_ratio * line_size:
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

# ---------------------- Original helpers (kept) ----------------------

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
    import re
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n(?=[a-zäöüß])", "", s)
    s = s.replace("\n", " ")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"\s+", " ", s).strip()
    return s

def detect_body_size(blocks):
    sizes = [b.get("size", 0.0) for b in blocks if len((b.get("text") or "")) > 40 and b.get("size",0)>0]
    if not sizes:
        sizes = [b.get("size", 0.0) for b in blocks if b.get("size",0)>0]
    xs = sorted(sizes)
    n = len(xs)
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

def merge_flow_blocks(blocks, gap_px=6.0, size_tol=0.6, x_slop=40.0):
    """
    Merge adjacent 'text'/'body' blocks on the same page that visually form
    one paragraph or list item (e.g., wrapped line under '1.' or 'a)').
    """
    if not blocks:
        return blocks

    blocks = sorted(blocks, key=page_sort_key)
    out = []
    i = 0
    while i < len(blocks):
        cur = blocks[i]
        i += 1

        if cur.get("kind") != "text" or cur.get("role") != "body":
            out.append(cur)
            continue

        while i < len(blocks):
            nxt = blocks[i]
            # must be same page + text body
            if not (nxt.get("kind") == "text" and nxt.get("role") == "body" and
                    nxt["page_index"] == cur["page_index"]):
                break

            # don't merge over a NEW list start like "2." or "b)"
            if looks_like_new_list_start(nxt.get("text","")):
                break

            # geometry: small vertical gap + same column (allow hanging indent)
            gap = max(0.0, nxt["y"] - (cur["y"] + cur["h"]))
            same_column = (abs(nxt["x"] - cur["x"]) <= x_slop) or (nxt["x"] >= cur["x"])
            if gap > gap_px or not same_column:
                break

            # style proximity: similar font & size (loose)
            fs_cur, fs_nxt = float(cur.get("size") or 0.0), float(nxt.get("size") or 0.0)
            if fs_cur and fs_nxt and abs(fs_cur - fs_nxt) > size_tol:
                break
            if font_base(cur.get("font","")) and font_base(nxt.get("font","")) and \
               font_base(cur.get("font","")) != font_base(nxt.get("font","")):
                break

            # merge nxt into cur
            i += 1
            t1 = (cur.get("text","") or "").rstrip()
            t2 = (nxt.get("text","") or "").lstrip()
            # de-hyphenate across blocks
            if t1.endswith("-") and (t2[:1].islower() or t2[:1].isdigit()):
                joined = t1[:-1] + t2
            else:
                joined = (t1 + " " + t2).strip()
            cur["text"] = normalize_text(joined)

            # merge runs preserving order
            r1 = cur.get("runs") or []
            r2 = nxt.get("runs") or []
            cur["runs"] = (r1 + r2) if (r1 or r2) else []

            # union bbox
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
    start_printed = None
    start_heading = None
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

# ---------------------- New: header/footer detection ----------------------

def mark_headers_footers(blocks: List[Dict[str,Any]], page_heights: Dict[int,float],
                         header_frac=0.10, footer_frac=0.12, min_ratio=0.6):
    """Mark repeating text near top/bottom as header/footer.
    Footer uses the *bottom edge* (y+h) so low strips are caught."""
    if not blocks:
        return
    page_count = len({b["page_index"] for b in blocks})
    top_counts, bot_counts = {}, {}

    for b in blocks:
        h = page_heights.get(b["page_index"], 842.0)
        top_y = h * header_frac
        bot_y = h * (1.0 - footer_frac)
        txt = (b.get("text","") or "").strip()
        if not txt:
            continue
        if b["y"] < top_y:
            top_counts[txt] = top_counts.get(txt, 0) + 1
        if (b["y"] + b["h"]) > bot_y:
            bot_counts[txt] = bot_counts.get(txt, 0) + 1

    # page numbers count as footer even if not repeating
    top_common = {t for t,c in top_counts.items() if c / page_count >= min_ratio and not re.fullmatch(r"\d{1,3}", t)}
    bot_common = {t for t,c in bot_counts.items() if c / page_count >= min_ratio or re.fullmatch(r"\d{1,3}", t)}

    for b in blocks:
        h = page_heights.get(b["page_index"], 842.0)
        top_y = h * header_frac
        bot_y = h * (1.0 - footer_frac)
        txt = (b.get("text","") or "").strip()
        if txt in top_common and b["y"] < top_y:
            b["role"] = "header"
        elif (txt in bot_common and (b["y"] + b["h"]) > bot_y) or re.fullmatch(r"\d{1,3}", txt):
            b["role"] = "footer"
        else:
            b.setdefault("role", "body")


# ---------------------- Build intermediate ----------------------

def build_intermediate(pdf_path, out_path, metadata_path, start_page_cli, start_heading_cli, lang,
                       heading_threshold, anchor_tolerance, max_pages=None, no_tables=False,
                       debug_csv=None, debug_jsonl=None, start_printed=None, printed_first_cli=None, infer_printed=True,
                       emit_spans=False):
    import re
    require_pymupdf()
    doc = fitz.open(pdf_path.as_posix())

    meta_start_printed = None
    meta_start_heading = None
    if metadata_path and metadata_path.exists():
        meta_start_printed, meta_start_heading = extract_start_from_metadata(metadata_path)

    # Precedence: CLI > metadata > defaults
    start_page = start_page_cli
    start_heading = start_heading_cli or meta_start_heading
    printed_first = printed_first_cli

    # If not given a PDF page, derive strictly from metadata printed page by scanning footer
    start_printed_effective = (start_printed if start_printed is not None else meta_start_printed)
    if start_page is None and start_printed_effective is not None:
        idx = detect_pdf_start_page_by_footer(doc, start_printed_effective)
        if idx is not None:
            start_page = idx + 1  # convert 0-based to 1-based
        else:
            # fallback to old filename inference only if footer scan failed
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
                # last resort
                print("WARNING: Could not determine the printed-first page; pass --printed-first or disable inference.", file=sys.stderr)

    if start_page is None:
        start_page = 1



    all_blocks, all_tables, all_images = [], [], []
    page_heights: Dict[int,float] = {}

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
            # Inline runs with style
            runs = []
            if ss:
                # enrich with style flags
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

                # 🔑 sort before detection (ensures leftmost digit is first)
                tmp_runs.sort(key=lambda r: (r["y"], r["x"]))

                # classify superscripts/subscripts
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
            blk["text"] = normalize_text(blk["text"])
            blk["page_w"], blk["page_h"] = w, h
            blk["page_index"] = i
            all_blocks.append(blk)

        for sp in spans:
            if sp.get("image"):
                img = dict(sp)
                img["page_index"] = i
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
                                grid = t.extract()
                                tbl["grid"] = grid
                            except Exception:
                                pass
                            all_tables.append(tbl)
            except Exception:
                pass

    # Order and compute spacing
    all_blocks.sort(key=page_sort_key)
    compute_gap_above(all_blocks)

    # Heading detection
    body_size = detect_body_size(all_blocks)
    for b in all_blocks:
        score = score_heading(b, b["page_w"], body_size)
        b["_score"] = score
        b["kind"] = "heading" if score >= heading_threshold else "text"
        b["level"] = 0
        b.setdefault("role", "body")

    assign_heading_levels(all_blocks)

    # Mark headers/footers across the processed pages
    mark_headers_footers(all_blocks, page_heights)

    # Anchor & stop (include everything UNTIL the next same/higher heading)
    anchor_idx, anchor_level, sim = find_anchor(all_blocks, start_page, start_heading, anchor_tolerance)
    if anchor_idx < 0 and start_heading:
        print(f"WARNING: Anchor heading not found (best similarity={sim:.2f}). Starting at page {start_page}.", file=sys.stderr)

    # Determine bottom boundary at the next same/higher heading
    stop_page_incl = last_page_index
    bottom_clip_page = None
    bottom_clip_y = None
    if anchor_idx >= 0 and anchor_level is not None:
        for j in range(anchor_idx + 1, len(all_blocks)):
            if all_blocks[j]["kind"] == "heading" and all_blocks[j].get("level", 99) <= anchor_level:
                # Skip paragraph-style subheadings like "§ 1", they are not top-level stops
                txtj = all_blocks[j].get("text", "")
                if re.match(r"^§\s*\d+", txtj):
                    continue
                # include content on that page ABOVE the next top-level heading, exclude the heading itself and below
                stop_page_incl = all_blocks[j]["page_index"]
                bottom_clip_page = all_blocks[j]["page_index"]
                bottom_clip_y = all_blocks[j]["y"]
                break

    # Top clip: include anchor itself and anything below its top y on the start page
    top_clip_page = None
    top_clip_y = None
    if anchor_idx >= 0:
        anchor_block = all_blocks[anchor_idx]
        top_clip_page = anchor_block["page_index"]
        top_clip_y = anchor_block["y"]

    sel_blocks, sel_tables, sel_images = [], [], []
    for b in all_blocks:
        pi = b["page_index"]
        if pi < start_page-1 or pi > stop_page_incl:
            continue
        if top_clip_page is not None and pi == top_clip_page and top_clip_y is not None and b["y"] < top_clip_y - 0.5:
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


    # Merge wrapped/hanging lines into single flow blocks
    sel_blocks = merge_flow_blocks(sel_blocks, gap_px=6.0, size_tol=0.6, x_slop=40.0)


    # Debug outputs
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

    # Build XML
    root = etree.Element("doc")
    root.set("{http://www.w3.org/XML/1998/namespace}lang", lang)
    meta = etree.SubElement(root, "meta")
    etree.SubElement(meta, "source").text = pdf_path.name
    start_el = etree.SubElement(meta, "start")
    start_el.set("page", str(start_page))
    if start_heading:
        start_el.set("heading", start_heading)
    summary = summarize(sel_blocks, sel_tables, sel_images, start_page, stop_page_incl,
                        printed_start=start_printed_effective, printed_first=printed_first)
    etree.SubElement(meta, "summary").text = json.dumps(summary, ensure_ascii=False)

    # Emit pages & content
    pages = {}
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
        el.text = b.get("text","")

        # Optional inline spans
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

    tree = etree.ElementTree(root)
    out_path.write_bytes(etree.tostring(tree, encoding="utf-8", xml_declaration=True, pretty_print=True))
    print(json.dumps({"wrote": str(out_path), **summary}, ensure_ascii=False))

# ---------------------- CLI ----------------------

def main():
    ap = argparse.ArgumentParser(description="Build intermediate.xml (annotated model) from PDF + metadata.")
    ap.add_argument("pdf", type=Path, help="Input PDF")
    ap.add_argument("--metadata", type=Path, default=None, help="Metadata XML (optional; auto-reads start page/heading)")
    ap.add_argument("--out", type=Path, default=Path("intermediate.xml"), help="Output path for intermediate XML")

    # Start position: either PDF page or printed page (auto-inferred first printed from filename)
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
    ap.add_argument("--emit-spans", action="store_true", help="Emit <span> with bold/italic/sup/sub in XML")
    args = ap.parse_args()

    infer_printed = not args.no_infer_printed

    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        sys.exit(1)

    build_intermediate(args.pdf, args.out, args.metadata, args.start_page, args.start_heading, args.lang,
                       args.heading_threshold, args.anchor_tolerance, args.max_pages, args.no_tables,
                       args.debug_csv, args.debug_jsonl, start_printed=args.start_printed,
                       printed_first_cli=args.printed_first, infer_printed=infer_printed,
                       emit_spans=args.emit_spans)

if __name__ == "__main__":
    main()

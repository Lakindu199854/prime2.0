import argparse, sys, json, csv, re
from pathlib import Path
from typing import List, Dict, Any, Optional
from lxml import etree

from .pdfio import (
    require_pymupdf, detect_pdf_start_page_by_footer, read_text_blocks, read_rawdict,
    iter_raw_spans, spans_in_block, fitz
)
from .textutils import normalize_text, most_common, median, PAGE_NUM_RE
from .inline_detection import detect_sup_sub_for_block
from .classification import (
    detect_body_size, score_heading, assign_heading_levels, find_anchor,
    page_sort_key, merge_flow_blocks, compute_gap_above
)
from .header_footer import (
    mark_headers_footers, extract_footer_page_number_from_blocks,
    detect_bottom_footnotes_from_blocks
)
from .xmlout import ensure_page_node, add_separator, copy_block_coords_and_style, bbox_union, copy_inline_spans_from_runs
from .lists import maybe_emit_inline_list
from .tables import detect_tables_from_text_blocks
from .textutils import RULE_RE

def stringify_runs(runs: List[Dict[str,Any]]) -> str:
    try:
        light=[]
        for r in runs or []:
            u = 1 if r.get("sup")=="sup" else (2 if r.get("sup")=="sub" else 0)
            light.append({"t":r.get("text",""),"b":1 if r.get("bold") else 0,"i":1 if r.get("italic") else 0,"u":u,"s":round(float(r.get("size",0.0)),2)})
        return json.dumps(light, ensure_ascii=False)
    except Exception:
        return ""

def extract_sup_markers_from_runs(runs: List[Dict[str,Any]]) -> List[str]:
    out=[]
    for r in runs or []:
        if r.get("sup")=="sup":
            t=(r.get("text") or "").strip()
            if re.fullmatch(r"[0-9]", t): out.append(t)
    return out

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

def extract_start_from_metadata(meta_path: Path):
    start_printed = None; start_heading = None
    try:
        tree = etree.parse(str(meta_path))
        vb = tree.find(".//verkblatt")
        if vb is not None and vb.get("seite"):
            try: start_printed = int(vb.get("seite"))
            except Exception: start_printed = None
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

def build_tree_direct(
    pdf_path: Path, metadata_path: Optional[Path], start_page_cli: Optional[int],
    start_heading_cli: Optional[str], lang: str, heading_threshold: float,
    anchor_tolerance: float, max_pages: Optional[int] = None, no_tables: bool = False,
    debug_csv: Optional[Path] = None, debug_jsonl: Optional[Path] = None,
    start_printed: Optional[int] = None, printed_first_cli: Optional[int] = None,
    infer_printed: bool = True, emit_spans: bool = False, out_xml: Path = Path("out_tree.xml")
) -> Dict[str, Any]:

    require_pymupdf()
    doc = fitz.open(pdf_path.as_posix())

    # --- derive start position ---
    meta_start_printed = None; meta_start_heading = None
    if metadata_path and metadata_path.exists():
        meta_start_printed, meta_start_heading = extract_start_from_metadata(metadata_path)

    start_page = start_page_cli
    start_heading = start_heading_cli or meta_start_heading
    printed_first = printed_first_cli

    start_printed_effective = (start_printed if start_printed is not None else meta_start_printed)
    if start_page is None and start_printed_effective is not None:
        idx = detect_pdf_start_page_by_footer(doc, start_printed_effective)
        if idx is not None:
            start_page = idx + 1
        else:
            if printed_first is None and infer_printed:
                m = re.search(r"_(\d{2,4})_(\d{2,4})", pdf_path.name)
                if m:
                    try: printed_first = int(m.group(1))
                    except Exception: printed_first = None
            if printed_first is not None:
                start_page = (start_printed_effective - printed_first + 1)
                if start_page < 1 or start_page > doc.page_count:
                    print(f"WARNING: Computed start-page {start_page} out of [1..{doc.page_count}].", file=sys.stderr)
            else:
                print("WARNING: Could not determine the printed-first page; pass --printed-first or --no-infer-printed.", file=sys.stderr)
    if start_page is None: start_page = 1

    # --- extract blocks/spans/images/tables ---
    all_blocks: List[Dict[str,Any]] = []
    all_tables: List[Dict[str,Any]] = []
    all_images: List[Dict[str,Any]] = []
    page_heights: Dict[int,float] = {}

    last_page_index = doc.page_count - 1
    if max_pages is not None:
        last_page_index = min(last_page_index, (start_page-1)+max_pages-1)

    for i in range(start_page-1, last_page_index+1):
        page = doc.load_page(i)
        w,h = page.rect.width, page.rect.height
        page_heights[i] = float(h)

        blk_list = read_text_blocks(page)
        raw = read_rawdict(page)
        spans = list(iter_raw_spans(raw))

        for blk in blk_list:
            ss = spans_in_block(spans, blk)
            runs=[]
            if ss:
                tmp=[]
                for s in ss:
                    r = {"text":s.get("text",""), "font":s.get("font",""),
                         "size":float(s.get("size",0.0) or 0.0),
                         "x":float(s.get("x",0.0)),"y":float(s.get("y",0.0)),
                         "w":float(s.get("w",0.0)),"h":float(s.get("h",0.0))}
                    from .textutils import is_bold_font, is_italic_font
                    r["bold"] = is_bold_font(r["font"])
                    r["italic"] = is_italic_font(r["font"])
                    tmp.append(r)
                tmp.sort(key=lambda r: (r["y"], r["x"]))
                sup_tags = detect_sup_sub_for_block(tmp)
                for r,tag in zip(tmp, sup_tags): r["sup"] = tag
                runs = tmp
                from .textutils import most_common, median
                blk["font"] = most_common([r["font"] for r in runs if r.get("font")])
                blk["size"] = median([r["size"] for r in runs if r.get("size")])
            else:
                blk["font"] = ""; blk["size"] = 0.0

            blk["runs"] = runs
            blk["text"] = normalize_text(blk.get("text",""))

            if not blk["text"]:
                if runs:
                    try:
                        joined = " ".join((r.get("text","") or "").strip()
                                          for r in sorted(runs, key=lambda r: (float(r.get("y",0.0)), float(r.get("x",0.0)))))
                        blk["text"] = normalize_text(joined)
                    except Exception: pass
                if not blk["text"]:
                    try:
                        rect = fitz.Rect(blk["x"], blk["y"], blk["x"]+blk["w"], blk["y"]+blk["h"])
                        blk["text"] = normalize_text(page.get_textbox(rect) or "")
                    except Exception: pass

            blk["page_w"], blk["page_h"] = w,h
            blk["page_index"] = i
            blk.setdefault("role","body")
            all_blocks.append(blk)

        # images
        for sp in spans:
            if sp.get("image"):
                im = dict(sp); im["page_index"] = i
                all_images.append(im)

        # tables
        if not no_tables:
            try:
                if hasattr(page,"find_tables"):
                    tf = page.find_tables()
                    tables = getattr(tf,"tables",[]) if tf else []
                    for t in tables:
                        bbox = getattr(t,"bbox",None) or getattr(t,"rect",None)
                        if bbox:
                            x0,y0,x1,y1 = bbox
                            tbl={"x":float(x0),"y":float(y0),"w":float(x1-x0),"h":float(y1-y0),
                                 "rows":getattr(t,"row_count", getattr(t,"nrows",0)),
                                 "cols":getattr(t,"col_count", getattr(t,"ncols",0)),
                                 "page_index":i}
                            try: tbl["grid"] = t.extract()
                            except Exception: pass
                            all_tables.append(tbl)
            except Exception:
                pass

    # Fallback table detection if requested
    if no_tables:
        all_tables.extend(detect_tables_from_text_blocks(all_blocks))

    # --- classify + anchors + range ---
    all_blocks.sort(key=page_sort_key)
    compute_gap_above(all_blocks)

    body_size = detect_body_size(all_blocks)
    for b in all_blocks:
        sc = score_heading(b, b["page_w"], body_size)
        b["_score"] = sc
        b["kind"] = "heading" if sc >= heading_threshold else "text"
        b["level"] = 0
    assign_heading_levels(all_blocks)
    mark_headers_footers(all_blocks, page_heights)

    anchor_idx, anchor_level, sim = find_anchor(all_blocks, start_page, start_heading, anchor_tolerance)
    if anchor_idx < 0 and start_heading:
        print(f"WARNING: Anchor heading not found (best similarity={sim:.2f}). Starting at page {start_page}.", file=sys.stderr)

    stop_page_incl = last_page_index
    bottom_clip_page = None; bottom_clip_y = None
    if anchor_idx >= 0 and anchor_level is not None:
        for j in range(anchor_idx+1, len(all_blocks)):
            if all_blocks[j]["kind"]=="heading" and all_blocks[j].get("level",99) <= anchor_level:
                if re.match(r"^§\s*\d+", all_blocks[j].get("text","") or ""):
                    continue
                stop_page_incl = all_blocks[j]["page_index"]
                bottom_clip_page = all_blocks[j]["page_index"]
                bottom_clip_y = all_blocks[j]["y"]
                break

    top_clip_page=None; top_clip_y=None
    if anchor_idx >= 0:
        a = all_blocks[anchor_idx]
        top_clip_page, top_clip_y = a["page_index"], a["y"]

    sel_blocks=[]; sel_tables=[]; sel_images=[]
    for b in all_blocks:
        pi=b["page_index"]
        if pi < start_page-1 or pi > stop_page_incl: continue
        if top_clip_page is not None and pi==top_clip_page and top_clip_y is not None and b["y"] < top_clip_y - 0.5 and b.get("role")!="header":
            continue
        if bottom_clip_page is not None and pi==bottom_clip_page and bottom_clip_y is not None and b["y"] >= bottom_clip_y - 0.5:
            continue
        sel_blocks.append(b)
    for t in all_tables:
        pi=t["page_index"]
        if pi < start_page-1 or pi > stop_page_incl: continue
        if top_clip_page is not None and pi==top_clip_page and top_clip_y is not None and t["y"] < top_clip_y - 0.5:
            continue
        if bottom_clip_page is not None and pi==bottom_clip_page and bottom_clip_y is not None and t["y"] >= bottom_clip_y - 0.5:
            continue
        sel_tables.append(t)
    for im in all_images:
        pi=im["page_index"]
        if pi < start_page-1 or pi > stop_page_incl: continue
        if top_clip_page is not None and pi==top_clip_page and top_clip_y is not None and im["y"] < top_clip_y - 0.5:
            continue
        if bottom_clip_page is not None and pi==bottom_clip_page and bottom_clip_y is not None and im["y"] >= bottom_clip_y - 0.5:
            continue
        sel_images.append(im)

    # merge wrapped paragraphs
    sel_blocks = merge_flow_blocks(sel_blocks, gap_px=6.0, size_tol=0.6, x_slop=40.0)

    # --- Debug outputs ---
    if debug_csv:
        with open(debug_csv, "w", newline="", encoding="utf-8") as f:
            wcsv = csv.writer(f, delimiter=";")
            wcsv.writerow(["page","x","y","w","h","font","size","score","kind","level","role","text","runs_json"])
            for b in sel_blocks:
                wcsv.writerow([b["page_index"]+1,
                               f"{b['x']:.2f}", f"{b['y']:.2f}", f"{b['w']:.2f}", f"{b['h']:.2f}",
                               b.get("font",""), f"{b.get('size',0):.1f}",
                               f"{b.get('_score',0):.2f}", b.get("kind","text"), b.get("level",0),
                               b.get("role","body"),
                               (b.get("text","")[:200]).replace("\n"," "),
                               stringify_runs(b.get("runs", []))])
    if debug_jsonl:
        with open(debug_jsonl, "w", encoding="utf-8") as f:
            for b in sel_blocks: f.write(json.dumps(b, ensure_ascii=False)+"\n")

    # --- Build XML ---
    out_root = etree.Element("document")
    out_meta = etree.SubElement(out_root, "meta")
    etree.SubElement(out_meta, "source").text = pdf_path.name
    out_pages = etree.SubElement(out_root, "pages")

    by_page: Dict[int, List[Dict[str,Any]]] = {}
    for b in sel_blocks: by_page.setdefault(b["page_index"], []).append(b)
    for pi in by_page: by_page[pi].sort(key=lambda b:(b["y"],b["x"]))

    for pdf_i in range(start_page-1, stop_page_incl+1):
        page = doc.load_page(pdf_i)
        page_w = f"{page.rect.width:.2f}"
        page_h = f"{page.rect.height:.2f}"
        pdf_h_float = float(page_h)
        page_blocks = by_page.get(pdf_i, [])

        printed = extract_footer_page_number_from_blocks(page_blocks)
        if printed is None and printed_first is not None:
            printed = printed_first + (pdf_i)
        if printed is None:
            printed = pdf_i + 1

        out_page = ensure_page_node(out_pages, printed, page_w, page_h)
        header_el = out_page.find("./header")
        body_el   = out_page.find("./body")
        footer_el = out_page.find("./footer")

        # Header paragraphs
        for hb in [x for x in page_blocks if x.get("role")=="header"]:
            hpara = etree.SubElement(header_el, "paragraph")
            copy_block_coords_and_style(hpara, hb)
            hpara.text = hb.get("text","")
            copy_inline_spans_from_runs(hpara, hb.get("runs"), emit_spans)

        # Body content
        heading_markers_on_page: List[str] = []

        class ListState:
            def __init__(self):
                self.decimal_open = False; self.decimal_start = None
                self.alpha_open   = False; self.alpha_start   = None
                self.ul_open      = False
            def close_alpha(self):
                self.alpha_open = False; self.alpha_start = None
            def close_decimal(self):
                self.close_alpha(); self.decimal_open = False; self.decimal_start = None
            def close_ul(self):
                self.ul_open = False

        state = ListState()
        current_decimal_list_el = None
        current_alpha_list_el   = None
        current_ul_list_el      = None

        from .textutils import DECIMAL_RE, ALPHA_RE, BULLET_RE
        for b in [x for x in page_blocks if x.get("role") != "header"]:
            txt = (b.get("text") or "").strip()
            if not txt: continue
            if b.get("role") == "footer" and PAGE_NUM_RE.match(txt): continue

            # horizontal rule / separator
            if RULE_RE.search(txt or ""):
                state.close_ul(); current_ul_list_el = None
                add_separator(body_el, b)
                continue

            # split mixed blocks (per-line list detection)
            if maybe_emit_inline_list(body_el, b, emit_spans, copy_inline_spans_from_runs):
                state.close_ul(); current_ul_list_el = None
                state.close_alpha(); current_alpha_list_el = None
                continue

            # ---- bullet list ----
            m_bul = BULLET_RE.match(txt)
            if m_bul:
                if not state.ul_open:
                    ch = m_bul.group(1)
                    marker = "bullet" if ch in ("•","·","▪","‣") else "dash"
                    current_ul_list_el = etree.SubElement(body_el, "list", type="unordered", marker=marker)
                    state.ul_open = True
                    state.close_alpha(); current_alpha_list_el = None

                content = (m_bul.group(2) or "").strip()
                it = etree.SubElement(current_ul_list_el, "item")
                copy_block_coords_and_style(it, b)
                p = etree.SubElement(it, "paragraph")
                copy_block_coords_and_style(p, b)
                p.text = content
                copy_inline_spans_from_runs(p, b.get("runs"), emit_spans)
                continue

            # headings
            if b.get("kind") == "heading":
                state.close_ul(); current_ul_list_el = None
                heading_markers_on_page.extend(extract_sup_markers_from_runs(b.get("runs")))
                h_el = etree.SubElement(body_el, "heading", level=str(b.get("level", "1")))
                copy_block_coords_and_style(h_el, b)
                h_el.text = txt
                copy_inline_spans_from_runs(h_el, b.get("runs"), emit_spans)
                state.close_decimal(); current_decimal_list_el = None
                current_alpha_list_el = None
                continue

            # numeric list (close UL first)
            state.close_ul(); current_ul_list_el = None
            m_dec = DECIMAL_RE.match(txt)

            if m_dec:
                dec_ms = list(re.finditer(r"(?<!\d)(\d+)\.\s+", txt))
                parts = []
                if len(dec_ms) >= 2:
                    for i, m in enumerate(dec_ms):
                        start = m.end()
                        end   = dec_ms[i + 1].start() if i + 1 < len(dec_ms) else len(txt)
                        chunk = txt[start:end].strip()
                        cut = re.search(r"\s\d+\.\d+\s", chunk)
                        if cut:
                            chunk = chunk[:cut.start()].rstrip()
                        parts.append((int(m.group(1)), chunk))
                else:
                    parts = [(int(m_dec.group(1)), m_dec.group(2).strip())]

                if not state.decimal_open:
                    current_decimal_list_el = etree.SubElement(body_el, "list", type="ordered", marker="decimal")
                    state.decimal_open = True
                    current_decimal_list_el.set("start", str(parts[0][0]))

                for _, content in parts:
                    it = etree.SubElement(current_decimal_list_el, "item")
                    copy_block_coords_and_style(it, b)
                    p = etree.SubElement(it, "paragraph")
                    copy_block_coords_and_style(p, b)
                    p.text = content

                state.close_alpha(); current_alpha_list_el = None
                continue

            # alpha sublist (close UL first)
            state.close_ul(); current_ul_list_el = None
            m_alp = ALPHA_RE.match(txt)
            if m_alp:
                alpha = m_alp.group(1); content = m_alp.group(2).strip()
                parent = None
                if state.decimal_open and current_decimal_list_el is not None:
                    items = current_decimal_list_el.findall("./item")
                    if items: parent = items[-1]
                if not state.alpha_open or current_alpha_list_el is None:
                    list_parent = parent if parent is not None else body_el
                    current_alpha_list_el = etree.SubElement(list_parent, "list", type="ordered", marker="lower-alpha")
                    start = ord(alpha) - ord('a') + 1
                    current_alpha_list_el.set("start", str(start))
                    state.alpha_open = True; state.alpha_start = start
                it = etree.SubElement(current_alpha_list_el, "item")
                copy_block_coords_and_style(it, b)
                p = etree.SubElement(it, "paragraph")
                copy_block_coords_and_style(p, b)
                p.text = content
                continue

            # normal paragraph
            state.close_ul(); current_ul_list_el = None
            para = etree.SubElement(body_el, "paragraph")
            copy_block_coords_and_style(para, b)
            para.text = txt
            copy_inline_spans_from_runs(para, b.get("runs"), emit_spans)

        # Tables
        for t in [t for t in sel_tables if t["page_index"] == pdf_i]:
            tbl_attrs = {"x": f"{t['x']:.2f}", "y": f"{t['y']:.2f}",
                         "w": f"{t['w']:.2f}", "h": f"{t['h']:.2f}",
                         "rows": str(t.get("rows", 0)), "cols": str(t.get("cols", 0))}
            tbl_el = etree.SubElement(body_el, "table", **tbl_attrs)
            grid = t.get("grid", t.get("content", "").split("\n"))
            if isinstance(grid, list):
                for row in grid:
                    tr = etree.SubElement(tbl_el, "tr")
                    if isinstance(row, list):
                        for cell in row:
                            td = etree.SubElement(tr, "td")
                            td.text = normalize_text(str(cell))
                    else:
                        for cell in row.split():
                            td = etree.SubElement(tr, "td")
                            td.text = normalize_text(cell)

        # Images
        for im in [im for im in sel_images if im["page_index"]==pdf_i]:
            etree.SubElement(body_el, "image",
                x=f"{im['x']:.2f}", y=f"{im['y']:.2f}", w=f"{im['w']:.2f}", h=f"{im['h']:.2f}"
            )

        # Footer page-number
        pn = etree.SubElement(footer_el, "page-number"); pn.text = str(printed)

        # Footnotes
        page_notes = detect_bottom_footnotes_from_blocks(page_blocks, pdf_h_float)
        if page_notes:
            for mk in heading_markers_on_page:
                if mk in page_notes:
                    info = page_notes[mk]
                    fn = etree.SubElement(body_el, "footnote", marker=mk,
                        x=info.get("x","0"), y=info.get("y","0"),
                        w=info.get("w","0"), h=info.get("h","0"))
                    fn.text = info.get("text","")

        # List bboxes
        for lst in body_el.findall(".//list[@type='ordered']"):
            st = lst.get("start")
            if st and st.isdigit() and int(st)>1:
                lst.set("continue","true")
            items = lst.findall("./item")
            bb = bbox_union(items)
            if bb:
                x,y,w,h = bb
                lst.set("x", f"{x:.2f}"); lst.set("y", f"{y:.2f}")
                lst.set("w", f"{w:.2f}"); lst.set("h", f"{h:.2f}")

    # write XML
    etree.ElementTree(out_root).write(str(out_xml), encoding="utf-8", xml_declaration=True, pretty_print=True)

    # summary
    return summarize(sel_blocks, all_tables, sel_images,
                     start_page, stop_page_incl,
                     printed_start=start_printed_effective, printed_first=printed_first)
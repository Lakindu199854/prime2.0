#!/usr/bin/env python3
import sys, re, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lxml import etree

# ---------- Regexes ----------
DECIMAL_RE   = re.compile(r"^\s*(\d+)\.\s+(.*)")
ALPHA_RE     = re.compile(r"^\s*([a-z])\)\s+(.*)")
RULE_RE      = re.compile(r"[_\-]{5,}\s*$")   # underline/HR lines
PAGE_NUM_RE  = re.compile(r"^\s*(\d{1,4})\s*$")
FOOTNOTE_LINE_RE = re.compile(r"^\s*(\d+)\s+(.+)")

# ---------- Helpers ----------
def parse_summary_meta(root) -> Tuple[Optional[int], Optional[int]]:
    """
    Read printed_start and printed_first from <meta><summary> JSON if present.
    """
    s_el = root.find("./meta/summary")
    if s_el is None or not (s_el.text or "").strip():
        return None, None
    try:
        j = json.loads(s_el.text)
        return (int(j.get("printed_start")) if j.get("printed_start") else None,
                int(j.get("printed_first")) if j.get("printed_first") else None)
    except Exception:
        return None, None

def get_page_blocks(page_el):
    """Yield blocks in reading order."""
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
    """Return superscript digit markers (1..9) from <span sup="1"> inside a block (used for footnotes)."""
    out: List[str] = []
    for sp in b.findall("./span"):
        if sp.get("sup") in ("1", "sup"):
            t = (sp.text or "").strip()
            if re.fullmatch(r"[0-9]", t):
                out.append(t)
    return out

def detect_bottom_footnotes(page_el: etree._Element, page_h: float) -> Dict[str, Dict[str,str]]:
    """
    Heuristic: footnotes live near the bottom (last ~18%), like: '2 Gült.-Verzeichnis Nr. 245'
    Returns {marker: {"text":..., "x":..., "y":..., "w":..., "h":...}} (strings).
    """
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
    """Create or reuse a page node for a given printed page index; set width/height when first created."""
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
    """Attach coords + block-level font/size/role to dst."""
    for attr in ("x","y","w","h","font","size","role"):
        val = src_block.get(attr)
        if val is not None:
            dst.set(attr, val)

def copy_inline_spans(dst_parent: etree._Element, src_block: etree._Element):
    """
    Copy <span> children from intermediate into dst_parent,
    preserving bold/italic/sup/size/font and x/y/w/h.

    IMPORTANT: only clear the parent text if the spans themselves have
    real text content. If spans are empty (as in your case), KEEP the
    parent text so you don't lose content.
    """
    spans = src_block.findall("./span")
    if not spans:
        return

    # Do spans carry any non-empty text?
    spans_have_text = any((sp.text or "").strip() for sp in spans)

    # Only clear the parent text if spans actually carry text
    if spans_have_text:
        dst_parent.text = None  # avoid double text if spans contain content

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
        # If the span has text, copy it; if it doesn't, leave it empty.
        # Parent text remains intact in that case.
        out.text = sp.text or ""


def bbox_union(items: List[etree._Element]) -> Optional[Tuple[float,float,float,float]]:
    """Compute union bbox from elements that have x,y,w,h."""
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
    """Track open ordered lists: decimal (1.) → alpha (a)) nesting."""
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

# ---------- Core ----------
def build_tree(intermediate_xml: Path, out_xml: Path):
    src = etree.parse(str(intermediate_xml))
    root = src.getroot()

    printed_start, printed_first = parse_summary_meta(root)

    # Output skeleton
    out_root = etree.Element("document")
    out_meta = etree.SubElement(out_root, "meta")
    etree.SubElement(out_meta, "source").text = (root.findtext("./meta/source") or "")
    out_pages = etree.SubElement(out_root, "pages")

    # Iterate PDF pages in order
    for page_el in root.findall("./page"):
        pdf_index = int(page_el.get("index"))
        page_w = page_el.get("width")
        page_h = page_el.get("height")
        pdf_h = float(page_h or "0")

        # Determine printed page number
        footer_blocks = [b for b in get_page_blocks(page_el) if is_footer_block(b)]
        printed = extract_footer_page_number(footer_blocks)
        if printed is None and printed_first is not None:
            printed = printed_first + (pdf_index - 1)
        if printed is None:
            printed = pdf_index  # fallback

        out_page = ensure_page_node(out_pages, printed, page_w, page_h)
        body = out_page.find("./body")
        footer = out_page.find("./footer")

        # Collect bottom footnotes (marker -> dict)
        page_notes = detect_bottom_footnotes(page_el, pdf_h)

        # Track markers in headings to attach footnotes after content
        heading_markers_on_page: List[str] = []

        # List tracking per page
        state = ListState()
        current_decimal_list_el = None
        current_alpha_list_el = None

        # Walk blocks skipping headers and bare page-number footers
        for b in [x for x in get_page_blocks(page_el) if not is_header_block(x)]:
            kind = b.get("kind")
            role = b.get("role")
            txt = block_text(b)
            if not txt:
                continue
            if role == "footer" and PAGE_NUM_RE.match(txt):
                continue

            # Horizontal rule
            if is_rule_block(b):
                add_separator(body, b)
                continue

            # Heading
            if kind == "heading":
                heading_markers_on_page.extend(extract_sup_markers(b))
                h_el = etree.SubElement(body, "heading", level=b.get("level", "1"))
                copy_block_coords_and_style(h_el, b)
                h_el.text = txt  # provisional; will be cleared if spans exist
                copy_inline_spans(h_el, b)
                # break list context
                state.close_decimal()
                current_decimal_list_el = None
                current_alpha_list_el = None
                continue

            # List items
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

                # item with bbox from the source block
                it = etree.SubElement(current_decimal_list_el, "item")
                copy_block_coords_and_style(it, b)

                p = etree.SubElement(it, "paragraph")
                copy_block_coords_and_style(p, b)
                p.text = content  # keep clean content (skip marker)
                # opening a decimal resets alpha level
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

            # Plain paragraph
            para = etree.SubElement(body, "paragraph")
            copy_block_coords_and_style(para, b)
            para.text = txt
            copy_inline_spans(para, b)

        # Footer: page number
        pn = etree.SubElement(footer, "page-number")
        pn.text = str(printed)

        # Footnotes attached after content (only for markers we saw in headings on this page)
        if page_notes:
            for mk in heading_markers_on_page:
                if mk in page_notes:
                    info = page_notes[mk]
                    fn = etree.SubElement(body, "footnote", marker=mk,
                        x=info.get("x","0"), y=info.get("y","0"),
                        w=info.get("w","0"), h=info.get("h","0"))
                    fn.text = info.get("text","")

        # Mark list continuation visually when start>1, and compute list bbox
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

    # Write output
    etree.ElementTree(out_root).write(str(out_xml), encoding="utf-8", xml_declaration=True, pretty_print=True)

def main():
    if len(sys.argv) < 3:
        print("Usage: python tree_xml_builder.py intermediate.xml out_tree.xml", file=sys.stderr)
        sys.exit(2)
    build_tree(Path(sys.argv[1]), Path(sys.argv[2]))

if __name__ == "__main__":
    main()

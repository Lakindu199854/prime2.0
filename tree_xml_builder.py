#!/usr/bin/env python3
import sys, re, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from lxml import etree

DECIMAL_RE = re.compile(r"^\s*(\d+)\.\s+(.*)")
ALPHA_RE   = re.compile(r"^\s*([a-z])\)\s+(.*)")
RULE_RE    = re.compile(r"[_\-]{5,}\s*$")  # a row of underscores/dashes ~ horizontal rule
PAGE_NUM_RE = re.compile(r"^\s*(\d{1,4})\s*$")

def parse_summary_meta(root) -> Tuple[Optional[int], Optional[int]]:
    """Read printed_start and printed_first from meta/summary JSON if available."""
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
    # Use the element text; spans (if present) are for styles/footnote markers
    return (b.text or "").strip()

def is_footer_block(b: etree._Element) -> bool:
    return (b.get("role") == "footer")

def is_header_block(b: etree._Element) -> bool:
    return (b.get("role") == "header")

def is_rule_block(b: etree._Element) -> bool:
    return bool(RULE_RE.search(block_text(b)))

def extract_sup_markers(b: etree._Element) -> List[str]:
    """Return any superscript digit markers found inside spans of this block, in order."""
    out = []
    for sp in b.findall("./span"):
        if sp.get("sup") in ("1", "sup"):
            t = (sp.text or "").strip()
            # Keep only simple numeric markers like 1..9
            if re.fullmatch(r"[0-9]", t):
                out.append(t)
    return out

def detect_bottom_footnotes(page_el: etree._Element, page_h: float) -> Dict[str, str]:
    """
    Heuristic: footnotes lines live near the bottom (last ~18%),
    and look like "2 Gült.-Verzeichnis Nr. 245".
    Returns {marker: text}.
    """
    notes = {}
    try:
        h_frac = 0.82  # bottom band threshold
        for b in page_el.findall("./block"):
            # prefer blocks already marked footer; otherwise use y+h band
            y = float(b.get("y", "0")); h = float(b.get("h", "0"))
            txt = block_text(b)
            if not txt:
                continue
            in_bottom_band = ((y + h) >= page_h * h_frac)
            looks_like_note = re.match(r"^\s*(\d+)\s+(.+)", txt)
            if (b.get("role") == "footer" and looks_like_note) or (in_bottom_band and looks_like_note):
                m = looks_like_note
                notes[m.group(1)] = m.group(2).strip()
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
                    pass
    return None

def ensure_page_node(doc_pages_el: etree._Element, printed_index: int) -> etree._Element:
    page_el = etree.SubElement(doc_pages_el, "page", index=str(printed_index))
    etree.SubElement(page_el, "body")
    etree.SubElement(page_el, "footer")
    return page_el

def add_separator(body_el: etree._Element):
    etree.SubElement(body_el, "separator", kind="rule")

class ListState:
    """
    Tracks current open lists across pages.
    We only need two levels for this doc: decimal -> lower-alpha.
    """
    def __init__(self):
        self.decimal_open = False
        self.decimal_start = None
        self.alpha_open = False
        self.alpha_start = None

    def close_alpha(self, body_el):
        if self.alpha_open:
            # find last alpha list and close by doing nothing (XML build already done)
            self.alpha_open = False
            self.alpha_start = None

    def close_decimal(self, body_el):
        # closing decimal implies alpha closes too
        self.close_alpha(body_el)
        if self.decimal_open:
            self.decimal_open = False
            self.decimal_start = None

def build_tree(intermediate_xml: Path, out_xml: Path):
    src = etree.parse(str(intermediate_xml))
    root = src.getroot()

    printed_start, printed_first = parse_summary_meta(root)

    # Build output skeleton
    out_root = etree.Element("document")
    out_pages = etree.SubElement(out_root, "pages")

    # Iterate PDF pages (ordered by @index)
    for page_el in root.findall("./page"):
        pdf_index = int(page_el.get("index"))
        pdf_w = float(page_el.get("width", "0") or 0.0)
        pdf_h = float(page_el.get("height", "0") or 0.0)

        # figure printed page
        footer_blocks = list(b for b in get_page_blocks(page_el) if is_footer_block(b))
        printed = extract_footer_page_number(footer_blocks)
        if printed is None and printed_first is not None:
            printed = printed_first + (pdf_index - 1)

        if printed is None:
            # fallback to pdf index if nothing else
            printed = pdf_index

        out_page = ensure_page_node(out_pages, printed)
        body = out_page.find("./body")
        footer = out_page.find("./footer")

        # Collect bottom footnotes (marker -> text)
        page_notes = detect_bottom_footnotes(page_el, pdf_h)

        # Capture heading-level sup markers to link with notes
        heading_markers_on_page: List[str] = []

        # List state across this page
        state = ListState()
        # Placeholders for the actual list elements being built
        current_decimal_list_el = None
        current_alpha_list_el = None

        # Iterate blocks in order, skipping headers & number-only page number footers
        blocks = [b for b in get_page_blocks(page_el) if not is_header_block(b)]
        for b in blocks:
            kind = b.get("kind")
            role = b.get("role")
            txt = block_text(b)
            if not txt:
                continue

            # Skip pure page-number footer here; we'll set footer after loop
            if role == "footer" and PAGE_NUM_RE.match(txt):
                continue

            # Horizontal rule / underline line
            if is_rule_block(b):
                add_separator(body)
                continue

            # Heading
            if kind == "heading":
                # record any footnote markers contained in the heading
                heading_markers_on_page.extend(extract_sup_markers(b))
                h_el = etree.SubElement(body, "heading", level=b.get("level", "1"))
                h_el.text = txt
                # headings break list context
                state.close_alpha(body)
                state.close_decimal(body)
                current_decimal_list_el = None
                current_alpha_list_el = None
                continue

            # Try list item detection
            m_dec = DECIMAL_RE.match(txt)
            m_alp = ALPHA_RE.match(txt)

            if m_dec:
                num = int(m_dec.group(1))
                content = m_dec.group(2).strip()

                # If decimal not open, start a new one
                if not state.decimal_open:
                    current_decimal_list_el = etree.SubElement(body, "list", type="ordered", marker="decimal")
                    current_decimal_list_el.set("start", str(num))
                    state.decimal_open = True
                    state.decimal_start = num
                    # when starting a new decimal, alpha (if any) must be closed
                    state.close_alpha(body)
                    current_alpha_list_el = None
                else:
                    # If we jumped (e.g., first item on a new page), update start on first encounter
                    if state.decimal_start is None:
                        current_decimal_list_el.set("start", str(num))
                        state.decimal_start = num

                # add the item
                it = etree.SubElement(current_decimal_list_el, "item")
                p = etree.SubElement(it, "paragraph")
                p.text = content
                # opening a new decimal also resets alpha nesting context
                current_alpha_list_el = None
                continue

            if m_alp:
                alpha = m_alp.group(1)
                content = m_alp.group(2).strip()
                # ensure a decimal context exists for nesting; if not, start a top-level alpha list
                parent = None
                if state.decimal_open and current_decimal_list_el is not None:
                    # nest under last decimal item
                    items = current_decimal_list_el.findall("./item")
                    if items:
                        parent = items[-1]
                # else: no decimal open; use body as parent

                if not state.alpha_open or current_alpha_list_el is None:
                    # starting a new alpha list
                    list_parent = parent if parent is not None else body
                    current_alpha_list_el = etree.SubElement(list_parent, "list", type="ordered", marker="lower-alpha")
                    # 'start' for alpha: map letter to ord
                    start = ord(alpha) - ord('a') + 1
                    current_alpha_list_el.set("start", str(start))
                    state.alpha_open = True
                    state.alpha_start = start
                # append item
                it = etree.SubElement(current_alpha_list_el, "item")
                p = etree.SubElement(it, "paragraph")
                p.text = content
                continue

            # Otherwise: normal paragraph
            para = etree.SubElement(body, "paragraph")
            para.text = txt

        # Footer: page number
        if printed is not None:
            pn = etree.SubElement(footer, "page-number")
            pn.text = str(printed)

        # Insert footnotes tied to markers we saw (if any)
        if page_notes:
            # Create footnotes in body, after main content (similar to your example)
            for mk in heading_markers_on_page:
                if mk in page_notes:
                    fn = etree.SubElement(body, "footnote", marker=mk)
                    fn.text = page_notes[mk]

        # Mark list continuation across *this* page boundary:
        # If page ends while a list is open, set a flag so next page can set continue="true".
        # We encode it on the last created list as a custom attribute to be read by the next loop,
        # but since we build page-by-page, we instead rely on detecting the first marker on the next
        # page and set 'start' accordingly. To reflect continuity visually, add 'continue="true"'
        # when a new alpha list starts with start>1 or when decimal doesn't start at 1.
        # (We already set 'start', now annotate 'continue' where start>1)
        for lst in body.findall(".//list[@type='ordered']"):
            st = lst.get("start")
            if st and st.isdigit() and int(st) > 1:
                lst.set("continue", "true")

    # Write
    etree.ElementTree(out_root).write(str(out_xml), encoding="utf-8", xml_declaration=True, pretty_print=True)

def main():
    if len(sys.argv) < 3:
        print("Usage: python tree_xml_builder.py intermediate.xml out_tree.xml", file=sys.stderr)
        sys.exit(2)
    build_tree(Path(sys.argv[1]), Path(sys.argv[2]))

if __name__ == "__main__":
    main()

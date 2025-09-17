import re
from lxml import etree
from typing import Dict, Any, List
from .textutils import DECIMAL_RE, ALPHA_RE, BULLET_RE, normalize_text
from .inline_detection import line_bbox
from .xmlout import copy_block_coords_and_style, bbox_union

def runs_to_lines_with_text(runs: List[Dict[str,Any]], y_tol: float = 2.5):
    # Return [(normalized_text_for_line, runs_in_line)]
    runs_sorted = sorted(runs or [], key=lambda r: (float(r.get("y",0.0)), float(r.get("x",0.0))))
    out=[]; cur=[]; cur_y=None
    for r in runs_sorted:
        y=float(r.get("y",0.0))
        if cur_y is None or abs(y-cur_y) <= y_tol:
            cur.append(r); cur_y = y if cur_y is None else (cur_y+y)/2.0
        else:
            txt = normalize_text("".join((x.get("text","") or "") for x in sorted(cur, key=lambda t:(t["y"],t["x"]))))
            out.append((txt, cur)); cur=[r]; cur_y=y
    if cur:
        txt = normalize_text("".join((x.get("text","") or "") for x in sorted(cur, key=lambda t:(t["y"],t["x"]))))
        out.append((txt, cur))
    return out

def maybe_emit_inline_list(body_el: etree._Element, block: Dict[str, Any], emit_spans: bool, copy_spans_fn) -> bool:
    """Split & emit when a single block mixes prose and list lines."""
    runs = block.get("runs") or []

    # ---------- Preferred: per-line using runs ----------
    if runs:
        L = runs_to_lines_with_text(runs, y_tol=2.5)
        segments = []
        list_count = text_count = 0

        for line_text, line_runs in L:
            m_dec = DECIMAL_RE.match(line_text)
            m_alp = ALPHA_RE.match(line_text)
            m_bul = BULLET_RE.match(line_text)

            if m_dec:
                list_count += 1
                segments.append(("ol-dec", m_dec.group(2).strip(), line_runs, int(m_dec.group(1))))
            elif m_alp:
                list_count += 1
                start = ord(m_alp.group(1)) - ord('a') + 1
                segments.append(("ol-alpha", m_alp.group(2).strip(), line_runs, start))
            elif m_bul:
                list_count += 1
                segments.append(("ul", m_bul.group(2).strip(), line_runs, m_bul.group(1)))
            else:
                text_count += 1
                segments.append(("text", line_text, line_runs, None))

        if (text_count > 0 and list_count > 0) or (list_count >= 2):
            def copy_coords_from_line(dst_el: etree._Element, ln_runs: List[Dict[str,Any]]):
                x,y,w,h = line_bbox(ln_runs)
                dst_el.set("x", f"{x:.2f}"); dst_el.set("y", f"{y:.2f}")
                dst_el.set("w", f"{w:.2f}"); dst_el.set("h", f"{h:.2f}")
                if block.get("font"): dst_el.set("font", f"{block.get('font')}")
                if block.get("size"): dst_el.set("size", f"{block.get('size')}")

            pos = 0
            if segments and segments[0][0] == "text":
                intro = []
                while pos < len(segments) and segments[pos][0] == "text":
                    intro.append(segments[pos]); pos += 1
                if intro:
                    para = etree.SubElement(body_el, "paragraph")
                    copy_coords_from_line(para, intro[0][2])
                    para.text = " ".join(s[1] for s in intro if s[1])
                    copy_spans_fn(para, intro[0][2], emit_spans)

            # contiguous list runs of same kind
            while pos < len(segments):
                kind = segments[pos][0]
                if kind not in ("ol-dec","ol-alpha","ul"):
                    tail = []
                    while pos < len(segments) and segments[pos][0] == "text":
                        tail.append(segments[pos]); pos += 1
                    if tail:
                        para = etree.SubElement(body_el, "paragraph")
                        copy_coords_from_line(para, tail[0][2])
                        para.text = " ".join(s[1] for s in tail if s[1])
                        copy_spans_fn(para, tail[0][2], emit_spans)
                    continue

                items = []
                while pos < len(segments) and segments[pos][0] == kind:
                    items.append(segments[pos]); pos += 1

                if kind == "ul":
                    mark = items[0][3]
                    marker = "bullet" if str(mark) in ("•","·","▪","‣") else "dash"
                    lst = etree.SubElement(body_el, "list", type="unordered", marker=marker)
                elif kind == "ol-dec":
                    lst = etree.SubElement(body_el, "list", type="ordered", marker="decimal")
                    lst.set("start", str(int(items[0][3])))
                else:
                    lst = etree.SubElement(body_el, "list", type="ordered", marker="lower-alpha")
                    lst.set("start", str(int(items[0][3])))

                for _, content, ln_runs, _extra in items:
                    it = etree.SubElement(lst, "item")
                    copy_coords_from_line(it, ln_runs)
                    p = etree.SubElement(it, "paragraph")
                    copy_coords_from_line(p, ln_runs)
                    p.text = content
                    copy_spans_fn(p, ln_runs, emit_spans)

                bb = bbox_union(lst.findall("./item"))
                if bb:
                    x,y,w,h = bb
                    lst.set("x", f"{x:.2f}"); lst.set("y", f"{y:.2f}")
                    lst.set("w", f"{w:.2f}"); lst.set("h", f"{h:.2f}")
            return True

    # ---------- Fallback: text-only split ----------
    full = (block.get("text") or "")

    def emit_paragraph(text: str):
        if not text.strip(): return
        para = etree.SubElement(body_el, "paragraph")
        copy_block_coords_and_style(para, block)
        para.text = normalize_text(text)

    dec_pat = re.compile(r"(?<!\d)(\d+)\.\s+")
    dec_ms = list(dec_pat.finditer(full))

    if len(dec_ms) >= 2:
        head = full[:dec_ms[0].start()].strip()
        if head: emit_paragraph(head)

        lst = etree.SubElement(body_el, "list", type="ordered", marker="decimal")
        start_num = int(dec_ms[0].group(1))
        lst.set("start", str(start_num))

        for i, m in enumerate(dec_ms):
            start = m.end()
            end   = dec_ms[i + 1].start() if i + 1 < len(dec_ms) else len(full)
            chunk = full[start:end].strip()
            cut = re.search(r"\s\d+\.\d+\s", chunk)
            if cut: chunk = chunk[:cut.start()].rstrip()
            it = etree.SubElement(lst, "item")
            copy_block_coords_and_style(it, block)
            p = etree.SubElement(it, "paragraph")
            copy_block_coords_and_style(p, block)
            p.text = normalize_text(chunk)
        return True

    alp_pat = re.compile(r"(?i)(?<![A-Za-z0-9])([a-z])\)\s+")
    alp_ms = list(alp_pat.finditer(full))
    if len(alp_ms) >= 2:
        head = full[:alp_ms[0].start()].strip()
        if head: emit_paragraph(head)
        lst = etree.SubElement(body_el, "list", type="ordered", marker="lower-alpha")
        start_alpha = ord(alp_ms[0].group(1).lower()) - ord('a') + 1
        lst.set("start", str(start_alpha))
        for i, m in enumerate(alp_ms):
            start = m.end()
            end   = alp_ms[i + 1].start() if i + 1 < len(alp_ms) else len(full)
            chunk = full[start:end].strip()
            it = etree.SubElement(lst, "item")
            copy_block_coords_and_style(it, block)
            p = etree.SubElement(it, "paragraph")
            copy_block_coords_and_style(p, block)
            p.text = normalize_text(chunk)
        return True

    m = re.search(r"(?s)^(?P<intro>.+?[.:;])\s*(?P<ch>[•·▪‣\-–—])\s+(?P<rest>.+)$", full)
    if m:
        emit_paragraph(m.group("intro").rstrip())
        ch = m.group("ch")
        marker = "bullet" if ch in ("•", "·", "▪", "‣") else "dash"
        lst = etree.SubElement(body_el, "list", type="unordered", marker=marker)
        it = etree.SubElement(lst, "item")
        copy_block_coords_and_style(it, block)
        p = etree.SubElement(it, "paragraph")
        copy_block_coords_and_style(p, block)
        p.text = normalize_text(m.group("rest"))
        return True

    return False
from typing import List, Dict, Any, Optional
import re
from .textutils import PAGE_NUM_RE, FOOTNOTE_LINE_RE

def mark_headers_footers(blocks: List[Dict[str,Any]],
                         page_heights: Dict[int,float],
                         header_frac: float = 0.10,
                         footer_frac: float = 0.12,
                         min_ratio: float = 0.6):
    if not blocks: return

    def norm_key(t: str) -> str:
        t = (t or "").strip()
        t = re.sub(r"\s+", " ", t)
        t = re.sub(r"\b\d+\b", "<NUM>", t)
        return t.lower()

    page_count = max(1, len({b["page_index"] for b in blocks}))
    top_counts, bot_counts = {}, {}
    for b in blocks:
        h = float(page_heights.get(b["page_index"], 842.0))
        top_y = h * header_frac; bot_y = h * (1.0 - footer_frac)
        raw = (b.get("text","") or "").strip()
        if not raw: continue
        key = norm_key(raw)
        if b["y"] < top_y: top_counts[key] = top_counts.get(key,0)+1
        if (b["y"] + b["h"]) > bot_y: bot_counts[key] = bot_counts.get(key,0)+1

    top_common = {k for k,c in top_counts.items() if (c/page_count) >= min_ratio}
    bot_common = {k for k,c in bot_counts.items() if (c/page_count) >= min_ratio}

    for b in blocks:
        h = float(page_heights.get(b["page_index"], 842.0))
        top_y = h*header_frac; bot_y = h*(1.0-footer_frac)
        raw = (b.get("text","") or "").strip()
        key = norm_key(raw)
        is_num = bool(re.fullmatch(r"\d{1,4}", raw))
        at_top = b["y"] < top_y
        at_bot = (b["y"] + b["h"]) > bot_y
        if at_top and key in top_common and not is_num:
            b["role"] = "header"
        elif at_bot and (key in bot_common or is_num):
            b["role"] = "footer"
        else:
            b.setdefault("role", "body")

def extract_footer_page_number_from_blocks(page_blocks: List[Dict[str,Any]]) -> Optional[int]:
    for b in page_blocks:
        if b.get("role") == "footer":
            m = PAGE_NUM_RE.match((b.get("text") or "").strip())
            if m:
                try: return int(m.group(1))
                except Exception: return None
    return None

def detect_bottom_footnotes_from_blocks(page_blocks: List[Dict[str,Any]], page_h: float) -> Dict[str, Dict[str,str]]:
    notes: Dict[str, Dict[str,str]] = {}
    h_frac = 0.82
    for b in page_blocks:
        y = float(b.get("y",0.0)); h = float(b.get("h",0.0))
        txt = (b.get("text") or "").strip()
        if not txt: continue
        in_bottom_band = ((y + h) >= page_h * h_frac)
        m = FOOTNOTE_LINE_RE.match(txt)
        if ((b.get("role") == "footer") and m) or (in_bottom_band and m):
            marker = m.group(1)
            notes[marker] = {
                "text": m.group(2).strip(),
                "x": f"{b.get('x',0)}", "y": f"{b.get('y',0)}",
                "w": f"{b.get('w',0)}", "h": f"{b.get('h',0)}"
            }
    return notes
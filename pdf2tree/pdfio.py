from pathlib import Path
from typing import Dict, Any, List, Optional
import re, sys
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

from .textutils import normalize_text

def require_pymupdf():
    if fitz is None:
        print("ERROR: PyMuPDF not installed. pip install PyMuPDF", file=sys.stderr)
        sys.exit(2)

def detect_pdf_start_page_by_footer(doc, target_printed: int, bottom_frac: float = 0.15) -> Optional[int]:
    try:
        for i in range(doc.page_count):
            p = doc.load_page(i)
            h = float(p.rect.height)
            cutoff = h*(1.0-bottom_frac)
            for b in (p.get_text("blocks") or []):
                if not (len(b)>=5 and isinstance(b[4], str)): continue
                y0, txt = float(b[1]), (b[4] or "").strip()
                if y0 >= cutoff:
                    t = normalize_text(txt)
                    if t == str(target_printed) or re.fullmatch(rf"\D*{target_printed}\D*", t):
                        return i
    except Exception:
        pass
    return None

def read_text_blocks(page) -> List[Dict[str,Any]]:
    out = []
    try:
        for b in page.get_text("blocks") or []:
            if len(b)>=5 and isinstance(b[4], str):
                x0, y0, x1, y1, txt = float(b[0]), float(b[1]), float(b[2]), float(b[3]), (b[4] or "").strip()
                if txt:
                    out.append({"x":x0,"y":y0,"w":x1-x0,"h":y1-y0,"text":txt})
    except Exception:
        pass
    return out

def read_rawdict(page):
    try:
        return page.get_text("rawdict") or {"blocks":[]}
    except Exception:
        return {"blocks":[]}

def iter_raw_spans(raw):
    for b in raw.get("blocks", []):
        if b.get("type",0) == 0:
            for ln in b.get("lines", []):
                for sp in ln.get("spans", []):
                    bbox = sp.get("bbox", [0,0,0,0])
                    yield {
                        "x": float(bbox[0]), "y": float(bbox[1]),
                        "w": float(bbox[2]-bbox[0]), "h": float(bbox[3]-bbox[1]),
                        "cx": float((bbox[0]+bbox[2])/2), "cy": float((bbox[1]+bbox[3])/2),
                        "font": sp.get("font",""), "size": float(sp.get("size",0.0)),
                        "text": sp.get("text",""), "flags": int(sp.get("flags",0))
                    }
        elif b.get("type") == 1:
            bbox = b.get("bbox", [0,0,0,0])
            yield {"image":True,"x":float(bbox[0]),"y":float(bbox[1]),"w":float(bbox[2]-bbox[0]),"h":float(bbox[3]-bbox[1])}

def spans_in_block(spans, blk):
    x0,y0 = blk["x"], blk["y"]; x1,y1 = x0+blk["w"], y0+blk["h"]
    out = []
    for sp in spans:
        if "image" in sp: continue
        cx,cy = sp["cx"], sp["cy"]
        if x0-1 <= cx <= x1+1 and y0-1 <= cy <= y1+1:
            out.append(sp)
    return out

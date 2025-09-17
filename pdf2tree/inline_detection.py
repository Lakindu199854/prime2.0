from typing import List, Dict, Any, Optional, Tuple
from .textutils import median

def group_spans_into_lines(spans: List[Dict[str, Any]], y_tol: float = 2.0) -> List[List[Dict[str, Any]]]:
    if not spans: return []
    ss = sorted(spans, key=lambda s: (float(s.get("y",0.0)), float(s.get("x",0.0))))
    lines, cur, cur_y = [], [], None
    for s in ss:
        y = float(s.get("y",0.0))
        if cur_y is None or abs(y - cur_y) <= y_tol:
            cur.append(s); cur_y = y if cur_y is None else (cur_y+y)/2.0
        else:
            lines.append(cur); cur = [s]; cur_y = y
    if cur: lines.append(cur)
    return lines

def detect_sup_sub_for_block(spans: List[Dict[str, Any]]) -> List[Optional[str]]:
    if not spans: return []
    SUPERS = {"¹","²","³","⁴","⁵","⁶","⁷","⁸","⁹","⁰"}
    SUBS   = {"₁","₂","₃","₄","₅","₆","₇","₈","₉","₀"}
    all_sizes = [float(s.get("size") or s.get("h") or 0.0) for s in spans]
    default_size = median([x for x in all_sizes if x>0]) or 10.0
    lines = group_spans_into_lines(spans, y_tol=2.0)
    idx_map = {id(s): i for i,s in enumerate(spans)}
    out = [None]*len(spans)
    soft = 0.10; small = 0.90

    for line in lines:
        baselines, sizes = [], []
        for s in line:
            sz = float(s.get("size",0.0)) or default_size
            y  = float(s.get("y",0.0))
            ref = min(sz, default_size)
            baselines.append(y + ref*0.8)
            sizes.append(sz)
        line_base = median(baselines)
        line_size = median([z for z in sizes if z>0]) or default_size

        for s, sz, base in zip(line, sizes, baselines):
            i = idx_map[id(s)]
            t = (s.get("text") or "").strip()
            delta = base - line_base
            tag = None
            if t in SUPERS: tag = "sup"
            elif t in SUBS: tag = "sub"
            else:
                if sz < small*line_size:
                    if delta < -soft*line_size: tag = "sup"
                    elif delta >  soft*line_size: tag = "sub"
            out[i] = tag
    return out

def line_bbox(line_runs: List[Dict[str, Any]]) -> Tuple[float,float,float,float]:
    xs=[]
    for r in line_runs:
        try:
            x=float(r.get("x",0)); y=float(r.get("y",0))
            w=float(r.get("w",0)); h=float(r.get("h",0))
            xs.append((x,y,w,h))
        except Exception:
            pass
    if not xs: return (0.0,0.0,0.0,0.0)
    x0=min(x for x,_,_,_ in xs); y0=min(y for _,y,_,_ in xs)
    x1=max(x+w for x,_,w,_ in xs); y1=max(y+h for _,y,_,h in xs)
    return (x0,y0,x1-x0,y1-y0)
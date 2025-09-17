import difflib, re
from typing import List, Dict, Any, Tuple
from .textutils import allcaps_ratio, font_base, looks_like_new_list_start, median

def detect_body_size(blocks):
    sizes = [b.get("size",0.0) for b in blocks if len((b.get("text") or ""))>40 and b.get("size",0)>0] or \
            [b.get("size",0.0) for b in blocks if b.get("size",0)>0]
    if not sizes: return 11.0
    return median(sizes)

def score_heading(b, page_w, body_size, thresh_regex_bonus=True):
    score = 0.0
    size = b.get("size",0.0); font = b.get("font",""); text = b.get("text","")
    centered = abs(((b["x"] + b["w"]/2) - (page_w/2))) < 20
    gap_above = b.get("_gap_above", 0.0)
    if size >= 1.30*body_size: score += 2.0
    if "Bold" in font or "Bd" in font or "bold" in font: score += 1.0
    if centered: score += 1.0
    if gap_above >= 12: score += 1.0
    if allcaps_ratio(text) >= 0.6: score += 0.5
    if re.match(r"^\s*\d+(?:\.\d+)*\s+\S", text): score += 0.8
    if thresh_regex_bonus and re.search(r"^(§\s*\d+|Einleitung|Inhaltsverzeichnis)$", text): score += 2.0
    if len(text) > 150: score -= 1.0
    return score

def assign_heading_levels(blocks):
    sizes = sorted({b["size"] for b in blocks if b.get("kind")=="heading" and b.get("size",0)>0}, reverse=True)
    tier = {sz: i+1 for i,sz in enumerate(sizes)}
    for b in blocks:
        if b.get("kind")=="heading": b["level"] = tier.get(b["size"], 1)

def similarity(a,b):
    a=(a or "").strip().lower(); b=(b or "").strip().lower()
    return difflib.SequenceMatcher(a=a,b=b).ratio()

def find_anchor(blocks, start_page, start_heading, anchor_tolerance):
    if not start_heading:
        for i,b in enumerate(blocks):
            if b["page_index"]+1 >= start_page and b.get("kind")=="heading":
                return i, b.get("level",1), 1.0
        return -1, None, 0.0
    best_i, best_sim = -1, 0.0
    for i,b in enumerate(blocks):
        if b["page_index"]+1 < start_page: continue
        if b.get("kind")!="heading": continue
        sim = similarity(start_heading, b["text"])
        if sim > best_sim: best_sim, best_i = sim, i
    if best_i>=0 and best_sim>=anchor_tolerance:
        return best_i, blocks[best_i].get("level",1), best_sim
    return -1, None, best_sim

def page_sort_key(b): return (b["page_index"], b["y"], b["x"])

def merge_flow_blocks(blocks, gap_px=6.0, size_tol=0.6, x_slop=40.0):
    if not blocks: return blocks
    blocks = sorted(blocks, key=page_sort_key)
    out=[]; i=0
    while i < len(blocks):
        cur = blocks[i]; i += 1
        if cur.get("kind")!="text" or cur.get("role")!="body":
            out.append(cur); continue
        while i < len(blocks):
            nxt = blocks[i]
            if not (nxt.get("kind")=="text" and nxt.get("role")=="body" and nxt["page_index"]==cur["page_index"]):
                break
            if looks_like_new_list_start(nxt.get("text","")): break
            gap = max(0.0, nxt["y"] - (cur["y"]+cur["h"]))
            same_col = (abs(nxt["x"]-cur["x"]) <= x_slop) or (nxt["x"] >= cur["x"])
            if gap > gap_px or not same_col: break
            fs_cur, fs_nxt = float(cur.get("size") or 0.0), float(nxt.get("size") or 0.0)
            if fs_cur and fs_nxt and abs(fs_cur-fs_nxt) > size_tol: break
            if font_base(cur.get("font","")) and font_base(nxt.get("font","")) and \
               font_base(cur.get("font","")) != font_base(nxt.get("font","")): break
            i += 1
            t1=(cur.get("text","") or "").rstrip(); t2=(nxt.get("text","") or "").lstrip()
            joined = (t1[:-1]+t2) if (t1.endswith("-") and (t2[:1].islower() or t2[:1].isdigit())) else (t1+" "+t2)
            cur["text"] = re.sub(r"\s+", " ", joined).strip()
            cur["runs"] = (cur.get("runs") or []) + (nxt.get("runs") or [])
            x0=min(cur["x"],nxt["x"]); y0=min(cur["y"],nxt["y"])
            x1=max(cur["x"]+cur["w"], nxt["x"]+nxt["w"]); y1=max(cur["y"]+cur["h"], nxt["y"]+nxt["h"])
            cur.update({"x":x0,"y":y0,"w":x1-x0,"h":y1-y0})
        out.append(cur)
    compute_gap_above(out)
    return out

def compute_gap_above(sorted_blocks):
    prev=None
    for b in sorted_blocks:
        if prev is None or prev["page_index"]!=b["page_index"]:
            b["_gap_above"]=999.0
        else:
            b["_gap_above"]=max(0.0, b["y"] - (prev["y"]+prev["h"]))
        prev=b
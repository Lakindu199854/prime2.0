from typing import Dict, Any, List, Optional, Tuple
from lxml import etree

def ensure_page_node(doc_pages_el: etree._Element, printed_index: int,
                     width: Optional[str], height: Optional[str]) -> etree._Element:
    existing = doc_pages_el.find(f"./page[@index='{printed_index}']")
    if existing is not None: return existing
    page_el = etree.SubElement(doc_pages_el, "page", index=str(printed_index))
    if width:  page_el.set("width", width)
    if height: page_el.set("height", height)
    etree.SubElement(page_el, "header")
    etree.SubElement(page_el, "body")
    etree.SubElement(page_el, "footer")
    return page_el

def add_separator(body_el: etree._Element, src_block: Dict[str,Any]):
    etree.SubElement(body_el, "separator",
        kind="rule",
        x=f"{src_block.get('x',0)}", y=f"{src_block.get('y',0)}",
        w=f"{src_block.get('w',0)}", h=f"{src_block.get('h',0)}"
    )

def copy_block_coords_and_style(dst: etree._Element, src: Dict[str,Any]):
    for attr in ("x","y","w","h","font","size","role"):
        if attr in src and src[attr] is not None:
            dst.set(attr, f"{src[attr]}")

def bbox_union(items: List[etree._Element]) -> Optional[Tuple[float,float,float,float]]:
    xs=[]
    for it in items:
        try:
            x=float(it.get("x","0")); y=float(it.get("y","0"))
            w=float(it.get("w","0")); h=float(it.get("h","0"))
            xs.append((x,y,w,h))
        except Exception: pass
    if not xs: return None
    x0=min(x for x,_,_,_ in xs); y0=min(y for _,y,_,_ in xs)
    x1=max(x+w for x,_,w,_ in xs); y1=max(y+h for _,y,_,h in xs)
    return (x0,y0,x1-x0,y1-y0)

def copy_inline_spans_from_runs(dst_parent: etree._Element, runs: List[Dict[str,Any]], emit_spans: bool):
    if not emit_spans or not runs: return
    if any((r.get("text") or "").strip() for r in runs): dst_parent.text = None
    for r in runs:
        sp = etree.SubElement(dst_parent, "span",
            bold="1" if r.get("bold") else "0",
            italic="1" if r.get("italic") else "0",
            sup=("1" if r.get("sup")=="sup" else ("-1" if r.get("sup")=="sub" else "0")),
            size=f"{float(r.get('size',0.0)):.2f}",
            x=f"{float(r.get('x',0.0)):.2f}", y=f"{float(r.get('y',0.0)):.2f}",
            w=f"{float(r.get('w',0.0)):.2f}", h=f"{float(r.get('h',0.0)):.2f}",
            font=r.get("font","")
        ); sp.text = r.get("text","")
import re
from typing import Dict, Any, List

def detect_tables_from_text_blocks(blocks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Fallback table detector from text blocks (best effort)."""
    tables = []
    for block in blocks:
        text = block.get("text", "")
        if re.search(r"(Jahr|<40 qm|ab \d+ bis <\d+|€)", text):
            table = {
                "x": block["x"], "y": block["y"], "w": block["w"], "h": block["h"],
                "rows": text.count("\n") + 1,
                "cols": max(1, len(re.findall(r"\s{2,}", text.split("\n")[0])) + 1),
                "page_index": block["page_index"],
                "content": text
            }
            tables.append(table)
    return tables
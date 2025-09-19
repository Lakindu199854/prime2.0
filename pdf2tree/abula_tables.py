# pdf2tree/tabula_tables.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd

def _dfs_to_grid(dfs: List[pd.DataFrame]) -> List[List[str]]:
    grids = []
    for df in dfs or []:
        if df is None or df.empty:
            continue
        # ensure strings, no NaNs
        grids.append(df.astype(str).fillna("").values.tolist())
    return grids

def extract_with_tabula(pdf_path: Path, start_pdf_idx: int, end_pdf_idx: int,
                        flavor: str = "auto") -> List[Dict[str, Any]]:
    """
    Return a list of table dicts:
       {"page_index": <0-based>, "rows": R, "cols": C, "grid": [[...], ...]}
    Coordinates are not available from tabula-py -> omit x/y/w/h.
    """
    try:
        import tabula  # requires Java
    except Exception as e:
        raise RuntimeError("tabula-py not installed (pip install tabula-py)") from e

    out: List[Dict[str, Any]] = []

    def read_page(page_no: int, lattice: bool, stream: bool):
        return tabula.read_pdf(
            pdf_path.as_posix(),
            pages=page_no,                  # 1-based for tabula
            multiple_tables=True,
            lattice=lattice,
            stream=stream,
            guess=True
        ) or []

    for pdf_i in range(start_pdf_idx, end_pdf_idx + 1):
        page_no = pdf_i + 1  # tabula is 1-based
        # try both flavors and pick the richer result
        t_lat = read_page(page_no, lattice=True,  stream=False)
        t_str = read_page(page_no, lattice=False, stream=True)

        def score(dfs):  # crude: prefer non-empty & more columns
            s = 0
            for df in dfs:
                if df is not None and not df.empty:
                    s += 1 + df.shape[1]
            return s

        chosen = t_lat if (flavor == "lattice" or (flavor == "auto" and score(t_lat) >= score(t_str))) \
                 else t_str

        for grid in _dfs_to_grid(chosen):
            rows = len(grid)
            cols = max((len(r) for r in grid), default=0)
            out.append({
                "page_index": pdf_i,
                "rows": rows,
                "cols": cols,
                "grid": grid
                # note: no x/y/w/h from tabula
            })
    return out

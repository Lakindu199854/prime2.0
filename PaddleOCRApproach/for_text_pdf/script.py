from pathlib import Path
from paddleocr import PPStructureV3

pdf_path = Path("A6.pdf").resolve()     # safer than a bare string
out_dir  = Path("out")
out_dir.mkdir(exist_ok=True)

pp = PPStructureV3(
    # document preprocessing
    use_doc_orientation_classify=False,   # toggle on if your pages are rotated
    use_doc_unwarping=False,              # set True only for warped scans (slower)

    # structure recognition toggles
    use_table_recognition=True,           # detect and parse tables
    use_formula_recognition=True,         # detect formulas/equations
    use_chart_recognition=False,          # set True if you want charts parsed
    use_region_detection=True             # detect text/headers/footers/paragraphs
    # device="gpu"                        # uncomment if you installed paddlepaddle-gpu
)

results = pp.predict(str(pdf_path))     # returns an iterator over pages

for page in results:
    # page.print()  # optional; prints block_label, bbox, etc.
    page.save_to_json(str(out_dir))      # structured JSON per page
    page.save_to_markdown(str(out_dir))  # markdown version per page

print("Done. Saved to:", out_dir)

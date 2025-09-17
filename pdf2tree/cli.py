import argparse, json, sys
from pathlib import Path
from .pipeline import build_tree_direct  # if not using a package, use: from pipeline import build_tree_direct

def main(argv=None):
    ap = argparse.ArgumentParser(description="Direct PDF -> Tree-XML (no intermediate).")
    ap.add_argument("pdf", type=Path, help="Input PDF")
    ap.add_argument("--metadata", type=Path, default=None, help="Metadata XML (optional)")
    ap.add_argument("--out-tree", type=Path, default=Path("out_tree.xml"), help="Output tree XML")

    # Start position / printed page mapping
    ap.add_argument("--start-page", type=int, default=None, help="Start page (PDF, 1-based)")
    ap.add_argument("--start-printed", type=int, default=None, help="Start printed page (e.g., 229)")
    ap.add_argument("--printed-first", type=int, default=None, help="Printed page number of PDF's first page (e.g., 213)")
    ap.add_argument("--no-infer-printed", action="store_true", help="Disable filename-based inference *_213_257*")

    # Content & heuristics
    ap.add_argument("--start-heading", type=str, default=None, help="Anchor heading text (overrides metadata)")
    ap.add_argument("--lang", type=str, default="de", help="xml:lang (unused in output)")
    ap.add_argument("--heading-threshold", type=float, default=3.0, help="Score ≥ threshold ⇒ heading")
    ap.add_argument("--anchor-tolerance", type=float, default=0.75, help="Fuzzy match 0..1")
    ap.add_argument("--max-pages", type=int, default=None, help="Process at most N pages from start")
    ap.add_argument("--no-tables", action="store_true", help="Disable native table detection (use fallback only)")

    # Debug / spans
    ap.add_argument("--debug-csv", type=Path, default=None, help="Write post-merge blocks CSV")
    ap.add_argument("--debug-jsonl", type=Path, default=None, help="Write post-merge blocks JSONL")
    ap.add_argument("--emit-spans", action="store_true", help="Emit inline <span> with bold/italic/sup/sub")

    args = ap.parse_args(argv)
    if not args.pdf.exists():
        print(f"ERROR: PDF not found: {args.pdf}", file=sys.stderr)
        return 1

    infer_printed = not args.no_infer_printed
    summary = build_tree_direct(
        pdf_path=args.pdf,
        metadata_path=args.metadata,
        start_page_cli=args.start_page,
        start_heading_cli=args.start_heading,
        lang=args.lang,
        heading_threshold=args.heading_threshold,
        anchor_tolerance=args.anchor_tolerance,
        max_pages=args.max_pages,
        no_tables=args.no_tables,
        debug_csv=args.debug_csv,
        debug_jsonl=args.debug_jsonl,
        start_printed=args.start_printed,
        printed_first_cli=args.printed_first,
        infer_printed=infer_printed,
        emit_spans=args.emit_spans,
        out_xml=args.out_tree,
    )

    print(json.dumps({"wrote_tree": str(args.out_tree), **summary}, ensure_ascii=False))
    return 0

if __name__ == "__main__":
    sys.exit(main())

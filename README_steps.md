# PDF → Intermediate → Final (Validated)

End‑to‑end pipeline to reproduce your **final XML** from a German **PDF** + **metadata XML** using an **intermediate model**, a **declarative mapping (YAML)**, and **schema validation (XSD)**.

---

## Files in this folder

- **HE_JMBL_2025_8_213_257.07.2025-veroeffentlichung.pdf** — input PDF.
- **HE_00901_2025.07.31_2026.01.01.xml** — input metadata (source of start page / heading, validity, etc.).
- **intermediate_builder.py** — builds **intermediate.xml** (annotated page model with blocks/tables/images).
- **mapping_spec.yaml** — declarative mapping rules from **intermediate.xml** → **final.xml**.
- **map_to_final.py** — executes `mapping_spec.yaml` to create **final.xml**.
- **final_schema_min.xsd** — XSD to validate **final.xml**.
- **validate.py** — validates **final.xml** against the XSD.

---

## Install (once)

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install --upgrade pip
pip install PyMuPDF lxml PyYAML
# Optional for tougher tables (use one):
# pip install camelot-py[cv]
# pip install pdfplumber
```

---

## Quickstart (your real files)

1) **Extract → intermediate.xml**

```bash
python intermediate_builder.py   HE_JMBL_2025_8_213_257.07.2025-veroeffentlichung.pdf   --metadata HE_00901_2025.07.31_2026.01.01.xml   --out intermediate.xml   --start-page 229   --start-heading "Verwaltungsvorschrift über die Berichtspflichten im Rahmen des Vollzugs lebenslanger Freiheitsstrafen in Hessen"
```

2) **Map → final.xml**

```bash
python map_to_final.py intermediate.xml mapping_spec.yaml --out final.xml
```

3) **Validate**

```bash
python validate.py final.xml final_schema_min.xsd
# Expect: "OK: XML is valid."  (or a list of errors)
```

---

## What is `intermediate.xml`? (shape & purpose)

A **clean, annotated snapshot** of the PDF used as staging before tagging to the final schema. You get **kind/level**, fonts, sizes, and **bbox** for positional rules.

```xml
<doc xml:lang="de">
  <meta>
    <source>HE_JMBL_2025_8_213_257.07.2025-veroeffentlichung.pdf</source>
    <start page="229" heading="…"/>
  </meta>
  <page index="229" width="595.00" height="842.00">
    <block kind="heading" level="1" font="…" size="22.0" x="70.8" y="213.6" w="480.0" h="20.1">Verwaltungsvorschrift …</block>
    <block kind="text"    font="…" size="11.0" x="70.8" y="240.0" w="480.0" h="12.5">Vom 16. Juli 2025 (JMBl. …)</block>
    <table x="84.7" y="385.0" w="441.7" h="130.0" rows="5" cols="3">
      <tr><td>…</td><td>…</td><td>…</td></tr>
    </table>
    <image x="…" y="…" w="…" h="…"/>
  </page>
</doc>
```

**Why it matters:** decouples PDF quirks from final tags, lets you write rules like “first text **below** H1 within 200px”.

---

## Mapping rules (YAML) — quick cheat sheet

Each rule declares **where to write** in the final XML and **how to find** the source in the intermediate.

```yaml
- path: /gesetz/rumpf/einzelvorschrift/ueberschrift   # target node (created if needed)
  source: block                                       # block | table | image | metadata (block is common)
  where:                                              # locator
    kind: "heading"                                   # filter by attributes
    level: 1
    regex: "^Verwaltungsvorschrift"
    below: { ref: "/gesetz/rumpf/einzelvorschrift/ueberschrift", within_px: 200 }
  attributes: { herausgeber: "amtlich" }              # set attributes on target
  take: text                                          # text | literal (literal uses rule.text)
  required: true                                      # fail build if not matched
```

The provided `map_to_final.py` (minimal version) supports:
- `where.kind`, `where.level`, `where.regex`
- `where.below` → positional constraint relative to a previously mapped `path`
- `attributes`, `take: text|literal`, `required`

> Need `between`, "table_as_rows", numbered lists, fuzzy matching in mapping, etc.? We can extend `map_to_final.py` when you’re ready.

---

## Troubleshooting

- **No match for required rule** (mapper exits): open `intermediate.xml` and check that the block exists with the expected `kind/level/regex`. Adjust `mapping_spec.yaml` or tweak the builder’s heading heuristics.
- **Validation fails**: check the error line/column; verify your `final.xml` tag/attribute names against `final_schema_min.xsd`.
- **Start anchor not found**: relax `--start-heading` text (it’s fuzzy matched at ~0.75) or omit it and rely on `--start-page` only.
- **Tables missing**: PyMuPDF’s `find_tables()` may not detect all tables. Consider installing `camelot-py[cv]` or `pdfplumber` and extending the builder.

---

## Suggested next steps

- Keep a **golden**: `cp final.xml golden_final.xml` and diff in CI to catch regressions.
- Promote `script.py` → `intermediate_builder.py` usage to keep the stack consistent.
- If you want a one‑liner UX, add a `Makefile` with `make build`, `make validate` targets.

---

## Command reference

```bash
# Build intermediate (anchor on page/heading; stop at next same/higher heading)
python intermediate_builder.py INPUT.pdf --metadata META.xml --start-page N --start-heading "…"

# Map to final
python map_to_final.py intermediate.xml mapping_spec.yaml --out final.xml

# Validate
python validate.py final.xml final_schema_min.xsd
```
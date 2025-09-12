#!/usr/bin/env python3
import argparse
from pathlib import Path
from lxml import etree

def main():
    ap = argparse.ArgumentParser(description="Validate final.xml against XSD")
    ap.add_argument("xml", type=Path)
    ap.add_argument("xsd", type=Path)
    args = ap.parse_args()

    xml_doc = etree.parse(str(args.xml))
    with open(args.xsd, "rb") as f:
        schema_doc = etree.parse(f)
    schema = etree.XMLSchema(schema_doc)

    ok = schema.validate(xml_doc)
    if ok:
        print("OK: XML is valid.")
    else:
        print("INVALID: XML failed validation.")
        for e in schema.error_log:
            print(f"- {e}")

if __name__ == "__main__":
    main()
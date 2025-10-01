import os, time
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# ---------- CONFIG ----------
PDF_PATH   = "input.pdf"            # Path to your PDF file
FILE_ID    = None                   # Reuse uploaded file id if you have one, else keep None
OUT_FILE   = "output.xml"           # Where the XML output will be saved
MAX_PARTS  = 12                     # Safety cap for multi-part outputs
MAX_TOKENS = 8000                   # Tokens per part (adjust if needed)

INITIAL_PROMPT = (
  "Convert the PDF into a correct XML structure. Capture headers, footers, heading levels, paragraphs, "
  "tables (rows/cols), figures/charts (as <Figure> with <Caption>/<Source>), lists, footnotes, and "
  "inline styles (bold/italic/superscript/subscript) only if explicitly present.\n"
  "Do not add or improvise data. Include a <Page> tag for each page.\n"
  "When there are images, insert a pointer like <Image src=\"images/xxx.jpg\"/> (actual extraction done later).\n"
  "Start with <Document> and end with </Document>. If output too long, stop cleanly and end with <!-- CONTINUE -->.\n"
)

CONTINUE_PROMPT = """
Continue the SAME XML from where you stopped.
- Do NOT repeat any content already produced.
- Follow the exact same rules and schema as before.
- If you finish the entire document, end with </Document>.
- If you still have more, end with <!-- CONTINUE -->.
Return ONLY XML (no prose).
"""

# ---------- FUNCTIONS ----------

def get_client():
    load_dotenv()
    endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "").rstrip("/")
    api_key  = os.getenv("AZURE_OPENAI_API_KEY")
    if not endpoint or not api_key:
        raise SystemExit("❌ Set AZURE_OPENAI_ENDPOINT and AZURE_OPENAI_API_KEY in .env")
    base_url = f"{endpoint}/openai/v1/"
    return OpenAI(api_key=api_key, base_url=base_url)


def ensure_file_uploaded(client: OpenAI, pdf_path: str, known_id: str|None) -> str:
    if known_id:
        return known_id
    with open(pdf_path, "rb") as f:
        up = client.files.create(file=f, purpose="assistants")
    # brief poll until processed
    for _ in range(10):
        fi = client.files.retrieve(up.id)
        if getattr(fi, "status", None) == "processed":
            break
        time.sleep(0.6)
    print(f"[upload] file_id={up.id} status={getattr(fi,'status',None)} bytes={getattr(fi,'bytes',None)}")
    return up.id


def ask(client: OpenAI, model: str, file_id: str, user_text: str, max_tokens: int):
    """Non-streaming call to Responses API"""
    resp = client.responses.create(
        model=model,
        max_output_tokens=max_tokens,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_file", "file_id": file_id},
                {"type": "input_text", "text": user_text},
            ],
        }],
    )

    # Collect all text safely
    chunks = []
    for item in resp.output:
        if not getattr(item, "content", None):
            continue
        for c in item.content:
            if getattr(c, "type", None) == "output_text":
                chunks.append(c.text)

    text = "".join(chunks)
    rid  = resp.id
    tot  = getattr(resp.usage, "total_tokens", 0)
    return text, rid, tot


# ---------- MAIN ----------

def main():
    client = get_client()
    model  = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-5-mini")

    if not FILE_ID and not Path(PDF_PATH).exists():
        raise SystemExit("❌ Set PDF_PATH or FILE_ID")

    file_id = ensure_file_uploaded(client, PDF_PATH, FILE_ID)

    out = Path(OUT_FILE)
    out.write_text("", encoding="utf-8")  # start fresh

    prompt = INITIAL_PROMPT
    total_tokens, response_ids = 0, []

    for part in range(1, MAX_PARTS + 1):
        print(f"[part {part}] generating...")
        text, rid, used = ask(client, model, file_id, prompt, MAX_TOKENS)

        response_ids.append(rid)
        total_tokens += used or 0

        # append output
        with out.open("a", encoding="utf-8") as f:
            f.write(text)

        if "<!-- CONTINUE -->" in text.upper():
            prompt = CONTINUE_PROMPT
            continue
        if text.strip().endswith("</Document>"):
            print(f"[done] Found closing </Document>")
            break
        else:
            print(f"[done] No continuation marker, stopping")
            break

    print(f"[saved] {out.resolve()}")
    print(f"[usage] total_tokens={total_tokens}")
    print(f"[responses] {response_ids}")


if __name__ == "__main__":
    main()

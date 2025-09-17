import re
from collections import Counter

# ---------- Regex (public) ----------
DECIMAL_RE         = re.compile(r"^\s*(\d+)\.\s+(.*)")
DECIMAL_START_ONLY = re.compile(r"^\s*\d+\.\s+")
ALPHA_RE           = re.compile(r"^\s*([a-z])\)\s+(.*)")
ALPHA_START_ONLY   = re.compile(r"^\s*[a-z]\)\s+", re.I)
RULE_RE            = re.compile(r"[_\-]{5,}\s*$")
PAGE_NUM_RE        = re.compile(r"^\s*(\d{1,4})\s*$")
FOOTNOTE_LINE_RE   = re.compile(r"^\s*(\d+)\s+(.+)")
BULLET_RE          = re.compile(r"^[\s\u00A0]*([•·▪‣\-\u2010\u2011\u2012\u2013\u2014\u2212])[\s\u00A0]+(.*)")
BULLET_START_ONLY  = re.compile(r"^[\s\u00A0]*[•·▪‣\-\u2010\u2011\u2012\u2013\u2014\u2212][\s\u00A0]+")

# ---------- Small helpers ----------
def looks_like_new_list_start(text: str) -> bool:
    t = (text or "").strip()
    return bool(
        DECIMAL_START_ONLY.match(t) or ALPHA_START_ONLY.match(t) or BULLET_START_ONLY.match(t)
    )

def font_base(font_name: str) -> str:
    f = (font_name or "").lower()
    for pat in ("-bold", "bold", "-bd", " bd", "-it", "italic", "oblique"):
        f = f.replace(pat, "")
    return f.strip()

def is_bold_font(font_name: str) -> bool:
    f = (font_name or "").lower()
    return ("bold" in f) or f.endswith("-bd") or "-bd" in f or "bd" == f[-2:]

def is_italic_font(font_name: str) -> bool:
    f = (font_name or "").lower()
    return ("italic" in f) or ("oblique" in f) or f.endswith("-it") or "-it" in f

def most_common(items):
    if not items: return ""
    return Counter(items).most_common(1)[0][0]

def median(nums):
    if not nums: return 0.0
    xs = sorted(nums); n = len(xs); m = n//2
    return xs[m] if n%2 else 0.5*(xs[m-1]+xs[m])

def allcaps_ratio(s):
    letters = [ch for ch in s if ch.isalpha()]
    if not letters: return 0.0
    caps = sum(1 for ch in letters if ch.isupper())
    return caps/len(letters)

def normalize_text(s):
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"-\n(?=[a-zäöüß])", "", s)
    s = s.replace("\n", " ")
    s = s.replace("ﬁ", "fi").replace("ﬂ", "fl")
    s = re.sub(r"\s+", " ", s).strip()
    return s
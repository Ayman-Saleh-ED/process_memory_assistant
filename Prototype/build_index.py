"""Read the PDFs in docs/ and save one chunk per page (with section title) to rag/chunks.json.
Run once:  python build_index.py
"""
import collections
import json
import re
from pathlib import Path

from pypdf import PdfReader

TITLES = {
    "control_valve_handbook": "Emerson Control Valve Handbook (6th ed.)",
    "fisher_dvc6200": "Fisher FIELDVUE DVC6200 Instruction Manual",
    "siemens_sipart_ps2": "Siemens SIPART PS2 HART Operating Instructions",
    "rosemount_3051": "Rosemount 3051 Reference Manual",
    "rosemount_3144p": "Rosemount 3144P Reference Manual",
    "rosemount_8700m": "Rosemount 8700M Magnetic Flowmeter Manual",
    "isa_closed_loop": "ISA: Closed-loop control troubleshooting",
    "aiche_cep_loop": "AIChE CEP: Control loop troubleshooting",
}


def line_key(line):
    return re.sub(r"\d+", "#", line.strip().lower())


def clean_pages(pages):
    """Remove running headers / footers and page numbers, join broken lines and words."""
    # lines that repeat on more than 20% of the pages are headers or footers
    counts = collections.Counter()
    for p in pages:
        for key in set(line_key(x) for x in p.split("\n") if x.strip()):
            counts[key] += 1
    repeated = {k for k, c in counts.items() if len(pages) >= 10 and c > 0.2 * len(pages) and len(k) < 120}

    kept = []
    for p in pages:
        lines = [x for x in p.split("\n") if x.strip() and line_key(x) not in repeated]
        lines = [x for x in lines if not re.fullmatch(r"\s*\d{1,4}\s*", x)]
        kept.append("\n".join(lines))

    # a word split over two lines ("pres-\nsure") is joined if the joined word is common in the document
    words = collections.Counter(re.findall(r"[a-z]{3,}", " ".join(kept).lower().replace("-\n", " ")))

    def join(m):
        joined = m.group(1) + m.group(2)
        return joined if words[joined.lower()] >= 3 else m.group(1) + "-" + m.group(2)

    result = []
    for p in kept:
        p = re.sub(r"([A-Za-z]{2,})-\n([a-z]{2,})", join, p)
        result.append(re.sub(r"\s+", " ", p).strip())
    return result


def outline_entries(reader):
    """Bookmarks of the PDF as a list of (page, title)."""
    entries = []

    def walk(items):
        for item in items:
            if isinstance(item, list):
                walk(item)
            else:
                try:
                    entries.append((reader.get_destination_page_number(item) + 1, str(item.title).strip()))
                except Exception:
                    pass

    try:
        walk(reader.outline)
    except Exception:
        pass
    return entries


def section_of(entries, page):
    before = [title for p, title in entries if p <= page]
    return before[-1] if before else ""


chunks = []
for pdf in sorted(Path("docs").glob("*.pdf")):
    name = pdf.stem
    reader = PdfReader(str(pdf))
    pages = clean_pages([p.extract_text() or "" for p in reader.pages])
    entries = outline_entries(reader)
    for page_number, text in enumerate(pages, start=1):
        if len(text) < 150 or text.count("....") > 3:   # skip empty pages and tables of contents
            continue
        chunks.append({"doc": name, "title": TITLES.get(name, name),
                       "section": section_of(entries, page_number), "page": page_number, "text": text})
    print(name, len(reader.pages), "pages,", len(entries), "bookmarks")

Path("rag").mkdir(exist_ok=True)
Path("rag/chunks.json").write_text(json.dumps(chunks), encoding="utf-8")
print(len(chunks), "chunks saved")

"""Extract readable text from lecture PDFs (drops page numbers / repeated headers, re-flows lines)."""
import re
from collections import Counter

from pypdf import PdfReader

_PAGE_NUMBER = re.compile(r"^(page\s*)?#(\s*(of|/)\s*#)?$")
_BULLET = re.compile(r"^([•\-*–·]|\d+[.)])\s")


def _normalise(line: str) -> str:
    return re.sub(r"\d+", "#", line.strip().lower())


def _strip_repeated_lines(pages: list[list[str]]) -> list[list[str]]:
    repeated = set()
    if len(pages) >= 3:
        counts = Counter(_normalise(l) for lines in pages for l in set(lines) if l.strip())
        threshold = max(3, int(0.5 * len(pages)))
        repeated = {line for line, c in counts.items() if c >= threshold}
    cleaned = []
    for lines in pages:
        keep = []
        for line in lines:
            norm = _normalise(line)
            if norm in repeated or _PAGE_NUMBER.match(norm):
                continue
            keep.append(line)
        cleaned.append(keep)
    return cleaned


def reflow(lines: list[str]) -> str:
    """Join hard-wrapped lines back into paragraphs; keep headings and bullets separate."""
    out: list[str] = []
    for raw in lines:
        line = raw.strip()
        if not line:
            out.append("")
            continue
        prev = out[-1] if out else ""
        wrapped = (
            prev
            and len(prev) > 40
            and not re.search(r"[.!?:;]$", prev)
            and not _BULLET.match(line)
        )
        if wrapped:
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    return "\n".join(out)


def extract_text_from_pdf(file) -> str:
    """`file` can be a path or a file-like object (e.g. Streamlit's UploadedFile)."""
    reader = PdfReader(file)
    pages = [(page.extract_text() or "").splitlines() for page in reader.pages]
    pages = _strip_repeated_lines(pages)
    return "\n\n".join(reflow(lines) for lines in pages if lines)

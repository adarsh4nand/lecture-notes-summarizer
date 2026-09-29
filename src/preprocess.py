"""Text cleaning and sentence splitting (regex based, no NLTK downloads needed)."""
import re

_ABBREVIATIONS = ["e.g.", "i.e.", "etc.", "vs.", "Fig.", "Eq.", "Dr.", "Mr.", "Mrs.", "Prof.", "No."]
_SENTENCE_END = re.compile(r'(?<=[.!?])\s+(?=[A-Z0-9"\'(\[])')
_BULLET_PREFIX = "•*-–· \t"


def clean_text(text: str) -> str:
    """Normalise line endings and whitespace, and re-join words hyphenated across lines."""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _protect(line: str) -> str:
    for abbr in _ABBREVIATIONS:
        line = line.replace(abbr, abbr.replace(".", "<DOT>"))
    return line


def _restore(line: str) -> str:
    return line.replace("<DOT>", ".")


def split_sentences(text: str, min_words: int = 4) -> list[str]:
    """Split text into sentences. Each line is treated as a boundary (bullets, headings).
    Fragments shorter than `min_words` words are dropped."""
    sentences = []
    for line in clean_text(text).split("\n"):
        line = line.strip()
        if not line:
            continue
        for piece in _SENTENCE_END.split(_protect(line)):
            piece = _restore(piece).lstrip(_BULLET_PREFIX).strip()
            if len(piece.split()) >= min_words:
                sentences.append(piece)
    return sentences


def word_count(text: str) -> int:
    return len(text.split())

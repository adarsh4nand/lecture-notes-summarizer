import pathlib

from src.extractive import textrank_summary, tfidf_summary
from src.keywords import extract_keywords
from src.pdf_utils import _strip_repeated_lines, reflow
from src.preprocess import split_sentences

NOTES = (pathlib.Path(__file__).parent.parent / "sample_notes" / "searching_algorithms.txt").read_text(encoding="utf-8")


def test_split_sentences_handles_abbreviations():
    sents = split_sentences("Use binary search, e.g. on sorted lists. It is fast. Complexity is O(log n) for this case.")
    assert sents[0].startswith("Use binary search, e.g. on sorted lists")


def test_extractive_returns_requested_count():
    for fn in (tfidf_summary, textrank_summary):
        out = fn(NOTES, n_sentences=3, as_list=True)
        assert len(out) == 3


def test_extractive_keeps_document_order():
    sents = split_sentences(NOTES)
    out = textrank_summary(NOTES, n_sentences=4, as_list=True)
    positions = [sents.index(s) for s in out]
    assert positions == sorted(positions)


def test_short_text_and_empty_text():
    assert tfidf_summary("") == ""
    assert textrank_summary("Only one short sentence is here.", n_sentences=3) == "Only one short sentence is here."


def test_keywords_not_empty():
    assert len(extract_keywords(NOTES, 5)) > 0


def test_pdf_reflow_joins_wrapped_lines():
    lines = ["Binary search halves the search space at every single", "step of the algorithm.", "Next Topic"]
    assert reflow(lines).splitlines()[0].endswith("step of the algorithm.")


def test_pdf_strips_page_numbers_and_headers():
    pages = [["CS201 Notes", "Some content here", "1"], ["CS201 Notes", "More content here", "2"], ["CS201 Notes", "Final content here", "3"]]
    cleaned = _strip_repeated_lines(pages)
    assert all("CS201 Notes" not in p and not p[-1].isdigit() for p in cleaned)

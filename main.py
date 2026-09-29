"""Command-line interface.

    python main.py sample_notes/searching_algorithms.txt --method textrank --sentences 4
    python main.py my_lecture.pdf --method abstractive
"""
import argparse
import os

from src.abstractive import abstractive_summary
from src.extractive import textrank_summary, tfidf_summary
from src.keywords import extract_keywords
from src.pdf_utils import extract_text_from_pdf
from src.preprocess import word_count


def read_input(path: str) -> str:
    if path.lower().endswith(".pdf"):
        return extract_text_from_pdf(path)
    with open(path, encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Summarize lecture notes (.txt or .pdf)")
    parser.add_argument("file")
    parser.add_argument("--method", choices=["tfidf", "textrank", "abstractive"], default="textrank")
    parser.add_argument("--sentences", type=int, default=None, help="sentences to keep (extractive)")
    parser.add_argument("--ratio", type=float, default=0.2, help="fraction of sentences to keep if --sentences unset")
    parser.add_argument("--keywords", type=int, default=8, help="number of key terms to print (0 to disable)")
    parser.add_argument("--out", type=str, help="optional path to save the summary")
    args = parser.parse_args()

    text = read_input(args.file)
    if args.method == "tfidf":
        summary = tfidf_summary(text, args.sentences, args.ratio)
    elif args.method == "textrank":
        summary = textrank_summary(text, args.sentences, args.ratio)
    else:
        summary = abstractive_summary(text)

    print(summary)
    print(f"\n[{word_count(text)} words -> {word_count(summary)} words]")
    if args.keywords:
        print("Key terms:", ", ".join(extract_keywords(text, args.keywords)))
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(summary)


if __name__ == "__main__":
    main()

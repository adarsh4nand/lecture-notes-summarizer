"""Evaluate summarizers with ROUGE.

Usage:
    python -m src.evaluate --n 20                 # CNN/DailyMail sample (needs internet)
    python -m src.evaluate --n 20 --skip-abstractive
    python -m src.evaluate --custom sample_notes  # your own notes + reference summaries
"""
import argparse
import csv
import os
import time

from rouge_score import rouge_scorer

from .extractive import textrank_summary, tfidf_summary

_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
METRICS = ["rouge1", "rouge2", "rougeL"]


def rouge_scores(predictions: list[str], references: list[str]) -> dict:
    """Average ROUGE F1 (x100) over all (prediction, reference) pairs."""
    totals = {m: 0.0 for m in METRICS}
    for pred, ref in zip(predictions, references):
        result = _SCORER.score(ref, pred)
        for m in METRICS:
            totals[m] += result[m].fmeasure
    n = max(len(predictions), 1)
    return {m: round(100 * totals[m] / n, 2) for m in METRICS}


def build_methods(skip_abstractive: bool) -> dict:
    methods = {
        "TF-IDF (3 sent.)": lambda t: tfidf_summary(t, n_sentences=3),
        "TextRank (3 sent.)": lambda t: textrank_summary(t, n_sentences=3),
    }
    if not skip_abstractive:
        from .abstractive import abstractive_summary

        methods["DistilBART"] = lambda t: abstractive_summary(t, max_len=90, min_len=30)
    return methods


def run(articles: list[str], references: list[str], skip_abstractive: bool) -> list[dict]:
    rows = []
    for name, fn in build_methods(skip_abstractive).items():
        start = time.time()
        preds = [fn(a) for a in articles]
        elapsed = time.time() - start
        rows.append({"method": name, **rouge_scores(preds, references), "sec_per_doc": round(elapsed / len(articles), 2)})
    return rows


def load_cnn_dailymail(n: int):
    from datasets import load_dataset

    ds = load_dataset("abisee/cnn_dailymail", "3.0.0", split=f"test[:{n}]")
    return list(ds["article"]), list(ds["highlights"])


def load_custom(folder: str):
    """Pairs `name.txt` with `name_reference.txt` inside `folder`."""
    articles, references = [], []
    for fname in sorted(os.listdir(folder)):
        if fname.endswith(".txt") and not fname.endswith("_reference.txt"):
            ref_path = os.path.join(folder, fname[:-4] + "_reference.txt")
            if os.path.exists(ref_path):
                with open(os.path.join(folder, fname), encoding="utf-8") as f:
                    articles.append(f.read())
                with open(ref_path, encoding="utf-8") as f:
                    references.append(f.read())
    return articles, references


def save_and_print(rows: list[dict], out_prefix: str) -> None:
    os.makedirs("results", exist_ok=True)
    csv_path = f"results/{out_prefix}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    header = "| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | sec/doc |\n|---|---|---|---|---|"
    lines = [f"| {r['method']} | {r['rouge1']} | {r['rouge2']} | {r['rougeL']} | {r['sec_per_doc']} |" for r in rows]
    table = header + "\n" + "\n".join(lines)
    with open(f"results/{out_prefix}.md", "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nSaved to {csv_path}")


def main():
    parser = argparse.ArgumentParser(description="ROUGE evaluation of the summarizers")
    parser.add_argument("--n", type=int, default=20, help="number of CNN/DailyMail test articles")
    parser.add_argument("--custom", type=str, help="folder with name.txt + name_reference.txt pairs")
    parser.add_argument("--skip-abstractive", action="store_true")
    args = parser.parse_args()

    if args.custom:
        articles, references = load_custom(args.custom)
        prefix = "rouge_custom"
    else:
        articles, references = load_cnn_dailymail(args.n)
        prefix = "rouge_cnn_dailymail"
    if not articles:
        raise SystemExit("No evaluation documents found.")
    print(f"Evaluating on {len(articles)} documents...\n")
    save_and_print(run(articles, references, args.skip_abstractive), prefix)


if __name__ == "__main__":
    main()

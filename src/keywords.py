"""Simple keyword / key-term extraction using TF-IDF over sentences."""
from sklearn.feature_extraction.text import TfidfVectorizer

from .preprocess import split_sentences


def extract_keywords(text: str, top_n: int = 10) -> list[str]:
    sentences = split_sentences(text)
    if len(sentences) < 2:
        return []
    try:
        vec = TfidfVectorizer(stop_words="english", ngram_range=(1, 2), max_df=0.9)
        matrix = vec.fit_transform(sentences)
    except ValueError:
        return []
    scores = matrix.sum(axis=0).A1
    terms = vec.get_feature_names_out()
    ranked = sorted(zip(terms, scores), key=lambda t: -t[1])
    picked: list[str] = []
    for term, _ in ranked:
        # skip a unigram already covered by a chosen bigram (and vice versa)
        if any(term in p or p in term for p in picked):
            continue
        picked.append(term)
        if len(picked) == top_n:
            break
    return picked

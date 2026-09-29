"""Extractive summarizers: TF-IDF scoring and TextRank."""
import numpy as np
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .preprocess import split_sentences


def _resolve_n(total: int, n_sentences, ratio: float) -> int:
    if n_sentences is None:
        n_sentences = round(total * ratio)
    return max(1, min(int(n_sentences), total))


def _tfidf_matrix(sentences):
    return TfidfVectorizer(stop_words="english").fit_transform(sentences)


def _top_indices(scores, n: int) -> list[int]:
    """Indices of the n best scores, returned in original document order."""
    return sorted(np.argsort(scores)[::-1][:n].tolist())


def _output(chosen: list[str], as_list: bool):
    return chosen if as_list else " ".join(chosen)


def tfidf_summary(text: str, n_sentences=None, ratio: float = 0.2, as_list: bool = False):
    """Score each sentence by the sum of its TF-IDF weights (length-normalised)."""
    sentences = split_sentences(text)
    if not sentences:
        return _output([], as_list)
    n = _resolve_n(len(sentences), n_sentences, ratio)
    if len(sentences) <= n:
        return _output(sentences, as_list)
    try:
        matrix = _tfidf_matrix(sentences)
    except ValueError:  # empty vocabulary (e.g. only stop words)
        return _output(sentences[:n], as_list)
    nnz = np.maximum(matrix.getnnz(axis=1), 1)
    scores = matrix.sum(axis=1).A1 / np.sqrt(nnz)
    return _output([sentences[i] for i in _top_indices(scores, n)], as_list)


def textrank_summary(text: str, n_sentences=None, ratio: float = 0.2, as_list: bool = False):
    """Build a sentence-similarity graph and rank sentences with PageRank."""
    sentences = split_sentences(text)
    if not sentences:
        return _output([], as_list)
    n = _resolve_n(len(sentences), n_sentences, ratio)
    if len(sentences) <= n:
        return _output(sentences, as_list)
    try:
        matrix = _tfidf_matrix(sentences)
    except ValueError:
        return _output(sentences[:n], as_list)
    sim = cosine_similarity(matrix)
    np.fill_diagonal(sim, 0)
    graph = nx.from_numpy_array(sim)
    try:
        ranks = nx.pagerank(graph, weight="weight", max_iter=200)
    except nx.PowerIterationFailedConvergence:
        ranks = {i: float(sim[i].sum()) for i in range(len(sentences))}
    scores = np.array([ranks[i] for i in range(len(sentences))])
    return _output([sentences[i] for i in _top_indices(scores, n)], as_list)

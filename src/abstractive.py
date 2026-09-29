"""Abstractive summarization with a pretrained seq2seq model (BART / T5 family).

Long documents are split into chunks that fit the model's input window, each chunk is
summarized, and (optionally) the combined chunk summaries are summarized once more.
"""
from .preprocess import split_sentences

DEFAULT_MODEL = "sshleifer/distilbart-cnn-12-6"  # small & fast; try "facebook/bart-large-cnn" for quality
_MODEL_CACHE: dict = {}


def _load(model_name: str):
    if model_name not in _MODEL_CACHE:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        device = "cuda" if torch.cuda.is_available() else "cpu"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
        _MODEL_CACHE[model_name] = (tokenizer, model, device)
    return _MODEL_CACHE[model_name]


def chunk_sentences(sentences: list[str], max_words: int = 450) -> list[str]:
    """Group consecutive sentences into chunks of at most ~max_words words."""
    chunks, current, count = [], [], 0
    for sentence in sentences:
        words = len(sentence.split())
        if current and count + words > max_words:
            chunks.append(" ".join(current))
            current, count = [], 0
        current.append(sentence)
        count += words
    if current:
        chunks.append(" ".join(current))
    return chunks


def _summarize_chunk(text: str, model_name: str, max_len: int, min_len: int) -> str:
    import torch

    tokenizer, model, device = _load(model_name)
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    input_len = inputs["input_ids"].shape[1]
    max_len = max(16, min(max_len, int(input_len * 0.8)))
    min_len = min(min_len, max_len // 2)
    with torch.no_grad():
        ids = model.generate(
            **inputs,
            num_beams=4,
            max_length=max_len,
            min_length=min_len,
            length_penalty=2.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
        )
    return tokenizer.decode(ids[0], skip_special_tokens=True).strip()


def abstractive_summary(
    text: str,
    model_name: str = DEFAULT_MODEL,
    chunk_words: int = 450,
    max_len: int = 130,
    min_len: int = 30,
    second_pass: bool = True,
) -> str:
    sentences = split_sentences(text)
    if not sentences:
        return ""
    chunks = chunk_sentences(sentences, chunk_words)
    partials = [_summarize_chunk(c, model_name, max_len, min_len) for c in chunks]
    combined = " ".join(partials)
    if second_pass and len(chunks) > 1 and len(combined.split()) > chunk_words // 2:
        return _summarize_chunk(combined, model_name, max_len * 2, min_len)
    return combined

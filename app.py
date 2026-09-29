"""Streamlit app:  streamlit run app.py"""
import streamlit as st

from src.extractive import textrank_summary, tfidf_summary
from src.keywords import extract_keywords
from src.pdf_utils import extract_text_from_pdf
from src.preprocess import split_sentences, word_count

st.set_page_config(page_title="Lecture Notes Summarizer", page_icon="📝", layout="wide")
st.title("📝 Lecture Notes Summarizer")
st.caption("Extractive (TF-IDF, TextRank) and abstractive (DistilBART) summarization")

with st.sidebar:
    st.header("Settings")
    method = st.selectbox("Method", ["TextRank", "TF-IDF", "Abstractive (DistilBART)", "Compare all"])
    n_sentences = st.slider("Sentences to keep (extractive)", 2, 20, 5)
    n_keywords = st.slider("Key terms", 0, 20, 8)
    st.markdown("---")
    st.markdown("Abstractive mode downloads a model (~1 GB) the first time it runs.")

tab_text, tab_file = st.tabs(["Paste text", "Upload file"])
text = ""
with tab_text:
    pasted = st.text_area("Paste your notes", height=280)
    if pasted:
        text = pasted
with tab_file:
    upload = st.file_uploader("PDF or TXT", type=["pdf", "txt"])
    if upload is not None:
        try:
            text = extract_text_from_pdf(upload) if upload.name.lower().endswith(".pdf") else upload.read().decode("utf-8", errors="ignore")
            st.success(f"Loaded {upload.name} - {word_count(text)} words")
        except Exception as exc:  # corrupted / encrypted PDFs
            st.error(f"Could not read the file: {exc}")


@st.cache_resource(show_spinner=False)
def _abstractive(text_in: str) -> str:
    from src.abstractive import abstractive_summary

    return abstractive_summary(text_in)


def run_method(name: str, source: str) -> str:
    if name == "TF-IDF":
        return tfidf_summary(source, n_sentences)
    if name == "TextRank":
        return textrank_summary(source, n_sentences)
    with st.spinner("Running the transformer model (first run downloads it)..."):
        return _abstractive(source)


def show(name: str, summary: str, source: str) -> None:
    st.subheader(name)
    st.write(summary)
    in_w, out_w = word_count(source), word_count(summary)
    c1, c2, c3 = st.columns(3)
    c1.metric("Original words", in_w)
    c2.metric("Summary words", out_w)
    c3.metric("Compression", f"{(1 - out_w / max(in_w, 1)) * 100:.0f}%")
    st.download_button("Download summary", summary, file_name=f"summary_{name.split()[0].lower()}.txt", key=f"dl_{name}")


if st.button("Summarize", type="primary"):
    if len(split_sentences(text)) < 3:
        st.warning("Please provide a longer text (at least a few full sentences).")
    else:
        if method == "Compare all":
            cols = st.columns(3)
            for col, name in zip(cols, ["TF-IDF", "TextRank", "Abstractive (DistilBART)"]):
                with col:
                    show(name, run_method(name, text), text)
        else:
            show(method, run_method(method, text), text)
        if n_keywords:
            st.subheader("Key terms")
            st.write(" ".join(f"`{k}`" for k in extract_keywords(text, n_keywords)))

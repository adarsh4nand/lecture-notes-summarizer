# 📝 Lecture Notes Summarizer

Turn long lecture notes (pasted text, `.txt` or `.pdf`) into short summaries and key terms.
The project implements **two families of summarizers** and compares them with ROUGE:

| Approach | Methods | Idea |
|---|---|---|
| Extractive | TF-IDF, TextRank | Select the most important existing sentences |
| Abstractive | DistilBART (Hugging Face) | Generate new sentences with a pretrained transformer |

<!-- Add a screenshot or GIF of the Streamlit app here -->
<!-- ![demo](results/demo.gif) -->

## Features
- PDF and text input, with header/footer/page-number cleanup and line re-flowing
- TF-IDF and TextRank extractive summarizers (built from scratch on scikit-learn + NetworkX)
- Abstractive summarization with chunking so long notes fit the model's input window
- Key-term extraction
- Streamlit web app with side-by-side method comparison and summary download
- CLI (`main.py`) and ROUGE evaluation script (`src/evaluate.py`)
- Unit tests (`pytest`)

## Project structure
```
lecture-notes-summarizer/
├── app.py                 # Streamlit UI
├── main.py                # command-line interface
├── requirements.txt
├── src/
│   ├── preprocess.py      # cleaning + sentence splitting
│   ├── extractive.py      # TF-IDF and TextRank
│   ├── abstractive.py     # transformer summarizer with chunking
│   ├── pdf_utils.py       # PDF text extraction
│   ├── keywords.py        # key-term extraction
│   └── evaluate.py        # ROUGE evaluation
├── sample_notes/          # example notes + reference summary
├── tests/
├── notebooks/             # your experiments
└── results/               # evaluation tables and plots
```

## Setup
```bash
git clone https://github.com/adarsh4nand/lecture-notes-summarizer.git
cd lecture-notes-summarizer
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```
> The abstractive model (~1 GB) is downloaded from Hugging Face the first time it is used.
> If you only need the extractive methods, `numpy scikit-learn networkx pypdf` are enough.

## Usage
**Web app**
```bash
streamlit run app.py
```

**Command line**
```bash
python main.py sample_notes/searching_algorithms.txt --method textrank --sentences 4
python main.py my_lecture.pdf --method abstractive
```

**Evaluation**
```bash
python -m src.evaluate --n 20                  # CNN/DailyMail sample
python -m src.evaluate --custom sample_notes   # your own notes + *_reference.txt
```

**Tests**
```bash
pytest
```

## Results
Run the evaluation and paste your table here (ROUGE F1 x100):

| Method | ROUGE-1 | ROUGE-2 | ROUGE-L | sec/doc |
|---|---|---|---|---|
| TF-IDF (3 sent.) | 23.62 | 11.20 | 14.17 | 0.0 |
| TextRank (3 sent.) | 43.33 | 15.25 | 23.33 | 0.0 |
| DistilBART | 37.41 | 11.68 | 20.14 | 8.04 |

TextRank achieved the highest ROUGE scores among the three methods on the custom sample, while TF-IDF and TextRank were substantially faster than DistilBART. DistilBART produced abstractive summaries but required about 8 seconds per document on CPU.

## Limitations
- Abstractive models can produce statements that are not in the source (hallucination).
- Equations, tables and diagrams in PDFs are not understood, only extracted text is used.
- Scanned PDFs need OCR first; this project reads text-based PDFs only.
- ROUGE measures word overlap, not factual correctness or readability.
- The summary quality of the pretrained model depends on its news-article training data.

## Future improvements
- Fine-tune a model on lecture-style data
- Add OCR for scanned notes
- Generate revision questions from the summary
- Add BERTScore as an additional metric

## License
MIT

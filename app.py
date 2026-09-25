import os

import joblib
import streamlit as st
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

# set ROBERTA_MODEL to a local folder to skip the download from the Hub
ROBERTA_SOURCE = os.getenv("ROBERTA_MODEL", "mikaso67/anime-sentiment-roberta")

EXAMPLES = {
    "Negation": "Not boring at all, actually it was amazing.",
    "Mixed feelings": "I really liked this anime, but it was kinda shitty at the end.",
    "Polite no": "I'm quite doubting the fact that some people are liking this anime, it's very unique but yeah i'll pass",
    "Easy one": "This anime was absolutely a masterpiece, I cried at the ending.",
}

RESULTS = [
    ("TF-IDF + logistic regression", "0.928", "0.952"),
    ("RoBERTa, 256 tokens", "0.950", "0.968"),
    ("RoBERTa, 512 tokens", "0.966", "0.978"),
]

st.set_page_config(page_title="TF-IDF vs RoBERTa", page_icon="📊", layout="centered")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;700&family=JetBrains+Mono&display=swap');

#MainMenu, footer, header {visibility: hidden;}
.block-container {padding-top: 3rem; max-width: 760px;}
html, body, [class*="css"], .stMarkdown, textarea, button {font-family: 'Inter', sans-serif !important;}

.hero {font-size: 2.2rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 4px;}
.tagline {color: #9aa0ab; margin-bottom: 2rem; font-size: 1.02rem;}

.card {border: 1px solid #2a2f3a; border-radius: 14px; padding: 18px 20px 14px; background: #161a21; height: 100%;}
.card .model {font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.08em; color: #9aa0ab;}
.card .how {font-size: 0.85rem; color: #6c7280; margin-bottom: 14px;}
.card .verdict {font-size: 1.9rem; font-weight: 700; line-height: 1.1;}
.card.positive .verdict {color: #3fb950;}
.card.negative .verdict {color: #f85149;}
.card .conf {font-family: 'JetBrains Mono', monospace; font-size: 0.85rem; color: #9aa0ab; margin: 4px 0 16px;}

.scale {position: relative; height: 8px; border-radius: 4px;
        background: linear-gradient(90deg, #f85149 0%, #3a3f4b 50%, #3fb950 100%);}
.scale .marker {position: absolute; top: -5px; width: 4px; height: 18px; border-radius: 2px;
                background: #f4f1ec; transform: translateX(-50%);}
.ends {display: flex; justify-content: space-between; font-size: 0.72rem; color: #6c7280; margin-top: 6px;}

.split {margin-top: 18px; padding: 12px 16px; border-left: 3px solid #4c8dff;
        background: rgba(76,141,255,0.08); color: #d6d3cf; font-size: 0.92rem;}

.results {width: 100%; border-collapse: collapse; font-size: 0.9rem; margin-top: 6px;}
.results {border: none !important;}
.results td, .results th {padding: 8px 6px; border: none !important; border-bottom: 1px solid #2a2f3a !important; text-align: left;}
.results th {color: #9aa0ab; font-weight: 500;}
.results td.num {font-family: 'JetBrains Mono', monospace; text-align: right;}
.results th.num {text-align: right;}
.results tr.best td {color: #4c8dff; font-weight: 500;}

.stButton button[kind="primary"] {font-weight: 600;}
.card {margin-bottom: 8px;}
[data-testid="stExpander"] {margin-top: 1.5rem;}
.foot {margin-top: 2.5rem; color: #6c7280; font-size: 0.8rem;}
.foot a {color: #9aa0ab;}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_tfidf():
    return joblib.load("models/tfidf_vectorizer.joblib"), joblib.load("models/tfidf_logreg.joblib")


@st.cache_resource
def load_roberta():
    tokenizer = AutoTokenizer.from_pretrained(ROBERTA_SOURCE)
    model = AutoModelForSequenceClassification.from_pretrained(ROBERTA_SOURCE)
    model.eval()
    return tokenizer, model


def tfidf_positive_prob(text):
    vectorizer, clf = load_tfidf()
    return float(clf.predict_proba(vectorizer.transform([text]))[0][1])


def roberta_positive_prob(text):
    tokenizer, model = load_roberta()
    inputs = tokenizer(text, truncation=True, max_length=512, return_tensors="pt")
    with torch.no_grad():
        logits = model(**inputs).logits
    return float(torch.softmax(logits, dim=-1)[0][1])


def card(name, how, p_pos):
    label = "positive" if p_pos >= 0.5 else "negative"
    confidence = max(p_pos, 1 - p_pos)
    return f"""
<div class="card {label}">
  <div class="model">{name}</div>
  <div class="how">{how}</div>
  <div class="verdict">{label}</div>
  <div class="conf">{confidence:.0%} confident</div>
  <div class="scale"><div class="marker" style="left:{p_pos * 100:.1f}%"></div></div>
  <div class="ends"><span>negative</span><span>positive</span></div>
</div>"""


def use_example():
    choice = st.session_state.example
    if choice:
        st.session_state.review = EXAMPLES[choice]


st.markdown(
    '<div class="hero">TF-IDF vs RoBERTa</div>'
    '<div class="tagline">Sentiment analysis on anime reviews. Paste a review and compare '
    "a bag-of-words baseline with a fine-tuned transformer.</div>",
    unsafe_allow_html=True,
)

st.pills("Examples", list(EXAMPLES), key="example", on_change=use_example)
review = st.text_area(
    "Review",
    key="review",
    height=140,
    placeholder="Write a review in English...",
    label_visibility="collapsed",
)

if st.button("Read it", type="primary", use_container_width=True):
    if not review.strip():
        st.warning("Write something first.")
    else:
        with st.spinner("Reading..."):
            p_tfidf = tfidf_positive_prob(review)
            p_roberta = roberta_positive_prob(review)

        left, right = st.columns(2)
        left.markdown(card("TF-IDF + LogReg", "counts which words appear", p_tfidf), unsafe_allow_html=True)
        right.markdown(card("RoBERTa", "reads words in context", p_roberta), unsafe_allow_html=True)

        if (p_tfidf >= 0.5) != (p_roberta >= 0.5):
            st.markdown(
                '<div class="split">The two models disagree. This usually happens with negations '
                "or a \"but\" that flips the sentence: TF-IDF sees the words, RoBERTa sees how "
                "they relate.</div>",
                unsafe_allow_html=True,
            )

with st.expander("How the models compare on the test set"):
    rows = "".join(
        f'<tr class="{"best" if i == len(RESULTS) - 1 else ""}"><td>{name}</td>'
        f'<td class="num">{acc}</td><td class="num">{f1}</td></tr>'
        for i, (name, acc, f1) in enumerate(RESULTS)
    )
    st.markdown(
        f'<table class="results"><tr><th>Model</th><th class="num">Accuracy</th>'
        f'<th class="num">F1</th></tr>{rows}</table>',
        unsafe_allow_html=True,
    )
    st.caption(
        "Trained on MyAnimeList reviews rated 1-4 (negative) or 8-10 (positive). "
        "Mid-range ratings were left out, so the models have no \"mixed\" option and "
        "will force a verdict on nuanced reviews."
    )

st.markdown(
    '<div class="foot">Code and notebooks on '
    '<a href="https://github.com/mikaso67/anime-sentiment-analysis">GitHub</a></div>',
    unsafe_allow_html=True,
)

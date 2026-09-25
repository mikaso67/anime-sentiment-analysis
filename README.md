# Anime review sentiment analysis

Binary sentiment classification on MyAnimeList reviews. I compare a classic TF-IDF + logistic regression baseline with a fine-tuned RoBERTa, and wrap both in a small Streamlit app to see where they disagree.

## Results

| Model | Accuracy | F1 |
|---|---|---|
| TF-IDF + logistic regression | 0.928 | 0.952 |
| RoBERTa, 256 tokens, 2 epochs | 0.950 | 0.968 |
| **RoBERTa, 512 tokens, 2 epochs** | **0.966** | **0.978** |
| RoBERTa, 512 tokens, 3 epochs | 0.964 | 0.977 |

F1 is for the positive class. The baseline's F1 is the mean over 5-fold cross-validation (std 0.002). The baseline gets 0.84 F1 on the negative class, which is the harder one since negatives are only ~21% of the data.

A few things I took from this:

- **Sequence length matters more than epochs.** Reviews are long (median around 300 words), so cutting them at 256 tokens throws away a lot of text. Going to 512 gave the biggest gain.
- **A third epoch doesn't help.** F1 stays flat and validation loss goes back up, so the model starts to overfit. I kept 2 epochs.
- **RoBERTa handles negation, TF-IDF doesn't.** On *"Not boring at all, actually it was amazing"*, TF-IDF sees "boring" and predicts negative. RoBERTa gets it right.

Note that the two models aren't evaluated on exactly the same test set: RoBERTa was fine-tuned on a stratified 10k subset (to keep GPU time reasonable), so its test set is 2,000 reviews versus 5,514 for the baseline. Same distribution, same labeling, but not a strict head-to-head.

## Data

[MyAnimeList Comment Dataset V2](https://www.kaggle.com/datasets/natlee/myanimelist-comment-dataset-v2) on Kaggle, about 224k reviews. The raw file is ~600 MB so it's not in the repo.

Reviews come with a 1-10 rating, not a sentiment label, so I built the labels myself:

- 1 to 4 → negative
- 8 to 10 → positive
- 5 to 7 → dropped

Mid-range reviews are usually mixed ("liked it but the ending ruined it"), and keeping them adds noise more than signal. This leaves 27,570 reviews from a 40k sample, 79% positive.

## Limitations

- **No "mixed" class.** Because 5-7 ratings were dropped, the models always force a positive/negative verdict, even on reviews that are genuinely mixed.
- **Indirect phrasing still fools RoBERTa.** *"I'm quite doubting the fact that some people are liking this anime... but yeah i'll pass"* is classified positive with high confidence. The negative part is a short idiom at the end, surrounded by positive vocabulary.
- **Truncation.** Even at 512 tokens, the longest reviews get cut.

## Repo structure

```
01_baseline_tfidf.ipynb       labeling, TF-IDF baseline, evaluation, cross-validation
02_roberta_finetuning.ipynb   RoBERTa fine-tuning on Colab (T4 GPU), length/epoch experiments
app.py                        Streamlit app comparing both models
models/                       saved TF-IDF vectorizer and classifier
```

## Running it

```bash
git clone https://github.com/mikaso67/anime-sentiment-analysis.git
cd anime-sentiment-analysis
pip install -r requirements.txt
```

To reproduce the baseline, download the dataset from Kaggle, put `reviews.csv` in `data/`, and run `01_baseline_tfidf.ipynb`. The RoBERTa notebook is meant for Colab with a GPU.

To run the app:

```bash
streamlit run app.py
```

The app expects the fine-tuned RoBERTa in `models/roberta/`, or a Hugging Face model id in the `ROBERTA_MODEL` environment variable.

## Stack

Python, pandas, scikit-learn, PyTorch, Hugging Face Transformers, Streamlit, Google Colab.

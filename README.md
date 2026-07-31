# Week 5 — NLP & Sentiment Analysis Dashboard

## Overview

This project cleans and analyzes real Amazon Alexa customer reviews, trains and compares three
text classification models to predict review sentiment (Positive / Negative), and deploys the
best-performing model in an interactive Streamlit dashboard. The full pipeline — loading,
cleaning, EDA, vectorization, model training, evaluation, and model selection — is documented and
executed end-to-end in the Part 1 notebook, with every result below pulled directly from its
executed outputs.

## Dataset

The working dataset (`data/amazon_reviews.csv`) is the Amazon Alexa Reviews dataset, containing
`rating`, `date`, `variation`, `verified_reviews` (review text), and `feedback` (1 = positive,
0 = negative). It follows the structure required by the task brief (review text + binary sentiment
label).

- **Raw shape:** 3,150 rows × 5 columns
- **After dropping nulls and duplicates:** 2,300 rows
- **After removing reviews that became empty post-cleaning:** 2,290 rows (final modeling set)
- **Class balance:** 2,095 Positive vs. 205 Negative — an imbalance ratio of roughly **10.2 : 1**

This imbalance is called out explicitly in the notebook and handled with stratified train/test
splitting and `class_weight="balanced"` models, since ignoring it would produce a model that looks
accurate on paper but is practically useless at catching negative reviews.

## Project Structure

```
.
├── notebook/
│   └── week5_nlp.ipynb        # Part 1 — full NLP pipeline, all cells executed with outputs
├── app.py                     # Part 2 — Streamlit dashboard entry point
├── data/
│   └── amazon_reviews.csv     # dataset
├── models/
│   ├── best_model.pkl         # best-performing trained classifier
│   ├── vectorizer.pkl         # fitted TF-IDF vectorizer
│   ├── model_name.txt         # name of the selected best model
│   ├── model_comparison.csv   # metrics table for all trained models
│   └── dataset_stats.json     # summary stats used by the dashboard Home page
├── assets/
│   ├── class_distribution.png
│   ├── wordclouds.png
│   ├── confusion_matrices.png
│   └── model_comparison.png
├── requirements.txt
└── README.md
```

## Part 1 — Notebook (`notebook/week5_nlp.ipynb`)

1. **Load and inspect** the dataset; check shape, nulls, and class distribution.
2. **Clean review text** — lowercasing, URL/HTML/punctuation removal, stopword removal (from
   scikit-learn's built-in list, not an external NLTK download, so the notebook runs fully
   offline/reproducibly), and suffix normalization.
3. **Word clouds** for positive vs. negative reviews, side by side.
4. **Vectorize** cleaned text with **TF-IDF** (unigrams + bigrams, max 5,000 features, `min_df=2`,
   `max_df=0.9`) — chosen over plain bag-of-words because it down-weights common, uninformative
   words and separates classes better for linear models. Reasoning documented in a markdown cell.
5. **Train three classifiers** (one more than the required minimum of two): Logistic Regression,
   Multinomial Naive Bayes, and a Linear SVM — all trained with `class_weight="balanced"` on a
   stratified 80/20 split.
6. **Evaluate** each with a full classification report and confusion matrix.
7. **Compare all models** in a bar chart and select the best one **by F1-score** (not raw
   accuracy, given the class imbalance) — reasoning documented in a markdown cell.
8. **Save** the best model, vectorizer, and supporting stats/metrics files used by the dashboard.

## Part 2 — Streamlit Dashboard (`app.py`)

Three pages, accessible via the sidebar:

- **Home** — project introduction, key dataset metrics, and a short explanation of what sentiment
  analysis is.
- **Data Overview** — class distribution table + chart, word clouds, model comparison table +
  chart, and confusion matrices.
- **Sentiment Predictor** — free-text input; on clicking "Predict Sentiment," the app applies the
  exact same cleaning pipeline used in Part 1, vectorizes the text with the saved TF-IDF
  vectorizer, and returns the predicted sentiment with a confidence score and a probability
  breakdown chart.

## How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) re-run the notebook end to end
jupyter nbconvert --to notebook --execute --inplace notebooks/week5_nlp.ipynb

# 3. Launch the dashboard
streamlit run app.py
```

## Screenshots

**Class distribution** — visualizing the ~10.2 : 1 positive/negative imbalance:

<img width="684" height="483" alt="image" src="https://github.com/user-attachments/assets/c6a25cab-1ace-4a88-b800-7a89b0ed0b95" />

**Word clouds** — positive reviews are dominated by *love, great, easy, work, use, sound*;
negative reviews cluster around *disappointed, return, poor, stopped working, waste*:

![Word clouds](https://github.com/user-attachments/assets/4b791f53-3a02-4821-97b9-981b51eb67e1)

**Confusion matrices** — all three trained models side by side on the held-out test set:

![Confusion matrices](https://github.com/user-attachments/assets/86131b06-442f-46fb-bb54-2942338cd204)

**Model comparison** — Accuracy, Precision, Recall, and F1-Score across all three models:

![Model comparison](https://github.com/user-attachments/assets/accfd318-e6b4-4f1e-8afa-5dc90ea02526)

## Quantitative Insights

**Train/test split:** 1,832 train / 458 test rows (stratified 80/20). Test set: 41 Negative, 417
Positive reviews.

| Model                      | Accuracy | Precision | Recall | F1-Score |
|-----------------------------|----------|-----------|--------|----------|
| Logistic Regression         | 0.9061   | 0.9583    | 0.9376 | 0.9479   |
| Multinomial Naive Bayes     | 0.9105   | 0.9105    | 1.0000 | 0.9531   |
| **Linear SVM (selected)**   | **0.9214** | 0.9300  | 0.9880 | **0.9581** |

The **Linear SVM** was selected as the production model based on F1-score and deployed in the
Sentiment Predictor page.

**Why raw accuracy is misleading here** — the per-class breakdown on the minority (Negative)
class tells a different story than the weighted-average table above:

| Model                    | Negative-class Precision | Negative-class Recall | Negative-class F1 |
|---------------------------|---------------------------|-------------------------|----------------------|
| Logistic Regression       | 0.48                      | 0.59                    | 0.53                 |
| Multinomial Naive Bayes   | 0.00                      | 0.00                    | 0.00                 |
| Linear SVM                | 0.67                      | 0.24                    | 0.36                 |

Naive Bayes reaches 91% overall accuracy while catching **zero** negative reviews — it simply
predicts "Positive" for almost everything, since positives dominate the training data. This is
the central quantitative argument for selecting by F1-score rather than accuracy, and for
reporting the minority-class breakdown at all: a dashboard viewer who saw only the top-line
accuracy number would have no way to know the Naive Bayes model is functionally useless for the
one thing the business actually cares about — flagging unhappy customers.

## My Work / Gaps Addressed Beyond the Base Requirements

- Explicit handling of class imbalance (stratified split + `class_weight="balanced"`), which the
  original brief did not call out.
- A third model (Linear SVM) trained beyond the required minimum of two, for a stronger
  comparison.
- Model selection justified by F1-score, with an explicit business-interpretation note (backed by
  the per-class negative-recall numbers above) on why accuracy alone is misleading on imbalanced
  data.
- Dashboard includes a full model comparison and confusion matrices on the Data Overview page
  (not just word clouds/class distribution) so the predictor's reliability is transparent to any
  viewer, plus a probability breakdown chart alongside every prediction rather than a bare label.
- Reasoning for each major methodological choice (TF-IDF over bag-of-words, F1 over accuracy,
  scikit-learn stopwords over an external download) documented in-line in markdown cells rather
  than left implicit.

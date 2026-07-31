# Week 5 — NLP & Sentiment Analysis Dashboard

Internship Task: Natural Language Processing & Sentiment Analysis Dashboard
Submission deadline: Friday, 31 July 2026

## Overview

This project cleans and analyzes real Amazon customer reviews, trains and compares multiple text
classification models to predict review sentiment (Positive / Negative), and deploys the best
model in an interactive Streamlit dashboard.

## Dataset

The working dataset (`data/amazon_reviews.csv`) is the Amazon Alexa Reviews dataset, containing
`rating`, `date`, `variation`, `verified_reviews` (review text), and `feedback` (1 = positive,
0 = negative). It follows the same structure required by the task brief (review text + binary
sentiment label) and is used as the dataset for this submission.

**Note:** the dataset is imbalanced (~92% positive / ~8% negative). This is called out explicitly
in the notebook and handled with stratified train/test splitting and class-weighted models, since
ignoring it would produce a model that looks accurate but is practically useless at catching
negative reviews.

## Project Structure

```
.
├── notebooks/
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
│   ├── outputs_class_distribution.png
│   ├── outputs_wordclouds.png
│   ├── outputs_confusion_matrices.png
│   └── outputs_model_comparison.png
├── requirements.txt
└── README.md
```

## Part 1 — Notebook (`notebooks/week5_nlp.ipynb`)

1. Load and inspect the dataset; check class distribution.
2. Clean review text (lowercasing, URL/HTML/punctuation removal, stopword removal, suffix
   normalization). Stopwords are sourced from scikit-learn's built-in list rather than an external
   NLTK download, so the notebook runs fully offline/reproducibly.
3. Generate word clouds for positive and negative reviews side by side.
4. Vectorize cleaned text with **TF-IDF** (unigrams + bigrams, 5000 features) — chosen over simple
   bag-of-words because it down-weights common, uninformative words and better separates classes
   for linear models. Reasoning documented in a markdown cell.
5. Train **three** classifiers (one more than the required minimum of two): Logistic Regression,
   Multinomial Naive Bayes, and a calibrated Linear SVM.
6. Evaluate each with a full classification report and a confusion matrix.
7. Compare all models in a bar chart and select the best one **by F1-score** (not raw accuracy,
   given the class imbalance) — reasoning documented in a markdown cell.
8. Save the best model, vectorizer, and supporting stats/metrics files used by the dashboard.

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

## Results Summary

| Model                    | Accuracy | Precision | Recall | F1-Score |
|---------------------------|----------|-----------|--------|----------|
| Logistic Regression       | 0.906    | 0.958     | 0.938  | 0.948    |
| Multinomial Naive Bayes   | 0.911    | 0.911     | 1.000  | 0.953    |
| **Linear SVM (selected)** | **0.921**| 0.930     | 0.988  | **0.958**|

The Linear SVM was selected as the production model based on F1-score and deployed in the
Sentiment Predictor page.

## Notes / Gaps Addressed Beyond the Base Requirements

- Explicit handling of class imbalance (stratified split + `class_weight="balanced"`), which the
  original brief did not call out.
- A third model (Linear SVM) trained beyond the required minimum of two, for a stronger
  comparison.
- Model selection justified by F1-score with an explicit business-interpretation note on why
  accuracy alone is misleading on imbalanced data.
- Dashboard includes a full model comparison and confusion matrices on the Data Overview page
  (not just word clouds/class distribution) so the predictor's reliability is transparent to any
  viewer, and a probability breakdown chart alongside every prediction rather than a bare label.

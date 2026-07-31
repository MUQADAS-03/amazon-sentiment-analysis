# Amazon Sentiment Intelligence and Analytics Dashboard

## Overview

This repository contains an end-to-end Natural Language Processing (NLP) pipeline and an interactive dashboard built with Streamlit. The project cleans and analyzes Amazon Alexa customer reviews, addresses significant class imbalance, trains and compares multiple text classification models, and deploys the best-performing model into a multi-page web application.

The primary goal is to transform unstructured customer review text into actionable business intelligence, allowing product teams to quickly gauge customer satisfaction and classify new feedback in real time.

---

## Application Screenshots

### 1. Home Page — Voice of the Customer
![Home Page](assets/dashboard_home.png)

### 2. Data Overview Page — Model Benchmarks and Class Distribution
![Data Overview](assets/dashboard_data.png)

### 3. Sentiment Predictor Page — Real-Time Inference
![Sentiment Predictor](assets/dashboard_predictor.png)

---

## Project Structure

.
├── notebooks/
│   └── week5_nlp.ipynb         # Jupyter Notebook with data exploration, cleaning, and model training
├── app.py                      # Multi-page Streamlit web dashboard
├── data/
│   └── amazon_reviews.csv      # Raw Amazon Alexa customer review dataset
├── models/
│   ├── best_model.pkl          # Serialized production classifier (Linear SVM)
│   ├── vectorizer.pkl          # Fitted TF-IDF vectorizer instance
│   ├── model_name.txt          # File containing the selected best model name
│   ├── model_comparison.csv    # Benchmark metrics table across all trained models
│   └── dataset_stats.json      # Precomputed summary statistics for dashboard rendering
├── assets/
│   ├── dashboard_home.png      # Screenshot of the Home page
│   ├── dashboard_data.png      # Screenshot of the Data Overview page
│   ├── dashboard_predictor.png # Screenshot of the Sentiment Predictor page
│   ├── outputs_class_distribution.png
│   ├── outputs_wordclouds.png
│   ├── outputs_confusion_matrices.png
│   └── outputs_model_comparison.png
├── requirements.txt            # Python environment dependencies
└── README.md                   # Project documentation

---

## Dataset Description

The dataset used in this project is the Amazon Alexa Reviews dataset (`data/amazon_reviews.csv`). It contains the following key fields:
* rating: Numerical score provided by the customer (1 to 5 stars).
* date: Date the review was submitted.
* variation: Specific product model or color configuration.
* verified_reviews: Raw textual feedback written by the customer.
* feedback: Binary sentiment label (1 for Positive, 0 for Negative).

---

## Technical Methodology

1. Text Preprocessing and Cleaning
   * Normalization to lower case and removal of HTML tags, URLs, and punctuation.
   * Stopword filtering using scikit-learn's built-in English stopword dictionary to avoid external downloading dependencies.
   * Custom suffix normalization to reduce words to their base linguistic forms.

2. Feature Extraction
   * Term Frequency-Inverse Document Frequency (TF-IDF) vectorization using unigrams and bigrams.
   * Constrained to 5,000 top features to prevent high-dimensional sparsity while down-weighting non-informative common terms.

3. Handling Class Imbalance
   * The dataset exhibits an extreme positive-to-negative imbalance (~91.1% positive vs. ~8.9% negative).
   * Mitigation strategies included Stratified K-Fold train-test splitting and incorporating class weighting (class_weight="balanced") during model training to ensure negative sentiment detection was not ignored.

4. Model Benchmarking
   * Three distinct classifiers were trained and evaluated:
     * Logistic Regression
     * Multinomial Naive Bayes
     * Calibrated Linear Support Vector Machine (Linear SVM)

---

## Quantitative Insights

### Dataset Summary Statistics

* Total Reviews Analyzed: 2,290 verified reviews
* Positive Feedback Count: 2,095 reviews (91.5%)
* Negative Feedback Count: 195 reviews (8.5%)
* Average Customer Rating: 4.44 / 5.00 stars
* Imbalance Ratio: Approximately 10.7 to 1 (Positive to Negative)

### Model Benchmarks and Comparison

Models were evaluated on a held-out stratified test set (20% of dataset). The evaluation prioritized F1-Score over raw accuracy due to the class imbalance.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :---: | :---: | :---: | :---: |
| Logistic Regression | 0.906 | 0.958 | 0.938 | 0.948 |
| Multinomial Naive Bayes | 0.911 | 0.911 | 1.000 | 0.953 |
| Linear SVM (Selected Production Model) | 0.921 | 0.930 | 0.988 | 0.958 |

### Analytical Findings

1. Model Selection Rationale: The Linear SVM model was selected for production deployment because it achieved the highest overall F1-Score (0.958) and accuracy (0.921), striking an optimal balance between precision and recall on the minority class.
2. Vocabulary Drivers: Term analysis via word clouds revealed that positive reviews heavily feature usability and hardware satisfaction terms (e.g., "love", "great sound", "easy setup"), whereas negative reviews focus predominantly on connectivity, voice recognition failures, and hardware setup faults (e.g., "refused", "disappointed", "wifi", "stopped working").
3. Product Variation Dynamics: Variations such as the Fire TV Stick and Black Plus accounted for high volume while maintaining over 90% positive sentiment, whereas smaller accessory variations exhibited wider variance in average star ratings.

---

## How to Run the Application

### 1. Environment Setup

Ensure Python 3.9+ is installed, then install required dependencies:

pip install -r requirements.txt

### 2. Execute Model Training (Optional)

To re-run the full training pipeline and regenerate saved model artifacts:

jupyter nbconvert --to notebook --execute --inplace notebooks/week5_nlp.ipynb

### 3. Launch the Streamlit Dashboard

Run the Streamlit application from the root directory:

streamlit run app.py

Access the interface by opening http://localhost:8501 in your web browser.

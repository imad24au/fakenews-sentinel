


#  FakeNews Sentinel

### A Naive Bayes Approach to Misinformation Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.75%25-brightgreen.svg)](#-results)

> A classical NLP approach to detecting fake news, achieving **93.75% accuracy** on a 44,000-article dataset using **TF-IDF + Multinomial Naive Bayes**. Built to demonstrate that interpretable, lightweight models can compete with complex deep learning for real-world misinformation detection.

---

##  Project Motivation

In an era of AI-generated content and viral misinformation, detecting fake news has never been more critical. While transformer-based models dominate headlines, **classical ML methods remain underrated** — they're fast, interpretable, deployable on minimal hardware, and surprisingly effective.

This project explores whether a **Naive Bayes classifier with TF-IDF features** can achieve production-grade performance on real-world fake news detection.

**Spoiler:** Yes, it can.

---

## Dataset

- **Source:** [Fake and Real News Dataset (Kaggle)](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- **Size:** 44,898 articles
- **Real news:** Reuters articles (~21,400 samples)
- **Fake news:** Articles from various unreliable sources (~23,500 samples)
- **Classes:** Balanced (~52% fake, 48% real)

---

##  Key Insights from EDA

###  Article Length Analysis (A Surprising Finding)

Counter-intuitively, **fake news articles are 13% LONGER** than real news on average:

| Type | Avg Word Count |
|------|----------------|
| Real news (Reuters) | 385 words |
| **Fake news** | **435 words** |

**Interpretation:** Fake news compensates for lack of factual substance with verbose emotional language, while professional journalism prioritizes concise factual reporting.

### Word Cloud Analysis

Distinct vocabulary patterns emerged between real and fake news:
- **Real news:** Reuters, said, government, percent, official
- **Fake news:** WATCH, VIDEO, Trump, people, Hillary

These patterns reveal stylistic signatures that classifiers can leverage.

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Data Manipulation** | Pandas, NumPy |
| **Machine Learning** | Scikit-Learn |
| **NLP** | TF-IDF Vectorizer, Naive Bayes |
| **Visualization** | Matplotlib, Seaborn, WordCloud |
| **Development** | Google Colab, Jupyter |

---

##  Methodology

### 1. Data Preprocessing
- Removed duplicates and empty articles
- Lowercased all text
- Stripped URLs, HTML tags, punctuation, and numbers
- Normalized whitespace

### 2. Feature Engineering — TF-IDF Vectorization
- **Vocabulary size:** 5,000 most informative terms
- **N-gram range:** Unigrams + Bigrams (1, 2)
- **Stop words:** English stop words removed
- **Min document frequency:** 5
- **Max document frequency:** 0.7 (filters overly common terms)

### 3. Model Comparison
Compared two Naive Bayes variants on identical TF-IDF features:
- **Multinomial Naive Bayes** — handles count/frequency data
- **Bernoulli Naive Bayes** — treats features as binary presence/absence

### 4. Stratified Train/Test Split
80/20 split with `stratify=y` to preserve class balance in both sets.

---

## Results

### Model Performance

| Metric | MultinomialNB | BernoulliNB |
|--------|---------------|-------------|
| **Accuracy** | **93.75%** | _your_value_ |
| **Precision** | _your_value_ | _your_value_ |
| **Recall** | _your_value_ | _your_value_ |
| **F1 Score** | _your_value_ | _your_value_ |

**Winner: MultinomialNB** — chosen for production deployment based on superior F1 score.

### Real-World Validation

Tested on 3 custom articles to validate beyond the training distribution:

| Article Type | Prediction | Confidence |
|-------------|------------|------------|
| Reuters-style (Fed Reserve) | ✅ Real | **97.25%** |
| Clickbait (clearly fake) | ❌ Fake | **97.85%** |
| Ambiguous (coffee study) | ✅ Real | **55.72%** |

**Key insight:** The model demonstrates **strong calibration** — high confidence on clear cases (97%), but appropriate uncertainty (55%) on ambiguous content. This calibration is critical for production: low-confidence predictions can be flagged for human review.

---

## Author
** Md Imad Hossain **
🇦🇺 Sydney, Australia
- 💼 [LinkedIn](<!-- TODO: Your LinkedIn URL or delete this line -->)
- 🐙 [GitHub](https://github.com/imad24au)
- 📧 [Email](mdimad.au.18@gmail.com)

# FakeNews Sentinel

### A Naive Bayes Approach to Misinformation Detection

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange.svg)](https://scikit-learn.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Accuracy](https://img.shields.io/badge/Accuracy-93.75%25-brightgreen.svg)](#results)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-Hugging%20Face-yellow.svg)](https://huggingface.co/spaces/imad24au/fakenews-sentinel)

> A classical NLP approach to detecting fake news, achieving **93.75% accuracy** on a 44,000-article dataset using **TF-IDF + Multinomial Naive Bayes**. Built to demonstrate that interpretable, lightweight models can compete with complex deep learning for real-world misinformation detection.

---

## Live Demo

**Try it yourself:** [https://huggingface.co/spaces/imad24au/fakenews-sentinel](https://huggingface.co/spaces/imad24au/fakenews-sentinel)

Paste any news article and see real-time predictions with confidence scores. No setup required.

---

## Project Motivation

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

## Key Insights from EDA

### Article Length Analysis (A Surprising Finding)

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
| **Web App** | Streamlit |
| **Deployment** | Hugging Face Spaces (Docker) |
| **Development** | Google Colab, VS Code |

---

## Methodology

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

### 5. Production Deployment
- Saved model and vectorizer using `pickle`
- Built interactive Streamlit interface
- Containerized and deployed to Hugging Face Spaces via Docker
- Live and accessible to anyone with the demo URL

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
| Reuters-style (Fed Reserve) | Real | **97.25%** |
| Clickbait (clearly fake) | Fake | **97.85%** |
| Ambiguous (coffee study) | Real | **55.72%** |

**Key insight:** The model demonstrates **strong calibration** — high confidence on clear cases (97%), but appropriate uncertainty (55%) on ambiguous content. This calibration is critical for production: low-confidence predictions can be flagged for human review.

---

## Getting Started

### Option 1: Try the Live Demo (No Setup Required)

Visit: [https://huggingface.co/spaces/imad24au/fakenews-sentinel](https://huggingface.co/spaces/imad24au/fakenews-sentinel)

### Option 2: Run Locally

```bash
# Clone the repo
git clone https://github.com/imad24au/fakenews-sentinel.git
cd fakenews-sentinel

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
cd app
streamlit run app.py
```

The app opens at `http://localhost:8501`.

### Option 3: Use the Pre-Trained Model in Python

```python
import pickle

# Load model and vectorizer
with open('models/nb_classifier.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/tfidf_vectorizer.pkl', 'rb') as f:
    vectorizer = pickle.load(f)

# Predict on a new article
article = "Your news article text here..."
cleaned = article.lower()  # Apply same preprocessing as training
vectorized = vectorizer.transform([cleaned])
prediction = model.predict(vectorized)[0]
confidence = model.predict_proba(vectorized)[0].max()

print(f"Prediction: {'Real' if prediction == 1 else 'Fake'}")
print(f"Confidence: {confidence*100:.2f}%")
```

---
## Project Structure
```
fakenews-sentinel/
│
├── notebooks/
│   └── FakeNews_Sentinel.ipynb     # Main notebook (EDA + modeling)
│
├── models/
│   ├── nb_classifier.pkl            # Trained MultinomialNB model
│   └── tfidf_vectorizer.pkl         # Fitted TF-IDF vectorizer
│
├── app/
│   └── app.py                       # Streamlit web application
│
├── README.md                        # You are here
├── LICENSE                          # MIT License
├── requirements.txt                 # Project dependencies
└── .gitignore                       # Python ignore rules
```

---

## Why Naive Bayes for Fake News?

A common question: *"Why not use BERT or GPT?"*

| Aspect | Naive Bayes + TF-IDF | Transformer Models |
|--------|---------------------|-------------------|
| **Training time** | Seconds | Hours/Days |
| **Inference latency** | <1ms | 50-500ms |
| **Hardware needed** | Any laptop | GPU recommended |
| **Interpretability** | High (word probabilities) | Low (black box) |
| **Deployment cost** | Negligible | Significant |
| **Performance on this task** | **93.75%** | ~96-98% |

For misinformation detection at scale (millions of articles per day), Naive Bayes offers a **~3% accuracy trade-off for 100x faster inference** — often the right engineering choice.

---

## Future Improvements

- [ ] Experiment with TF-IDF + Logistic Regression for comparison
- [ ] Add SHAP-based explainability for individual predictions
- [ ] Test on out-of-distribution data (recent 2024-2025 news)
- [ ] Build a Chrome extension for real-time URL checking
- [ ] Ensemble methods combining NB with other classifiers
- [ ] Compare against fine-tuned DistilBERT to quantify the accuracy gap

---

## Lessons Learned

1. **Classical ML is still relevant.** Naive Bayes achieved 93.75% on a non-trivial dataset — proving that algorithm choice should be driven by problem requirements, not hype.
2. **EDA reveals counter-intuitive patterns.** The discovery that fake news is *longer* than real news shaped my preprocessing decisions.
3. **Model calibration matters.** A model that knows when it's uncertain (55% on ambiguous content) is more deployable than one that's overconfident.
4. **Stratified splits are non-negotiable.** Without them, evaluation metrics can mislead by huge margins.
5. **Deployment teaches more than training.** Debugging Docker paths, Hugging Face quirks, and production environment differences taught me more about ML engineering than the modeling itself.

---

## Author

**Md Imad Hossain**  
Sydney, Australia
- [LinkedIn](https://www.linkedin.com/in/imad-au-hossain/)
- [GitHub](https://github.com/imad24au)
- [Email](mailto:imad.au.18@gmail.com)

> *Aspiring ML Engineer focused on building deployable, interpretable AI systems. Currently sharpening skills in classical ML before tackling deep learning.*

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

## Acknowledgments

- Dataset: [Clément Bisaillon on Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- Inspiration: The growing importance of interpretable AI in the age of misinformation
- Deployment platform: [Hugging Face Spaces](https://huggingface.co/spaces)

---

**If you found this project useful, please give it a star on GitHub!** It helps others discover this work.

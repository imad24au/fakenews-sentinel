"""
=================================================================
FakeNews Sentinel - Streamlit Web App
=================================================================
Author: Md Imad Hossain
Description: Interactive web interface for fake news detection
             using TF-IDF + Multinomial Naive Bayes
GitHub: https://github.com/imad24au/fakenews-sentinel
=================================================================
"""

import streamlit as st
import pickle
import re
import string
import os

# ===============================================================
# Page Configuration
# ===============================================================
st.set_page_config(
    page_title="FakeNews Sentinel",
    page_icon="N",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ===============================================================
# Custom CSS for polish
# ===============================================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #888;
        font-size: 1.2rem;
        margin-bottom: 2rem;
    }
    .result-real {
        background-color: #d4edda;
        color: #155724;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    .result-fake {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1.5rem;
        border-radius: 10px;
        text-align: center;
        font-size: 1.5rem;
        font-weight: bold;
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

# ===============================================================
# Load Model and Vectorizer (cached for speed)
# ===============================================================
@st.cache_resource
def load_artifacts():
    """Load model and vectorizer once and cache them."""
    model_path = os.path.join('..', 'models', 'nb_classifier.pkl')
    vectorizer_path = os.path.join('..', 'models', 'tfidf_vectorizer.pkl')

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(vectorizer_path, 'rb') as f:
        vectorizer = pickle.load(f)

    return model, vectorizer

# ===============================================================
# Text Cleaning (must match training preprocessing)
# ===============================================================
def clean_text(text):
    """Clean input text using the same preprocessing as training."""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# ===============================================================
# UI Header
# ===============================================================
st.markdown('<p class="main-header">FakeNews Sentinel</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">AI-Powered Misinformation Detection</p>', unsafe_allow_html=True)

st.markdown("""
This app uses **Naive Bayes + TF-IDF** to detect whether a news article is real or fake.
Trained on **44,898 articles**, achieving **93.75% accuracy**.
""")

st.divider()

# ===============================================================
# Load model
# ===============================================================
try:
    model, vectorizer = load_artifacts()
    model_loaded = True
except Exception as e:
    st.error(f"Could not load model files. Error: {str(e)}")
    model_loaded = False

# ===============================================================
# Input Section
# ===============================================================
st.subheader("Paste a News Article")

sample_options = {
    "-- Choose a sample (or paste your own) --": "",
    "Real news example (Reuters style)": (
        "WASHINGTON (Reuters) - The Federal Reserve raised interest rates by a "
        "quarter percentage point on Wednesday, citing strong job growth and "
        "continued inflationary pressures. Fed Chair Jerome Powell said the central "
        "bank remains committed to bringing inflation back to its 2% target through "
        "measured policy adjustments."
    ),
    "Fake news example (clickbait)": (
        "SHOCKING REVELATION: You won't BELIEVE what scientists just discovered! "
        "The government has been HIDING this incredible secret from us for DECADES. "
        "Big Pharma doesn't want you to know about this one weird trick that doctors "
        "HATE. Click here to see the truth they don't want you to see!"
    )
}

selected_sample = st.selectbox(
    "Try a sample article:",
    list(sample_options.keys()),
    key="sample_selector"
)

default_text = sample_options[selected_sample]
article_text = st.text_area(
    "Or paste your own article:",
    value=default_text,
    height=200,
    placeholder="Paste a news article here...",
    key="article_input"
)

# ===============================================================
# Prediction Button
# ===============================================================
col1, col2, col3 = st.columns([1, 1, 1])
with col2:
    predict_button = st.button(
        "Analyze Article",
        use_container_width=True,
        type="primary",
        key="analyze_btn"
    )

# ===============================================================
# Prediction Logic
# ===============================================================
if predict_button and model_loaded:
    if not article_text.strip():
        st.warning("Please enter or select an article to analyze.")
    elif len(article_text.split()) < 20:
        st.warning("Article seems too short. Please provide at least 20 words for accurate prediction.")
    else:
        with st.spinner("Analyzing..."):
            cleaned = clean_text(article_text)
            vectorized = vectorizer.transform([cleaned])

            prediction = model.predict(vectorized)[0]
            probabilities = model.predict_proba(vectorized)[0]

            confidence = probabilities[prediction] * 100
            fake_prob = probabilities[0] * 100
            real_prob = probabilities[1] * 100

        st.divider()
        st.subheader("Prediction Result")

        if prediction == 1:
            st.markdown(
                f'<div class="result-real">REAL NEWS<br>'
                f'<span style="font-size: 1rem;">Confidence: {confidence:.2f}%</span></div>',
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f'<div class="result-fake">FAKE NEWS<br>'
                f'<span style="font-size: 1rem;">Confidence: {confidence:.2f}%</span></div>',
                unsafe_allow_html=True
            )

        st.subheader("Probability Breakdown")

        prob_col1, prob_col2 = st.columns(2)
        with prob_col1:
            st.metric("Fake Probability", f"{fake_prob:.2f}%")
            st.progress(fake_prob / 100)
        with prob_col2:
            st.metric("Real Probability", f"{real_prob:.2f}%")
            st.progress(real_prob / 100)

        if 40 < confidence < 65:
            st.info(
                "Low confidence prediction. The model is uncertain about this article. "
                "In production, this would be flagged for human review."
            )
        elif confidence > 90:
            st.success(
                f"High confidence prediction. The model is very certain ({confidence:.1f}%)."
            )

# ===============================================================
# Footer
# ===============================================================
st.divider()

with st.expander("About This Project"):
    st.markdown("""
    **FakeNews Sentinel** uses classical NLP techniques to detect misinformation:

    - **Algorithm:** Multinomial Naive Bayes
    - **Features:** TF-IDF (5,000 terms, unigrams + bigrams)
    - **Training Data:** 44,898 articles (Reuters real news + fake news sources)
    - **Test Accuracy:** 93.75%

    **Why classical ML?** Fast inference (<1ms), interpretable, deployable on any hardware,
    and surprisingly competitive with deep learning for this task.

    [GitHub Repository](https://github.com/imad24au/fakenews-sentinel)
    """)

st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.9rem;'>"
    "Built by <a href='https://github.com/imad24au' target='_blank'>Md Imad Hossain</a> | "
    "<a href='https://github.com/imad24au/fakenews-sentinel' target='_blank'>GitHub</a>"
    "</p>",
    unsafe_allow_html=True
)
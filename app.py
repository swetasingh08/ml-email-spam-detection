from pathlib import Path
import re
import string

import joblib
import numpy as np
import streamlit as st
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "models" / "spam_model.pkl"
VECTORIZER_PATH = BASE_DIR / "models" / "tfidf_vectorizer.pkl"


st.set_page_config(page_title="Spam Classifier", page_icon="email", layout="wide")


@st.cache_resource
def load_stop_words():
    try:
        return set(stopwords.words("english"))
    except LookupError:
        return set(ENGLISH_STOP_WORDS)


@st.cache_resource
def load_models():
    if not MODEL_PATH.exists() or not VECTORIZER_PATH.exists():
        missing = [
            str(path)
            for path in (MODEL_PATH, VECTORIZER_PATH)
            if not path.exists()
        ]
        raise FileNotFoundError(f"Missing model artifact(s): {', '.join(missing)}")

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


stop_words = load_stop_words()
stemmer = PorterStemmer()
model, vectorizer = load_models()


def clean_text(text):
    """Apply the same preprocessing used when the saved vectorizer was trained."""
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()

    tokens = text.split()
    tokens = [stemmer.stem(token) for token in tokens if token not in stop_words]
    return " ".join(tokens)


def predict_spam(message):
    cleaned = clean_text(message)
    transformed = vectorizer.transform([cleaned])
    prediction = model.predict(transformed)[0]

    if hasattr(model, "predict_proba"):
        ham_prob, spam_prob = model.predict_proba(transformed)[0]
    else:
        decision = model.decision_function(transformed)[0]
        spam_prob = 1 / (1 + np.exp(-decision))
        ham_prob = 1 - spam_prob

    label = "Spam" if prediction == 1 else "Ham"
    confidence = max(ham_prob, spam_prob)
    return label, confidence, ham_prob, spam_prob


def use_example():
    st.session_state.message = st.session_state.example_message


examples = [
    "",
    "Congratulations! You won a free iPhone. Click now!",
    "Hey, are we meeting tomorrow?",
    "URGENT: Your bank account is locked. Verify immediately!",
    "Limited time loan offer. Get cash instantly!",
]


st.title("Email/SMS Spam Classifier")

st.sidebar.title("Examples")
st.sidebar.selectbox("Choose example:", examples, key="example_message")
st.sidebar.button("Use Example", on_click=use_example, disabled=not st.session_state.example_message)

user_input = st.text_area("Enter message:", key="message", height=160)

if st.button("Classify", type="primary"):
    if user_input.strip():
        label, confidence, ham_prob, spam_prob = predict_spam(user_input)

        if label == "Spam":
            st.error("Spam detected")
        else:
            st.success("Not spam")

        st.metric("Confidence", f"{confidence:.2%}")

        col1, col2 = st.columns(2)
        col1.metric("Ham probability", f"{ham_prob:.2%}")
        col2.metric("Spam probability", f"{spam_prob:.2%}")
    else:
        st.warning("Please enter a message.")

st.markdown("---")
st.info("Model: Logistic Regression with TF-IDF")

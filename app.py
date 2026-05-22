import streamlit as st
import joblib
import re
import string
import numpy as np
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

# ---- NLTK setup ----
@st.cache_resource
def download_nltk():
    nltk.download('stopwords')
    nltk.download('punkt')

download_nltk()

# ---- Load model ----
@st.cache_resource
def load_models():
    model = joblib.load('models/spam_model.pkl')
    vectorizer = joblib.load('models/tfidf_vectorizer.pkl')
    return model, vectorizer

model, vectorizer = load_models()

# ---- Preprocessing (MATCHES NOTEBOOK) ----
stop_words = set(stopwords.words('english'))
ps = PorterStemmer()

def clean_text(text):
    text = text.lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()

    tokens = text.split()
    tokens = [ps.stem(t) for t in tokens if t not in stop_words]

    return ' '.join(tokens)

# ---- Prediction ----
def predict_spam(message):
    cleaned = clean_text(message)
    transformed = vectorizer.transform([cleaned])

    prediction = model.predict(transformed)[0]

    # SAFE probability handling
    if hasattr(model, "predict_proba"):
        prob = model.predict_proba(transformed)[0]
        spam_prob = prob[1]
        ham_prob = prob[0]
    else:
        decision = model.decision_function(transformed)[0]
        spam_prob = 1 / (1 + np.exp(-decision))
        ham_prob = 1 - spam_prob

    label = "Spam" if prediction == 1 else "Ham"
    confidence = max(spam_prob, ham_prob)

    return label, confidence


# ---- UI ----
st.set_page_config(page_title="Spam Classifier", page_icon="📧", layout="wide")

st.title("📧 Email/SMS Spam Classifier")

user_input = st.text_area("Enter message:")

# Example messages
st.sidebar.title("Examples")

example = st.sidebar.selectbox(
    "Choose example:",
    [
        "",
        "Congratulations! You won a free iPhone. Click now!",
        "Hey, are we meeting tomorrow?",
        "URGENT: Your bank account is locked. Verify immediately!",
        "Limited loan offer. Get cash instantly!"
    ]
)

if st.sidebar.button("Use Example"):
    user_input = example

# ---- Prediction ----
if st.button("Classify"):
    if user_input.strip():

        label, confidence = predict_spam(user_input)

        if label == "Spam":
            st.error(f"⚠️ Spam Detected")
        else:
            st.success(f"✅ Not Spam")

        st.metric("Confidence", f"{confidence:.2%}")

    else:
        st.warning("Please enter a message.")

# ---- Footer ----
st.markdown("---")
st.info("Model: Logistic Regression / Linear SVM with TF-IDF")
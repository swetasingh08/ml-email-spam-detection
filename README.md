# 📧 ml-email-spam-detection

> 🚀 Real-time email & SMS spam classification using Streamlit and machine learning

![GitHub stars](https://img.shields.io/github/stars/swetasingh08/ml-email-spam-detection?style=for-the-badge&logo=github) ![GitHub forks](https://img.shields.io/github/forks/swetasingh08/ml-email-spam-detection?style=for-the-badge&logo=github) ![GitHub issues](https://img.shields.io/github/issues/swetasingh08/ml-email-spam-detection?style=for-the-badge&logo=github) ![Last commit](https://img.shields.io/github/last-commit/swetasingh08/ml-email-spam-detection?style=for-the-badge&logo=github) ![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white) ![License](https://img.shields.io/badge/license-MIT-green?style=for-the-badge)

---

## 📑 Table of Contents

- [📝 Description](#-description)
- [🎬 Live Demo](#-live-demo)
- [✨ Key Features](#-key-features)
- [🎯 Use Cases](#-use-cases)
- [🛠️ Tech Stack](#️-tech-stack)
- [⚡ Quick Start](#-quick-start)
- [📦 Key Dependencies](#-key-dependencies)
- [📁 Project Structure](#-project-structure)
- [🛠️ Development Setup](#️-development-setup)
- [👥 Contributing](#-contributing)
- [📜 License](#-license)

---

## 📝 Description

📬 **ml-email-spam-detection** is a lightweight web application designed to classify text messages as either 🚫 **spam** or ✅ **ham** in real time. The project provides an accessible, interactive interface for evaluating text content, helping users quickly identify potential spam or malicious messages without complex local setups.

🧠 Under the hood, the system loads a pre-trained TF-IDF vectorizer and classification model utilizing joblib. The text processing pipeline relies on NLTK for text preprocessing, featuring fallback capabilities to scikit-learn's English stopwords when necessary. The user interface is driven by Streamlit, ensuring efficient performance by caching resource-heavy model load operations.

🔧 This application is structured to facilitate both interactive usage and model development. With designated directories for training notebooks and dataset assets, it serves as a clean baseline for developers looking to understand or extend natural language processing classification workflows.

---

## 🎬 Live Demo

<p>
  🌐 <strong>Try it now:</strong>
  <a href="https://ml-email-spam-detection-pduigrzyujnqqg2emix8cy.streamlit.app/" target="_blank">
    ml-email-spam-detection.streamlit.app
  </a>
</p>

🚀 Experience the spam classifier in action — no installation required! Simply visit the deployed app, paste any message, and get instant spam/ham predictions with confidence scores.

---


## ✨ Key Features

- 💬 **Interactive Streamlit Web UI** — Clean, wide-layout web interface for typing or pasting messages to get instant spam classification feedback
- ⚙️ **Robust Stopword Fallback Pipeline** — Integrates NLTK stopwords with automatic fallback to scikit-learn's English stopword lists for reliable text preprocessing
- 📦 **Serialized Model Loading** — Loads pre-trained TF-IDF vectorizer and model pipeline files from local directories using Joblib serialization
- ⚡ **Resource Caching Support** — Optimizes load times and memory usage by caching model and vectorizer assets via Streamlit's resource caching
- 📓 **Modular Project Organization** — Dedicated directories for raw data, serialized models, and Jupyter notebooks supporting iterative development
- 📊 **Confidence Scoring** — Displays both ham and spam probability scores alongside final classification for transparent decision-making
- 🧪 **Example Messages** — Built-in sidebar with pre-loaded examples for quick testing and demonstration

---

## 🎯 Use Cases

- 🧪 Testing and evaluating custom text blocks for spam content through an interactive web dashboard
- 📚 Learning or demonstrating how to deploy scikit-learn and NLTK models within a streamlined web application framework
- 🔬 Experimenting with NLP text classification and preprocessing techniques using the provided notebooks and data directories
- 🛡️ Building a foundation for email/SMS filtering systems or content moderation tools
- 🎓 Educational resource for understanding TF-IDF vectorization and logistic regression in spam detection

---

## 🛠️ Tech Stack

- 🐍 **Python** (v3.10+)
- 🎈 **Streamlit** — Web interface
- 🧮 **Scikit-learn** — ML models & TF-IDF vectorization
- 📝 **NLTK** — Natural language preprocessing
- 💾 **Joblib** — Model serialization
- 🐼 **Pandas** — Data manipulation
- 🔢 **NumPy** — Numerical operations

---

## ⚡ Quick Start

```bash
# 📥 1. Clone the repository
git clone https://github.com/swetasingh08/ml-email-spam-detection.git

# 🐍 2. Create & activate virtual environment
python -m venv venv && source venv/bin/activate
# Windows: venv\Scripts\activate

# 📦 3. Install dependencies
pip install -r requirements.txt

# 🚀 4. Launch the application
streamlit run app.py
```
---

## 📦 Key Dependencies

| Package       | Version |
|---------------|---------|
| streamlit     | latest  |
| pandas        | latest  |
| numpy         | latest  |
| scikit-learn  | latest  |
| nltk          | latest  |
| joblib        | latest  |

---

## 📁 Project Structure

```
📦 ml-email-spam-detection
├── 📜 LICENSE
├── 🚀 app.py                          # Streamlit web application
├── 📂 data/
│   └── 📊 spam.csv                    # Dataset (SMS Spam Collection)
├── 📂 models/
│   ├── 🧠 spam_model.pkl              # Trained classification model
│   └── 🔤 tfidf_vectorizer.pkl        # Fitted TF-IDF vectorizer
├── 📂 notebooks/
│   └── 📓 spam_classifier.ipynb       # Training & evaluation notebook
└── 📄 requirements.txt                # Python dependencies
```
---

## 🛠️ Development Setup

### 🐍 Python Setup
1. 📥 Install Python (v3.10+ recommended)
2. 🔧 Create virtual environment: `python -m venv venv && source venv/bin/activate` (Windows: `venv\Scripts\activate`)
3. 📦 Install dependencies: `pip install -r requirements.txt`
4. 📓 Explore the notebook: `jupyter notebook notebooks/spam_classifier.ipynb`
5. 🚀 Run the app: `streamlit run app.py`

##

### 🧪 Model Training
The Jupyter notebook (`spam_classifier.ipynb`) contains the complete pipeline:
- 📊 Exploratory Data Analysis
- 🧹 Text preprocessing (lowercasing, punctuation removal, stopword filtering, stemming)
- 🔤 TF-IDF vectorization (bigrams, 20K max features)
- 🤖 Model training (Multinomial Naive Bayes & Logistic Regression)
- 📈 Performance evaluation & visualization
- 💾 Model serialization

---

## 👥 Contributing

🙌 Contributions are welcome! Here's the standard flow:

1. 🍴 **Fork** the repository
2. 📥 **Clone** your fork: `git clone https://github.com/YOUR_USERNAME/ml-email-spam-detection.git`
3. 🌿 **Branch**: `git checkout -b feature/your-feature`
4. ✏️ **Commit**: `git commit -m 'feat: add some feature'`
5. ⬆️ **Push**: `git push origin feature/your-feature`
6. 🔃 **Open** a pull request

📏 Please follow the existing code style and include tests for new behavior where applicable.

---

## 📜 License

📄 This project is licensed under the **MIT** License — see the [LICENSE](LICENSE) file for details.

---

<p align="center">
  🛡️ Built with ❤️ for a spam-free inbox | ⭐ Star this repo if you found it useful!
</p>

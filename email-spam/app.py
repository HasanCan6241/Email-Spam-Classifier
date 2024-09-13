import streamlit as st
import pickle
import re
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nltk.download('punkt')
nltk.download('stopwords')

st.markdown(
    """
    <style>
    body {
        background-color: #f5f7fa;
    }
    .main {
        background-color: white;
        padding: 20px;
        border-radius: 8px;
        box-shadow: 0px 4px 6px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 3em;
        text-align: center;
        font-weight: bold;
        color: #2c3e50;
    }
    .footer {
        font-size: 0.85em;
        color: gray;
        text-align: center;
        margin-top: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Başlık kısmı

st.markdown('<div class="title">📧 Email Spam Classifier</div> <br>', unsafe_allow_html=True)
st.write("Analyze and classify emails as **Spam** or **Not Spam** using AI-powered models.")

with open('tf_idf_word_vectorizer.pkl', 'rb') as vectorizer_file:
    tf_idf_vectorizer = pickle.load(vectorizer_file)

with open('lg_best_grid_model.pkl', 'rb') as model_file:
    lg_best_grid = pickle.load(model_file)




def clean_text(text):
    text = text.lower()
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'[^\w\s]', '', text)
    words = nltk.tokenize.word_tokenize(text)
    stop_words = set(stopwords.words('english'))  # İngilizce stopwords
    stemmer = PorterStemmer()
    words = [word for word in words if word not in stop_words]
    words = [stemmer.stem(word) for word in words]
    return ' '.join(words)


st.markdown("### ✉️ Enter Email Text Below:")
email_text = st.text_area("", "", placeholder="Type your email content here...")

if st.button("🔍 Classify Email"):
    if email_text:
        cleaned_text = clean_text(email_text)
        X_tf_idf_word = tf_idf_vectorizer.transform([cleaned_text])
        predicted_label = lg_best_grid.predict(X_tf_idf_word)
        label = "🚫 Spam" if predicted_label[0] == 1 else "✅ Not Spam"

        # Sonucun stilize edilmesi
        st.write("---")
        st.subheader(f"Result: {label}")
    else:
        st.warning("⚠️ Please enter some text to classify.")

st.markdown('<div class="footer">Made with 💻 by AI Enthusiasts | Powered by Machine Learning</div>',
            unsafe_allow_html=True)

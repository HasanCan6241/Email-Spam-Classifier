import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from nltk.corpus import stopwords
import nltk.tokenize
from wordcloud import WordCloud, STOPWORDS
import re
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import cross_val_score
import pickle


nltk.download('stopwords')
nltk.download('punkt')

df=pd.read_csv('combined_data.csv')

print(df.head())

print(df.shape)

print(df.info())

df.isnull().sum()

sns.barplot(x=df['label'].value_counts().index, y=df['label'].value_counts().values)

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

def clean_text(text):
    text = text.lower()

    text = re.sub(r'\d+', '', text)

    text = re.sub(r'[^\w\s]', '', text)

    words= nltk.tokenize.word_tokenize(text)

    words = [word for word in words if word not in stop_words]

    words = [stemmer.stem(word) for word in words]

    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

tf = df["text"].apply(lambda x: pd.value_counts(x.split(" "))).sum(axis=0).reset_index()

tf.columns = ["words", "tf"]

print(tf.head())

tf[tf["tf"] > 500].plot.bar(x="words", y="tf")
plt.show()

x=df['text']
y=df['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)

tf_idf_word_vectorizer = TfidfVectorizer()
X_tf_idf_word = tf_idf_word_vectorizer.fit_transform(x)

lg_params={
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver': ['liblinear']
}
lg_model = LogisticRegression()
lg_best_grid = GridSearchCV(lg_model, lg_params, cv=5, n_jobs=-1, verbose=1).fit(X_tf_idf_word, y)
cross_val_score(lg_model, X_tf_idf_word, y, cv=5, n_jobs=-1, verbose=1)


with open('lg_best_grid_model.pkl', 'wb') as file:
    pickle.dump(lg_best_grid, file)

with open('tf_idf_word_vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(tf_idf_word_vectorizer, vectorizer_file)
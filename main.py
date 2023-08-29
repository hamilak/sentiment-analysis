import pandas as pd
import numpy as np
import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load csv data
df = pd.read_csv('nlpsentimentdata/data.csv')

print(df.head(5))
print(df.isnull().sum())
print(df.info())
print(df.describe())

print(df["Sentence"].head(5))


# Lower casing text
def lowercase_text(text):
    return text.lower()


# Removing symbols
def remove_sym(text):
    return re.sub(r'[^\w\s]|[$]', '', text)


# Removing numbers
def remove_nums(text):
    return re.sub(r'\d+', '', text)


# Remove whitespace
def remove_whitespace(text):
    return re.sub(r'\s+', '', text).strip()


# Remove stopwords
# nltk.download('stopwords')
stop_words = set(stopwords.words('english'))


def remove_stopwords(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    return ' '.join(filtered_words)


# Lemmatization
# nltk.download('wordnet')
lemmatizer = WordNetLemmatizer()


def lemmatize_text(text):
    words = text.split()
    lemmatized_words = [lemmatizer.lemmatize(word) for word in words]
    return ' '.join(lemmatized_words)


# Clean the text data
def clean_text(text):
    text = lowercase_text(text)
    text = remove_sym(text)
    text = remove_stopwords(text)
    text = remove_whitespace(text)
    text = remove_nums(text)
    text = lemmatize_text(text)
    return text


df["Sentence"] = df["Sentence"].apply(clean_text)
print(df["Sentence"])


# Calculate sentiment using textblob
def analyze_sentiment(text):
    blob = TextBlob(text)
    sentiment_polarity = blob.sentiment.polarity

    if sentiment_polarity > 0:
        return 'positive'
    elif sentiment_polarity < 0:
        return 'negative'
    else:
        return 'neutral'


df["Predicted Sentiment"] = df["Sentence"].apply(analyze_sentiment)
# print(df)
accuracy_tb = accuracy_score(df["Sentiment"], df["Predicted Sentiment"])
print(f"TB accuracy is{accuracy_tb}")

# Split the data train and test sets
X = df["Sentence"]
y = df["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Initialize and SVM classifier
classifier = SVC()
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy is {accuracy}")


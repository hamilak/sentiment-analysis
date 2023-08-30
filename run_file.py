import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from textblob import TextBlob
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix
from clean_data import clean_text


# Load csv data
df = pd.read_csv('nlpsentimentdata/data.csv')

print(df.head(5))
print(df.isnull().sum())
print(df.info())
print(df.describe())

print(df["Sentence"].head(5))
print(df.groupby('Sentiment').count())


df["Sentence"] = df["Sentence"].apply(clean_text)
# print(df["Sentence"])


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
print("TB accuracy is", accuracy_tb)
print(df.groupby('Predicted Sentiment').count())


# Split the data train and test sets
X = df["Sentence"]
y = df["Sentiment"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)


# Initialize and fit the TF-IDF vectorizer
vectorizer = TfidfVectorizer(max_features=1000)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


# Initialize and train SVM classifier
classifier = SVC()
classifier.fit(X_train_tfidf, y_train)

y_pred = classifier.predict(X_test_tfidf)
# print(y_pred)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy is", accuracy)

cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
ax = sns.heatmap(cm, annot=True, cmap="Blues", fmt="g", cbar=False)
# plt.xlabel('Predicted')
# plt.ylabel('Actual')
# plt.title('Confusion matrix')
# plt.show()


import re
import string
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


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

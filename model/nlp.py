import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

lemmatizer = WordNetLemmatizer()


def clean_text(text):
    text = text.lower()

    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

    words = word_tokenize(text)

    words = [
        lemmatizer.lemmatize(word)
        for word in words
        if word.strip()
    ]

    return words
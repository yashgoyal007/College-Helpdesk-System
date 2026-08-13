import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# NLTK data is installed during the Render build
# and stored in /opt/render/project/src/nltk_data

NLTK_DATA_DIR = "/opt/render/project/src/nltk_data"

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


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
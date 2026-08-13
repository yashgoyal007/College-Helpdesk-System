import os
import re
import nltk

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# Use a fixed NLTK data folder inside the Render project
NLTK_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "nltk_data")

os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)


# Check required NLTK resources
try:
    nltk.data.find("tokenizers/punkt_tab")
except LookupError:
    nltk.download("punkt_tab", download_dir=NLTK_DATA_DIR)


try:
    nltk.data.find("corpora/wordnet")
except LookupError:
    nltk.download("wordnet", download_dir=NLTK_DATA_DIR)


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
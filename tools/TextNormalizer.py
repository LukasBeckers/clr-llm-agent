import nltk

nltk.download("stopwords")
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import string
import re


class TextNormalizer:
    def __init__(self, language="english"):
        """
        Initializes the TextNormalizer with a specified language for stopwords.

        Args:
            language (str): Language for stopwords. Default is 'english'.
        """
        # Download stopwords if not already downloaded
        try:
            self.stop_words = set(stopwords.words(language))
        except LookupError:
            nltk.download("stopwords")
            self.stop_words = set(stopwords.words(language))

        # Initialize the stemmer
        self.stemmer = PorterStemmer()

        # Create a translation table for removing punctuation
        self.punct_table = str.maketrans("", "", string.punctuation)

    def __call__(self, text):
        """
        Normalizes the input text by lowercasing, removing punctuation,
        removing stopwords, and applying stemming.

        Args:
            text (str): The text to normalize.

        Returns:
            str: The normalized text.
        """
        # Lowercase the text
        text = text.lower()

        # Remove punctuation
        text = text.translate(self.punct_table)

        # Remove numerical digits (optional)
        text = re.sub(r"\d+", "", text)

        # Tokenize the text into words
        tokens = text.split()

        # Remove stopwords and apply stemming
        processed_tokens = [
            self.stemmer.stem(word)
            for word in tokens
            if word not in self.stop_words
        ]

        return processed_tokens

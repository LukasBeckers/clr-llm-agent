import textstat
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize

# Ensure NLTK resources are downloaded
import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')

class CohMetrix:
    def __init__(self):
        pass

    def analyze(self, document):
        """
        Analyzes a single document for basic Coh-Metrix-like metrics.

        :param document: Preprocessed document (list of tokens)
        :return: Dictionary with metrics
        """
        text = ' '.join(document)
        metrics = {
            'readability_score': textstat.flesch_reading_ease(text),
            'syllable_count': textstat.syllable_count(text),
            'lexical_diversity': len(set(document)) / len(document) if len(document) > 0 else 0,
            'average_sentence_length': textstat.sentence_count(text) / len(document) if len(document) > 0 else 0
        }
        # POS tagging
        pos_tags = pos_tag(document)
        pos_counts = {}
        for word, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        metrics['pos_counts'] = pos_counts
        return metrics
    
    def analyze_corpus(self, documents):
        """
        Analyzes a corpus of documents.

        :param documents: List of preprocessed documents (list of tokens)
        :return: List of analysis dictionaries
        """
        return [self.analyze(doc) for doc in documents]

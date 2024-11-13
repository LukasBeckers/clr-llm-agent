import textstat
from nltk import pos_tag
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from typing import List, Dict, Any, Optional
import copy

# Ensure NLTK resources are downloaded
import nltk

nltk.download("punkt")
nltk.download("averaged_perceptron_tagger")
nltk.download('averaged_perceptron_tagger_eng')
nltk.download("wordnet")


class CohMetrix:
    def __init__(self) -> None:
        pass

    def analyze(
        self, document: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyzes a single document for basic Coh-Metrix-like metrics.

        :param document: A dictionary representing a document with various fields.
        :return: Dictionary with metrics.
        """
        # Create a copy to ensure the original document is not modified
        doc_copy = copy.deepcopy(document)
        
        # Extract the "Abstract Normalized" field
        abstract_normalized: Optional[List[str]] = doc_copy.get("Abstract Normalized")
        
        if not isinstance(abstract_normalized, list):
            raise ValueError(
                "The document does not contain an 'Abstract Normalized' field "
                "or it is not a list of tokens."
            )
        
        text = " ".join(abstract_normalized)
        metrics: Dict[str, Any] = {
            "readability_score": textstat.flesch_reading_ease(text),
            "syllable_count": textstat.syllable_count(text),
            "lexical_diversity": (
                len(set(abstract_normalized)) / len(abstract_normalized)
                if len(abstract_normalized) > 0
                else 0
            ),
            "average_sentence_length": (
                textstat.sentence_count(text) / len(abstract_normalized)
                if len(abstract_normalized) > 0
                else 0
            ),
        }
        
        # POS tagging
        pos_tags = pos_tag(abstract_normalized)
        pos_counts: Dict[str, int] = {}
        for _, tag in pos_tags:
            pos_counts[tag] = pos_counts.get(tag, 0) + 1
        metrics["pos_counts"] = pos_counts
        
        return metrics

    def analyze_corpus(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Analyzes a corpus of documents.

        :param documents: List of document dictionaries.
        :return: List of analysis dictionaries.
        """
        return [self.analyze(doc) for doc in documents]

    def __call__(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Makes the CohMetrix instance callable. Analyzes the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing analysis results.
        """
        analysis_results = self.analyze_corpus(documents)
        return {"analysis": analysis_results}

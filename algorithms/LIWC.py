from typing import List, Dict, Any, Optional
import copy


class LIWC:
    def __init__(self, dictionary: Optional[Dict[str, List[str]]] = None) -> None:
        """
        Initializes the LIWC analyzer.

        :param dictionary: Dictionary mapping categories to lists of words.
                           If None, defaults to the sample dictionary.
        """
        if dictionary is None:
            # Default to the sample dictionary
            self.dictionary: Dict[str, List[str]] = {
                'positive_emotion': ['happy', 'joy', 'love', 'excellent', 'good'],
                'negative_emotion': ['sad', 'anger', 'hate', 'bad', 'terrible'],
                'cognitive_process': ['think', 'know', 'understand', 'reason', 'believe'],
                'social': ['friend', 'family', 'people', 'social', 'community']
            }
        else:
            self.dictionary: Dict[str, List[str]] = dictionary

    def analyze(self, document: List[str]) -> Dict[str, int]:
        """
        Analyzes a single document for LIWC categories.

        :param document: Preprocessed document (list of tokens)
        :return: Dictionary with category counts
        """
        analysis: Dict[str, int] = {category: 0 for category in self.dictionary}
        for word in document:
            for category, words in self.dictionary.items():
                if word in words:
                    analysis[category] += 1
        return analysis

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, int]]]:
        """
        Makes the LIWC instance callable. Analyzes the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing analysis results
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy: List[Dict[str, Any]] = copy.deepcopy(documents)

        # Extract "Abstract Normalized" from each document
        abstracts: List[List[str]] = []
        for doc in documents_copy:
            abstract_normalized = doc.get("Abstract Normalized")
            if not isinstance(abstract_normalized, list) or not all(isinstance(token, str) for token in abstract_normalized):
                raise ValueError(
                    "Each document must contain an 'Abstract Normalized' field as a list of strings."
                )
            abstracts.append(abstract_normalized)

        # Analyze each abstract
        analysis_results: List[Dict[str, int]] = [self.analyze(doc) for doc in abstracts]

        return {'analysis': analysis_results}

from typing import List, Dict, Any


class LIWC:
    def __init__(self, dictionary: Dict[str, List[str]] = None):
        """
            Initializes the LIWC analyzer.

            :param dictionary: Dictionary mapping categories to lists of words.
                               If None, defaults to the sample dictionary.
        """
        if dictionary is None:
            # Default to the sample dictionary
            self.dictionary = {
                'positive_emotion': ['happy', 'joy', 'love', 'excellent', 'good'],
                'negative_emotion': ['sad', 'anger', 'hate', 'bad', 'terrible'],
                'cognitive_process': ['think', 'know', 'understand', 'reason', 'believe'],
                'social': ['friend', 'family', 'people', 'social', 'community']
            }
        else:
            self.dictionary = dictionary

    def analyze(self, document: List[str]) -> Dict[str, int]:
        """
            Analyzes a single document for LIWC categories.

            :param document: Preprocessed document (list of tokens)
            :return: Dictionary with category counts
        """
        analysis = {category: 0 for category in self.dictionary}
        for word in document:
            for category, words in self.dictionary.items():
                if word in words:
                    analysis[category] += 1
        return analysis

    def __call__(self, documents: List[List[str]]) -> Dict[str, List[Dict[str, int]]]:
        """
            Makes the LIWC instance callable. Analyzes the provided documents.

            :param documents: List of preprocessed documents (list of tokens)
            :return: Dictionary containing analysis results
        """
        analysis_results = [self.analyze(doc) for doc in documents]
        return {'analysis': analysis_results}

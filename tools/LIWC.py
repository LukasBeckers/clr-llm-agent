"""# Sample LIWC-like dictionary
LIWC_DICTIONARY = {
    'positive_emotion': ['happy', 'joy', 'love', 'excellent', 'good'],
    'negative_emotion': ['sad', 'anger', 'hate', 'bad', 'terrible'],
    'cognitive_process': ['think', 'know', 'understand', 'reason', 'believe'],
    'social': ['friend', 'family', 'people', 'social', 'community']
}"""


class LIWC:
    def __init__(self, dictionary=None):
        """
        Initializes the LIWC analyzer.

        :param dictionary: Dictionary mapping categories to lists of words
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

    def analyze(self, document):
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
    
    def analyze_corpus(self, documents):
        """
        Analyzes a corpus of documents.

        :param documents: List of preprocessed documents (list of tokens)
        :return: List of analysis dictionaries
        """
        return [self.analyze(doc) for doc in documents]

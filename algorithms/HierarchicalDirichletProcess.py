import gensim
from gensim import corpora
from gensim.models import HdpModel
from typing import List, Dict, Any, Tuple

class HierarchicalDirichletProcess:
    def __init__(self, 
                 random_state: int = 42, 
                 top_n: int = 10):
        """
        Initializes the HDP model.

        :param random_state: Seed for reproducibility.
        :param top_n: Number of top words per topic to retrieve.
        """
        self.random_state = random_state
        self.top_n = top_n
        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, documents: List[List[str]]):
        """
        Fits the HDP model to the documents.

        :param documents: List of preprocessed documents (list of tokens).
        """
        # Create a Gensim dictionary and corpus
        self.dictionary = corpora.Dictionary(documents)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]

        # Initialize and train the HDP model
        self.model = HdpModel(corpus=self.corpus,
                              id2word=self.dictionary,
                              random_state=self.random_state)

    def get_topics(self) -> List[str]:
        """
        Retrieves the topics.

        :return: List of topics with top words.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        
        topics = []
        for topic_id, topic in self.model.print_topics(num_topics=-1, num_words=self.top_n):
            topics.append(topic)
        return topics

    def get_document_topics(self, document: List[str]) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens).
        :return: List of (topic_id, probability) tuples.
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        
        bow = self.dictionary.doc2bow(document)
        return self.model[bow]

    def __call__(self, documents: List[List[str]]) -> Dict[str, Any]:
        """
        Makes the HierarchicalDirichletProcess instance callable. Performs HDP on the provided documents.

        :param documents: List of preprocessed documents (list of tokens).
        :return: Dictionary containing model results.
        """
        self.fit(documents)
        topics = self.get_topics()
        result = {
            'random_state': self.random_state,
            'top_n': self.top_n,
            'num_topics': len(topics),
            'topics': topics
        }
        return result

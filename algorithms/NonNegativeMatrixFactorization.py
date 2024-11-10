import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from typing import List, Dict, Any, Tuple


class NonNegativeMatrixFactorization:
    def __init__(
        self,
        num_topics: int = 10,
        random_state: int = 42,
        max_iter: int = 200,
        top_n: int = 10
    ):
        """
        Initializes the NMF model with specified hyperparameters.

        :param num_topics: Number of topics to extract.
        :param random_state: Seed for reproducibility.
        :param max_iter: Maximum number of iterations during training.
        :param top_n: Number of top words per topic.
        """
        self.num_topics = num_topics
        self.random_state = random_state
        self.max_iter = max_iter
        self.top_n = top_n
        self.vectorizer = TfidfVectorizer()
        self.model = NMF(
            n_components=self.num_topics,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.feature_names = None

    def fit(self, documents: List[List[str]]):
        """
        Fits the NMF model to the documents.

        :param documents: List of preprocessed documents (list of tokens as strings).
        """
        # Join tokens back to strings for TfidfVectorizer
        joined_docs = [' '.join(doc) for doc in documents]
        tfidf = self.vectorizer.fit_transform(joined_docs)
        self.model.fit(tfidf)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_topics(self) -> List[Tuple[int, List[str]]]:
        """
        Retrieves the topics.

        :return: List of topics with top words.
        """
        topics = []
        for idx, topic in enumerate(self.model.components_):
            top_features = [self.feature_names[i] for i in topic.argsort()[:-self.top_n - 1:-1]]
            topics.append((idx, top_features))
        return topics

    def get_document_topics(self, document: List[str]) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens).
        :return: List of (topic_id, weight) tuples.
        """
        joined_doc = ' '.join(document)
        tfidf = self.vectorizer.transform([joined_doc])
        topic_distribution = self.model.transform(tfidf)[0]
        return list(enumerate(topic_distribution))

    def __call__(self, documents: List[List[str]]) -> Dict[str, Any]:
        """
        Makes the NonNegativeMatrixFactorization instance callable. Performs NMF topic modeling on the provided documents.

        :param documents: List of preprocessed documents (list of tokens as strings).
        :return: Dictionary containing the trained model, feature names, and topics.
        """
        self.fit(documents)
        topics = self.get_topics()
        result = {
            'num_topics': self.num_topics,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'top_n': self.top_n,
            'topics': topics,
            'model': self.model,
            'feature_names': self.feature_names.tolist() if self.feature_names is not None else None
        }
        return result

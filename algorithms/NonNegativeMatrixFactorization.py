import gensim
from gensim import corpora
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import NMF
from typing import List, Dict, Any, Tuple, Optional
import copy


class NonNegativeMatrixFactorization:
    def __init__(
        self,
        num_topics: int = 10,
        random_state: int = 42,
        max_iter: int = 200,
        top_n: int = 10
    ) -> None:
        """
        Initializes the NMF model with specified hyperparameters.

        :param num_topics: Number of topics to extract.
        :param random_state: Seed for reproducibility.
        :param max_iter: Maximum number of iterations during training.
        :param top_n: Number of top words per topic.
        """
        self.num_topics: int = num_topics
        self.random_state: int = random_state
        self.max_iter: int = max_iter
        self.top_n: int = top_n
        self.vectorizer: TfidfVectorizer = TfidfVectorizer()
        self.model: Optional[NMF] = None
        self.feature_names: Optional[List[str]] = None

    def fit(self, documents: List[List[str]]) -> None:
        """
        Fits the NMF model to the documents.

        :param documents: List of preprocessed documents (list of tokens as strings).
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy: List[List[str]] = copy.deepcopy(documents)

        # Join tokens back to strings for TfidfVectorizer
        joined_docs: List[str] = [' '.join(doc) for doc in documents_copy]
        tfidf = self.vectorizer.fit_transform(joined_docs)
        self.model = NMF(
            n_components=self.num_topics,
            random_state=self.random_state,
            max_iter=self.max_iter
        )
        self.model.fit(tfidf)
        self.feature_names = self.vectorizer.get_feature_names_out()

    def get_topics(self) -> List[Tuple[int, List[str]]]:
        """
        Retrieves the topics.

        :return: List of topics with top words.
        """
        if self.model is None or self.feature_names is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        
        topics: List[Tuple[int, List[str]]] = []
        for idx, topic in enumerate(self.model.components_):
            top_features = [self.feature_names[i] for i in topic.argsort()[:-self.top_n - 1:-1]]
            topics.append((idx, top_features))
        return topics

    def get_document_topics(self, document: List[str]) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens).
        :param top_n: Number of top topics to return.
        :return: List of (topic_id, weight) tuples.
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        if not isinstance(document, list) or not all(isinstance(token, str) for token in document):
            raise ValueError("The document must be a list of strings.")
        
        joined_doc: str = ' '.join(document)
        tfidf = self.vectorizer.transform([joined_doc])
        topic_distribution: List[float] = self.model.transform(tfidf)[0]
        top_topics: List[Tuple[int, float]] = sorted(enumerate(topic_distribution), key=lambda x: x[1], reverse=True)[:self.top_n]
        return top_topics

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the NonNegativeMatrixFactorization instance callable. Performs NMF topic modeling on the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing the trained model, feature names, and topics.
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

        # Fit the model
        self.fit(abstracts)

        # Retrieve topics
        topics: List[Tuple[int, List[str]]] = self.get_topics()

        result: Dict[str, Any] = {
            'num_topics': self.num_topics,
            'random_state': self.random_state,
            'max_iter': self.max_iter,
            'top_n': self.top_n,
            'topics': topics,
            'model': self.model,
            'feature_names': self.feature_names.tolist() if self.feature_names is not None else None
        }
        return result

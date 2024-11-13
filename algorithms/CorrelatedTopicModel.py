import tomotopy as tp
from typing import List, Dict, Any, Tuple, Optional
import copy


class CorrelatedTopicModel:
    def __init__(
        self,
        num_topics: int = 10,
        alpha: str = "symmetric",
        eta: str = "auto",
        seed: int = 42,
        iterations: int = 1000,
        top_n: int = 10
    ) -> None:
        """
        Initializes the CTM model using tomotopy.

        :param num_topics: Number of topics to extract
        :param alpha: Hyperparameter for topic distribution
        :param eta: Hyperparameter for word distribution
        :param seed: Seed for reproducibility
        :param iterations: Number of training iterations
        :param top_n: Number of top words per topic
        """
        self.num_topics: int = num_topics
        self.alpha: str = alpha
        self.eta: str = eta
        self.seed: int = seed
        self.iterations: int = iterations
        self.top_n: int = top_n
        self.model: tp.CTModel = tp.CTModel(
            k=self.num_topics, alpha=self.alpha, eta=self.eta, seed=self.seed
        )
        self.dictionary: Optional[tp.Dictionary] = None

    def fit(self, documents: List[List[str]]) -> None:
        """
        Fits the CTM model to the documents.

        :param documents: List of preprocessed documents (list of tokens)
        """
        self.model.reset()
        for doc in documents:
            self.model.add_doc(doc)
        self.model.train(self.iterations)

    def get_topics(self) -> List[Tuple[int, List[str]]]:
        """
        Retrieves the topics.

        :return: List of topics with top words
        """
        topics: List[Tuple[int, List[str]]] = []
        for i in range(self.num_topics):
            words = self.model.get_topic_words(i, self.top_n)
            topics.append((i, [word for word, _ in words]))
        return topics

    def get_document_topics(self, document: List[str]) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens)
        :return: List of (topic_id, probability) tuples
        """
        doc = self.model.make_doc(document)
        self.model.add_doc(doc)
        self.model.train(0)  # Infer without training
        topic_dist = doc.get_topic_dist()
        top_topics = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:self.top_n]
        return top_topics

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the CorrelatedTopicModel instance callable. Performs topic modeling on the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing model results.
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy: List[Dict[str, Any]] = copy.deepcopy(documents)

        # Extract "Abstract Normalized" from each document
        abstracts: List[List[str]] = []
        for doc in documents_copy:
            abstract_normalized = doc.get("Abstract Normalized")
            if not isinstance(abstract_normalized, list) or not all(
                isinstance(token, str) for token in abstract_normalized
            ):
                raise ValueError(
                    "Each document must contain an 'Abstract Normalized' field as a list of strings."
                )
            abstracts.append(abstract_normalized)

        # Fit the model
        self.fit(abstracts)

        # Retrieve topics
        topics = self.get_topics()

        result: Dict[str, Any] = {
            'num_topics': self.num_topics,
            'topics': topics,
            'alpha': self.alpha,
            'eta': self.eta,
            'seed': self.seed
        }
        return result

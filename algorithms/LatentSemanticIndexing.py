import gensim
from gensim import corpora
from gensim.models import LsiModel
from typing import List, Dict, Any, Tuple, Optional
import copy


class LatentSemanticIndexing:
    def __init__(
        self,
        num_topics: int = 10,
        num_words: int = 10,
        random_state: Optional[int] = None,
        chunksize: int = 1000,
    ) -> None:
        """
        Initializes the LSI model with specified hyperparameters.

        :param num_topics: The number of topics to extract.
        :param num_words: Number of top words per topic.
        :param random_state: Seed for random number generator for reproducibility.
        :param chunksize: Number of documents to be used in each training chunk.
        """
        self.num_topics: int = num_topics
        self.num_words: int = num_words
        self.random_state: Optional[int] = random_state
        self.chunksize: int = chunksize

        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[List[List[Tuple[int, int]]]] = None
        self.model: Optional[LsiModel] = None

    def fit(self, documents: List[List[str]]) -> None:
        """
        Fits the LSI model to the documents.

        :param documents: List of preprocessed documents (list of tokens).
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy: List[List[str]] = copy.deepcopy(documents)

        # Create a Gensim dictionary and corpus
        self.dictionary = corpora.Dictionary(documents_copy)
        # Optionally, filter extremes to limit the number of features
        # self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents_copy]

        # Initialize and train the LSI model
        self.model = LsiModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            random_state=self.random_state,
            chunksize=self.chunksize,
        )

    def get_topics(self) -> List[str]:
        """
        Retrieves the topics.

        :return: List of topics with top words.
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        return self.model.print_topics(num_topics=self.num_topics, num_words=self.num_words)

    def get_document_topics(self, document: List[str], top_n: int = 10) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens).
        :param top_n: Number of top topics to return.
        :return: List of (topic_id, probability) tuples.
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        if not isinstance(document, list) or not all(isinstance(token, str) for token in document):
            raise ValueError("The document must be a list of strings.")
        bow = self.dictionary.doc2bow(document)
        topic_distribution = self.model.get_document_topics(bow, minimum_probability=0)
        top_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[:top_n]
        return top_topics

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the LatentSemanticIndexing instance callable. Performs LSI topic modeling on the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing the trained model, dictionary, corpus, and topics.
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
        topics: List[str] = self.get_topics()

        result: Dict[str, Any] = {
            'num_topics': self.num_topics,
            'num_words': self.num_words,
            'chunksize': self.chunksize,
            'random_state': self.random_state,
            'topics': topics,
            'model': self.model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
        }
        return result

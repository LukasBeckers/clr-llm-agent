import gensim
from gensim import corpora
from gensim.models import LsiModel
from typing import List, Dict, Any, Tuple


class LatentSemanticIndexing:
    def __init__(
        self,
        num_topics: int = 10,
        num_words: int = 10,
        random_state: int = None,
        chunksize: int = 1000,
    ):
        """
        Initializes the LSI model with specified hyperparameters.

        :param num_topics: The number of topics to extract.
        :param num_words: Number of top words per topic.
        :param random_state: Seed for random number generator for reproducibility.
        :param chunksize: Number of documents to be used in each training chunk.
        """
        self.num_topics = num_topics
        self.num_words = num_words
        self.random_state = random_state
        self.chunksize = chunksize

        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, documents: List[List[str]]):
        """
        Fits the LSI model to the documents.

        :param documents: List of preprocessed documents (list of tokens).
        """
        self.dictionary = corpora.Dictionary(documents)
        # Optionally, filter extremes to limit the number of features
        # self.dictionary.filter_extremes(no_below=5, no_above=0.5)
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
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
        bow = self.dictionary.doc2bow(document)
        return self.model.get_document_topics(bow, minimum_probability=0)

    def __call__(self, documents: List[List[str]]) -> Dict[str, Any]:
        """
        Makes the LatentSemanticIndexing instance callable. Performs LSI topic modeling on the provided documents.

        :param documents: List of preprocessed documents (list of tokens).
        :return: Dictionary containing the trained model, dictionary, corpus, and topics.
        """
        self.fit(documents)
        topics = self.get_topics()
        result = {
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

import gensim
from gensim import corpora
from gensim.models import LdaModel
from typing import List, Dict, Any, Tuple


class ProbabilisticLatentSemanticAnalysis:
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 10,
        random_state: int = 42,
        alpha: str = 'asymmetric',
    ):
        """
        Initializes the PLSA model with specified hyperparameters.

        :param num_topics: Number of topics to extract.
        :param passes: Number of passes through the corpus during training.
        :param random_state: Seed for reproducibility.
        :param alpha: Hyperparameter that affects the sparsity of the document-topic distribution.
                      Typically set to 'symmetric' or 'asymmetric' for PLSA-like behavior.
        """
        self.num_topics = num_topics
        self.passes = passes
        self.random_state = random_state
        self.alpha = alpha
        self.dictionary = None
        self.corpus = None
        self.model = None

    def fit(self, documents: List[List[str]]):
        """
        Fits the PLSA model to the documents.

        :param documents: List of preprocessed documents (list of tokens).
        """
        # Create a dictionary representation of the documents.
        self.dictionary = corpora.Dictionary(documents)
        # Convert documents to Bag-of-Words format.
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents]
        # Initialize and train the LDA model with alpha set to 'asymmetric' to mimic PLSA.
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            random_state=self.random_state,
            alpha=self.alpha,
            eta='auto',  # You can make eta a parameter if needed.
        )

    def get_topics(self) -> List[str]:
        """
        Retrieves the topics.

        :return: List of topics with top words.
        """
        return self.model.print_topics(num_topics=self.num_topics, num_words=10)

    def get_document_topics(self, document: List[str]) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens).
        :return: List of (topic_id, probability) tuples.
        """
        if self.model is None or self.dictionary is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        # Convert the document to Bag-of-Words format.
        bow = self.dictionary.doc2bow(document)
        # Get the topic distribution for the document.
        return self.model.get_document_topics(bow, minimum_probability=0)

    def __call__(self, documents: List[List[str]]) -> Dict[str, Any]:
        """
        Makes the ProbabilisticLatentSemanticAnalysis instance callable.
        Performs PLSA topic modeling on the provided documents.

        :param documents: List of preprocessed documents (list of tokens).
        :return: Dictionary containing the trained model, dictionary, corpus, and topics.
        """
        # Fit the model to the documents.
        self.fit(documents)
        # Retrieve the topics.
        topics = self.get_topics()
        # Prepare the result dictionary.
        result = {
            'num_topics': self.num_topics,
            'passes': self.passes,
            'random_state': self.random_state,
            'alpha': self.alpha,
            'topics': topics,
            'model': self.model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
        }
        return result

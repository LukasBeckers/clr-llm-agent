import gensim
from gensim import corpora
from gensim.models import LdaModel
from typing import List, Dict, Any, Tuple, Optional
import copy


class ProbabilisticLatentSemanticAnalysis:
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 10,
        random_state: int = 42,
        alpha: str = 'asymmetric',
        top_n: int = 10
    ) -> None:
        """
        Initializes the PLSA model with specified hyperparameters.

        :param num_topics: Number of topics to extract.
        :param passes: Number of passes through the corpus during training.
        :param random_state: Seed for reproducibility.
        :param alpha: Hyperparameter that affects the sparsity of the document-topic distribution.
                      Typically set to 'symmetric' or 'asymmetric' for PLSA-like behavior.
        :param top_n: Number of top words per topic.
        """
        self.num_topics: int = num_topics
        self.passes: int = passes
        self.random_state: int = random_state
        self.alpha: str = alpha
        self.top_n: int = top_n
        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[List[List[Tuple[int, int]]]] = None
        self.model: Optional[LdaModel] = None

    def fit(self, documents: List[List[str]]) -> None:
        """
        Fits the PLSA model to the documents.

        :param documents: List of preprocessed documents (list of tokens).
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy: List[List[str]] = copy.deepcopy(documents)

        # Create a dictionary representation of the documents.
        self.dictionary = corpora.Dictionary(documents_copy)
        # Convert documents to Bag-of-Words format.
        self.corpus = [self.dictionary.doc2bow(doc) for doc in documents_copy]
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
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call the instance with documents first.")
        return self.model.print_topics(num_topics=self.num_topics, num_words=self.top_n)

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
        # Convert the document to Bag-of-Words format.
        bow = self.dictionary.doc2bow(document)
        # Get the topic distribution for the document.
        return self.model.get_document_topics(bow, minimum_probability=0)

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the ProbabilisticLatentSemanticAnalysis instance callable.
        Performs PLSA topic modeling on the provided documents.

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
            'passes': self.passes,
            'random_state': self.random_state,
            'alpha': self.alpha,
            'topics': topics,
            'model': self.model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
        }
        return result

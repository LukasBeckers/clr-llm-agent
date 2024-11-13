import gensim
from gensim import corpora
from gensim.models import LdaModel
from typing import List, Dict, Any, Tuple, Optional
import copy


class LatentDirichletAllocation:
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 1,
        iterations: int = 50,
        alpha: Any = "symmetric",
        beta: Any = "auto",
        random_state: Optional[int] = None,
        chunksize: int = 1000,
    ) -> None:
        """
        Initializes the LDA model with specified hyperparameters.

        :param num_topics: The number of topics to extract.
        :param passes: Number of passes through the corpus during training.
        :param iterations: Maximum number of iterations through the corpus when inferring the topic distribution.
        :param alpha: Hyperparameter that affects the sparsity of the document-topic distribution.
                      Can be 'symmetric', 'asymmetric', or a list of values.
        :param beta: Hyperparameter that affects the sparsity of the topic-word distribution.
                     Can be 'auto', 'symmetric', or a float value.
        :param random_state: Seed for random number generator for reproducibility.
        :param chunksize: Number of documents to be used in each training chunk.
        """
        self.num_topics: int = num_topics
        self.passes: int = passes
        self.iterations: int = iterations
        self.alpha: Any = alpha
        self.beta: Any = beta
        self.random_state: Optional[int] = random_state
        self.chunksize: int = chunksize

        self.dictionary: Optional[corpora.Dictionary] = None
        self.corpus: Optional[List[List[Tuple[int, int]]]] = None
        self.model: Optional[LdaModel] = None

    def __call__(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Performs LDA topic modeling on the provided documents.

        :param documents: List of dictionaries, each containing at least the key "Abstract Normalized".
        :return: Dictionary containing the trained model, dictionary, corpus, and topics.
        """
        # Create a deep copy to ensure original documents are not modified
        documents_copy: List[Dict[str, Any]] = copy.deepcopy(documents)

        # Step 1: Extract and tokenize the "Abstract Normalized" text
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

        # Step 2: Create a dictionary and filter out extremes to limit the number of features
        self.dictionary = corpora.Dictionary(abstracts)
        # Optionally, you can add dictionary filtering here if needed
        # For example:
        # self.dictionary.filter_extremes(no_below=5, no_above=0.5)

        # Step 3: Convert documents to Bag-of-Words format
        self.corpus = [self.dictionary.doc2bow(text) for text in abstracts]

        # Step 4: Initialize and train the LDA model
        self.model = LdaModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            alpha=self.alpha,
            eta=self.beta,
            random_state=self.random_state,
            chunksize=self.chunksize,
            update_every=1,
            per_word_topics=True,
        )

        # Step 5: Retrieve topics
        topics: List[str] = self.model.print_topics(
            num_topics=self.num_topics, num_words=10
        )

        result: Dict[str, Any] = {
            "num_topics": self.num_topics,
            "passes": self.passes,
            "iterations": self.iterations,
            "alpha": self.alpha,
            "beta": self.beta,
            "random_state": self.random_state,
            "chunksize": self.chunksize,
            "topics": topics,
            "model": self.model,
            "dictionary": self.dictionary,
            "corpus": self.corpus,
        }

        return result

    def get_document_topics(
        self, document: List[str], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: Preprocessed document (list of tokens).
        :param top_n: Number of top topics to return.
        :return: List of (topic_id, probability) tuples.
        """
        if self.model is None or self.dictionary is None:
            raise ValueError(
                "Model has not been trained yet. Call the instance with documents first."
            )

        if not isinstance(document, list) or not all(
            isinstance(token, str) for token in document
        ):
            raise ValueError("The document must be a list of strings.")

        bow = self.dictionary.doc2bow(document)
        topic_distribution = self.model.get_document_topics(
            bow, minimum_probability=0
        )
        top_topics = sorted(
            topic_distribution, key=lambda x: x[1], reverse=True
        )[:top_n]
        return top_topics

import gensim
from gensim import corpora
from gensim.models import LdaModel
from typing import List, Dict, Any, Tuple


class LatentDirichletAllocation:
    def __init__(
        self,
        num_topics: int = 10,
        passes: int = 1,
        iterations: int = 50,
        alpha: Any = "symmetric",
        beta: Any = "auto",
        random_state: int = None,
        chunksize: int = 1000,
    ):
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
        self.num_topics = num_topics
        self.passes = passes
        self.iterations = iterations
        self.alpha = alpha
        self.beta = beta
        self.random_state = random_state
        self.chunksize = chunksize

        self.dictionary = None
        self.corpus = None
        self.model = None

    def __call__(
        self,
        documents: List[Dict[str, Any]],
    ) -> Dict[str, Any]:
        """
        Performs LDA topic modeling on the provided documents.

        :param documents: List of dictionaries, each containing at least the key "Abstract Normalized".
        :return: Dictionary containing the trained model, dictionary, corpus, and topics.
        """
        # Step 1: Extract and tokenize the "Abstract Normalized" text
        texts = []
        for doc in documents:
            abstract = doc.get("Abstract Normalized", "")
            if isinstance(abstract, str):
                tokens = abstract.split()  # Assuming text is already normalized and tokenized by spaces
                texts.append(tokens)
            else:
                texts.append([])

        # Step 2: Create a dictionary and filter out extremes to limit the number of features
        self.dictionary = corpora.Dictionary(texts)
        # Optionally, you can add dictionary filtering here if needed
        # For example:
        # self.dictionary.filter_extremes(no_below=5, no_above=0.5)

        # Step 3: Convert documents to Bag-of-Words format
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

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
        topics = self.model.print_topics(num_topics=self.num_topics, num_words=10)

        result = {
            'num_topics': self.num_topics,
            'passes': self.passes,
            'iterations': self.iterations,
            'alpha': self.alpha,
            'beta': self.beta,
            'random_state': self.random_state,
            'chunksize': self.chunksize,
            'topics': topics,
            'model': self.model,
            'dictionary': self.dictionary,
            'corpus': self.corpus,
        }

        return result

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
        topic_distribution = self.model.get_document_topics(bow, minimum_probability=0)
        top_topics = sorted(topic_distribution, key=lambda x: x[1], reverse=True)[:top_n]
        return top_topics

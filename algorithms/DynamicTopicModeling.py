import gensim
from gensim import corpora
from gensim.models import LdaSeqModel
from typing import List, Dict, Any, Tuple
from collections import defaultdict


class DynamicTopicModeling:
    def __init__(
        self,
        num_topics: int = 10,
        time_key: str = "time",
        passes: int = 1,
        iterations: int = 100,
        alpha: Any = "symmetric",
        eta: Any = "auto",
        decay: float = 0.5,
        offset: float = 1.0,
        random_state: int = None,
        chunksize: int = 2000,
        evaluate_every: int = 0,
        gamma_threshold: float = 0.001,
    ):
        """
        Initializes the Dynamic Topic Modeling class with specified hyperparameters.

        :param num_topics: Number of topics to extract.
        :param time_key: The key in each document dict that indicates the time slice.
        :param passes: Number of passes through the corpus during training.
        :param iterations: Maximum number of iterations through the corpus when inferring the topic distribution.
        :param alpha: Hyperparameter that affects the sparsity of the document-topic distribution.
        :param eta: Hyperparameter that affects the sparsity of the topic-word distribution.
        :param decay: Controls the exponential decay for weights assigned to old documents.
        :param offset: Controls how fast the exponential decay rates the weights.
        :param random_state: Seed for random number generator for reproducibility.
        :param chunksize: Number of documents to be used in each training chunk.
        :param evaluate_every: How often to evaluate perplexity, etc. Set to 0 to disable.
        :param gamma_threshold: Threshold for the convergence of variational EM.
        """
        self.num_topics = num_topics
        self.time_key = time_key
        self.passes = passes
        self.iterations = iterations
        self.alpha = alpha
        self.eta = eta
        self.decay = decay
        self.offset = offset
        self.random_state = random_state
        self.chunksize = chunksize
        self.evaluate_every = evaluate_every
        self.gamma_threshold = gamma_threshold

        self.dictionary = None
        self.corpus = None
        self.time_slices = None
        self.model = None

    def fit(self, documents: List[Dict[str, Any]]):
        """
        Fits the Dynamic Topic Model to the provided documents.

        :param documents: List of dictionaries, each containing at least the keys "Abstract Normalized" and the specified time key.
        """
        # Step 1: Extract and tokenize the "Abstract Normalized" text and group by time slices
        texts_by_time = defaultdict(list)
        for doc in documents:
            abstract = doc.get("Abstract Normalized", "")
            time = doc.get(self.time_key, None)
            if time is None:
                raise ValueError(
                    f"Document is missing the time key '{self.time_key}'."
                )
            if isinstance(abstract, str):
                tokens = (
                    abstract.split()
                )  # Assuming text is already normalized and tokenized by spaces
                texts_by_time[time].append(tokens)
            else:
                # If "Abstract Normalized" is not a string, skip or handle accordingly
                texts_by_time[time].append([])

        # Step 2: Sort the time slices in chronological order
        sorted_times = sorted(texts_by_time.keys())
        texts = []
        self.time_slices = []
        for time in sorted_times:
            current_texts = texts_by_time[time]
            texts.extend(current_texts)
            self.time_slices.append(len(current_texts))

        if not self.time_slices:
            raise ValueError("No documents found after processing.")

        # Step 3: Create a dictionary and filter out extremes to limit the number of features
        self.dictionary = corpora.Dictionary(texts)
        # Optionally, you can add dictionary filtering here if needed
        # For example:
        # self.dictionary.filter_extremes(no_below=5, no_above=0.5)

        # Step 4: Convert documents to Bag-of-Words format
        self.corpus = [self.dictionary.doc2bow(text) for text in texts]

        # Step 5: Initialize and train the Dynamic Topic Model
        self.model = LdaSeqModel(
            corpus=self.corpus,
            id2word=self.dictionary,
            time_slice=self.time_slices,
            num_topics=self.num_topics,
            passes=self.passes,
            iterations=self.iterations,
            alpha=self.alpha,
            eta=self.eta,
            decay=self.decay,
            offset=self.offset,
            random_state=self.random_state,
            chunksize=self.chunksize,
            evaluate_every=self.evaluate_every,
            gamma_threshold=self.gamma_threshold,
        )

    def get_topics(self, top_n: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Retrieves the topics across all time slices.

        :param top_n: Number of top words per topic per time slice.
        :return: A list where each element corresponds to a time slice and contains a list of topics, each topic being a list of (word, probability) tuples.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call the instance with documents first."
            )

        all_topics = []
        for t in range(len(self.time_slices)):
            time_topics = []
            for topic_id in range(self.num_topics):
                topic_terms = self.model.print_topic(topicid=topic_id, time=t)
                # Parse the topic terms into (word, probability) tuples
                terms = [
                    (term.split("*")[1].strip('"'), float(term.split("*")[0]))
                    for term in topic_terms.split(" + ")
                ]
                time_topics.append((topic_id, terms))
            all_topics.append(time_topics)
        return all_topics

    def get_document_topics(
        self, document: Dict[str, Any], top_n: int = 10
    ) -> List[Tuple[int, float]]:
        """
        Gets the topic distribution for a single document.

        :param document: A dictionary containing at least the keys "Abstract Normalized" and the specified time key.
        :param top_n: Number of top topics to return.
        :return: List of (topic_id, probability) tuples.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call the instance with documents first."
            )

        abstract = document.get("Abstract Normalized", "")
        time = document.get(self.time_key, None)
        if time is None:
            raise ValueError(
                f"Document is missing the time key '{self.time_key}'."
            )

        tokens = abstract.split() if isinstance(abstract, str) else []
        bow = self.dictionary.doc2bow(tokens)
        # Infer topic distribution for the document
        topic_dist = self.model.inference([bow])[0]
        # Flatten the list and sort
        topic_probs = sorted(topic_dist, key=lambda x: x[1], reverse=True)[
            :top_n
        ]
        return topic_probs

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the DynamicTopicModeling instance callable. Performs dynamic topic modeling on the provided documents.

        :param documents: List of dictionaries, each containing at least the keys "Abstract Normalized" and the specified time key.
        :return: Dictionary containing model results.
        """
        self.fit(documents)
        topics = self.get_topics(top_n=10)
        result = {
            "num_topics": self.num_topics,
            "time_slices": self.time_slices,
            "topics": topics,
            "alpha": self.alpha,
            "eta": self.eta,
            "decay": self.decay,
            "offset": self.offset,
            "random_state": self.random_state,
        }
        return result

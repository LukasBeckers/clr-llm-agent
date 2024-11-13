import gensim
from gensim import corpora
from gensim.models import LdaSeqModel
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict
import copy
from datetime import datetime
import logging

class DynamicTopicModeling:
    def __init__(
        self,
        num_topics: int = 5,  # Reduced number of topics
        passes: int = 1,
        iterations: int = 50,  # Reduced iterations
        alpha: Any = "symmetric",  # Start with symmetric alpha
        eta: Any = "auto",  # Explicitly set eta
        random_state: Optional[int] = None,
        chunksize: int = 1000,  # Adjusted chunksize
        **kwargs: Any
    ) -> None:
        """
        Initializes the Dynamic Topic Modeling class with specified hyperparameters.
        """
        self.num_topics: int = num_topics
        self.passes: int = passes
        self.iterations: int = iterations
        self.alpha: Any = alpha
        self.eta: Any = eta
        self.random_state: Optional[int] = random_state
        self.chunksize: int = chunksize

        # Fixed time key as per your requirement
        self.time_key: str = "PublicationDate"

        # Initialize attributes
        self.dictionary: corpora.Dictionary = corpora.Dictionary()
        self.corpus: List[List[Tuple[int, int]]] = []
        self.time_slices: List[int] = []
        self.model: Optional[LdaSeqModel] = None

        # Store additional parameters internally if needed
        self.additional_params: Dict[str, Any] = kwargs

    def _parse_publication_date(self, date_str: str) -> Optional[datetime]:
        """
        Parses the PublicationDate string into a datetime object.
        """
        for fmt in ("%Y-%b", "%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _generate_alpha_vector(self) -> List[float]:
        """
        Generates an asymmetric alpha vector for the number of topics.
        """
        alpha_vector = [0.1] + [0.05] * (self.num_topics - 1)
        return alpha_vector

    def fit(self, documents: List[Dict[str, Any]]) -> None:
        """
        Fits the Dynamic Topic Model to the provided documents.
        """
        logging.info("Starting the fitting process.")
        documents_copy: List[Dict[str, Any]] = copy.deepcopy(documents)

        # Step 1: Extract and group documents by time slices
        texts_by_time: defaultdict = defaultdict(list)
        for doc in documents_copy:
            abstract = doc.get("Abstract Normalized", "")
            publication_date = doc.get(self.time_key, None)

            if publication_date is None:
                continue  # Skip documents without PublicationDate

            if isinstance(publication_date, str):
                parsed_date = self._parse_publication_date(publication_date)
                if parsed_date is None:
                    continue  # Skip if date parsing fails
            else:
                continue  # Skip if PublicationDate is not a string

            if isinstance(abstract, list) and all(isinstance(token, str) for token in abstract):
                if len(abstract) < 3:
                    continue  # Skip very short abstracts
                texts_by_time[parsed_date].append(abstract)
            else:
                continue  # Skip documents with invalid Abstract Normalized

        # Step 2: Sort time slices chronologically
        sorted_times: List[datetime] = sorted(texts_by_time.keys())
        texts: List[List[str]] = []
        self.time_slices = []
        for time in sorted_times:
            current_texts = texts_by_time[time]
            texts.extend(current_texts)
            self.time_slices.append(len(current_texts))

        if not self.time_slices:
            raise ValueError("No valid documents found after processing.")

        # Ensure each time slice has enough documents
        min_docs_per_slice = 5
        filtered_texts = []
        filtered_time_slices = []
        for time, texts_in_time in texts_by_time.items():
            if len(texts_in_time) >= min_docs_per_slice:
                filtered_texts.extend(texts_in_time)
                filtered_time_slices.append(len(texts_in_time))

        if not filtered_time_slices:
            raise ValueError("No time slices have the minimum required number of documents.")

        self.time_slices = filtered_time_slices
        self.dictionary = corpora.Dictionary(filtered_texts)
        self.corpus = [self.dictionary.doc2bow(text) for text in filtered_texts]

        # Step 5: Prepare the 'alphas' parameter
        if isinstance(self.alpha, str):
            if self.alpha.lower() == 'asymmetric':
                alphas = self._generate_alpha_vector()
            elif self.alpha.lower() == 'symmetric':
                alphas = 0.01
            else:
                raise ValueError(f"Unsupported alpha value: {self.alpha}")
        elif isinstance(self.alpha, (list, tuple)):
            if len(self.alpha) != self.num_topics:
                raise ValueError(f"Length of alpha vector ({len(self.alpha)}) does not match num_topics ({self.num_topics}).")
            alphas = self.alpha
        elif isinstance(self.alpha, (float, int)):
            alphas = float(self.alpha)
        else:
            raise ValueError(f"Unsupported type for alpha: {type(self.alpha)}")

        logging.info("Initializing LdaSeqModel with the following parameters:")
        logging.info(f"Number of Topics: {self.num_topics}")
        logging.info(f"Alpha: {alphas}")
        logging.info(f"Iterations: {self.iterations}")
        logging.info(f"Chunksize: {self.chunksize}")
        logging.info(f"Eta: {self.eta}")

        # Step 6: Initialize and train the Dynamic Topic Model

        try:
            self.model = LdaSeqModel(
                corpus=self.corpus,
                id2word=self.dictionary,
                time_slice=self.time_slices,
                num_topics=self.num_topics,
                passes=self.passes,
                lda_inference_max_iter=self.iterations,
                alphas=alphas,
                random_state=self.random_state,
                chunksize=self.chunksize,
                initialize='gensim',
            )

            
            logging.info("LdaSeqModel trained successfully.")
        except Exception as e:
            logging.error("An error occurred during LdaSeqModel training.")
            logging.error(str(e))
            raise

    def get_topics(self, top_n: int = 10) -> List[List[Tuple[str, float]]]:
        """
        Retrieves the topics across all time slices.

        :param top_n: Number of top words per topic per time slice.
        :return: A list where each element corresponds to a time slice and contains a list of topics,
                 each topic being a list of (word, probability) tuples.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call the instance with documents first."
            )

        all_topics: List[List[Tuple[str, float]]] = []
        for t in range(len(self.time_slices)):
            time_topics: List[Tuple[str, float]] = []
            for topic_id in range(self.num_topics):
                topic_terms = self.model.print_topic(topicid=topic_id, time=t, top_n=top_n)
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

        :param document: A dictionary containing at least the keys "Abstract Normalized" and "PublicationDate".
        :param top_n: Number of top topics to return.
        :return: List of (topic_id, probability) tuples.
        """
        if self.model is None:
            raise ValueError(
                "Model has not been trained yet. Call the instance with documents first."
            )

        abstract = document.get("Abstract Normalized", "")
        publication_date = document.get(self.time_key, None)

        if publication_date is None:
            raise ValueError(
                f"Document is missing the time key '{self.time_key}'."
            )

        if isinstance(publication_date, str):
            parsed_date = self._parse_publication_date(publication_date)
            if parsed_date is None:
                raise ValueError("PublicationDate format is invalid.")
        else:
            raise ValueError("PublicationDate must be a string.")

        if isinstance(abstract, list) and all(isinstance(token, str) for token in abstract):
            tokens = abstract
        else:
            raise ValueError(
                "The 'Abstract Normalized' field must be a list of strings."
            )

        bow = self.dictionary.doc2bow(tokens)
        # Infer topic distribution for the document
        topic_dist = self.model.inference([bow])[0]
        # Flatten the list and sort
        topic_probs = sorted(topic_dist, key=lambda x: x[1], reverse=True)[:top_n]
        return topic_probs

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the DynamicTopicModeling instance callable. Performs dynamic topic modeling on the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing model results.
        """
        self.fit(documents)
        topics = self.get_topics(top_n=10)
        result: Dict[str, Any] = {
            "num_topics": self.num_topics,
            "time_slices": self.time_slices,
            "topics": topics,
            "alpha": self.alpha,
            "random_state": self.random_state,
            "chunksize": self.chunksize,
            "passes": self.passes,
            "iterations": self.iterations,
            "model": self.model,
            "dictionary": self.dictionary,
            "corpus": self.corpus,
        }
        return result

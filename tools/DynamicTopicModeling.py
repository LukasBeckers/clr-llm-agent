import gensim
from gensim import corpora
from gensim.models import LdaSeqModel
from typing import List, Dict, Any, Tuple
from collections import defaultdict

class DynamicTopicModeling:
    def __init__(self):
        """
        Initializes the Dynamic Topic Modeling class.
        No arguments are required during initialization.
        """
        pass

    def __call__(
        self,
        documents: List[Dict[str, Any]],
        num_topics: int = 10,
        time_key: str = 'time',
        passes: int = 1,
        iterations: int = 100,
        alpha: Any = 'symmetric',
        eta: Any = 'auto',
        decay: float = 0.5,
        offset: float = 1.0,
        random_state: int = None,
        chunksize: int = 2000,
        evaluate_every: int = 0,
        gamma_threshold: float = 0.001
    ) -> Tuple[LdaSeqModel, corpora.Dictionary, List[List[tuple]], List[int]]:
        """
        Performs Dynamic Topic Modeling on the provided documents.

        Parameters:
            documents (List[Dict[str, Any]]): 
                A list of dictionaries, each containing at least the keys "Abstract Normalized" and the specified time key.
            num_topics (int, optional): 
                The number of topics to extract. Default is 10.
            time_key (str, optional):
                The key in each document dict that indicates the time slice. Default is 'time'.
            passes (int, optional): 
                Number of passes through the corpus during training. Default is 1.
            iterations (int, optional): 
                Maximum number of iterations through the corpus when inferring the topic distribution. Default is 100.
            alpha (str or list, optional): 
                Hyperparameter that affects the sparsity of the document-topic distribution. 
                Can be 'symmetric', 'asymmetric', or a list of values. Default is 'symmetric'.
            eta (str or float, optional): 
                Hyperparameter that affects the sparsity of the topic-word distribution. 
                Can be 'auto', 'symmetric', or a float value. Default is 'auto'.
            decay (float, optional):
                Controls the exponential decay for weights assigned to old documents. Default is 0.5.
            offset (float, optional):
                Controls how fast the exponential decay rates the weights. Default is 1.0.
            random_state (int, optional): 
                Seed for random number generator for reproducibility. Default is None.
            chunksize (int, optional): 
                Number of documents to be used in each training chunk. Default is 2000.
            evaluate_every (int, optional):
                How often to evaluate perplexity, etc. Set to 0 to disable. Default is 0.
            gamma_threshold (float, optional):
                Threshold for the convergence of variational EM. Default is 0.001.

        Returns:
            Tuple containing:
                - LdaSeqModel: The trained Dynamic Topic Model.
                - corpora.Dictionary: The dictionary mapping of word IDs to words.
                - List[List[tuple]]: The corpus in BoW format.
                - List[int]: The list indicating the number of documents in each time slice.
        """
        # Step 1: Extract and tokenize the "Abstract Normalized" text and group by time slices
        texts_by_time = defaultdict(list)
        for doc in documents:
            abstract = doc.get("Abstract Normalized", "")
            time = doc.get(time_key, None)
            if time is None:
                raise ValueError(f"Document is missing the time key '{time_key}'.")
            if isinstance(abstract, str):
                tokens = abstract.split()  # Assuming text is already normalized and tokenized by spaces
                texts_by_time[time].append(tokens)
            else:
                # If "Abstract Normalized" is not a string, skip or handle accordingly
                texts_by_time[time].append([])

        # Step 2: Sort the time slices in chronological order
        sorted_times = sorted(texts_by_time.keys())
        texts = []
        time_slices = []
        for time in sorted_times:
            current_texts = texts_by_time[time]
            texts.extend(current_texts)
            time_slices.append(len(current_texts))

        if not time_slices:
            raise ValueError("No documents found after processing.")

        # Step 3: Create a dictionary and filter out extremes to limit the number of features
        dictionary = corpora.Dictionary(texts)
        # Optionally, you can add dictionary filtering here if needed
        # For example:
        # dictionary.filter_extremes(no_below=5, no_above=0.5)

        # Step 4: Convert documents to Bag-of-Words format
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Step 5: Initialize and train the Dynamic Topic Model
        lda_seq = LdaSeqModel(
            corpus=corpus,
            id2word=dictionary,
            time_slice=time_slices,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=eta,
            decay=decay,
            offset=offset,
            random_state=random_state,
            chunksize=chunksize,
            evaluate_every=evaluate_every,
            gamma_threshold=gamma_threshold
        )

        return lda_seq, dictionary, corpus, time_slices

# Example Usage
if __name__ == "__main__":
    # Sample documents with a 'time' key
    sample_documents = [
        {"Abstract Normalized": "machine learning algorithms for data analysis", "time": "2020"},
        {"Abstract Normalized": "deep learning and neural networks in computer vision", "time": "2020"},
        {"Abstract Normalized": "statistical models and data mining techniques", "time": "2021"},
        {"Abstract Normalized": "natural language processing and text mining", "time": "2021"},
        {"Abstract Normalized": "reinforcement learning and artificial intelligence", "time": "2022"},
        {"Abstract Normalized": "graph neural networks and their applications", "time": "2022"},
        {"Abstract Normalized": "transfer learning and domain adaptation methods", "time": "2023"},
        {"Abstract Normalized": "explainable AI and model interpretability", "time": "2023"}
    ]

    # Initialize the Dynamic Topic Modeling class
    dtm = DynamicTopicModeling()

    # Perform Dynamic Topic Modeling analysis
    model, dictionary, corpus, time_slices = dtm(
        documents=sample_documents,
        num_topics=2,
        time_key='time',
        passes=10,
        iterations=100,
        alpha='auto',
        eta='auto',
        decay=0.5,
        offset=1.0,
        random_state=42,
        chunksize=100,
        evaluate_every=10,
        gamma_threshold=0.001
    )

    # Print the topics for each time slice
    for t in range(len(time_slices)):
        print(f"Time Slice {t+1}:")
        for topic_id in range(model.num_topics):
            topic_terms = model.print_topic(topicid=topic_id, time=t)
            print(f"  Topic {topic_id}: {topic_terms}")
        print("\n")

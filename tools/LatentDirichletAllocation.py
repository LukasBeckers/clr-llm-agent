import gensim
from gensim import corpora
from gensim.models import LdaModel


class LatentDirichletAllocation:
    def __init__(self):
        """
        Initializes the LDA model class.
        No arguments are required during initialization.
        """
        pass

    def __call__(
        self,
        documents: List[Dict[str, Any]],
        num_topics: int = 10,
        passes: int = 1,
        iterations: int = 50,
        alpha: Any = "symmetric",
        beta: Any = "auto",
        random_state: int = None,
        chunksize: int = 1000,
    ) -> Tuple[LdaModel, corpora.Dictionary, List[List[tuple]]]:
        """
        Performs LDA topic modeling on the provided documents.

        Parameters:
            documents (List[Dict[str, Any]]):
                A list of dictionaries, each containing at least the key "Abstract Normalized".
            num_topics (int, optional):
                The number of topics to extract. Default is 10.
            passes (int, optional):
                Number of passes through the corpus during training. Default is 1.
            iterations (int, optional):
                Maximum number of iterations through the corpus when inferring the topic distribution. Default is 50.
            alpha (str or list, optional):
                Hyperparameter that affects the sparsity of the document-topic distribution.
                Can be 'symmetric', 'asymmetric', or a list of values. Default is 'symmetric'.
            beta (str or float, optional):
                Hyperparameter that affects the sparsity of the topic-word distribution.
                Can be 'auto', 'symmetric', or a float value. Default is 'auto'.
            random_state (int, optional):
                Seed for random number generator for reproducibility. Default is None.
            chunksize (int, optional):
                Number of documents to be used in each training chunk. Default is 1000.

        Returns:
            Tuple containing:
                - LdaModel: The trained LDA model.
                - corpora.Dictionary: The dictionary mapping of word IDs to words.
                - List[List[tuple]]: The corpus in BoW format.
        """
        # Step 1: Extract and tokenize the "Abstract Normalized" text
        texts = []
        for doc in documents:
            abstract = doc.get("Abstract Normalized", "")
            if isinstance(abstract, str):
                tokens = (
                    abstract.split()
                )  # Assuming text is already normalized and tokenized by spaces
                texts.append(tokens)
            else:
                # If "Abstract Normalized" is not a string, skip or handle accordingly
                texts.append([])

        # Step 2: Create a dictionary and filter out extremes to limit the number of features
        dictionary = corpora.Dictionary(texts)
        # Optionally, you can add dictionary filtering here if needed
        # For example:
        # dictionary.filter_extremes(no_below=5, no_above=0.5)

        # Step 3: Convert documents to Bag-of-Words format
        corpus = [dictionary.doc2bow(text) for text in texts]

        # Step 4: Initialize and train the LDA model
        lda_model = LdaModel(
            corpus=corpus,
            id2word=dictionary,
            num_topics=num_topics,
            passes=passes,
            iterations=iterations,
            alpha=alpha,
            eta=beta,
            random_state=random_state,
            chunksize=chunksize,
            update_every=1,
            per_word_topics=True,
        )

        return lda_model, dictionary, corpus

from algorithms.IAlgorithm import IAlgorithm
from tomotopy import LDAModel, TermWeight, ParallelScheme
from typing import Dict, List, Optional, Any, Union, Iterable, Callable, Tuple
from kneed import KneeLocator
import copy
import numpy as np
import os
import matplotlib.pyplot as plt


class LatentDirichletAllocation(IAlgorithm):
    algorithm_description = ""

    def __init__(
        self,
        # Model Parameters
        tw: Union[int, TermWeight] = TermWeight.ONE,
        min_cf: int = 0,
        min_df: int = 0,
        rm_top: int = 0,
        k: int = 6,
        alpha: Union[float, Iterable[float]] = 0.1,
        eta: float = 0.01,
        seed: int = 42,
        transform: Optional[Callable] = None,
        # Training Parameters
        iter: int = 5000,
        workers: int = 10,
        parallel=ParallelScheme.DEFAULT,
        freeze_topics: bool = False,
        callback_interval: int = 10,
        callback: Optional[Callable] = None,
        show_progress: bool = False,
        **kwargs: Any 
    ) -> None:
        """
        LDA Algorithm, the base topic modeling algorithm on which most other
        tm algorithms are based.

        tomotopy LDA Model initialization parameters:

        tw : Union[int, TermWeight]
            term weighting scheme in TermWeight. The default value is
            TermWeight.ONE
        min_cf : int
            minimum collection frequency of words. Words with a smaller collection frequency than min_cf are excluded from the model. The default value is 0, which means no words are excluded.
        min_df : int
            minimum document frequency of words. Words with a smaller document frequency than min_df are excluded from the model. The default value is 0, which means no words are excluded
        rm_top : int
            the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more. The default value is 0, which means no top words are removed.
        k : Optional(int)
            the number of topics between 1 ~ 32767
            if k == None topic models from 0 to 100 will be calculated and the
            optimal number of topics will be found by searching the knee of the
            perplexity number of topics curve.
        alpha : Union[float, Iterable[float]]
            hyperparameter of Dirichlet distribution for document-topic, given as a single float in case of symmetric prior and as a list with length k of float in case of asymmetric prior.
        eta : float
            hyperparameter of Dirichlet distribution for topic-word
        seed : int
            random seed. The default value is a random number from std::random_device{} in C++
        transform : Callable[dict, dict]
            a callable object to manipulate arbitrary keyword arguments for a specific topic model

        Training Parameters
        iter : int
            the number of iterations of Gibbs-sampling
        workers : int
            an integer indicating the number of workers to perform samplings. If workers is 0, the number of cores in the system will be used.
        parallel : Union[int, ParallelScheme]
            the parallelism scheme for training. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.
        freeze_topics : bool
            prevents to create a new topic when training. Only valid for HLDAModel
        callback_interval : int
            the intderval of calling callback function. If callback_interval <= 0, callback function is called at the beginning and the end of training.
        callback : Callable[[LDAModel, int, int], None]
            a callable object which is called every callback_interval iterations. It receives three arguments: the current model, the current number of iterations, and the total number of iterations.
        show_progress : bool
            If True, it shows progress bar during training using tqdm package.

        """
        # Setting the model Parameters
        self.tw = tw
        self.min_cf = min_cf
        self.min_df = min_df
        self.rm_top = rm_top
        self.k = k
        self.alpha = alpha
        self.eta = eta
        self.seed = seed
        self.transform = transform

        # Setting the training parameters
        self.iter = iter
        self.workers = workers
        self.parallel = parallel
        self.freeze_topics = freeze_topics
        self.callback_interval = callback_interval
        self.callback = callback
        self.show_progress = show_progress

    def _visualize(self, results: Dict[str, Union[Dict, str, List]]) -> Dict[str, Union[Dict, str, List]]:
        """
        Creates Visualizations of the generated results including:

        - Word Clouds for each topic.
        - Histogram showing word counts per topic with labels.
        """
        import matplotlib.pyplot as plt
        from wordcloud import WordCloud
        import os

        # Ensure the 'visualizations' directory exists
        if not os.path.exists("visualizations"):
            os.makedirs("visualizations")

        # Plotting Word Clouds for each topic
        topic_words = results["Topic Words"]
        num_topics = len(topic_words)
        
        # Determine the grid size for plotting all word clouds in one figure
        cols = 3  # Number of columns in the grid
        rows = (num_topics + cols - 1) // cols  # Calculate number of rows

        fig, axes = plt.subplots(rows, cols, figsize=(6, 6))
        axes = axes.flatten()

        for i, (topic_key, words) in enumerate(topic_words.items()):
            # Convert list of tuples into a dictionary for WordCloud
            word_freq = {word: prob for word, prob in words}
            # Generate word cloud
            wordcloud = WordCloud(width=400,
                                  height=400, background_color='white').generate_from_frequencies(word_freq)
            # Plot word cloud
            ax = axes[i]
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            ax.set_title(f"Topic {i}", fontsize=14)

        # Hide any unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis('off')

        plt.tight_layout()
        wordclouds_filepath = os.path.join("visualizations", "TopicWordClouds.png")
        plt.savefig(wordclouds_filepath)
        plt.close(fig)

        # Adding the word clouds to the results
        results["Topic Word Clouds Explanation"] = """
        A plot that shows the word clouds for each topic. Each word cloud represents the top words in the topic, with the size of each word corresponding to its probability in the topic.
        """
        results["Topic Word Clouds"] = wordclouds_filepath

        # Plotting Histogram of Word Counts per Topic
        counts_per_topic = results["Counts Per Topic"]  # Assuming this is a list of word counts per topic
        topics = list(range(len(counts_per_topic)))  # List of topic indices

        fig, ax = plt.subplots(figsize=(6, 6))
        bars = ax.bar(topics, counts_per_topic)
        ax.set_title("Word Counts per Topic")
        ax.set_xlabel("Topic")
        ax.set_ylabel("Word Count")
        ax.set_xticks(topics)
        ax.set_xticklabels([f"Topic {i}" for i in topics],
                           rotation="vertical")

        # Adding labels on top of each bar
        for bar in bars:
            yval = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                yval + max(counts_per_topic) * 0.01,  # Adjust position above the bar
                f'{int(yval)}',
                ha='center',
                va='bottom'
            )

        histogram_filepath = os.path.join("visualizations", "WordCountsPerTopic.png")
        fig.tight_layout()
        fig.savefig(histogram_filepath)
        plt.close(fig)

        # Adding the histogram to the results
        results["Word Counts per Topic Explanation"] = """
        A histogram that shows the total word counts for each topic.
        This provides an overview of how many words are associated with each topic.
        Each bar is labeled with the exact word count to make it easier to read.
        """
        results["Word Counts per Topic"] = histogram_filepath

        return results

    def _find_k(self, documents: List[Dict[str, Any]]) -> Tuple[int, LDAModel]:
        """
        Finds the optimal number of topics (k) by training models with different k
        values and selecting the one with the highest coherence score.

        Args:
            documents (List[Dict[str, Any]]): The list of documents to analyze.

        Returns:
            Tuple[int, LDAModel]: The optimal number of topics and the trained model.
        """
        # Prepare the corpus
        corpus = [document["AbstractNormalized"] for document in documents]

        # Define the range of k values to test
        min_k = 2
        max_k = 20
        ks = range(min_k, max_k + 1)

        coherence_scores = []
        models = []

        for k in ks:
            print(f"Training model with k = {k}")

            # Create a model with the current k
            model = LDAModel(
                tw=self.tw,
                min_cf=self.min_cf,
                min_df=self.min_df,
                rm_top=self.rm_top,
                k=k,
                alpha=self.alpha,
                eta=self.eta,
                seed=self.seed,
                transform=self.transform,
            )

            # Add documents to the model
            for words in corpus:
                model.add_doc(words)

            # Train the model
            model.train(
                iter=self.iter,
                workers=self.workers,
                parallel=self.parallel,
                freeze_topics=self.freeze_topics,
                show_progress=self.show_progress,
            )

            # Compute the coherence score using the 'c_v' metric
            cm = tp.coherence.CoherenceModel(model=model, coherence='c_v')
            coherence = cm.get_score()

            coherence_scores.append(coherence)
            models.append(model)

            print(f"Coherence Score for k = {k}: {coherence}")

        # Find the k with the maximum coherence score
        max_coherence = max(coherence_scores)
        best_index = coherence_scores.index(max_coherence)
        best_k = ks[best_index]
        best_model = models[best_index]

        print(f"Optimal number of topics (k): {best_k} with Coherence Score: {max_coherence}")
        
        return best_k, best_model
            
    def __call__(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:

        # Making a deep copy of documents, so that the original documents will
        # not be changed in this algorithm.
        print("In LDA")
        documents = copy.deepcopy(documents)
        # remove all documents that have no key Abstarct Normalized
        len_documents_initially = len(documents)
        documents = [
            document
            for document in documents
            if "AbstractNormalized" in document.keys()
        ]
        print("Normalized Documents")
        # Performing the Analysis
        if self.k is None:
            # Searching for the optimal value of k if k=None
            k, model = self._find_k(documents)
        else:
            # Performing the normal LDA if k is not None
            print("Creating the model")
            model = LDAModel(
                tw=self.tw,
                min_cf=self.min_cf,
                min_df=self.min_df,
                rm_top=self.rm_top,
                k=self.k,
                alpha=self.alpha,
                eta=self.eta,
                seed=self.seed,
                transform=self.transform,
            )
            print("Adding the Documents")
            for document in documents:
                words = document["AbstractNormalized"]
                model.add_doc(words)

            print("Training the model")
            model.train(
                iter=self.iter,
                workers=self.workers,
                parallel=self.parallel,
                freeze_topics=self.freeze_topics,
                show_progress=self.show_progress,
            )
        print("Extracting the results")
     
        # Extracting the results
        results = {
            "Documents Analyzed": len(documents),
            "Documents discarded because publication date could not be parsed": len(
                documents
            )
            - len_documents_initially,
            "Topic Words Explanation": """
Topic Word distribution for every topic for the top 10 words.
It is presented in form of a dictionary with items:
    topic<k>_timestamp<t>: [(<word>, probability)...]
            """,
            "Topic Words": {},
            "Counts Per Topic Explanation": """
The number of words allocated to each topic in the form of [n_words_topic_1, 
n_words_topic_2....])
            """,
            "Counts Per Topic": model.get_count_by_topics(),
            "Hyperparameters Explanation": """LDA Algorithm, the base topic modeling algorithm on which most other
        tm algorithms are based.

        tomotopy LDA Model initialization parameters:

        tw : Union[int, TermWeight]
            term weighting scheme in TermWeight. The default value is
            TermWeight.ONE
        min_cf : int
            minimum collection frequency of words. Words with a smaller collection frequency than min_cf are excluded from the model. The default value is 0, which means no words are excluded.
        min_df : int
            minimum document frequency of words. Words with a smaller document frequency than min_df are excluded from the model. The default value is 0, which means no words are excluded
        rm_top : int
            the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more. The default value is 0, which means no top words are removed.
        k : Optional(int)
            the number of topics between 1 ~ 32767
            if k == None topic models from 0 to 100 will be calculated and the
            optimal number of topics will be found by searching the knee of the
            perplexity number of topics curve.
        alpha : Union[float, Iterable[float]]
            hyperparameter of Dirichlet distribution for document-topic, given as a single float in case of symmetric prior and as a list with length k of float in case of asymmetric prior.
        eta : float
            hyperparameter of Dirichlet distribution for topic-word
        seed : int
            random seed. The default value is a random number from std::random_device{} in C++
        transform : Callable[dict, dict]
            a callable object to manipulate arbitrary keyword arguments for a specific topic model

        Training Parameters
        iter : int
            the number of iterations of Gibbs-sampling
        workers : int
            an integer indicating the number of workers to perform samplings. If workers is 0, the number of cores in the system will be used.
        parallel : Union[int, ParallelScheme]
            the parallelism scheme for training. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.""",
            "Hyperparameters": {
                "tw": self.tw,
                "min_cf": self.min_cf,
                "min_df": self.min_df,
                "rm_top": self.rm_top,
                "k": self.k,
                "alpha": self.alpha,
                "eta": self.eta  ,
                "seed": self.seed,
                "transform": self.transform ,
                "iter": self.iter
            }
        }

        print("after Results initialization")
        
        for k in range(self.k):
            topic_words = model.get_topic_words(k, top_n=10)
            results["Topic Words"]["topic{}".format(k)] = topic_words

        print("After Adding Topic Words")

        # # Adding visualizations to the results

        results = self._visualize(results=results)

        print("After adding Visualizations")

        return results

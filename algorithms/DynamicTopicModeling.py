from algorithms.IAlgorithm import IAlgorithm
from tomotopy import DTModel, TermWeight, ParallelScheme
from tomotopy.utils import Corpus
from typing import List, Dict, Any, Tuple, Optional, Union, Callable
from collections import defaultdict
import copy
from datetime import datetime, timedelta
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os


class DynamicTopicModeling(IAlgorithm):
    algorithm_description = """

This text should help you in two cases: 
    Selecting the algorithm for the right task and 
    calibrating the algorithm after it is selected. 
    
KEY_FOR_ALGORITHM_SELECTION: DynamicTopicModeling

Description: 
    Dynamic Topic Modeling is an adaptation of the normal 
    LatentDirichletAllocation algorithm. In contrast to the normal LDA
    it analyzes how the distribution and composition of topics varies 
    over time. 

    You can set in how many evenly distributed timepoints the dataset will be
    split (based on the PublicationDate of the articles in the dataset), and 
    the topic distributions will be Calculated for each time-step. 
    Choose the number of time-steps wisely, a to low number can lead to too low
    resolution to see fine grained structures in the dataset while a to high 
    number of timesteps can lead to datasparsity at some timesteps. Keep the 
    size of your dataset in mind when choosing the number of timepoints, the 
    default is 20.

    the algorithms is structured in the following way:
        it is a python class in which all hyperparameters are set in the 
        __init__ function

        it also has a __call__ method which only takes the dataset as input and
        returns a dictionary with the results. 

    The results of this algorithms when it eccecuted successfully are the 
    following: 
        1. The topic word distribution at every timepoint for every topic. 
        2. The counts of words per topic at every timepoint for every topic. 

        3. Based on the topic_counts a Plot that shows the word counts per 
           topic overtime.
        3.1 The topic distributions overtime. For all topics at all timepoints. 
        4. A plot of the top n words for every topic at each timepoint. 

    parameters that you can set for this algorithm are:  

        tw : Union[int, TermWeight]
            term weighting scheme in TermWeight. The default value is TermWeight.ONE

        min_cf : int
            minimum collection frequency of words. Words with a smaller collection frequency than min_cf are excluded from the model. The default value is 0, which means no words are excluded.

        min_df : int
            minimum document frequency of words. Words with a smaller document frequency than min_df are excluded from the model. The default value is 0, which means no words are excluded

        rm_top : int
            the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more. The default value is 0, which means no top words are removed.

        k : int
            the number of topics between 1 ~ 32767

        t : int
            the number of timpoints

        alpha_var : float
            transition variance of alpha (per-document topic distribution)

        eta_var : float
            variance of eta (topic distribution of each document) from its alpha

        phi_var : float
            transition variance of phi (word distribution of each topic)

        lr_a : float
            shape parameter a greater than zero, for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        lr_b : float
            shape parameter b greater than or equal to zero, for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        lr_c : float
            shape parameter c with range (0.5, 1], for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        seed : int
            random seed. default value is a random number from std::random_device{} in C++
            Set the seed for reproducability always to 42 please. 

        corpus : Corpus
            a list of documents to be added into the model

        transform : Callable[dict, dict]
            a callable object to manipulate arbitrary keyword arguments for a specific topic model

        Train Parameters

        iter : int
            the number of iterations of Gibbs-sampling

        workers : int
            an integer indicating the number of workers to perform samplings. If workers is 0, the number of cores in the system will be used.

        parallel : Union[int, ParallelScheme]
            the parallelism scheme for training. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.

        freeze_topics : bool
            prevents to create a new topic when training. Only valid for HLDAModel

        callback_interval : int
            the interval of calling callback function. If callback_interval <= 0, callback function is called at the beginning and the end of training.

        callback : Callable[[LDAModel, int, int], None]
            a callable object which is called every callback_interval iterations. It receives three arguments: the current model, the current number of iterations, and the total number of iterations.

        show_progress : bool
            If True, it shows progress bar during training using tqdm package.
    
        default Values are: 

        tw: Union[TermWeight, int] = TermWeight.ONE,
        min_cf: int = 0,
        min_df: int = 0,
        rm_top: int = 0,
        k: int = 10,
        t: int = 20,
        alpha_var: float = 0.1,
        eta_var: float = 0.1,
        phi_var: float = 0.1,
        lr_a: float = 0.01,
        lr_b: float = 0.1,
        lr_c: float = 0.55,
        seed: int = 42,
        corpus: Optional[Corpus] = None,
        transform: Optional[Callable[dict, dict]] = None,
        # Train parameters to be set by agent
        iter=1000,
        freeze_topics=False,
        # Train Parameters to be set by human (default config)
        workers: int = 4,
        parallel: Union[int, ParallelScheme] = 1,
        callback_interval=10,
        callback=None,
        show_progress=False,
    
        """

    def __init__(
        self,
        # DTM init params
        tw: Union[TermWeight, int] = TermWeight.ONE,
        min_cf: int = 0,
        min_df: int = 0,
        rm_top: int = 0,
        k: int = 10,
        t: int = 10,
        alpha_var: float = 0.1,
        eta_var: float = 0.1,
        phi_var: float = 0.1,
        lr_a: float = 0.01,
        lr_b: float = 0.1,
        lr_c: float = 0.55,
        seed: int = 42,
        corpus: Optional[Corpus] = None,
        transform: Optional[Callable[dict, dict]] = None,
        # Train parameters to be set by agent
        iter=1000,
        freeze_topics=False,
        # Train Parameters to be set by human (default config)
        workers: int = 10,
        parallel: Union[int, ParallelScheme] = 1,
        callback_interval=10,
        callback=None,
        show_progress=False,
        # General parameters
        **kwargs: Any,
    ) -> None:
        """
        DT Parameters

        tw : Union[int, TermWeight]
            term weighting scheme in TermWeight. The default value is TermWeight.ONE

        min_cf : int
            minimum collection frequency of words. Words with a smaller collection frequency than min_cf are excluded from the model. The default value is 0, which means no words are excluded.

        min_df : int
            minimum document frequency of words. Words with a smaller document frequency than min_df are excluded from the model. The default value is 0, which means no words are excluded

        rm_top : int
            the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more. The default value is 0, which means no top words are removed.

        k : int
            the number of topics between 1 ~ 32767

        t : int
            the number of timpoints

        alpha_var : float
            transition variance of alpha (per-document topic distribution)

        eta_var : float
            variance of eta (topic distribution of each document) from its alpha

        phi_var : float
            transition variance of phi (word distribution of each topic)

        lr_a : float
            shape parameter a greater than zero, for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        lr_b : float
            shape parameter b greater than or equal to zero, for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        lr_c : float
            shape parameter c with range (0.5, 1], for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        seed : int
            random seed. default value is a random number from std::random_device{} in C++

        corpus : Corpus
            a list of documents to be added into the model

        transform : Callable[dict, dict]
            a callable object to manipulate arbitrary keyword arguments for a specific topic model

        Train Parameters

        iter : int
            the number of iterations of Gibbs-sampling

        workers : int
            an integer indicating the number of workers to perform samplings. If workers is 0, the number of cores in the system will be used.

        parallel : Union[int, ParallelScheme]
            the parallelism scheme for training. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.

        freeze_topics : bool
            prevents to create a new topic when training. Only valid for HLDAModel

        callback_interval : int
            the interval of calling callback function. If callback_interval <= 0, callback function is called at the beginning and the end of training.

        callback : Callable[[LDAModel, int, int], None]
            a callable object which is called every callback_interval iterations. It receives three arguments: the current model, the current number of iterations, and the total number of iterations.

        show_progress : bool
            If True, it shows progress bar during training using tqdm package.

        """
        self.tw=tw
        self.min_cf=min_cf
        self.min_df=min_df
        self.rm_top=rm_top
        self.k=k
        self.t=t
        self.alpha_var=alpha_var
        self.eta_var=eta_var
        self.phi_var=phi_var
        self.lr_a=lr_a
        self.lr_b=lr_b
        self.lr_c=lr_c
        self.seed=seed
        self.corpus= corpus
        self.transform = transform
      
        # Store the training parameters
        self.iter = iter
        self.freeze_topics = freeze_topics
        self.workers = workers
        self.parallel = parallel
        self.callback_interval = 10
        self.callback = callback
        self.show_progress = show_progress

        # Intitialize Model
        self.model = DTModel(
            tw=tw,
            min_cf=min_cf,
            min_df=min_df,
            rm_top=rm_top,
            k=k,
            t=t,
            alpha_var=0.1,
            eta_var=eta_var,
            phi_var=phi_var,
            lr_a=lr_a,
            lr_b=lr_b,
            lr_c=lr_c,
            seed=42,
            corpus=corpus,
            transform=transform,
        )

    def _parse_publication_date(self, date_str: str) -> Optional[datetime]:
        """
        Parses the PublicationDate string into a datetime object.
        """
        date_str = date_str[:8]
        for fmt in ("%Y-%b", "%Y"):
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        return None

    def _visualize(
        self, results: Dict[str, Union[Dict, str, List]]
    ) -> Dict[str, Union[Dict, str, List]]:
        """
        Creates Visualizations of the generated results including.

        Topic distributions over time.
        Normalized Topic distributions over time.
        Top Topic Words over time.
        """
        # Plotting the Topic word counts over time
        timepoints = results["Timepoints"]
        counts_per_topic = np.array(results["Counts Per Topic"])
        # Counts per topic is in form timepoints, topics
        # we need it in topics, timepoints
        counts_per_topic = counts_per_topic.T

        fig, ax = plt.subplots()

        ax.set_title("Topic Word Counts")
        ax.set_xticks(np.arange(len(timepoints)))
        ax.set_xticklabels(
            [
                " - ".join(
                    [
                        str(date.year) + " " + str(date.month)
                        for date in timepoint
                    ]
                )
                for timepoint in timepoints
            ],
            rotation="vertical",
        )
        ax.set_xlabel("Time Intervals")
        ax.set_ylabel("Words Per Topic")

        # Extracting the Topic Names, i.e. the most prominent words of a topic
        # over all timepoints. This will be given to subsequent LLM Agents to
        # Better analyze the results.
        topic_names = {}
        for k in range(self.model.k):
            topic_word_distribution = {}
            for t in range(self.t):
                topic_words = self.model.get_topic_words(
                    k, timepoint=t, top_n=10
                )
                for word, value in topic_words:
                    if word in topic_word_distribution.keys():
                        topic_word_distribution[word] += value
                    else:
                        topic_word_distribution[word] = value

            topic_word_distribution = {
                k: v
                for k, v in sorted(
                    topic_word_distribution.items(),
                    key=lambda item: item[1],
                    reverse=True,
                )
            }
            # Get top 10 names
            names = []
            for i, topic_name in enumerate(
                topic_word_distribution.keys(), start=1
            ):
                names.append(topic_name)
                if i % 10 == 0:
                    topic_names[k] = names
                    break

        # Adding the topic names to the results
        results[
            "Topic Names Explanation"
        ] = """
Top 10 words for each topic based on the unified distributions of every 
timepoint for every topic. 

Presented in this form: 

<topic_number>: ["<top word>", "<2nd top word>" ...]

"""
        results["Topic Names"] = topic_names
        # Plotting the Topic Wordcounts overtime
        for topic, timepoint_counts in enumerate(counts_per_topic):
            ax.plot(
                np.arange(len(timepoints)),
                timepoint_counts,
                label=f"Topic: {topic} {topic_names[topic][:2]}",
            )

        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join("visualizations", "DynamicTopicWordCounts.png")
        )

        results[
            "DynamicTopicWordCounts Explanation"
        ] = """
        A plot that shows the distribution of words per topic for each timepoint
        it is based on the data from Counts Per Topic. 

        In the plot each topic is named by a number and the first name from 
        "Topic Names" so use the number (n_topic starting from zero for the 
        first topic described in "Topic Names" and so on) or the name to
        discribe the plot. 
"""
        results["DynamicTopicWordCounts"] = "DynamicTopicWordCounts.png"

        # Dynamic Topic Distribution

        time_stamps_word_counts = [
            np.sum(timestamp_word_counts)
            for timestamp_word_counts in counts_per_topic.T
        ]

        topic_distributions = [
            list(
                np.array(topic_word_counts) / np.array(time_stamps_word_counts)
            )
            for topic_word_counts in counts_per_topic
        ]

        results[
            "Topic Distributions Explanation"
        ] = """
        A list of values derived from Topic Names transposed. 
        each timepoint is normalized by the sum of the wordcounts of all topics
        at this timepoint so it coresponds to the topic distribution at this 
        timepoint. 
"""
        results["Topic Distributions"] = topic_distributions

        fig, ax = plt.subplots()

        ax.set_title("Dynamic Topic Distribution")
        ax.set_xticks(np.arange(len(timepoints)))
        ax.set_xticklabels(
            [
                " - ".join(
                    [
                        str(date.year) + " " + str(date.month)
                        for date in timepoint
                    ]
                )
                for timepoint in timepoints
            ],
            rotation="vertical",
        )
        ax.set_xlabel("Time Intervals")
        ax.set_ylabel("Distribution of Topics")

        for topic, timepoint_counts in enumerate(topic_distributions):
            ax.plot(
                np.arange(len(timepoints)),
                timepoint_counts,
                label=f"Topic: {topic} {topic_names[topic][:2]}",
            )

        ax.legend()
        fig.tight_layout()
        fig.savefig(
            os.path.join("visualizations", "DynamicTopicDistributions.png")
        )

        results[
            "DynamicTopicDistributions Explanation"
        ] = """
        A plot that shows the distribution of topics for each timepoint.
        This plot is based on the data from "Topic Distributions"

        In the plot each topic is named by a number and the first two names 
        from "Topic Names" so use the number (n_topic starting from zero for
        the first topic described in "Topic Names" and so on) or the name to
        discribe the plot. 
"""
        results["DynamicTopicWordCounts"] = "DynamicTopicWordCounts.png"

        # Top Words over time

        top10_topic_names_overtime = {}
        for k in range(self.model.k):
            top10_topic_names_overtime[f"Topic {k}"] = []
            for t in range(self.t):
                topic_words = self.model.get_topic_words(
                    k, timepoint=t, top_n=10
                )
                topic_words = [
                    word
                    for word, value in sorted(
                        topic_words, key=lambda x: x[1], reverse=True
                    )
                ]
                top10_topic_names_overtime[f"Topic {k}"].append(
                    topic_words[:10]
                )

        topics = sorted(top10_topic_names_overtime.keys())
        num_timepoints = self.t

        cell_text = []
        for topic in topics:
            row = []
            for t in range(num_timepoints):
                top_words = top10_topic_names_overtime[topic][t][:1]
                words_str = ", ".join(top_words)
                row.append(words_str)
            cell_text.append(row)

        fig, ax = plt.subplots(figsize=(12, len(topics) * 0.5))

        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.axis("off")

        table = ax.table(
            cellText=cell_text,
            rowLabels=topics,
            colLabels=[f"Timepoint {t+1}" for t in range(num_timepoints)],
            loc="center",
            cellLoc="center",
            rowLoc="center",
        )

        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(
            1, 1.5
        )  # You may need to adjust scaling based on your data

        plt.tight_layout()
        fig.savefig(os.path.join("visualizations", "TopicWordsOverTime.png"))
        data = {}
        for t in range(num_timepoints):
            column_data = []
            for topic in topics:
                top_words = top10_topic_names_overtime[topic][
                    t
                ]  # Get all top words
                words_str = ", ".join(top_words)
                column_data.append(words_str)
            data[f"Timepoint {t+1}"] = column_data

        # Create the DataFrame
        df = pd.DataFrame(data, index=topics)

        # Save DataFrame to Excel
        df.to_excel(
            os.path.join("visualizations", "topic_words_over_time.xlsx")
        )

        results[
            "Topic Words Over Time Explanation"
        ] = """
        This is a table plot of the "Topic Words" data. it shows for each 
        timepoint the top word for each topic. 
"""
        results["Topic Words Over Time Explanation"] = (
            "topic_words_over_time.xlsx"
        )

        return results

    def __call__(self, documents: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Makes the DynamicTopicModeling instance callable.
        Performs dynamic topic modeling on the provided documents.

        :param documents: List of document dictionaries.
        :return: Dictionary containing model results.
        """
        # Debug
        print({
                "tw": self.tw,
                "min_cf": self.min_cf,
                "min_df": self.min_df,
                "rm_top": self.rm_top,
                "k": self.k,
                "t": self.t,
                "alpha_var": self.alpha_var,
                "eta_var": self.eta_var,
                "phi_var": self.phi_var,
                "lr_a": self.lr_a,
                "lr_b": self.lr_b,
                "lr_c": self.lr_c,
                "seed": self.seed,
                "corpus": self.corpus,
                "iter":self.iter,
                "freeze_topics":self.freeze_topics
            })
        ####

        # Always deepcopy documents at strart of algorithm call to pervent
        # documents from changing
        documents = copy.deepcopy(documents)
        # Converting the time-stamp strings into python time-stamps
        publication_dates = []
        documents_preprocessed = []
        for document in documents:
            publication_date_parsed = self._parse_publication_date(
                document["PublicationDate"]
            )
            if publication_date_parsed is not None:
                document["PublicationDate"] = publication_date_parsed
                documents_preprocessed.append(document)
                publication_dates.append(publication_date_parsed)

        # Generating date-ranges corresponding to timepoints

        min_date, max_date = (min(publication_dates), max(publication_dates))

        days_step = ((max_date - min_date).days + 1) / self.t
        time_step = timedelta(days=days_step)

        intervals = [
            [min_date + time_step * i, min_date + time_step * (i + 1)]
            for i in range(self.t)
        ]

        # Assigning timempoints to the documents
        for i, document in enumerate(documents_preprocessed):
            for timepoint, (low, high) in enumerate(intervals):
                # print(time_point, low, high, low < document["PublicationDate"] < high)
                if low <= document["PublicationDate"] < high:
                    document["Timepoint"] = timepoint

        for i, document in enumerate(documents_preprocessed):
            if "Timepoint" not in document.keys():
                print(i, document.keys())
            pass

        # Adding the documents to the model
        for document in documents_preprocessed:
            self.model.add_doc(
                words=document["AbstractNormalized"],
                timepoint=document["Timepoint"],
            )

        # Adding at least one document per timeslot to have no empty timepoints
        for timepoint in range(self.t):
            self.model.add_doc(words=[" "], timepoint=timepoint)

        # Running the algorithm
        self.model.train(
            iter=self.iter,
            workers=self.workers,
            parallel=self.parallel,
            freeze_topics=self.freeze_topics,
            # callback_interval=self.callback_interval,
            # callback=self.callback,
            show_progress=self.show_progress,
        )

        # Gathering the results
        results = {
            "Timepoints Explanation": """
The time intervals which were analyzed in this DynamicTopicModeling. In all 
results everytime a time point is mentioned these date-intervals are meant. 
""",
            "Timepoints": intervals,
            "Documents Analyzed": len(documents_preprocessed),
            "Documents discarded because publication date could not be parsed": len(
                documents
            )
            - len(documents_preprocessed),
            "Topic Words Explanation": """
Topic Word distribution for every topic and every timestamp.
It is presented in form of a dictionary with items:
    topic<k>_timestamp<t>: [(<word>, probability)...]
            """,
            "Topic Words": {},
            "Counts Per Topic Explanation": """
The number of words allocated to each timepoint and topic in the shape 
([num_timepoints, num_topics]...)
            """,
            "Counts Per Topic": self.model.get_count_by_topics(),
            "Hyperparameters Explanation": """DT Parameters

        tw : Union[int, TermWeight]
            term weighting scheme in TermWeight. The default value is TermWeight.ONE

        min_cf : int
            minimum collection frequency of words. Words with a smaller collection frequency than min_cf are excluded from the model. The default value is 0, which means no words are excluded.

        min_df : int
            minimum document frequency of words. Words with a smaller document frequency than min_df are excluded from the model. The default value is 0, which means no words are excluded

        rm_top : int
            the number of top words to be removed. If you want to remove too common words from model, you can set this value to 1 or more. The default value is 0, which means no top words are removed.

        k : int
            the number of topics between 1 ~ 32767

        t : int
            the number of timpoints

        alpha_var : float
            transition variance of alpha (per-document topic distribution)

        eta_var : float
            variance of eta (topic distribution of each document) from its alpha

        phi_var : float
            transition variance of phi (word distribution of each topic)

        lr_a : float
            shape parameter a greater than zero, for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        lr_b : float
            shape parameter b greater than or equal to zero, for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        lr_c : float
            shape parameter c with range (0.5, 1], for SGLD step size calculated as e_i = a * (b + i) ^ (-c)

        seed : int
            random seed. default value is a random number from std::random_device{} in C++

        corpus : Corpus
            a list of documents to be added into the model

        transform : Callable[dict, dict]
            a callable object to manipulate arbitrary keyword arguments for a specific topic model

        Train Parameters

        iter : int
            the number of iterations of Gibbs-sampling

        workers : int
            an integer indicating the number of workers to perform samplings. If workers is 0, the number of cores in the system will be used.

        parallel : Union[int, ParallelScheme]
            the parallelism scheme for training. the default value is ParallelScheme.DEFAULT which means that tomotopy selects the best scheme by model.

        freeze_topics : bool
            prevents to create a new topic when training. Only valid for HLDAModel

        callback_interval : int
            the interval of calling callback function. If callback_interval <= 0, callback function is called at the beginning and the end of training.

        callback : Callable[[LDAModel, int, int], None]
            a callable object which is called every callback_interval iterations. It receives three arguments: the current model, the current number of iterations, and the total number of iterations.

        show_progress : bool
            If True, it shows progress bar during training using tqdm package.""",
            "Hyperparameters": {
                "tw": self.tw,
                "min_cf": self.min_cf,
                "min_df": self.min_df,
                "rm_top": self.rm_top,
                "k": self.k,
                "t": self.t,
                "alpha_var": self.alpha_var,
                "eta_var": self.eta_var,
                "phi_var": self.phi_var,
                "lr_a": self.lr_a,
                "lr_b": self.lr_b,
                "lr_c": self.lr_c,
                "seed": self.seed,
                "corpus": self.corpus,
                "iter":self.iter,
                "freeze_topics":self.freeze_topics
            }
        }

        for k in range(self.model.k):
            for t in range(self.t):
                topic_words = self.model.get_topic_words(
                    k, timepoint=t, top_n=10
                )
                results["Topic Words"][
                    "topic{}_timepoint{}".format(k, t)
                ] = topic_words
        
        # Adding the visualizations
        results = self._visualize(results)

        return results


if __name__ == "__main__":
    pass

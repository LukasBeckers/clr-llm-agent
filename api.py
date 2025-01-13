from algorithms import algorithms
from step1.ReasoningResearchQuestionClassifier import (
    ReasoningResearchQuestionClassifier,
)
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from step3.prompts import algorithms_selector_prompt_v2
from step4.HyperParameterGuessor import HyperParameterGuessor
from step4.prompts import (
    hyperparamter_selection_prompts,
    multi_algorithm_prompt,
)
from step4.ResultsParser import ResultsParser
from step5.ResultsAnalyzer import ResultsAnalyzer
from step6.LaTeXPaperGenerator import LaTeXPaperGenerator
from step6.prompts import latex_paper_prompt

from agents.ReasoningResponseParser import ReasoningResponseParser
from agents.utils import json_to_dict

from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv
import os
import pickle as pk
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from typing import Generator
import time
import json
from queue import Queue
from uuid import uuid4
import uuid
from typing import List, Callable, Tuple
from pydantic import BaseModel
import asyncio
from matplotlib import pyplot as plt


load_dotenv()

email = os.getenv("EMAIL_ADDRESS")
algorithms_selector_prompt = algorithms_selector_prompt_v2
base_model = "gpt-4o"


class Message:
    def __init__(
        self,
        words,
        step: int,
        type: str,
        start_answer_token: str,
        stop_answer_token: str,
        callback: Callable = None,
        url: str = None,
    ):
        """
        step = int 0-5 and type = "reasoning" or "result" or "image"
        """
        self.id = uuid4()
        self.words = words
        self.step = step
        self.type = type
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token
        self.callback = callback
        self.url = url

    def to_json(self):
        return {
            "id": str(self.id),
            "step": self.step,
            "type": self.type,
            "start_answer_token": self.start_answer_token,
            "stop_answer_token": self.stop_answer_token,
            "url": self.url,
        }


class Step_1:
    def __init__(self, llm: str = "gpt-4o"):
        self.step_id = 0
        self.current_substep = 0
        self.research_question_class = None
        self.finished = False
        self.research_question = ""
        self.llm = llm

    def __call__(self, message: str) -> Message:
        """
        Makes the class instance callable to route the message to the correct sub-step
        based on the current state of the step.

        Parameters:
        - message (str): The incoming message to process.
        """
        if self.current_substep == 0:
            # Begin with the research_question; this is special for step 1
            research_question = message
            self.research_question = research_question

            return self.substep_0(research_question)

        elif self.current_substep == 1:
            # In this step, there is only one substep. If substep is one, it means that
            # a message was committed to the step even though it is finished.

            if message == "":
                # Empty message is equal to a confirmation; nothing happens
                response = Message(
                    words=[
                        word + " "
                        for word in "This step is already completed, provide critique of the last substep or continue with the next step!".split(
                            " "
                        )
                    ],
                    step=0,
                    type="reasoning",
                    start_answer_token=self.start_answer_token,
                    stop_answer_token=self.stop_answer_token,
                )

                return [response]

            else:
                # Non-empty message is meant as critique, so the step is repeated.
                self.finished = False
                critique = (
                    f"Previous Answer to this task: {self.research_question_class}. \n"
                    f"User-Critique: {message}"
                )
                return self.substep_0(
                    self.research_question, critique=critique
                )

    def substep_0(
        self,
        research_question: str,
        critique: str = "First try so no critique",
    ) -> Tuple[List[Message], callable]:
        """
        Handles the first substep by classifying the research question and parsing the response.

        Parameters:
        - research_question (str): The research question to classify.
        - critique (str): Optional critique to refine the classification.
        """
        research_question_classifier = ReasoningResearchQuestionClassifier(
            self.llm
        )
        self.start_answer_token = (
            research_question_classifier.start_answer_token
        )
        self.stop_answer_token = research_question_classifier.stop_answer_token

        # Classifying the research question
        response = research_question_classifier(
            research_question, critique=critique
        )

        message = Message(
            words=response,
            step=0,
            type="reasoning",
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
            callback=self._finish_answer,
        )

        return [message]

    def _finish_answer(self, full_output: List[str]) -> List[Message]:
        print("Begin Finish Answer")
        answer_parser = ReasoningResponseParser(
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )

        for token in full_output:

            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break

        self.research_question_class = answer_parser.answer

        message = Message(
            words=[
                word + " " for word in self.research_question_class.split(" ")
            ],
            step=0,
            type="result",
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )

        print("Created Answere in Finish Answere", message.to_json())

        self.current_substep = 1
        self.finished = True

        return [message]


class Step_2:
    def __init__(self, llm: str = "gpt-4o"):
        """
        Initializes Step_2 with the specified language model.

        Args:
            llm (str, optional): The base model to use. Defaults to "gpt-4o".
        """
        self.step_id = 1
        self.current_substep = 0
        self.research_question = ""
        self.research_question_class = ""
        self.llm = llm
        self.critique = "First attempt so no critique"
        self.search_strings = []
        self.finished = False
        self.dataset = {}

    def __call__(
        self,
        research_question: str,
        research_question_class: str,
        message: str,
    ) -> Message:
        """
        Makes the class instance callable to route the message to the correct sub-step
        based on the current state of the step.

        Args:
            research_question (str): The research question to generate search queries for.
            classification_result (str): The classification result of the research question.
            message (str): The incoming message to process, typically a critique or confirmation.
        """
        print("In Call of step 2", message)
        # Update the internal state with the latest research question and classification result
        self.research_question = research_question
        self.research_question_class = research_question_class

        # Update critique based on the incoming message
        if message.strip() != "":
            self.critique = f"Previous Answer to this task: {self.search_strings}. \nUser-Critique: {message}"
        else:
            self.critique = "No Critique"

        # Route the message based on the current substep
        if self.current_substep == 0:
            # Begin the search string generation substep
            return self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Handle user critique after generating search strings in the last step
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.finished = True
                response = Message(
                    words=[
                        word + " "
                        for word in "This step is already completed, provide critique of the last substep or continue with the next step!".split(
                            " "
                        )
                    ],
                    step=1,
                    type="reasoning",
                    start_answer_token=self.start_answer_token,
                    stop_answer_token=self.stop_answer_token,
                )

                return [response]
            else:
                # Non-empty message is treated as a critique; regenerate search strings
                self.finished = False
                return self.substep_0(self.critique)
        else:
            raise ValueError(f"Invalid substep state: {self.current_substep}")

    def substep_0(
        self, critique: str = "First attempt so no critique"
    ) -> Message:
        """
        Generates search strings based on the research question and classification result.

        Args:
            critique (str, optional): User critique to refine the search string generation.
                                      Defaults to "First attempt so no critique".
        """
        # Initialize the search query generator
        search_query_generator = ReasoningSearchQueryGenerator(
            self.llm, temperature=0
        )

        # Generate search strings
        response = search_query_generator(
            research_question=self.research_question,
            classification_result=self.research_question_class,
            critic=critique,
        )

        self.start_answer_token = search_query_generator.start_answer_token
        self.stop_answer_token = search_query_generator.stop_answer_token

        message = Message(
            words=response,
            step=1,
            type="reasoning",
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
            callback=self._finish_answer,
        )

        return [message]

    def _finish_answer(self, full_output: List[str]) -> Message:
        # Initialize the response parser
        answer_parser = ReasoningResponseParser(
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )

        # Parse the streaming response
        for token in full_output:
            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break
            print(token, end="")  # Optional: For debugging or logging purposes

        raw_search_strings = answer_parser.answer

        try:
            search_strings = []

            for search_string_and_source in raw_search_strings.split("),"):
                parts = search_string_and_source.strip().rstrip(")").split(",")
                if len(parts) != 2:
                    continue  # Skip malformed entries
                search_string, data_source = parts
                search_string = search_string.strip().strip("['\"]").strip()
                data_source = data_source.strip().strip("'\"").strip()
                search_strings.append((search_string, data_source))

            self.search_strings = search_strings
        except Exception as e:
            print(f"\nError parsing search strings: {e}")
            # Optionally, you can set a flag or raise an exception
            return

        message = Message(
            words=[str(wordpair) + " " for wordpair in self.search_strings],
            step=1,
            type="result",
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )

        self.current_substep = 1
        self.finished = True

        # Downloading the datasets from the datasources
        data_loader = DataLoader(email=email)
        # data_set = data_loader(search_strings=search_strings[:])
        # data_set = data_loader(search_strings=[("Glymph* OR Brain_Clearance", "pub_med")])

        # with open(os.path.join("temp", "dataset"), "wb") as f:
        #     pk.dump(data_set, f)

        with open(os.path.join("temp", "dataset"), "rb") as f:
            dataset = pk.load(f)

        self.dataset = dataset

        return [message]


class Step_3:
    def __init__(self, llm: str = "gpt-4o"):
        """
        Initializes Step_3 with only the LLM parameter.

        Args:
            llm (str, optional): The base model to use. Defaults to "gpt-4o".
        """
        self.step_id = 2
        self.current_substep = 0
        self.dataset = []
        self.research_question = ""
        self.classification_result = ""
        self.basic_dataset_evaluation = ""
        self.basic_dataset_description = ""
        self.llm = llm
        self.critique = "First attempt so no critique"
        self.selected_algorithms = []
        self.finished = False

    def __call__(
        self,
        dataset: list = None,
        research_question: str = "",
        research_question_class: str = "",
        message: str = "",
    ):
        """
        Routes the incoming message to the appropriate substep based on the current state.

        This method also initializes the necessary data if provided.

        Args:
            dataset (list, optional): The dataset to be analyzed.
            research_question (str, optional): The research question.
            classification_result (str, optional): The classification result from Step_1.
            basic_dataset_evaluation (str, optional): Evaluation of the dataset from BasicDatasetAnalyzer.
            basic_dataset_description (str, optional): Description of the dataset from BasicDatasetAnalyzer.
            message (str, optional): The user input message, typically a critique or confirmation.
        """
        # Initialize data if provided
        if dataset is not None:
            self.dataset = dataset
        if research_question:
            self.research_question = research_question
        if research_question_class:
            self.classification_result = research_question_class

        # Update critique based on the message
        if message.strip() != "":
            self.critique = f"Previous Answer to this task: {self.selected_algorithms}. \nUser-Critique: {message}"
        else:
            self.critique = "No Critique"

        # Route to the appropriate substep
        if self.current_substep == 0:
            # Begin normalization and basic dataset analysis
            return self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Begin algorithm selection
            return self.substep_1(self.critique)
        elif self.current_substep == 2:
            # Handle user critique after algorithm selection
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.finished = True
                print("Step 3 is finished.")
                response = Message(
                    words=[
                        word + " "
                        for word in "This step is already completed, provide critique of the last substep or continue with the next step!".split(
                            " "
                        )
                    ],
                    step=2,
                    type="reasoning",
                    start_answer_token=self.start_answer_token,
                    stop_answer_token=self.stop_answer_token,
                )

                return [response]

            else:
                # Non-empty message is treated as a critique; redo algorithm selection
                self.finished = False
                return self.substep_1(self.critique)
        else:
            raise ValueError(f"Invalid substep state: {self.current_substep}")

    def plot_publications_over_time(self, data):
        # Prepare data
        years = list(data.keys())
        counts = list(data.values())

        # Create a figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(years, counts)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Publications")
        ax.set_title("Publications Over Time")

        publications_over_time_path = os.path.abspath(
            os.path.join(
                "gui", "public", "visualizations", "PublicationsOverTime.png"
            )
        )

        fig.savefig(publications_over_time_path)

        return os.path.join("visualizations", "PublicationsOverTime.png")

    def substep_0(self, critique: str = "First attempt so no critique"):
        """
        Normalizes the text, removes data points without abstracts, and performs basic dataset analysis.

        Args:
            critique (str, optional): User critique to refine the dataset analysis. Defaults to "First attempt so no critique".
        """
        print(
            "Substep 3.0: Normalizing text and performing basic dataset analysis"
        )

        # Normalize the Text from the Dataset
        text_normalizer = TextNormalizer()

        # Create a copy of the dataset to avoid modifying the original during iteration
        normalized_dataset = []
        for data_point in self.dataset[:]:
            try:
                data_point["AbstractNormalized"] = text_normalizer(
                    data_point["Abstract"]
                )
                normalized_dataset.append(data_point)
            except KeyError:
                # Removing datapoints that have no abstract
                continue

        self.dataset = normalized_dataset

        # Perform basic analysis of the dataset
        basic_dataset_analyzer = BasicDatasetAnalyzer(llm=self.llm)

        evaluation, description = basic_dataset_analyzer(self.dataset)

        self.basic_dataset_evaluation = evaluation
        self.basic_dataset_description = description

        print("Dataset Evaluation:", self.basic_dataset_evaluation)
        print("Dataset Description:", self.basic_dataset_description)

        plot_url = self.plot_publications_over_time(
            self.basic_dataset_evaluation["Publications Over Time"]
        )

        messages = [
            Message(
                words=[
                    word + " "
                    for word in str(self.basic_dataset_evaluation).split(" ")
                ],
                step=2,
                type="result",
                start_answer_token="",
                stop_answer_token="",
            ),
            Message(
                words=[
                    word + " "
                    for word in str(self.basic_dataset_description).split(" ")
                ],
                step=2,
                type="result",
                start_answer_token="",
                stop_answer_token="",
            ),
            Message(
                words=["Publications over Time"],
                step=2,
                type="image",
                start_answer_token="",
                stop_answer_token="",
                url=plot_url,
            ),
        ]

        self.current_substep = 1  # Move to the next substep

        return messages

    def substep_1(self, critique: str = "First attempt so no critique"):
        """
        Selects algorithms based on the research question, classification, and dataset evaluation.

        Args:
            critique (str, optional): User critique to refine the algorithm selection. Defaults to "First attempt so no critique".
        """
        print(
            "Substep 3.1: Selecting algorithms based on research question and dataset evaluation"
        )

        # Select the algorithms based on the research question, the classification
        # of the research question and the basic dataset evaluation
        algorithm_selector = AlgorithmsSelector(
            prompt_explanation=algorithms_selector_prompt, llm=self.llm
        )

        self.start_answer_token = algorithm_selector.start_answer_token
        self.stop_answer_token = algorithm_selector.stop_answer_token

        response = algorithm_selector(
            research_question=self.research_question,
            research_question_class=self.classification_result,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
            critic=critique,
        )

        messages = [
            Message(
                words=response,
                step=2,
                type="reasoning",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
                callback=self.finish_answer_substep1,
            )
        ]

        return messages

    def finish_answer_substep1(
        self, full_response: List[str]
    ) -> List[Message]:
        answer_parser = ReasoningResponseParser(
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )
        # Parse the streaming response
        for token in full_response:
            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break

        reasoning = answer_parser.reasoning
        algorithms_raw = answer_parser.answer

        # Process the algorithms
        try:
            selected_algorithms = []
            for algorithm in algorithms_raw.split(","):
                algorithm = algorithm.strip(', "()[]"`\n\t')
                algorithm = algorithm.strip("'")
                if algorithm:  # Ensure it's not empty
                    selected_algorithms.append(algorithm)
            self.selected_algorithms = selected_algorithms
            print("\nSelected Algorithms:", self.selected_algorithms)
        except Exception as e:
            self.selected_algorithms = []
            print(f"\nError parsing selected algorithms: {e}")
            # Optionally, handle the error or retry

        self.current_substep = 2  # Move to the critique handling substep
        self.finished = True

        messages = [
            Message(
                words=[
                    word for word in str(self.selected_algorithms).split(" ")
                ],
                step=2,
                type="result",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
            )
        ]

        return messages


class Step_4:
    def __init__(self, llm: str = "gpt-4o"):
        """
        Initializes Step_4 with the language model.

        Args:
            llm (str, optional): The language model to use. Defaults to "gpt-4o".
        """
        self.step_id = 3
        self.current_substep = 0
        self.llm = llm
        self.finished = False

        # Attributes to be set during the first call
        self.selected_algorithms = []
        self.research_question = ""
        self.classification_result = ""
        self.basic_dataset_evaluation = ""
        self.dataset = []

        # Internal state
        self.critique = "First Attempt no critique yet."
        self.hyperparameters_dict = {}
        self.calibrated_algorithms = {}
        self.results = {}
        self.parsed_results = {}

    def __call__(
        self,
        message: str,
        selected_algorithms: list = None,
        research_question: str = None,
        research_question_class: str = None,
        basic_dataset_evaluation: str = None,
        dataset: list = None,
    ):
        """
        Routes the incoming message to the appropriate substep based on the current state.

        Parameters:
            message (str): The user input message, typically a critique or confirmation.
            selected_algorithms (list, optional): List of algorithms selected in Step 3.
            research_question (str, optional): The main research question.
            classification_result (str, optional): The classification result from Step 1.
            basic_dataset_evaluation (str, optional): Evaluation of the dataset from Step 3.
            dataset (list, optional): The dataset to perform analysis on.
        """
        # Initialize attributes on the first call
        if self.current_substep == 0 and any(
            param is not None
            for param in [
                selected_algorithms,
                research_question,
                research_question_class,
                basic_dataset_evaluation,
                dataset,
            ]
        ):
            if selected_algorithms is not None:
                self.selected_algorithms = selected_algorithms
            if research_question is not None:
                self.research_question = research_question
            if research_question_class is not None:
                self.classification_result = research_question_class
            if basic_dataset_evaluation is not None:
                self.basic_dataset_evaluation = basic_dataset_evaluation
            if dataset is not None:
                self.dataset = dataset

        # Update critique based on the message
        if message.strip() != "":
            if self.current_substep == 1:
                self.critique = f"Previous Answer to this task: {self.hyperparameters_dict}. \nUser-Critique: {message}"
            elif self.current_substep == 2:
                self.critique = f"Previous Results: {self.results}. \nUser-Critique: {message}"
            elif self.current_substep == 3:
                self.critique = f"Previous Parsed Results: {self.parsed_results}. \nUser-Critique: {message}"
            else:
                self.critique = f"Previous Answer to this task: {getattr(self, 'parsed_results', 'N/A')}. \nUser-Critique: {message}"
        else:
            self.critique = "No Critique"

        # Route to the appropriate substep
        if self.current_substep == 0:
            # Begin hyperparameter selection and algorithm calibration
            return self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Handle user critique after hyperparameter selection
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.current_substep = 2
                return self.substep_1()  # Proceed to run algorithms
            else:
                # Non-empty message is treated as a critique; redo hyperparameter selection
                return self.substep_0(self.critique)
        elif self.current_substep == 2:
            # Handle user critique after running algorithms
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.current_substep = 3
                return self.substep_2()  # Proceed to parse results
            else:
                # Non-empty message is treated as a critique; rerun algorithms or adjust hyperparameters
                return self.substep_1()
        elif self.current_substep == 3:
            # Handle user critique after parsing results
            if message.strip() == "":
                # Empty message signifies confirmation to finish Step 4
                self.finished = True
                response = Message(
                    words=[
                        word + " "
                        for word in "This step is already completed, provide critique of the last substep or continue with the next step!".split(
                            " "
                        )
                    ],
                    step=3,
                    type="reasoning",
                    start_answer_token=self.start_answer_token,
                    stop_answer_token=self.stop_answer_token,
                )

                return [response]

            else:
                # Non-empty message is treated as a critique; rerun analysis or adjust previous steps
                self.finished = False
                return self.substep_2()
        else:
            raise ValueError(f"Invalid substep state: {self.current_substep}")

    def substep_0(self, critique: str = "First attempt so no critique"):
        """
        Generates hyperparameters for the selected algorithms and calibrates them.

        Args:
            critique (str, optional): User critique to refine hyperparameter selection. Defaults to "First attempt so no critique".
        """
        print(
            "Substep 4.0: Generating hyperparameters and calibrating algorithms"
        )

        # Construct the hyperparameter selection prompt
        hyper_parameter_guessor_prompt = ""
        for algorithm_name in self.selected_algorithms:
            try:
                hyper_parameter_guessor_prompt += (
                    hyperparamter_selection_prompts[algorithm_name]
                )
            except KeyError:
                print(
                    f"Warning: No hyperparameter prompt found for {algorithm_name}. Skipping."
                )
                continue

        hyper_parameter_guessor_prompt += multi_algorithm_prompt

        # Initialize the HyperParameterGuessor
        hyper_parameter_guessor = HyperParameterGuessor(
            prompt_explanation=hyper_parameter_guessor_prompt,
            llm=self.llm,
        )

        self.start_answer_token = hyper_parameter_guessor.start_answer_token
        self.stop_answer_token = hyper_parameter_guessor.stop_answer_token

        # Generate hyperparameters
        response = hyper_parameter_guessor(
            research_question=self.research_question,
            research_question_class=self.classification_result,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
            critic=critique,
        )

        messages = [
            Message(
                words=response,
                step=3,
                type="reasoning",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
                callback=self._finalize_answer,
            )
        ]

        return messages

    def _finalize_answer(self, full_output):
        # Initialize the response parser
        answer_parser = ReasoningResponseParser(
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )

        # Parse the streaming response
        for token in full_output:
            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break
            print(token, end="")  # Optional: For debugging or logging purposes

        reasoning = answer_parser.reasoning
        hyperparameters_raw = answer_parser.answer

        # Convert the JSON output to a dictionary
        self.hyperparameters_dict = json_to_dict(hyperparameters_raw)

        if self.hyperparameters_dict is None:
            print("Error: Failed to parse hyperparameters JSON.")
            return

        print("\nParsed Hyperparameters:", self.hyperparameters_dict)

        # Calibrate the algorithms with the parsed hyperparameters
        try:
            self.calibrated_algorithms = {
                algorithm_name: algorithms[algorithm_name](**guessing_results)
                for algorithm_name, guessing_results in self.hyperparameters_dict.items()
            }
        except Exception as e:
            print(f"Error calibrating algorithms: {e}")
            self.calibrated_algorithms = {}

        self.current_substep = 1  # Move to the next substep

        messages = [
            Message(
                words=[json.dumps(self.hyperparameters_dict)],
                step=3,
                type="result",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
            )
        ]

        return messages

    def substep_1(self):
        """
        Runs the calibrated algorithms on the dataset and collects results.
        """
        print("\nSubstep 4.1: Running calibrated algorithms on the dataset")

        self.results = {}

        for algorithm_name, algorithm in self.calibrated_algorithms.items():

            if algorithm_name not in [
                "LatentDirichletAllocation",
                "DynamicTopicModeling",
            ]:
                continue

            try:
                self.results[algorithm_name] = algorithm(self.dataset)
                print(f"Algorithm '{algorithm_name}' executed successfully.")
            except Exception as e:
                self.results[algorithm_name] = str(e)
                print(f"Error running algorithm '{algorithm_name}': {e}")

        import pickle as pk

        with open(os.path.join("temp", "results.pk"), "wb") as f:
            pk.dump(self.results, f)

        print("\nCollected Results:", self.results)
        self.current_substep = 2  # Move to the next substep

        messages = []

        print(
            "RSULTS DYNAMIC TOPIC MODELING",
            self.results["DynamicTopicModeling"],
        )

        for algorithm, results in self.results.items():
            if type(results) == str:
                messages.append(
                    Message(
                        words=[f"{algorithm}: ", results],
                        step=3,
                        type="result",
                        start_answer_token=self.start_answer_token,
                        stop_answer_token=self.stop_answer_token,
                    )
                )
            else:
                for key, value in results.items():
                    if type(value) == str:

                        messages.append(
                            Message(
                                words=[f"{algorithm}: ", key, value],
                                step=3,
                                type="result",
                                start_answer_token=self.start_answer_token,
                                stop_answer_token=self.stop_answer_token,
                            )
                        )
                        path_split = value.split("\\")
                        print("Path Split", path_split)
                        if value.endswith(".png"):
                            messages.append(
                                Message(
                                    words=[f"{algorithm} {key}"],
                                    step=3,
                                    type="image",
                                    start_answer_token="",
                                    stop_answer_token="",
                                    url=os.path.join(
                                        path_split[-2], path_split[-1]
                                    ),
                                ),
                            )
                    else:
                        messages.append(
                            Message(
                                words=[f"{algorithm}: ", key, str(value)],
                                step=3,
                                type="result",
                                start_answer_token=self.start_answer_token,
                                stop_answer_token=self.stop_answer_token,
                            )
                        )

        print("Returning Messages in Step 4 substep_1", messages)
        return messages

    def substep_2(self):
        """
        Parses the raw results obtained from the algorithms.
        """
        print("\nSubstep 4.2: Parsing the results")

        # Initialize the ResultsParser
        results_parser = ResultsParser()

        # Parse the results
        self.parsed_results = results_parser(results=self.results)

        print("\nParsed Results:", self.parsed_results)
        self.current_substep = 3  # Move to the critique handling substep
        self.finished = True

        messages = [
            Message(
                words=[self.parsed_results],
                step=3,
                type="result",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
            )
        ]

        return messages


class Step_5:
    def __init__(
        self,
        llm: str = "gpt-4o",
    ):
        """
        Initializes Step_5 with necessary parameters from previous steps.

        Args:

            llm (str, optional): The language model to use. Defaults to "gpt-4o".
        """
        self.step_id = 4
        self.current_substep = 0
        self.research_question = ""
        self.classification_result = ""
        self.parsed_algorithm_results = ""
        self.search_strings = ""
        self.basic_dataset_evaluation = ""
        self.llm = llm
        self.critique = "First attempt so no critique"
        self.analysis_result = ""
        self.finished = False

    def __call__(
        self,
        message: str,
        research_question: str,
        classification_result: str,
        parsed_algorithm_results: dict,
        search_strings: list,
        basic_dataset_evaluation: str,
        hyperparameters: str,
    ):
        """
        Routes the incoming message to the appropriate substep based on the current state.

        Args:
            message (str): The user input message, typically a critique or confirmation.
        """
        self.research_question = research_question
        self.classification_result = classification_result
        self.parsed_algorithm_results = parsed_algorithm_results
        self.search_strings = search_strings
        self.basic_dataset_evaluation = basic_dataset_evaluation
        self.hyperparameters = hyperparameters

        if message != "":
            self.critique = f"Previous Answer to this task: {self.analysis_result}. \nUser-Critique: {message}"
        else:
            self.critique = "No Critique"

        if self.current_substep == 0:
            # Begin analysis of the results
            return self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Handle user critique after analysis
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.finished = True
                response = Message(
                    words=[
                        word + " "
                        for word in "This step is already completed, provide critique of the last substep or continue with the next step!".split(
                            " "
                        )
                    ],
                    step=2,
                    type="reasoning",
                    start_answer_token=self.start_answer_token,
                    stop_answer_token=self.stop_answer_token,
                )
                return response
            else:
                # Non-empty message is treated as a critique; redo analysis
                self.finished = False
                return self.substep_0(self.critique)
        else:
            raise ValueError(f"Invalid substep state: {self.current_substep}")

    def substep_0(self, critique: str = "First attempt so no critique"):
        """
        Analyzes the parsed algorithm results using the ResultsAnalyzer.

        Args:
            critique (str, optional): User critique to refine the analysis. Defaults to "First attempt so no critique".
        """
        print("Substep 5.0: Analyzing the results")

        # Initialize the ResultsAnalyzer
        results_analyzer = ResultsAnalyzer(llm=self.llm)

        self.start_answer_token = results_analyzer.start_answer_token
        self.stop_answer_token = results_analyzer.stop_answer_token

        # Generate the analysis
        response = results_analyzer(
            research_question=self.research_question,
            research_question_class=self.classification_result,
            parsed_algorithm_results=self.parsed_algorithm_results,
            search_strings=self.search_strings,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
            critique=critique,
            hyperparameters=self.hyperparameters,
        )

        messages = [
            Message(
                words=response,
                step=4,
                type="reasoning",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
                callback=self._finish_answer,
            )
        ]

        return messages

    def _finish_answer(self, full_output: List[str]) -> List[Message]:
        # Initialize the response parser
        answer_parser = ReasoningResponseParser(
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )
        # Parse the streaming response
        for token in full_output:
            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break

        self.analysis_result = answer_parser.full_output.strip()
        print("\n\nAnalysis Result:", self.analysis_result)

        self.current_substep = 1  # Move to the next substep
        self.finished = True

        messages = [
            Message(
                words=[self.analysis_result],
                step=4,
                type="result",
                start_answer_token=self.start_answer_token,
                stop_answer_token=self.stop_answer_token,
            )
        ]

        return messages


class Step_6:
    def __init__(self, llm: str):
        self.step_id = 5
        self.current_substep = 0
        self.llm = llm
        self.finished = False

    def __call__(self, message: str, analysis_result: str):
        print("In Step 6!")
        pdf_generator = LaTeXPaperGenerator(llm=self.llm)
        pdf_generator(analysis_results=analysis_result)
        self.finished = True

        messages = [
            Message(
                words=["PDF was created"],
                step=5,
                type="result",
                start_answer_token="<START_LATEX>",
                stop_answer_token="<STOP_LATEX>",
            )
        ]
        print("Created Message is Step 6")

        return messages


class API:
    def __init__(self, llm: str = base_model):
        self.current_step = 0
        self.llm = llm
        self.steps = {
            0: Step_1(llm),
            1: Step_2(llm),
            2: Step_3(llm),
            3: Step_4(llm),
            4: Step_5("o1-mini"),
            5: Step_6("o1-mini"),
        }
        self.messages = Queue()
        self.current_message = None

    def get_allowed_steps(self):
        allowed_steps = [False for _ in range(len(self.steps))]

        for i, step in self.steps.items():
            # first step is always allowed
            if i == 0:
                allowed_steps[i] = True

            if i == len(allowed_steps) - 1:
                break  # last step

            if step.finished:
                allowed_steps[i + 1] = True
        return allowed_steps

    async def get_step_states(self):
        step_states = {
            i: {"isallowed": allowed}
            for i, allowed in enumerate(self.get_allowed_steps())
        }

        for i, step in self.steps.items():
            step_states[i]["finished"] = step.finished

        return step_states

    async def upload_user_message(self, user_message: str, step: int):

        # If an older step is rerun, clear newersteps

        for i, self_step in self.steps.items():
            if i > step:
                self_step.finished = False
                self_step.current_substep = 0

        print("User Message", user_message)
        if user_message == "Go on, do your thing!":
            user_message = ""

        print("User Message", user_message)
        # checking if the step is allowed
        if not self.get_allowed_steps()[step]:
            return self.get_step_states()
        else:
            if step == 0:
                response_messages = self.steps[0](user_message)
                for response_message in response_messages:
                    self.messages.put(response_message)

            if step == 1:
                response_messages = self.steps[1](
                    self.steps[0].research_question,
                    self.steps[0].research_question_class,
                    user_message,
                )
                for response_message in response_messages:
                    self.messages.put(response_message)

            if step == 2:
                response_messages = self.steps[2](
                    dataset=self.steps[1].dataset,
                    research_question=self.steps[0].research_question,
                    research_question_class=self.steps[
                        0
                    ].research_question_class,
                    message=user_message,
                )
                for response_message in response_messages:
                    self.messages.put(response_message)

            if step == 3:
                response_messages = self.steps[3](
                    message=user_message,
                    selected_algorithms=self.steps[2].selected_algorithms,
                    dataset=self.steps[1].dataset,
                    research_question=self.steps[0].research_question,
                    research_question_class=self.steps[
                        0
                    ].research_question_class,
                    basic_dataset_evaluation=self.steps[
                        2
                    ].basic_dataset_evaluation,
                )

                for response_message in response_messages:
                    self.messages.put(response_message)

            if step == 4:
                response_messages = self.steps[4](
                    message=user_message,
                    research_question=self.steps[0].research_question,
                    classification_result=self.steps[
                        0
                    ].research_question_class,
                    parsed_algorithm_results=self.steps[3].parsed_results,
                    search_strings=self.steps[1].search_strings,
                    basic_dataset_evaluation=self.steps[
                        2
                    ].basic_dataset_evaluation,
                    hyperparameters=self.steps[3].hyperparameters_dict,
                )

                for response_message in response_messages:
                    self.messages.put(response_message)

            if step == 5:
                response_messages = self.steps[5](
                    message=user_message,
                    analysis_result=self.steps[4].analysis_result,
                )
                for response_message in response_messages:
                    self.messages.put(response_message)

    async def get_current_message(self):
        if self.current_message is None:
            try:
                self.current_message = self.messages.get(timeout=0.01)

            except Exception as e:

                self.current_message = None

        if self.current_message is not None:
            message_json = self.current_message.to_json()
            # special handeling of image messages
            if message_json["type"] == "image":
                self.current_message = None
            return message_json

    async def stream_current_message(self):

        if self.current_message is None:
            return

        response_parser = ReasoningResponseParser(
            start_answer_token=self.current_message.start_answer_token,
            stop_answer_token=self.current_message.stop_answer_token,
        )

        full_message = ""
        all_tokens = []

        for token in self.current_message.words:
            try:
                try:
                    token = token.choices[0].delta.content
                except Exception:
                    pass
                if token is None:
                    continue

                all_tokens.append(token)
                # print(token, end="")
                full_message += token
                if self.current_message.type == "reasoning":
                    # the full resoponse from open ai contains the whole message
                    # here we only yield the reasoning tokens
                    token_type = response_parser(token=token)
                    if token_type == "reasoning":
                        print("Yielding Token: ", token)
                        yield token

                # Result Messages are already filtered so we yield all tokens.
                if self.current_message.type == "result":
                    yield token
            except Exception as e:
                print("Exception: ", e)
                continue

        if self.current_message.callback is not None:
            print("Found Callback")
            answer_messages = self.current_message.callback(all_tokens)

            for message in answer_messages:
                print("Answere Message Text", message.words)
                print("Answer message type", message.type)
                self.messages.put(message)

        self.current_message = None  # Reset the current message


class UserMessage(BaseModel):
    text: str
    step: int


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # or ["*"] for testing
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

api = API(llm=base_model)


@app.get("/status")
async def status():
    return await api.get_step_states()


@app.post("/user_message")
async def user_message(data: UserMessage):
    print("Data of usermessage", data)
    await api.upload_user_message(user_message=data.text, step=data.step)


@app.get("/current_message")
async def current_message():
    result = await api.get_current_message()

    print("Result of current_message api call", result)
    return result


@app.get("/stream_current_message")
async def stream_current_message():
    # Suppose api.stream_current_message() is an async generator that yields tokens
    async def sse_generator():
        async for token in api.stream_current_message():
            # SSE format
            yield token
            # "Nudge" flush 
            await asyncio.sleep(0)

    return StreamingResponse(
        sse_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no"
        }
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

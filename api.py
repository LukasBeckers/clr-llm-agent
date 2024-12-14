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
    ):
        """
        step = int 0-5 and type = "reasoning" or "result"
        """
        self.id = uuid4()
        self.words = words
        self.step = step
        self.type = type
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token
        self.callback = callback

    def to_json(self):
        return json.dumps(
            {
                "id": str(self.id),
                "step": self.step,
                "type": self.type,
                "start_answer_token": self.start_answer_token,
                "stop_answer_token": self.stop_answer_token,
            }
        )


class Step_1:
    def __init__(self, llm: str = "gpt-4o"):
        self.step_id = 0
        self.current_substep = 0
        self.research_question_class = None
        self.finished = False
        self.research_question = ""
        self.llm = llm

    def __call__(self, message: str):
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
                return
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
            words=self.research_question_class,
            step=0,
            type="result",
            start_answer_token=self.start_answer_token,
            stop_answer_token=self.stop_answer_token,
        )

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

    def __call__(
        self,
        research_question: str,
        research_question_class: str,
        message: str,
    ):
        """
        Makes the class instance callable to route the message to the correct sub-step
        based on the current state of the step.

        Args:
            research_question (str): The research question to generate search queries for.
            classification_result (str): The classification result of the research question.
            message (str): The incoming message to process, typically a critique or confirmation.
        """
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
                return
            else:
                # Non-empty message is treated as a critique; regenerate search strings
                self.finished = False
                return self.substep_0(self.critique)
        else:
            raise ValueError(f"Invalid substep state: {self.current_substep}")

    def substep_0(self, critique: str = "First attempt so no critique"):
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

        # Initialize the response parser
        answer_parser = ReasoningResponseParser(
            start_answer_token=search_query_generator.start_answer_token,
            stop_answer_token=search_query_generator.stop_answer_token,
        )

        # Generate search strings
        response = search_query_generator(
            research_question=self.research_question,
            classification_result=self.research_question_class,
            critic=critique,
        )

        # Parse the streaming response
        for chunk in response:
            token = chunk.choices[0].delta.content
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

        self.current_substep = 1
        self.finished = True


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
        classification_result: str = "",
        basic_dataset_evaluation: str = "",
        basic_dataset_description: str = "",
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
        if classification_result:
            self.classification_result = classification_result
        if basic_dataset_evaluation:
            self.basic_dataset_evaluation = basic_dataset_evaluation
        if basic_dataset_description:
            self.basic_dataset_description = basic_dataset_description

        # Update critique based on the message
        if message.strip() != "":
            self.critique = f"Previous Answer to this task: {self.selected_algorithms}. \nUser-Critique: {message}"
        else:
            self.critique = "No Critique"

        # Route to the appropriate substep
        if self.current_substep == 0:
            # Begin normalization and basic dataset analysis
            self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Begin algorithm selection
            self.substep_1(self.critique)
        elif self.current_substep == 2:
            # Handle user critique after algorithm selection
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.finished = True
                print("Step 3 is finished.")
                return
            else:
                # Non-empty message is treated as a critique; redo algorithm selection
                self.finished = False
                self.substep_1(self.critique)
        else:
            raise ValueError(f"Invalid substep state: {self.current_substep}")

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

        self.current_substep = 1  # Move to the next substep

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

        answer_parser = ReasoningResponseParser(
            start_answer_token=algorithm_selector.start_answer_token,
            stop_answer_token=algorithm_selector.stop_answer_token,
        )

        answer_parser.reset()

        response = algorithm_selector(
            research_question=self.research_question,
            rq_class=self.classification_result,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
            critic=critique,
        )

        # Parse the streaming response
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break
            print(token, end="")  # Optional: For debugging or logging purposes

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
            print(f"\nError parsing selected algorithms: {e}")
            # Optionally, handle the error or retry

        self.current_substep = 2  # Move to the critique handling substep
        self.finished = True


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
        classification_result: str = None,
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
                classification_result,
                basic_dataset_evaluation,
                dataset,
            ]
        ):
            if selected_algorithms is not None:
                self.selected_algorithms = selected_algorithms
            if research_question is not None:
                self.research_question = research_question
            if classification_result is not None:
                self.classification_result = classification_result
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
            self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Handle user critique after hyperparameter selection
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.current_substep = 2
                self.substep_1()  # Proceed to run algorithms
            else:
                # Non-empty message is treated as a critique; redo hyperparameter selection
                self.substep_0(self.critique)
        elif self.current_substep == 2:
            # Handle user critique after running algorithms
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.current_substep = 3
                self.substep_2()  # Proceed to parse results
            else:
                # Non-empty message is treated as a critique; rerun algorithms or adjust hyperparameters
                self.substep_1()
        elif self.current_substep == 3:
            # Handle user critique after parsing results
            if message.strip() == "":
                # Empty message signifies confirmation to finish Step 4
                self.finished = True
                print("Step 4 completed successfully.")
                return
            else:
                # Non-empty message is treated as a critique; rerun analysis or adjust previous steps
                self.finished = False
                self.substep_2()
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

        # Initialize the response parser
        answer_parser = ReasoningResponseParser(
            start_answer_token=hyper_parameter_guessor.start_answer_token,
            stop_answer_token=hyper_parameter_guessor.stop_answer_token,
        )

        # Generate hyperparameters
        response = hyper_parameter_guessor(
            research_question=self.research_question,
            research_question_class=self.classification_result,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
            critic=critique,
        )

        # Parse the streaming response
        for chunk in response:
            token = chunk.choices[0].delta.content
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

    def substep_1(self):
        """
        Runs the calibrated algorithms on the dataset and collects results.
        """
        print("\nSubstep 4.1: Running calibrated algorithms on the dataset")

        self.results = {}

        for algorithm_name, algorithm in self.calibrated_algorithms.items():
            try:
                self.results[algorithm_name] = algorithm(self.dataset)
                print(f"Algorithm '{algorithm_name}' executed successfully.")
            except Exception as e:
                self.results[algorithm_name] = str(e)
                print(f"Error running algorithm '{algorithm_name}': {e}")

        print("\nCollected Results:", self.results)
        self.current_substep = 2  # Move to the next substep

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

        if message != "":
            self.critique = f"Previous Answer to this task: {self.analysis_result}. \nUser-Critique: {message}"
        else:
            self.critique = "No Critique"

        if self.current_substep == 0:
            # Begin analysis of the results
            self.substep_0(self.critique)
        elif self.current_substep == 1:
            # Handle user critique after analysis
            if message.strip() == "":
                # Empty message signifies confirmation to proceed
                self.finished = True
                return
            else:
                # Non-empty message is treated as a critique; redo analysis
                self.finished = False
                self.substep_0(self.critique)
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

        # Initialize the response parser
        answer_parser = ReasoningResponseParser(
            start_answer_token=results_analyzer.start_answer_token,
            stop_answer_token=results_analyzer.stop_answer_token,
        )

        # Generate the analysis
        response = results_analyzer(
            research_question=self.research_question,
            research_question_class=self.classification_result,
            parsed_algorithm_results=self.parsed_algorithm_results,
            search_strings=self.search_strings,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
            critique=critique,
        )

        # Parse the streaming response
        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answer_parser(token)
            if not token_type:
                break
            print(token, end="")  # Optional: For debugging or logging purposes

        self.analysis_result = answer_parser.full_output.strip()
        print("\n\nAnalysis Result:", self.analysis_result)

        self.current_substep = 1  # Move to the next substep
        self.finished = True


class Step_6:
    def __init__(self, llm: str):
        self.step_id = 5
        self.current_substep = 0
        self.llm = llm
        self.finished = False

    def __call__(self, message: str, analysis_result: str):
        pdf_generator = LaTeXPaperGenerator(llm=self.llm)
        pdf_generator(analysis_results=analysis_result)
        self.finished = True


class API:
    def __init__(self, llm: str = base_model):
        self.current_step = 0
        self.llm = llm
        self.steps = {
            0: Step_1(llm),
            1: Step_2(llm),
            2: Step_3(llm),
            3: Step_4(llm),
            4: Step_5(llm),
            5: Step_6(llm),
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

        return json.dumps(step_states)

    async def upload_user_message(self, user_message: str, step: int):

        # checking if the step is allowed
        if not self.get_allowed_steps()[step]:
            return self.get_step_states()
        else:
            if step == 0:
                response_messages = self.steps[0](user_message)
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
                if type(token) != str:
                    token = token.choices[0].delta.content
                if token is None:
                    continue

                all_tokens.append(token)
                print(token, end="")
                full_message += token
                if self.current_message.type == "reasoning":
                    # the full resoponse from open ai contains the whole message
                    # here we only yield the reasoning tokens
                    token_type = response_parser(token=token)
                    if token_type == "reasoning":
                        yield token
            except Exception as e:
                print("Exception: ", e)
                continue

        if self.current_message.callback is not None:
            answer_messages = self.current_message.callback(all_tokens)

            for message in answer_messages:
                self.messages.put(message)

        self.current_message = None  # Reset the current message


class UserMessage(BaseModel):
    text: str
    step: int


app = FastAPI()
api = API(llm=base_model)


@app.get("/status")
async def status():
    return await api.get_step_states()


@app.post("/user_message")
async def user_message(data: UserMessage):
    await api.upload_user_message(user_message=data.text, step=data.step)


@app.get("/current_message")
async def current_message():
    return await api.get_current_message()


@app.get("/stream_current_message")
async def stream_current_message():
    return StreamingResponse(
        api.stream_current_message(), media_type="text/plain"
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)

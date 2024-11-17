# hyperparameter_guessor.py

from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import re
from typing import Tuple, Optional, Dict, Any
from agents.utils import json_to_dict  
from agents.ReasoningTextGenerator import ReasoningTextGenerator


class HyperParameterGuessor(ReasoningTextGenerator):
    def __init__(
        self,
        prompt_explanation: str,
        llm: ChatOpenAI,
        start_answer_token: str = "<START_HYPERPARAMETERS>",
        stop_answer_token: str = "<STOP_HYPERPARAMETERS>",
    ):
        """
        Initializes the HyperParameterGuessor with a specific prompt explanation, language model,
        and tokens to encapsulate the final generated hyperparameters.

        Args:
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            start_answer_token (str): Token indicating the start of the final hyperparameters.
            stop_answer_token (str): Token indicating the end of the final hyperparameters.
        """
        super().__init__(
            prompt_explanation=prompt_explanation,
            llm=llm,
            start_answer_token=start_answer_token,
            stop_answer_token=stop_answer_token,
        )

    def __call__(
        self,
        research_question: str,
        research_question_class: str,
        basic_dataset_evaluation: str,
        critic: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Generates hyperparameters based on the provided research question, its classification,
        and the basic dataset evaluation using the LLM, along with reasoning steps.

        Args:
            research_question (str): The research question guiding the analysis.
            research_question_class (str): Classification/category of the research question.
            basic_dataset_evaluation (str): A basic evaluation of the dataset.
            critic (Optional[str]): Optional critic or additional context.

        Returns:
            Dict[str, Any]: A dictionary containing the hyper_parameters and reasoning_steps.
        """
        # Construct the input prompt for the ReasoningTextGenerator
        if critic:
            input_text = f"""
            Research Question: "{research_question}"
            Research Question Classification: {research_question_class}
            Critic: "{critic}"
            Dataset Basic Evaluation: {basic_dataset_evaluation}

            Please generate a set of hyperparameters for the specified algorithm based on the Research Question and the basic evaluation of the Dataset.
            """
        else:
            input_text = f"""
            Research Question: "{research_question}"
            Research Question Classification: {research_question_class}
            Dataset Basic Evaluation: {basic_dataset_evaluation}


            Please generate a set of hyperparameters for the specified algorithm based on the Research Question and the basic evaluation of the Dataset.
            """

        # Generate hyperparameters and reasoning steps
        hyperparameters_raw, reasoning_steps = self.generate(input_text)

        # Convert the JSON output to a dictionary
        hyperparameters_dict = json_to_dict(hyperparameters_raw)

        if hyperparameters_dict is None:
            raise ValueError("Failed to parse hyperparameters JSON.")

        return {
            "hyper_parameters": hyperparameters_dict.get("hyper_parameters", {}),
            "reasoning_steps": reasoning_steps
        }

# hyperparameter_guessor.py

import os
from typing import Optional, Dict, Any
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from agents.ReasoningTextGenerator import ReasoningTextGenerator

# Assuming json_to_dict is defined in agents.utils
from agents.utils import json_to_dict  

class HyperParameterGuessor(ReasoningTextGenerator):
    def __init__(
        self,
        prompt_explanation: str,
        llm: str = "gpt-4o-mini",  # possible options: "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        temperature: float = 1.0,
        start_answer_token: str = "<START_HYPERPARAMETERS>",
        stop_answer_token: str = "<STOP_HYPERPARAMETERS>",
    ):
        """
        Initializes the HyperParameterGuessor with a specific prompt explanation, language model,
        and tokens to encapsulate the final generated hyperparameters.

        Args:
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            llm (str): The language model to use.
            temperature (float): The temperature parameter for model output creativity.
            start_answer_token (str): Token indicating the start of the final hyperparameters.
            stop_answer_token (str): Token indicating the end of the final hyperparameters.
        """
        super().__init__(
            prompt_explanation=prompt_explanation,
            llm=llm,
            temperature=temperature,
            start_answer_token=start_answer_token,
            stop_answer_token=stop_answer_token,
        )

    def __call__(
        self,
        research_question: str,
        research_question_class: str,
        basic_dataset_evaluation: str,
        critic: Optional[str] = None,
    ) -> str:
        """
        Generates hyperparameters based on the provided research question, its classification,
        and the basic dataset evaluation using the LLM, along with reasoning steps.

        Args:
            research_question (str): The research question guiding the analysis.
            research_question_class (str): Classification/category of the research question.
            basic_dataset_evaluation (str): A basic evaluation of the dataset.
            critic (Optional[str]): Optional critic or additional context.

        Returns:
            str: The raw response from the LLM containing reasoning and generated hyperparameters.
        """
        # Construct the input prompt for the ReasoningTextGenerator

        input_text = f"""
Research Question: "{research_question}"
Research Question Classification: {research_question_class}"
Dataset Basic Evaluation: {basic_dataset_evaluation}

Please generate a set of hyperparameters for the specified algorithms based on the Research Question and the basic evaluation of the Dataset.
"""

        # Generate hyperparameters and reasoning steps
        response = self.generate(input_text, critique=critic)

        return response

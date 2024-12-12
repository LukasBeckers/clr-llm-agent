import os
from typing import Dict, Optional, Union, List, Tuple
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from agents.ReasoningTextGenerator import ReasoningTextGenerator

# Assuming algorithms_selector_prompt_v2 is defined in step3.prompts
from step3.prompts import algorithms_selector_prompt_v2


class AlgorithmsSelector(ReasoningTextGenerator):
    def __init__(
        self,
        llm: str = "gpt-4o-mini",  # possible options: "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        prompt_explanation: str = algorithms_selector_prompt_v2,
        temperature: float = 1.0,
        start_answer_token: str = "<START_ALGORITHMS>",
        stop_answer_token: str = "<STOP_ALGORITHMS>",
    ):
        """
        Initializes the AlgorithmsSelector with a specific prompt explanation, language model,
        and tokens to encapsulate the final list of algorithms.

        Args:
            llm (str): The language model to use.
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            temperature (float): The temperature parameter for model output creativity.
            start_answer_token (str): Token indicating the start of the final algorithms list.
            stop_answer_token (str): Token indicating the end of the final algorithms list.
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
        Generates a list of algorithms to analyze the dataset in respect to the research question
        and its classification, along with optional critique feedback.

        Args:
            research_question (str): The original research question.
            research_question_class (str): The classification of the research question 
                                           (e.g., Explicating, Envisioning, Relating, Debating).
            basic_dataset_evaluation (str): Basic evaluation or description of the dataset.
            critic (Optional[str]): Feedback on previous algorithm selections to refine the generation.

        Returns:
            str: The raw response from the LLM containing reasoning and generated algorithms.
        """
        # Construct the input prompt for the ReasoningTextGenerator
        if critic:
            input_text = f"""
Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Critic: "{critic}"
Dataset Basic Evaluation: {basic_dataset_evaluation}"

Please generate a list of all algorithms you plan to use to 
analyze the dataset in respect to the Research Question and also 
based on the basic evaluation of the Dataset.
"""
        else:
            input_text = f"""
Research Question: "{research_question}"
Research Question Classification: {research_question_class}"
Dataset Basic Evaluation: {basic_dataset_evaluation}"

Please generate a list of all algorithms you plan to use to 
analyze the dataset in respect to the Research Question and also 
based on the basic evaluation of the Dataset.
"""

        # Generate the algorithms and reasoning
        response = self.generate(input_text, critique=critic)

        return response

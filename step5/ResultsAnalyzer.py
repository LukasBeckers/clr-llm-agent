import os
from typing import List, Tuple, Optional, Dict, Union
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from agents.TextGenerator import TextGenerator

# Assuming results_analyzer_prompt is defined in step5.prompts
from step5.prompts import results_analyzer_prompt


class ResultsAnalyzer(TextGenerator):
    def __init__(
        self,
        llm: str = "gpt-4o-mini",  # possible options: "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        prompt_explanation: str = results_analyzer_prompt,
        temperature: float = 1.0,
        max_tokens: Optional[int] = 15000,  # Increased max tokens for detailed analysis
        start_image_token: str = "<START_IMAGE>",
        stop_image_token: str = "<STOP_IMAGE>",
        start_image_description: str = "<START_IMAGE_DESCRIPTION>",
        stop_image_description: str = "<STOP_IMAGE_DESCRIPTION>",
    ):
        """
            Initializes the ResultsAnalyzer with specific prompt explanations, language model,
            and tokens to encapsulate images and their descriptions.

            Args:
                llm (str): The language model to use.
                prompt_explanation (str): A detailed explanation of the task for the LLM.
                temperature (float): The temperature parameter for model output creativity.
                max_tokens (Optional[int]): The maximum number of tokens for the generated output.
                start_image_token (str): Token indicating the start of an image.
                stop_image_token (str): Token indicating the end of an image.
                start_image_description (str): Token indicating the start of an image description.
                stop_image_description (str): Token indicating the end of an image description.
        """
        self.start_image_token = start_image_token
        self.stop_image_token = stop_image_token
        self.start_image_description = start_image_description
        self.stop_image_description = stop_image_description

        # Format the prompt_explanation with the provided tokens
        formatted_prompt_explanation = prompt_explanation.format(
            start_image_token=start_image_token,
            stop_image_token=stop_image_token,
            start_image_description=start_image_description,
            stop_image_description=stop_image_description,
        )

        super().__init__(
            prompt_explanation=formatted_prompt_explanation,
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            start_answer_token="<START_ANALYSIS>",
            stop_answer_token="<STOP_ANALYSIS>",
        )

    def __call__(
        self,
        research_question: str,
        research_question_class: str,
        basic_dataset_evaluation: str,
        hyperparameters: str, 
        parsed_algorithm_results: str,
        search_strings: str,
        critique: Optional[str] = None,
    ) -> str:
        """
        Analyzes the research results based on various inputs and returns the raw response from the LLM.

        Args:
            research_question (str): The original research question.
            research_question_class (str): The classification of the research question 
                                           (e.g., Explicating, Envisioning, Relating, Debating).
            basic_dataset_evaluation (str): Basic evaluation metrics or summaries of the dataset.
            hyperparameters (str): Details about the hyperparameters used in the algorithms.
            parsed_algorithm_results (str): The results obtained from the algorithms after parsing.
            search_strings (str): The search strings used to generate the dataset, along with their data sources.
            critique (Optional[str]): Optional feedback to refine the analysis based on previous attempts.

        Returns:
            str: The raw response from the LLM containing reasoning and generated analysis.
        """
        # Construct the input prompt for the ResultsAnalyzer
        input_text = f"""
        Research Question: "{research_question}"
        Research Question Classification: {research_question_class}
        Search Strings used to Generate Dataset (search_string, database): {search_strings} 
        Dataset Basic Evaluation: {basic_dataset_evaluation}
        Hyperparameters: {hyperparameters}
        Algorithm Analysis Results: {parsed_algorithm_results}
        """

        # Generate the analysis
        response = self.generate(input_text, critique=critique)

        return response

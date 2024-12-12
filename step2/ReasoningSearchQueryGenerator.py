import os
from typing import List, Tuple, Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from agents.ReasoningTextGenerator import ReasoningTextGenerator

# Assuming pubmed_query_generation_system_prompt is defined in step2.prompts
from step2.prompts import pubmed_query_generation_system_prompt


class  ReasoningSearchQueryGenerator(ReasoningTextGenerator):
    def __init__(
        self,
        llm: str = "gpt-4o-mini",  # possible options: "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        prompt_explanation: str = pubmed_query_generation_system_prompt,
        temperature: float = 1.0,
        start_answer_token: str = "<START_SEARCH_STRINGS>",
        stop_answer_token: str = "<STOP_SEARCH_STRINGS>",
    ):
        """
        Initializes the ReasoningSearchQueryGenerator with a specific prompt explanation, language model,
        and tokens to encapsulate the final search strings.

        Args:
            llm (str): The language model to use.
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            temperature (float): The temperature parameter for model output creativity.
            start_answer_token (str): Token indicating the start of the final search strings.
            stop_answer_token (str): Token indicating the end of the final search strings.
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
        classification_result: str,
        critic: Optional[str] = None,
    ) -> str:
        """
        Generates PubMed-compatible search queries based on the research question,
        its classification, and optional critic feedback.

        Args:
            research_question (str): The original research question.
            classification_result (str): The classification of the research question 
                                         (e.g., Explicating, Envisioning, Relating, Debating).
            critic (Optional[str]): Feedback on previous search strings to refine the generation.

        Returns:
            str: The raw response from the LLM containing reasoning and generated search strings.
        """
        # Construct the input prompt for the ReasoningTextGenerator
        if critic:
            input_text = f"""
Research Question: "{research_question}"
Classification: {classification_result}
Critic: "{critic}"

Please generate a list of 3 to 5 PubMed-compatible search strings based on the above information.
"""
        else:
            input_text = f"""
Research Question: "{research_question}"
Classification: {classification_result}"

Please generate a list of 3 to 5 PubMed-compatible search strings based on the above information.
"""

        # Generate the search strings and reasoning
        response = self.generate(input_text, critique=critic)

        return response

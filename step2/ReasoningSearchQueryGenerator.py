from step2.prompts import (
    pubmed_query_generation_system_prompt,
)
from langchain_community.chat_models import ChatOpenAI
from agents.ReasoningTextGenerator import ReasoningTextGenerator
from typing import List, Tuple, Optional


class ReasoningSearchQueryGenerator(ReasoningTextGenerator):
    def __init__(
        self,
        llm: ChatOpenAI,
        prompt_explanation: str = pubmed_query_generation_system_prompt,
        start_answer_token: str = "<START_SEARCH_STRINGS>",
        stop_answer_token: str = "<STOP_SEARCH_STRINGS>",
    ):
        """
        Initializes the ReasoningSearchQueryGenerator with a specific prompt explanation, language model,
        and tokens to encapsulate the final search strings.

        Args:
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            start_answer_token (str): Token indicating the start of the final search strings.
            stop_answer_token (str): Token indicating the end of the final search strings.
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            additional_context (Optional[str]): Any additional context or instructions for the LLM.
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
        classification_result: str,
        critic: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, str]], str]:
        """
        Generates a list of PubMed-compatible search queries based on the research question,
        its classification, and optional critic feedback, along with the reasoning steps.

        Args:
            research_question (str): The original research question.
            classification_result (str): The classification of the research question (Explicating, Envisioning, Relating, Debating).
            critic (Optional[str]): Feedback on previous search strings to refine the generation.

        Returns:
            Tuple[List[str], str]: A tuple containing the list of generated PubMed search strings and the reasoning steps.
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
        raw_search_strings, reasoning_steps = self.generate(input_text)

        search_strings = []

        for search_string_and_source in raw_search_strings.split("),"):
            search_string, data_source = search_string_and_source.split(",")
            search_string = (
                search_string.strip()
                .strip("[()'']")
                .strip()
                .strip("[()'']")
                .strip("\n")
                .strip()
            )
            data_source = (
                data_source.strip()
                .strip("[()'']")
                .strip()
                .strip('"')
                .strip("[()'']")
                .strip("\n")
            )
            print("search_string", search_string)
            print("data_source", data_source)
            search_strings.append((search_string, data_source))

        return search_strings, reasoning_steps

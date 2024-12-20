from step5.prompts import results_analyzer_prompt
from langchain_community.chat_models import ChatOpenAI
from agents.TextGenerator import TextGenerator
from typing import List, Tuple, Optional, Dict, Union


class ResultsAnalyzer(TextGenerator):
    def __init__(
        self,
        llm: ChatOpenAI,
        prompt_explanation: str = results_analyzer_prompt,
        start_image_token: str = "<START_IMAGE>",
        stop_image_token: str = "<STOP_IMAGE>",
        start_image_description: str = "<START_IMAGE_DISCRIPTION",
        stop_image_description: str = "STOP_IMAGE_DISCRIPTION",
    ):

        self.start_image_token = start_image_token
        self.stop_image_token = stop_image_token
        self.start_image_description = start_image_description
        self.stop_image_description = stop_image_description

        super().__init__(
            prompt_explanation=prompt_explanation.format(
                start_image_token=start_image_token,
                stop_image_token=stop_image_token,
                start_image_description=start_image_description,
                stop_image_description=stop_image_description,
            ),
            llm=llm,
        )

    def __call__(
        self,
        research_question: str,
        research_question_class: str,
        basic_dataset_evaluation: str,
        parsed_algorithm_results: str,
        search_strings: str,
    ) -> str:

        # Construct the input prompt for the ResultsAnalyzer

        input_text = f"""
        Research Question: "{research_question}"
        Research Question Classification: {research_question_class}
        Search Strings used to Generate Dataset (search_string, database): {search_strings} 
        Dataset basic Evaluation: {basic_dataset_evaluation}
        Algorithm Analysis Results: {parsed_algorithm_results}
        """

        # Generate the algorithms
        results = self.generate(input_text)

        return {
            "Analysis Results": results,
        }

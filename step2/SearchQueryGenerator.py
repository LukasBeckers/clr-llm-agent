# step2/SearchQueryGenerator.py

from step2.serach_query_system_prompts import pubmed_query_generation_system_prompt
from agents.TextGenerator import TextGenerator
from langchain_community.chat_models import ChatOpenAI
from typing import Optional, List


class SearchQueryGenerator:
    def __init__(
        self, 
        llm: ChatOpenAI,
        prompt_explanation: str = pubmed_query_generation_system_prompt,
        additional_context: Optional[str] = None
    ):
        """
        Initializes the SearchQueryGenerator with a specific prompt explanation and language model.

        Args:
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            additional_context (Optional[str]): Any additional context or instructions for the LLM.
        """
        super().__init__(
            prompt_explanation=prompt_explanation,
            llm=llm,
            additional_context=additional_context
        )
    
    def __call__(
        self, 
        research_question: str, 
        classification_result: str, 
        critic: Optional[str] = None
    ) -> List[str]:
        """
        Generates a list of PubMed-compatible search queries based on the research question, 
        its classification, and optional critic feedback.

        Args:
            research_question (str): The original research question.
            classification_result (str): The classification of the research question (Explicating, Envisioning, Relating, Debating).
            critic (Optional[str]): Feedback on previous search strings to refine the generation.

        Returns:
            List[str]: A list of generated PubMed search strings.
        """
        # Construct the input prompt for the TextGenerator
        if critic:
            input_text = f"""
            Research Question: "{research_question}"
            Classification: {classification_result}
            Critic: "{critic}"
            """
        else:
            input_text = f"""
            Research Question: "{research_question}"
            Classification: {classification_result}
            """
        
        # Generate the raw search strings text
        raw_search_strings = self.generate(input_text)
        
        # Process the raw text to extract individual search strings
        # Assuming the model outputs search strings separated by commas
        search_strings = [s.strip() for s in raw_search_strings.split(',') if s.strip()]
        
        return search_strings
    



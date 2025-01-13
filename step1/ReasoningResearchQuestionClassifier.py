from langchain_community.chat_models import ChatOpenAI
from agents.ReasoningBaseClassifier import ReasoningBaseClassifier
from typing import Tuple, List
import openai


class ReasoningResearchQuestionClassifier(ReasoningBaseClassifier):
    def __init__(
        self,
        llm: ChatOpenAI,
        start_answer_token: str = "<START_ANSWER>",
        stop_answer_token: str = "<STOP_ANSWER>",
        valid_classes: List[str] = [
            "Explicating",
            "Envisioning",
            "Relating",
            "Debating",
        ],
    ):
        """
        Initializes the ReasoningResearchQuestionClassifier with specific valid classes,
        a prompt explanation, and tokens to encapsulate the final classification result.

        Args:
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            start_answer_token (str): Token indicating the start of the final answer.
            stop_answer_token (str): Token indicating the end of the final answer.
        """
        # Define the valid classes for classification

        # Provide a detailed explanation to guide the LLM in classification
        prompt_explanation = """
        **Classification Definitions:**
        
        1. **Explicating**: 
           - Research questions that seek to clarify, describe, or provide detailed explanations of existing concepts, theories, or phenomena.
           - Example: "What are the underlying mechanisms of the Technology Acceptance Model (TAM)?"
        
        2. **Envisioning**: 
           - Research questions that aim to develop new theories, frameworks, or explore future possibilities and scenarios.
           - Example: "How can artificial intelligence be integrated into TAM to predict future technology trends?"
        
        3. **Relating**: 
           - Research questions that investigate the relationships, connections, or interactions between different concepts, variables, or phenomena.
           - Example: "What is the relationship between user satisfaction and TAM in mobile application usage?"
        
        4. **Debating**: 
           - Research questions that challenge, critique, or engage in scholarly debate about existing theories, practices, or findings.
           - Example: "Does TAM adequately account for cultural differences in technology adoption?"
        
        **Instructions:**
        - Carefully analyze the research question provided.
        - Determine which of the four categories (**Explicating**, **Envisioning**, **Relating**, **Debating**) best fits the nature and intent of the question.
        - Ensure that the classification aligns with the definitions provided above.
        """

        # Initialize the ReasoningBaseClassifier with the defined classes, explanation, and tokens
        super().__init__(
            valid_classes=valid_classes,
            prompt_explanation=prompt_explanation,
            llm=llm,
            start_answer_token=start_answer_token,
            stop_answer_token=stop_answer_token,
        )

    def __call__(self, text: str, critique:str = "") -> openai.Stream:
        """
        Classifies a given research question into one of the predefined classes and provides reasoning.

        Args:
            text (str): The research question to classify.

        Returns:
            Tuple[str, str]: A tuple containing the classification result and the reasoning steps.
        """
        # Run the classification chain with the provided text
        response = self.classify(text=text, critique=critique)
        return response

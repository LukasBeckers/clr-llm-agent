# step1/ResearchQuestionClassifier.py

from agents.BaseClassifier import BaseClassifier
from langchain_community.chat_models import ChatOpenAI
from typing import List


class ResearchQuestionClassifier(BaseClassifier):
    def __init__(self, llm: ChatOpenAI):
        """
        Initializes the ResearchQuestionClassifier with specific valid classes and a prompt explanation.

        Args:
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
        """
        # Define the valid classes for classification
        valid_classes = ['Explicating', 'Envisioning', 'Relating', 'Debating']
        
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
        
        # Initialize the BaseClassifier with the defined classes and explanation
        super().__init__(
            valid_classes=valid_classes,
            prompt_explanation=prompt_explanation,
            llm=llm
        )
    
    
    def __call__(self, text: str) -> str:
        """
        Classifies a given research question into one of the predefined classes.

        Args:
            text (str): The research question to classify.

        Returns:
            str: The classification result.
        """
        # Run the classification chain with the provided text
        classification = self.classify(text=text)

        return classification
        


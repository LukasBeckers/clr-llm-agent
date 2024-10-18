# agents/BaseClassifier.py
from langchain.prompts import PromptTemplate  # Updated import
from langchain_community.chat_models import ChatOpenAI  # Updated import
from langchain.chains import LLMChain  # Continue using LLMChain
from typing import List

class BaseClassifier:
    def __init__(self, 
                 valid_classes: List[str],
                 llm: ChatOpenAI
            ):
        self.prompt_template = PromptTemplate(
            input_variables=["text", "valid_classes"],
            template="""
            You are a text classifier. Classify the following text into one of the categories: {valid_classes}.

            Text: "{text}"

            Classification:
            """
        )
        self.valid_classes = valid_classes
        self.llm = llm

        self.classification_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )

    def _validate_result(self, result: str) -> bool:
        """
        Returns True if the result is in the valid_classes 
        and False if not. 
        """
        result = result.strip()
        return result in self.valid_classes

    def _correct_result(self, result: str) -> str:
        """
        Takes the implementation results and optimizes them, 
        should be used when _validate_result returns False

        Currently just a mock implementation
        """
        return result

    def __call__(self, text: str) -> str: 
        classification = self.classification_chain.run(
            text=text,
            valid_classes=self.valid_classes
        )

        validity = self._validate_result(classification)

        if validity:
            return classification
        else:
            return self._correct_result(classification)

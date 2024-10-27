from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import re
from typing import List, Tuple


class ReasoningBaseClassifier:
    def __init__(
        self,
        valid_classes: List[str],
        prompt_explanation: str,
        llm: ChatOpenAI,
        start_answer_token: str,
        stop_answer_token: str
    ):
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token

        self.prompt_template = PromptTemplate(
            input_variables=["text", "valid_classes", "prompt_explanation"],
            template=f"""
You are a text classifier. Classify the following text into one of the categories: {{valid_classes}}.

Further Explanation of your task: 
{{prompt_explanation}}

Please provide your reasoning step by step. After completing your reasoning, provide the classification result enclosed within `{self.start_answer_token}` and `{self.stop_answer_token}`.

Text: "{{text}}"

Response:
"""
        )
        self.valid_classes = valid_classes
        self.prompt_explanation = prompt_explanation
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

    def _split_output(self, output: str) -> Tuple[str, str]:
        """
        Splits the model output into reasoning steps and the final answer.

        Args:
            output (str): The raw output from the LLM.

        Returns:
            Tuple[str, str]: A tuple containing the final answer and the reasoning steps.
        """
        pattern = re.escape(self.start_answer_token) + r'(.*?)' + re.escape(self.stop_answer_token)
        match = re.search(pattern, output, re.DOTALL)
        if match:
            final_answer = match.group(1).strip()
            reasoning_steps = output.replace(match.group(0), '').strip()
            return final_answer, reasoning_steps
        else:
            # If tokens not found, treat entire output as reasoning with no final answer
            return "", output.strip()

    def classify(self, text: str) -> Tuple[str, str]:
        """
        Classifies the given text and returns the final answer along with reasoning steps.

        Args:
            text (str): The text to classify.

        Returns:
            Tuple[str, str]: A tuple containing the final answer and reasoning steps.
        """
        output = self.classification_chain.run(
            text=text,
            valid_classes=self.valid_classes,
            prompt_explanation=self.prompt_explanation
        )

        final_answer, reasoning_steps = self._split_output(output)

        if self._validate_result(final_answer):
            return final_answer, reasoning_steps
        else:
            corrected_answer = self._correct_result(final_answer)
            return corrected_answer, reasoning_steps
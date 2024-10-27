from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
import re
from typing import Tuple, Optional


class ReasoningTextGenerator:
    def __init__(
        self,
        prompt_explanation: str,
        llm: ChatOpenAI,
        start_answer_token: str,
        stop_answer_token: str,
    ):
        """
        Initializes the ReasoningTextGenerator with a specific prompt explanation, language model,
        and tokens to encapsulate the final generated text.

        Args:
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            start_answer_token (str): Token indicating the start of the final answer.
            stop_answer_token (str): Token indicating the end of the final answer.
        """
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token

        # Define the prompt template with placeholders for prompt_explanation and input_text
        self.prompt_template = PromptTemplate(
            input_variables=["prompt_explanation", "input_text"],
            template=f"""
{{prompt_explanation}}

Input: "{{input_text}}"

Please provide your reasoning step by step. After completing your reasoning, provide the generated text enclosed within `{self.start_answer_token}` and `{self.stop_answer_token}`.

Generated Text:
"""
        )

        self.prompt_explanation = prompt_explanation
        self.llm = llm

        # Initialize the LLMChain with the defined prompt template
        self.generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )

    def _split_output(self, output: str) -> Tuple[str, str]:
        """
        Splits the model output into reasoning steps and the final generated text.

        Args:
            output (str): The raw output from the LLM.

        Returns:
            Tuple[str, str]: A tuple containing the final generated text and the reasoning steps.
        """
        pattern = re.escape(self.start_answer_token) + r'(.*?)' + re.escape(self.stop_answer_token)
        match = re.search(pattern, output, re.DOTALL)
        if match:
            final_text = match.group(1).strip()
            reasoning_steps = output.replace(match.group(0), '').strip()
            return final_text, reasoning_steps
        else:
            # If tokens not found, treat entire output as reasoning with no final text
            return "", output.strip()

    def _validate_result(self, result: str) -> bool:
        """
        Optional: Implement validation logic for the generated text if needed.

        Args:
            result (str): The final generated text.

        Returns:
            bool: True if valid, False otherwise.
        """
        # Placeholder for validation logic
        return True

    def _correct_result(self, result: str) -> str:
        """
        Optional: Implement correction logic for invalid generated text.

        Args:
            result (str): The final generated text.

        Returns:
            str: The corrected generated text.
        """
        # Placeholder for correction logic
        return result

    def generate(self, input_text: str) -> Tuple[str, str]:
        """
        Generates text based on the provided input using the LLM, along with reasoning steps.

        Args:
            input_text (str): The input text to guide the text generation.

        Returns:
            Tuple[str, str]: A tuple containing the final generated text and reasoning steps.
        """
        # Run the LLMChain with the current prompt_explanation and provided input_text
        output = self.generation_chain.run(
            prompt_explanation=self.prompt_explanation,
            input_text=input_text
        )

        # Split the output into final text and reasoning
        final_text, reasoning_steps = self._split_output(output)

        # Optionally validate the final text
        if self._validate_result(final_text):
            return final_text, reasoning_steps
        else:
            # Optionally correct the final text
            corrected_text = self._correct_result(final_text)
            return corrected_text, reasoning_steps
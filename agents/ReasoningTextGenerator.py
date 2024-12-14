import os
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate

class ReasoningTextGenerator:
    def __init__(
        self,
        prompt_explanation: str,
        llm: str = "gpt-4o-mini",  # possible are "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        temperature: float = 1.0,
        start_answer_token: str = "`",
        stop_answer_token: str = "`",
    ):
        """
        Initializes the ReasoningTextGenerator with a prompt explanation and options for the LLM.

        Args:
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            llm (str): The language model to use.
            temperature (float): The temperature parameter for model output creativity.
            start_answer_token (str): Token indicating the start of the final answer.
            stop_answer_token (str): Token indicating the end of the final answer.
        """
        self.prompt_explanation = prompt_explanation
        self.llm = llm
        self.temperature = temperature
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token

        # Define the prompt template
        self.prompt_template = PromptTemplate(
            input_variables=["input_text", "critique"],
            template=f"""

Critique of previous attempt by the user, take this seriously: 
"{{critique}}"

Input: "{{input_text}}"

Please provide your reasoning step by step. After completing your reasoning, provide the generated text enclosed within `{self.start_answer_token}` and `{self.stop_answer_token}`.

Generated Text:
"""
        )

        load_dotenv()
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def generate(self, input_text: str, critique: str = ""):
        """
        Generates text based on the provided input using the LLM, along with reasoning steps.
        The reasoning and final answer extraction is done outside of this class, so we return the raw streaming response.

        Args:
            input_text (str): The input text to guide generation.
            critique (str): Optional critique of a previous attempt.

        Returns:
            The raw streaming response from the OpenAI API.
        """

        prompt = self.prompt_template.format(
            input_text=input_text,
            critique=critique
        )

        prompt = self.prompt_explanation + prompt

        response = self.client.chat.completions.create(
            model=self.llm,
            messages=[
                {
                    "role": "user",
                    "content": prompt,
                }
            ],
            temperature=self.temperature,
            stream=True,
        )

        return response

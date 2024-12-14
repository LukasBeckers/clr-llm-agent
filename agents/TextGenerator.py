import os
from typing import Optional
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate

# Assuming necessary environment variables are loaded elsewhere or here
load_dotenv()


class TextGenerator:
    def __init__(
        self,
        prompt_explanation: str,
        llm: str = "gpt-4o-mini",  # possible options: "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        temperature: float = 1.0,
        max_tokens: Optional[int] = 1000,  # Default max tokens set to 1000
        start_answer_token: str = "`",
        stop_answer_token: str = "`",
    ):
        """
        Initializes the TextGenerator with a specific prompt explanation, language model,
        and tokens to encapsulate the final generated text.

        Args:
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            llm (str): The language model to use.
            temperature (float): The temperature parameter for model output creativity.
            max_tokens (Optional[int]): The maximum number of tokens for the generated output.
            start_answer_token (str): Token indicating the start of the final answer.
            stop_answer_token (str): Token indicating the end of the final answer.
        """
        self.prompt_explanation = prompt_explanation
        self.llm = llm
        self.temperature = temperature
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token

        # Define the prompt template with placeholders for prompt_explanation and input_text
        self.prompt_template = PromptTemplate(
            input_variables=["input_text", "critique"],
            template=f"""

Critique of previous attempt by the user, take this seriously: 
"{{critique}}"

Input: "{{input_text}}"

Please generate the text based on the input. After completing your generation, provide the generated text enclosed within `{self.start_answer_token}` and `{self.stop_answer_token}`.

Generated Text:
""",
        )

        # Initialize the OpenAI client
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        if self.llm not in ["o1-mini", "o1-preview"]:
            self.client.max_tokens = max_tokens

    def generate(self, input_text: str, critique: str = "") -> str:
        """
        Generates text based on the provided input using the LLM.
        The reasoning and final answer extraction is done outside of this class, so we return the raw streaming response.

        Args:
            input_text (str): The input text to guide the text generation.
            critique (str): Optional critique of a previous attempt.

        Returns:
            str: The raw response from the LLM containing reasoning and generated text.
        """
        # Construct the input prompt
        prompt = self.prompt_template.format(
            input_text=input_text,
            critique=critique
        )

        prompt = self.prompt_explanation + prompt

        # Generate the text using the OpenAI API
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

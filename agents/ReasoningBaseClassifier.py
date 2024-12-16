# agents/ReasoningBaseClassifier.py

import re
from typing import List, Tuple
import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
from langchain.prompts import PromptTemplate


class ReasoningBaseClassifier:
    def __init__(
        self,
        valid_classes: List[str],
        prompt_explanation: str,
        llm: str = "gpt-4o-mini",  # possible are "gpt-4o-mini" "gpt-4" "gpt-4o" "o1-mini"
        temperature: float = 1.0,
        start_answer_token: str = "`",
        stop_answer_token: str = "`",
    ):
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token
        self.valid_classes = valid_classes
        self.prompt_explanation = prompt_explanation

        self.prompt_template = PromptTemplate(
            input_variables=["text", "critique"],
            template=f"""
You are a text classifier. Classify the following text into one of the categories: {self.valid_classes}.

Further Explanation of your task: 
{self.prompt_explanation}

Critique of previous attempt by the user, take this seriously, if the user demands a certain outcome follow the users will even though you might disagree! 
"{{critique}}"

Please provide your reasoning step by step. After completing your reasoning, provide the classification result enclosed within `{self.start_answer_token}` and `{self.stop_answer_token}`.

Text: "{{text}}"

Response:
""",
        )
        self.valid_classes = valid_classes
        self.prompt_explanation = prompt_explanation
        self.llm = llm
        self.temperature = temperature
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def classify(self, text: str, critique: str="") -> openai.Stream:
        """
        Asynchronously classifies the given text, updating the state step by step.

        Args:
            text (str): The text to classify.
        """
        prompt = self.prompt_template.format(
                        text=text, critique=critique
                    )
        
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

import re
from typing import Tuple


class ReasoningResponseParser:
    def __init__(
        self,
        start_answer_token: str = "`",
        stop_answer_token: str = "`",
    ):
        self.start_answer_token = start_answer_token
        self.stop_answer_token = stop_answer_token

        self.full_output = ""
        self.reasoning = ""
        self.answer = ""

    def __call__(self, token: str) -> str:

        if self.start_answer_token in self.full_output:
            self.answer += token
            self.answer = self.answer.replace(self.start_answer_token, "")
            self.answer = self.answer.replace(self.stop_answer_token, "")
            self.full_output += token
            return "answere"

        else:
            self.reasoning += token
            self.reasoning = self.reasoning.replace(
                self.start_answer_token, ""
            )
            self.reasoning = self.reasoning.replace(self.stop_answer_token, "")
            self.full_output += token
            return "reasoning"        

    def reset(self):
        self.full_output = ""
        self.reasoning = ""
        self.answer = ""

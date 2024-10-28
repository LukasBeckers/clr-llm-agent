from step3.prompts import (
    algorithms_selector_prompt
)
from langchain_community.chat_models import ChatOpenAI
from agents.ReasoningTextGenerator import ReasoningTextGenerator
from typing import List, Tuple, Optional


class AlgorithmsSelector(ReasoningTextGenerator):
    def __init__(
        self,
        llm: ChatOpenAI,
        prompt_explanation: str = algorithms_selector_prompt,
        start_answer_token: str = "<START_SEARCH_STRINGS>",
        stop_answer_token: str = "<STOP_SEARCH_STRINGS>",
    ):

        super().__init__(
            prompt_explanation=prompt_explanation,
            llm=llm,
            start_answer_token=start_answer_token,
            stop_answer_token=stop_answer_token,
        )

    def __call__(
        self,
        research_question: str,
        research_queston_class: str,
        basic_dataset_evaluation: str,
        critic: Optional[str] = None,
    ) -> Tuple[List[Tuple[str, str]], str]:

        # Construct the input prompt for the ReasoningTextGenerator
        if critic:
            input_text = f"""
            Research Question: "{research_question}"
            Research Question Classification: {research_queston_class}
            Critic: "{critic}"
            Dataset basic Evaluation: {basic_dataset_evaluation}


            Please generate a list of all algorithms you plan to use to 
            analze the dataset in respect to the Research Question and also 
            based on the basic evaluation of the Dataset. 
            """
        else:
            input_text = f"""
            Research Question: "{research_question}"
            Research Question Classification: {research_queston_class}
            Dataset basic Evaluation: {basic_dataset_evaluation}


            Please generate a list of all algorithms you plan to use to 
            analze the dataset in respect to the Research Question and also 
            based on the basic evaluation of the Dataset. 
            """

        # Generate the algorithms
        algorithms_raw, reasoning_steps = self.generate(input_text)

        algorithms = []

        for algorithm in algorithms_raw.split(","):
            algorithm = algorithm.strip(', "()[]"`')
            algorithms.append(algorithm)

        return algorithms, reasoning_steps

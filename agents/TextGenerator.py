from langchain.prompts import PromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from typing import Optional

class TextGenerator:
    def __init__(
        self, 
        prompt_explanation: str, 
        llm: ChatOpenAI,
        max_tokens: Optional[int] = 1000,  # Default max tokens set to 1000
    ):
        """
        Initializes the TextGenerator with a specific prompt explanation and language model.

        Args:
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            max_tokens (Optional[int]): The maximum number of tokens for the generated output.
        """
        # Define the prompt template with placeholders for prompt_explanation and input_text
        self.prompt_template = PromptTemplate(
            input_variables=["prompt_explanation", "input_text"],
            template="""
            {prompt_explanation}

            Input: "{input_text}"

            Generated Text:
            """
        )
        
        self.prompt_explanation = prompt_explanation
        self.llm = llm
        
        # Set the max tokens in the LLM instance
        self.llm.max_tokens = max_tokens
        
        # Initialize the LLMChain with the defined prompt template
        self.generation_chain = LLMChain(
            llm=self.llm,
            prompt=self.prompt_template
        )
    
    def generate(self, input_text: str) -> str:
        """
        Generates text based on the provided input using the LLM.

        Args:
            input_text (str): The input text to guide the text generation.

        Returns:
            str: The generated text from the LLM.
        """
        # Run the LLMChain with the current prompt_explanation and provided input_text
        generated_text = self.generation_chain.run(
            prompt_explanation=self.prompt_explanation,
            input_text=input_text
        )
        return generated_text

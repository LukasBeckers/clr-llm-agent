import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Updated import

def create_llm(llm_name: str, temperature: float = 0.2) -> ChatOpenAI:
    # Load environment variables from .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    llm = ChatOpenAI(
        model_name=llm_name, 
        openai_api_key=openai_api_key, 
        temperature=temperature
    )
    return llm


# Replace "gpt-4o-mini" with your actual model name, e.g., "gpt-4"
gpt_4o_mini = create_llm("gpt-4")

import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI  # Updated import

def create_llm(llm_name: str, temperature: float = 0.2) -> ChatOpenAI:
    # Load environment variables from .env file
    load_dotenv()
    openai_api_key = os.getenv("OPENAI_API_KEY")

    if llm_name == "o1-preview" or llm_name == "o1-mini":
        temperature = 1

    llm = ChatOpenAI(
        model_name=llm_name, 
        openai_api_key=openai_api_key, 
        temperature=temperature
    )
    return llm

gpt_4o_mini = create_llm("gpt-4o-mini")
gpt_4 = create_llm("gpt-4")
gpt_4o = create_llm("gpt-4o")
o1_mini = create_llm("o1-mini")

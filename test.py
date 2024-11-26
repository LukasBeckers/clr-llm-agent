from tools.TextNormalizer import TextNormalizer
from algorithms.DynamicTopicModeling import DynamicTopicModeling
import pickle as pk
import os
import time
from agents.TextGenerator import TextGenerator
from agents.LLMs import create_llm



if __name__ == "__main__":  
    test_llm = create_llm("o1-mini")
    text_gen = TextGenerator("", llm=test_llm)

    print(text_gen.generate("HI!"))


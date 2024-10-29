from step1.ReasoningResearchQuestionClassifier import (
    ReasoningResearchQuestionClassifier,
)
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from step3.prompts import algorithms_selector_prompt
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from agents.LLMs import gpt_4o_mini, gpt_4, gpt_4o
from dotenv import load_dotenv
import os
import pickle as pk

load_dotenv()

email = os.getenv("EMAIL_ADDRESS")

base_model = gpt_4o_mini

if __name__ == "__main__":

    # Step 1
    research_question_classifier = ReasoningResearchQuestionClassifier(base_model)

    rq =  "How has the Research concerning the glymphatic System Changed over time?"
    output = research_question_classifier(rq)

    rq_class, reasoning_rq = output

    print("Research Question: ", rq)
    print("Reasoning for Question Classification: ", reasoning_rq)
    print("Research Question Classification: ", rq_class)

    # Step 2
    pubmed_search_string_generator = ReasoningSearchQueryGenerator(base_model)

    search_strings, reasoning_search_strings = pubmed_search_string_generator(
        research_question=rq,
        classification_result=rq_class
    )

    print("Reasoning Search Strings: ", reasoning_search_strings)
    print("Search Strings: ", search_strings)

    data_loader = DataLoader(email = email)
    data_set = data_loader(search_strings=search_strings[:2])

    with open(os.path.join("temp", "dataset"), "wb") as f:
        pk.dump(data_set, f)

    with open(os.path.join("temp", "dataset"), "rb") as f:
        data_set = pk.load(f)

    text_normalizer = TextNormalizer()

    for data_point in data_set:
        try:
            data_point["Abstract Normalized"] = text_normalizer(
                data_point["Abstract"]
            )
        except KeyError:
            pass
    
    # Step3
    algorithm_selector = AlgorithmsSelector(
        prompt_explanation=algorithms_selector_prompt, llm=base_model
    )

    basic_dataset_analyzer = BasicDatasetAnalyzer(llm=base_model)

    basic_dataset_evalutation, basic_dataset_description = (
        basic_dataset_analyzer(data_set)
    )

    print("Dataset Evaluation", basic_dataset_evalutation)
    print("Dataset_description", basic_dataset_description)

    selected_algorithms, reasoning_steps = algorithm_selector(
       rq, rq_class, basic_dataset_evaluation=basic_dataset_evalutation
    )

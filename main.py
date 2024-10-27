from step1.ReasoningResearchQuestionClassifier import (
    ReasoningResearchQuestionClassifier,
)
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from agents.LLMs import gpt_4o_mini
from dotenv import load_dotenv
import os
import pickle as pk

load_dotenv()

email = os.getenv("EMAIL_ADDRESS")


if __name__ == "__main__":

    # # Step 1
    # research_question_classifier = ReasoningResearchQuestionClassifier(gpt_4o_mini)

    # rq =  "How has the Research concerning the glymphatic System Changed over time?"
    # output = research_question_classifier(rq)

    # rq_class, reasoning_rq = output

    # print("Research Question: ", rq)
    # print("Reasoning for Question Classification: ", reasoning_rq)
    # print("Research Question Classification: ", rq_class)

    # # Step 2
    # pubmed_search_string_generator = ReasoningSearchQueryGenerator(gpt_4o_mini)

    # search_strings, reasoning_search_strings = pubmed_search_string_generator(
    #     research_question=rq,
    #     classification_result=rq_class
    # )

    # print("Reasoning Search Strings: ", reasoning_search_strings)
    # print("Search Strings: ", search_strings)

    # data_loader = DataLoader(email = email)
    # data_set = data_loader(search_strings=search_strings[:2])

    # with open(os.path.join("temp", "dataset"), "wb") as f:
    #     pk.dump(data_set, f)

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

    print(data_set[0])

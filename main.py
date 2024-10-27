from step1.ReasoningResearchQuestionClassifier import ReasoningResearchQuestionClassifier
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from tools.PubmedDataloader import PubmedDataloader
from agents.LLMs import gpt_4o_mini


if __name__=="__main__":

    # Step 1
    research_question_classifier = ReasoningResearchQuestionClassifier(gpt_4o_mini)

    rq =  "How has the Research concerning the glymphatic System Changed over time?"
    output = research_question_classifier(rq)

    rq_class, reasoning_rq = output

    print("Research Question: ", rq)
    print("Reasoning for Question Classification: ", reasoning_rq)
    print("Research Question Classification: ", rq_class)

    # Step 2
    pubmed_search_string_generator = ReasoningSearchQueryGenerator(gpt_4o_mini)

    search_strings, reasoning_search_strings = pubmed_search_string_generator(
        research_question=rq,
        classification_result=rq_class
    )

    print("Reasoning Search Strings: ", reasoning_search_strings)
    print("Search Strings: ", search_strings)
    
    data_loader = PubmedDataloader("http://127.0.0.1:5112")
    search_strings = ["Brain_Clearance"]

    for search_string in search_strings: 
        json_data = data_loader.search_and_download(search_string=search_string, email="beckers@time.rwth-aachen.de")
        print("Json Data", json_data)

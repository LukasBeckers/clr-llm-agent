from step1.ResearchQuestionClassifer import ResearchQuestionClassifier
from step2.SearchQueryGenerator import SearchQueryGenerator

from agents.LLMs import gpt_4o_mini


if __name__=="__main__":

    # Step 1
    research_question_classifier = ResearchQuestionClassifier(gpt_4o_mini)

    rq =  "How has the Research concerning the glymphatic System Changed over time?"
    rq_class = research_question_classifier(rq)

    # Step 2
    pubmed_search_string_generator = SearchQueryGenerator(gpt_4o_mini)

    search_strings = pubmed_search_string_generator(
        research_question=rq,
        classification_result=rq_class
    )
    print(search_strings)

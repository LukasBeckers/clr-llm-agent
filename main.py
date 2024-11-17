from algorithms import algorithms
from step1.ReasoningResearchQuestionClassifier import (
    ReasoningResearchQuestionClassifier,
)
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from step3.prompts import algorithms_selector_prompt_v2
from step4.HyperParameterGuessor import HyperParameterGuessor
from step4.prompts import hyperparamter_selection_prompts
from step4.ResultsParser import ResultsParser
from step5.ResultsAnalyzer import ResultsAnalyzer

from agents.LLMs import gpt_4o_mini, gpt_4, gpt_4o, o1_mini	
from dotenv import load_dotenv
import os
import pickle as pk

load_dotenv()

email = os.getenv("EMAIL_ADDRESS")
algorithms_selector_prompt = algorithms_selector_prompt_v2
base_model = gpt_4o

if __name__ == "__main__":

    # Step 1
    research_question_classifier = ReasoningResearchQuestionClassifier(
        base_model
    )

    # Classifing the research question
    research_question = "How has the Research concerning the glymphatic System Changed over time?"
    output = research_question_classifier(research_question)

    rq_class, reasoning_rq = output

    print("Research Question: ", research_question)
    print("Reasoning for Question Classification: ", reasoning_rq)
    print("Research Question Classification: ", rq_class)

    # Step 2

    # Generating search-strings
    pubmed_search_string_generator = ReasoningSearchQueryGenerator(base_model)

    search_strings, reasoning_search_strings = pubmed_search_string_generator(
        research_question=research_question, classification_result=rq_class
    )

    print("Reasoning Search Strings: ", reasoning_search_strings)
    print("Search Strings: ", search_strings)

    # Downloading the datasets from the datasources
    data_loader = DataLoader(email=email)
    #data_set = data_loader(search_strings=search_strings[:])
    # data_set = data_loader(search_strings=[("Glymph* OR Brain_Clearance", "pub_med")])

    # with open(os.path.join("temp", "dataset"), "wb") as f:
    #     pk.dump(data_set, f)

    with open(os.path.join("temp", "dataset"), "rb") as f:
        dataset = pk.load(f)

    # Step3

    # Perfom basic analysis of the dataset (no publications, trend over time
    # date-range etc.)

    # Normalize the Text from the Dataset
    text_normalizer = TextNormalizer()

    for data_point in dataset[:]:
        try:
            data_point["AbstractNormalized"] = text_normalizer(
                data_point["Abstract"]
            )
        # Removint datapoints that have no abstract
        except KeyError:
            dataset.remove(data_point)

    basic_dataset_analyzer = BasicDatasetAnalyzer(llm=base_model)

    basic_dataset_evalutation, basic_dataset_description = (
        basic_dataset_analyzer(dataset)
    )

    print("Dataset Evaluation", basic_dataset_evalutation)
    print("Dataset_description", basic_dataset_description)

    # Select the algorithms based on the research question, the classification
    # of the research question and the basic dataset evaluation
    algorithm_selector = AlgorithmsSelector(
        prompt_explanation=algorithms_selector_prompt, llm=base_model
    )

    algorithm_selector_output = algorithm_selector(
        research_question,
        rq_class,
        basic_dataset_evaluation=basic_dataset_evalutation,
    )
    selected_algorithms = algorithm_selector_output["algorithms"]
    algorithm_selector_reasoning_steps = algorithm_selector_output[
        "reasoning_steps"
    ]

    print("Selecting Algorithms", algorithm_selector_reasoning_steps)
    print("Selected Algorithms", selected_algorithms)

    # Step 4        

    # Calibrate the algorithms
    hyper_parameter_guessors = {
        algorithm_name: HyperParameterGuessor(
            prompt_explanation=hyperparamter_selection_prompts[algorithm_name],
            llm=base_model
        )
        for algorithm_name in selected_algorithms
    }

    hyper_parameters = {
        algorithm_name: hyper_parameter_guessor(
            research_question=research_question,
            research_question_class=rq_class,
            basic_dataset_evaluation=basic_dataset_evalutation,
        )
        for algorithm_name, hyper_parameter_guessor in hyper_parameter_guessors.items()
    }

    print(hyper_parameters["DynamicTopicModeling"])

    calibrated_algorithms = {
        algorithm_name: algorithms[algorithm_name](
            **guessing_results["hyper_parameters"]
        )
        for algorithm_name, guessing_results in hyper_parameters.items()
    }

    # Perform the analysis

    results = {
        algorithm_name: algorithm(dataset)
        for algorithm_name, algorithm in calibrated_algorithms.items()
    }

    # Parse the results
    results_parser = ResultsParser()
    results = results_parser(results=results)

    # Step5 

    # Analyze the Results
    results_analyzer = ResultsAnalyzer(llm=o1_mini)

    analysis_result = results_analyzer(results)

    print("ANALYSIS RESULTS", analysis_result)

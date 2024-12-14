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
from step4.prompts import (
    hyperparamter_selection_prompts,
    multi_algorithm_prompt,
)
from step4.ResultsParser import ResultsParser
from step5.ResultsAnalyzer import ResultsAnalyzer
from step6.LaTeXPaperGenerator import LaTeXPaperGenerator
from step6.prompts import latex_paper_prompt

from agents.ReasoningResponseParser import ReasoningResponseParser
from agents.utils import json_to_dict

from dotenv import load_dotenv
import os
import pickle as pk

load_dotenv()

email = os.getenv("EMAIL_ADDRESS")
algorithms_selector_prompt = algorithms_selector_prompt_v2
base_model = "gpt-4o"

if __name__ == "__main__":

    # Step 1
    print("Begin Step 1")
    critique = "First attempt so no critique"
    research_question_classifier = ReasoningResearchQuestionClassifier(
        base_model
    )
    answere_parser = ReasoningResponseParser(
        start_answer_token=research_question_classifier.start_answer_token,
        stop_answer_token=research_question_classifier.stop_answer_token,
    )

    while 1:
        answere_parser.reset()
        # Classifing the research question
        research_question = "How has the Research concerning the glymphatic System Changed over time?"
        response = research_question_classifier(
            research_question, critique=critique
        )

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            if not token_type: break
            print(token, end="")

        reasoning_rq = answere_parser.reasoning
        rq_class = answere_parser.answer

        user_input = input(
            "Do you have any critique? if not commit an empty message"
        )

        if user_input == "":
            break

        else:
            critique = f"Previous Answere to this task: {rq_class}. \n User-Critique: {user_input}"

    # Step 2
    print("Begin Step 2")
    critique = "First attempt so no critique"
    # Generating search-strings
    pubmed_search_string_generator = ReasoningSearchQueryGenerator(
        base_model, temperature=0
    )
    answere_parser = ReasoningResponseParser(
        start_answer_token=pubmed_search_string_generator.start_answer_token,
        stop_answer_token=pubmed_search_string_generator.stop_answer_token,
    )

    while 1:
        answere_parser.reset()
        response = pubmed_search_string_generator(
            research_question=research_question,
            classification_result=rq_class,
            critic=critique,
        )

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            if not token_type: break
            print(token, end="")

        reasoning = answere_parser.reasoning
        raw_search_strings = answere_parser.answer

        try:
            search_strings = []

            for search_string_and_source in raw_search_strings.split("),"):
                search_string, data_source = search_string_and_source.split(
                    ","
                )
                search_string = (
                    search_string.strip()
                    .strip("[()'']")
                    .strip()
                    .strip("[()'']")
                    .strip("\n")
                    .strip()
                )
                data_source = (
                    data_source.strip()
                    .strip("[()'']")
                    .strip()
                    .strip('"')
                    .strip("[()'']")
                    .strip("\n")
                )
        except Exception:
            continue

        user_input = input(
            "Do you have any critique? if not commit an empty message"
        )

        if user_input == "":
            break

        else:
            critique = f"Previous Answere to this task: {raw_search_strings}. \n User-Critique: {user_input}"

    # Downloading the datasets from the datasources
    data_loader = DataLoader(email=email)
    # data_set = data_loader(search_strings=search_strings[:])
    # data_set = data_loader(search_strings=[("Glymph* OR Brain_Clearance", "pub_med")])

    # with open(os.path.join("temp", "dataset"), "wb") as f:
    #     pk.dump(data_set, f)

    with open(os.path.join("temp", "dataset"), "rb") as f:
        dataset = pk.load(f)

    # Step3
    print("Begin Step 3")
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

    answere_parser = ReasoningResponseParser(
        start_answer_token=algorithm_selector.start_answer_token,
        stop_answer_token=algorithm_selector.stop_answer_token,
    )

    critique = "First attempt so no critique"
    while True:
        answere_parser.reset()
        response = algorithm_selector(
            research_question,
            rq_class,
            basic_dataset_evaluation=basic_dataset_evalutation,
        )

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            if not token_type: break
            print(token, end="")

        reasoning = answere_parser.reasoning
        algorithms_raw = answere_parser.answer

        user_input = input(
            "Do you have any critique? if not commit an empty message"
        )

        if user_input == "":

            break

        else:
            critique = f"Previous Answere to this task: {algorithms_raw}. \n User-Critique: {user_input}"

    selected_algorithms = []
    for algorithm in algorithms_raw.split(","):
        algorithm = algorithm.strip(', "()[]"`\n\t')
        algorithm = algorithm.strip("'")
        selected_algorithms.append(algorithm)

    print("Selected Algorithms", selected_algorithms, "Selected Algorithms")
    # Step 4

    print("Begin Step 4")

    # Calibrate the algorithms

    hyper_parameter_guessor_prompt = """"""

    for algorithm_name in selected_algorithms:
        try:
            hyper_parameter_guessor_prompt += hyperparamter_selection_prompts[
                algorithm_name
            ]
        except KeyError:
            continue

    hyper_parameter_guessor_prompt += multi_algorithm_prompt

    hyper_parameter_guessor = HyperParameterGuessor(
        prompt_explanation=hyper_parameter_guessor_prompt,
        llm=base_model,
    )

    answere_parser = ReasoningResponseParser(
        start_answer_token=hyper_parameter_guessor.start_answer_token,
        stop_answer_token=hyper_parameter_guessor.stop_answer_token,
    )

    critique = "First Attempt no citique yet."

    while True:
        answere_parser.reset()
        response = hyper_parameter_guessor(
            research_question=research_question,
            research_question_class=rq_class,
            basic_dataset_evaluation=basic_dataset_evalutation,
            critic=critique,
        )

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            if not token_type: break
            print(token, end="")

        reasoning = answere_parser.reasoning
        hyperparameters_raw = answere_parser.answer

        # Convert the JSON output to a dictionary
        hyperparameters_dict = json_to_dict(hyperparameters_raw)

        if hyperparameters_dict is None:
            raise ValueError("Failed to parse hyperparameters JSON.")

        print(hyperparameters_dict)

        user_input = input(
            "Do you have any critique? if not commit an empty message"
        )

        if user_input == "":

            break

        else:
            critique = f"Previous Answere to this task: {hyperparameters_raw}. \n User-Critique: {user_input}"

    calibrated_algorithms = {
        algorithm_name: algorithms[algorithm_name](
            **guessing_results
        )
        for algorithm_name, guessing_results in hyperparameters_dict.items()
    }

    # Perform the analysis

    results = {}

    for algorithm_name, algorithm in calibrated_algorithms.items():

        try:
            results[algorithm_name] = algorithm(dataset)
        except Exception as e:
            results[algorithm_name] = e

    # Parse the results

    results_parser = ResultsParser()
    results = results_parser(results=results)

    # Step5
    print("Begin Step 5")
    # Analyze the Results
    results_analyzer = ResultsAnalyzer(llm=base_model)

    critique = "First Try so no critique"

    while True: 
        response = results_analyzer(
            research_question=research_question,
            research_question_class=rq_class,
            parsed_algorithm_results=results,
            search_strings=search_strings,
            basic_dataset_evaluation=basic_dataset_evalutation,
            critique=critique
        )

        for chunk in response:
            token = chunk.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            print(token, end="")

        analysis_result = answere_parser.full_output

        user_input = input(
            "Do you have any critique? if not commit an empty message"
        )

        if user_input == "":

            break

        else:
            critique = f"Previous Answere to this task: {analysis_result}. \n User-Critique: {user_input}"


    # Step6

    # Generate PDF

    pdf_generator = LaTeXPaperGenerator(llm=base_model)

    pdf_generator(analysis_results=analysis_result)

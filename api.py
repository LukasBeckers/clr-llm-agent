from openai import OpenAI


from flask import Flask, request, Response, jsonify
import os
import pickle as pk
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
email = os.getenv("EMAIL_ADDRESS")

# Importing your existing logic (make sure these are in your PYTHONPATH)
from step1.ReasoningResearchQuestionClassifier import ReasoningResearchQuestionClassifier
from agents.ReasoningResponseParser import ReasoningResponseParser
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from step3.prompts import algorithms_selector_prompt_v2

# Your model settings
base_model = "gpt-4o"

app = Flask(__name__)

# Global session state
session_state = {
    "current_step": 0,  # current step 0=step1, 1=step2, 2=step3
    "phase": None,      # 'generation' or 'waiting_for_critique'
    "research_question": None,
    "rq_class": None,
    "reasoning_rq": None,
    "critique": None,
    "search_strings": None,
    "dataset": None,
    "basic_dataset_evaluation": None,
    "basic_dataset_description": None,
    "selected_algorithms": None,
}

# Initialize LLM classes
research_question_classifier = ReasoningResearchQuestionClassifier(base_model)
search_query_generator = ReasoningSearchQueryGenerator(base_model, temperature=0)
algorithm_selector = AlgorithmsSelector(prompt_explanation=algorithms_selector_prompt_v2, llm=base_model)

def stream_llm_response(response_iter, parser):
    """Stream tokens from an OpenAI response. 
       parser is a ReasoningResponseParser instance to separate reasoning/answer.
    """
    for chunk in response_iter:
        token = chunk.choices[0].delta.content
        if token is not None:
            token_type = parser(token)
            # You can format the stream as needed:
            # For example, prefix token_type so frontend knows if it's reasoning or answer
            yield f"{token_type.upper()}:{token}"

@app.route("/api", methods=["POST"])
def api_endpoint():
    # The client sends JSON: {"step": int, "user_message": str}
    req_data = request.get_json(force=True)
    step = req_data.get("step")
    user_message = req_data.get("user_message", "")

    # Validate step
    if step not in [0, 1, 2]:
        return jsonify({"error": "Only steps 0 to 2 are implemented."}), 400

    # Ensure steps are done in order
    if step != session_state["current_step"]:
        return jsonify({"error": f"Invalid step. Expected step {session_state['current_step']}."}), 400

    # Handle the logic depending on step and phase
    if step == 0:
        # Step 1 logic from original code, but step=0 in API
        return handle_step_0(user_message)
    elif step == 1:
        # Step 2 logic from original code
        return handle_step_1(user_message)
    elif step == 2:
        # Step 3 logic from original code
        return handle_step_2(user_message)


def handle_step_0(user_message):
    """
    Step 0 (original Step 1):
    - If phase is None or 'generation', we treat user_message as research question (if not already set).
    - Run classification, stream tokens.
    - Then wait for critique if needed.
    - If already waiting for critique, use user_message as critique.
    """
    if session_state["phase"] is None:
        # Initial entry into step 0
        # user_message is the research question
        session_state["research_question"] = user_message
        session_state["critique"] = "First attempt so no critique"
        session_state["phase"] = "generation"
        return stream_step_0_classification()

    elif session_state["phase"] == "waiting_for_critique":
        # We are waiting for critique
        if user_message.strip() == "":
            # No critique means we finalize step 0
            session_state["current_step"] = 1  # move to next step
            session_state["phase"] = None
            return jsonify({"message": "Step 0 completed. Move to step 1."})
        else:
            # Apply critique and run classification again
            session_state["critique"] = f"Previous Answer: {session_state['rq_class']}. User-Critique: {user_message}"
            session_state["phase"] = "generation"
            return stream_step_0_classification()


def stream_step_0_classification():
    # Run the research question classification
    answere_parser = ReasoningResponseParser(
        start_answer_token=research_question_classifier.start_answer_token,
        stop_answer_token=research_question_classifier.stop_answer_token
    )
    response = research_question_classifier(
        session_state["research_question"],
        critique=session_state["critique"]
    )

    def generate():
        for token_data in response:
            token = token_data.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            yield f"{token_type.upper()}:{token}"

        # After streaming finishes:
        session_state["reasoning_rq"] = answere_parser.reasoning
        session_state["rq_class"] = answere_parser.answer

        # Now we wait for critique
        session_state["phase"] = "waiting_for_critique"
        yield "\nDONE:Classification complete. Please provide critique (empty if none)."

    return Response(generate(), mimetype='text/plain')


def handle_step_1(user_message):
    """
    Step 1 (original Step 2):
    Similar logic:
    - If phase is None or 'generation', we run the search query generation.
    - Then wait for critique.
    - If waiting_for_critique and user_message empty means done, else incorporate critique and run again.
    """
    if session_state["phase"] is None:
        # Initial entry into step 1
        session_state["critique"] = "First attempt so no critique"
        session_state["phase"] = "generation"
        return stream_step_1_search_query()

    elif session_state["phase"] == "waiting_for_critique":
        # We are waiting for critique
        if user_message.strip() == "":
            # No critique means we finalize step 1
            session_state["current_step"] = 2  # move to next step
            session_state["phase"] = None
            return jsonify({"message": "Step 1 completed. Move to step 2."})
        else:
            # Apply critique and run again
            session_state["critique"] = f"Previous Answer: {session_state.get('search_strings','')}. User-Critique: {user_message}"
            session_state["phase"] = "generation"
            return stream_step_1_search_query()


def stream_step_1_search_query():
    answere_parser = ReasoningResponseParser(
        start_answer_token=search_query_generator.start_answer_token,
        stop_answer_token=search_query_generator.stop_answer_token
    )

    response = search_query_generator(
        research_question=session_state["research_question"],
        classification_result=session_state["rq_class"],
        critic=session_state["critique"]
    )

    def generate():
        raw_search_strings = ""
        for token_data in response:
            token = token_data.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            yield f"{token_type.upper()}:{token}"
        # Parsing after done
        reasoning = answere_parser.reasoning
        raw_search_strings = answere_parser.answer

        # Attempt parsing search strings
        try:
            search_strings = []
            for s in raw_search_strings.split("),"):
                ss, ds = s.split(",")
                ss = ss.strip(" '()\"\n")
                ds = ds.strip(" '()\"\n")
                if ss and ds:
                    search_strings.append((ss, ds))
            session_state["search_strings"] = search_strings
        except Exception:
            # If parsing fails, we can handle it or ask for critique again
            pass

        # Now we wait for critique
        session_state["phase"] = "waiting_for_critique"
        yield "\nDONE:Search query generation complete. Please provide critique (empty if none)."

    return Response(generate(), mimetype='text/plain')


def handle_step_2(user_message):
    """
    Step 2 (original Step 3):
    - If phase is None or 'generation', we run the dataset analysis and algorithm selection.
    - Then wait for critique.
    - If waiting_for_critique and user_message empty means done, else incorporate critique and run again.
    """
    if session_state["phase"] is None:
        # Initial entry into step 2
        # Load dataset, normalize texts, basic analysis, algorithm selection
        session_state["critique"] = "First attempt so no critique"
        session_state["phase"] = "generation"
        return stream_step_2_analysis_and_selection()

    elif session_state["phase"] == "waiting_for_critique":
        # waiting for critique
        if user_message.strip() == "":
            # No critique means we finalize step 2
            session_state["current_step"] = 3  # move to next step (which isn't implemented yet)
            session_state["phase"] = None
            return jsonify({"message": "Step 2 completed. Move to step 3."})
        else:
            # Apply critique and run again (if needed)
            session_state["critique"] = f"Previous Answer: {session_state.get('selected_algorithms','')}. User-Critique: {user_message}"
            session_state["phase"] = "generation"
            return stream_step_2_analysis_and_selection()


def stream_step_2_analysis_and_selection():
    # Here we do dataset loading (already done?), normalization, basic analysis, algorithm selection
    # This is a simplified version:
    # In your original code you load dataset from file, we assume dataset is already available.

    # Make sure dataset is loaded from "temp/dataset"
    if session_state["dataset"] is None:
        with open(os.path.join("temp", "dataset"), "rb") as f:
            dataset = pk.load(f)
        # Normalize text
        text_normalizer = TextNormalizer()
        normalized_data = []
        for data_point in dataset:
            if "Abstract" in data_point:
                data_point["AbstractNormalized"] = text_normalizer(data_point["Abstract"])
                normalized_data.append(data_point)
        dataset = normalized_data
        session_state["dataset"] = dataset

    # Basic dataset analysis
    basic_dataset_analyzer = BasicDatasetAnalyzer(llm=base_model)
    eval_res, description = basic_dataset_analyzer(session_state["dataset"])
    session_state["basic_dataset_evaluation"] = eval_res
    session_state["basic_dataset_description"] = description

    # Algorithm selection
    answere_parser = ReasoningResponseParser(
        start_answer_token=algorithm_selector.start_answer_token,
        stop_answer_token=algorithm_selector.stop_answer_token,
    )
    response = algorithm_selector(
        session_state["research_question"],
        session_state["rq_class"],
        basic_dataset_evaluation=session_state["basic_dataset_evaluation"],
    )

    def generate():
        algorithms_raw = ""
        for token_data in response:
            token = token_data.choices[0].delta.content
            if token is None:
                break
            token_type = answere_parser(token)
            yield f"{token_type.upper()}:{token}"

        reasoning = answere_parser.reasoning
        algorithms_raw = answere_parser.answer
        selected_algorithms = []
        for a in algorithms_raw.split(","):
            alg = a.strip(', "()[]`\n\t').strip("'")
            if alg:
                selected_algorithms.append(alg)

        session_state["selected_algorithms"] = selected_algorithms

        # Now we wait for critique
        session_state["phase"] = "waiting_for_critique"
        yield "\nDONE:Algorithm selection complete. Please provide critique (empty if none)."

    return Response(generate(), mimetype='text/plain')


if __name__ == "__main__":
    app.run(debug=True, port=5000)

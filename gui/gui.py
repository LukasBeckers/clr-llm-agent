import logging
import tkinter as tk
from tkinter import scrolledtext, messagebox, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from algorithms import algorithms
from step1.ReasoningResearchQuestionClassifier import (
    ReasoningResearchQuestionClassifier,
)
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from step3.prompts import algorithms_selector_prompt_v2
from step4.HyperParameterGuessor import HyperParameterGuessor
from step4.prompts import hyperparamter_selection_prompts
from step4.ResultsParser import ResultsParser
from step5.ResultsAnalyzer import ResultsAnalyzer
from step6.LaTeXPaperGenerator import LaTeXPaperGenerator
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from agents.LLMs import gpt_4o_mini, gpt_4o
from dotenv import load_dotenv
import os
import pickle as pk

load_dotenv()
email = os.getenv("EMAIL_ADDRESS")
base_model = gpt_4o
algorithms_selector_prompt = algorithms_selector_prompt_v2


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Automated Computational Literature Review")
        self.geometry("1200x900")
        self.configure(bg="#f4f4f4")

        # Initialize variables
        self.current_step = 0
        self.data_set = None
        self.rq_class = None
        self.search_strings = None
        self.basic_dataset_evaluation = None
        self.basic_dataset_description = None
        self.selected_algorithms = None
        self.hyper_parameters = None
        self.results = None
        self.analysis_result = None

        # Create modern style
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TNotebook", background="#ffffff", borderwidth=0)
        style.configure(
            "TNotebook.Tab",
            background="#e0e0e0",
            font=("Helvetica", 12),
            padding=(10, 5),
        )
        style.map("TNotebook.Tab", background=[("selected", "#ffffff")])
        style.configure("TFrame", background="#f4f4f4")
        style.configure("TLabel", background="#f4f4f4")

        # Header Section
        self.header_frame = ttk.Frame(self)
        self.header_frame.pack(pady=20)

        self.rq_label = ttk.Label(
            self.header_frame,
            text="Enter your research question:",
            font=("Helvetica", 16),
        )
        self.rq_label.pack()

        self.rq_entry = ttk.Entry(
            self.header_frame, width=100, font=("Helvetica", 12)
        )
        self.rq_entry.pack(pady=10)

        # Insert default research question
        default_rq = "How has the research concerning the glymphatic system changed over time?"
        self.rq_entry.insert(0, default_rq)

        # Start Button
        self.start_button = ttk.Button(
            self.header_frame, text="Start", command=self.start_process
        )
        self.start_button.pack(pady=5)

        # Notebook Tabs for Steps
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(expand=True, fill="both", padx=20, pady=20)
        self.notebook.bind("<<NotebookTabChanged>>", self.on_tab_change)

        # Create Frames for each step
        self.steps_frames = []
        for i in range(6):
            step_frame = ttk.Frame(self.notebook)
            self.notebook.add(step_frame, text=f"Step {i+1}")
            self.steps_frames.append(step_frame)
            self.create_step_content(step_frame, i + 1)

        # Disable all tabs except the first one
        for i in range(6):
            self.notebook.tab(i, state="disabled")

    def create_step_content(self, frame, step_number):
        step_label = ttk.Label(
            frame,
            text=f"Details for Step {step_number}",
            font=("Helvetica", 16),
        )
        step_label.pack(pady=10)

        text_area = scrolledtext.ScrolledText(
            frame, wrap="word", font=("Helvetica", 12), height=20
        )
        text_area.pack(expand=True, fill="both", padx=20, pady=10)
        text_area.tag_config("reasoning", foreground="blue")
        text_area.tag_config("result", foreground="green")
        text_area.tag_config("header", font=("Helvetica", 14, "bold"))
        setattr(self, f"step_{step_number}_text", text_area)

        if step_number == 5:
            # Canvas for images in step 5
            canvas_frame = ttk.Frame(frame)
            canvas_frame.pack(pady=10)
            setattr(self, f"step_{step_number}_canvas_frame", canvas_frame)

        # Start Button for each step
        start_button = ttk.Button(
            frame,
            text="Start Step",
            command=lambda s=step_number: self.run_step(s),
        )
        start_button.pack(pady=5)
        setattr(self, f"step_{step_number}_start_button", start_button)

    def start_process(self):
        # Disable the Start button
        self.start_button.config(state=tk.DISABLED)
        self.rq = self.rq_entry.get()
        if not self.rq.strip():
            messagebox.showwarning(
                "Input Error", "Please enter a research question."
            )
            self.start_button.config(state=tk.NORMAL)
            return

        # Initialize the process
        self.current_step = 0  # Start from step 0
        self.notebook.tab(0, state="normal")
        self.notebook.select(0)
        # Disable all tabs except the first one
        for i in range(1, 6):
            self.notebook.tab(i, state="disabled")
        # Wait for user to click 'Start' in Step 1

    def on_tab_change(self, event):
        selected_tab = event.widget.index("current")
        if selected_tab > self.current_step:
            messagebox.showwarning(
                "Access Denied",
                "Please complete the current step before proceeding.",
            )
            self.notebook.select(self.current_step)

    def enable_next_tab(self, tab_id: int):
        if self.current_step > tab_id:
            # you are redoing a step
            self.current_step = tab_id + 1
            return
        if self.current_step < 5:
            self.current_step += 1
            self.notebook.tab(self.current_step, state="normal")
            # Do not automatically select the next tab

    def run_step(self, step_number):
        if step_number == 1:
            self.run_step1()
        elif step_number == 2:
            self.run_step2()
        elif step_number == 3:
            self.run_step3()
        elif step_number == 4:
            self.run_step4()
        elif step_number == 5:
            self.run_step5()
        elif step_number == 6:
            self.run_step6()

    def run_step1(self):
        text_area = self.step_1_text
        text_area.delete(1.0, tk.END)
        text_area.insert(
            tk.END, f"Step 1: Research Question Classification\n", "header"
        )

        # Initialize the classifier
        research_question_classifier = ReasoningResearchQuestionClassifier(
            base_model
        )

        # Run the classifier
        output = research_question_classifier(self.rq)

        self.rq_class, reasoning_rq = output

        # Display the reasoning in blue
        text_area.insert(
            tk.END,
            f"Reasoning for Question Classification:\n{reasoning_rq}\n",
            "reasoning",
        )

        # Display the classification result in green
        text_area.insert(
            tk.END,
            f"Research Question Classification: {self.rq_class}\n",
            "result",
        )

        # Enable the next tab
        self.enable_next_tab(tab_id=0)
        # Stay on the current tab and wait for user to proceed

    def run_step2(self):
        text_area = self.step_2_text
        text_area.delete(1.0, tk.END)
        text_area.insert(
            tk.END,
            f"Step 2: Generate Search Strings and Load Data\n",
            "header",
        )

        # Generate search strings
        pubmed_search_string_generator = ReasoningSearchQueryGenerator(
            base_model
        )

        search_strings, reasoning_search_strings = (
            pubmed_search_string_generator(
                research_question=self.rq, classification_result=self.rq_class
            )
        )

        # Display reasoning in blue
        text_area.insert(
            tk.END,
            f"Reasoning Search Strings:\n{reasoning_search_strings}\n",
            "reasoning",
        )

        # Display search strings in green
        text_area.insert(
            tk.END, f"Search Strings:\n{search_strings}\n", "result"
        )

        # Save search strings for use in data loading
        self.search_strings = search_strings

        # Load data
        text_area.insert(tk.END, f"Loading data...\n")
        self.update()  # Force update of GUI

        data_loader = DataLoader(email=email)
        # For demonstration, we're skipping actual data loading
        # self.data_set = data_loader(search_strings=search_strings[:2])

        # Load dataset from disk (as per your existing code)
        with open(os.path.join("temp", "dataset"), "rb") as f:
            self.data_set = pk.load(f)

        # Display number of publications found
        num_publications = len(self.data_set)
        text_area.insert(
            tk.END,
            f"Number of publications found: {num_publications}\n",
            "result",
        )

        # Normalize text
        text_normalizer = TextNormalizer()
        for data_point in self.data_set[:]:
            try:
                data_point["Abstract Normalized"] = text_normalizer(
                    data_point["Abstract"]
                )
            except KeyError:
                self.data_set.remove(data_point)

        # Enable the next tab
        self.enable_next_tab(tab_id=2)
        # Stay on the current tab and wait for user to proceed

    def run_step3(self):
        text_area = self.step_3_text
        text_area.delete(1.0, tk.END)
        text_area.insert(
            tk.END,
            f"Step 3: Analyze Dataset and Select Algorithms\n",
            "header",
        )

        # Analyze dataset
        basic_dataset_analyzer = BasicDatasetAnalyzer(llm=base_model)

        self.basic_dataset_evaluation, self.basic_dataset_description = (
            basic_dataset_analyzer(self.data_set)
        )

        # Display dataset evaluation
        text_area.insert(
            tk.END,
            f"Dataset Evaluation:\n{self.basic_dataset_evaluation}\n",
            "result",
        )
        text_area.insert(
            tk.END,
            f"Dataset Description:\n{self.basic_dataset_description}\n",
            "result",
        )

        # Plot the 'Publications Over Time' data
        publications_over_time = self.basic_dataset_evaluation.get(
            "Publications Over Time", {}
        )
        if publications_over_time:
            self.plot_publications_over_time(publications_over_time)

        # Initialize the algorithm selector
        algorithm_selector = AlgorithmsSelector(
            prompt_explanation=algorithms_selector_prompt_v2, llm=base_model
        )

        # Select algorithms based on rq, rq_class, and basic_dataset_evaluation
        algorithm_selector_output = algorithm_selector(
            self.rq,
            self.rq_class,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
        )
        self.selected_algorithms = algorithm_selector_output["algorithms"]
        reasoning_steps = algorithm_selector_output["reasoning_steps"]

        # Display reasoning steps in blue
        text_area.insert(
            tk.END,
            f"Algorithm Selection Reasoning Steps:\n{reasoning_steps}\n",
            "reasoning",
        )

        # Display selected algorithms in green
        text_area.insert(
            tk.END,
            f"Selected Algorithms:\n{self.selected_algorithms}\n",
            "result",
        )

        # Enable the next tab
        self.enable_next_tab(tab_id=2)
        # Stay on the current tab and wait for user to proceed

    def plot_publications_over_time(self, data):
        # Prepare data
        years = list(data.keys())
        counts = list(data.values())

        # Create a figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(years, counts)
        ax.set_xlabel("Year")
        ax.set_ylabel("Number of Publications")
        ax.set_title("Publications Over Time")

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self.steps_frames[2])
        canvas.draw()
        canvas.get_tk_widget().pack()

    def run_step4(self):
        logging.info("Starting Step 4: Calibrate and Run Algorithms")
        text_area = self.step_4_text
        text_area.delete(1.0, tk.END)
        text_area.insert(
            tk.END, f"Step 4: Calibrate and Run Algorithms\n", "header"
        )

        try:
            # Calibrate the algorithms
            logging.debug("Calibrating algorithms")
            hyper_parameter_guessors = {
                algorithm_name: HyperParameterGuessor(
                    prompt_explanation=hyperparamter_selection_prompts[
                        algorithm_name
                    ],
                    llm=base_model,
                )
                for algorithm_name in self.selected_algorithms
            }

            self.hyper_parameters = {
                algorithm_name: hyper_parameter_guessor(
                    research_question=self.rq,
                    research_question_class=self.rq_class,
                    basic_dataset_evaluation=self.basic_dataset_evaluation,
                )
                for algorithm_name, hyper_parameter_guessor in hyper_parameter_guessors.items()
            }

            # Debugging, reducing the number of iterations for demonstration
            if "DynamicTopicModeling" in self.hyper_parameters:
                self.hyper_parameters["DynamicTopicModeling"]["hyper_parameters"]["t"] = 10
                self.hyper_parameters["DynamicTopicModeling"]["hyper_parameters"]["k"] = 5
                self.hyper_parameters["DynamicTopicModeling"]["hyper_parameters"]["iter"] = 500
                logging.debug("Reduced iterations for DynamicTopicModeling for debugging")
            else:
                raise KeyError("DynamicTopicModeling algorithm not found in hyper_parameters.")

            # Display hyperparameters in green
            for alg_name, params in self.hyper_parameters.items():
                text_area.insert(
                    tk.END,
                    f"Hyperparameters for {alg_name}:\n{params}\n",
                    "result",
                )
                logging.debug(f"Hyperparameters for {alg_name}: {params}")

            # Run the algorithms
            self.results = {}

            self.calibrated_algorithms = {
                algorithm_name: algorithms[algorithm_name](
                    **guessing_results["hyper_parameters"]
                )
                for algorithm_name, guessing_results in self.hyper_parameters.items()
            }

            for algorithm_name, algorithm in self.calibrated_algorithms.items():
                try:
                    logging.debug(f"Running algorithm: {algorithm_name}")
                    self.results[algorithm_name] = algorithm(self.data_set)
                    logging.debug(f"Algorithm {algorithm_name} completed successfully")
                except Exception as e:
                    self.results[algorithm_name] = e
                    logging.error(f"Error running {algorithm_name}: {e}")
                    text_area.insert(
                        tk.END,
                        f"Error running {algorithm_name}: {e}\n",
                        "reasoning",
                    )

            # Display results
            text_area.insert(
                tk.END,
                f"Algorithm Results:\n{self.results}\n",
                "result",
            )
            logging.info("Step 4 completed successfully")

            # Enable the next tab
            self.enable_next_tab(tab_id=3)

        except Exception as e:
            logging.exception("An error occurred in Step 4")
            messagebox.showerror("Error", f"An error occurred in Step 4:\n{e}")
            self.start_button.config(state=tk.NORMAL)


    def run_step5(self):
        text_area = self.step_5_text
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"Step 5: Analyze Results\n", "header")

        # Parse the results
        results_parser = ResultsParser()
        parsed_results = results_parser(results=self.results)

        # Analyze the Results
        results_analyzer = ResultsAnalyzer(llm=base_model)

        self.analysis_result = results_analyzer(
            research_question=self.rq,
            research_question_class=self.rq_class,
            parsed_algorithm_results=parsed_results,
            search_strings=self.search_strings,
            basic_dataset_evaluation=self.basic_dataset_evaluation,
        )

        # Display analysis result
        text_area.insert(
            tk.END, f"Analysis Result:\n{self.analysis_result}\n", "result"
        )

        # If there are images in the results, display them
        self.display_results_images(parsed_results)

        # Enable the next tab
        self.enable_next_tab(tab_id=4)
        # Stay on the current tab and wait for user to proceed

    def display_results_images(self, parsed_results):
        canvas_frame = self.step_5_canvas_frame

        for result in parsed_results.values():
            if "image" in result:
                fig = result["image"]
                canvas = FigureCanvasTkAgg(fig, master=canvas_frame)
                canvas.draw()
                canvas.get_tk_widget().pack()

    def run_step6(self):
        text_area = self.step_6_text
        text_area.delete(1.0, tk.END)
        text_area.insert(tk.END, f"Step 6: Generate PDF\n", "header")

        # Generate PDF
        pdf_generator = LaTeXPaperGenerator(base_model)
        pdf_generator(analysis_results=self.analysis_result)

        text_area.insert(tk.END, f"PDF Generated Successfully.\n", "result")

        messagebox.showinfo(
            "Process Completed",
            "The automated literature review process is completed.",
        )


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    main()

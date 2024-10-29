import tkinter as tk
from tkinter import scrolledtext, messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.pyplot as plt
from step1.ReasoningResearchQuestionClassifier import ReasoningResearchQuestionClassifier
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from step3.prompts import algorithms_selector_prompt
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from agents.LLMs import gpt_4o_mini, gpt_4o
from dotenv import load_dotenv
import os

load_dotenv()
email = os.getenv("EMAIL_ADDRESS")
base_model = gpt_4o

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Research Question Analysis")
        self.geometry("800x600")

        # Input field for the research question
        self.rq_label = tk.Label(self, text="Enter your research question:")
        self.rq_label.pack(pady=5)

        self.rq_entry = tk.Entry(self, width=100)
        self.rq_entry.pack(pady=5)

        # Start button
        self.start_button = tk.Button(self, text="Start", command=self.start_process)
        self.start_button.pack(pady=5)

        # Output area
        self.output_text = scrolledtext.ScrolledText(self, wrap=tk.WORD, width=100, height=20)
        self.output_text.pack(pady=5)

        # Next button
        self.next_button = tk.Button(self, text="Next", command=self.next_step, state=tk.DISABLED)
        self.next_button.pack(pady=5)

        # Initialize step counter
        self.current_step = 0

        # Tags for coloring text
        self.output_text.tag_config('reasoning', foreground='blue')
        self.output_text.tag_config('result', foreground='green')
        self.output_text.tag_config('header', font=('Helvetica', 14, 'bold'))

    def start_process(self):
        # Disable the Start button
        self.start_button.config(state=tk.DISABLED)

        # Get the research question
        self.rq = self.rq_entry.get()
        if not self.rq.strip():
            messagebox.showwarning("Input Error", "Please enter a research question.")
            self.start_button.config(state=tk.NORMAL)
            return

        # Clear the output area
        self.output_text.delete(1.0, tk.END)

        # Initialize the step counter
        self.current_step = 1

        # Start the first step
        self.step1()

    def next_step(self):
        self.current_step += 1
        self.next_button.config(state=tk.DISABLED)

        if self.current_step == 2:
            self.step2()
        elif self.current_step == 3:
            self.step3()
        else:
            self.output_text.insert(tk.END, "Process completed.\n", 'header')

    def step1(self):
        self.output_text.insert(tk.END, f"Step 1: Research Question Classification\n", 'header')

        # Initialize the classifier
        research_question_classifier = ReasoningResearchQuestionClassifier(base_model)

        # Run the classifier
        output = research_question_classifier(self.rq)

        rq_class, reasoning_rq = output

        # Display the reasoning in blue
        self.output_text.insert(tk.END, f"Reasoning for Question Classification:\n{reasoning_rq}\n", 'reasoning')

        # Display the classification result in green
        self.output_text.insert(tk.END, f"Research Question Classification: {rq_class}\n", 'result')

        # Save the classification result for use in next step
        self.rq_class = rq_class

        # Enable the Next button
        self.next_button.config(state=tk.NORMAL)

    def step2(self):
        self.output_text.insert(tk.END, f"\nStep 2: Generate Search Strings and Load Data\n", 'header')

        # Generate search strings
        pubmed_search_string_generator = ReasoningSearchQueryGenerator(base_model)

        search_strings, reasoning_search_strings = pubmed_search_string_generator(
            research_question=self.rq,
            classification_result=self.rq_class
        )

        # Display reasoning in blue
        self.output_text.insert(tk.END, f"Reasoning Search Strings:\n{reasoning_search_strings}\n", 'reasoning')

        # Display search strings in green
        self.output_text.insert(tk.END, f"Search Strings:\n{search_strings}\n", 'result')

        # Save search strings for use in data loading
        self.search_strings = search_strings

        # Load data
        self.output_text.insert(tk.END, f"Loading data...\n")
        self.update()  # Force update of GUI

        data_loader = DataLoader(email=email)
        data_set = data_loader(search_strings=search_strings[:2])  # Adjust as needed

        # Save data_set for use in next step
        self.data_set = data_set

        # Display number of publications found
        num_publications = len(data_set)
        self.output_text.insert(tk.END, f"Number of publications found: {num_publications}\n", 'result')

        # Normalize text
        text_normalizer = TextNormalizer()
        for data_point in data_set:
            try:
                data_point["Abstract Normalized"] = text_normalizer(data_point["Abstract"])
            except KeyError:
                pass

        # Enable Next button
        self.next_button.config(state=tk.NORMAL)

    def step3(self):
        self.output_text.insert(tk.END, f"\nStep 3: Analyze Dataset and Select Algorithms\n", 'header')

        # Analyze dataset
        basic_dataset_analyzer = BasicDatasetAnalyzer(llm=base_model)

        basic_dataset_evaluation, basic_dataset_description = basic_dataset_analyzer(self.data_set)

        # Display dataset evaluation
        self.output_text.insert(tk.END, f"Dataset Evaluation:\n{basic_dataset_evaluation}\n", 'result')
        self.output_text.insert(tk.END, f"Dataset Description:\n{basic_dataset_description}\n", 'result')

        # Plot the 'Publications Over Time' data
        publications_over_time = basic_dataset_evaluation.get('Publications Over Time', {})
        if publications_over_time:
            self.plot_publications_over_time(publications_over_time)

        # Initialize the algorithm selector
        algorithm_selector = AlgorithmsSelector(
            prompt_explanation=algorithms_selector_prompt,
            llm=base_model
        )

        # Select algorithms based on rq, rq_class, and basic_dataset_evaluation
        selected_algorithms, reasoning_steps = algorithm_selector(
            self.rq, self.rq_class, basic_dataset_evaluation=basic_dataset_evaluation
        )

        # Display reasoning steps in blue
        self.output_text.insert(tk.END, f"Algorithm Selection Reasoning Steps:\n{reasoning_steps}\n", 'reasoning')

        # Display selected algorithms in green
        self.output_text.insert(tk.END, f"Selected Algorithms:\n{selected_algorithms}\n", 'result')

        # Indicate process completed
        self.output_text.insert(tk.END, "Process completed.\n", 'header')

    def plot_publications_over_time(self, data):
        # Prepare data
        years = list(data.keys())
        counts = list(data.values())

        # Create a figure
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(years, counts)
        ax.set_xlabel('Year')
        ax.set_ylabel('Number of Publications')
        ax.set_title('Publications Over Time')

        # Embed the plot in the tkinter window
        canvas = FigureCanvasTkAgg(fig, master=self)
        canvas.draw()
        canvas.get_tk_widget().pack()


def main():
    app = App()
    app.mainloop()


if __name__ == "__main__":
    app = App()
    app.mainloop()

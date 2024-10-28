import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox
from tkinter import filedialog
import threading
import os
import pickle as pk
from dotenv import load_dotenv
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Import your modules
from step1.ReasoningResearchQuestionClassifier import ReasoningResearchQuestionClassifier
from step2.ReasoningSearchQueryGenerator import ReasoningSearchQueryGenerator
from step3.AlgorithmsSelector import AlgorithmsSelector
from step3.prompts import algorithms_selector_prompt
from tools.DataLoader import DataLoader
from tools.TextNormalizer import TextNormalizer
from tools.BasicDatasetAnalyzer import BasicDatasetAnalyzer
from agents.LLMs import gpt_4o_mini  # Assuming gpt_4o_mini is your base model

# Load environment variables
load_dotenv()
email = os.getenv("EMAIL_ADDRESS")
base_model = gpt_4o_mini

class ResearchApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Research Pipeline GUI")
        self.create_widgets()
        self.current_step = 0
        self.steps = [
            self.step1_classify_question,
            self.step2_generate_search_strings,
            self.step3_load_and_normalize_data,
            self.step4_analyze_dataset,
            self.step5_select_algorithms
        ]
        self.results = {}
    
    def create_widgets(self):
        # Input Frame
        input_frame = ttk.Frame(self.root, padding="10")
        input_frame.pack(fill=tk.X)

        ttk.Label(input_frame, text="Research Question:").pack(side=tk.LEFT)
        self.question_var = tk.StringVar()
        self.question_entry = ttk.Entry(input_frame, textvariable=self.question_var, width=80)
        self.question_entry.pack(side=tk.LEFT, padx=5)

        self.start_button = ttk.Button(input_frame, text="Start", command=self.start_process)
        self.start_button.pack(side=tk.LEFT)

        # Display Area
        display_frame = ttk.Frame(self.root, padding="10")
        display_frame.pack(fill=tk.BOTH, expand=True)

        self.display = scrolledtext.ScrolledText(display_frame, wrap=tk.WORD, state='disabled')
        self.display.pack(fill=tk.BOTH, expand=True)

        # Configure text tags once
        self.display.tag_configure("blue", foreground="blue")
        self.display.tag_configure("green", foreground="green")

        # Next Button
        self.next_button = ttk.Button(self.root, text="Next Step", command=self.next_step, state='disabled')
        self.next_button.pack(pady=10)

        # Plot Area (Initially Hidden)
        self.plot_frame = ttk.Frame(self.root, padding="10")
        # self.plot_frame.pack(fill=tk.BOTH, expand=True)
    
    def start_process(self):
        question = self.question_var.get().strip()
        if not question:
            messagebox.showwarning("Input Error", "Please enter a research question.")
            return
        self.start_button.config(state='disabled')
        self.append_text("Starting process...\n", "blue")
        self.results['research_question'] = question
        self.next_button.config(state='normal')
    
    def next_step(self):
        if self.current_step < len(self.steps):
            step_function = self.steps[self.current_step]
            threading.Thread(target=step_function).start()
            self.current_step += 1
        else:
            messagebox.showinfo("Process Complete", "All steps have been completed.")
            self.next_button.config(state='disabled')
    
    def append_text(self, text, color="black"):
        self.display.config(state='normal')
        
        # Get the current end index before insertion
        start_index = self.display.index(tk.END)
        
        # Insert the text
        self.display.insert(tk.END, text)
        
        # Get the new end index after insertion
        end_index = self.display.index(tk.END)
        
        # Apply the tag to the inserted text
        if color in ["blue", "green"]:
            self.display.tag_add(color, start_index, end_index)
        
        self.display.see(tk.END)
        self.display.config(state='disabled')
    
    def step1_classify_question(self):
        try:
            self.append_text("Step 1: Classifying Research Question...\n", "blue")
            research_question_classifier = ReasoningResearchQuestionClassifier(base_model)
            rq = self.results['research_question']
            output = research_question_classifier(rq)
            rq_class, reasoning_rq = output
            self.results['rq_class'] = rq_class
            self.results['reasoning_rq'] = reasoning_rq

            self.append_text(f"Reasoning for Question Classification: {reasoning_rq}\n", "green")
            self.append_text(f"Research Question Classification: {rq_class}\n", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Error in Step 1: {e}")
    
    def step2_generate_search_strings(self):
        try:
            self.append_text("Step 2: Generating Search Strings...\n", "blue")
            pubmed_search_string_generator = ReasoningSearchQueryGenerator(base_model)
            rq = self.results['research_question']
            rq_class = self.results['rq_class']
            search_strings, reasoning_search_strings = pubmed_search_string_generator(
                research_question=rq,
                classification_result=rq_class
            )
            self.results['search_strings'] = search_strings
            self.results['reasoning_search_strings'] = reasoning_search_strings

            self.append_text(f"Reasoning Search Strings: {reasoning_search_strings}\n", "green")
            self.append_text(f"Search Strings: {search_strings}\n", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Error in Step 2: {e}")
    
    def step3_load_and_normalize_data(self):
        try:
            self.append_text("Step 3: Loading and Normalizing Data...\n", "blue")
            search_strings = self.results.get('search_strings', [])[:2]
            data_loader = DataLoader(email=email)
            data_set = data_loader(search_strings=search_strings)
            self.results['data_set'] = data_set

            # Save and load dataset
            os.makedirs("temp", exist_ok=True)
            with open(os.path.join("temp", "dataset"), "wb") as f:
                pk.dump(data_set, f)
            with open(os.path.join("temp", "dataset"), "rb") as f:
                data_set = pk.load(f)
            self.results['data_set'] = data_set

            text_normalizer = TextNormalizer()
            for data_point in data_set:
                try:
                    data_point["Abstract Normalized"] = text_normalizer(data_point["Abstract"])
                except KeyError:
                    pass
            self.results['data_set_normalized'] = data_set

            self.append_text(f"Number of Publications Found: {len(data_set)}\n", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Error in Step 3: {e}")
    
    def step4_analyze_dataset(self):
        try:
            self.append_text("Step 4: Analyzing Dataset...\n", "blue")
            data_set = self.results.get('data_set_normalized', [])
            basic_dataset_analyzer = BasicDatasetAnalyzer(llm=base_model)
            basic_dataset_evaluation, basic_dataset_description = basic_dataset_analyzer(data_set)
            self.results['basic_dataset_evaluation'] = basic_dataset_evaluation
            self.results['basic_dataset_description'] = basic_dataset_description

            self.append_text(f"Dataset Evaluation: {basic_dataset_evaluation}\n", "green")
            self.append_text(f"Dataset Description: {basic_dataset_description}\n", "green")

            # Plot Publications Over Time
            pub_over_time = basic_dataset_evaluation.get('Publications Over Time', {})
            years = sorted(pub_over_time.keys())
            counts = [pub_over_time[year] for year in years]

            fig, ax = plt.subplots(figsize=(6,4))
            ax.plot(years, counts, marker='o')
            ax.set_title('Publications Over Time')
            ax.set_xlabel('Year')
            ax.set_ylabel('Number of Publications')
            ax.grid(True)

            # Embed plot in tkinter
            canvas = FigureCanvasTkAgg(fig, master=self.plot_frame)
            canvas.draw()
            canvas.get_tk_widget().pack()
            self.plot_frame.pack(fill=tk.BOTH, expand=True)

        except Exception as e:
            messagebox.showerror("Error", f"Error in Step 4: {e}")
    
    def step5_select_algorithms(self):
        try:
            self.append_text("Step 5: Selecting Algorithms...\n", "blue")
            rq = self.results.get('research_question')
            rq_class = self.results.get('rq_class')
            basic_dataset_evaluation = self.results.get('basic_dataset_evaluation')
            algorithm_selector = AlgorithmsSelector(
                prompt_explanation=algorithms_selector_prompt, llm=base_model
            )
            selected_algorithms = algorithm_selector(
                rq, rq_class, basic_dataset_evaluation=basic_dataset_evaluation
            )
            self.results['selected_algorithms'] = selected_algorithms

            self.append_text(f"Selected Algorithms: {selected_algorithms}\n", "green")
        except Exception as e:
            messagebox.showerror("Error", f"Error in Step 5: {e}")

def main():
    root = tk.Tk()
    app = ResearchApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()

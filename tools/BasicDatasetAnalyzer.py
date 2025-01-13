import os
from openai import OpenAI
from dotenv import load_dotenv
from collections import defaultdict
from datetime import datetime
from typing import List, Dict, Tuple, Optional

class BasicDatasetAnalyzer:
    """
    A class to perform basic analysis of a dataset of publications and generate descriptive summaries using OpenAI's API.

    Attributes:
        model_name (str): The name of the OpenAI model to use for generating descriptions.
        temperature (float): The temperature parameter for the OpenAI model to control creativity.
    """

    def __init__(
        self,

        llm: str = "gpt-4",  # Default to "gpt-4", can be overridden
        temperature: float = 1.0
    ):
        """
        Initializes the BasicDatasetAnalyzer with a specified OpenAI language model.

        Args:
            llm (str): The OpenAI language model to use (e.g., "gpt-4", "gpt-3.5-turbo").
            temperature (float): The temperature parameter for model output creativity.
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key not found. Please set OPENAI_API_KEY in your environment.")
        
        self.model_name = llm
        self.temperature = temperature
        self.dataset = []  # Initialize an empty dataset
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


    def analyze_dataset(self) -> Dict:
        """
        Performs basic analysis on the dataset, including total publications, date range, and publications per year.

        Returns:
            Dict: A dictionary containing the analysis results.
        """
        total_publications = len(self.dataset)
        publication_years = []

        for record in self.dataset:
            pub_date_str = record.get("PublicationDate", "")
            try:
                # Attempt to parse the publication date in 'YYYY-MMM' format, e.g., '2016-Jun'
                pub_date = datetime.strptime(pub_date_str, "%Y-%b")
                publication_years.append(pub_date.year)
            except ValueError:
                try:
                    # Attempt to parse the publication date in 'YYYY-MM-DD' format, e.g., '2016-02-01'
                    pub_date = datetime.strptime(pub_date_str, "%Y-%m-%d")
                    publication_years.append(pub_date.year)
                except ValueError:
                    # If date parsing fails, skip the record or handle accordingly
                    continue

        if not publication_years:
            date_range = "Unknown"
            publications_over_time = {}
        else:
            start_year = min(publication_years)
            end_year = max(publication_years)
            date_range = f"{start_year}-{end_year}"

            # Calculate publications per year
            publications_over_time = defaultdict(int)
            for year in publication_years:
                publications_over_time[year] += 1

            # Convert defaultdict to regular dict and sort by year
            publications_over_time = dict(sorted(publications_over_time.items()))

        analysis = {
            "Total Publications": total_publications,
            "Date Range": date_range,
            "Publications Over Time": publications_over_time,
        }

        return analysis

    def generate_description(self, prompt: str) -> str:
        """
        Generates a descriptive summary using OpenAI's ChatCompletion API based on the provided prompt.

        Args:
            prompt (str): The prompt to send to the OpenAI model.

        Returns:
            str: The generated descriptive summary.
        """
        try:
            response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[
                    {"role": "system", "content": "You are an expert at describing datasets."},
                    {"role": "user", "content": prompt}
                ],
            temperature=self.temperature,
            stream=False,
        )

            # Extract the generated text
            description = response.choices[0].message.content.strip()
            return description
        except Exception as e:
            # Handle exceptions from OpenAI API
            print(f"Error: {e}")
            return ""

    def describe_analysis(self, analysis: Dict) -> str:
        """
        Uses the OpenAI API to generate a descriptive summary of the dataset analysis.

        Args:
            analysis (Dict): The analysis results containing total publications, date range, and publications per year.

        Returns:
            str: A descriptive summary generated by the OpenAI model.
        """
        prompt = (
            f"Provide a concise summary based on the following dataset analysis:\n\n"
            f"Total Publications: {analysis['Total Publications']}\n"
            f"Date Range of Publications: {analysis['Date Range']}\n"
            f"Number of Publications Over Time:\n"
        )

        for year, count in analysis["Publications Over Time"].items():
            prompt += f"  {year}: {count} publications\n"

        prompt += (
            "\nDescribe the dataset characteristics and any observable trends."
        )

        description = self.generate_description(prompt)
        return description

    def __call__(self, dataset: List[Dict]) -> Tuple[Dict, str]:
        """
        Executes the dataset analysis and generates a descriptive summary.

        Args:
            dataset (List[Dict]): A list of publication records, each represented as a dictionary.

        Returns:
            Tuple[Dict, str]: A tuple containing the analysis dictionary and the descriptive summary.
        """
        self.dataset = dataset
        analysis = self.analyze_dataset()
        print("Analysis:", analysis)
        description = self.describe_analysis(analysis)
        return analysis, description

# Example Usage
if __name__ == "__main__":
    # Sample dataset
    sample_dataset = [
        {"PublicationDate": "2016-Jun", "Title": "Study on AI"},
        {"PublicationDate": "2017-07", "Title": "Research on ML"},
        {"PublicationDate": "2018-08-15", "Title": "Deep Learning Advances"},
        {"PublicationDate": "2019-09-10", "Title": "Neural Networks"},
        {"PublicationDate": "2020-10", "Title": "AI in Healthcare"},
    ]

    # Initialize the analyzer with the desired OpenAI model
    analyzer = BasicDatasetAnalyzer(llm="gpt-4", temperature=0.7)

    # Perform analysis and generate description
    analysis_result, descriptive_summary = analyzer(sample_dataset)

    print("\nDescriptive Summary:")
    print(descriptive_summary)

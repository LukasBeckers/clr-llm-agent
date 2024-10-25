# step2/SearchQueryGenerator.py

from agents.TextGenerator import TextGenerator
from langchain_community.chat_models import ChatOpenAI
from typing import Optional, List

class SearchQueryGenerator:
    def __init__(
        self, 
        llm: ChatOpenAI,
        prompt_explanation: str = """
       You are an AI assistant specialized in generating PubMed search queries based on research questions and their classifications. 
    Your task is to create a list of search strings that adhere to PubMed's search syntax rules. 
    Use the following guidelines to generate the search strings:

    **PubMed Search Syntax:**
    - Use quotes and underscores to indicate that two words must appear next to each other in the title or abstract.
    Example: "machine_learning"
    - Use "AND" to indicate that both words must be present in the title or abstract, but not necessarily next to each other.
    Example: "machine AND learning"
    - Use "OR" to indicate that at least one of the words must be present in the title or abstract.
    Example: "machine OR learning"

    **Inputs:**
    - **Research Question:** The original research question provided by the user.
    - **Classification Result:** One of the following categories indicating the nature of the research question: Explicating, Envisioning, Relating, Debating.
    - **Critic (Optional):** Feedback on previous search strings that were generated, indicating issues or areas for improvement.

    **Task:**
    - Generate a list of 3 to 5 PubMed-compatible search strings that effectively capture the essence of the research question based on its classification.
    - If a critic is provided, adjust the search strings to address the feedback and improve their relevance and accuracy.

    **Output:**
    - A list of search strings separated by commas, e.g., "search_string1, search_string2, search_string3"

    ---

    **Examples:**

    ### **Example 1**

    - **Research Question:** "What advancements have been made in renewable energy technologies over the past decade?"
    - **Classification Result:** Explicating
    - **Search Strings:**
    "sustainable_energy AND technological_advancements AND last_10_years", "green_energy_innovations AND renewable_tech_progress", "clean_energy AND technology_evolution AND decade_review", "alternative_energy_sources AND tech_developments AND recent_trends", "renewable_power AND innovation_breakthroughs AND ten_year_summary"

    ### **Example 2**

    - **Research Question:** "How does remote work influence employee productivity and job satisfaction?"
    - **Classification Result:** Relating
    - **Search Strings:**
    "telecommuting AND employee_efficiency AND job_happiness", "remote_employment AND work_performance AND job_morale", "virtual_work AND productivity_levels AND employee_contentment", "home_office AND work_output AND job_satisfaction", "distributed_workforce AND performance_metrics AND employee_wellbeing"

    ### **Example 3**

    - **Research Question:** "How has artificial intelligence transformed the landscape of modern healthcare?"
    - **Classification Result:** Debating
    - **Search Strings:**
    "AI_in_healthcare AND medical_innovation", "artificial_intelligence AND health_services AND transformation", "machine_learning AND healthcare_industry AND technological_change", "intelligent_systems AND modern_medicine AND healthcare_evolution", "AI_applications AND health_sector AND innovation_impact"

    ---

    **Guidelines for Incorporating Examples:**

    1. **Diverse Research Questions:** Ensure that the examples cover a range of classifications (Explicating, Envisioning, Relating, Debating) to demonstrate how search strings vary based on the nature of the research question.

    2. **Use of Synonyms and Related Terms:** Notice how different terms are used to express the same or similar concepts, enhancing the breadth of the search.

    3. **Boolean Operators:** Observe the strategic use of "AND" to combine key concepts, ensuring comprehensive coverage of the research topic.

    4. **Specificity and Relevance:** Each search string is tailored to encapsulate the core elements of the research question while adhering to PubMed's syntax rules.

    By following these examples and guidelines, the AI assistant can generate more nuanced and effective PubMed search queries that go beyond the literal wording of the research questions, capturing related concepts and enhancing the retrieval of relevant information.
    """,
        additional_context: Optional[str] = None
    ):
        """
        Initializes the SearchQueryGenerator with a specific prompt explanation and language model.

        Args:
            llm (ChatOpenAI): An instance of the ChatOpenAI language model.
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            additional_context (Optional[str]): Any additional context or instructions for the LLM.
        """
        self.text_generator = TextGenerator(
            prompt_explanation=prompt_explanation,
            llm=llm,
            additional_context=additional_context
        )
    
    def __call__(
        self, 
        research_question: str, 
        classification_result: str, 
        critic: Optional[str] = None
    ) -> List[str]:
        """
        Generates a list of PubMed-compatible search queries based on the research question, 
        its classification, and optional critic feedback.

        Args:
            research_question (str): The original research question.
            classification_result (str): The classification of the research question (Explicating, Envisioning, Relating, Debating).
            critic (Optional[str]): Feedback on previous search strings to refine the generation.

        Returns:
            List[str]: A list of generated PubMed search strings.
        """
        # Construct the input prompt for the TextGenerator
        if critic:
            input_text = f"""
            Research Question: "{research_question}"
            Classification: {classification_result}
            Critic: "{critic}"

            Please generate a list of 3 to 5 PubMed-compatible search strings based on the above information.
            """
        else:
            input_text = f"""
            Research Question: "{research_question}"
            Classification: {classification_result}

            Please generate a list of 3 to 5 PubMed-compatible search strings based on the above information.
            """
        
        # Generate the raw search strings text
        raw_search_strings = self.text_generator.generate(input_text)
        
        # Process the raw text to extract individual search strings
        # Assuming the model outputs search strings separated by commas
        search_strings = [s.strip() for s in raw_search_strings.split(',') if s.strip()]
        
        return search_strings

import os
import subprocess
from typing import Optional, Dict
from dotenv import load_dotenv
from openai import OpenAI
from langchain.prompts import PromptTemplate
from agents.TextGenerator import TextGenerator
import shutil

# Assuming latex_paper_prompt is defined in step6.prompts
from step6.prompts import latex_paper_prompt


class LaTeXPaperGenerator(TextGenerator):
    def __init__(
        self,
        llm: str = "gpt-4o-mini",  # possible options: "gpt-4o-mini", "gpt-4", "gpt-4o", "o1-mini"
        prompt_explanation: str = latex_paper_prompt,
        temperature: float = 1.0,
        max_tokens: Optional[int] = 15000,  # Increased max tokens for detailed LaTeX generation
        start_image_token: str = "<START_IMAGE>",
        stop_image_token: str = "<STOP_IMAGE>",
        start_image_description: str = "<START_IMAGE_DESCRIPTION>",
        stop_image_description: str = "<STOP_IMAGE_DESCRIPTION>",
        start_latex_token: str = "<START_LATEX>",
        stop_latex_token: str = "<STOP_LATEX>",
    ):
        """
        Initializes the LaTeXPaperGenerator with specific prompt explanations, language model,
        and tokens to encapsulate LaTeX code and image descriptions.

        Args:
            llm (str): The language model to use.
            prompt_explanation (str): A detailed explanation of the task for the LLM.
            temperature (float): The temperature parameter for model output creativity.
            max_tokens (Optional[int]): The maximum number of tokens for the generated output.
            start_image_token (str): Token indicating the start of an image.
            stop_image_token (str): Token indicating the end of an image.
            start_image_description (str): Token indicating the start of an image description.
            stop_image_description (str): Token indicating the end of an image description.
            start_latex_token (str): Token indicating the start of LaTeX code.
            stop_latex_token (str): Token indicating the end of LaTeX code.
        """
        self.start_image_token = start_image_token
        self.stop_image_token = stop_image_token
        self.start_image_description = start_image_description
        self.stop_image_description = stop_image_description
        self.start_latex_token = start_latex_token
        self.stop_latex_token = stop_latex_token

        # Format the prompt_explanation with the provided tokens
        formatted_prompt_explanation = prompt_explanation.format(
            start_image_token=start_image_token,
            stop_image_token=stop_image_token,
            start_image_description=start_image_description,
            stop_image_description=stop_image_description,
            start_latex_token=start_latex_token,
            stop_latex_token=stop_latex_token,
        )

        super().__init__(
            prompt_explanation=formatted_prompt_explanation,
            llm=llm,
            temperature=temperature,
            max_tokens=max_tokens,
            start_answer_token=start_latex_token,
            stop_answer_token=stop_latex_token,
        )

    def __call__(
        self,
        analysis_results: str,
        critique: Optional[str] = None,
    ) -> Optional[Dict[str, str]]:
        """
        Generates a LaTeX-formatted paper based on the analysis results, compiles it to PDF,
        and returns the path to the generated PDF.

        Args:
            analysis_results (str): The analysis results to be included in the paper.
            critique (Optional[str]): Optional feedback to refine the LaTeX generation based on previous attempts.

        Returns:
            Optional[Dict[str, str]]: A dictionary containing the PDF path if successful, otherwise None.
        """
        # Construct the input prompt for the LaTeXPaperGenerator
        input_text = f"""
        Analysis Results:
        {analysis_results}
        """

        # Generate the LaTeX code with optional critique
        raw_response = self.generate(input_text, critique=critique)

        response = ""
        for chunk in raw_response:
            try:
                token = chunk.choices[0].delta.content
            except Exception:
                pass
            if token is None:
                continue
            
            response += token

        raw_response = response

        # Convert the streaming response to a complete string if necessary
        if isinstance(raw_response, list):
            # If the response is a list of chunks (streaming), concatenate them
            generated_response = ''.join([chunk.get('choices', [{}])[0].get('delta', {}).get('content', '') for chunk in raw_response])
        else:
            # If the response is a single string
            generated_response = raw_response

        # Extract the LaTeX code between the start and stop tokens
        latex_code = self._extract_between_tokens(
            generated_response,
            self.start_answer_token,
            self.stop_answer_token
        )

        if not latex_code:
            print("LaTeX tokens not found in the response.")
            return None

        pdf_name = "generated_paper"
        temp_dir = "temp"

        # Ensure the temp directory exists
        os.makedirs(temp_dir, exist_ok=True)

        # Define file paths
        tex_filename = os.path.join(temp_dir, f"{pdf_name}.tex")
        pdf_filename = os.path.join(temp_dir, f"{pdf_name}.pdf")

        # Save the LaTeX code to a .tex file
        with open(tex_filename, "w", encoding="utf-8") as tex_file:
            tex_file.write(latex_code)

        # Compile the LaTeX code to PDF using pdflatex
        pdflatex_path = self._get_pdflatex_path()

        if not pdflatex_path:
            print("pdflatex executable not found.")
            return None

        try:
            subprocess.run(
                [
                    pdflatex_path,
                    "-interaction=nonstopmode",
                    "-output-directory",
                    temp_dir,
                    tex_filename,
                ],
                check=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error compiling LaTeX document: {e.stderr.decode('utf-8')}")
            return None

        # Verify that the PDF was created
        if not os.path.isfile(pdf_filename):
            print("PDF compilation failed; PDF file not found.")
            return None

        return {
            "PDF Path": pdf_filename,
        }

    def _extract_between_tokens(self, text: str, start_token: str, stop_token: str) -> str:
        """
        Extracts the substring between start_token and stop_token.

        Args:
            text (str): The complete text containing the tokens.
            start_token (str): The token indicating the start of the desired substring.
            stop_token (str): The token indicating the end of the desired substring.

        Returns:
            str: The extracted substring, or an empty string if tokens are not found.
        """
        start_index = text.find(start_token)
        stop_index = text.find(stop_token, start_index + len(start_token))

        if start_index != -1 and stop_index != -1:
            return text[start_index + len(start_token):stop_index].strip()
        return ""

    def _get_pdflatex_path(self) -> Optional[str]:
        """
        Retrieves the path to the pdflatex executable. Modify this method based on your system's configuration.

        Returns:
            Optional[str]: The path to pdflatex if found, otherwise None.
        """
        # Common paths for pdflatex on different operating systems
        possible_paths = [
            # Windows (modify according to your installation)
            "C:\\Program Files\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe",
            "C:\\Users\\Lukas\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe",
            # macOS/Linux (assuming pdflatex is in PATH)
            "pdflatex",
        ]

        for path in possible_paths:
            if os.path.isfile(path) and os.access(path, os.X_OK):
                return path

        # If pdflatex is in PATH (for macOS/Linux)
        if shutil.which("pdflatex"):
            return "pdflatex"

        return None

from step6.prompts import latex_paper_prompt
from langchain_community.chat_models import ChatOpenAI
from agents.TextGenerator import TextGenerator
import os
import subprocess


class LaTeXPaperGenerator(TextGenerator):
    def __init__(
        self,
        llm: ChatOpenAI,
        prompt_explanation: str = latex_paper_prompt,
        start_image_token: str = "<START_IMAGE>",
        stop_image_token: str = "<STOP_IMAGE>",
        start_image_description: str = "<START_IMAGE_DESCRIPTION>",
        stop_image_description: str = "<STOP_IMAGE_DESCRIPTION>",
    ):
        self.start_image_token = start_image_token
        self.stop_image_token = stop_image_token
        self.start_image_description = start_image_description
        self.stop_image_description = stop_image_description

        super().__init__(
            prompt_explanation=prompt_explanation,
            llm=llm,
        )

    def __call__(self, analysis_results: str) -> str:

        pdf_name = "clr"
        # Ensure the temp directory exists
        if not os.path.exists("temp"):
            os.makedirs("temp")

        # Construct the input prompt for the LaTeXPaperGenerator
        input_text = f"""
        Analysis Results:
        {analysis_results}
        """

        # Generate the LaTeX code
        latex_code = self.generate(input_text)

        # Save the LaTeX code to a .tex file
        tex_filename = f"temp/{pdf_name}.tex"
        with open(tex_filename, "w", encoding="utf-8") as tex_file:
            tex_file.write(latex_code)

        # Compile the LaTeX code to PDF using pdflatex
        cwd = os.getcwd()
        temp_dir = os.path.join(cwd, "temp")
        try:
            subprocess.run(
                [
                    "C:\\Users\\Lukas\\AppData\\Local\\Programs\\MiKTeX\\miktex\\bin\\x64\\pdflatex.exe",
                    "-interaction=nonstopmode",
                    "-output-directory",
                    temp_dir,
                    tex_filename,
                ],
                check=True,
            )
        except subprocess.CalledProcessError as e:
            print(f"Error compiling LaTeX document: {e}")
            return None

        # The PDF should be saved as temp/<pdf_name>.pdf
        pdf_path = f"temp/{pdf_name}.pdf"

        return {
            "PDF Path": pdf_path,
        }


latex_paper_prompt = """
You will be presented with the analysis results of a research question.

Your task is to write a LaTeX article document in the style of a scientific paper based on these results.

Include appropriate sections such as Abstract, Introduction, Methodology, Results, Discussion, Conclusion.

Make sure to use proper LaTeX formatting and include any necessary packages in the preamble.

If the analysis results contain images indicated by special tokens (e.g., {start_image_token}path_to_image{stop_image_token}), include the images in the LaTeX document using the \\includegraphics command.

Ensure that you properly handle any image descriptions provided between {start_image_description} and {stop_image_description} tokens, and include them as figure captions.

Return only the LaTeX code, and do not include any explanations.

Do not include references or acnowledgments!

make sure that images fit on a page, by starting a new page before inserting the 
image. 

The linespacing should be 1.5

"""
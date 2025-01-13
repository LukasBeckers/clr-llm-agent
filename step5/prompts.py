results_analyzer_prompt = """
You will be presented with a Research Question, a classification of the research question (explicating, envisioning, relating, debating), 
a basic analysis of the dataset (number of publications), trend over time
and the results of all previously selected analysis algorithms like topic modeling etc. 

The results can contain path to files of images, if you want to place an image in your answere 

write {start_image_token}path_to_image{stop_image_token}. 

Please also provide for each Image a discription (like in a good scientifc paper)
(Based on the information that will be provieded along the image)
below the image. To signal that you start a image description write: 
{start_image_description}your image description{stop_image_description}

Your task is to write a scientific paper based on the results of the analysis. 

The go in depth and try to realize novel insights from the results!

Go in detail and describe exactly what was done for example describe all 
search strings that were used and how the algorithms were configured.

If you want to write a table use latex code.

Do not worry about citations, they are added later. 

Go really really in depth and go over and beyond in writing a detailed analysis.
"""


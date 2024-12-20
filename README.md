# Automation of Computational Literaturereviews (CLR)


## The 6 Steps of a CLR
| **Step** | **Step Title**                  | **Description**                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               |
|----------|---------------------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **1**    | **Begin with a Conceptual Goal** | - **Identify CLR Objectives:** Determine the primary aim of the review (e.g., explicating, envisioning, relating, debating).<br>- **Align Goals with Research Needs:** Ensure the goals advance understanding beyond descriptive summaries.<br>- **Balance Feasibility and Utility:** Consider technical feasibility and the usefulness of the outcomes.<br>- **Iterative Goal Setting:** Be prepared to refine goals based on initial findings and insights.                           |
| **2**    | **Operationalize the CLR**        | - **Define the Knowledge Domain:** Specify the scope, including keywords, research fields, timeframes, and journal sources.<br>- **Compile the Text Corpus:** Collect relevant articles from comprehensive databases like Web of Science or Scopus.<br>- **Apply Scoping Rules:** Develop and iterate search strings to balance breadth and specificity.<br>- **Manual Screening:** Ensure the corpus excludes unsuitable texts (e.g., retracted articles, predatory journals).<br>- **Extract Meta-Data:** Gather titles, authors, journals, and citation counts for reproducibility. |
| **3**    | **Choose a Computational Technique** | - **Match Techniques to Goals:** Select algorithms that align with the CLR’s conceptual objectives (e.g., ATC for explicating, LDA for envisioning).<br>- **Assess Accessibility:** Choose methods compatible with the research team's technical skills and available resources.<br>- **Select Algorithm Types:** Consider supervised, unsupervised, or dictionary-based techniques based on review goals.<br>- **Understand Trade-Offs:** Balance control and customizability versus ease of use, especially for novice users.                                |
| **4**    | **Perform the Content Analysis**   | - **Data Preprocessing:** Convert texts to machine-readable formats, normalize text (e.g., lowercasing, removing stopwords), and handle compound words using n-grams.<br>- **Calibrate Algorithms:** Set appropriate parameters for chosen algorithms (e.g., number of topics in LDA).<br>- **Execute Analysis:** Run the computational techniques to generate outputs like topic distributions or semantic networks.<br>- **Validate Outputs:** Check for accuracy and plausibility, often requiring manual review by domain experts.<br>- **Enhance Outputs:** Use visualization and additional analyses to deepen insights. |
| **5**    | **Generate Original Insights**     | - **Interpret Results:** Apply domain knowledge to make sense of algorithmic outputs.<br>- **Align with Goals:** Ensure insights support the initial conceptual objectives.<br>- **Iterative Refinement:** Continuously refine analyses and revisit earlier steps as needed to uncover meaningful patterns.<br>- **Identify Knowledge Gaps:** Highlight areas for future research or theoretical development.<br>- **Collaborate Across Teams:** Work with interdisciplinary teams to enhance interpretation and validity.                                           |
| **6**    | **Present the Findings**           | - **Choose a Synthesis Form:** Select how to present results (e.g., research agenda, taxonomy, alternative models, meta-theory).<br>- **Communicate Clearly:** Use visualizations and structured narratives to make findings accessible.<br>- **Link to Objectives:** Clearly state how the findings address the CLR’s conceptual goals.<br>- **Leverage for Theory Building:** Integrate CLR insights with existing theories or develop new theoretical frameworks.<br>- **Document Methodology:** Provide a transparent account of the CLR process for replicability.           |



# Step 1

### Begin with a conceptual goal.


In this work everything will begin with a research question.

This research question will be classified into Explicating, Envisioning, Relating, Debating


# Step 2

A LLM Agent will generate search_strings based on the research question and 
the classification result. 

The dataset will be downloaded automatically, duplicates will be dropped and 
another LLM Agent will analyze the text corpus and 


# Step 3

Unsupervised Techniques

    Latent Dirichlet Allocation (LDA)
    Latent Semantic Indexing/Analysis (LSI/LSA)
    Probabilistic Latent Semantic Analysis (PLSA)
    Non-negative Matrix Factorization (NMF)
    Correlated Topic Models
    Hierarchical Dirichlet Processes (HDP)
    Computer-Assisted Clustering (CAC)

Dictionary-Based Techniques

    Linguistic Inquiry and Word Count (LIWC)
    Coh-Metrix


Latent Dirichlet Allocation (LDA)
Latent Semantic Indexing/Analysis (LSI/LSA)
Probabilistic Latent Semantic Analysis (PLSA)
Non-negative Matrix Factorization (NMF)
Hierarchical Dirichlet Processes (HDP)
Correlated Topic Models
Computer-Assisted Clustering (CAC)
Leximancer
Linguistic Inquiry and Word Count (LIWC)
Coh-Metrix
N-grams
Network Analysis
Regression Analysis
Heatmaps
Automated Frame Analysis (AFA)
Automated Narrative Analysis (ANA)
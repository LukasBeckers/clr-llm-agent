hyperparamter_selection_prompts = {
    "CorrelatedTopicModel": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the CorrelatedTopicModel (CTM) algorithm based on the provided research question, its classification, and the dataset analysis.

The CorrelatedTopicModel has the following hyperparameters:
- **num_topics** (int): The number of topics to extract. Possible values: 5 to 50.
- **alpha** (str): Hyperparameter for topic distribution. Possible values: "symmetric", "asymmetric".
- **eta** (str): Hyperparameter for word distribution. Possible values: "auto", "fixed".
- **seed** (int): Seed for reproducibility. Any integer value.
- **iterations** (int): Number of iterations to perform.
- **top_n** (int): Number of results to show.

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **alpha**: Controls the sparsity of the topic distribution.
- **eta**: Controls the sparsity of the word distribution within topics.
- **seed**: Ensures reproducibility of results.
- **iterations**: Potentially increase fit at the cost of higher runtime. 
- **top_n**: Lower = more concise result, higher = wider result with more info

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace "value" with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "alpha": "value",
        "eta": "value",
        "seed": value,
        "iterations": value,
        "top_n": value
    }
}
<STOP_HYPERPARAMETERS>
""",

    "DynamicTopicModeling": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the DynamicTopicModeling algorithm based on the provided research question, its classification, and the dataset analysis.

The DynamicTopicModeling has the following hyperparameters:

- **k** (int): Number of topics to extract. Possible values: 1 to 32767.
- **t** (int): Number of timepoints. Depends on the dataset's temporal granularity.
- **alpha_var** (float): Transition variance of alpha (per-document topic distribution). Possible values: 0.01 to 1.0.
- **eta_var** (float): Variance of eta (topic distribution of each document) from its alpha. Possible values: 0.01 to 1.0.
- **phi_var** (float): Transition variance of phi (word distribution of each topic). Possible values: 0.01 to 1.0.
- **lr_a** (float): Shape parameter 'a' for SGLD step size. Must be greater than 0.
- **lr_b** (float): Shape parameter 'b' for SGLD step size. Must be greater than or equal to 0.
- **lr_c** (float): Shape parameter 'c' for SGLD step size. Range: (0.5, 1].
- **iter** (int): Number of iterations of Gibbs-sampling. Possible values: 1000 to 10000.
- **seed** (int): Seed for reproducibility. Any integer value.
- **tw** (str): Term weighting scheme. Possible values: "one", "idf", "pmi", "dbi".
- **min_cf** (int): Minimum collection frequency of words. Possible values: 0 to 10.
- **min_df** (int): Minimum document frequency of words. Possible values: 0 to 10.
- **rm_top** (int): Number of top words to remove. Possible values: 0 to 50.

Each hyperparameter influences the model as follows:

- **k**: Determines the granularity of topic extraction.
- **t**: Defines the number of timepoints in the data.
- **alpha_var**: Controls how much the per-document topic distribution can change over time.
- **eta_var**: Controls the variance of topic distributions from alpha.
- **phi_var**: Controls how much the word distribution of each topic can change over time.
- **lr_a**, **lr_b**, **lr_c**: Parameters for SGLD step size, affecting convergence speed and stability.
- **iter**: Determines the number of sampling iterations, affecting convergence.
- **seed**: Ensures reproducibility of results.
- **tw**: Affects the term weighting scheme used in the model.
- **min_cf**: Excludes words with low collection frequency, reducing noise.
- **min_df**: Excludes words appearing in few documents, reducing noise.
- **rm_top**: Removes too common words from the model.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace `"value"` with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "k": value,
        "t": value,
        "alpha_var": value,
        "eta_var": value,
        "phi_var": value,
        "lr_a": value,
        "lr_b": value,
        "lr_c": value,
        "iter": value,
        "seed": value,
        "tw": "value",
        "min_cf": value,
        "min_df": value,
        "rm_top": value
    }
}
<STOP_HYPERPARAMETERS>
"""
,

    "HierarchicalDirichletProcess": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the HierarchicalDirichletProcess (HDP) algorithm based on the provided research question, its classification, and the dataset analysis.

The HierarchicalDirichletProcess has the following hyperparameters:
- **random_state** (int): Seed for reproducibility. Any integer value.
- **top_n** (int): Number of top words per topic to retrieve. Possible values: 5 to 20.

Each hyperparameter influences the model as follows:
- **random_state**: Ensures reproducibility of results.
- **top_n**: Determines the number of top words displayed for each topic.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace "value" with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "random_state": value,
        "top_n": value
    }
}
<STOP_HYPERPARAMETERS>
""",

   "LatentDirichletAllocation": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the LatentDirichletAllocation (LDA) algorithm based on the provided research question, its classification, and the dataset analysis.

The LatentDirichletAllocation has the following hyperparameters:

- **k** (int): Number of topics to extract. Possible values: 1 to 32767.
- **alpha** (float or list of floats): Hyperparameter of Dirichlet distribution for document-topic distribution. If float, symmetric prior; if list, asymmetric prior with length k.
- **eta** (float): Hyperparameter of Dirichlet distribution for topic-word distribution. Possible values: 0.001 to 0.1.
- **iter** (int): Number of iterations of Gibbs-sampling. Possible values: 1000 to 10000.
- **seed** (int): Seed for reproducibility. Any integer value.
- **tw** (str): Term weighting scheme. Possible values: "one", "idf", "pmi", "dbi".
- **min_cf** (int): Minimum collection frequency of words. Possible values: 0 to 10.
- **min_df** (int): Minimum document frequency of words. Possible values: 0 to 10.
- **rm_top** (int): Number of top words to remove. Possible values: 0 to 50.

Each hyperparameter influences the model as follows:

- **k**: Determines the granularity of topic extraction.
- **alpha**: Controls the sparsity of the document-topic distribution.
- **eta**: Controls the sparsity of the topic-word distribution.
- **iter**: Determines the number of sampling iterations, affecting convergence.
- **seed**: Ensures reproducibility of results.
- **tw**: Affects the term weighting scheme used in the model.
- **min_cf**: Excludes words with low collection frequency, reducing noise.
- **min_df**: Excludes words appearing in few documents, reducing noise.
- **rm_top**: Removes too common words from the model.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace `"value"` with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "k": value,
        "alpha": value,
        "eta": value,
        "iter": value,
        "seed": value,
        "tw": "value",
        "min_cf": value,
        "min_df": value,
        "rm_top": value
    }
}
<STOP_HYPERPARAMETERS>
"""
,

    "LatentSemanticIndexing": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the LatentSemanticIndexing (LSI) algorithm based on the provided research question, its classification, and the dataset analysis.

The LatentSemanticIndexing has the following hyperparameters:
- **num_topics** (int): The number of topics to extract. Possible values: 5 to 50.
- **num_words** (int): Number of top words per topic. Possible values: 5 to 20.
- **random_state** (int): Seed for reproducibility. Any integer value.
- **chunksize** (int): Number of documents per training chunk. Possible values: 500 to 2000.

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **num_words**: Specifies the number of top words displayed for each topic.
- **random_state**: Ensures reproducibility of results.
- **chunksize**: Manages memory usage and training speed.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace "value" with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "num_words": value,
        "random_state": value,
        "chunksize": value
    }
}
<STOP_HYPERPARAMETERS>
""",

    "LIWC": """
You are an expert in configuring linguistic and readability analysis tools. Your task is to suggest optimal hyperparameters for the LIWC (Linguistic Inquiry and Word Count) analyzer based on the provided research question, its classification, and the dataset analysis.

The LIWC has the following hyperparameters:
- **dictionary** (Dict[str, List[str]]): Mapping of categories to lists of words. Possible values: Custom dictionaries or default sample dictionary.

Each hyperparameter influences the model as follows:
- **dictionary**: Determines the categories and corresponding words used for analysis.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace the categories and words with your suggested values.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "dictionary": {
            "category1": ["word1", "word2", "..."],
            "category2": ["word1", "word2", "..."],
            "..."
        }
    }
}
<STOP_HYPERPARAMETERS>

""",

    "NonNegativeMatrixFactorization": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the NonNegativeMatrixFactorization (NMF) algorithm based on the provided research question, its classification, and the dataset analysis.

The NonNegativeMatrixFactorization has the following hyperparameters:
- **num_topics** (int): Number of topics to extract. Possible values: 5 to 50.
- **random_state** (int): Seed for reproducibility. Any integer value.
- **max_iter** (int): Maximum number of iterations during training. Possible values: 100 to 500.
- **top_n** (int): Number of top words per topic. Possible values: 5 to 20.

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **random_state**: Ensures reproducibility of results.
- **max_iter**: Sets the limit on training iterations.
- **top_n**: Specifies the number of top words displayed for each topic.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace "value" with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "random_state": value,
        "max_iter": value,
        "top_n": value
    }
}
<STOP_HYPERPARAMETERS>
""",

    "ProbabilisticLatentSemanticAnalysis": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the ProbabilisticLatentSemanticAnalysis (PLSA) algorithm based on the provided research question, its classification, and the dataset analysis.

The ProbabilisticLatentSemanticAnalysis has the following hyperparameters:
- **num_topics** (int): Number of topics to extract. Possible values: 5 to 50.
- **passes** (int): Number of passes through the corpus during training. Possible values: 5 to 20.
- **random_state** (int): Seed for reproducibility. Any integer value.
- **alpha** (str): Controls the sparsity of the document-topic distribution. Possible values: "symmetric", "asymmetric".

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **passes**: Affects how thoroughly the model learns from the corpus.
- **random_state**: Ensures reproducibility of results.
- **alpha**: Controls the sparsity of the document-topic distribution.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters.

Your response should output only the JSON content between <START_HYPERPARAMETERS> and <STOP_HYPERPARAMETERS>, without any code blocks or additional formatting. Do not include any explanations or descriptions. Here is the exact format you should use:

Replace "value" with your suggested values for each hyperparameter.

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "passes": value,
        "random_state": value,
        "alpha": "value"
    }
}
<STOP_HYPERPARAMETERS>
""",
}

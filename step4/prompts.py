# hyperparameter_selection_prompts.py

hyperparamter_selection_prompts = {
    "CorrelatedTopicModel": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the CorrelatedTopicModel (CTM) algorithm based on the provided research question, its classification, and the dataset analysis.

The CorrelatedTopicModel has the following hyperparameters:
- **num_topics** (int): The number of topics to extract. Possible values: 5 to 50.
- **alpha** (str): Hyperparameter for topic distribution. Possible values: "symmetric", "asymmetric".
- **eta** (str): Hyperparameter for word distribution. Possible values: "auto", "fixed".
- **seed** (int): Seed for reproducibility. Any integer value.

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **alpha**: Controls the sparsity of the topic distribution.
- **eta**: Controls the sparsity of the word distribution within topics.
- **seed**: Ensures reproducibility of results.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "alpha": value,
        "eta": value,
        "seed": value
    }
}
<STOP_HYPERPARAMETERS>
""",

    "DynamicTopicModeling": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the DynamicTopicModeling algorithm based on the provided research question, its classification, and the dataset analysis.

The DynamicTopicModeling has the following hyperparameters:
- **num_topics** (int): Number of topics to extract. Possible values: 5 to 50.
- **time_key** (str): The key in each document that indicates the time slice. Typically "time".
- **passes** (int): Number of passes through the corpus during training. Possible values: 1 to 10.
- **iterations** (int): Maximum number of iterations during training. Possible values: 100 to 1000.
- **alpha** (Any): Controls sparsity of document-topic distribution. Possible values: "symmetric", "asymmetric".
- **eta** (Any): Controls sparsity of topic-word distribution. Possible values: "auto", "fixed".
- **decay** (float): Exponential decay rate for old documents. Possible values: 0.1 to 0.9.
- **offset** (float): Controls how fast the exponential decay rates the weights. Possible values: 0.5 to 2.0.
- **random_state** (int): Seed for reproducibility. Any integer value.
- **chunksize** (int): Number of documents per training chunk. Possible values: 500 to 5000.
- **evaluate_every** (int): Frequency of evaluation metrics. Possible values: 0 (disable) or positive integers.
- **gamma_threshold** (float): Convergence threshold for variational EM. Possible values: 0.0001 to 0.01.

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **time_key**: Specifies the document attribute used for time slicing.
- **passes**: Affects how thoroughly the model learns from the corpus.
- **iterations**: Determines the depth of training iterations.
- **alpha**: Controls the sparsity of the document-topic distribution.
- **eta**: Controls the sparsity of the topic-word distribution.
- **decay**: Balances the influence of older documents.
- **offset**: Adjusts the rate at which decay affects document weights.
- **random_state**: Ensures reproducibility of results.
- **chunksize**: Manages memory usage and training speed.
- **evaluate_every**: Determines how often evaluation metrics are computed.
- **gamma_threshold**: Sets the threshold for convergence in the EM algorithm.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "time_key": value,
        "passes": value,
        "iterations": value,
        "alpha": value,
        "eta": value,
        "decay": value,
        "offset": value,
        "random_state": value,
        "chunksize": value,
        "evaluate_every": value,
        "gamma_threshold": value
    }
}
<STOP_HYPERPARAMETERS>
""",

    "HierarchicalDirichletProcess": """
You are an expert in configuring machine learning models. Your task is to suggest optimal hyperparameters for the HierarchicalDirichletProcess (HDP) algorithm based on the provided research question, its classification, and the dataset analysis.

The HierarchicalDirichletProcess has the following hyperparameters:
- **random_state** (int): Seed for reproducibility. Any integer value.
- **top_n** (int): Number of top words per topic to retrieve. Possible values: 5 to 20.

Each hyperparameter influences the model as follows:
- **random_state**: Ensures reproducibility of results.
- **top_n**: Determines the number of top words displayed for each topic.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

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
- **num_topics** (int): The number of topics to extract. Possible values: 5 to 50.
- **passes** (int): Number of passes through the corpus during training. Possible values: 1 to 10.
- **iterations** (int): Maximum number of iterations during training. Possible values: 50 to 500.
- **alpha** (Any): Controls the sparsity of the document-topic distribution. Possible values: "symmetric", "asymmetric".
- **beta** (Any): Controls the sparsity of the topic-word distribution. Possible values: "auto", "fixed".
- **random_state** (int): Seed for reproducibility. Any integer value.
- **chunksize** (int): Number of documents per training chunk. Possible values: 500 to 2000.

Each hyperparameter influences the model as follows:
- **num_topics**: Determines the granularity of topic extraction.
- **passes**: Affects how thoroughly the model learns from the corpus.
- **iterations**: Determines the depth of training iterations.
- **alpha**: Controls the sparsity of the document-topic distribution.
- **beta**: Controls the sparsity of the topic-word distribution.
- **random_state**: Ensures reproducibility of results.
- **chunksize**: Manages memory usage and training speed.

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "passes": value,
        "iterations": value,
        "alpha": value,
        "beta": value,
        "random_state": value,
        "chunksize": value
    }
}
<STOP_HYPERPARAMETERS>
""",

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

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

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

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "dictionary": {
            "category1": ["word1", "word2", ...],
            "category2": ["word1", "word2", ...],
            ...
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

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

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

Please analyze the research question, its classification, and the dataset characteristics to suggest appropriate values for these hyperparameters. After your reasoning, output the hyperparameters in JSON format as specified.

---

Research Question: "{research_question}"
Research Question Classification: {research_question_class}
Dataset Basic Evaluation: {basic_dataset_evaluation}

---

Your response should follow this structure:

<START_HYPERPARAMETERS>
{
    "hyper_parameters": {
        "num_topics": value,
        "passes": value,
        "random_state": value,
        "alpha": value
    }
}
<STOP_HYPERPARAMETERS>
""",
}

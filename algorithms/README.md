# Algorithms

Welcome to the Algorithms directory of the Automated Computational Literature Review project. This folder encompasses a collection of robust algorithms designed to facilitate comprehensive literature analysis through automated means.

## Overview

Each algorithm within this directory is meticulously organized into its own Python file, ensuring clarity and ease of maintenance. The structure adheres to a consistent naming convention and interface design, promoting seamless integration and usability across the project.
Directory Structure

```
/algorithms
├── CohMetrix.py
├── ComputerAssistedClustering.py
├── CorrelatedTopicModel.py
├── DynamicTopicModeling.py
├── HierarchicalDirichletProcess.py
├── LatentDirichletAllocation.py
├── LatentSemanticIndexing.py
├── ProbabilisticLatentSemanticAnalysis.py
└── NonNegativeMatrixFactorization.py
```

One Class per File: Each Python file contains exactly one class. The class name matches the filename, ensuring intuitive navigation and usage. For example, CohMetrix.py contains the CohMetrix class.

Class Design
Initialization (**init**)

All hyperparameters are configured within the **init** method of each class. This design choice centralizes configuration, allowing users to customize algorithm behavior upon instantiation.

Example:

```
class CohMetrix:
    def __init__(self, param1=default1, param2=default2, ...):
        """
        Initializes the CohMetrix analyzer with specified hyperparameters.

        :param param1: Description of param1
        :param param2: Description of param2
        """
        self.param1 = param1
        self.param2 = param2
        # Initialize other components or models here

```

### Callable Interface

Each algorithm class implements a \_\_call\_\_ method, enabling instances to be invoked as functions. This method accepts only the documents as input and returns the results as a dictionary, encapsulating all relevant outputs.

Method Signature:
```
def __call__(self, documents: List[List[str]]) -> Dict[str, Any]:
```
"""
Executes the algorithm on the provided documents.

    :param documents: A list of documents, each represented as a list of tokens (strings).
    :return: A dictionary containing the algorithm's results.
    """

Key Features:

    Input:
        documents: A list of preprocessed documents, where each document is a list of tokens (strings).

    Output:
        A dictionary containing all pertinent results generated by the algorithm. The structure of this dictionary varies depending on the specific algorithm but generally includes metrics, model parameters, and other relevant data.

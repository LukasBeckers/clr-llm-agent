from algorithms.CohMetrix import CohMetrix
from algorithms.ComputerAssistedClustering import ComputerAssistedClustering
from algorithms.CorrelatedTopicModel import CorrelatedTopicModel
from algorithms.DynamicTopicModeling import DynamicTopicModeling
from algorithms.HierarchicalDirichletProcess import (
    HierarchicalDirichletProcess,
)
from algorithms.LatentDirichletAllocation import LatentDirichletAllocation
from algorithms.LatentSemanticIndexing import LatentSemanticIndexing
from algorithms.LIWC import LIWC
from algorithms.NonNegativeMatrixFactorization import (
    NonNegativeMatrixFactorization,
)
from algorithms.ProbabilisticLatentSemanticAnalysis import (
    ProbabilisticLatentSemanticAnalysis,
)


algorithms = {
    "CohMetrix": CohMetrix,
    "ComputerAssistedClustering": ComputerAssistedClustering,
    "CorrelatedTopicModel": CorrelatedTopicModel,
    "DynamicTopicModeling": DynamicTopicModeling,
    "HierarchicalDirichletProcess": HierarchicalDirichletProcess,
    "LatentDirichletAllocation": LatentDirichletAllocation,
    "LatentSemanticIndexing": LatentSemanticIndexing,
    "LIWC": LIWC,
    "NonNegativeMatrixFactorization": NonNegativeMatrixFactorization,
    "ProbabilisticLatentSemanticAnalysis": ProbabilisticLatentSemanticAnalysis,
}

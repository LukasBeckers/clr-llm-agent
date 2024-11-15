from abc import ABC, abstractmethod
from typing import Dict, Union, List, Optional, Any


class IAlgorithm(ABC):
    @property
    @abstractmethod
    def algorithm_description(self) -> str:
        pass

    @abstractmethod
    def __call__(
        self, documents: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        pass

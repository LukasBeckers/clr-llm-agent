from abc import ABC, abstractmethod
from typing import List, Union
from pandas import DataFrame


class IDataSource(ABC):
    @abstractmethod
    def search(self, query: Union[str, List[str]]) -> DataFrame:
        pass

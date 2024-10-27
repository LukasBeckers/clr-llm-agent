from tools.data_sources import data_sources
from typing import Dict, List, Tuple


class DataLoader:
    def __init__(self, data_sources: Dict = data_sources) -> None:
        self.data_sources = data_sources

    def __call__(search_strings: List[Tuple[str, str]]) -> dict[str:str]:
        

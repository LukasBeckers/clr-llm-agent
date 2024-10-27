from tools.data_sources import data_sources
from typing import Dict, List, Tuple


class DataLoader:
    def __init__(self, email: str, data_sources: Dict = data_sources) -> None:
        self.data_sources = data_sources
        self.email = email

    def __call__(self, search_strings: List[Tuple[str, str]]) -> dict[str:str]:
        results = []

        for search_string, data_source_key in search_strings:
            try:
                result = self.data_sources[data_source_key](
                    search_string, self.email
                )
                results.append(result)
            except KeyError:
                pass

        combined_results = []

        # This has to be changed for other datasources to remove duplicates
        for result in results:
            print("Result", len(result))
            for key, value in result.items():
                print("Key", key, type(key))
                print("Value", len(value), type(value))
                for publication in value: 
                    if not publication in combined_results:
                        combined_results.append(publication)

        print("Combined_Results", len(combined_results))

        return combined_results

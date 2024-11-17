from typing import Dict, List, Optional, Any, Union, Iterable, Callable, Tuple


class ResultsParser():
    """
    If results need to be parsed implement it here
    """
    def __init__(self):
        pass

    def __call__(self, results: Dict[str, Union[Dict, str, List]]) -> str: 
        return str(results)
        




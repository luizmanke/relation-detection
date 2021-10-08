import numpy as np
from typing import Callable, Dict, Tuple
from .metaheuristics.harmony_search import HarmonySearch


class BaseMetaheuristic:

    available_methods_ = {
        "harmony_search": HarmonySearch
    }

    def __init__(
            self,
            method_name: str,
            objective_function: Callable,
            features: Dict[str, list]
    ) -> None:
        self.method_ = self.available_methods_[method_name](
            objective_function,
            features
        )

    def get_best_population(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.method_.get_best_population(x, y)

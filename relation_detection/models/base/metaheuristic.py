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
            features: Dict[str, list],
            n_runs: int = 10,
            n_generations: int = 500,
            population_size: int = 200
    ) -> None:
        self.method_ = self.available_methods_[method_name](
            objective_function,
            features,
            n_runs,
            n_generations,
            population_size
        )

    def fit_and_return(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        return self.method_.fit_and_return(x, y)

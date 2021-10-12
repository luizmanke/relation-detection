import numpy as np
from typing import Callable, List, Tuple
from .metaheuristics.harmony_search import HarmonySearch


class BaseMetaheuristic:

    available_methods_ = {
        "harmony_search": HarmonySearch
    }

    def __init__(
            self,
            method_name: str,
            objective_function: Callable,
            n_runs: int = 5,
            n_generations: int = 200,
            population_size: int = 50,
            n_individuals: int = 100
    ) -> None:
        self.method_ = self.available_methods_[method_name](
            objective_function,
            n_runs,
            n_generations,
            population_size,
            n_individuals
        )

    def fit_and_return(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[list], List[list]]:
        return self.method_.fit_and_return(x, y)

import numpy as np
import random
from bisect import bisect_left
from multiprocessing import cpu_count
from pyharmonysearch.harmony_search import harmony_search
from typing import Callable, Dict, Optional, Tuple


class HarmonySearch:

    def __init__(self, objective_function: Callable, features: Dict[str, list]) -> None:
        self.objective_function_ = objective_function
        self.features_ = features
        self.num_processes_ = cpu_count()
        self.num_iterations_ = self.num_processes_ * 5

    def get_best_population(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:

        self.x_ = x
        self.y_ = y

        results = harmony_search(
            ObjectiveClass(self._mock_objective_function, self.features_),
            self.num_processes_,
            self.num_iterations_
        )

        solutions_per_run = []
        score_history_per_run = []
        for run in results.harmony_histories:

            solutions = []
            score_history = []
            for i, generation in enumerate(run[1:]):

                score_of_generation = []
                for solution, score in generation["harmonies"]:
                    score_of_generation.append(score)
                    if i == len(run) - 2:
                        solutions.append(solution)
                score_history.append(score_of_generation)

            score_history_per_run.append(score_history)
            solutions_per_run.append(solutions)

        return np.array(solutions_per_run), np.array(score_history_per_run)

    def _mock_objective_function(self, vector: np.ndarray) -> float:
        return self.objective_function_(self.x_, self.y_, vector)


class ObjectiveClass:

    def __init__(self, objective_function: Callable, features: Dict[str, list]) -> None:
        self.objective_function_ = objective_function
        self.features_ = features
        self._create_value_ranges()

    def get_fitness(self, vector: list) -> float:
        return self.objective_function_(vector)

    def use_random_seed(self) -> bool:
        return False

    def maximize(self) -> bool:
        return True

    def get_hms(self) -> int:
        return 250

    def get_hmcr(self) -> float:
        return 0.75

    def get_max_imp(self) -> int:
        return 50_000

    def get_par(self) -> float:
        return 0.5

    def get_mpai(self) -> int:
        return 5

    def is_variable(self, i: int) -> bool:
        return True

    def is_discrete(self, i: int) -> bool:
        return True

    def get_index(self, i: int, value: int):
        return bisect_left(self._value_ranges[i], value)

    def get_num_discrete_values(self, i: int) -> int:
        return len(self._value_ranges[i])

    def get_num_parameters(self) -> int:
        return len(self._value_ranges)

    def get_value(self, i: int, specific_index: Optional[int] = None) -> str:
        if specific_index is None:
            index = random.randint(0, len(self._value_ranges[i])-1)
        else:
            index = specific_index
        return self._value_ranges[i][index]

    def _create_value_ranges(self):
        self._value_ranges = []
        for _, range in self.features_.items():
            self._value_ranges.append(np.arange(range[0], range[1]+1))

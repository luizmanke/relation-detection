import numpy as np
import random
from bisect import bisect_left
from pyharmonysearch.harmony_search import HarmonySearch as Algorithm
from tqdm import tqdm
from typing import Callable, Dict, List, Optional, Tuple


class HarmonySearch:

    def __init__(
            self,
            objective_function: Callable,
            features: Dict[str, list],
            n_runs: int,
            n_generations: int,
            population_size: int
    ) -> None:
        self.objective_function_ = objective_function
        self.features_ = features
        self.n_runs_ = n_runs
        self.n_generations_ = n_generations
        self.population_size_ = population_size

    def fit_and_return(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        self.x_ = x
        self.y_ = y
        results_per_run = self._fit()
        solutions, history = self._refactor_results(results_per_run)
        return solutions, history

    def _fit(self) -> list:
        results_per_run = []
        for _ in tqdm(range(self.n_runs_)):
            algorithm = Algorithm(
                ObjectiveClass(
                    self._mock_objective_function,
                    self.features_,
                    self.n_generations_,
                    self.population_size_
                )
            )
            _, _, _, history = algorithm.run(self._create_initial_population().tolist())
            results_per_run.append(history)
        return results_per_run

    def _mock_objective_function(self, vector: np.ndarray) -> float:
        return self.objective_function_(self.x_, self.y_, vector)

    def _create_initial_population(self) -> np.ndarray:
        indexes = [random.randint(0, len(self.x_)-1) for _ in range(self.population_size_)]
        return self.x_[indexes]

    def _refactor_results(self, results_per_run: List[list]) -> Tuple[np.ndarray, np.ndarray]:

        solutions_per_run = []
        score_history_per_run = []
        for run in results_per_run:

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


class ObjectiveClass:

    def __init__(
            self,
            objective_function: Callable,
            features: Dict[str, list],
            n_generations: int,
            population_size: int
    ) -> None:
        self.objective_function_ = objective_function
        self.features_ = features
        self.n_generations_ = n_generations
        self.population_size_ = population_size
        self._create_value_ranges()

    def get_fitness(self, vector: list) -> float:
        return self.objective_function_(vector)

    def use_random_seed(self) -> bool:
        return False

    def maximize(self) -> bool:
        return True

    def get_hms(self) -> int:
        return self.population_size_

    def get_hmcr(self) -> float:
        return 0.75

    def get_max_imp(self) -> int:
        return self.n_generations_ * self.population_size_

    def get_par(self) -> float:
        return 0.5

    def get_mpai(self) -> int:
        return 5

    def is_variable(self, i: int) -> bool:
        return True

    def is_discrete(self, i: int) -> bool:
        return True

    def get_index(self, i: int, value: int):
        index = bisect_left(self._value_ranges[i], value)
        if not self._value_ranges[i][index] == value:
            raise Exception("Value not in range.")
        return index

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

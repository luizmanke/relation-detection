import numpy as np
import random
from copy import deepcopy
from multiprocessing import Pool
from pyharmonysearch.harmony_search import HarmonySearch as BaseAlgorithm
from typing import Callable, List, Tuple


class HarmonySearch:

    def __init__(
            self,
            objective_function: Callable,
            n_runs: int,
            n_generations: int,
            population_size: int,
            n_individuals: int
    ) -> None:
        self.objective_function_ = objective_function
        self.n_runs_ = n_runs
        self.n_generations_ = n_generations
        self.population_size_ = population_size
        self.n_individuals_ = n_individuals

    def fit_and_return(self, x: np.ndarray, y: np.ndarray) -> Tuple[List[list], List[list]]:
        self.x_ = x
        self.y_ = y
        self._create_value_ranges(x, y)
        solution, history = self._fit()
        return solution, history

    def _fit(self) -> Tuple[List[list], List[list]]:

        pool = Pool(processes=5)
        results = [pool.apply_async(self._run) for _ in range(self.n_runs_)]
        pool.close()  # no more tasks will be submitted to the pool
        pool.join()  # wait for all tasks to finish before moving on

        best_harmony = []
        best_fitness = float("-inf")
        history_per_run = []
        for result in results:
            harmony, fitness, _, history = result.get()
            if fitness > best_fitness:
                best_fitness = fitness
                best_harmony = harmony
            history_per_run.append(history)

        return best_harmony, self._refactor_history(history_per_run)

    def _run(self):
        algorithm = Algorithm(
            ObjectiveClass(
                self._mock_objective_function,
                self.n_generations_,
                self.population_size_,
                self.value_ranges_
            )
        )
        return algorithm.run(self._create_initial_population())

    def _mock_objective_function(self, vector: np.ndarray) -> float:
        return self.objective_function_(self.x_, self.y_, vector)

    def _create_value_ranges(self, x: np.ndarray, y: np.ndarray) -> None:
        x_positive = x[y == 1]
        x_unique = np.unique(x_positive, axis=0)
        self.value_ranges_ = []
        for _ in range(self.n_individuals_):
            self.value_ranges_.append(x_unique.tolist())

    def _create_initial_population(self) -> List[list]:
        population = []
        for _ in range(self.population_size_):
            population.append([
                self.value_ranges_[i][random.randint(0, len(self.value_ranges_[i])-1)]
                for i in range(self.n_individuals_)
            ])
        return population

    def _refactor_history(self, history_per_run: List[list]) -> List[list]:

        solutions_per_run = []
        for generations in history_per_run:

            solutions_per_generation = []
            for _, generation in enumerate(generations[1:]):

                solutions_per_harmony = []
                for solution, score in generation["harmonies"]:
                    solutions_per_harmony.append(solution)

                solutions_per_generation.append(solutions_per_harmony)
            solutions_per_run.append(solutions_per_generation)

        return solutions_per_run


class Algorithm(BaseAlgorithm):

    def _pitch_adjustment(self, harmony, i):

        harmony[i] = deepcopy(harmony[i])
        indexes = np.arange(0, len(harmony[i]))
        np.random.shuffle(indexes)

        if random.random() < 0.5:

            for index in indexes:
                if harmony[i][index] != 1:  # ANY
                    harmony[i][index] = 1
                    break
        else:

            original_values = np.array(self._obj_fun.value_ranges_[i])
            copy_values = original_values.copy()
            for j, item in enumerate(harmony[i]):
                if item == 1:  # ANY
                    copy_values[:, j] = 1

            filtered_values = original_values[(copy_values == harmony[i]).all(axis=1)]
            np.random.shuffle(filtered_values)

            for index in indexes:
                if harmony[i][index] == 1:  # ANY
                    harmony[i][index] = filtered_values[0][index]
                    break


class ObjectiveClass:

    def __init__(
            self,
            objective_function: Callable,
            n_generations: int,
            population_size: int,
            value_ranges: list
    ) -> None:
        self.objective_function_ = objective_function
        self.n_generations_ = n_generations
        self.population_size_ = population_size
        self.value_ranges_ = value_ranges

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
        return 0.1

    def get_num_parameters(self) -> int:
        return len(self.value_ranges_)

    def get_value(self, i: int) -> int:
        return self.value_ranges_[i][random.randint(0, len(self.value_ranges_[i])-1)]

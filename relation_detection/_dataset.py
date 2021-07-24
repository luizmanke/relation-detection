import numpy as np
from typing import List, Tuple
from .datasets.dbpedia import DBpedia


class Dataset:

    available_sets_ = {
        "dbpedia": DBpedia
    }

    def __init__(self) -> None:
        pass

    def list_available_sets(self) -> List[str]:
        return list(self.available_sets_.keys())

    def load(self, dataset_name: str) -> None:
        assert dataset_name in self.available_sets_
        self.dataset_ = self.available_sets_[dataset_name]()
        self.dataset_.load()

    def get_data(self) -> Tuple[List[dict], np.ndarray, List[str]]:
        return self.dataset_.get_data()

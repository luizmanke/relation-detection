import numpy as np
from typing import List, Optional, Tuple
from .datasets.dbpedia import DBpedia
from .datasets.news import News


class Dataset:

    available_sets_ = {
        "dbpedia": DBpedia,
        "news": News
    }

    def __init__(self, dataset_name: str, file_path: Optional[str] = None) -> None:
        self._assert_dataset_name(dataset_name)
        self.dataset_ = self.available_sets_[dataset_name](file_path)
        self.dataset_.load()  # type: ignore

    def get_data(self) -> Tuple[List[dict], np.ndarray, List[str]]:
        return self.dataset_.get_data()  # type: ignore

    def _assert_dataset_name(self, dataset_name:str) -> None:
        if dataset_name not in self.available_sets_:
            raise Exception(f"Dataset name not in {self.available_sets_.keys()}")

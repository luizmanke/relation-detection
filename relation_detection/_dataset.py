import numpy as np
from typing import List, Optional, Tuple
from .datasets.dbpedia import DBpedia
from .datasets.news import News


class Dataset:

    available_sets_ = {
        "dbpedia": DBpedia,
        "news": News
    }

    def __init__(self) -> None:
        pass

    def list_available_sets(self) -> List[str]:
        return list(self.available_sets_.keys())

    def load(self, dataset_name: str, file_path: Optional[str] = None) -> None:
        assert dataset_name in self.available_sets_
        self.dataset_ = self.available_sets_[dataset_name](file_path)
        self.dataset_.load()  # type: ignore

    def get_data(self) -> Tuple[List[dict], np.ndarray, List[str]]:
        return self.dataset_.get_data()  # type: ignore

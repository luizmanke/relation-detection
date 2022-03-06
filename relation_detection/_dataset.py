import numpy as np
from typing import List, Tuple
from .datasets.base import BaseDataset
from .datasets.dbpedia import DBpedia
from .datasets.news import News


class Dataset:

    available_sets_ = {
        "dbpedia": DBpedia,
        "news": News
    }

    def __init__(self, dataset_name: str, file_path: str) -> None:
        self.dataset_: BaseDataset = self.available_sets_[dataset_name](file_path)

    def get_data(self) -> Tuple[List[dict], np.ndarray, List[str]]:
        return self.dataset_.get_data()

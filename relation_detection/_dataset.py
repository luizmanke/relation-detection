from random import Random
from typing import Any, Dict, Optional
from .datasets.dbpedia import DBpedia
from .datasets.news import News


class Dataset:

    available_sets_ = {
        "dbpedia": DBpedia,
        "news": News
    }

    def __init__(
        self,
        dataset_name: str,
        file_path: str,
        n_samples: Optional[int] = None
    ) -> None:
        self.dataset_ = self.available_sets_[dataset_name](file_path)
        self.n_samples_ = n_samples

    def get_data(self) -> Dict[str, Any]:
        if self.n_samples_:
            return self._random_sample()
        return self.dataset_.get_data()

    def _random_sample(self) -> Dict[str, Any]:
        samples, labels, groups = self.dataset_.get_data()
        indexes = [i for i in range(len(samples))]
        Random(42).shuffle(indexes)
        indexes_selected = indexes[:self.n_samples_]
        return {
            "samples": [samples[i] for i in indexes_selected],
            "labels": labels[indexes_selected],
            "groups": [groups[i] for i in indexes_selected]
        }

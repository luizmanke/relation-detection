import numpy as np
from abc import abstractclassmethod
from typing import List, Tuple


class BaseDataset:

    @abstractclassmethod
    def get_data(self) -> Tuple[List[dict], np.ndarray, List[str]]:
        pass

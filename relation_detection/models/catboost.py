import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool  # type: ignore
from sklearn.model_selection import GroupShuffleSplit
from typing import List, Optional, Tuple


class CatBoost(CatBoostClassifier):

    def __init__(self):
        CatBoostClassifier.__init__(self, random_state=42)

    def fit(self, samples: List[dict], y: np.ndarray, groups: List[str]) -> None:
        sentences = self._get_surroundings(samples)
        df = self._to_pandas(sentences, y)
        df_train, df_test = self._train_test_split(df, groups)
        self._fit_model(df_train, df_test)

    def predict(self, samples: List[dict]) -> np.ndarray:
        sentences = self._get_surroundings(samples)
        df = self._to_pandas(sentences)
        return CatBoostClassifier.predict(self, df)

    @staticmethod
    def _get_surroundings(samples: List[dict]) -> List[dict]:
        surroundings = []
        for sample in samples:
            surroundings.append({
                "prefix": " ".join(sample["tokens"][:sample["index_1"]]),
                "middle": " ".join(sample["tokens"][sample["index_1"]+1:sample["index_2"]]),
                "suffix": " ".join(sample["tokens"][sample["index_2"]+1:])
            })
        return surroundings

    @staticmethod
    def _to_pandas(sentences: List[dict], y: Optional[np.ndarray] = None) -> pd.DataFrame:
        items = sentences
        if y is not None:
            for item, label in zip(items, y):
                item["label"] = label
        return pd.DataFrame(items)

    @staticmethod
    def _train_test_split(
            df: pd.DataFrame, groups: List[str]
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        gss = GroupShuffleSplit(train_size=0.8, n_splits=2, random_state=42)
        indexes_train, indexes_test = next(gss.split(df, groups=groups))
        return df.iloc[indexes_train], df.iloc[indexes_test]

    def _fit_model(self, df_train: pd.DataFrame, df_test: pd.DataFrame) -> None:
        FEATURES = ["prefix", "middle", "suffix"]
        train_pool = Pool(
            df_train[FEATURES],
            df_train["label"],
            text_features=FEATURES
        )
        eval_pool = Pool(
            df_test[FEATURES],
            df_test["label"],
            text_features=FEATURES
        )
        CatBoostClassifier.fit(self, train_pool, eval_set=eval_pool, verbose=False)

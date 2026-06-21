"""Serializable production model bundle shared by training and serving."""

from dataclasses import dataclass
from typing import Any, List

import numpy as np
import pandas as pd


@dataclass
class ModelBundle:
    """Keep fitted target encoder, estimator and feature contract together."""

    preprocessor: Any
    estimator: Any
    feature_names: List[str]
    categorical_columns: List[str]
    log_target: bool = False

    def _transform(self, X):
        frame = pd.DataFrame(X).copy()
        for column in self.feature_names:
            if column not in frame:
                frame[column] = "__MISSING__" if column in self.categorical_columns else 0.0
        frame = frame[self.feature_names]
        for column in self.categorical_columns:
            frame[column] = frame[column].fillna("__MISSING__").astype(str)
        numeric = [c for c in self.feature_names if c not in self.categorical_columns]
        frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce").fillna(0.0)
        return self.preprocessor.transform(frame)

    def predict(self, X):
        prediction = np.asarray(self.estimator.predict(self._transform(X)))
        return np.expm1(prediction).clip(min=0) if self.log_target else prediction

    def predict_proba(self, X):
        return self.estimator.predict_proba(self._transform(X))

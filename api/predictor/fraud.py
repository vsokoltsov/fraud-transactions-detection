from typing import Union
import os
import numpy as np
import numpy.typing as npt
from api.config import Settings
import joblib
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from api.types import DataFrame

LINEAR_REGRESSION = "linear_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"


class FraudPredictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Union[
            LogisticRegression, RandomForestClassifier, xgboost.XGBClassifier
        ] = self._init_model()

    def predict_proba(self, df: DataFrame) -> npt.NDArray[np.float64]:
        return self.model.predict_proba(df)

    def _init_model(
        self,
    ) -> Union[LogisticRegression, RandomForestClassifier, xgboost.XGBClassifier]:
        if self.settings.model == LINEAR_REGRESSION:
            return joblib.load(
                os.path.join(self.settings.models_path, f"{self.settings.model}.pkl")
            )
        elif self.settings.model == RANDOM_FOREST:
            return joblib.load(
                os.path.join(self.settings.models_path, f"{self.settings.model}.pkl")
            )
        elif self.settings.model == XGBOOST:
            model = xgboost.XGBClassifier()
            model.load_model(
                os.path.join(self.settings.models_path, f"{self.settings.model}.json")
            )
            return model
        else:
            raise ValueError("Model is unknown")

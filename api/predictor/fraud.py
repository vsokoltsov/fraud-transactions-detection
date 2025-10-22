from typing import Union, List, Dict
import os
import numpy as np
import numpy.typing as npt
from api.config import Settings
import joblib
import xgboost
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from api.types import DataFrame
import yaml

LINEAR_REGRESSION = "linear_regression"
RANDOM_FOREST = "random_forest"
XGBOOST = "xgboost"


class FraudPredictor:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.model: Union[
            LogisticRegression, RandomForestClassifier, xgboost.XGBClassifier
        ] = self._init_model()
        self.features = self._load_features()
        self.threshold = self._threshold_for_model(
            os.path.join(self.settings.models_path, 'thresholds.yaml')
        )


    def predict_proba(self, df: DataFrame) -> npt.NDArray[np.float64]:
        """
        Calculate probability of given transaction to be fraud

        Args:
            df (pandas.DataFrame): Dataframe with all necessary for prediction features
        Returns:
            npt.NDArray: Numpy array with possibilites of belonging to both of the classes
        """
        return self.model.predict_proba(df)

    def _init_model(
        self,
    ) -> Union[LogisticRegression, RandomForestClassifier, xgboost.XGBClassifier]:
        """
        Initializes ML model.
        Possible options:
        - LinearRegression
        - RandomForest
        - XGBoost

        :returns: Previously saved model instance
        """
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

    def _load_features(self) -> List[str]:
        """Load list of features from yaml file"""
        features = []
        with open(self.settings.features_path, "r", encoding="utf-8") as f:
            lines = yaml.safe_load(f)
            features.extend(lines)

        return features

    def _load_model_thresholds(self, thresholds_path: str) -> Dict[str, float]:
        """Load models' thresholds from yaml file"""
        thresholds: Dict[str, float] = {}
        with open(thresholds_path, "r", encoding="utf-8") as f:
            thresholds = yaml.safe_load(f)
        return thresholds

    def _threshold_for_model(self, thresholds_path: str) -> float:
        """Retrieve threshold for given model"""
        thresholds = self._load_model_thresholds(thresholds_path)
        threshold = thresholds.get(self.settings.model)
        if not threshold:
            raise ValueError(f"There is no threshold for model '{self.settings.model}' in {thresholds_path}")
        return float(threshold)


import numpy as np
import xgboost as xgb

from .base import BaseModel


class XGBoostModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__("xgboost")
        self.config = config
        self.mode = config.get("mode", "regression")
        base_params = {
            "n_estimators": config.get("n_estimators", 200),
            "max_depth": config.get("max_depth", 6),
            "learning_rate": config.get("learning_rate", 0.05),
            "subsample": config.get("subsample", 0.8),
            "colsample_bytree": config.get("colsample_bytree", 0.8),
            "random_state": 42,
        }
        if self.mode == "classification":
            base_params["objective"] = "binary:logistic"
            base_params["eval_metric"] = "logloss"
        else:
            base_params["objective"] = "reg:squarederror"
            base_params["eval_metric"] = "rmse"
        self.params = base_params

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        if self.mode == "classification":
            self.model = xgb.XGBClassifier(**self.params)
        else:
            self.model = xgb.XGBRegressor(**self.params)
        self.model.fit(X, y, verbose=False)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        if self.mode == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)

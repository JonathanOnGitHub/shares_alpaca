import numpy as np

from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

from .base import BaseModel
from .lstm_model import LSTMModel
from .xgb_model import XGBoostModel


class LinearModel(BaseModel):
    def __init__(self, mode: str = "regression"):
        super().__init__("linear")
        self.mode = mode
        if mode == "classification":
            from sklearn.linear_model import LogisticRegression
            self.model = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
        else:
            self.model = Ridge(alpha=1.0)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        self.model.fit(X, y)
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        if self.mode == "classification":
            return self.model.predict_proba(X)[:, 1]
        return self.model.predict(X)


class EnsembleModel:
    def __init__(self, config: dict):
        self.config = config
        self.mode = config.get("mode", "regression")
        self.weights = config.get("weights", {"lstm": 0.4, "xgboost": 0.4, "linear": 0.2})
        self.models: dict[str, BaseModel] = {}

        lstm_cfg = {**config.get("lstm", {}), "mode": self.mode}
        xgb_cfg = {**config.get("xgboost", {}), "mode": self.mode}

        if config.get("lstm", {}).get("enabled", True):
            self.models["lstm"] = LSTMModel(lstm_cfg)
        if config.get("xgboost", {}).get("enabled", True):
            self.models["xgboost"] = XGBoostModel(xgb_cfg)
        if config.get("linear", {}).get("enabled", True):
            self.models["linear"] = LinearModel(mode=self.mode)

    @property
    def is_trained(self) -> bool:
        return all(m.is_trained for m in self.models.values())

    def train(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        results = {}
        lstm_model = self.models.get("lstm")
        seq_len = lstm_model.sequence_length if lstm_model else 0

        for name, model in self.models.items():
            if isinstance(model, LSTMModel):
                model.train(X, y)
            else:
                model.train(X, y)

        for name, model in self.models.items():
            preds = self._predict_model(model, X)
            aligned_y = y[-len(preds):] if len(preds) < len(y) else y[:len(preds)]
            if self.mode == "classification":
                from sklearn.metrics import log_loss
                eps = 1e-12
                clipped = np.clip(preds, eps, 1 - eps)
                results[name] = float(log_loss(aligned_y, clipped))
            else:
                mse = mean_squared_error(aligned_y, preds)
                results[name] = float(np.sqrt(mse))

        return results

    def _predict_model(self, model: BaseModel, X: np.ndarray) -> np.ndarray:
        preds = model.predict(X)
        return preds

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("All models must be trained before ensemble prediction")

        all_preds = []
        min_len = None

        for name, model in self.models.items():
            weight = self.weights.get(name, 0.0)
            if weight <= 0:
                continue
            preds = self._predict_model(model, X)
            if len(preds) == 0:
                continue
            all_preds.append((weight, preds))
            if min_len is None or len(preds) < min_len:
                min_len = len(preds)

        if not all_preds or min_len == 0:
            return np.array([])

        weighted_sum = np.zeros(min_len)
        total_weight = 0.0
        for weight, preds in all_preds:
            weighted_sum += weight * preds[:min_len]
            total_weight += weight

        result = weighted_sum / total_weight
        if self.mode == "classification":
            result = result - 0.5  # center probabilities around 0
        return result

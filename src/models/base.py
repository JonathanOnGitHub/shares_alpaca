from abc import ABC, abstractmethod

import numpy as np


class BaseModel(ABC):
    def __init__(self, name: str):
        self.name = name
        self.model = None
        self.is_trained = False

    @abstractmethod
    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        ...

    def save(self, path: str) -> None:
        import joblib
        joblib.dump(self.model, f"{path}/{self.name}.joblib")

    def load(self, path: str) -> None:
        import joblib
        self.model = joblib.load(f"{path}/{self.name}.joblib")
        self.is_trained = True

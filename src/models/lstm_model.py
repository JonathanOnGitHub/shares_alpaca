import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping

from .base import BaseModel


class LSTMModel(BaseModel):
    def __init__(self, config: dict):
        super().__init__("lstm")
        self.config = config
        self.sequence_length = config.get("sequence_length", 60)
        self.hidden_units = config.get("hidden_units", [128, 64, 32])
        self.dropout = config.get("dropout", 0.2)
        self.epochs = config.get("epochs", 50)
        self.batch_size = config.get("batch_size", 32)
        self.learning_rate = config.get("learning_rate", 0.001)
        self.mode = config.get("mode", "regression")

    def _build_model(self, input_shape: tuple):
        model = Sequential()
        model.add(Input(shape=input_shape))
        for i, units in enumerate(self.hidden_units):
            return_seq = i < len(self.hidden_units) - 1
            model.add(LSTM(units, return_sequences=return_seq))
            model.add(Dropout(self.dropout))
        if self.mode == "classification":
            model.add(Dense(1, activation="sigmoid"))
            loss = "binary_crossentropy"
            metrics = ["accuracy"]
        else:
            model.add(Dense(1))
            loss = "mse"
            metrics = ["mae"]
        model.compile(
            optimizer=Adam(learning_rate=self.learning_rate),
            loss=loss,
            metrics=metrics,
        )
        return model

    def _create_sequences(
        self, X: np.ndarray, y: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        X_seq, y_seq = [], []
        n = len(X)
        if n <= self.sequence_length:
            return np.empty((0, self.sequence_length, X.shape[1]), dtype=np.float32), np.empty((0,), dtype=np.float32)
        for i in range(self.sequence_length, n):
            X_seq.append(X[i - self.sequence_length : i])
            y_seq.append(y[i])
        return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)

    def train(self, X: np.ndarray, y: np.ndarray) -> None:
        X_seq, y_seq = self._create_sequences(X, y)
        if len(X_seq) == 0:
            raise ValueError(f"Training data too short (need > {self.sequence_length} samples, got {len(X)})")
        bs = min(self.batch_size, len(X_seq))
        self.model = self._build_model((self.sequence_length, X.shape[1]))
        early_stop = EarlyStopping(
            monitor="loss", patience=5, restore_best_weights=True
        )
        self.model.fit(
            X_seq,
            y_seq,
            epochs=self.epochs,
            batch_size=bs,
            callbacks=[early_stop],
            verbose=1,
        )
        self.is_trained = True

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self.is_trained:
            raise RuntimeError("Model must be trained before prediction")
        X_seq, _ = self._create_sequences(X, np.zeros(len(X)))
        if len(X_seq) == 0:
            return np.array([])
        return self.model.predict(X_seq, verbose=0).flatten()

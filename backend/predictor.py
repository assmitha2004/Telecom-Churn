"""
Prediction logic for the churn model.

This file uses ONLY the Python standard library. It reads backend/artifacts/model.json
(created by backend/train.py) and applies the model with simple maths:

    z_j    = (x_j - mean_j) / scale_j            (StandardScaler)
    logit  = intercept + sum_j weight_j * (z_j - pca_mean_j)   (PCA + Logistic Regression, merged)
    score  = 1 / (1 + exp(-logit))               (sigmoid)

Because there is no pandas / scikit-learn here, the API is tiny and starts fast.
That is important for serverless hosting such as Vercel.
"""

import json
import math
import os

ARTIFACT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "artifacts", "model.json")


class PredictionError(ValueError):
    """Raised when the input from the user is not valid."""


class ChurnModel:
    def __init__(self, path: str = ARTIFACT_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model file not found: {path}. Run 'python backend/train.py' to create it."
            )
        with open(path) as f:
            a = json.load(f)

        self.meta = a["meta"]
        self.feature_cols = a["feature_cols"]
        self.labels = a["labels"]
        self.mean = a["mean"]
        self.scale = a["scale"]
        self.pca_mean = a["pca_mean"]
        self.weights = a["weights"]
        self.intercept = a["intercept"]
        self.defaults = a["defaults"]
        self.key_features = a["key_features"]
        self.presets = a["presets"]
        self.metrics = a["metrics"]
        self.roc = a["roc"]
        self.importances = a["importances"]
        self.stats = a["stats"]

        self._index = {name: i for i, name in enumerate(self.feature_cols)}
        self._baseline = self._score(self.defaults)

    # ------------------------------------------------------------------ helpers
    def _logit(self, values: dict) -> float:
        total = self.intercept
        for j, name in enumerate(self.feature_cols):
            z = (values[name] - self.mean[j]) / self.scale[j]
            total += self.weights[j] * (z - self.pca_mean[j])
        return total

    @staticmethod
    def _sigmoid(x: float) -> float:
        # Safe for very large positive or negative numbers
        if x >= 0:
            return 1.0 / (1.0 + math.exp(-x))
        e = math.exp(x)
        return e / (1.0 + e)

    def _score(self, values: dict) -> float:
        return self._sigmoid(self._logit(values))

    def _risk_level(self, score: float) -> str:
        bands = self.meta["risk_bands"]
        if score < bands["low_below"]:
            return "Low"
        if score >= bands["high_from"]:
            return "High"
        return "Medium"

    # ------------------------------------------------------------------ public
    def predict(self, data: dict) -> dict:
        """
        data: {feature_name: number}. Missing features use the typical (median) value.
        Unknown feature names are ignored and returned in 'ignored'.
        """
        if not isinstance(data, dict):
            raise PredictionError("Send a JSON object like {\"total_ic_mou_8\": 120}.")

        values = dict(self.defaults)
        provided, ignored = [], []
        for name, raw in data.items():
            if name not in self._index:
                ignored.append(name)
                continue
            if isinstance(raw, bool) or raw is None or raw == "":
                raise PredictionError(f"'{name}' must be a number.")
            try:
                number = float(raw)
            except (TypeError, ValueError):
                raise PredictionError(f"'{name}' must be a number.")
            if not math.isfinite(number):
                raise PredictionError(f"'{name}' must be a finite number.")
            if abs(number) > 1e9:
                raise PredictionError(f"'{name}' is too large.")
            values[name] = number
            provided.append(name)

        score = self._score(values)

        # How much did each value you entered move the result, compared with a typical customer?
        # (change in logit; positive = more churn risk)
        effects = []
        for name in provided:
            j = self._index[name]
            delta = self.weights[j] * (values[name] - self.defaults[name]) / self.scale[j]
            if abs(delta) > 1e-9:
                effects.append(
                    {
                        "feature": name,
                        "label": self.labels[name],
                        "value": values[name],
                        "typical": round(self.defaults[name], 4),
                        "effect": round(delta, 4),
                        "direction": "raises risk" if delta > 0 else "lowers risk",
                    }
                )
        effects.sort(key=lambda e: abs(e["effect"]), reverse=True)

        return {
            "score": round(score, 4),
            "prediction": "Churn" if score >= self.meta["threshold"] else "No Churn",
            "risk_level": self._risk_level(score),
            "baseline_score": round(self._baseline, 4),
            "threshold": self.meta["threshold"],
            "factors": effects[:6],
            "provided_count": len(provided),
            "default_count": len(self.feature_cols) - len(provided),
            "ignored": ignored,
        }

    def schema(self) -> dict:
        return {
            "key_features": self.key_features,
            "presets": self.presets,
            "total_features": len(self.feature_cols),
            "threshold": self.meta["threshold"],
            "risk_bands": self.meta["risk_bands"],
            "baseline_score": round(self._baseline, 4),
        }


_model = None


def get_model() -> ChurnModel:
    """Load the model file once and reuse it."""
    global _model
    if _model is None:
        _model = ChurnModel()
    return _model

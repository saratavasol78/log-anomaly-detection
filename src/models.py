"""دو مدل بدون‌نظارت: Isolation Forest و Autoencoder."""
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler


def _prep(scaler, X, fit=False):
    """log1p برای مهار دامنه‌ی شمارش‌ها، سپس استانداردسازی."""
    Z = np.log1p(X)
    return scaler.fit_transform(Z) if fit else scaler.transform(Z)


class IForestDetector:
    def __init__(self, contamination=0.02, seed=42):
        self.scaler = StandardScaler()
        self.model = IsolationForest(n_estimators=200, contamination=contamination, random_state=seed)

    def fit(self, X):
        self.model.fit(_prep(self.scaler, X, fit=True))
        return self

    def score(self, X):
        # امتیاز بزرگ‌تر = ناهنجارتر
        return -self.model.score_samples(_prep(self.scaler, X))


class AutoEncoderDetector:
    """Autoencoder با گلوگاه (bottleneck) که فقط روی داده‌ی عادی آموزش می‌بیند.

    ورودی را به خودش نگاشت می‌کند؛ هرچه خطای بازسازی بیشتر باشد،
    پنجره از رفتار عادی دورتر است.
    """

    def __init__(self, hidden=(16, 4, 16), max_iter=800, seed=42):
        self.scaler = StandardScaler()
        self.net = MLPRegressor(
            hidden_layer_sizes=hidden, activation="relu",
            max_iter=max_iter, random_state=seed, learning_rate_init=1e-2,
        )

    def fit(self, X):
        Z = _prep(self.scaler, X, fit=True)
        self.net.fit(Z, Z)
        return self

    def score(self, X):
        Z = _prep(self.scaler, X)
        return ((self.net.predict(Z) - Z) ** 2).mean(axis=1)

"""definitions for experiment 3 module
Multiclass classification
Wavelet features
Random forest feature selection
Most-Recent Label strategy
Drop windows with NaN label
"""

import numpy as np

from sklearn.metrics import accuracy_score, get_scorer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier

from processing.feature_mappers import TorchWaveletFeatureMapper
from processing.label_mappers import TorchMulticlassMRLStrategy

from .base_experiment import BaseExperiment
from dataset.dataset import MAEDataset


MAX_LEVEL = 10


def sample(trial, *args, **kwargs):
    return Experiment(
        level=trial.suggest_int("level", 4, MAX_LEVEL, step=1),
        stride=trial.suggest_int("stride", 10, 10),
        n_features=trial.suggest_float("n_features", 0.1, 1.0),
        normal_balance=trial.suggest_int("normal_balance", 1, 10, step=1),
    )


class Experiment(BaseExperiment):
    """the docstring"""

    def __init__(
        self,
        level,
        stride,
        n_features,
        normal_balance,
        *args,
        **kwargs,
    ):
        super().__init__()

        # save params
        self.window_size = 2**level
        self.level = level
        self.n_features = n_features
        self.stride = stride
        self.normal_balance = normal_balance

        self._init_raw_mappers()
        self._init_preprocessor()

    def raw_transform(self, event, transient_only=True, no_nans=True):
        # filter tags and set zeros to nans
        tags = event["tags"][self.selected_tags].replace(0, np.nan)
        labels = event["labels"]
        event_type = event["event_type"]

        if transient_only and MAEDataset.TRANSIENT_CLASS[event_type]:
            transients = labels.values != event_type
            tags = tags[transients]
            labels = labels[transients]

        features = self._feature_mapper(tags, event_type)
        labels = self._label_mapper(labels, event_type)

        # drop windows with NaN label
        if no_nans:
            notnan = labels.notna()
            features = features[notnan]
            labels = labels[notnan]

        return features, labels, event_type

    def metric_name(self):
        return "accuracy"

    def metric_rf(self):
        return get_scorer("accuracy")

    def metric_lgbm(self):
        def acc(preds, train_data):
            preds_ = np.argmax(np.reshape(preds, (self.num_classes, -1)), axis=0)
            return "accuracy", accuracy_score(train_data.get_label(), preds_), True

        return acc

    def fit(self, X, y=None):
        X = self._scaler.fit_transform(X)
        X = self._imputer.fit_transform(X)
        self._forest.fit(X, y)

    def transform(self, X, y=None):
        X = self._scaler.transform(X)
        X = self._imputer.transform(X)
        # filter most important features
        importances = self._forest.feature_importances_
        importance_order = np.argsort(importances)
        X = X[:, importance_order[: int(self.n_features * importances.size)]]
        return X, y

    def _init_raw_mappers(self):
        offset = 2**MAX_LEVEL - self.window_size
        self._feature_mapper = TorchWaveletFeatureMapper(
            level=self.level, stride=self.stride, offset=offset
        )
        self._label_mapper = TorchMulticlassMRLStrategy(
            window_size=self.window_size,
            stride=self.stride,
            offset=offset,
        )

    def _init_preprocessor(self):
        # z-score
        self._scaler = StandardScaler()
        # remove nans
        self._imputer = SimpleImputer(strategy="mean")
        # feature selection
        self._forest = RandomForestClassifier(n_jobs=-1)

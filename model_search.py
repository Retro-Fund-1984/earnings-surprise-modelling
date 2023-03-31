import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from xgboost import XGBClassifier

from sklearn.metrics._scorer import _BaseScorer
from sklearn.utils.multiclass import type_of_target
from sklearn.model_selection import PredefinedSplit

from mlresearch.metrics import get_scorer
from mlresearch.utils import check_pipelines, load_datasets
from rlearn.model_selection import ModelSearchCV
from rlearn.reporting import report_model_search_results


class _ProbaScorer(_BaseScorer):
    def _score(self, method_caller, clf, X, y, sample_weight=None):
        """Evaluate predicted probabilities for X relative to y_true.
        Parameters
        ----------
        method_caller : callable
            Returns predictions given an estimator, method name, and other
            arguments, potentially caching results.
        clf : object
            Trained classifier to use for scoring. Must have a `predict_proba`
            method; the output of that is used to compute the score.
        X : {array-like, sparse matrix}
            Test data that will be fed to clf.predict_proba.
        y : array-like
            Gold standard target values for X. These must be class labels,
            not probabilities.
        sample_weight : array-like, default=None
            Sample weights.
        Returns
        -------
        score : float
            Score function applied to prediction of estimator on X.
        """

        y_type = type_of_target(y)
        y_pred = method_caller(clf, "predict_proba", X)
        if y_type == "binary" and y_pred.shape[1] <= 2:
            # `y_type` could be equal to "binary" even in a multi-class
            # problem: (when only 2 class are given to `y_true` during scoring)
            # Thus, we need to check for the shape of `y_pred`.

            # OVERRIDEN: clf.classes_ doesn't exist apparently... Not sure
            # what was happening there. Bug from SKlearn perhaps?
            y_pred = y_pred[:, -1]
        if sample_weight is not None:
            return self._sign * self._score_func(
                y, y_pred, sample_weight=sample_weight, **self._kwargs
            )
        else:
            return self._sign * self._score_func(y, y_pred, **self._kwargs)

    def _factory_args(self):
        return ", needs_proba=True"


def precision_at_k(y_true, y_score, k=10):
    top_idx = np.argsort(y_score)[-k:]
    return y_true[top_idx].sum() / k


precisions = {
    f"p@{i}": _ProbaScorer(precision_at_k, 1, {"k": i})
    for i in [65, 100, 195, 350, 500]
}

CONFIG = {
    "scaling": [
        ("NONE", None, {}),
        ("MINMAX", MinMaxScaler(), {}),
        ("STANDARD", StandardScaler(), {}),
    ],
    "classifiers": [
        ("CONSTANT", DummyClassifier(strategy="prior"), {}),
        (
            "LR",
            LogisticRegression(max_iter=10000),
            {
                "penalty": ["l1", "l2"],
                "solver": ["saga"],
                "C": [0.00001, 0.0001, 0.001, 0.01, 0.1, 1, 10],
                "class_weight": [None, "balanced"],
            },
        ),
        (
            "RF",
            RandomForestClassifier(),
            {
                "max_depth": [None] + [1, 5, 10, 20, 50, 100],
                "n_estimators": [10**i for i in range(5)],
                "max_features": ["sqrt", "log2"],
                "min_samples_split": [2, 5, 10],
            },
        ),
        (
            "XGB",
            XGBClassifier(),
            {
                "learning_rate": [
                    0.1,
                    0.2,
                    0.3,
                    0.4,
                    0.5,
                    0.6,
                    0.7,
                    0.8,
                    0.9,
                    1.0,
                ],
                "n_estimators": [10**i for i in range(5)],
                "max_depth": [None] + [1, 5, 10, 20, 50, 100],
                "reg_alpha": [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0],
            },
        ),
        (
            "MLP",
            MLPClassifier(),
            {
                "hidden_layer_sizes": [
                    (5,),
                    (10,),
                    (50,),
                    (100,),
                    (5, 5),
                    (10, 10),
                    (50, 50),
                    (100, 100),
                ],
                "activation": ["logistic", "tanh", "relu"],
                "solver": ["lbfgs", "adam"],
            },
        ),
    ],
    "scoring": {
        "accuracy": get_scorer("accuracy"),
        "f1_macro": get_scorer("f1_macro"),
        "geometric_mean_score_macro": get_scorer("geometric_mean_score_macro"),
        **precisions,
    },
    "random_state": 42,
    "n_runs": 3,
    "n_jobs": -1,
    "verbose": 1,
}

if __name__ == "__main__":
    # Read data
    DATA_PATH = "matrices/"
    DATE = "2023mar28"
    data = dict(load_datasets(DATA_PATH, target_exists=False, suffix=DATE + ".csv"))

    # Set up data, with train/test split mask
    keys = np.sort(list(data.keys()))
    x_data = [data[k] for k in keys if k.startswith("X")]
    y_data = [data[k] for k in keys if k.startswith("Y")]
    X = pd.concat(x_data).drop(columns=["Unnamed: 0", "target"]).set_index("symbol")
    y = pd.concat(y_data).drop(columns="Unnamed: 0").squeeze().astype(int)
    ps = np.concatenate(
        [
            # -1 means train, 0 means test
            np.zeros(data.shape[0]) + (-1 if "TRAIN" in key else 0)
            for key, data in zip([k for k in keys if k.startswith("Y")], y_data)
        ]
    )
    ps = PredefinedSplit(ps)

    # Set up pipelines
    estimators, param_grids = check_pipelines(
        CONFIG["scaling"],
        CONFIG["classifiers"],
        random_state=CONFIG["random_state"],
        n_runs=CONFIG["n_runs"],
    )

    # Define and fit experiment
    experiment = ModelSearchCV(
        estimators=estimators,
        param_grids=param_grids,
        scoring=CONFIG["scoring"],
        n_jobs=CONFIG["n_jobs"],
        cv=ps,
        verbose=CONFIG["verbose"],
        return_train_score=True,
        refit=False,
    ).fit(X, y)

    results = report_model_search_results(experiment)

    # Save results
    pd.DataFrame(experiment.cv_results_).to_pickle(
        f"results/experiment_results{DATE}.pkl"
    )

"""
Get the investment strategy.
"""
from copy import deepcopy
import numpy as np
import pandas as pd
from datetime import datetime
from yahooquery import Ticker
from mlresearch.utils import check_pipelines

from esurprise.utils import parallel_loop
from scripts.model_search import CONFIG

REFERENCE_METRIC = "mean_test_p@65"

df = pd.read_pickle("results/experiment_results2023apr10.pkl")
X_train = pd.read_csv("matrices/_old/X_test_2023apr10.csv").set_index("symbol")
y_train = pd.read_csv("matrices/_old/y_test_2023apr10.csv").values.squeeze()
X_future = pd.read_csv("matrices/_futures/X_future_2023apr10.csv").set_index("symbol")


def get_earnings_date(symbol):
    events = Ticker(symbol).calendar_events[symbol]
    if "earnings" in events.keys():
        dates = events["earnings"]["earningsDate"]
        return dates
    else:
        return np.nan


# Get parameters and model with best performance
best_test_perf = (
    df.groupby("param_est_name")[
        ["mean_test_f1_macro", "mean_test_p@65", "mean_test_p@195"]
    ]
    .max()
    .sort_values([REFERENCE_METRIC])
)

best_model = best_test_perf.iloc[-1]

model = df[df[REFERENCE_METRIC] == best_model[REFERENCE_METRIC]][
    ["param_est_name", "params"]
].iloc[0]
params = {"__".join(k.split("__")[1:]): v for k, v in model.params.items()}
params.pop("")

estimators, param_grids = check_pipelines(
    CONFIG["scaling"],
    CONFIG["classifiers"],
    random_state=CONFIG["random_state"],
    n_runs=CONFIG["n_runs"],
)

# Set up and train classifier
classifier = deepcopy(dict(estimators)[model.param_est_name]).set_params(**params)
classifier.fit(X_train, y_train)

# Get probabilities and top 65 list
probs = classifier.predict_proba(X_future)[:, -1]
strategy = (
    pd.DataFrame(probs, index=X_future.index, columns=["prob"])
    .sort_values("prob", ascending=False)
    .iloc[:65]
)

# Get earning dates
earnings_dates = parallel_loop(
    get_earnings_date, strategy.index, n_jobs=-1, progress_bar=True
)

strategy["start_earn_date"] = pd.to_datetime(
    [":".join(i[0].split(":")[:-1]) for i in earnings_dates]
)
strategy["end_earn_date"] = pd.to_datetime(
    [":".join(i[-1].split(":")[:-1]) if len(i) > 1 else np.nan for i in earnings_dates]
)

strategy.sort_values("start_earn_date")

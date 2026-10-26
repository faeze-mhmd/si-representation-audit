                                                                                        

import hashlib
import json
from pathlib import Path
from time import perf_counter
import warnings

import numpy as np
from scipy.stats import rankdata
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier

from .baselines import classification_summary
from .data import parse_optdigits

TEST_SHA256 = "6ebb3d2fee246a4e99363262ddf8a00a3c41bee6014c373ed9d9216ba7f651b8"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def load_official_test(path):
    payload = Path(path).read_bytes()
    require(hashlib.sha256(payload).hexdigest() == TEST_SHA256, "official test hash mismatch")
    data = parse_optdigits(payload, sample_id_start=3823)
    require(data.features.shape == (1797, 64), "wrong test shape")
    return data


def rank_agreement(scores, outcome):
                                                                                    
    x, y = np.asarray(scores, dtype=float), np.asarray(outcome, dtype=float)
    require(x.ndim == y.ndim == 1 and x.shape == y.shape and len(x) >= 2, "invalid rank vectors")
    require(np.isfinite(x).all() and np.isfinite(y).all(), "nonfinite rank scores")
    rho = None if np.ptp(x) == 0 or np.ptp(y) == 0 else float(np.corrcoef(rankdata(x), rankdata(y))[0, 1])
    counts = {"concordant": 0, "discordant": 0, "predictor_tie_only": 0, "outcome_tie_only": 0, "both_tied": 0}
    for i in range(len(x)):
        for j in range(i):
            a, b = np.sign(x[i] - x[j]), np.sign(y[i] - y[j])
            key = ("both_tied" if a == b == 0 else "predictor_tie_only" if a == 0 else
                   "outcome_tie_only" if b == 0 else "concordant" if a == b else "discordant")
            counts[key] += 1
    return {"n_models": len(x), "spearman_rho": rho, "pairs": counts}


def fixed_estimators(config, *, probes=False):
    linear = LogisticRegression(C=config["probe_logistic_C"] if probes else 1.0,
                                max_iter=config["probe_logistic_max_iter"] if probes else 1000,
                                solver="lbfgs", random_state=2027)
    nearest = KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=1)
    if probes:
        return [("linear_probe", linear), ("cross_split_1nn_probe", nearest)]
    return [("majority", DummyClassifier(strategy="most_frequent")),
            ("logistic_regression", linear), ("euclidean_1nn", nearest)]


def fit_and_validate(estimator, train_x, train_y, validation_x, validation_y, expected):
                                                                                      
    start = perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        estimator.fit(train_x, train_y)
        predicted = estimator.predict(validation_x)
    summary = classification_summary(validation_y, predicted)
    require(summary == expected, "validation reproduction mismatch; test evaluation prohibited")
    return {"parameters": estimator.get_params(deep=False), "validation_reproduced": True,
            "fit_and_validation_seconds": perf_counter() - start,
            "warnings": [str(w.message) for w in caught]}


def mean_sample_sd(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": float(np.mean(values)), "sample_sd": float(np.std(values, ddof=1))}

                                                                               

from time import perf_counter
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .data import validate_manifest


def classification_summary(labels, predicted):
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predicted, labels=np.arange(10), zero_division=0,
    )
    return {
        "n": len(labels), "accuracy": float(accuracy_score(labels, predicted)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(labels, predicted, labels=np.arange(10)).tolist(),
        "per_class": {str(i): {"precision": float(precision[i]), "recall": float(recall[i]),
                               "f1": float(f1[i]), "support": int(support[i])} for i in range(10)},
    }


def fit_input_baselines(pool, manifest):
                                                                                  
    validate_manifest(pool, manifest)
    training = np.array(manifest["train_rows"])
    validation = np.array(manifest["validation_rows"])
                                                                                
    x_train = pool.features[training].astype(np.float64) / 16.0
    x_validation = pool.features[validation].astype(np.float64) / 16.0
    y_train, y_validation = pool.labels[training], pool.labels[validation]
    models = (
        ("majority", DummyClassifier(strategy="most_frequent")),
        ("logistic_regression", LogisticRegression(C=1.0, max_iter=1000, solver="lbfgs", random_state=2027)),
        ("euclidean_1nn", KNeighborsClassifier(n_neighbors=1, algorithm="brute", metric="euclidean", n_jobs=1)),
    )
    records = []
    with threadpool_limits(limits=1):
        for name, model in models:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                start = perf_counter()
                model.fit(x_train, y_train)
                fit_seconds = perf_counter() - start
                start = perf_counter()
                validation_prediction = model.predict(x_validation)
                predict_seconds = perf_counter() - start
                train_prediction = model.predict(x_train)
            errors = np.flatnonzero(y_validation != validation_prediction)
            record = {
                "model": name, "parameters": model.get_params(deep=False),
                "training": classification_summary(y_train, train_prediction),
                "validation": classification_summary(y_validation, validation_prediction),
                "fit_seconds": fit_seconds, "validation_predict_seconds": predict_seconds,
                "warnings": [{"category": type(w.message).__name__, "message": str(w.message)} for w in caught],
                "validation_error_count": len(errors),
                "error_examples_policy": "first 30 errors in ascending original row order; aggregate metrics use all validation examples",
                "validation_error_examples": [{"sample_id": int(pool.sample_ids[validation[i]]),
                                       "true": int(y_validation[i]), "predicted": int(validation_prediction[i])} for i in errors[:30]],
            }
            if hasattr(model, "n_iter_"):
                record["optimizer_iterations"] = model.n_iter_.tolist()
            records.append(record)
    return {
        "evaluation_split": "fixed inner validation only; no test predictions or test metrics",
        "normalization": "feature / 16.0; no fitted scaler",
        "threads": 1, "seed": 2027,
        "knn_ties": "scikit-learn brute-force implementation on ascending original training row IDs; no cross-library tie equivalence claimed",
        "training_metric_limit": "in-sample; 1-NN sees its own training queries and training accuracy is not leave-one-out SI",
        "models": records,
    }

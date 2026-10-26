                                                                                    

from time import perf_counter
import warnings

import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits

from .baselines import classification_summary


def fixed_readouts(config, *, probes=False):
    linear=LogisticRegression(C=config["probe_logistic_C"],max_iter=config["probe_logistic_max_iter"],
                              solver="lbfgs",random_state=config["probe_seed"])
    nearest=KNeighborsClassifier(n_neighbors=1,algorithm="brute",metric="euclidean",n_jobs=1)
    if probes:
        return [("linear_probe",linear),("cross_split_1nn_probe",nearest)]
    return [("majority",DummyClassifier(strategy="most_frequent")),
            ("logistic_regression",linear),("euclidean_1nn",nearest)]


def predict_batched(estimator,features, *, batch_size=256):
    if isinstance(batch_size,bool) or not isinstance(batch_size,int) or batch_size<1 or len(features)<1:
        raise ValueError("nonempty queries and positive integer batch size required")
    return np.concatenate([estimator.predict(features[i:i+batch_size]) for i in range(0,len(features),batch_size)])


def fit_readout(estimator,train_x,train_y,query_x,query_y):
                                                                             
    with threadpool_limits(limits=1),warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        start=perf_counter()
        estimator.fit(train_x,train_y)
        fit_seconds=perf_counter()-start
        start=perf_counter()
        prediction=predict_batched(estimator,query_x)
        predict_seconds=perf_counter()-start
    summary=classification_summary(query_y,prediction)
    record={"parameters":estimator.get_params(deep=False),"validation":summary,
            "fit_seconds":fit_seconds,"predict_seconds":predict_seconds,
            "warnings":[str(w.message) for w in caught],
            "iterations":np.asarray(estimator.n_iter_).tolist() if hasattr(estimator,"n_iter_") else None}
    return record,prediction

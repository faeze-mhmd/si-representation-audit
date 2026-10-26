                                                                          

from pathlib import Path
import os
import sys
import unittest
import warnings

os.environ["LOKY_MAX_CPU_COUNT"]="1"

import numpy as np
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"src"))
from si_audit.fashion_readouts import predict_batched,fit_readout


class FashionReadoutTests(unittest.TestCase):
    def test_bounded_queries_preserve_exact_tie_predictions(self):
        x=np.array([[0.,0.],[0.,0.],[2.,0.],[4.,0.]],dtype=np.float32)
        y=np.array([0,1,2,3])
        model=KNeighborsClassifier(n_neighbors=1,algorithm="brute",metric="euclidean",n_jobs=1).fit(x,y)
        queries=np.array([[0.,0.],[1.,0.],[3.,0.],[4.,0.],[2.,0.]],dtype=np.float32)
        np.testing.assert_array_equal(predict_batched(model,queries,batch_size=2),model.predict(queries))

    def test_only_training_fitted_queries_bounded_and_warnings_retained(self):
        class Spy:
            def fit(self,x,y):
                self.x,self.y=x.copy(),y.copy()
                self.query_lengths=[]
                warnings.warn("recorded fitting diagnostic",UserWarning)
            def predict(self,x):
                self.query_lengths.append(len(x))
                return np.zeros(len(x),dtype=int)
            def get_params(self,deep=False): return {}
        train_x=np.array([[1.],[2.]])
        query_x=np.arange(600).reshape(-1,1)+100
        spy=Spy()
        record,predicted=fit_readout(spy,train_x,np.array([0,1]),query_x,np.zeros(600,dtype=int))
        np.testing.assert_array_equal(spy.x,train_x)
        self.assertEqual(spy.query_lengths,[256,256,88])
        self.assertEqual(record["warnings"],["recorded fitting diagnostic"])
        self.assertEqual(len(predicted),600)
        self.assertEqual(record["validation"]["accuracy"],1.0)


if __name__=="__main__":
    unittest.main()

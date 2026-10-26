import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from sklearn.dummy import DummyClassifier

from si_audit.fashion_final import claim_once,checked_readout
from si_audit.fashion_readouts import fit_readout


class FashionFinalGuards(unittest.TestCase):
    def test_existing_claim_is_preserved(self):
        with tempfile.TemporaryDirectory() as directory:
            claim_once(directory,{'plan_sha256':'original'})
            path=Path(directory)/'evaluation_started.json'
            before=path.read_bytes()
            with self.assertRaises(FileExistsError):
                claim_once(directory,{'plan_sha256':'replacement'})
            self.assertEqual(path.read_bytes(),before)
            self.assertEqual(json.loads(before)['plan_sha256'],'original')

    def test_validation_gate_accepts_identical_fixed_readout(self):
        x=np.arange(8).reshape(4,2)
        y=np.array([0,0,0,1])
        expected,prediction=fit_readout(DummyClassifier(strategy='most_frequent'),x,y,x,y)
        result=checked_readout(DummyClassifier(strategy='most_frequent'),x,y,x,y,expected,prediction)
        self.assertEqual(result['validation'],expected['validation'])

    def test_equal_metrics_do_not_hide_prediction_drift(self):
        expected={'validation':{'accuracy':0.5},'parameters':{},'warnings':[],'iterations':None}
        with patch('si_audit.fashion_final.fit_readout',return_value=(expected,np.array([1,0]))):
            with self.assertRaisesRegex(ValueError,'prediction drift'):
                checked_readout(None,None,None,None,None,expected,np.array([0,1]))

    def test_warning_and_parameter_drift_stop_gate(self):
        expected={'validation':{},'parameters':{'C':1},'warnings':['cap'],'iterations':[1000]}
        for key,value in [('warnings',[]),('parameters',{'C':2}),('iterations',[999]),('validation',{'accuracy':1})]:
            replay={**expected,key:value}
            with self.subTest(key=key),patch('si_audit.fashion_final.fit_readout',return_value=(replay,np.array([0]))):
                with self.assertRaisesRegex(ValueError,key):
                    checked_readout(None,None,None,None,None,expected,np.array([0]))

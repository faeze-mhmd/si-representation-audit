                                                                                       

import copy
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

os.environ["LOKY_MAX_CPU_COUNT"]="1"

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.cnn import build_fashion_cnn
from si_audit.fashion_training import cnn_bundle,cnn_from_bundle,evaluate_cnn,run_cnn


class FashionTrainingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def fixture(self):
        config=json.loads((ROOT/"configs/fashion_cnn_v1.json").read_text())
        config.update(epochs=3,primary_epoch=3,checkpoint_epochs=[0,1,3],batch_size=7,
            budgets={"A":20,"B":40},permutation_count=3,si_distance_batch_size=4)
        generator=torch.Generator().manual_seed(2042)
        data={}
        for split,offset in (("train",0),("validation",100)):
            data[split]={"x":torch.rand(20,1,28,28,generator=generator),"y":torch.arange(20)%10,
                "ids":np.arange(offset,offset+20),"si_positions":np.arange(20),"rows":np.arange(offset,offset+20)}
        data["validation"]["sensitivity_positions"]=np.arange(20)
        return config,data

    def test_portable_cnn_preserves_logits_and_rng_and_rejects_corruption(self):
        model=build_fashion_cnn(11)
        bundle=cnn_bundle(model,{"fixture":True})
        before=torch.get_rng_state().clone()
        restored=cnn_from_bundle(bundle)
        self.assertTrue(torch.equal(before,torch.get_rng_state()))
        x=torch.ones(3,1,28,28)
        with torch.no_grad(): self.assertTrue(torch.equal(model(x),restored(x)))
        corrupt=copy.deepcopy(bundle)
        corrupt["payload_sha256"]="wrong"
        with self.assertRaisesRegex(ValueError,"payload mismatch"): cnn_from_bundle(corrupt)
        with self.assertRaises(ValueError): cnn_from_bundle({**bundle,"architecture":"MLP"})

    def test_cross_entropy_weights_unequal_query_batches_correctly(self):
        model=build_fashion_cnn(11)
        x=torch.rand(19,1,28,28,generator=torch.Generator().manual_seed(11))
        y=torch.arange(19)%10
        summary,predicted=evaluate_cnn(model,x,y,detailed=True,batch_size=7)
        with torch.no_grad():
            expected=float(torch.nn.functional.cross_entropy(model(x),y))
            np.testing.assert_array_equal(predicted,model(x).argmax(1).numpy())
        self.assertAlmostEqual(summary["cross_entropy"],expected,places=6)
        self.assertEqual(sum(map(sum,summary["confusion_matrix"])),19)

    def test_interrupted_optimizer_loader_rng_and_measurements_reproduce(self):
        config,data=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            full,_=run_cnn(data,config,{"fixture":"v1"},Path(directory)/"full",setting="A",seed=11)
            partial,disposition=run_cnn(data,config,{"fixture":"v1"},Path(directory)/"resumed",setting="A",seed=11,stop_after_epoch=1)
            self.assertFalse(partial["completed"])
            self.assertEqual(disposition,"interrupted_for_test")
            text_path=Path(directory)/"resumed/metrics.json"
            stale_text=text_path.read_bytes()
            advanced,_=run_cnn(data,config,{"fixture":"v1"},Path(directory)/"resumed",setting="A",seed=11,stop_after_epoch=2)
            self.assertEqual(advanced["last_epoch"],2)
                                                                              
                                                                         
            text_path.write_bytes(stale_text)
            restored,disposition=run_cnn(data,config,{"fixture":"v1"},Path(directory)/"resumed",setting="A",seed=11)
            self.assertEqual(disposition,"resumed")
            for key in ("final_model_state_sha256","epoch_order_sha256","snapshots","history","final_prediction"):
                self.assertEqual(full[key],restored[key],key)
            for first,second in zip(full["probes"],restored["probes"]):
                self.assertEqual(first["validation"],second["validation"])
            reused,disposition=run_cnn(data,config,{"fixture":"v1"},Path(directory)/"resumed",setting="A",seed=11)
            self.assertEqual(disposition,"reused_completed")
            self.assertEqual(reused["final_model_state_sha256"],full["final_model_state_sha256"])

    def test_changed_config_or_provenance_cannot_resume(self):
        config,data=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            run_cnn(data,config,{"fixture":"v1"},directory,setting="A",seed=11,stop_after_epoch=1)
            with self.assertRaisesRegex(ValueError,"another source/config/split"):
                run_cnn(data,{**config,"learning_rate":.002},{"fixture":"v1"},directory,setting="A",seed=11)
            with self.assertRaises(ValueError):
                run_cnn(data,config,{"fixture":"v2"},directory,setting="A",seed=11)

    def test_corrupt_checkpoint_tensor_rejected(self):
        config,data=self.fixture()
        with tempfile.TemporaryDirectory() as directory:
            run_cnn(data,config,{"fixture":"v1"},directory,setting="A",seed=11,stop_after_epoch=1)
            path=Path(directory)/"resume.pt"
            state=torch.load(path,weights_only=True)
            state["model"]["classifier.bias"]+=.1
            with path.open("wb") as handle: torch.save(state,handle)
            with self.assertRaisesRegex(ValueError,"tensor digest mismatch"):
                run_cnn(data,config,{"fixture":"v1"},directory,setting="A",seed=11)


if __name__=="__main__":
    unittest.main()

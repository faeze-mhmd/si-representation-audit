                                                                           

import copy
import gzip
import hashlib
from pathlib import Path
import struct
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"src"))
from si_audit.fashion_data import (FashionData,parse_idx_pair,_read_official,load_training_pool,
                                   make_manifest,validate_manifest,exact_duplicate_audit)


def fixture(n=2):
    images=np.arange(n*784,dtype=np.uint8).reshape(n,28,28)
    labels=np.arange(n,dtype=np.uint8)%10
    return struct.pack(">IIII",2051,n,28,28)+images.tobytes(),struct.pack(">II",2049,n)+labels.tobytes()


class FashionDataTests(unittest.TestCase):
    def test_strict_pair_preserves_pixels_labels_and_ids(self):
        images,labels=fixture()
        data=parse_idx_pair(images,labels,expected_n=2,sample_id_start=60000)
        self.assertEqual(data.images.shape,(2,28,28))
        self.assertEqual(data.images.dtype,np.uint8)
        np.testing.assert_array_equal(data.images.reshape(-1),np.arange(1568,dtype=np.uint8))
        self.assertEqual(data.labels.tolist(),[0,1])
        self.assertEqual(data.sample_ids.tolist(),[60000,60001])

    def test_malformed_idx_rejected(self):
        images,labels=fixture()
        cases=[(b"",labels),(images,labels[:-1]),(images+b"x",labels),
               (struct.pack(">IIII",1,2,28,28)+images[16:],labels),
               (struct.pack(">IIII",2051,2,27,28)+images[16:],labels),
               (images,struct.pack(">II",2049,3)+labels[8:]),
               (images,labels[:8]+bytes([0,10]))]
        for a,b in cases:
            with self.subTest(lengths=(len(a),len(b))),self.assertRaises(ValueError):
                parse_idx_pair(a,b)
        with self.assertRaises(ValueError): parse_idx_pair(images,labels,expected_n=60000)

    def test_checksum_rejected_before_decompression_and_inflate_cap_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"train-labels-idx1-ubyte.gz"
            path.write_bytes(b"invalid")
            with patch("si_audit.fashion_data.gzip.GzipFile") as decoder:
                with self.assertRaisesRegex(ValueError,"checksum mismatch"): _read_official(path,128)
                decoder.assert_not_called()
            packed=gzip.compress(b"x"*1024)
            path.write_bytes(packed)
            with patch.dict("si_audit.fashion_data.PUBLISHED_MD5",{path.name:hashlib.md5(packed).hexdigest()}):
                with self.assertRaisesRegex(ValueError,"exceeds"): _read_official(path,128)

    def test_training_loader_never_opens_test_files(self):
        images,labels=fixture()
        opened=[]
        def read(path,cap):
            opened.append(path.name)
            return (images if "images" in path.name else labels),{}
        with patch("si_audit.fashion_data._read_official",side_effect=read):
            with self.assertRaisesRegex(ValueError,"count"): load_training_pool(Path("unused"))
        self.assertEqual(opened,["train-images-idx3-ubyte.gz","train-labels-idx1-ubyte.gz"])

    def _manifest_case(self):
        config=__import__("json").loads((ROOT/"configs/fashion_cnn_v1.json").read_text())
        config.update(official_training_n=400,validation_n=100,budgets={"A":100,"B":200},
                      si_train_n=20,si_validation_primary_n=20,si_validation_sensitivity_n=50)
        data=FashionData(np.zeros((400,28,28),dtype=np.uint8),np.repeat(np.arange(10),40),np.arange(400),{"fixture":"v1"})
        return data,config

    def test_nested_balanced_subsets_are_reproducible_and_disjoint(self):
        data,config=self._manifest_case()
        first=make_manifest(data,config)
        self.assertEqual(first,make_manifest(data,config))
        validate_manifest(data,first,config)
        a,b=map(set,[first["budget_rows"][k] for k in ("A","B")])
        val=set(first["validation_rows"])
        self.assertTrue(a<b)
        self.assertFalse(b&val)
        self.assertTrue(set(first["si_train_rows"])<=a)
        self.assertTrue(set(first["si_validation_rows"])<=set(first["si_validation_sensitivity_rows"])<=val)
        for key in ("si_train_rows","si_validation_rows","si_validation_sensitivity_rows"):
            rows=first[key]
            self.assertEqual(rows,sorted(set(rows)))
            self.assertEqual(len(set(np.bincount(data.labels[rows],minlength=10))),1)

    def test_manifest_tampering_and_invalid_sizes_rejected(self):
        data,config=self._manifest_case()
        manifest=make_manifest(data,config)
        corrupt=copy.deepcopy(manifest)
        corrupt["budget_rows"]["A"][0]=manifest["validation_rows"][0]
        with self.assertRaisesRegex(ValueError,"manifest differs"): validate_manifest(data,corrupt,config)
        changed=copy.deepcopy(config)
        changed["split_seed"]+=1
        with self.assertRaises(ValueError): validate_manifest(data,manifest,changed)
        with self.assertRaises(ValueError): make_manifest(data,{**config,"si_train_n":21})

    def test_exact_duplicates_retain_conflicts_and_actual_split_roles(self):
        images=np.stack([np.zeros((28,28),dtype=np.uint8),np.ones((28,28),dtype=np.uint8),
                         np.full((28,28),2,dtype=np.uint8),np.zeros((28,28),dtype=np.uint8)])
        train=FashionData(images,np.array([0,1,2,0]),np.arange(4),{})
        test=FashionData(images[[1,0]],np.array([4,0]),np.arange(60000,60002),{})
        manifest={"validation_rows":[1,3],"budget_rows":{"A":[0],"B":[0,2]}}
        result=exact_duplicate_audit(train,test,manifest)
        self.assertEqual(result["duplicate_groups"],2)
        self.assertEqual(result["rows_in_duplicate_groups"],5)
        self.assertEqual(result["conflicting_label_groups"],1)
        self.assertEqual(result["boundaries"]["pool__validation"]["groups"],1)
        self.assertEqual(result["boundaries"]["A__test"]["pairs"],1)
        self.assertEqual(result["boundaries"]["validation__test"]["pairs"],2)
        self.assertEqual(result["boundaries"]["validation__test"]["conflicting_label_groups"],1)


if __name__=="__main__":
    unittest.main()

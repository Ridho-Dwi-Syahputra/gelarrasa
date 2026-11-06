"""Prove accepted source anchors really have untied multi-source support."""
import unittest
import numpy as np
from stage16_source_consensus import independent_anchors,CLASSES
from stage15_second_cnn_refinement import load_second_features

class SourceContractTests(unittest.TestCase):
    def test_anchor_consensus_and_key_coverage(self):
        keys,labels,clean,votes,sources=independent_anchors()
        self.assertEqual(len(keys),2400)
        self.assertEqual(len(set(keys)),2400)
        self.assertEqual(sources.shape,(4,2400,8))
        np.testing.assert_array_equal(labels,votes.argmax(1))
        ordered=np.sort(votes,axis=1)
        np.testing.assert_array_equal(clean,(ordered[:,-1]>=2)&(ordered[:,-1]>ordered[:,-2]))
        self.assertTrue((np.bincount(labels[clean],minlength=len(CLASSES))>0).all())

    def test_new_feature_branches_preserve_ids_and_scale(self):
        keys=None
        for branch in ['second_dino','second_all']:
            k,x,fk,xf=load_second_features(branch)
            if keys is not None:np.testing.assert_array_equal(k,keys)
            keys=k
            self.assertEqual(len(k),2400);self.assertEqual(len(fk),256)
            self.assertEqual(x.dtype,np.float32);self.assertEqual(xf.dtype,np.float32)
            np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-5)
            np.testing.assert_allclose(np.linalg.norm(xf,axis=1),1,atol=1e-5)

if __name__=='__main__':unittest.main()

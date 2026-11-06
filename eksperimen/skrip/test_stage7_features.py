"""Check hybrid feature precision and head compatibility before fitting."""
import unittest
import numpy as np
import torch
from stage7_waste_semantic_training import feature_blocks,make_head

class FeatureTest(unittest.TestCase):
    def test_all_branches_keep_float32_and_unit_scale(self):
        rng=np.random.default_rng(11)
        waste=rng.normal(size=(4,2048)).astype('float32');waste/=np.linalg.norm(waste,axis=1,keepdims=True)
        dino=rng.normal(size=(4,384)).astype('float32');dino/=np.linalg.norm(dino,axis=1,keepdims=True)
        for branch in ['waste','dino','hybrid']:
            x=feature_blocks(waste,dino,branch)
            self.assertEqual(x.dtype,np.float32)
            np.testing.assert_allclose(np.linalg.norm(x,axis=1),1,atol=1e-6)
            for kind in ['linear','mlp']:
                head=make_head(kind,x.shape[1]);logits=head(torch.from_numpy(x))
                self.assertEqual(tuple(logits.shape),(4,8));self.assertTrue(torch.isfinite(logits).all())

if __name__=='__main__':unittest.main()

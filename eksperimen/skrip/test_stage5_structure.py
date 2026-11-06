"""Focused proofs for patch geometry and weighted prototype merging."""
import unittest
from types import SimpleNamespace
import numpy as np
from stage5_patch_features import valid_patch_mask
from stage5_material_structure import weighted_ward, blocks


class StructureTest(unittest.TestCase):
    def test_patch_mask_excludes_padding_and_tracks_flip(self):
        row=SimpleNamespace(width=50,height=200)
        mask=valid_patch_mask(row,224).reshape(16,16)
        self.assertTrue(mask[:,6:10].all())
        self.assertFalse(mask[:,:6].any())
        self.assertFalse(mask[:,10:].any())
        flipped=valid_patch_mask(row,224,True).reshape(16,16)
        np.testing.assert_array_equal(flipped,np.fliplr(mask))

    def test_weighted_merge_preserves_nearby_prototypes(self):
        points=np.array([[0.,0.],[.01,0.],[10.,10.],[10.01,10.]])
        mapping=weighted_ward(points,np.array([10,20,30,40]),2)
        self.assertEqual(mapping[0],mapping[1])
        self.assertEqual(mapping[2],mapping[3])
        self.assertNotEqual(mapping[0],mapping[2])
        self.assertEqual(set(mapping),{0,1})

    def test_hybrid_blocks_have_predeclared_mass(self):
        rng=np.random.default_rng(11)
        x=rng.normal(size=(5,3,384));x/=np.linalg.norm(x,axis=2,keepdims=True)
        y=blocks(x.reshape(5,1152),'hybrid')
        np.testing.assert_allclose((y[:,:384]**2).sum(1),.5)
        np.testing.assert_allclose((y[:,384:768]**2).sum(1),.25)
        np.testing.assert_allclose((y[:,768:]**2).sum(1),.25)
        np.testing.assert_allclose((y*y).sum(1),1.)


if __name__=='__main__':unittest.main()

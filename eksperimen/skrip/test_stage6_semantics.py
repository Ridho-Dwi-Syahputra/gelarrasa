"""Synthetic, non-competition proofs for ontology mapping and normalization."""
import unittest
import numpy as np
from stage6_semantic_ontology import CLASSES,object_evidence,combine_evidence,combine_calibrated

class SemanticTest(unittest.TestCase):
    def names(self):
        return ['wine bottle','comic book','laptop','milk can','plastic bag','carton','sock','banana','unmapped object']
    def test_each_public_object_anchor_points_to_its_class(self):
        logits=np.full((8,9),-20.,dtype='float32')
        logits[np.arange(8),np.arange(8)]=20
        evidence,_=object_evidence(logits,self.names())
        np.testing.assert_array_equal(evidence.argmax(1),np.arange(8))
    def test_material_probability_uses_named_channels(self):
        materials=['glass','paper','metal','plastic','fabric','leather','food']
        logits=np.full((1,9),-20.,dtype='float32');logits[0,-1]=20
        priors,_,_=combine_evidence(logits,self.names(),np.array([[1.,0,0,0,0,0,0]]),materials)
        self.assertEqual(CLASSES[priors.argmax(1)[0]],'glass')
        self.assertTrue(np.isfinite(priors).all())
        np.testing.assert_allclose(priors.sum(1),1.,atol=1e-6)
    def test_cardboard_has_explicit_non_llm_source(self):
        logits=np.full((1,9),-20.,dtype='float32');logits[0,5]=20
        materials=['glass','paper','metal','plastic','fabric','leather','food']
        p,_,_=combine_evidence(logits,self.names(),np.array([[0.,1.,0.,0.,0.,0.,0.]]),materials)
        self.assertEqual(CLASSES[p.argmax(1)[0]],'cardboard')
    def test_electronic_object_not_relabelled_as_its_plastic_casing(self):
        logits=np.full((1,9),-20.,dtype='float32');logits[0,2]=20
        materials=['glass','paper','metal','plastic','fabric','leather','food']
        p,_,_=combine_calibrated(logits,self.names(),np.array([[0.,0.,0.,1.,0.,0.,0.]]),materials)
        self.assertEqual(CLASSES[p.argmax(1)[0]],'electronic')
    def test_missing_evidence_does_not_create_false_confidence(self):
        logits=np.full((1,9),-1000.,dtype='float32');logits[0,-1]=1000
        materials=['glass','paper','metal','plastic','fabric','leather','food','other']
        p,_,_=combine_calibrated(logits,self.names(),np.array([[0.,0.,0.,0.,0.,0.,0.,1.]]),materials)
        np.testing.assert_allclose(p,np.full((1,8),.125))

if __name__=='__main__':unittest.main()

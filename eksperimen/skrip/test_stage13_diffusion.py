"""A synthetic separated graph must propagate only its supplied seed classes."""
import unittest
import numpy as np
from stage13_graph_semantic_refinement import fit_diffusion,extend

class DiffusionTests(unittest.TestCase):
    def test_inductive_extension_with_two_disconnected_components(self):
        x=np.array([[1,0],[.99,.01],[0,1],[.01,.99]],dtype='float32')
        x/=np.linalg.norm(x,axis=1,keepdims=True)
        labels=np.array([0,0,1,1]);seeds=np.array([True,False,True,False])
        distribution,steps=fit_diffusion(x,labels,seeds,k=1,alpha=.95)
        np.testing.assert_array_equal(distribution.argmax(1),labels)
        np.testing.assert_allclose(distribution.sum(1),1,atol=1e-6)
        before=x.copy();query=np.array([[.98,.02],[.02,.98]],dtype='float32');query/=np.linalg.norm(query,axis=1,keepdims=True)
        probability=extend(query,x,distribution,k=1)
        np.testing.assert_array_equal(probability.argmax(1),[0,1])
        np.testing.assert_array_equal(x,before)
        self.assertTrue(np.isfinite(probability).all())
        self.assertGreater(steps,1)

if __name__=='__main__':unittest.main()

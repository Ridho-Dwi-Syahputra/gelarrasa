"""Fixed full-source posterior descriptors for views of the SAME34human sources."""
import json,time
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.random_projection import SparseRandomProjection
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import NOTES,OLD,sha,dump,unit

def main():
    start=time.time();src=NOTES/'.runtime/human_source_aug_features.npz';z=np.load(src,allow_pickle=False)
    x=unit(z['cnn_features'].reshape(-1,768));assert x.shape==(272,768)
    metadata=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(metadata['repository'],'model.safetensors',revision=metadata['revision'],local_files_only=True);assert sha(weight)==metadata['checkpoint_sha256']
    dump(NOTES/'json/augmented_source_dictionary_plan.json',{'source_sha256':sha(src),'sources':34,'views_per_source':8,'new_labels':0,'test_rows':0,'projection':'all21841sqrtsoftmaxT1/T2;fixedSSR512seed20261108','view0':'normalizedcache;othersRAW;allunitnormalizebeforeverifiedLNscaleinverse','source_code_sha256':sha(__file__)})
    with safe_open(weight,framework='np') as f:
        gamma=f.get_tensor('head.norm.weight').astype(np.float64);beta=f.get_tensor('head.norm.bias').astype(np.float64);fc=f.get_tensor('head.fc.weight');bias=f.get_tensor('head.fc.bias')
    projection=SparseRandomProjection(n_components=512,random_state=20261108,dense_output=True).fit(sparse.csr_matrix((1,21841),dtype=np.float32));blocks=[[],[]];recovered=[]
    with threadpool_limits(limits=2):
        for first in range(0,len(x),32):
            u=x[first:first+32].astype(np.float64);aa=((u/gamma)**2).sum(1);bb=-2*(u*beta/gamma**2).sum(1);cc=((beta/gamma)**2).sum()-768
            scale=(-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa);recovered.extend(scale.tolist());logits=(u*scale[:,None]).astype(np.float32)@fc.T+bias
            for i,t in enumerate([1.,2.]):blocks[i].append(np.asarray(projection.transform(np.sqrt(softmax(logits/t,axis=1))),dtype=np.float32))
    features=np.stack([np.concatenate(v) for v in blocks],1).reshape(34,8,2,512)
    target=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False);lookup=dict(zip(target['keys'],range(len(target['keys']))));expected=target['features'][[lookup[k] for k in z['keys']]]
    parity=float(np.max(np.abs(features[:,0]-expected)));assert parity<2e-5,parity
    known=np.linalg.norm(z['cnn_features'][:,1:],axis=-1);estimated=np.array(recovered).reshape(34,8)[:,1:];norm_relative=float(np.max(np.abs(estimated-known)/known));assert norm_relative<1e-4,norm_relative
    out=NOTES/'.runtime/augmented_source_dictionary.npz';np.savez_compressed(out,keys=z['keys'],features=features,source_sha256=z['source_sha256'])
    dump(NOTES/'json/augmented_source_dictionary_extraction.json',{'status':'passed','sources':34,'views_per_source':8,'original_view_max_error':parity,'raw_scale_max_relative_error':norm_relative,'feature_sha256':sha(out),'seconds':time.time()-start})
    print(json.dumps({'status':'passed','original_view_max_error':parity,'raw_scale_relative_error':norm_relative,'seconds':time.time()-start}),flush=True)

if __name__=='__main__':main()

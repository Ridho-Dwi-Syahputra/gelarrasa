"""Stream ALL21841source probabilities into fixed projected Hellinger features.

No ontology labels, top-k truncation, image reads, test fitting or Torch model.
Input is the previously verified organizer-TRAIN-only normalized feature cache.
"""
import json
import time
import numpy as np
from scipy import sparse
from scipy.special import softmax
from sklearn.random_projection import SparseRandomProjection
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import NOTES,OLD,sha,dump


def main():
    start=time.time();path=NOTES/'.runtime/convnext_train_features.npz';z=np.load(path,allow_pickle=False);keys=z['keys'];assert len(keys)==1904 and all(k.startswith('train:') for k in keys) and not any(k in ['train:1662','train:1885'] for k in keys)
    source=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(source['repository'],'model.safetensors',revision=source['revision'],local_files_only=True);assert sha(weight)==source['checkpoint_sha256']
    with safe_open(weight,framework='np') as state:
        gamma=state.get_tensor('head.norm.weight').astype(np.float64);beta=state.get_tensor('head.norm.bias').astype(np.float64)
        fc=state.get_tensor('head.fc.weight');bias=state.get_tensor('head.fc.bias')
    original=np.load(OLD/'output/embeddings/stage11_ontology_cnn_v1/original224.npz',allow_pickle=False);lookup=dict(zip(original['keys'],range(len(original['keys']))));expected_top=original['top5_indices'][[lookup[k] for k in keys]]
    plan={'method':'all_source_distribution_projected_Hellinger_v1','train_rows':1904,'test_rows':0,'held_rows':0,'category_targets_read':False,
          'source':{k:source[k] for k in ['repository','revision','checkpoint_sha256']},'source_feature_sha256':sha(path),
          'distribution':'ALL21841softmaxcoordinates,temperatures1and2,noontologyprojectionandNOtop-ktruncation',
          'projection':'sqrtprobability,512fixedSparseRandomProjectionseed20261108,fittedononeshape-onlyzerorow;notdatafitting',
          'memory':'stream32rows,Numpyheadtensorsonly,noTorchbackbone','source_code_sha256':sha(__file__)}
    dump(NOTES/'json/full_source_distribution_feature_plan.json',plan)
    projection=SparseRandomProjection(n_components=512,random_state=20261108,dense_output=True);projection.fit(sparse.csr_matrix((1,21841),dtype=np.float32))
    blocks=[[],[]];top1=0;top5=0;mass_error=0.
    with threadpool_limits(limits=2):
        for first in range(0,1904,32):
            u=z['features'][first:first+32].astype(np.float64);aa=((u/gamma)**2).sum(1);bb=-2*(u*beta/gamma**2).sum(1);cc=((beta/gamma)**2).sum()-768
            scale=(-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa);logits=(u*scale[:,None]).astype(np.float32)@fc.T+bias
            ordered=np.argpartition(-logits,4,axis=1)[:,:5];order=np.argsort(-np.take_along_axis(logits,ordered,1),axis=1);ordered=np.take_along_axis(ordered,order,1)
            expected=expected_top[first:first+len(u)];top1+=int((ordered[:,0]==expected[:,0]).sum());top5+=int((ordered==expected).all(1).sum())
            for i,temperature in enumerate([1.,2.]):
                p=softmax(logits/temperature,axis=1);mass_error=max(mass_error,float(np.max(np.abs(p.sum(1)-1))))
                features=np.asarray(projection.transform(np.sqrt(np.maximum(p,0))),dtype=np.float32);assert np.isfinite(features).all() and features.shape==(len(u),512);blocks[i].append(features)
    parity={'top1_exact_fraction':top1/1904,'top5_exact_row_fraction':top5/1904,'softmax_mass_error':mass_error};assert top1/1904>=.999 and top5/1904>=.995
    output=NOTES/'.runtime/full_source_distribution_features.npz';np.savez_compressed(output,keys=keys,features=np.stack([np.concatenate(b) for b in blocks],1),source_sha256=z['source_sha256'])
    dump(NOTES/'json/full_source_distribution_feature_extraction.json',{'status':'passed','train_rows':1904,'test_rows':0,'source_parity':parity,'feature_sha256':sha(output),'all_source_probability_mass_retained_before_projection':True,'source_code_sha256':sha(__file__),'seconds':time.time()-start})
    print(json.dumps({'parity':parity,'seconds':time.time()-start}),flush=True)


if __name__=='__main__':main()

"""Numeric source concept indices for the eight cached views of34genuine sources."""
import json
import numpy as np
from scipy.special import softmax
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import NOTES,OLD,sha,unit

def load_augmented_source_concepts(keys):
    cache=NOTES/'.runtime/augmented_source_concepts.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=False);lookup=dict(zip(z['keys'],range(len(z['keys']))));ix=[lookup[k] for k in keys]
        return z['indices'][ix],z['probabilities'][ix],z['cnn_features'][ix],sha(cache)
    z=np.load(NOTES/'.runtime/human_source_aug_features.npz',allow_pickle=False);x=unit(z['cnn_features'].reshape(-1,768));metadata=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(metadata['repository'],'model.safetensors',revision=metadata['revision'],local_files_only=True);assert sha(weight)==metadata['checkpoint_sha256']
    with safe_open(weight,framework='np') as f:
        gamma=f.get_tensor('head.norm.weight').astype(np.float64);beta=f.get_tensor('head.norm.bias').astype(np.float64);fc=f.get_tensor('head.fc.weight');bias=f.get_tensor('head.fc.bias')
    indices=[];probs=[]
    with threadpool_limits(limits=2):
        for first in range(0,len(x),32):
            u=x[first:first+32].astype(np.float64);aa=((u/gamma)**2).sum(1);bb=-2*(u*beta/gamma**2).sum(1);cc=((beta/gamma)**2).sum()-768;scale=(-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa)
            p=softmax((u*scale[:,None]).astype(np.float32)@fc.T+bias,axis=1);idx=np.argpartition(-p,9,axis=1)[:,:10];order=np.argsort(-np.take_along_axis(p,idx,1),axis=1);idx=np.take_along_axis(idx,order,1);indices.append(idx);probs.append(np.take_along_axis(p,idx,1))
    np.savez_compressed(cache,keys=z['keys'],indices=np.concatenate(indices).reshape(34,8,10),probabilities=np.concatenate(probs).reshape(34,8,10),cnn_features=x.reshape(34,8,768))
    return load_augmented_source_concepts(keys)

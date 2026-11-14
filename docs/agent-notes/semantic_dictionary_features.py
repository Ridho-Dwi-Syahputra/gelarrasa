"""Train-only full source classifier distributions, without ontology mapping.

Recover the pre-logits scale from the affine LayerNorm variance constraint;
verify against cached source top5/eight-class evidence before using any result.
Random projection fits shape only, never train/test statistics or labels.
"""
import json
import time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from sklearn.random_projection import SparseRandomProjection
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file
from fewshot_train_only_probe import ROOT,NOTES,OLD,sha,dump


def main():
    start=time.time();torch.set_num_threads(4)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});frame=m[m.split.eq('train')&~m.is_crop&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str);assert len(keys)==1904 and not frame.parent_id.isin(['1662','1885']).any()
    source=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(source['repository'],'model.safetensors',revision=source['revision'],local_files_only=True);assert sha(weight)==source['checkpoint_sha256']
    features_path=NOTES/'.runtime/convnext_train_features.npz';z=np.load(features_path,allow_pickle=False);assert np.array_equal(z['keys'],keys) and np.array_equal(z['source_sha256'],frame.source_sha256.to_numpy(str))
    old_path=OLD/'output/embeddings/stage11_ontology_cnn_v1/original224.npz';old=np.load(old_path,allow_pickle=False)
    lookup=dict(zip(old['keys'],range(len(old['keys']))));ids=np.array([lookup[k] for k in keys]);expected_top=old['top5_indices'][ids];expected_evidence=old['evidence'][ids]
    # All arrays sliced to the1904eligibleTRAIN keys before any computation.
    plan={'method':'full_21841_source_distribution_train_descriptor_v1','train_rows':1904,'test_rows':0,'new_images':False,'category_targets_read':False,
          'reason':'oldontologyonlycapturespartofprobabilitymass;retainfullsourcedictionarywithoutmanualclassmapping',
          'norm_reconstruction':'positive quadratic root from sum(((s*u-beta)/gamma)^2)=768;LayerNormepsilon approximationvalidatedagainstcachedsourceoutputs',
          'source_temperatures':[1.,2.],'top_source_dimensions':128,'descriptor':'sqrtretainedsourceprobabilityrenormalized;fixed512SparseRandomProjectionpertemperature',
          'projection':'fitononezero-rowshapeonly,nodatastatistics,seed20261108','source':{k:source[k] for k in ['repository','revision','checkpoint_sha256']},
          'source_feature_sha256':sha(features_path),'source_code_sha256':sha(__file__),'no_VLM_or_external_dataset':True}
    dump(NOTES/'json/semantic_dictionary_feature_plan.json',plan)
    state=load_file(weight,device='cpu');gamma=state['head.norm.weight'].numpy().astype(np.float64);beta=state['head.norm.bias'].numpy().astype(np.float64)
    assert np.min(np.abs(gamma))>1e-8
    u=z['features'].astype(np.float64);aa=((u/gamma)**2).sum(1);bb=-2*(u*beta/gamma**2).sum(1);cc=((beta/gamma)**2).sum()-768
    scale=(-bb+np.sqrt(bb**2-4*aa*cc))/(2*aa);assert np.isfinite(scale).all() and (scale>0).all()
    raw=torch.from_numpy((u*scale[:,None]).astype(np.float32));w=state['head.fc.weight'];bias=state['head.fc.bias']
    with torch.inference_mode():logits=torch.nn.functional.linear(raw,w,bias)
    assert logits.shape==(1904,21841) and torch.isfinite(logits).all()
    actual_top=logits.topk(5,dim=1).indices.numpy();top1=float((actual_top[:,0]==expected_top[:,0]).mean());top5=float((actual_top==expected_top).all(1).mean())
    mapping_path=OLD/'output/embeddings/stage11_ontology_cnn_v1/category_mapping.npy';mapping=np.load(mapping_path,allow_pickle=False)
    with torch.inference_mode():prob=logits.softmax(1).numpy()
    evidence=np.stack([prob[:,mapping==k].sum(1) for k in range(8)],1);max_error=float(np.max(np.abs(evidence-expected_evidence)))
    parity={'top1_exact_fraction':top1,'top5_exact_row_fraction':top5,'maximum_mapped_evidence_error':max_error,'scope':'source-cacheparity,notcompetitionAccuracy'}
    assert top1>=.999 and top5>=.995 and max_error<.001,parity
    projection=SparseRandomProjection(n_components=512,random_state=20261108,dense_output=True)
    projection.fit(sparse.csr_matrix((1,21841),dtype=np.float32))
    blocks=[];retained=[]
    for temperature in plan['source_temperatures']:
        with torch.inference_mode():p=(logits/temperature).softmax(1);value,index=p.topk(128,dim=1)
        val=value.numpy();col=index.numpy();mass=val.sum(1);retained.append({'temperature':temperature,'retained_mass_quantiles':np.quantile(mass,[0,.25,.5,.75,1]).tolist()})
        root=np.sqrt(val/np.maximum(mass[:,None],1e-12));matrix=sparse.csr_matrix((root.ravel(),(np.repeat(np.arange(1904),128),col.ravel())),shape=(1904,21841))
        result=np.asarray(projection.transform(matrix),dtype=np.float32);assert result.shape==(1904,512) and np.isfinite(result).all();blocks.append(result)
    path=NOTES/'.runtime/semantic_dictionary_features.npz';np.savez_compressed(path,keys=keys,features=np.stack(blocks,1),source_sha256=frame.source_sha256.to_numpy(str),recovered_source_scale=scale.astype(np.float32))
    dump(NOTES/'json/semantic_dictionary_feature_extraction.json',{'status':'passed','train_rows':1904,'test_rows':0,'source_parity':parity,'retained_source_mass':retained,'feature_sha256':sha(path),'seconds':time.time()-start,'source_code_sha256':sha(__file__)})
    print(json.dumps({'parity':parity,'retained':retained,'seconds':time.time()-start},indent=2),flush=True)


if __name__=='__main__':main()

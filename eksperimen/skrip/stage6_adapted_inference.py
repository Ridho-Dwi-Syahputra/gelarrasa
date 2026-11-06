"""Infer all images only if final-block adaptation won the declared protocol."""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import pandas as pd
from PIL import Image
import torch,timm
from huggingface_hub import hf_hub_download
from torchvision.transforms.functional import to_tensor,normalize
from stage6_lastblock_training import Tail
from stage6_material_resolution import network
from stage6_semantic_ontology import CLASSES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage6_lastblock_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4)
    trained=json.loads((REPORT/'lastblock_training.json').read_text())
    assert trained['status']=='passed'
    if not trained['selected_adaptation']:
        print('Adaptation did not beat its control; named-head predictions retained.',flush=True);return
    checkpoint=OUT/'selected_tail.pt';assert sha(checkpoint)==trained['checkpoint_sha256']
    state=torch.load(checkpoint,map_location='cpu',weights_only=True)
    pointer=REPORT/'selected_material_run.json'
    if pointer.exists() and state['plan']['base_output']!=json.loads(pointer.read_text())['output']:
        print('Tail pilot belongs to an earlier semantic model; current selected family retained.',flush=True);return
    info=json.loads((ROOT/'output/embeddings/stage2_repr_v1/dino_checkpoint.json').read_text())
    weight=hf_hub_download(info['repository'],'model.safetensors',revision=info['revision'],local_files_only=True)
    model=timm.create_model(info['model'],pretrained=True,num_classes=0,dynamic_img_size=True,pretrained_cfg_overlay={'file':weight}).eval()
    tail=Tail(copy.deepcopy(model.blocks[-1]),copy.deepcopy(model.norm),network(state['kind']))
    tail.load_state_dict(state['state_dict']);tail.eval()
    model.blocks[-1]=tail.block;model.norm=tail.norm;model.eval()
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    prefix=np.load(ROOT/'output/embeddings/stage6_lastblock_v1/original224.npy',mmap_mode='r')
    keys=pd.read_csv(ROOT/'output/embeddings/stage6_lastblock_v1/original224_keys.csv').key.tolist()
    cached=dict(zip(keys,range(len(keys))));positions=dict(zip(p.key,range(len(p))))
    features=np.zeros((len(p),384),dtype='float32')
    with torch.inference_mode():
        for first in range(0,len(keys),16):
            rows=keys[first:first+16]
            tensor=torch.from_numpy(np.array(prefix[first:first+len(rows)],dtype='float32'))
            features[[positions[k] for k in rows]]=tail.features(tensor).numpy()
        uncached=p[~p.key.isin(cached)].copy();started=time.time()
        for first in range(0,len(uncached),8):
            batch=uncached.iloc[first:first+8];images=[]
            for row in batch.itertuples():
                with Image.open(ROOT/row.preview_path) as src:im=src.convert('RGB')
                images.append(normalize(to_tensor(im),[.485,.456,.406],[.229,.224,.225]))
            values=torch.nn.functional.normalize(model(torch.stack(images)),dim=1).numpy()
            features[batch.index]=values
            if first==0 or (first+8)%200==0:print(f'adapted inference {min(first+8,len(uncached))}/{len(uncached)} {time.time()-started:.0f}s',flush=True)
        probability=tail.head(torch.from_numpy(features)).softmax(-1).numpy()
    assert np.isfinite(features).all() and np.allclose(np.linalg.norm(features,axis=1),1,atol=1e-5)
    assert probability.shape==(2400,8) and np.allclose(probability.sum(1),1,atol=1e-6)
    base=ROOT/json.loads((REPORT/'selected_material_run.json').read_text())['output'] if (REPORT/'selected_material_run.json').exists() else ROOT/'output/groups/stage6_material_resolution_v1'
    original=np.load(base/'named_probabilities.npz',allow_pickle=False)
    assert np.array_equal(original['keys'],p.key.to_numpy(str));prior=original['prior_probabilities']
    pred=probability.argmax(1);ordered=np.sort(probability,axis=1)
    result=p[['key','split','image_id','pixel_sha256','processed_path']].copy()
    result['predicted_class']=np.array(CLASSES)[pred];result['uncalibrated_probability']=probability.max(1)
    result['margin']=ordered[:,-1]-ordered[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    result['cnn_prior_disagrees']=pred!=prior.argmax(1)
    result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='CNN_evidence_graph_head_then_last_ViT_block_adaptation_v1'
    result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=p.key.to_numpy(str),probabilities=probability)
    np.savez_compressed(OUT/'adapted_features.npz',keys=p.key.to_numpy(str),features=features)
    record=dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),
        checkpoint_sha256=sha(checkpoint),probability_sha256=sha(OUT/'named_probabilities.npz'),
        features_sha256=sha(OUT/'adapted_features.npz'),source_sha256=sha(__file__),
        rows=2400,prefix_cached_rows=len(keys),full_forward_rows=len(uncached),reference_archive_read=False,
        precision='cached rows use validated float16 prefix; other rows use float32 full encoder')
    (REPORT/'final_prediction_frozen.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    print('Adapted predictions frozen',flush=True)

if __name__=='__main__':main()

"""Image-only object evidence from the existing pinned ResNet50, no labels.

All category names are the upstream ImageNet ontology. This is auxiliary
evidence, not a ready eight-class waste classifier or a ground-truth label.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torchvision.models import resnet50,ResNet50_Weights
from torchvision.transforms.functional import to_tensor,normalize

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/embeddings/stage6_semantic_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(4)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    probe=pd.read_csv(ROOT/'output/embeddings/stage2_repr_v1/view_probe_ids.csv')
    probe=p.set_index('key').loc[probe.key].reset_index()
    weights=ResNet50_Weights.IMAGENET1K_V2
    model=resnet50(weights=weights).eval()
    info=dict(model='torchvision resnet50 IMAGENET1K_V2',url=weights.url,categories=weights.meta['categories'],
        reference_labels_read=False,external_images_downloaded=False,source_sha256=sha(__file__),
        interpretation='Only upstream1000 object classes; no waste labels read or produced here.')
    (REPORT/'object_evidence_provenance.json').write_text(json.dumps(info,indent=2)+'\n',encoding='utf-8')
    for view,table,flip in [('original224',p,False),('hflip224',probe,True)]:
        blocks=[];start=time.time()
        with torch.inference_mode():
            for first in range(0,len(table),12):
                tensors=[]
                for row in table.iloc[first:first+12].itertuples():
                    with Image.open(ROOT/row.preview_path) as source:im=source.convert('RGB')
                    if flip:im=ImageOps.mirror(im)
                    tensors.append(normalize(to_tensor(im),[.485,.456,.406],[.229,.224,.225]))
                blocks.append(model(torch.stack(tensors)).cpu().numpy().astype('float32'))
                if first==0 or (first+12)%240==0:print(f'ResNet object {view} {min(first+12,len(table))}/{len(table)} {time.time()-start:.0f}s',flush=True)
        logits=np.vstack(blocks);assert logits.shape==(len(table),1000) and np.isfinite(logits).all()
        np.savez_compressed(OUT/f'resnet_objects_{view}.npz',keys=table.key.to_numpy(str),logits=logits)
    print('Object evidence completed',flush=True)

if __name__=='__main__':main()

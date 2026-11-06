"""Run authors' non-VLM material CNN on whole-frame and center appearances.

Raw CNN evidence is not the final competition classifier. No archive labels,
LLM output, or external training images are read. Native images remain intact.
"""
from pathlib import Path
import hashlib,json,time,sys
ROOT=Path(__file__).resolve().parents[2]
# OpenCV5 removed its Caffe reader. Keep the legacy inference dependency local
# to this subprocess, without replacing the project's OpenCV5 installation.
legacy=ROOT/'output/runtime/opencv4'
if legacy.exists():sys.path.insert(0,str(legacy))
import cv2
import numpy as np
import pandas as pd
from PIL import Image,ImageOps

OUT=ROOT/'output/embeddings/stage6_semantic_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'
MODEL=ROOT/'output/pretrained/minc15/model'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def views(row,flip):
    with Image.open(ROOT/row.processed_path) as source:im=source.convert('RGB')
    # Two fixed views preserve all-frame context and a central material patch.
    full=im.resize((224,224),Image.Resampling.LANCZOS)
    w,h=im.size;side=max(1,round(min(w,h)*.7));left=(w-side)//2;top=(h-side)//2
    center=im.crop((left,top,left+side,top+side)).resize((224,224),Image.Resampling.LANCZOS)
    if flip:full=ImageOps.mirror(full);center=ImageOps.mirror(center)
    return [cv2.cvtColor(np.asarray(v),cv2.COLOR_RGB2BGR) for v in [full,center]]

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);cv2.setNumThreads(4)
    checkpoint=json.loads((REPORT/'material_checkpoint.json').read_text())
    for name,data in checkpoint['files'].items():assert sha(MODEL/name)==data['sha256']
    categories=(MODEL/'categories.txt').read_text().splitlines()
    net=cv2.dnn.readNetFromCaffe(str(MODEL/'deploy-googlenet.prototxt'),str(MODEL/'minc-googlenet.caffemodel'))
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV);net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    probe=pd.read_csv(ROOT/'output/embeddings/stage2_repr_v1/view_probe_ids.csv')
    probe=p.set_index('key').loc[probe.key].reset_index()
    provenance=dict(categories=categories,model='MINC authors GoogLeNet Caffe CNN',
        architecture='image-only, not SigLIP/CLIP or another VLM',mean_bgr=[104,117,124],
        mean_source='Bell etal CVPR2015 section4.1, RGB124/117/104',
        views=['full frame resized224 (aspect distorted)','center square .7 of shorter side resized224'],
        aggregation='mean of two-view probabilities, predeclared',checkpoint=checkpoint,opencv=cv2.__version__,
        reference_archive_read=False,source_sha256=sha(__file__))
    (REPORT/'material_evidence_provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    for name,table,flip in [('original224',p,False),('hflip224',probe,True)]:
        blocks=[];disagreements=[];started=time.time()
        for first in range(0,len(table),8):
            batch=[image for row in table.iloc[first:first+8].itertuples() for image in views(row,flip)]
            blob=cv2.dnn.blobFromImages(batch,scalefactor=1,size=(224,224),mean=(104,117,124),swapRB=False,crop=False)
            net.setInput(blob);probs=net.forward().reshape(len(batch),-1)
            assert probs.shape[1]==len(categories) and np.isfinite(probs).all()
            assert np.allclose(probs.sum(1),1,atol=1e-4)
            paired=probs.reshape(-1,2,len(categories));blocks.append(paired.mean(1));disagreements.extend(paired[:,0].argmax(1)!=paired[:,1].argmax(1))
            if first==0 or (first+8)%200==0:print(f'Material CNN {name} {min(first+8,len(table))}/{len(table)} {time.time()-started:.0f}s',flush=True)
        np.savez_compressed(OUT/f'minc_materials_{name}.npz',keys=table.key.to_numpy(str),probabilities=np.vstack(blocks),view_disagrees=np.asarray(disagreements))
    print('Material evidence completed',flush=True)

if __name__=='__main__':main()

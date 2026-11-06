"""Pinned recycling-domain ResNet CNN as a base, no external images or APIs.

Source class order and preprocessing are read from the author's pinned app
as data, never executed. Downstream named heads are trained on competition fit.
"""
from pathlib import Path
import ast,hashlib,json,time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
from torchvision import transforms
from torchvision.models import resnet50
from huggingface_hub import hf_hub_download

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/embeddings/stage7_waste_cnn_v1'
REPORT=ROOT/'reports/errors/stage7_waste_cnn_20261106'
REPO='kendrickfff/my_resnet50_garbage_classificationv1.2'
REVISION='c689678ec70cf53baa0a081644d29901e59f4a4a'
APP_REPO='kendrickfff/resnet50_garbage_classification_v1.2'
APP_REVISION='ab3b912a730f2b6f1986aba8986fbc0d8e5e36b5'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    path=hf_hub_download(REPO,'resnet50_garbage_classificationv1.2.pth',revision=REVISION)
    app=hf_hub_download(APP_REPO,'app.py',repo_type='space',revision=APP_REVISION)
    tree=ast.parse(Path(app).read_text(encoding='utf-8'))
    assignments=[node for node in ast.walk(tree) if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='class_names' for t in node.targets)]
    assert len(assignments)==1
    classes=ast.literal_eval(assignments[0].value)
    assert classes==['battery','biological','brown-glass','cardboard','clothes','green-glass','metal','paper','plastic','shoes','trash','white-glass']
    state=torch.load(path,map_location='cpu',weights_only=True)
    model=resnet50(weights=None);model.fc=torch.nn.Linear(2048,len(classes));model.load_state_dict(state);model.eval()
    weight=model.fc.weight.detach().clone();bias=model.fc.bias.detach().clone();model.fc=torch.nn.Identity()
    transform=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    record=dict(repository=REPO,revision=REVISION,checkpoint_sha256=sha(path),license='MIT per model card',
        model='torchvision ResNet50, 12-class recycling-domain pretrained base, image-only CNN',
        pretraining_dataset='mostafaabla/garbage-classification, per uploader; no dataset images downloaded',
        generic_ImageNet_initialization=False,source_class_order=classes,
        source_app=f'https://huggingface.co/spaces/{APP_REPO}/blob/{APP_REVISION}/app.py',source_app_sha256=sha(app),
        source_app_executed=False,transform='authors Resize256 ->CenterCrop224 ->ImageNet mean/std',
        reference_archive_read=False,external_image_dataset_downloaded=False,source_sha256=sha(__file__),
        qualification='Task-specific pretrained base, not trained from scratch here. Participant head adaptation and provenance are required; no automatic claim of organizer approval.')
    (REPORT/'checkpoint_and_ontology.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    probe=pd.read_csv(ROOT/'output/embeddings/stage2_repr_v1/view_probe_ids.csv');probe=p.set_index('key').loc[probe.key].reset_index()
    for name,table,flip in [('original224',p,False),('hflip224',probe,True)]:
        features=[];probabilities=[];started=time.time()
        with torch.inference_mode():
            for first in range(0,len(table),12):
                images=[]
                for row in table.iloc[first:first+12].itertuples():
                    with Image.open(ROOT/row.processed_path) as src:im=src.convert('RGB')
                    if flip:im=ImageOps.mirror(im)
                    images.append(transform(im))
                raw=model(torch.stack(images));prob=torch.nn.functional.linear(raw,weight,bias).softmax(-1)
                features.append(torch.nn.functional.normalize(raw,dim=1).numpy().astype('float32'));probabilities.append(prob.numpy().astype('float32'))
                if first==0 or (first+12)%240==0:print(f'waste CNN {name} {min(first+12,len(table))}/{len(table)} {time.time()-started:.0f}s',flush=True)
        np.savez_compressed(OUT/f'{name}.npz',keys=table.key.to_numpy(str),features=np.vstack(features),probabilities=np.vstack(probabilities))
    print('Waste-domain CNN features ready',flush=True)

if __name__=='__main__':main()

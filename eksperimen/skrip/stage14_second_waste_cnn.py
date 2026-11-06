"""Second independent image-only waste CNN, pinned, no remote code execution."""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd,torch
from PIL import Image,ImageOps
from torchvision import transforms
from torchvision.models import efficientnet_v2_s
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/embeddings/stage14_second_waste_cnn_v1'
REPORT=ROOT/'reports/errors/stage14_second_waste_cnn_20261106'
REPO='attilaultzindur/garbage_classifier_effnetv2s_ft'
REV='35f28e0f432e4d874d88bb214da6cb96553f10fd'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    path=hf_hub_download(REPO,'model.safetensors',revision=REV)
    config_path=hf_hub_download(REPO,'config.json',revision=REV)
    config=json.loads(Path(config_path).read_text());classes=config['label_names']
    assert classes==['battery','biological','cardboard','clothes','glass','metal','paper','plastic','shoes','trash']
    model=efficientnet_v2_s(weights=None);model.classifier=torch.nn.Sequential(torch.nn.Linear(1280,256),torch.nn.ReLU(inplace=True),torch.nn.Dropout(.5),torch.nn.Linear(256,10))
    model.load_state_dict(load_file(path,device='cpu'));model.eval()
    head=model.classifier;model.classifier=torch.nn.Identity()
    transform=transforms.Compose([transforms.Resize((224,224)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
    record=dict(repository=REPO,revision=REV,checkpoint_sha256=sha(path),config_sha256=sha(config_path),source_class_order=classes,
        model='torchvision efficientnet_v2_s, image-only task-specific pretrained CNN; 1280D embedding',
        transform='author handler: warp224 RGB, ImageNet mean/std',remote_code_executed=False,source_sha256=sha(__file__),
        license='not declared in model card',external_images_downloaded=False,pretraining_dataset_unspecified=True,pretraining_overlap_unknown=True,
        reference_archive_read=False,qualification='Experimental pretrained base. Requires participant adaptation; no claim of independent accuracy or organizer approval.')
    (REPORT/'checkpoint_and_ontology.json').write_text(json.dumps(record,indent=2)+'\n',encoding='utf-8')
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    ids=pd.read_csv(ROOT/'output/embeddings/stage2_repr_v1/view_probe_ids.csv');probe=p.set_index('key').loc[ids.key].reset_index()
    for name,table,flip in [('original224',p,False),('hflip224',probe,True)]:
        features=[];probabilities=[];started=time.time()
        with torch.inference_mode():
            for first in range(0,len(table),12):
                batch=[]
                for row in table.iloc[first:first+12].itertuples():
                    with Image.open(ROOT/row.processed_path) as src:im=src.convert('RGB')
                    if flip:im=ImageOps.mirror(im)
                    batch.append(transform(im))
                raw=model(torch.stack(batch));prob=head(raw).softmax(-1)
                features.append(torch.nn.functional.normalize(raw,dim=1).numpy().astype('float32'));probabilities.append(prob.numpy().astype('float32'))
                if first==0 or (first+12)%240==0:print(f'second waste CNN {name} {min(first+12,len(table))}/{len(table)} {time.time()-started:.0f}s',flush=True)
        np.savez_compressed(OUT/f'{name}.npz',keys=table.key.to_numpy(str),features=np.vstack(features),probabilities=np.vstack(probabilities))
    print('Second waste CNN features ready',flush=True)

if __name__=='__main__':main()

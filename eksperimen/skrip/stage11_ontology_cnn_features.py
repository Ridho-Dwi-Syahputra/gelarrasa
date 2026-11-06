"""Generic ImageNet22k ConvNeXt + public WordNet ontology, no VLM or images.

The ontology maps public classifier synsets to task categories. It is lexical
metadata, not competition annotations. Unmapped probability mass is retained.
"""
from pathlib import Path
import hashlib,json,time,warnings
import numpy as np,pandas as pd
from PIL import Image,ImageOps
import nltk,torch,timm
from timm.data import ImageNetInfo,create_transform,resolve_model_data_config
from huggingface_hub import hf_hub_download
from stage6_semantic_ontology import CLASSES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/embeddings/stage11_ontology_cnn_v1'
REPORT=ROOT/'reports/errors/stage11_ontology_cnn_20261106'
REPO='timm/convnext_tiny.fb_in22k'
REVISION='342f2588bb2ea79d793d0712c9be16ec6ae0149a'
ROOTS={
 'glass':['glass.n.01','glass.n.02','glassware.n.01'],
 'paper':['paper.n.01','newspaper.n.01','paper.n.04','document.n.01','book.n.01'],
 'electronic':['electronic_equipment.n.01','electrical_device.n.01','battery.n.02','computer.n.01','computer_hardware.n.01'],
 'metal':['metallic_element.n.01','alloy.n.01','metalware.n.01','can.n.01'],
 'plastic':['plastic.n.01','plastic_bag.n.01','credit_card.n.01'],
 'cardboard':['cardboard.n.01','carton.n.02'],
 'textile':['fabric.n.01','garment.n.01','leather.n.01'],
 'foodwaste':['food.n.01','food.n.02','garbage.n.01']}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    wordnet_dir=Path.home()/'.cache/gelar_rasa_wordnet';nltk.data.path.insert(0,str(wordnet_dir))
    from nltk.corpus import wordnet as wn
    root_synsets={}
    unavailable=[]
    for category,names in ROOTS.items():
        for name in names:
            try:root_synsets[wn.synset(name)]=CLASSES.index(category)
            except Exception:unavailable.append(name)
    info=ImageNetInfo('imagenet-22k');names=info.label_names();assert len(names)==21841
    mapping=np.full(len(names),-1,dtype='int32');unresolved=[]
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i,name in enumerate(names):
            synset=wn.synset_from_pos_and_offset(name[0],int(name[1:]))
            if synset is None:unresolved.append(name);continue
            distances=dict(synset.hypernym_distances())
            matches=[(distances[root],category) for root,category in root_synsets.items() if root in distances]
            if matches:mapping[i]=min(matches)[1]
    assert set(mapping[mapping>=0])==set(range(8))
    np.save(OUT/'category_mapping.npy',mapping)
    weight=hf_hub_download(REPO,'model.safetensors',revision=REVISION,local_files_only=True)
    model=timm.create_model('convnext_tiny.fb_in22k',pretrained=True,pretrained_cfg_overlay={'file':weight}).eval()
    config=resolve_model_data_config(model);transform=create_transform(**config,is_training=False)
    record=dict(repository=REPO,revision=REVISION,checkpoint_sha256=sha(weight),architecture='ConvNeXt image-only supervised CNN',
        pretrained_dataset='ImageNet22k, generic base',classes=21841,wordnet_version=wn.get_version(),ontology_roots=ROOTS,
        missing_roots=unavailable,unresolved_synsets=unresolved,mapped_counts=dict(zip(CLASSES,np.bincount(mapping[mapping>=0],minlength=8).tolist())),
        mapping_policy='Closest hypernym root, each public model class at most one category; unmapped mass never treated as known.',
        mapping_sha256=sha(OUT/'category_mapping.npy'),transform=config,source_sha256=sha(__file__),
        external_image_dataset_downloaded=False,external_lexical_ontology_used=True,competition_labels_read=False,reference_archive_read=False,
        source='https://huggingface.co/timm/convnext_tiny.fb_in22k',ontology_source='https://www.nltk.org/howto/wordnet.html')
    (REPORT/'checkpoint_and_ontology.json').write_text(json.dumps(record,indent=2,default=str)+'\n',encoding='utf-8')
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    probe=pd.read_csv(ROOT/'output/embeddings/stage2_repr_v1/view_probe_ids.csv');probe=p.set_index('key').loc[probe.key].reset_index()
    projection=torch.from_numpy((mapping[:,None]==np.arange(8)[None,:]).astype('float32'))
    for name,table,flip in [('original224',p,False),('hflip224',probe,True)]:
        feature_blocks=[];scores=[];tops=[];started=time.time()
        with torch.inference_mode():
            for first in range(0,len(table),12):
                inputs=[]
                for row in table.iloc[first:first+12].itertuples():
                    with Image.open(ROOT/row.processed_path) as src:im=src.convert('RGB')
                    if flip:im=ImageOps.mirror(im)
                    inputs.append(transform(im))
                spatial=model.forward_features(torch.stack(inputs));raw=model.forward_head(spatial,pre_logits=True)
                logits=model.get_classifier()(raw);probability=logits.softmax(-1)
                scores.append((probability@projection).numpy().astype('float32'))
                feature_blocks.append(torch.nn.functional.normalize(raw,dim=1).numpy().astype('float32'))
                tops.append(probability.topk(5,dim=1).indices.numpy().astype('int32'))
                if first==0 or (first+12)%240==0:print(f'ontology CNN {name} {min(first+12,len(table))}/{len(table)} {time.time()-started:.0f}s',flush=True)
        evidence=np.vstack(scores);coverage=evidence.sum(1)
        conditional=evidence/np.maximum(coverage[:,None],1e-8)
        np.savez_compressed(OUT/f'{name}.npz',keys=table.key.to_numpy(str),features=np.vstack(feature_blocks),
            evidence=evidence,coverage=coverage,conditional=conditional,top5_indices=np.vstack(tops))
    print('Ontology CNN extraction complete',flush=True)

if __name__=='__main__':main()

"""Global specificity fallback for electrical lighting; train-only, no human targets."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
import nltk
import torch
import timm
from timm.data import ImageNetInfo, create_transform, resolve_model_data_config
from huggingface_hub import hf_hub_download
from fewshot_train_only_probe import ROOT, NOTES, OLD, sha, dump


def main():
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    roots=['electric_lamp.n.01','fluorescent_lamp.n.01','sodium-vapor_lamp.n.01','neon_lamp.n.01','light-emitting_diode.n.01']
    dump(NOTES/'json/lighting_specificity_plan.json',{
        'roots':roots,'generic_shape_fallback':'bulb.n.04 is shape only, never itself an electrical label',
        'gate':'top1 electrical-lighting descendant OR generic bulb shape top1 and at least2 specific lighting descendants in top5',
        'horizontal_flip_same_gate_required':True,'scope':'all working full-frame train; no supplied ID targeting',
        'human_labels_used':0,'test_images_processed':0,'review_labels_used':False,'active_data_modified':False})
    nltk.data.path.insert(0,str(Path.home()/'.cache/gelar_rasa_wordnet'))
    from nltk.corpus import wordnet as wn
    descendants=set()
    for name in roots:
        root=wn.synset(name);descendants.add(root);descendants.update(root.closure(lambda s:s.hyponyms()))
    vocabulary=ImageNetInfo('imagenet-22k').label_names()
    positions=dict(zip(vocabulary,range(len(vocabulary))))
    lighting={positions[f'{s.pos()}{s.offset():08d}'] for s in descendants if f'{s.pos()}{s.offset():08d}' in positions}
    shape=wn.synset('bulb.n.04');generic=positions[f'{shape.pos()}{shape.offset():08d}']
    def gate(values):
        hits=sum(int(v) in lighting for v in values)
        return bool(int(values[0]) in lighting or (int(values[0])==generic and hits>=2)),hits
    cached=np.load(OLD/'output/embeddings/stage11_ontology_cnn_v1/original224.npz',allow_pickle=False)
    indices=dict(zip(cached['keys'],range(len(cached['keys']))))
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').copy()
    assert len(frame)==1906
    candidates=[]
    for row in frame.itertuples():
        values=cached['top5_indices'][indices[row.parent_key]];accepted,hits=gate(values)
        if accepted and row.assigned_class!='electronic':candidates.append((row,values,hits))
    flip=np.load(OLD/'output/embeddings/stage11_ontology_cnn_v1/hflip224.npz',allow_pickle=False)
    flip_lookup=dict(zip(flip['keys'],flip['top5_indices']))
    missing=[row for row,values,hits in candidates if row.parent_key not in flip_lookup]
    if missing:
        torch.set_num_threads(4)
        info=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
        weight=hf_hub_download(info['repository'],'model.safetensors',revision=info['revision'],local_files_only=True)
        assert sha(weight)==info['checkpoint_sha256']
        model=timm.create_model('convnext_tiny.fb_in22k',pretrained=True,pretrained_cfg_overlay={'file':weight}).eval()
        transform=create_transform(**resolve_model_data_config(model),is_training=False)
        with torch.inference_mode():
            for start in range(0,len(missing),8):
                batch=[];items=missing[start:start+8]
                for row in items:
                    assert sha(ROOT/row.source_path)==row.source_sha256
                    with Image.open(ROOT/row.source_path) as source:im=ImageOps.mirror(ImageOps.exif_transpose(source).convert('RGB'))
                    batch.append(transform(im))
                predictions=model(torch.stack(batch)).topk(5,dim=1).indices.cpu().numpy()
                for row,values in zip(items,predictions):flip_lookup[row.parent_key]=values
    decisions=[]
    result=frame[['parent_key','parent_id','assigned_class']].rename(columns={'parent_key':'key','parent_id':'image_id','assigned_class':'baseline_class'}).copy()
    result['predicted_class']=result.baseline_class
    for row,values,hits in candidates:
        accepted,other_hits=gate(flip_lookup[row.parent_key])
        if accepted:result.loc[result.key.eq(row.parent_key),'predicted_class']='electronic'
        decisions.append({'image_id':row.parent_id,'baseline_class':row.assigned_class,'original_specific_hits':hits,
                          'hflip_specific_hits':other_hits,'accepted':accepted,
                          'original_top5':[wn.synset_from_pos_and_offset(vocabulary[int(i)][0],int(vocabulary[int(i)][1:])).name() for i in values]})
    result.to_csv(NOTES/'table-csv/lighting_specificity_selected.csv',index=False)
    report={'status':'candidate_frozen','manual_training_labels_used':0,'test_rows_processed':0,
            'train_rows_scanned':1906,'accepted_changes':int(result.predicted_class.ne(result.baseline_class).sum()),
            'decisions':decisions,'prediction_sha256':sha(NOTES/'table-csv/lighting_specificity_selected.csv'),
            'active_data_modified':False,'semantic_accuracy':None,'protected_sha256':protected}
    assert all(sha(ROOT/name)==value for name,value in protected.items())
    dump(NOTES/'json/lighting_specificity_frozen.json',report)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

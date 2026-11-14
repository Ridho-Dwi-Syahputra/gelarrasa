"""Global source-ontology coverage repair, no per-image or manual targets.

Only train images are processed. WordNet synsets are public classifier metadata.
New assignments require top-five semantic support and horizontal-view agreement.
"""
import ast
from pathlib import Path
import json
import time
import warnings

import nltk
import numpy as np
import pandas as pd
from PIL import Image,ImageOps
import torch
import timm
from timm.data import ImageNetInfo,create_transform,resolve_model_data_config
from huggingface_hub import hf_hub_download
from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump


ADDITIONS={
 'paper':['paper_towel.n.01','tissue.n.02','toilet_tissue.n.01'],
 'electronic':['calculator.n.02','electrical_cable.n.01'],
 'metal':['coin.n.01'],
 'plastic':['polyethylene.n.01','polystyrene.n.01'],
 'textile':['footwear.n.01'],
 'foodwaste':['bone.n.01','bone.n.02','skull.n.01','shell.n.04','shell.n.05','peel.n.02','rind.n.01']}


def main():
    started=time.time();torch.set_num_threads(4)
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'global WordNet source-class coverage extension with top-five/hflip checks','new_ontology_roots':ADDITIONS,
          'scope':'public classifier class metadata; no per-image rules or reference labels',
          'minimum_same_category_labels_in_top5':2,'minimum_rank_evidence_share':.6,'rank_weights':[1.,.5,.25,.125,.0625],
          'only_changed_source_ontology_top1':True,'horizontal_view_agreement_required':True,
          'manual_labels_consumed':0,'test_rows_processed':0,'reference_archive_read':False,
          'source_semantics_are_weak_priors_not_verified_material_truth':True,'external_image_data_downloaded':False,
          'external_lexical_metadata_used':True,'protected_sha256':protected,'researcher_has_seen_prior_private_audits':True}
    dump(NOTES/'json/automatic_ontology_plan.json',plan)
    tree=ast.parse((OLD/'eksperimen/skrip/stage11_ontology_cnn_features.py').read_text())
    old_roots=next(ast.literal_eval(node.value) for node in tree.body if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='ROOTS' for t in node.targets))
    nltk.data.path.insert(0,str(Path.home()/'.cache/gelar_rasa_wordnet'))
    from nltk.corpus import wordnet as wn
    root_synsets=[];definitions={}
    for category,names in old_roots.items():
        for name in names:
            try:root_synsets.append((wn.synset(name),CLASSES.index(category)))
            except Exception:pass
    for category,names in ADDITIONS.items():
        for name in names:
            synset=wn.synset(name);root_synsets.append((synset,CLASSES.index(category)));definitions[name]=synset.definition()
    names=ImageNetInfo('imagenet-22k').label_names();old_map=np.load(OLD/'output/embeddings/stage11_ontology_cnn_v1/category_mapping.npy',allow_pickle=False)
    mapping=np.full(len(names),-1,dtype='int32')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i,name in enumerate(names):
            syn=wn.synset_from_pos_and_offset(name[0],int(name[1:]))
            if syn is None:continue
            distances={}
            for node,distance in syn.hypernym_distances():distances[node]=min(distances.get(node,10**9),distance)
            matches=[(distances[node],category) for node,category in root_synsets if node in distances]
            if matches:mapping[i]=min(matches)[1]
    frame=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});frame=frame[frame.split.eq('train')&~frame.is_crop][['parent_key','parent_id','source_path','source_sha256','assigned_class']].rename(columns={'parent_key':'key','parent_id':'image_id'}).sort_values('image_id').reset_index(drop=True)
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    z=np.load(OLD/'output/embeddings/stage11_ontology_cnn_v1/original224.npz',allow_pickle=False);index={key:i for i,key in enumerate(z['keys'])}
    top5=z['top5_indices'][[index[key] for key in frame.key]]
    def evidence(indices):
        labels=mapping[indices];category=int(labels[0])
        if category<0:return None
        weights=np.array(plan['rank_weights']);known=labels>=0
        support=int((labels==category).sum());share=float(weights[labels==category].sum()/max(weights[known].sum(),1e-8))
        return {'category':CLASSES[category],'same_category_top5':support,'rank_evidence_share':share}
    candidates=[]
    for i,indices in enumerate(top5):
        if mapping[indices[0]]==old_map[indices[0]]:continue
        e=evidence(indices)
        if e is not None and e['same_category_top5']>=2 and e['rank_evidence_share']>=.6 and e['category']!=frame.iloc[i].assigned_class:
            candidates.append((i,e))
    print(f'Global ontology candidate changes before hflip: {len(candidates)}',flush=True)
    flip_lookup={}
    flip=np.load(OLD/'output/embeddings/stage11_ontology_cnn_v1/hflip224.npz',allow_pickle=False)
    for key,values in zip(flip['keys'],flip['top5_indices']):flip_lookup[str(key)]=values
    missing=[i for i,e in candidates if frame.iloc[i].key not in flip_lookup]
    if missing:
        info=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
        weight=hf_hub_download(info['repository'],'model.safetensors',revision=info['revision'],local_files_only=True)
        assert sha(weight)==info['checkpoint_sha256']
        model=timm.create_model('convnext_tiny.fb_in22k',pretrained=True,pretrained_cfg_overlay={'file':weight}).eval()
        transform=create_transform(**resolve_model_data_config(model),is_training=False)
        with torch.inference_mode():
            for start in range(0,len(missing),8):
                rows=missing[start:start+8];batch=[]
                for i in rows:
                    path=ROOT/frame.iloc[i].source_path;assert sha(path)==frame.iloc[i].source_sha256
                    with Image.open(path) as source:im=ImageOps.mirror(ImageOps.exif_transpose(source).convert('RGB'))
                    batch.append(transform(im))
                values=model(torch.stack(batch)).topk(5,dim=1).indices.cpu().numpy()
                for i,indices in zip(rows,values):flip_lookup[frame.iloc[i].key]=indices
    results=[];predicted=frame.assigned_class.to_numpy().copy()
    for i,e in candidates:
        other=evidence(flip_lookup[frame.iloc[i].key]);accepted=other is not None and other['category']==e['category'] and other['same_category_top5']>=2 and other['rank_evidence_share']>=.6
        if accepted:predicted[i]=e['category']
        results.append({'key':frame.iloc[i].key,'image_id':frame.iloc[i].image_id,'baseline_category':frame.iloc[i].assigned_class,'ontology_category':e['category'],'original_support':e['same_category_top5'],'original_rank_share':e['rank_evidence_share'],'hflip_agrees':accepted,'source_top1_synset':str(names[top5[i,0]])})
    frame['predicted_class']=predicted;frame['method']='expanded_public_ontology_with_hflip_v1';frame['ground_truth_verified']=False
    output=NOTES/'table-csv/automatic_ontology_selected.csv';frame.to_csv(output,index=False)
    pd.DataFrame(results).to_csv(NOTES/'table-csv/automatic_ontology_decisions.csv',index=False)
    report={'status':'frozen_before_reference_audit','method':'expanded_public_ontology_with_hflip_v1','manual_labels_consumed':0,'train_rows':1906,'test_rows_processed':0,
            'newly_mapped_source_classes':int(((old_map<0)&(mapping>=0)).sum()),'changed_source_class_mappings':int((old_map!=mapping).sum()),
            'candidate_changes':len(candidates),'accepted_changes':int((predicted!=frame.assigned_class.to_numpy()).sum()),'definitions':definitions,
            'prediction_sha256':sha(output),'source_code_sha256':sha(__file__),'protected_sha256':protected,'semantic_accuracy':None,'active_data_modified':False,'seconds':time.time()-started}
    dump(NOTES/'json/automatic_ontology_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k not in ['definitions','protected_sha256']},indent=2),flush=True)


if __name__=='__main__':main()

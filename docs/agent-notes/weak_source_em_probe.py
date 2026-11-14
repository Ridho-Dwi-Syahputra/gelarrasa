"""Train-only weak-source error model, selected using 35 budgeted human anchors.

Reads cached non-VLM model outputs predating human refinement. No annotation
archive, review inventory, test row, or submission enters fit or selection.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT / 'docs/agent-notes'
OLD = ROOT / 'arsip/sebelum_rapi_20261107'
CLASSES = ['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_arrays(keys):
    translation={'battery':'electronic','biological':'foodwaste','brown-glass':'glass',
                 'green-glass':'glass','white-glass':'glass','clothes':'textile','shoes':'textile',
                 **{c:c for c in CLASSES}}
    sources=[];coverage=[];paths=[]
    def load(relative):
        path=OLD/relative;paths.append(path);z=np.load(path,allow_pickle=False)
        index={k:i for i,k in enumerate(z['keys'])};return z,np.array([index[k] for k in keys])
    z,i=load('output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz')
    sources.append(z['prior_probabilities'][i]);coverage.append(np.ones(len(keys)))
    for stage,folder,metadata in [
        (7,'stage7_waste_cnn_v1','reports/errors/stage7_waste_cnn_20261106/checkpoint_and_ontology.json'),
        (14,'stage14_second_waste_cnn_v1','reports/errors/stage14_second_waste_cnn_20261106/checkpoint_and_ontology.json')]:
        z,i=load(f'output/embeddings/{folder}/original224.npz')
        order=json.loads((OLD/metadata).read_text())['source_class_order']
        mapped=np.zeros((len(keys),8))
        for col,label in enumerate(order):
            if label in translation:mapped[:,CLASSES.index(translation[label])]+=z['probabilities'][i,col]
        mass=mapped.sum(1);sources.append(mapped/np.maximum(mass[:,None],1e-12));coverage.append(mass)
    z,i=load('output/embeddings/stage11_ontology_cnn_v1/original224.npz')
    sources.append(z['conditional'][i]);coverage.append(z['coverage'][i])
    return np.stack(sources,1),np.stack(coverage,1),{str(p.relative_to(ROOT)):digest(p) for p in paths}


def fit_em(observed,active,gold,diagonal,human_weight):
    n,s,k=observed.shape
    target=np.eye(k)[np.maximum(gold,0)]
    human=gold>=0
    weighted=observed*active[:,:,None]
    responsibility=weighted.sum(1)+.05
    responsibility/=responsibility.sum(1,keepdims=True)
    responsibility[human]=target[human]
    sample_weight=np.where(human,human_weight,1.)
    prior=.2*np.ones((s,k,k))+diagonal*np.eye(k)[None]
    for iteration in range(80):
        theta=prior+np.einsum('nc,nsk,n->sck',responsibility,weighted,sample_weight,optimize=True)
        theta/=theta.sum(2,keepdims=True)
        class_prior=responsibility.sum(0)+20
        class_prior/=class_prior.sum()
        logits=np.log(class_prior)[None]+np.einsum('nsk,sck->nc',weighted,np.log(np.maximum(theta,1e-10)),optimize=True)
        updated=softmax(logits,axis=1);updated[human]=target[human]
        change=np.max(np.abs(updated-responsibility));responsibility=updated
        if change<1e-6:break
    return theta,class_prior,iteration+1


def predict(observed,active,theta,prior):
    logits=np.log(prior)[None]+np.einsum('nsk,sck->nc',observed*active[:,:,None],np.log(np.maximum(theta,1e-10)),optimize=True)
    return softmax(logits,axis=1)


def main():
    protected={p:digest(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    config=[{'diagonal':d,'human_weight':h,'threshold':t} for d,h,t in itertools.product([2.,10.],[10.,50.],[.5,.8])]
    plan={'method':'human_anchored_soft_Dawid_Skene_EM','configs':config,'max_iterations':80,
          'fit_scope':'eligible_train_full_frames_only','human_sources':35,'test_rows_in_fit':0,
          'selection':'max35source_group_OOFmacro_recall_then_matches_then_list_order',
          'publication_guard':'>=29of35andmacro>=.8;atleast2reliable_sourcesagree;posterior>=.9;max40changes',
          'archive_or_full_review_read_by_fit':False,'new_manual_labels':0,
          'limitation':'reused35developmentanchors;conditional_source_independence_is_approximate',
          'references':['https://arxiv.org/abs/1711.10160','https://doi.org/10.2307/2346806']}
    (NOTES/'json/weak_source_em_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=m[m.split.eq('train')&~m.is_crop&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str);assert len(keys)==1887
    anchors=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(anchors)==35
    lookup=dict(zip(keys,range(len(keys))));positions=np.array([lookup[k] for k in anchors.key])
    gold=np.full(len(frame),-1);gold[positions]=[CLASSES.index(c) for c in anchors.assigned_class]
    group_info=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    folds=group_info.fold.to_numpy();groups=group_info.group.to_numpy()
    observed,coverage,sources=source_arrays(keys)
    history=[];outputs=[]
    for cfg in config:
        active=(observed.max(2)>=cfg['threshold'])&(coverage>=np.array([0.,.5,.5,.1])[None])
        oof=np.zeros((len(frame),8));iterations=[]
        for fold in range(5):
            fit=folds!=fold;query=~fit
            assert not set(groups[fit])&set(groups[query])
            theta,prior,count=fit_em(observed[fit],active[fit],gold[fit],cfg['diagonal'],cfg['human_weight'])
            oof[query]=predict(observed[query],active[query],theta,prior);iterations.append(count)
        pred=oof[positions].argmax(1)
        record={**cfg,'matches':int((pred==gold[positions]).sum()),
                'macro_recall':float(balanced_accuracy_score(gold[positions],pred)),'max_iterations':max(iterations)}
        history.append(record);outputs.append(oof)
        print(record,flush=True)
    winner=max(range(len(history)),key=lambda i:(history[i]['macro_recall'],history[i]['matches']))
    cfg=config[winner];active=(observed.max(2)>=cfg['threshold'])&(coverage>=np.array([0.,.5,.5,.1])[None])
    theta,prior,iterations=fit_em(observed,active,gold,cfg['diagonal'],cfg['human_weight'])
    probability=predict(observed,active,theta,prior);primary=probability.argmax(1)
    support=((observed.argmax(2)==primary[:,None])&active).sum(1)
    accepted=history[winner]['matches']>=29 and history[winner]['macro_recall']>=.8
    gate=(probability.max(1)>=.9)&(support>=2)&(gold<0)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate&=primary!=current
    order=np.argsort(-probability.max(1),kind='stable');selected=order[gate[order]][:40]
    gate[:]=False
    if accepted:gate[selected]=True
    candidate=frame[['parent_key','parent_id','assigned_class']].copy()
    candidate['candidate_class']=np.array(CLASSES)[primary]
    candidate['posterior']=probability.max(1);candidate['source_support']=support
    candidate['accepted_change']=gate
    candidate['proposed_class']=np.where(gate,np.array(CLASSES)[primary],candidate.assigned_class)
    output=NOTES/'table-csv/weak_source_em_candidate.csv';candidate.to_csv(output,index=False)
    pd.DataFrame(history).to_csv(NOTES/'table-csv/weak_source_em_development.csv',index=False)
    report={'status':'frozen_before_posthoc_audit','winner':history[winner],'human_sources':35,
            'development_guard_passed':bool(accepted),'proposed_changes':int(gate.sum()),
            'source_output_sha256':sources,'candidate_sha256':digest(output),
            'test_rows_in_fit':0,'archive_labels_read_by_experiment':False,'active_outputs_modified':False,
            'semantic_accuracy':None,'protected_sha256':protected}
    (NOTES/'json/weak_source_em_frozen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert all(digest(ROOT/p)==h for p,h in protected.items())
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':main()

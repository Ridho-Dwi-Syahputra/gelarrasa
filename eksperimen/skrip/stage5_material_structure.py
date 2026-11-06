"""Global/local and 32-prototype ablation. No reference labels or display names.

This tests limited ideas motivated by IIC and material recognition, not a
replication of IIC/SPICE or a semantic category classifier.
"""
from pathlib import Path
import hashlib
import html
import itertools
import json
import os
import time
import joblib
import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
FEATURE=ROOT/'output/embeddings/stage5_patch_v1'
OUT=ROOT/'output/groups/stage5_material_structure_v1'
REPORT=ROOT/'reports/errors/stage5_material_structure_20261106'
SEEDS=[11,29,47]


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')


def blocks(x,kind):
    if kind=='global':return x[:,:384]
    if kind=='local':return normalize(np.concatenate([x[:,384:768],x[:,768:]],axis=1)).astype('float32')
    assert kind=='hybrid'
    return np.concatenate([x[:,:384]*np.sqrt(.5),x[:,384:768]*.5,x[:,768:]*.5],axis=1).astype('float32')


def weighted_ward(centers,counts,target=8):
    centers=centers.astype('float64').copy();counts=counts.astype('float64').copy()
    members=[[i] for i in range(len(centers))]
    while len(members)>target:
        cost=cdist(centers,centers,'sqeuclidean')*counts[:,None]*counts[None,:]/(counts[:,None]+counts[None,:])
        cost[np.tril_indices(len(members))]=np.inf
        a,b=np.unravel_index(np.argmin(cost),cost.shape)
        mass=counts[a]+counts[b]
        centers[a]=(centers[a]*counts[a]+centers[b]*counts[b])/mass
        counts[a]=mass;members[a]+=members[b]
        centers=np.delete(centers,b,axis=0);counts=np.delete(counts,b);members.pop(b)
    mapping=np.empty(sum(map(len,members)),dtype=int)
    for group,ids in enumerate(members):mapping[ids]=group
    return mapping


def predict(model,x):
    y=model['kmeans'].predict(blocks(x,model['features']))
    return y if model['merge'] is None else model['merge'][y]


def gallery(p,labels,features,model,path):
    path.parent.mkdir(parents=True,exist_ok=True)
    distances=model['kmeans'].transform(blocks(features,model['features'])).min(1)
    parts=['<!doctype html><html lang="id"><meta charset="utf-8"><title>Prototipe visual</title>',
        '<style>body{font:16px sans-serif;max-width:1500px;margin:30px auto}img{width:150px;height:150px;object-fit:contain}figure{display:inline-block;width:170px;margin:8px;vertical-align:top}figcaption{font-size:12px}</style>',
        '<h1>32 prototipe visual untuk tinjauan tim</h1><p>Belum diberi kategori material. Contoh terdekat dan jauh ditampilkan; bukan label per gambar atau pemetaan berdasarkan arsip.</p>']
    for group in sorted(set(labels)):
        ids=np.flatnonzero((labels==group)&p.split.eq('train').to_numpy())
        order=ids[np.argsort(distances[ids],kind='stable')]
        shown=list(dict.fromkeys([*order[:8],*order[-4:]]))
        parts.append(f'<h2>Prototipe {group+1}: {len(ids)} train</h2>')
        for idx in shown:
            r=p.iloc[idx];relative=os.path.relpath(ROOT/r.processed_path,path.parent).replace('\\','/')
            parts.append(f'<figure><a href="{html.escape(relative)}"><img loading="lazy" src="{html.escape(relative)}"></a><figcaption>{html.escape(r.key)} | jarak {distances[idx]:.3f}</figcaption></figure>')
    parts.append('</html>');path.write_text('\n'.join(parts),encoding='utf-8')


def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    plan=json.loads((REPORT/'plan_before_run.json').read_text())
    integrity=json.loads((REPORT/'feature_integrity.json').read_text());assert integrity['status']=='passed'
    for name,digest in integrity['feature_hashes'].items():assert sha(FEATURE/name)==digest
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    z=np.load(FEATURE/'original224.npz',allow_pickle=False);x=z['features']
    assert np.array_equal(z['keys'],p.key.to_numpy(str))
    positions=dict(zip(p.key,range(len(p))))
    split_path=ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv'
    split=pd.read_csv(split_path).set_index('key')
    fit_keys=split.index[split.role.eq('fit_train')];val_keys=split.index[split.role.eq('heldout_train')]
    fi=np.array([positions[k] for k in fit_keys]);vi=np.array([positions[k] for k in val_keys])
    assert len(fi)==1538 and len(vi)==381 and not set(split.loc[fit_keys,'candidate_group'])&set(split.loc[val_keys,'candidate_group'])
    assert not set(p.iloc[fi].pixel_sha256)&set(p.iloc[vi].pixel_sha256)
    flip=np.load(FEATURE/'hflip224.npz',allow_pickle=False);scale=np.load(FEATURE/'original336.npz',allow_pickle=False)
    assert np.array_equal(flip['keys'],scale['keys'])
    subset=[i for i,k in enumerate(flip['keys']) if k in set(val_keys)]
    paired_ids=np.array([positions[k] for k in flip['keys'][subset]])
    assert len(subset)==87
    dump(REPORT/'clustering_plan_before_fit.json',dict(source_sha256=sha(__file__),
        feature_integrity_sha256=sha(REPORT/'feature_integrity.json'),split_sha256=sha(split_path),
        fit_keys=fit_keys.tolist(),heldout_keys=val_keys.tolist(),seeds=SEEDS,
        kinds=['global','local','hybrid'],clusters=[8,32],kmeans_n_init=10,max_iter=300,
        hybrid_squared_block_weights=[.5,.25,.25],prototype_merging='weighted Ward, 32 to 8',
        selection_weights=dict(flip=.4,scale=.4,cross_rep_neighbors=.2),minimum_heldout_group=5,
        fine_diagnostic='hybrid_k32_s11',reference_archive_read=False))
    rows=[];predictions={};models={};stability=[]
    with threadpool_limits(limits=4):
        neighbor_sets=[]
        for kind in ['global','local']:
            b=blocks(x,kind);sim=b[vi]@b[fi].T
            neighbor_sets.append(np.argsort(-sim,axis=1,kind='stable')[:,:10])
        def evaluate(labels,flip_labels,scale_labels,k):
            counts=np.bincount(labels[vi],minlength=k)
            neighbor=float(np.mean([(labels[vi,None]==labels[fi][nn]).mean() for nn in neighbor_sets]))
            fa=float((labels[paired_ids]==flip_labels).mean());sa=float((labels[paired_ids]==scale_labels).mean())
            return dict(flip_agreement=fa,scale_agreement=sa,cross_rep_neighbor_agreement=neighbor,
                development_score=.4*fa+.4*sa+.2*neighbor,occupied=int((counts>0).sum()),
                min_heldout_group=int(counts.min()),fit_count=len(fi),heldout_count=len(vi),view_count=len(subset))
        for kind in ['global','local','hybrid']:
            transformed=blocks(x,kind)
            for k in [8,32]:
                for seed in SEEDS:
                    km=KMeans(n_clusters=k,n_init=10,max_iter=300,random_state=seed).fit(transformed[fi])
                    for merge in [False,True] if k==32 and kind in ['global','hybrid'] else [False]:
                        tag=f'{kind}_k{k}'+('_ward8' if merge else '')+f'_s{seed}'
                        model=dict(kmeans=km,features=kind,merge=weighted_ward(km.cluster_centers_,np.bincount(km.labels_,minlength=k)) if merge else None)
                        labels=predict(model,x);fp=predict(model,flip['features'][subset]);sp=predict(model,scale['features'][subset]);final_k=8 if merge else k
                        metrics=evaluate(labels,fp,sp,final_k)
                        rows.append(dict(candidate=tag,kind=kind,clusters=final_k,seed=seed,**metrics))
                        predictions[tag]=labels;models[tag]=model
                        print(f'{tag} flip={metrics["flip_agreement"]:.3f} scale={metrics["scale_agreement"]:.3f} occupied={metrics["occupied"]}',flush=True)
        # Existing teacher/student is a control evaluated with the SAME new proxies.
        for tag,path in [('current_student',ROOT/'output/groups/stage4_self_training_pilot_v1/selected_fit_head.pt'),
                         ('stage2_full_train_control',ROOT/'output/groups/stage2_neighbor_v1/head.pt')]:
            c=torch.load(path,map_location='cpu',weights_only=True);w=c['state_dict']['weight'].numpy();b=c['state_dict']['bias'].numpy()
            labels=(x[:,:384]@w.T+b).argmax(1)
            fp=(flip['features'][subset,:384]@w.T+b).argmax(1);sp=(scale['features'][subset,:384]@w.T+b).argmax(1)
            metrics=evaluate(labels,fp,sp,8)
            rows.append(dict(candidate=tag,kind='head_control',clusters=8,seed=-1,selection_eligible=tag=='current_student',**metrics))
            predictions[tag]=labels
        frame=pd.DataFrame(rows);frame['selection_eligible']=frame.candidate.ne('stage2_full_train_control')&frame.clusters.eq(8)&frame.occupied.eq(8)&frame.min_heldout_group.ge(5)
        eligible=frame[frame.selection_eligible].sort_values(['development_score','candidate'],ascending=[False,True])
        assert len(eligible)>0
        selected=eligible.iloc[0].candidate
        for kind in ['global','local','hybrid']:
            for k in [8,32]:
                for a,b in itertools.combinations(SEEDS,2):
                    aa=f'{kind}_k{k}_s{a}';bb=f'{kind}_k{k}_s{b}'
                    stability.append(dict(kind=kind,clusters=k,seed_a=a,seed_b=b,heldout_ari=float(adjusted_rand_score(predictions[aa][vi],predictions[bb][vi]))))
    frame.to_csv(REPORT/'development_metrics.csv',index=False);pd.DataFrame(stability).to_csv(REPORT/'seed_stability.csv',index=False)
    assignments=p[['key','split','image_id','pixel_sha256']].copy()
    for name,labels in predictions.items():assignments[name]=labels
    assignments.to_csv(OUT/'all_frozen_assignments.csv',index=False)
    assignments[['key','split','image_id',selected]].to_csv(OUT/'selected_assignments.csv',index=False)
    for name,model in models.items():joblib.dump(model,OUT/(name+'.joblib'))
    # Fine grouping is a PREDECLARED diagnostic, never selected for reference purity.
    gallery(p,predictions['hybrid_k32_s11'],x,models['hybrid_k32_s11'],REPORT/'galleries/prototypes32.html')
    result=dict(status='passed',candidate_fits=18,assignment_conditions=len(predictions),selected_by_proxy=selected,
        selected_metrics=eligible.iloc[0].to_dict(),current_student_metrics=frame[frame.candidate.eq('current_student')].iloc[0].to_dict(),
        fine_diagnostic='hybrid_k32_s11',fit_images=len(fi),heldout_images=len(vi),group_overlap=0,
        test_used_for_fitting_or_selection=False,reference_archive_read=False,semantic_mapping_created=False,
        baseline_replaced=False,semantic_accuracy=None,seconds=time.time()-started,
        note='Development proxies can favor nuisance features. 32 groups are prototypes, not 32 classes or a claim of higher eight-class accuracy.')
    dump(REPORT/'summary.json',result)
    protected=['configs/data_contract.json','output/groups/stage2_neighbor_v1/assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',selected=selected,
        assignments_sha256=sha(OUT/'all_frozen_assignments.csv'),summary_sha256=sha(REPORT/'summary.json'),
        source_sha256=sha(__file__),split_sha256=sha(split_path),
        protected_inputs={v:sha(ROOT/v) for v in protected}))
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

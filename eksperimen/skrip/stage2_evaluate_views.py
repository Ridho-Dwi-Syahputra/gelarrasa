"""Unlabeled view/geometry comparison; no archived labels or categories read."""
from pathlib import Path
import json
import hashlib
import html
import os
import itertools
import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.metrics import silhouette_score, adjusted_rand_score
from sklearn.preprocessing import normalize
from threadpoolctl import threadpool_limits
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
REP=ROOT/'reports/errors/stage2_20261105'
FEAT=ROOT/'output/embeddings/stage2_repr_v1'
BRANCHES=['nopca','pca64','pca128','pca256']


def load(path, keys):
    z=np.load(path,allow_pickle=False)
    assert len(set(z['keys']))==len(z['keys'])
    lookup={k:i for i,k in enumerate(z['keys'])}
    assert set(keys)==set(z['keys'])
    return normalize(z['features'][[lookup[k] for k in keys]]).astype('float32')


def predict(model,x):
    x=normalize(x).astype('float32')
    if model['reducer'] is not None:
        x=normalize(model['reducer'].transform(x)).astype('float32')
    return model['kmeans'].predict(x)


def topk(x,pool,h,hp,k=10):
    s=x@pool.T
    s[h[:,None]==hp[None,:]]=-np.inf
    return np.argsort(-s,axis=1,kind='stable')[:,:k]


def retention(a,b):
    return (a[:,:,None]==b[:,None,:]).any(2).mean(1)


def main():
    df=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    keys=df.key.to_numpy(str)
    fit=df[df.split.eq('train')].drop_duplicates('pixel_sha256').index.to_numpy()
    probe=pd.read_csv(FEAT/'view_probe_ids.csv',dtype={'image_id':str})
    lookup={k:i for i,k in enumerate(keys)}
    ids=np.array([lookup[k] for k in probe.key])
    assert len(ids)==256 and df.iloc[ids].split.eq('train').all()
    raw={
        'resnet50':load(ROOT/'output/embeddings/stage1_v1/resnet50_features.npz',keys),
        'dinov2':load(FEAT/'dinov2_features.npz',keys)}
    views={name:{v:load(FEAT/f'{name}_{v}_probe.npz',probe.key.to_numpy(str))
                 for v in ['hflip224','original336']} for name in raw}
    methods={}
    for name in raw:
        for branch in BRANCHES:
            m=joblib.load(ROOT/f'output/groups/stage2_repr_v1/{name}/{branch}_seed11_model.joblib')
            methods[name+'_'+branch]=(name,lambda x,m=m:predict(m,x))
    original=dict(reducer=joblib.load(ROOT/'output/embeddings/stage1_v1/cnn_pca.joblib'),
                  kmeans=joblib.load(ROOT/'output/embeddings/stage1_v1/primary_kmeans.joblib'))
    methods['stage1_actual_resnet_pca64']=('resnet50',lambda x:predict(original,x))
    head_file=ROOT/'output/groups/stage2_neighbor_v1/head.pt'
    if head_file.exists():
        checkpoint=torch.load(head_file,map_location='cpu',weights_only=True)
        w=checkpoint['state_dict']['weight'].numpy(); b=checkpoint['state_dict']['bias'].numpy()
        methods['dinov2_neighbor_refinement']=('dinov2',lambda x:(x@w.T+b).argmax(1))
    summaries=[];per_row=[];partitions={};reference_scores=[]
    sample=np.sort(np.random.default_rng(11).choice(fit,1000,replace=False))
    hashes=df.pixel_sha256.to_numpy()
    with threadpool_limits(limits=4):
        for method,(name,fn) in methods.items():
            labels=fn(raw[name]);partitions[method]=labels
            if method=='stage1_actual_resnet_pca64':
                saved=pd.read_csv(ROOT/'output/groups/stage1_v1/cluster_assignments.csv')
                assert np.array_equal(labels,saved.cluster_index.to_numpy())
            for ref,ref_x in raw.items():
                reference_scores.append(dict(method=method,reference_space=ref,
                                             silhouette=float(silhouette_score(ref_x[sample],labels[sample]))))
            nn0=topk(raw[name][ids],raw[name][fit],hashes[ids],hashes[fit])
            for view,changed in views[name].items():
                lab=fn(changed)
                nnv=topk(changed,raw[name][fit],hashes[ids],hashes[fit])
                kept=retention(nn0,nnv)
                cosine=(changed*raw[name][ids]).sum(1)
                summary=dict(method=method,view=view,n=len(ids),
                             unchanged_count=int((lab==labels[ids]).sum()),
                             unchanged_fraction=float((lab==labels[ids]).mean()),
                             ari=float(adjusted_rand_score(labels[ids],lab)),
                             knn10_retention=float(kept.mean()),embedding_cosine=float(cosine.mean()))
                summaries.append(summary)
                for j,i in enumerate(ids):
                    per_row.append(dict(key=keys[i],method=method,view=view,
                                        original_cluster=int(labels[i]),view_cluster=int(lab[j]),
                                        unchanged=bool(lab[j]==labels[i]),knn10_retention=float(kept[j]),
                                        embedding_cosine=float(cosine[j])))
    pd.DataFrame(summaries).to_csv(REP/'view_robustness.csv',index=False)
    pd.DataFrame(per_row).to_csv(REP/'view_robustness_per_image.csv',index=False)
    pd.DataFrame(reference_scores).to_csv(REP/'shared_space_silhouette.csv',index=False)
    pairwise=[]
    for a,b in itertools.combinations(partitions,2):
        pairwise.append(dict(method_a=a,method_b=b,ari_train=float(adjusted_rand_score(partitions[a][fit],partitions[b][fit]))))
    pd.DataFrame(pairwise).to_csv(REP/'method_pairwise_ari.csv',index=False)
    # All anonymous partitions remain exploratory. Show every image in a train/test gallery
    # for the preregistered PCA128 DINO branch, not chosen using archived annotations.
    assignments=pd.read_csv(ROOT/'output/groups/stage2_repr_v1/dinov2/pca128_assignments.csv')
    assert np.array_equal(assignments.key,df.key)
    gal=REP/'galleries';gal.mkdir(exist_ok=True)
    def pic(row):
        source=(ROOT/row['preview_path']).resolve()
        rel=Path(os.path.relpath(source,gal)).as_posix()
        return f'<figure><img loading="lazy" src="{html.escape(rel)}"><figcaption>{html.escape(row["key"])}</figcaption></figure>'
    style='<style>body{font:16px sans-serif;margin:24px}section{display:flex;flex-wrap:wrap}figure{width:140px;margin:5px}img{width:140px;height:140px;object-fit:contain}h2{margin-top:35px}</style>'
    cards=['<!doctype html><meta charset="utf-8">',style,'<h1>DINOv2 PCA128: anonymous clusters</h1>',
           '<p>All 2,400 images; train-only fitted partition. No semantic category mapping. Not a validated submission dataset.</p>']
    for split in ['train','test']:
        for k in range(8):
            selected=df[df.split.eq(split)&assignments.cluster_index.eq(k)]
            cards.append(f'<h2>{split}: cluster_{k:02d}, n={len(selected)}</h2><section>')
            cards.extend(pic(r) for r in selected.to_dict('records'));cards.append('</section>')
    (gal/'dinov2_pca128_clusters.html').write_text('\n'.join(cards),encoding='utf-8')
    ti=np.flatnonzero(df.split.eq('test'))
    nb=topk(raw['dinov2'][ti],raw['dinov2'][fit],hashes[ti],hashes[fit],k=5)
    rows=[];cards=['<!doctype html><meta charset="utf-8">',style,'<h1>DINOv2: all 480 test images with five train neighbors</h1>',
                  '<p>Exact pixel duplicates excluded. Similarity is not proof of category correctness. No class names.</p>']
    for j,i in enumerate(ti):
        cards.append(f'<h2>{keys[i]}</h2><section>'+pic(df.iloc[i].to_dict()))
        for rank,ni in enumerate(fit[nb[j]],1):
            rows.append(dict(test_key=keys[i],train_key=keys[ni],rank=rank,
                             cosine=float(raw['dinov2'][i]@raw['dinov2'][ni])))
            cards.append(pic(df.iloc[ni].to_dict()))
        cards.append('</section>')
    pd.DataFrame(rows).to_csv(REP/'dinov2_test_train_top5.csv',index=False)
    (gal/'dinov2_test_neighbors.html').write_text('\n'.join(cards),encoding='utf-8')
    plot=pd.DataFrame(summaries).pivot(index='method',columns='view',values='unchanged_fraction')
    fig,ax=plt.subplots(figsize=(11,6))
    plot.plot.barh(ax=ax,xlim=(0,1),color=['#277da8','#f29c46'])
    ax.set_xlabel('Fraction with unchanged cluster (256 fixed train probes)')
    ax.set_ylabel('');ax.set_title('View consistency is not semantic Accuracy')
    fig.tight_layout();fig.savefig(REP/'view_consistency.png',dpi=160);plt.close(fig)
    (REP/'view_evaluation_notes.json').write_text(json.dumps(dict(
        test_used_for_fitting=False,archive_labels_used=False,semantic_accuracy=None,
        n_probe=256,probe_selection='random unique train seed20261105, before seeing view results',
        caution='Train probe stability, not heldout Accuracy; resolution change uses 224-trained cluster model. A changed assignment may be better or worse.',
        original_stage1_reproduced_exactly=True,gallery_cluster_branch='dinov2_pca128',
        gallery_all_images=2400,gallery_test_neighbors=480,
        selection_status='Exploratory review branch only; no active dataset promotion',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2),encoding='utf-8')
    print(pd.DataFrame(summaries)[['method','view','unchanged_count','unchanged_fraction']].to_string(index=False))
    print(pd.DataFrame(reference_scores).pivot(index='method',columns='reference_space',values='silhouette').to_string())


if __name__=='__main__':
    main()

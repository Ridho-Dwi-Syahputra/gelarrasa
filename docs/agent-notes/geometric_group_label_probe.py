"""Train-only high-coverage geometric source consistency and group-heldout vote."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
NOTES=ROOT/'docs/agent-notes'
CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']


def geometry(path_a,path_b):
    images=[cv2.imdecode(np.fromfile(path,dtype=np.uint8),cv2.IMREAD_GRAYSCALE) for path in [path_a,path_b]]
    sift=cv2.SIFT_create(nfeatures=2000)
    pairs=[sift.detectAndCompute(image,None) for image in images]
    if any(desc is None for _,desc in pairs):return {'qualified':False,'matches':0}
    match=cv2.BFMatcher().knnMatch(pairs[0][1],pairs[1][1],k=2)
    good=[a for a,b in match if a.distance<.75*b.distance]
    if len(good)<8:return {'qualified':False,'matches':len(good)}
    aa=np.float32([pairs[0][0][m.queryIdx].pt for m in good]);bb=np.float32([pairs[1][0][m.trainIdx].pt for m in good])
    cv2.setRNGSeed(20261109)
    transform,inliers=cv2.findHomography(aa,bb,cv2.RANSAC,3.)
    if transform is None:return {'qualified':False,'matches':len(good)}
    mask=inliers.ravel().astype(bool);fraction=float(mask.mean())
    hull=[float(cv2.contourArea(cv2.convexHull(points[mask]))/image.size) if mask.sum()>=3 else 0.
          for points,image in zip([aa,bb],images)]
    qualified=len(good)>=40 and fraction>=.9 and min(hull)>=.3 and np.isfinite(transform).all()
    return {'qualified':bool(qualified),'matches':len(good),'inliers':int(mask.sum()),'inlier_fraction':fraction,
            'hull_fraction_a':hull[0],'hull_fraction_b':hull[1]}


def main():
    paths=['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']
    protected={p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in paths}
    plan={'method':'high_coverage_SIFT_RANSAC_and_three_group_heldout_classifiers',
          'geometry':{'ratio':.75,'min_matches':40,'min_inlier_fraction':.9,'min_hull_fraction_both':.3,'ransac_pixels':3.},
          'vote':'>=5of6pooledpredictions_for_pair;new_class_must_be_present_in_pair;human_anchors_protected',
          'models':['logistic_C3','svm_rbf_C4','cosine_knn7'],'test_rows_in_fit':0,
          'archive_or_review_read_by_fit':False,'new_manual_labels':0,'active_outputs_modified':False}
    (NOTES/'json/geometric_group_label_plan.json').write_text(json.dumps(plan,indent=2),encoding='utf-8')
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    fit=m[m.split.eq('train')&m.training_eligible].reset_index(drop=True)
    full=fit[~fit.is_crop].copy();g=full.groupby('parent_group').assigned_class.nunique()
    z=np.load(ROOT/'data/processed/dinov2_small_embeddings.npz',allow_pickle=False);lookup=dict(zip(z['keys'],range(len(z['keys']))))
    x=z['features'][[lookup[k] for k in fit.sample_key]];y=fit.assigned_class.to_numpy(str)
    evidence=[];candidate=full[['parent_key','parent_id','assigned_class']].copy();candidate['proposed_class']=candidate.assigned_class
    with threadpool_limits(limits=4):
        for group in g[g>1].index:
            rows=full[full.parent_group.eq(group)]
            if len(rows)!=2 or rows.human_reviewed.any():continue
            a,b=rows.iloc[0],rows.iloc[1]
            metric=geometry(ROOT/a.source_path,ROOT/b.source_path)
            record={'group':group,'key_a':a.parent_key,'key_b':b.parent_key,**metric,'accepted':False}
            if metric['qualified']:
                train=~fit.parent_group.eq(group).to_numpy();query=~train
                assert set(fit.loc[train,'parent_group']).isdisjoint(set(fit.loc[query,'parent_group']))
                models=[LogisticRegression(C=3,max_iter=1200,random_state=20261109),
                        SVC(C=4,kernel='rbf',gamma='scale'),
                        KNeighborsClassifier(n_neighbors=7,weights='distance',metric='cosine')]
                votes=[]
                for model in models:
                    model.fit(x[train],y[train]);votes.extend(model.predict(x[query]).tolist())
                classes,counts=np.unique(votes,return_counts=True);best=int(counts.argmax());winner=str(classes[best])
                accepted=int(counts[best])>=5 and winner in set(rows.assigned_class)
                record.update(votes=votes,winner=winner,winning_votes=int(counts[best]),accepted=bool(accepted))
                if accepted:candidate.loc[candidate.parent_key.isin(rows.parent_key),'proposed_class']=winner
            evidence.append(record)
    candidate['changed']=candidate.proposed_class.ne(candidate.assigned_class)
    output=NOTES/'table-csv/geometric_group_label_candidate.csv';candidate.to_csv(output,index=False)
    report={'status':'frozen_before_posthoc_audit','evidence':evidence,'proposed_changes':int(candidate.changed.sum()),
            'candidate_sha256':hashlib.sha256(output.read_bytes()).hexdigest(),
            'archive_or_review_read_by_fit':False,'test_rows_in_fit':0,'active_outputs_modified':False}
    (NOTES/'json/geometric_group_label_frozen.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    assert all(hashlib.sha256((ROOT/p).read_bytes()).hexdigest()==h for p,h in protected.items())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

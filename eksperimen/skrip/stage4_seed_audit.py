"""Prepare model-based seed/reserve masks; no semantic/reference labels are read.

This is a selection audit, not a new trained model or semantic accuracy test.
Rebuild the teacher within fit folds before evaluating a future student.
"""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/stage4_seed_audit_v1'
REPORT=ROOT/'reports/errors/stage4_seed_audit_20261106'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    paths=[ROOT/'output/manifests/stage1_v1/image_profile.csv',
           ROOT/'output/groups/stage1_v1/cluster_assignments.csv',
           ROOT/'output/groups/stage2_neighbor_v1/assignments.csv',
           ROOT/'output/groups/stage2_neighbor_v1/head.pt',
           ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz']
    before={str(p.relative_to(ROOT)):sha(p) for p in paths}
    p=pd.read_csv(paths[0],dtype={'image_id':str})
    old=pd.read_csv(paths[1],dtype={'image_id':str})
    a=pd.read_csv(paths[2],dtype={'image_id':str})
    features=np.load(paths[4],allow_pickle=False)
    assert np.array_equal(features['keys'],p.key.to_numpy(str))
    mask=p.split.eq('train').to_numpy();x=features['features'][mask]
    d=p[mask][['key','image_id','pixel_sha256']].copy().reset_index(drop=True)
    d=d.merge(a[['key','cluster_index']],on='key',validate='one_to_one')
    d=d.merge(old[['key','cluster_index','relative_margin']],on='key',suffixes=('','_resnet'),validate='one_to_one')
    checkpoint=torch.load(paths[3],map_location='cpu',weights_only=True)
    assert checkpoint['config']['features_sha256']==before[str(paths[4].relative_to(ROOT))]
    head=torch.nn.Linear(384,8);head.load_state_dict(checkpoint['state_dict']);head.eval()
    with torch.inference_mode():probability=head(torch.from_numpy(x)).softmax(-1).numpy()
    assert np.array_equal(probability.argmax(1),d.cluster_index)
    d['head_probability_uncalibrated']=probability.max(1)
    sorted_probability=np.sort(probability,axis=1)
    d['head_margin']=sorted_probability[:,-1]-sorted_probability[:,-2]
    unique=~d.pixel_sha256.duplicated()
    unique_indices=np.where(unique)[0]
    with threadpool_limits(limits=4):similarity=x@x[unique_indices].T
    similarity[d.pixel_sha256.to_numpy()[:,None]==d.iloc[unique_indices].pixel_sha256.to_numpy()[None,:]]=-np.inf
    neighbors=unique_indices[np.argsort(-similarity,axis=1,kind='stable')[:,:10]]
    d['neighbor_agreement']=np.mean(d.cluster_index.to_numpy()[neighbors]==d.cluster_index.to_numpy()[:,None],axis=1)
    # Align two anonymous partitions on unique TRAIN only, without class names.
    overlap=np.zeros((8,8),dtype='int64')
    for r in d[unique].itertuples():overlap[r.cluster_index,r.cluster_index_resnet]+=1
    left,right=linear_sum_assignment(-overlap)
    old_to_current=dict(zip(right,left))
    d['two_model_agreement']=d.cluster_index.eq(d.cluster_index_resnet.map(old_to_current))
    d['pixel_unique']=unique
    d['seed_candidate']=(unique & d.head_probability_uncalibrated.ge(.95)&d.head_margin.ge(.5)&d.neighbor_agreement.ge(.9)&d.two_model_agreement)
    d['selection_reason']=['seed_candidate' if seed else 'duplicate_source_preserved' if not u else 'ambiguous_reserve' for seed,u in zip(d.seed_candidate,unique)]
    d['semantic_label_assigned']=False
    d['promoted_to_training']=False
    groups=pd.read_csv(ROOT/'output/groups/stage1_v1/groups.csv')
    folds=pd.read_csv(ROOT/'output/folds/stage1_v1/DRAFT_group_folds.csv')
    fold_map=folds.groupby('candidate_group').fold.first()
    d=d.merge(groups[['key','candidate_group']],on='key',validate='one_to_one')
    d['parent_fold']=d.candidate_group.map(fold_map)
    assert d.groupby('candidate_group').parent_fold.nunique().max()==1
    assert len(d)==1920 and d.key.is_unique and int(unique.sum())==1919
    assert d.neighbor_agreement.between(0,1).all()
    OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    d.to_csv(OUT/'seed_and_reserve_manifest.csv',index=False)
    d[d.seed_candidate].to_csv(OUT/'seed_candidates.csv',index=False)
    d[~d.seed_candidate].to_csv(OUT/'reserve_preserved.csv',index=False)
    counts=d[unique].groupby('cluster_index').agg(train_unique=('key','size'),seed_candidates=('seed_candidate','sum'))
    counts.to_csv(REPORT/'anonymous_group_seed_counts.csv')
    result=dict(status='passed',scope='Read-only preparation for self-training; no student trained',
        train_rows=1920,unique_train=1919,seed_candidates=int(d.seed_candidate.sum()),
        ambiguous_reserve=int((unique&~d.seed_candidate).sum()),duplicate_rows_preserved=int((~unique).sum()),
        filters=dict(probability=.95,margin=.5,neighbor_consensus=.9,model_agreement=True),
        reference_archive_read=False,display_names_read=False,reference_agreement_optimized=False,
        test_used=False,semantic_accuracy=None,labels_assigned=False,raw_files_deleted=0,
        student_trained=False,training_promoted=False,
        limitation='Current teacher was previously refit on all unique train. This audit does not establish held-out performance. Future teacher/filter/student must be built inside fit groups.',
        source_hashes=before,script_sha256=sha(Path(__file__)))
    for relative,digest in before.items():assert sha(ROOT/relative)==digest
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

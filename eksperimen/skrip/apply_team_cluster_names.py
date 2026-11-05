"""Apply a completed team-level cluster mapping without changing memberships.

This is deliberately separate from exploration. It does not guess class names,
read historical annotations, edit individual assignments, or generate a submission.
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

import pandas as pd


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def apply(root, mapping_path, version):
    if not version or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in version):
        raise ValueError('Use a new simple version name.')
    source=root/'data/processed/stage1_v1'
    assignment_path=root/'output/groups/stage1_v1/cluster_assignments.csv'
    provenance=json.loads((assignment_path.parent/'cluster_provenance.json').read_text())
    if file_hash(assignment_path)!=provenance['cluster_assignment_sha256']:
        raise ValueError('Cluster assignments changed after run; inspect provenance first.')
    classes=json.loads((root/'configs/data_contract.json').read_text())['class_order']
    mapping=pd.read_csv(mapping_path,dtype=str).fillna('')
    required={'cluster_id','waste_category','mapped_by_team_member','evidence'}
    if not required.issubset(mapping):
        raise ValueError('Mapping is missing required columns.')
    if mapping.cluster_id.duplicated().any() or not mapping.waste_category.isin(classes).all():
        raise ValueError('Every cluster needs one allowed category, without duplicate cluster rows.')
    if mapping.mapped_by_team_member.str.strip().eq('').any() or mapping.evidence.str.strip().eq('').any():
        raise ValueError('Record the team member and basis for each cluster-level name.')
    assignments=pd.read_csv(assignment_path,dtype={'image_id':str})
    if set(mapping.cluster_id)!=set(assignments.cluster_id):
        raise ValueError('Mapping must cover exactly the frozen cluster IDs.')
    destination=root/'data/processed'/version
    if destination.exists():
        raise FileExistsError('Use a new output version; existing named data are never overwritten.')
    joined=assignments.merge(mapping[list(required)],on='cluster_id',validate='many_to_one')
    # Only train gets category folders; test remains exploratory cluster assignment.
    train=joined[joined.split.eq('train')].copy()
    destination.mkdir(parents=True)
    for category in classes:
        (destination/'train'/category).mkdir(parents=True)
    for r in train.itertuples():
        src=root/r.processed_path
        dest=destination/'train'/r.waste_category/(r.image_id+'.png')
        try:
            os.link(src,dest)
        except OSError:
            shutil.copy2(src,dest)
    train.to_csv(destination/'train_labels_from_team_cluster_mapping.csv',index=False)
    mapping.to_csv(destination/'team_cluster_mapping.csv',index=False)
    manifest=dict(source_assignment_sha256=file_hash(assignment_path),
                  mapping_sha256=file_hash(mapping_path), categories=classes,
                  train_images=len(train), class_counts=train.waste_category.value_counts().to_dict(),
                  mapping_level='whole cluster', per_image_membership_edits=False,
                  category_ground_truth_verified=False, classifier_finetuned=False,
                  test_category_prediction_exported=False)
    (destination/'provenance.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print(json.dumps(manifest,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parents[2])
    parser.add_argument('--mapping',type=Path,required=True)
    parser.add_argument('--version',required=True)
    args=parser.parse_args()
    apply(args.root.resolve(),args.mapping.resolve(),args.version)

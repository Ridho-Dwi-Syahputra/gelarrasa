"""Verify frozen student results, THEN compare to the private reference archive."""
from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd
import torch
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage4_self_training_pilot_v1'
REPORT=ROOT/'reports/errors/stage4_self_training_20261106'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    torch.set_num_threads(4)
    run=json.loads((REPORT/'summary.json').read_text());assert run['status']=='passed'
    assert not run['reference_archive_read'] and not run['display_mapping_read']
    protected={p.name:sha(p) for p in [OUT/'selected_fit_head.pt',OUT/'teacher_fit_head.pt',OUT/'candidate_assignments.csv']}
    assert protected['selected_fit_head.pt']==run['student_fit_checkpoint_sha256']
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    z=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    assert np.array_equal(z['keys'],p.key.to_numpy(str));features=z['features']
    teacher=pd.read_csv(OUT/'teacher_fit_assignments.csv',dtype={'image_id':str})
    student=pd.read_csv(OUT/'candidate_assignments.csv',dtype={'image_id':str})
    checkpoint=torch.load(OUT/'selected_fit_head.pt',map_location='cpu',weights_only=True)
    fit=set(checkpoint['fit_keys']);held=set(checkpoint['heldout_keys'])
    assert not fit&held and len(fit)==1538 and len(held)==381
    assert p[p.key.isin(fit|held)].split.eq('train').all()
    seeds=pd.read_csv(OUT/'fit_seed_mask.csv');assert set(seeds.key)==fit
    assert int(seeds.seed_candidate.sum())==run['seed_fit_images']
    assert not set(seeds.key)&held
    head=torch.nn.Linear(384,8);head.load_state_dict(checkpoint['state_dict']);head.eval()
    with torch.inference_mode():probability=head(torch.from_numpy(features)).softmax(-1).numpy()
    cached=np.load(OUT/'candidate_probabilities.npz',allow_pickle=False)
    assert np.array_equal(cached['keys'],p.key.to_numpy(str))
    assert np.isfinite(probability).all() and np.allclose(probability.sum(1),1,atol=1e-6)
    assert np.allclose(probability,cached['probabilities'],atol=1e-6)
    assert np.array_equal(probability.argmax(1),student.anonymous_group)
    metrics=pd.read_csv(REPORT/'development_metrics.csv')
    assert np.isfinite(metrics[['neighbor_agreement','flip_agreement','selection_score']]).all().all()
    assert metrics.candidate.nunique()==1+run['new_student_models_trained']
    index=dict(zip(p.key,range(len(p))))
    fi=np.array([index[k] for k in checkpoint['fit_keys']]);vi=np.array([index[k] for k in checkpoint['heldout_keys']])
    with threadpool_limits(limits=4):similarity=features[vi]@features[fi].T
    neighbors=np.argsort(-similarity,axis=1,kind='stable')[:,:10]
    predicted=probability.argmax(1)
    reproduced=float((predicted[vi,None]==predicted[fi][neighbors]).mean())
    assert np.isclose(reproduced,run['selected_development']['neighbor_agreement'],atol=1e-6)
    # Re-examine the model's own reserve before loading any archive reference.
    with threadpool_limits(limits=4):fit_similarity=features[fi]@features[fi].T
    np.fill_diagonal(fit_similarity,-np.inf)
    near=np.argsort(-fit_similarity,axis=1,kind='stable')[:,:10]
    fit_prediction=predicted[fi]
    agreement=(fit_prediction[:,None]==fit_prediction[near]).mean(1)
    fit_probability=probability[fi];ordered=np.sort(fit_probability,axis=1)
    paired=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    with torch.inference_mode():paired_prediction=head(torch.from_numpy(paired['features'])).argmax(-1).numpy()
    flip_map=dict(zip(paired['keys'],paired_prediction))
    reassess=seeds.set_index('key').loc[checkpoint['fit_keys']].reset_index()
    reassess['student_probability_uncalibrated']=fit_probability.max(1)
    reassess['student_margin']=ordered[:,-1]-ordered[:,-2]
    reassess['neighbor_agreement']=agreement
    reassess['agrees_frozen_teacher']=fit_prediction==teacher.iloc[fi].anonymous_group.to_numpy()
    reassess['flip_tested']=[k in flip_map for k in reassess.key]
    reassess['flip_stable']=[k in flip_map and flip_map[k]==label for k,label in zip(reassess.key,fit_prediction)]
    reassess['next_round_candidate']=(~reassess.seed_candidate & reassess.student_probability_uncalibrated.ge(.95)&
        reassess.student_margin.ge(.5)&reassess.neighbor_agreement.ge(.9)&reassess.agrees_frozen_teacher&reassess.flip_stable)
    second=pd.read_csv(OUT/'fit_seed_mask_round2.csv').set_index('key')
    reassess['used_in_round2_pilot']=reassess.key.map(second.reentered_from_reserve)
    reassess['promoted_to_active_training_labels']=False
    reassess.to_csv(OUT/'reserve_reassessment_model_only.csv',index=False)
    rechecked=int((~reassess.seed_candidate).sum());returned=int(reassess.next_round_candidate.sum())
    # Freeze a display-only alignment using FIT predictions, before reading references.
    baseline=pd.read_csv(ROOT/'output/groups/stage2_neighbor_v1/assignments.csv')
    overlap=np.zeros((8,8),dtype=int)
    for a,b in zip(teacher.iloc[fi].anonymous_group,baseline.iloc[fi].cluster_index):overlap[a,b]+=1
    left,right=linear_sum_assignment(-overlap);alignment=dict(zip(left,right))
    display=json.loads((ROOT/'configs/display_label_mapping.json').read_text())
    names={int(k):v['display_name'] for k,v in display['groups'].items()}
    mixed={int(k):v['mixed'] for k,v in display['groups'].items()}
    d=student[student.split.eq('train')][['key','image_id']].copy()
    teacher_index=teacher.set_index('key').anonymous_group
    student_index=student.set_index('key').anonymous_group
    d['teacher_display']=d.key.map(teacher_index).map(alignment).map(names)
    d['student_display']=d.key.map(student_index).map(alignment).map(names)
    d['mixed_teacher']=d.key.map(teacher_index).map(alignment).map(mixed)
    d['mixed_student']=d.key.map(student_index).map(alignment).map(mixed)
    d['role']=np.where(d.key.isin(fit),'fit_train',np.where(d.key.isin(held),'heldout_train','duplicate_preserved'))
    # Post-hoc reference read starts only here, after all learning/selection is done.
    rows=[]
    for folder in (ROOT/'data/Dataset DSC GelarRasa 2026/train/label').iterdir():
        if folder.is_dir():
            for image in folder.iterdir():
                if image.is_file() and image.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                    rows.append(dict(image_id=image.stem,archive_reference=folder.name))
    reference=pd.DataFrame(rows);assert len(reference)==1920 and reference.image_id.is_unique
    d=d.merge(reference,on='image_id',validate='one_to_one')
    d['shared_single_material_cohort']=~d.mixed_teacher&~d.mixed_student
    d['teacher_agrees_reference']=d.teacher_display.eq(d.archive_reference)
    d['student_agrees_reference']=d.student_display.eq(d.archive_reference)
    d['audit_action']='review_only_not_used_for_model_selection_or_training'
    cohort=d[d.shared_single_material_cohort]
    result=dict(status='passed',student_models_trained=run['new_student_models_trained'],checkpoint_predictions_verified=2400,
        selected_development_neighbor_agreement_reproduced=reproduced,fit_holdout_overlap=0,
        reference_read_after_model_selection=True,reference_used_for_training_filtering_or_selection=False,
        ambiguous_fit_rows_reexamined=rechecked,next_round_model_consistent_candidates=returned,
        reserve_candidates_used_in_round2_pilot=int(reassess.used_in_round2_pilot.sum()),active_label_promotions=0,
        display_alignment_fit_only=True,shared_single_material_cohort=len(cohort),
        teacher_agrees_reference=int(cohort.teacher_agrees_reference.sum()),
        student_agrees_reference=int(cohort.student_agrees_reference.sum()),
        toward_reference=int((~cohort.teacher_agrees_reference&cohort.student_agrees_reference).sum()),
        away_from_reference=int((cohort.teacher_agrees_reference&~cohort.student_agrees_reference).sum()),
        mixed_or_changed_to_mixed_not_assessed=int((~d.shared_single_material_cohort).sum()),
        semantic_accuracy=None,baseline_replaced=False,
        note='Reference agreement is private post-hoc diagnostics, not official Accuracy. Candidate was frozen before comparison and is not selected or retrained using this audit.',
        case_0046=d[d.image_id.eq('0046')][['teacher_display','student_display','archive_reference','role']].to_dict('records'))
    for name,digest in protected.items():assert sha(OUT/name)==digest
    target=ROOT/'data/processed/stage4_frozen_student_reference_audit_only';target.mkdir(parents=True,exist_ok=True)
    d.to_csv(target/'posthoc_reference_queue.csv',index=False)
    (REPORT/'frozen_student_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

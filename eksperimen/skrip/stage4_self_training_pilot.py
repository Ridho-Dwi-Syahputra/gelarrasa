"""Train-only teacher/student pilot on frozen image features; no reference labels.

Uses the existing random 256-image flip probe. This is a small head-training
experiment, not backbone fine-tuning or semantic Accuracy evaluation.
"""
from pathlib import Path
import copy
import hashlib
import json
import time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
import stage2_neighbor_refinement as base

ROOT=Path(__file__).resolve().parents[2]
REPORT=ROOT/'reports/errors/stage4_self_training_20261106'
OUT=ROOT/'output/groups/stage4_self_training_pilot_v1'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    started=time.time();torch.set_num_threads(4);REPORT.mkdir(parents=True,exist_ok=True);OUT.mkdir(parents=True,exist_ok=True)
    profile=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    features_path=ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz'
    z=np.load(features_path,allow_pickle=False);assert np.array_equal(z['keys'],profile.key.to_numpy(str))
    x=z['features'];unique=profile[profile.split.eq('train')].drop_duplicates('pixel_sha256').copy()
    positions=dict(zip(profile.key,range(len(profile))))
    split_path=ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv'
    split=pd.read_csv(split_path);d=unique.merge(split[['key','role','candidate_group']],on='key',validate='one_to_one')
    fit=d[d.role.eq('fit_train')].copy();held=d[d.role.eq('heldout_train')].copy()
    assert not set(fit.candidate_group)&set(held.candidate_group)
    assert not set(fit.pixel_sha256)&set(held.pixel_sha256)
    fi=np.array([positions[k] for k in fit.key]);vi=np.array([positions[k] for k in held.key])
    xt=torch.from_numpy(x[fi]);xv=torch.from_numpy(x[vi])
    hashes=fit.pixel_sha256.to_numpy();hv=held.pixel_sha256.to_numpy()
    nn_fit=base.graph(x[fi],x[fi],hashes,hashes);nn_val=base.graph(x[vi],x[fi],hv,hashes)
    flip_path=ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz'
    flip=np.load(flip_path,allow_pickle=False);flip_index=dict(zip(flip['keys'],range(len(flip['keys']))))
    paired_fit=np.array([i for i,k in enumerate(fit.key) if k in flip_index],dtype=int)
    paired_val=np.array([i for i,k in enumerate(held.key) if k in flip_index],dtype=int)
    xf=torch.from_numpy(flip['features'][[flip_index[fit.iloc[i].key] for i in paired_fit]])
    vf=torch.from_numpy(flip['features'][[flip_index[held.iloc[i].key] for i in paired_val]])
    assert len(paired_fit)>0 and len(paired_val)>0
    plan=dict(task='Anonymous pseudo-label teacher/student pilot on frozen DINO features',teacher_seeds=[11,47],
       teacher_epochs=60,teacher_entropy_weight=2.,student_epochs=80,student_lr=.003,max_self_label_rounds=2,
       configurations=[dict(name='seed_ce_control',view_weight=0.),dict(name='seed_ce_view05',view_weight=.5),dict(name='seed_ce_view2',view_weight=2.)],
       seed_rules=dict(probability=.95,margin=.5,neighbor_agreement=.9,two_teacher_agreement=True),
       selection='Development holdout: neighbor agreement + 0.25*flip agreement; all eight groups occupied with at least five heldout samples. No semantic label or reference agreement.',
       split_reused_from_previous_development=True,fit_images=len(fit),heldout_images=len(held),
       flip_fit=len(paired_fit),flip_heldout=len(paired_val),reference_labels_read=False,display_names_read=False,
       test_used_for_training_or_selection=False,backbone_finetuned=False,
       feature_sha256=sha(features_path),flip_feature_sha256=sha(flip_path),split_sha256=sha(split_path))
    (REPORT/'plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    history=[]
    def evaluate(head):
        head.eval()
        with torch.inference_mode():
            p=head(xt).softmax(-1);q=head(xv).softmax(-1)
            pred=q.argmax(-1);train_pred=p.argmax(-1)
            neighbor=float((pred[:,None]==train_pred[nn_val]).float().mean())
            flip_agreement=float((pred[paired_val]==head(vf).argmax(-1)).float().mean())
            counts=torch.bincount(pred,minlength=8)
            return dict(neighbor_agreement=neighbor,flip_agreement=flip_agreement,
                        selection_score=neighbor+.25*flip_agreement,min_heldout_group=int(counts.min()),
                        occupied_heldout_groups=int((counts>0).sum()))
    def teacher(seed):
        torch.manual_seed(seed)
        with threadpool_limits(limits=4):head=base.initialize(x[fi],seed)
        optimizer=torch.optim.Adam(head.parameters(),lr=.03)
        for epoch in range(60):
            head.train();p=head(xt).softmax(-1);consistency,entropy=base.components(p,p,nn_fit)
            loss=consistency-2*entropy;optimizer.zero_grad();loss.backward();optimizer.step()
        head.eval();print(f'Teacher seed {seed} fit-only complete',flush=True);return head
    teachers=[teacher(seed) for seed in [11,47]];primary=teachers[1]
    # Targets and boolean masks are later saved by autograd's indexing ops.
    # no_grad creates ordinary tensors, unlike inference_mode tensors.
    with torch.no_grad():
        p0=teachers[0](xt).softmax(-1);p=primary(xt).softmax(-1)
        y=p.argmax(-1);other=p0.argmax(-1)
        overlap=np.zeros((8,8),dtype=int)
        for a,b in zip(y.numpy(),other.numpy()):overlap[a,b]+=1
        left,right=linear_sum_assignment(-overlap);mapping=dict(zip(right,left))
        agreement=torch.tensor([mapping[int(k)] for k in other])==y
        top=p.topk(2,dim=1).values
        neighbor_agreement=(y[:,None]==y[nn_fit]).float().mean(1)
        seeds=(top[:,0]>=.95)&((top[:,0]-top[:,1])>=.5)&(neighbor_agreement>=.9)&agreement
    assert int(seeds.sum())>0
    seed_table=fit[['key','candidate_group','pixel_sha256']].copy();seed_table['seed_candidate']=seeds.numpy()
    seed_table['pseudo_group']=y.numpy();seed_table.to_csv(OUT/'fit_seed_mask.csv',index=False)
    baseline=evaluate(primary);history.append(dict(candidate='teacher_control',epoch=60,**baseline))
    best_score=baseline['selection_score'];best_name='teacher_control';best_epoch=60;best_state=copy.deepcopy(primary.state_dict())
    for config in plan['configurations']:
        torch.manual_seed(20261106);head=torch.nn.Linear(384,8);head.load_state_dict(primary.state_dict())
        optimizer=torch.optim.Adam(head.parameters(),lr=.003)
        for epoch in range(1,81):
            head.train();logits=head(xt);q=logits.softmax(-1)
            loss=F.cross_entropy(logits[seeds],y[seeds])
            consistency,entropy=base.components(q,q,nn_fit);loss=loss+.2*(consistency-2*entropy)
            if config['view_weight']:
                original=q[paired_fit];alternative=head(xf).softmax(-1);mean=(original+alternative)/2
                js=.5*((original*(original.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean()+
                        (alternative*(alternative.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean())
                loss=loss+config['view_weight']*js
            optimizer.zero_grad();loss.backward();optimizer.step()
            if epoch%10==0:
                metrics=evaluate(head);history.append(dict(candidate=config['name'],epoch=epoch,**metrics))
                if metrics['min_heldout_group']>=5 and metrics['selection_score']>best_score+.000001:
                    best_score=metrics['selection_score'];best_name=config['name'];best_epoch=epoch;best_state=copy.deepcopy(head.state_dict())
        print(config['name'],'complete',flush=True)
    student=torch.nn.Linear(384,8);student.load_state_dict(best_state);student.eval()
    # One predeclared re-entry round, gated entirely by model/view evidence.
    # Neither this update nor the stopping/selection score reads reference labels.
    with torch.no_grad():
        refreshed=student(xt).softmax(-1);refreshed_y=refreshed.argmax(-1);refreshed_top=refreshed.topk(2,dim=1).values
        refreshed_neighbor=(refreshed_y[:,None]==refreshed_y[nn_fit]).float().mean(1)
        flip_ok=torch.zeros(len(fit),dtype=torch.bool)
        flip_ok[paired_fit]=student(xf).argmax(-1)==refreshed_y[paired_fit]
        reenter=(~seeds)&refreshed_top[:,0].ge(.95)&((refreshed_top[:,0]-refreshed_top[:,1])>=.5)&refreshed_neighbor.ge(.9)&refreshed_y.eq(y)&flip_ok
        round2_mask=seeds|reenter
        round2_y=y.clone();round2_y[reenter]=refreshed_y[reenter]
    round2_table=seed_table.copy();round2_table['reentered_from_reserve']=reenter.numpy()
    round2_table['round2_seed_candidate']=round2_mask.numpy();round2_table['pseudo_group']=round2_y.numpy()
    round2_table.to_csv(OUT/'fit_seed_mask_round2.csv',index=False)
    torch.save(dict(state_dict=copy.deepcopy(best_state),input_dimension=384,fit_keys=fit.key.tolist(),heldout_keys=held.key.tolist(),config=plan),OUT/'selected_round1_fit_head.pt')
    if reenter.any():
        head=torch.nn.Linear(384,8);head.load_state_dict(best_state)
        optimizer=torch.optim.Adam(head.parameters(),lr=.003)
        for epoch in range(1,81):
            head.train();logits=head(xt);q=logits.softmax(-1)
            consistency,entropy=base.components(q,q,nn_fit)
            loss=F.cross_entropy(logits[round2_mask],round2_y[round2_mask])+.2*(consistency-2*entropy)
            optimizer.zero_grad();loss.backward();optimizer.step()
            if epoch%10==0:
                metrics=evaluate(head);history.append(dict(candidate='round2_reentered_seeds',epoch=epoch,**metrics))
                if metrics['min_heldout_group']>=5 and metrics['selection_score']>best_score+.000001:
                    best_score=metrics['selection_score'];best_name='round2_reentered_seeds';best_epoch=epoch;best_state=copy.deepcopy(head.state_dict())
        print(f'Round 2 complete: {int(reenter.sum())} model-selected reserve candidates',flush=True)
    records=pd.DataFrame(history);records.to_csv(REPORT/'development_metrics.csv',index=False)
    student.load_state_dict(best_state);student.eval()
    torch.save(dict(state_dict=primary.state_dict(),input_dimension=384,fit_keys=fit.key.tolist(),
                    heldout_keys=held.key.tolist(),config=plan),OUT/'teacher_fit_head.pt')
    torch.save(dict(state_dict=best_state,input_dimension=384,fit_keys=fit.key.tolist(),heldout_keys=held.key.tolist(),
                    selection_frozen_before_reference_audit=True,config=plan),OUT/'selected_fit_head.pt')
    # Inference on all images only after training and candidate selection are frozen.
    with torch.inference_mode():
        probabilities=student(torch.from_numpy(x)).softmax(-1).numpy()
        teacher_probability=primary(torch.from_numpy(x)).softmax(-1).numpy()
    teacher_prediction=profile[['key','split','image_id','pixel_sha256']].copy()
    teacher_prediction['anonymous_group']=teacher_probability.argmax(1)
    teacher_prediction['uncalibrated_probability']=teacher_probability.max(1)
    teacher_prediction.to_csv(OUT/'teacher_fit_assignments.csv',index=False)
    prediction=profile[['key','split','image_id','pixel_sha256']].copy();prediction['anonymous_group']=probabilities.argmax(1)
    prediction['uncalibrated_probability']=probabilities.max(1);prediction.to_csv(OUT/'candidate_assignments.csv',index=False)
    np.savez_compressed(OUT/'candidate_probabilities.npz',keys=profile.key.to_numpy(str),probabilities=probabilities)
    selected=evaluate(student)
    result=dict(status='passed',new_student_models_trained=3+int(reenter.any()),teachers_trained=2,seed_fit_images=int(seeds.sum()),
        reserve_reentered_in_round2=int(reenter.sum()),round2_fit_seed_images=int(round2_mask.sum()),max_self_label_rounds=2,
        ambiguous_fit_images_preserved=int((~seeds).sum()),selected_candidate=best_name,selected_epoch=best_epoch,
        baseline_development=baseline,selected_development=selected,
        proxy_improved=selected['selection_score']>baseline['selection_score'],
        fit_images=len(fit),heldout_images=len(held),heldout_group_overlap=0,flip_fit_images=len(paired_fit),flip_heldout_images=len(paired_val),
        semantic_accuracy=None,backbone_finetuned=False,reference_archive_read=False,display_mapping_read=False,
        baseline_replaced=False,active_labels_promoted=False,student_fit_checkpoint_sha256=sha(OUT/'selected_fit_head.pt'),
        teacher_fit_checkpoint_sha256=sha(OUT/'teacher_fit_head.pt'),
        assignment_sha256=sha(OUT/'candidate_assignments.csv'),seconds=time.time()-started,
        caveat='Selection uses a reused development split and unlabeled proxies on a small flip subset. This is not semantic Accuracy, an independent final holdout, or a proven leaderboard improvement.')
    (REPORT/'summary.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__':main()

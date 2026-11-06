"""Train a named image-only model from non-VLM CNN ontology evidence.

Unlike an anonymous cluster-name substitution, every class has independent
CNN evidence, train-only graph propagation and a newly trained prediction head.
No assistant labels, display mapping or reference archive are read.
"""
from pathlib import Path
import argparse,copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from stage6_semantic_ontology import CLASSES,OBJECTS,MATERIALS,combine_evidence,combine_calibrated

ROOT=Path(__file__).resolve().parents[2]
FEATURE=ROOT/'output/embeddings/stage6_semantic_v1'
OUT=ROOT/'output/groups/stage6_material_resolution_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(data,indent=2,ensure_ascii=False,default=str)+'\n',encoding='utf-8')

def network(kind):
    return torch.nn.Linear(384,8) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(384,128),torch.nn.GELU(),torch.nn.Linear(128,8))

def main():
    global OUT,REPORT
    parser=argparse.ArgumentParser();parser.add_argument('--variant',choices=['raw','calibrated'],default='raw');args=parser.parse_args()
    source_report=REPORT
    if args.variant=='calibrated':
        OUT=ROOT/'output/groups/stage6_material_resolution_calibrated_v2'
        REPORT=source_report/'calibrated_v2'
    REPORT.mkdir(parents=True,exist_ok=True)
    start=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    z=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    assert np.array_equal(z['keys'],p.key.to_numpy(str));x=z['features'];positions=dict(zip(p.key,range(len(p))))
    objects=json.loads((source_report/'object_evidence_provenance.json').read_text())
    materials=json.loads((source_report/'material_evidence_provenance.json').read_text())
    probabilities={};object_strength={};source_files=[]
    for view in ['original224','hflip224']:
        op=FEATURE/f'resnet_objects_{view}.npz';mp=FEATURE/f'minc_materials_{view}.npz'
        ob=np.load(op,allow_pickle=False);ma=np.load(mp,allow_pickle=False)
        assert np.array_equal(ob['keys'],ma['keys'])
        combine=combine_calibrated if args.variant=='calibrated' else combine_evidence
        prob,signal,missing=combine(ob['logits'],objects['categories'],ma['probabilities'],materials['categories'])
        probabilities[view]=(ob['keys'],prob);object_strength[view]=signal;source_files.extend([op,mp])
        if view=='original224':assert np.array_equal(ob['keys'],p.key.to_numpy(str))
    split_path=ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv'
    split=pd.read_csv(split_path).set_index('key')
    fit_keys=split.index[split.role.eq('fit_train')];val_keys=split.index[split.role.eq('heldout_train')]
    fi=np.array([positions[k] for k in fit_keys]);vi=np.array([positions[k] for k in val_keys])
    assert len(fi)==1538 and len(vi)==381 and not set(split.loc[fit_keys,'candidate_group'])&set(split.loc[val_keys,'candidate_group'])
    assert not set(p.iloc[fi].pixel_sha256)&set(p.iloc[vi].pixel_sha256)
    flip=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    assert np.array_equal(flip['keys'],probabilities['hflip224'][0])
    flip_lookup=dict(zip(flip['keys'],range(len(flip['keys']))))
    fv=np.array([flip_lookup[k] for k in val_keys if k in flip_lookup]);vpaired=np.array([positions[k] for k in val_keys if k in flip_lookup])
    ft=np.array([flip_lookup[k] for k in fit_keys if k in flip_lookup]);fpaired=np.array([positions[k] for k in fit_keys if k in flip_lookup])
    original_prior=probabilities['original224'][1]
    plan=dict(task='Named train-only graph/student from non-VLM semantic CNN evidence',class_order=CLASSES,
        difference_from_prior_runs='Uses public pretrained image-model category names, not LLM labels or interpretations of anonymous clusters.',
        ontology=dict(objects=OBJECTS,materials=MATERIALS,missing_object_names=missing),
        fusion=('Conditional source probabilities with source-coverage weights and explicit whole-object electronic/cardboard priority.' if args.variant=='calibrated' else '0.65 material evidence +0.35 max object probability; electronic/cardboard use1.5 max object probability. Normalize across eight.'),
        variant=args.variant,
        fit_keys=fit_keys.tolist(),heldout_keys=val_keys.tolist(),split_sha256=sha(split_path),
        configurations=[dict(alpha=a,head=h) for a in [.4,.7,.9] for h in ['linear','mlp']],
        graph=dict(neighbors=10,temperature=.07,iterations=60,anchor_probability=.75,anchor_neighbor_consensus=.7),
        training=dict(epochs=160,lr=.01,soft_target_cross_entropy=True,view_loss=.2),
        selection='Development .35 flip stability +.35 independent pretrained-CNN evidence agreement +.30 train-neighbor agreement; all8 occupied and min5 heldout.',
        score_is_semantic_accuracy=False,reference_archive_read=False,assistant_display_names_read=False,
        test_in_training_or_selection=False,backbone_finetuned=False,head_trained_on_competition_fit=True,
        sources={str(path.relative_to(ROOT)):sha(path) for path in source_files},
        source_sha256=sha(__file__),ontology_source_sha256=sha(ROOT/'eksperimen/skrip/stage6_semantic_ontology.py'))
    dump(REPORT/'training_plan_before_fit.json',plan)
    with threadpool_limits(limits=4):
        similarity=x[fi]@x[fi].T;np.fill_diagonal(similarity,-np.inf)
        nn=np.argsort(-similarity,axis=1,kind='stable')[:,:10]
        edge=softmax(np.take_along_axis(similarity,nn,axis=1)/.07,axis=1)
        near_val=np.argsort(-(x[vi]@x[fi].T),axis=1,kind='stable')[:,:10]
    prior=original_prior[fi]
    anchor=prior.max(1)>=.75
    anchor&=(prior.argmax(1)[:,None]==prior.argmax(1)[nn]).mean(1)>=.7
    torch.manual_seed(20261106)
    xt=torch.from_numpy(x[fi]);xv=torch.from_numpy(x[vi]);xf=torch.from_numpy(flip['features'][fv])
    fit_flip_positions=np.array([list(fi).index(i) for i in fpaired]);fit_flip=torch.from_numpy(flip['features'][ft])
    history=[];best=None;models=[];targets=[]
    for alpha in [.4,.7,.9]:
        target=prior.copy()
        for _ in range(60):
            target=(1-alpha)*prior+alpha*(target[nn]*edge[:,:,None]).sum(1)
            target[anchor]=prior[anchor]
            target/=target.sum(1,keepdims=True)
        targets.append(target)
        for kind in ['linear','mlp']:
            torch.manual_seed(20261106);head=network(kind)
            optimizer=torch.optim.Adam(head.parameters(),lr=.01,weight_decay=1e-4)
            t=torch.from_numpy(target.astype('float32'))
            for epoch in range(1,161):
                head.train();logits=head(xt);q=logits.softmax(-1)
                loss=-(t*logits.log_softmax(-1)).sum(1).mean()
                alternative=head(fit_flip).softmax(-1);original=q[fit_flip_positions];mean=(original+alternative)/2
                js=.5*((original*(original.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean()+
                       (alternative*(alternative.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean())
                loss=loss+.2*js
                optimizer.zero_grad();loss.backward();optimizer.step()
                if epoch%10==0:
                    head.eval()
                    with torch.inference_mode():
                        fit_prediction=head(xt).argmax(1).numpy();v=head(xv).argmax(1).numpy()
                        flip_prediction=head(xf).argmax(1).numpy()
                    vlookup=dict(zip(vi,v));paired=np.array([vlookup[k] for k in vpaired])
                    stability=float((paired==flip_prediction).mean())
                    teacher_agreement=float((v==original_prior[vi].argmax(1)).mean())
                    neighbor=float((v[:,None]==fit_prediction[near_val]).mean())
                    counts=np.bincount(v,minlength=8)
                    score=.35*stability+.35*teacher_agreement+.30*neighbor
                    row=dict(alpha=alpha,head=kind,epoch=epoch,flip_stability=stability,cnn_evidence_agreement=teacher_agreement,
                        neighbor_agreement=neighbor,score=score,occupied=int((counts>0).sum()),min_class=int(counts.min()))
                    history.append(row)
                    # Preserve even rare predicted classes; occupancy remains a diagnostic.
                    eligible=row['occupied']==8 and row['min_class']>=5
                    if eligible and (best is None or score>best['score']+1e-7):
                        best=dict(**row,state=copy.deepcopy(head.state_dict()))
            models.append(dict(alpha=alpha,head=kind,state=copy.deepcopy(head.state_dict())))
            print(f'semantic head {kind} alpha{alpha} complete, occupied={history[-1]["occupied"]}',flush=True)
    if best is None:
        # Do not fabricate class quotas to make an ineligible model look complete.
        raise RuntimeError('No model covered all8 heldout classes; inspect source priors before any promotion.')
    head=network(best['head']);head.load_state_dict(best['state']);head.eval()
    torch.save(dict(state_dict=best.pop('state'),kind=best['head'],class_order=CLASSES,fit_keys=fit_keys.tolist(),heldout_keys=val_keys.tolist(),
        selected=best,plan=plan,reference_archive_read=False),OUT/'selected_semantic_head.pt')
    with torch.inference_mode():final=head(torch.from_numpy(x)).softmax(-1).numpy()
    prediction=final.argmax(1);ordered=np.sort(final,axis=1)
    records=p[['key','split','image_id','pixel_sha256','processed_path']].copy()
    records['predicted_class']=np.array(CLASSES)[prediction]
    records['uncalibrated_probability']=final.max(1);records['margin']=ordered[:,-1]-ordered[:,-2]
    records['entropy']=-(final*np.log(np.maximum(final,1e-8))).sum(1)
    records['cnn_prior_disagrees']=prediction!=original_prior.argmax(1)
    records['needs_review']=records.uncalibrated_probability.lt(.65)|records.margin.lt(.25)|records.cnn_prior_disagrees
    records['ground_truth_verified']=False;records['method']='CNN_material_object_evidence_train_only_graph_student_v1'
    records.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=p.key.to_numpy(str),probabilities=final,prior_probabilities=original_prior)
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    pd.DataFrame(targets[1],index=fit_keys,columns=CLASSES).to_csv(OUT/'alpha07_fit_soft_targets.csv',index_label='key')
    result=dict(status='passed',heads_trained=len(models),selected=best,class_order=CLASSES,fit_images=len(fi),heldout_images=len(vi),
        group_overlap=0,train_counts=records[records.split.eq('train')].predicted_class.value_counts().to_dict(),
        test_counts=records[records.split.eq('test')].predicted_class.value_counts().to_dict(),
        mixed_predicted_labels=int((~records.predicted_class.isin(CLASSES)).sum()),
        train_needs_review=int(records[records.split.eq('train')].needs_review.sum()),
        reference_archive_read=False,semantic_accuracy=None,baseline_replaced=False,backbone_finetuned=False,
        uncertainty_preserved=True,seconds=time.time()-start,assignment_sha256=sha(OUT/'named_assignments.csv'))
    dump(REPORT/'summary.json',result)
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),
        checkpoint_sha256=sha(OUT/'selected_semantic_head.pt'),summary_sha256=sha(REPORT/'summary.json'),source_sha256=sha(__file__),
        ontology_sha256=sha(ROOT/'eksperimen/skrip/stage6_semantic_ontology.py'),protected={n:sha(ROOT/n) for n in protected}))
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

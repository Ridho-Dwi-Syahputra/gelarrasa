"""Trusted-gradient noisy-example reweighting on frozen image-only features.

Linear-head specialization of the epsilon-zero one-step meta-gradient, with an
additional clean supervised loss. Not a reproduction of Ren18's full recipe.
Only34existing genuine sources are human targets; full review is never read.
"""
import datetime
import json
import time
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from selective_expert_refinement import wilson_lower
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


def alignment(x,logits,y,clean_x,clean_logits,clean_y,clean_w):
    residual=logits.softmax(1)-F.one_hot(y,logits.shape[1])
    clean_residual=(clean_logits.softmax(1)-F.one_hot(clean_y,logits.shape[1]))*clean_w[:,None]
    gw=clean_residual.T@clean_x;gb=clean_residual.sum(0)
    return ((residual@gw)*x).sum(1)+residual@gb


def numerical_check():
    torch.manual_seed(17);x=torch.randn(5,4,dtype=torch.float64);cx=torch.randn(3,4,dtype=torch.float64)
    y=torch.tensor([0,1,2,1,0]);cy=torch.tensor([2,0,1]);w=torch.tensor([.2,.3,.5],dtype=torch.float64)
    theta=torch.randn(3,4,dtype=torch.float64,requires_grad=True);bias=torch.randn(3,dtype=torch.float64,requires_grad=True)
    eps=torch.zeros(5,dtype=torch.float64,requires_grad=True);logits=F.linear(x,theta,bias)
    noisy=(F.cross_entropy(logits,y,reduction='none')*eps).sum();gt,gb=torch.autograd.grad(noisy,(theta,bias),create_graph=True)
    eta=.1;clean=(F.cross_entropy(F.linear(cx,theta-eta*gt,bias-eta*gb),cy,reduction='none')*w).sum()
    meta=torch.autograd.grad(clean,eps)[0];closed=alignment(x,logits,y,cx,F.linear(cx,theta,bias),cy,w)
    error=float((meta+eta*closed).abs().max().detach());assert error<1e-10
    return {'check':'closed_form_matches_autograd_epsilon_zero_virtual_SGD','max_absolute_error':error}


def main():
    start=time.time();prefix='gradient_reweight';torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    check=numerical_check();prior_lower=wilson_lower(12,12)
    plan={'method':'trusted_gradient_linear_example_reweighting_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'reference':'https://proceedings.mlr.press/v80/ren18a.html','adaptation':'epsilonzeroone-stepSGDlinearheadgradientalignment,plus.5cleanCE;notfullpaperrecipe;curatedhumansnotunbiasedvalidation',
          'targets':'coldautomaticconsensuscoreand34humanONLY;activeassignmentcategoriesunused',
          'banks':['base_regional','cnn_base'],'controls':['uniform','gradient_align'],'epochs':30,'checkpoints':[15,30],
          'optimizer':'SGDlr.3,momentum.9,weightdecay.001;gradclip5','steps_per_epoch':12,'auto_batch':128,
          'clean_batch':'oneof8cachedviewsforeachfit-humansource,sourceandclassbalanced;noquerysourceviews',
          'warmup_epochs':2,'reweight':'max(0,auto_gradient dot classbalancedclean_gradient)*coldconsensusconfidence,normalize;uniformfallbackifallsumzero',
          'objective':'weightedautoCE+.5classbalancedcleanCE','normalizer':'centeringouterTRAINonlythenrowL2',
          'folds':'fivegeometrysourcegroups;queryexcludedfromauto,cleanmeta,cleanCE,andnormalizer',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],'selection':'globalbestmacrothenmatches;elsebestprecisionfirstWilsonranking',
          'global_guard':'>=25/34,macro>.75+1e-12,and>=10coveredat>=.9precision',
          'risk_guard':'wholematches>=20,macro>=.65,coverage>=12,precision>=.95,Wilsonranking>=prior12/12lower+.02',
          'wilson_limit':'rankingonlyonreused34development,NOTconfidencecertificate','baseline_risk_lower':prior_lower,
          'acceptance':'selectedthresholdandbothfinalTRAINgraphsagree;humanprotected','maximum_changes':20,
          'cap':'descendingselectedprobability,stablekeyorder','private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'evaluation_limit':'all34developmentreused;notuntouchedAccuracy;graphnotcalibrationvalidation',
          'numerical_gradient_check':check,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];human=np.zeros(n,bool);human[pos]=True
    y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold;core=a.automatic_anchor.to_numpy(bool)&~human;confidence=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);aug=np.load(NOTES/'.runtime/human_source_aug_features.npz',allow_pickle=False)
    assert np.array_equal(b['keys'],keys) and np.array_equal(c['keys'],keys) and np.array_equal(aug['keys'],s.key.to_numpy(str))
    bg=unit(b['features'][:,:768]);ag=unit(aug['features'][:,:,:768]);cg=unit(c['features']);ac=unit(aug['cnn_features'])
    banks={'base_regional':(unit(np.c_[bg,unit(b['features'][:,2304:])]),unit(np.concatenate([ag,unit(aug['features'][:,:,2304:])],-1))),
           'cnn_base':(unit(np.c_[cg,bg]),unit(np.concatenate([ac,ag],-1)))}
    outputs={};training=[];development=[];target=torch.from_numpy(y).long();clean_target=torch.from_numpy(gold).long()
    for bank_number,(bank,(raw,raw_views)) in enumerate(banks.items()):
        for mode in plan['controls']:
            values={epoch:np.zeros((n,8),np.float32) for epoch in plan['checkpoints']}
            for fold in range(5):
                fit=np.flatnonzero((folds!=fold)&core);query=np.flatnonzero(folds==fold);hs=np.flatnonzero(folds[pos]!=fold)
                assert not set(groups[np.r_[fit,pos[hs]]])&set(groups[query])
                center=raw[folds!=fold].mean(0);x=torch.from_numpy(unit(raw-center).astype(np.float32));views=torch.from_numpy(unit(raw_views-center).astype(np.float32))
                rng=np.random.default_rng(20261108+bank_number*100+fold);torch.manual_seed(20261108+bank_number*100+fold)
                model=nn.Linear(x.shape[1],8);nn.init.normal_(model.weight,std=.01);nn.init.zeros_(model.bias)
                optimizer=torch.optim.SGD(model.parameters(),lr=.3,momentum=.9,weight_decay=.001)
                class_count=np.bincount(gold[hs],minlength=8);cw=torch.from_numpy((1/np.maximum(class_count[gold[hs]],1)).astype(np.float32));cw/=cw.sum()
                fallback=0;kept=[]
                for epoch in range(30):
                    model.train();losses=[]
                    for _ in range(12):
                        ids=rng.choice(fit,128,replace=len(fit)<128);view=rng.integers(8,size=len(hs));cx=views[hs,view];ax=x[ids]
                        logits=model(ax);clean_logits=model(cx);aw=torch.from_numpy(confidence[ids].astype(np.float32))
                        if mode=='gradient_align' and epoch>=2:
                            with torch.no_grad():aw*=alignment(ax,logits,target[ids],cx,clean_logits,clean_target[hs],cw).clamp_min(0)
                            kept.append(float((aw>0).float().mean()))
                            if float(aw.sum())<=1e-12:aw=torch.from_numpy(confidence[ids].astype(np.float32));fallback+=1
                        aw/=aw.sum();loss=(F.cross_entropy(logits,target[ids],reduction='none')*aw).sum()+.5*(F.cross_entropy(clean_logits,clean_target[hs],reduction='none')*cw).sum()
                        assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
                    if epoch+1 in values:
                        model.eval()
                        with torch.inference_mode():values[epoch+1][query]=model(x[query]).softmax(1).numpy()
                        training.append({'bank':bank,'mode':mode,'fold':fold,'epoch':epoch+1,'loss':float(np.mean(losses)),'auto_fit':len(fit),'human_fit_sources':len(hs),'mean_positive_weight_fraction':float(np.mean(kept)) if kept else None,'uniform_fallback_steps':fallback})
                print('Gradient reweight',bank,mode,'fold',fold+1,'/5 seconds',round(time.time()-start),flush=True)
            for epoch,p in values.items():
                cfg={'bank':bank,'mode':mode,'epoch':epoch,'anchor_matches':int((p[pos].argmax(1)==gold).sum()),'development_balanced_recall':float(balanced_accuracy_score(gold,p[pos].argmax(1)))}
                name=f'{bank}_{mode}_{epoch}';outputs[name]=p
                for threshold in plan['threshold_grid']:
                    covered=p[pos].max(1)>=threshold;count=int(covered.sum());good=int((covered&(p[pos].argmax(1)==gold)).sum());precision=good/count if count else 0.;lower=wilson_lower(good,count)
                    global_ok=cfg['anchor_matches']>=25 and cfg['development_balanced_recall']>.75+1e-12 and count>=10 and precision>=.9
                    risk_ok=cfg['anchor_matches']>=20 and cfg['development_balanced_recall']>=.65 and count>=12 and precision>=.95 and lower>=prior_lower+.02
                    development.append({**cfg,'name':name,'threshold':threshold,'covered':count,'correct':good,'precision':precision,'wilson_ranking':lower,'global_eligible':global_ok,'risk_eligible':risk_ok})
    global_options=[v for v in development if v['global_eligible']];risk_options=[v for v in development if v['risk_eligible']]
    if global_options:chosen=max(global_options,key=lambda v:(v['development_balanced_recall'],v['anchor_matches'],v['covered'],v['threshold']));objective='global'
    elif risk_options:chosen=max(risk_options,key=lambda v:(v['wilson_ranking'],v['covered'],v['development_balanced_recall'],v['threshold']));objective='risk_first'
    else:chosen=max(development,key=lambda v:(v['development_balanced_recall'],v['anchor_matches'],v['wilson_ranking']));objective='rejected'
    accepted=objective!='rejected';p=outputs[chosen['name']];primary=p.argmax(1);score=p.max(1)
    base_region=banks['base_regional'][0];cnn=banks['cnn_base'][0]
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(unit(v-v.mean(0)),p,pos,gold) for v in [base_region,cnn]])
    support=(graphs==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=~human&(primary!=current)&(score>=chosen['threshold'])&(support==2)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'robust_neural_assignment_snapshot',score[i],'uncalibrated_neural_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(training).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_gradient_reweight_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[chosen],qualified_heads=[chosen],selected_knn_heads=[],
        human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],confidence_calibration={'selected':chosen,'independent_validation':False,'objective':objective},human_diagnostic_protocol=plan['evaluation_limit'],accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()),new_vit_propagation_changes=0,
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'],numerical_gradient_check=check)
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','objective':objective,'prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline','seconds','numerical_gradient_check']},indent=2),flush=True)


if __name__=='__main__':main()

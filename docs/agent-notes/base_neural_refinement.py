"""Noise-aware nonlinear assignment heads on DINOv2 Base, grouped train only.

Targets are cold independent-model consensus, not historical routing labels.
Only the existing34budgeted participant sources are human targets.
"""
import json
import time
import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    start=time.time();prefix='base_neural';torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    feature_path=NOTES/'.runtime/dinov2_base_train_features.npz'
    plan={'method':'DINOv2Base_grouped_CE_GCE_assignment_heads_v1','train_rows':1904,'test_rows_in_fit':0,'human_sources':34,'new_manual_labels':0,
          'targets':'automatic_full_seed_targets.automatic_class forcore,not historicalrouting;34humanoverridesonly',
          'banks':['base_cls','base_local','base_context'],'architecture':'LayerNorm,Linear256,GELU,Dropout.15,Linear8',
          'losses':['CE','GCE(q=.7,CEwarmup5)'],'human_weights':[8.,32.],'checkpoints':[20,40],
          'optimizer':'AdamWlr.001,weightdecay.001,gradclip5,batch256','epochs':40,'folds':5,
          'automatic_filter':'insidefitfoldonly,contradictory nearesthumancategorycosine>=.65andmargin>=.02;neverremovehuman',
          'selection':'perbankbest34anchor groupedbalancedrecall;primarybestbank;qualifybanks>=.625andwithin.1ofbest',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoveragewith>=10coveredand>=.9empiricalselectiveprecision;tiesstricter',
          'acceptance':'allqualifiedheadsmeetlearnedminimumprobabilityandbothnormalizedimagegraphsagree;humanprotected',
          'previous_auto_refinements':'also minimumprobability>=max(.95,learnedthreshold)',
          'maximum_changes':40,'cap':'descending minimumprobability,stablekeyorder',
          'guard':'primary>=23/34andbalancedrecall>=.725andthresholdexistsand>=2qualifiedbanks',
          'evaluation_limit':'development reusedforselection/calibration;notfinalsemanticAccuracy;fullgraphnotusedforcalibrationevaluation',
          'private_review_or_archive_targets_read':False,'feature_sha256':sha(feature_path),'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    z=np.load(feature_path,allow_pickle=False);assert np.array_equal(z['keys'],keys) and np.array_equal(z['source_sha256'],frame.source_sha256.to_numpy(str))
    f=z['features'];cls=unit(f[:,:768]);roi=unit(f[:,2304:]);patch=unit(f[:,768:1536])
    banks={'base_cls':cls,'base_local':unit(np.c_[cls,roi]),'base_context':unit(np.c_[cls,roi,patch])}
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=g.group.to_numpy(),g.fold.to_numpy()
    seed=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seed.key]);gold=np.array([CLASSES.index(c) for c in seed.assigned_class])
    assert len(pos)==34 and seed.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold
    eligible=a.automatic_anchor.to_numpy(bool)|human;base=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    target=torch.from_numpy(y).long();is_human=torch.from_numpy(human);history=[];selections=[];selected_probabilities=[];all_outputs=[]
    for bank_number,(name,values) in enumerate(banks.items()):
        x=torch.from_numpy(values.astype('float32'));sim=values@values.T;fit_folds=[]
        for fold in range(5):
            fit=np.flatnonzero((folds!=fold)&eligible);query=np.flatnonzero(folds==fold);assert not set(groups[fit])&set(groups[query])
            anchors=fit[human[fit]];near=np.stack([sim[np.ix_(fit,anchors[y[anchors]==c])].max(1) if (y[anchors]==c).any() else np.full(len(fit),-np.inf) for c in range(8)],axis=1)
            cat=near.argmax(1);value=near[np.arange(len(fit)),cat].copy();near[np.arange(len(fit)),cat]=-np.inf
            remove=(cat!=y[fit])&(value>=.65)&(value-near.max(1)>=.02)&~human[fit]
            fit_folds.append((fit[~remove],query,int(remove.sum())))
        candidates=[]
        for objective in ['CE','GCE']:
            for human_weight in [8.,32.]:
                outputs={epoch:np.zeros((n,8),np.float32) for epoch in [20,40]}
                for fold,(fit,query,excluded) in enumerate(fit_folds):
                    torch.manual_seed(20261108+bank_number*100+fold);rng=np.random.default_rng(20261108+bank_number*100+fold)
                    model=nn.Sequential(nn.LayerNorm(x.shape[1]),nn.Linear(x.shape[1],256),nn.GELU(),nn.Dropout(.15),nn.Linear(256,8))
                    optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                    weight=torch.from_numpy(base.astype('float32'));weight[is_human]=human_weight
                    first_loss=None;last_loss=None
                    for epoch in range(40):
                        model.train();order=rng.permutation(fit);total=0.;denom=0
                        for offset in range(0,len(order),256):
                            batch=order[offset:offset+256];logits=model(x[batch]);ce=F.cross_entropy(logits,target[batch],reduction='none')
                            if objective=='GCE' and epoch>=5:
                                py=logits.softmax(1).gather(1,target[batch,None]).squeeze(1).clamp_min(1e-7)
                                items=torch.where(is_human[batch],ce,(1-py.pow(.7))/.7)
                            else:items=ce
                            w=weight[batch];loss=(items*w).sum()/w.sum();assert torch.isfinite(loss)
                            optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step()
                            total+=float(loss.detach())*len(batch);denom+=len(batch)
                        last_loss=total/denom
                        if first_loss is None:first_loss=last_loss
                        if epoch+1 in outputs:
                            model.eval()
                            with torch.inference_mode():outputs[epoch+1][query]=model(x[query]).softmax(1).numpy()
                    history.append({'bank':name,'objective':objective,'human_weight':human_weight,'fold':fold,'fit':len(fit),'excluded_automatic':excluded,
                                    'first_loss':first_loss,'last_loss':last_loss})
                for epoch,p in outputs.items():
                    assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5)
                    pred=p[pos].argmax(1);quality=float(balanced_accuracy_score(gold,pred));matches=int((pred==gold).sum())
                    candidates.append(({'bank':name,'objective':objective,'human_weight':human_weight,'epoch':epoch,'development_balanced_recall':quality,'anchor_matches':matches},p))
                print('Base neural completed:',name,objective,'humanweight',human_weight,'seconds',round(time.time()-start),flush=True)
        chosen,p=max(candidates,key=lambda v:v[0]['development_balanced_recall']);selections.append(chosen);selected_probabilities.append(p)
        all_outputs.extend([c[0] for c in candidates]);print('Selected Base neural bank:',chosen,flush=True)
    winner=max(range(len(selections)),key=lambda i:selections[i]['development_balanced_recall']);chosen=selections[winner];best=chosen['development_balanced_recall']
    trusted=[i for i,h in enumerate(selections) if h['development_balanced_recall']>=.625 and h['development_balanced_recall']>=best-.1]
    primary=selected_probabilities[winner].argmax(1);score=np.min([selected_probabilities[i][np.arange(n),primary] for i in trusted],axis=0) if trusted else np.zeros(n)
    correct=primary[pos]==gold;calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());matches=int((covered&correct).sum())
        calibration.append({'threshold':t,'covered':count,'correct':matches,'precision':matches/count if count else None,'eligible':count>=10 and matches/count>=.9})
    choices=[r for r in calibration if r['eligible']];cal=max(choices,key=lambda r:(r['covered'],r['threshold'])) if choices else None
    threshold=cal['threshold'] if cal else float('inf');graph=np.load(NOTES/'.runtime/normalized_base_outputs.npz',allow_pickle=False);assert np.array_equal(graph['keys'],keys)
    graph_support=(graph['graphs']==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);preserve=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    gate=(score>=threshold)&(graph_support==2)&~human&(primary!=current)&(~preserve|(score>=max(.95,threshold)))
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    accepted=cal is not None and len(trusted)>=2 and chosen['anchor_matches']>=23 and best>=.725
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'robust_neural_assignment_snapshot',score[i],'uncalibrated_neural_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(all_outputs).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_probability':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(selected_probabilities),accepted=gate)
    report=json.loads((NOTES/'json/normalized_base_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_base_neural_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selections,selected_knn_heads=[],
                  human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=best,
                  confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},qualified_heads=[selections[i] for i in trusted],
                  accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()),new_vit_propagation_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),
                  source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':main()

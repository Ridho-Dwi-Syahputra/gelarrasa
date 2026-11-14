"""Source-group supervised metric adaptation on frozen image-only features.

34 existing genuine sources only; cold automatic labels explicitly remain noisy.
Human augmentations do not create new annotation sources or validation samples.
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
from fewshot_train_only_probe import ROOT, NOTES, CLASSES, sha, dump


def unit(x): return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)


class MetricHead(nn.Module):
    def __init__(self, dimensions):
        super().__init__()
        self.encoder=nn.Sequential(nn.LayerNorm(dimensions),nn.Linear(dimensions,128),nn.GELU(),nn.Dropout(.1))
        self.projection=nn.Linear(128,64);self.classifier=nn.Linear(128,8)

    def forward(self,x):
        h=self.encoder(x)
        return self.classifier(h),F.normalize(self.projection(h),dim=-1)


def main():
    start=time.time();prefix='source_contrastive';torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'source_group_supervised_contrastive_feature_adaptation_v1','train_rows':1904,'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'targets':'same34humanandcoldautomaticconsensusonly;noactivepseudo-assignmenttargets',
          'banks':['base_global','base_regional','cnn_base'],'architecture':'LayerNorm,128GELUdropout.1,64normalizedprojectionand8classifier',
          'loss':'human-weightedCE+0.2SupCon(tau.1);humanweight8,automaticconfidenceweight;allanchorscontribute;positivesmustshareclassanddiffergeometrysource',
          'augmentation':'8cachedviewsfor34human,randomoneviewperselectedsource;not8independentlabels',
          'batch':'96coldautomatic+32randomfit-humansources;12steps/epoch;querygroupsneverfit',
          'training':'AdamWlr.001,weightdecay.001,gradclip5,40epochs;check20and40',
          'automatic_filter':'fit-onlynearesthumancontradictioncos>=.65andmargin>=.02;humanprotected',
          'selection':'perbankbest34group-heldoutmacrorecall,thenmatches;overallbestprimary;qualifybanksmacro>=.625andwithin.1',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],'threshold_selection':'maxcoverage>=10at>=.9empiricalprecision;tiesstricter',
          'acceptance':'allqualifiedheadsmeetthreshold,bothnormalizedTRAINgraphsagree;previousrefinementsscore>=max(.95,threshold);humanprotected',
          'guard':'>=25/34andmacro>.75and>=2qualifiedbanksandthresholdexists','maximum_changes':40,
          'cap':'descendingminimumprobability,stablekeyorder','evaluation_limit':'34reuseddevelopmentnotindependentAccuracy;graphnotcalibrationvalidation',
          'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__),
          'reference':'https://proceedings.neurips.cc/paper/2020/hash/d89a66c7c80a29b1bdbab0f2a1a94af8-Abstract.html','replication':'adaptedprojectionheadandmixednoisy-targetloss,notpublishedfull-backboneSupConrecipe'}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all()
    lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    group_table=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups=group_table.group.to_numpy();folds=group_table.fold.to_numpy();group_code=pd.factorize(groups)[0]
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold;human=np.zeros(n,bool);human[pos]=True
    core=a.automatic_anchor.to_numpy(bool)&~human;base=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.)
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);aug=np.load(NOTES/'.runtime/human_source_aug_features.npz',allow_pickle=False)
    assert np.array_equal(b['keys'],keys) and np.array_equal(c['keys'],keys) and np.array_equal(aug['keys'],s.key.to_numpy(str))
    bg=unit(b['features'][:,:768]);ag=unit(aug['features'][:,:,:768]);cg=unit(c['features']);ac=unit(aug['cnn_features'])
    banks={'base_global':(bg,ag),'base_regional':(unit(np.c_[bg,unit(b['features'][:,2304:])]),unit(np.concatenate([ag,unit(aug['features'][:,:,2304:])],-1))),
           'cnn_base':(unit(np.c_[cg,bg]),unit(np.concatenate([ac,ag],-1)))}
    outputs={};history=[];training=[]
    for bank_number,(name,(values,views)) in enumerate(banks.items()):
        x=torch.from_numpy(values.astype(np.float32));av=torch.from_numpy(views.astype(np.float32));p_by_epoch={e:np.zeros((n,8),np.float32) for e in [20,40]};sim=values@values.T
        for fold in range(5):
            rng=np.random.default_rng(20261108+bank_number*100+fold);torch.manual_seed(20261108+bank_number*100+fold)
            fit=np.flatnonzero((folds!=fold)&core);hs=np.flatnonzero(folds[pos]!=fold);query=np.flatnonzero(folds==fold)
            assert not set(groups[np.r_[fit,pos[hs]]])&set(groups[query])
            nearest=np.stack([sim[np.ix_(fit,pos[hs[gold[hs]==k]])].max(1) if (gold[hs]==k).any() else np.full(len(fit),-np.inf) for k in range(8)],1)
            category=nearest.argmax(1);value=nearest[np.arange(len(fit)),category].copy();nearest[np.arange(len(fit)),category]=-np.inf
            fit=fit[~((category!=y[fit])&(value>=.65)&(value-nearest.max(1)>=.02))]
            model=MetricHead(x.shape[1]);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
            for epoch in range(40):
                model.train();losses=[]
                for _ in range(12):
                    auto=rng.choice(fit,96,replace=len(fit)<96);human_source=rng.choice(hs,32,replace=True);view=rng.integers(8,size=32)
                    batch=torch.cat([x[auto],av[human_source,view]]);labels=torch.from_numpy(np.r_[y[auto],gold[human_source]]).long()
                    owners=torch.from_numpy(np.r_[group_code[auto],group_code[pos[human_source]]]);weights=torch.from_numpy(np.r_[base[auto],np.full(32,8.)].astype(np.float32))
                    logits,z=model(batch);ce=F.cross_entropy(logits,labels,reduction='none');scores=z@z.T/.1
                    different_source=owners[:,None]!=owners[None,:];positive=different_source&(labels[:,None]==labels[None,:]);valid=positive.any(1)
                    logden=torch.logsumexp(scores.masked_fill(~different_source,-1e9),1,keepdim=True)
                    contrast=-((scores-logden)*positive).sum(1)/positive.sum(1).clamp_min(1)
                    assert valid.any();loss=(ce*weights).sum()/weights.sum()+.2*(contrast[valid]*weights[valid]).sum()/weights[valid].sum()
                    assert torch.isfinite(loss);optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();losses.append(float(loss.detach()))
                if epoch+1 in p_by_epoch:
                    model.eval()
                    with torch.inference_mode():p_by_epoch[epoch+1][query]=model(x[query])[0].softmax(1).numpy()
                    training.append({'bank':name,'fold':fold,'epoch':epoch+1,'loss':float(np.mean(losses)),'fit_auto':len(fit),'human_fit_sources':len(hs)})
            print('Source contrastive',name,'fold',fold+1,'/5 seconds',round(time.time()-start),flush=True)
        for epoch,p in p_by_epoch.items():
            outputs[(name,epoch)]=p;history.append({'bank':name,'epoch':epoch,'matches':int((p[pos].argmax(1)==gold).sum()),'development_balanced_recall':float(balanced_accuracy_score(gold,p[pos].argmax(1)))})
    selected=[max([r for r in history if r['bank']==bank],key=lambda r:(r['development_balanced_recall'],r['matches'])) for bank in banks]
    winner=max(selected,key=lambda r:(r['development_balanced_recall'],r['matches']));best=winner['development_balanced_recall'];qualified=[r for r in selected if r['development_balanced_recall']>=.625 and r['development_balanced_recall']>=best-.1]
    primary=outputs[(winner['bank'],winner['epoch'])].argmax(1);score=np.min([outputs[(r['bank'],r['epoch'])][np.arange(n),primary] for r in qualified],axis=0) if qualified else np.zeros(n)
    calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());good=int((covered&(primary[pos]==gold)).sum());calibration.append({'threshold':t,'covered':count,'correct':good,'precision':good/count if count else None,'eligible':count>=10 and good/count>=.9})
    feasible=[r for r in calibration if r['eligible']];cal=max(feasible,key=lambda r:(r['covered'],r['threshold'])) if feasible else None
    accepted=bool(winner['matches']>=25 and best>.75+1e-12 and len(qualified)>=2 and cal is not None)
    graph=np.load(NOTES/'.runtime/normalized_base_outputs.npz',allow_pickle=False);assert np.array_equal(graph['keys'],keys);graph_support=(graph['graphs']==primary).sum(0)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);preserve=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy();threshold=cal['threshold'] if cal else np.inf
    gate=~human&(primary!=current)&(score>=threshold)&(graph_support==2)&(~preserve|(score>=max(.95,threshold)))
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');chosen=rank[gate[rank]][:40];gate[:]=False;gate[chosen]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'robust_neural_assignment_snapshot',score[i],'uncalibrated_neural_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(training).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack([outputs[(r['bank'],r['epoch'])] for r in selected]),accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_contrastive_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=winner,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,selected_knn_heads=[],qualified_heads=qualified,
        human_leave_source_group_out_matches=winner['matches'],human_holdout_rows=34,development_balanced_recall=best,human_diagnostic_protocol=plan['evaluation_limit'],confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},
        accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()),new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','accepted_changes':int(gate.sum()),'guard':plan['guard'],'prediction_sha256':sha(dest),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline','seconds']},indent=2),flush=True)


if __name__=='__main__':main()

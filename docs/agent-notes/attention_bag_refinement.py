"""Grouped weakly supervised attention on frozen image patches, organizer train only."""
import json
import time
import datetime
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from selective_expert_refinement import wilson_lower
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


class BagHead(nn.Module):
    def __init__(self,mode):
        super().__init__();self.mode=mode
        self.fine=nn.Sequential(nn.LayerNorm(192),nn.Linear(192,32),nn.GELU())
        self.coarse=nn.Sequential(nn.LayerNorm(768),nn.Linear(768,32),nn.GELU())
        self.v=nn.ModuleList([nn.Linear(32,16),nn.Linear(32,16)])
        self.u=nn.ModuleList([nn.Linear(32,16),nn.Linear(32,16)])
        self.w=nn.ModuleList([nn.Linear(16,1),nn.Linear(16,1)])
        dim=64 if mode=='gated_local' else 832
        self.head=nn.Sequential(nn.LayerNorm(dim),nn.Linear(dim,128),nn.GELU(),nn.Dropout(.15),nn.Linear(128,8))
    def pool(self,x,weight,i):
        if self.training:
            dropped=weight*(torch.rand_like(weight)>.1);weight=torch.where(dropped.sum(1,keepdim=True)>0,dropped,weight)
        if self.mode=='mean_global':a=weight/weight.sum(1,keepdim=True).clamp_min(1e-6)
        else:
            score=self.w[i](torch.tanh(self.v[i](x))*torch.sigmoid(self.u[i](x))).squeeze(-1)
            score=score+weight.clamp_min(1e-8).log();a=score.softmax(1)
        return F.normalize((a[:,:,None]*x).sum(1),dim=1)
    def forward(self,fine,coarse,fw,cw,global_x):
        f=self.pool(self.fine(fine),fw,0);c=self.pool(self.coarse(coarse),cw,1)
        x=torch.cat([f,c],dim=1) if self.mode=='gated_local' else torch.cat([f,c,global_x],dim=1)
        return self.head(x)


def main():
    start=time.time();prefix='attention_bag';torch.set_num_threads(4);torch.use_deterministic_algorithms(True)
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    index_path=NOTES/'.runtime/convnext_patch_token_index.json';index=json.loads(index_path.read_text())
    plan={'method':'grouped_image_patch_attention_assignment_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'source':'frozenConvNeXt192/768patches,optionalDINOBaseCLS;localheadtrainedonorganizertrain',
          'targets':'coldautomaticconsensuscoreand34human;notcurrentprocessedprogramlabels',
          'models':['mean_global','gated_global','gated_local'],'objectives':['CE','GCE(q=.7,CEwarmup5)'],
          'pooling':'Ilse-stylegatedtanh/sigmoidattention,softautomaticROIprior;notexactpaperreplicationorsegmentation',
          'projection':32,'epochs':20,'checkpoints':[10,20],'human_weight':8.,'batch_size':128,
          'optimizer':'AdamWlr.001,weightdecay.001,gradclip5;patchdropout.1',
          'validation':'fivegeometrysourcegroupfolds,queryexcludedfromtargets;frozenpatches,noqueryfitstatistics',
          'selection':'bestobjective/checkpointperarchitectureby34developmentbalancedrecall;thenbestarchitecture',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoverageat>=.9empiricalprecisionand>=10covered;tiesstricter',
          'guard':'primary>=24/34andmacro>=.75,orselectiveWilsonrank>=previous15/16rank+.02with>=16coveredandmacro>=.65,>=20matches;thresholdrequired',
          'acceptance':'learnedprobabilitythresholdandatleast1of3finalTRAINgraphsagree;humanprotected',
          'maximum_changes':40,'cap':'descendingneuralprobability,stablekeyorder',
          'evaluation_limit':'34anchorsreusedforselection/calibration,notfinalindependentAccuracy;Wilsonrankingnotriskcertificate;graphnotusedforcalibration',
          'reference':'https://proceedings.mlr.press/v80/ilse18a.html','private_review_or_archive_read':False,
          'patch_index_sha256':sha(index_path),'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904 and np.array_equal(index['keys'],keys)
    assert np.array_equal(index['source_sha256'],frame.source_sha256.to_numpy(str))
    arrays={}
    for name,record in index['files'].items():
        path=ROOT/record['path'];assert sha(path)==record['sha256'];arrays[name]=np.load(path,mmap_mode='r',allow_pickle=False)
    bf=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);assert np.array_equal(bf['keys'],keys)
    gx=torch.from_numpy(unit(bf['features'][:,:768]).astype(np.float32))
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups,folds=g.group.to_numpy(),g.fold.to_numpy();s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)))
    pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True;y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold
    eligible=a.automatic_anchor.to_numpy(bool)|human;weight=torch.from_numpy(np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.).astype(np.float32));weight[human]=8.
    target=torch.from_numpy(y).long();hm=torch.from_numpy(human)
    def batch_input(ids):return [torch.from_numpy(np.asarray(arrays[name][ids],dtype=np.float32)) for name in ['fine','coarse','fine_weight','coarse_weight']]+[gx[ids]]
    selected=[];probabilities=[];development=[];history=[]
    for arch_number,arch in enumerate(plan['models']):
        candidates=[]
        for objective in plan['objectives']:
            outputs={epoch:np.zeros((n,8),np.float32) for epoch in plan['checkpoints']}
            for fold in range(5):
                fit=np.flatnonzero((folds!=fold)&eligible);query=np.flatnonzero(folds==fold);assert not set(groups[fit])&set(groups[query])
                torch.manual_seed(20261108+fold+100*arch_number);rng=np.random.default_rng(20261108+fold+100*arch_number)
                model=BagHead(arch);optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.001)
                losses=[]
                for epoch in range(20):
                    model.train();order=rng.permutation(fit);total=0.;denom=0
                    for offset in range(0,len(order),128):
                        ids=order[offset:offset+128];logits=model(*batch_input(ids));ce=F.cross_entropy(logits,target[ids],reduction='none')
                        if objective=='GCE(q=.7,CEwarmup5)' and epoch>=5:
                            py=logits.softmax(1).gather(1,target[ids,None]).squeeze(1).clamp_min(1e-7);items=torch.where(hm[ids],ce,(1-py.pow(.7))/.7)
                        else:items=ce
                        loss=(items*weight[ids]).sum()/weight[ids].sum();assert torch.isfinite(loss)
                        optimizer.zero_grad(set_to_none=True);loss.backward();nn.utils.clip_grad_norm_(model.parameters(),5.);optimizer.step();total+=float(loss.detach())*len(ids);denom+=len(ids)
                    losses.append(total/denom)
                    if epoch+1 in outputs:
                        model.eval()
                        with torch.inference_mode():
                            for offset in range(0,len(query),128):
                                ids=query[offset:offset+128];outputs[epoch+1][ids]=model(*batch_input(ids)).softmax(1).numpy()
                history.append({'architecture':arch,'objective':objective,'fold':fold,'fit_rows':len(fit),'first_loss':losses[0],'last_loss':losses[-1]})
                print('Patch head:',arch,objective,'fold',fold+1,'seconds',round(time.time()-start),flush=True)
            for epoch,p in outputs.items():
                assert np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5);pred=p[pos].argmax(1)
                info={'architecture':arch,'objective':objective,'epoch':epoch,'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'anchor_matches':int((pred==gold).sum())}
                candidates.append((info,p));development.append(info)
        best,p=max(candidates,key=lambda v:v[0]['development_balanced_recall']);selected.append(best);probabilities.append(p)
        print('Selected patch architecture:',best,flush=True)
    winner=max(range(len(selected)),key=lambda i:selected[i]['development_balanced_recall']);chosen=selected[winner];p=probabilities[winner];primary=p.argmax(1);score=p.max(1);correct=primary[pos]==gold
    calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());good=int((correct&covered).sum())
        calibration.append({'threshold':t,'covered':count,'correct':good,'precision':good/count if count else None,'eligible':count>=10 and good/count>=.9,'wilson_ranking_score':wilson_lower(good,count)})
    feasible=[v for v in calibration if v['eligible']];cal=max(feasible,key=lambda v:(v['covered'],v['threshold'])) if feasible else None
    global_guard=chosen['anchor_matches']>=24 and chosen['development_balanced_recall']>=.75
    selective_guard=cal is not None and cal['covered']>=16 and cal['wilson_ranking_score']>=wilson_lower(15,16)+.02 and chosen['development_balanced_recall']>=.65 and chosen['anchor_matches']>=20
    accepted=cal is not None and (global_guard or selective_guard);threshold=cal['threshold'] if cal else float('inf')
    cf=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);tf=np.load(NOTES/'.runtime/convnext_texture_features.npz',allow_pickle=False)
    assert np.array_equal(cf['keys'],keys) and np.array_equal(tf['keys'],keys)
    base=unit(np.c_[unit(bf['features'][:,:768]),unit(bf['features'][:,2304:])]);base=unit(base-base.mean(0))
    features=[base,unit(np.c_[unit(cf['features']),unit(bf['features'][:,:768])]),unit(np.c_[unit(tf['covariance']),unit(bf['features'][:,:768])])]
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(x,p,pos,gold) for x in features])
    support=(graphs==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=(score>=threshold)&(support>=1)&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:40];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'robust_neural_assignment_snapshot',score[i],'uncalibrated_neural_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'neural_probability':score,'graph_support':support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=np.stack(probabilities),graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_attention_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,
                  primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,selected_knn_heads=[],qualified_heads=[chosen],
                  human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],
                  human_diagnostic_protocol=plan['validation'],confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},
                  accepted_by_development_guard=bool(accepted),proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()),new_vit_propagation_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],seconds=time.time()-start)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':main()

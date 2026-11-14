"""Sparse train-only anchor diffusion with genuinely fold-excluded graph fitting.

Motivation: a dense automatic-label prior can overwhelm the few reviewed sources.
Only34previously budgeted labels are used; no remaining-review targets are read.
"""
import time
import json
import datetime
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main():
    start=time.time();prefix='sparse_anchor_graph'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    assert n==1904
    hashes={}
    def take(path):
        hashes[str(path.relative_to(ROOT))]=sha(path)
        z=np.load(path,allow_pickle=False);lookup=dict(zip(z['keys'],range(len(z['keys']))))
        return z['features'][[lookup[k] for k in keys]].astype('float32')
    vit=unit(take(NOTES/'.runtime/waste_vit_train_features.npz')[:,:768])
    focus=unit(take(NOTES/'.runtime/object_focus_vit_features.npz')[:,:768])
    det=unit(take(NOTES/'.runtime/litter_object_features.npz')[:,:2688])
    dino=unit(take(OLD/'output/embeddings/stage2_repr_v1/dinov2_features.npz'))
    banks={'vit_dino':unit(np.c_[vit,dino]),'focus_dino':unit(np.c_[focus,dino]),
           'detector_vit_dino':unit(np.c_[det,vit,dino]),'vit_full_focus':unit(np.c_[vit,focus])}
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    folds,groups=g.fold.to_numpy(),g.group.to_numpy();idx=dict(zip(keys,range(n)))
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');pos=np.array([idx[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class])
    assert len(pos)==34 and seeds.assigned_class.value_counts().le(5).all()
    human=np.zeros(n,bool);human[pos]=True
    y=np.array([CLASSES.index(c) for c in a.automatic_class]);y[pos]=gold
    core=a.automatic_anchor.to_numpy(bool)&~human
    reliable=a.reliable_vote_count.to_numpy(float)
    configs=[{'automatic_per_class':k,'automatic_mass':w,'alpha':alpha} for k in [5,15] for w in [.02,.2] for alpha in [.95,.99]]
    plan={'method':'sparse_train_only_anchor_graph_v1','train_rows':n,'test_rows':0,'human_sources':34,'new_manual_labels':0,
          'targets':'only34existing participant seeds and fixed cold automatic-consensus anchors',
          'automatic_selection':'per-class reliable-vote priority,then within-class feature centrality;source-group unique;recomputed inside fit partition',
          'graph':'20nearestpositivecosine^4,symmetric,row-normalized;human clamped;automatic weak restart prior;1500maxiterations',
          'validation':'fivegeometrygroupfolds;build graph and select anchors ONLY onfit;query excluded then20neighbor interpolation',
          'configs':configs,'feature_banks':list(banks),'selection':'per-bank highest34anchor balancedrecall;then primary bank bysame metric',
          'gate':'primary agrees>=3/4banks;candidate cosine>=.65andnearest-otherclassmargin>=.01in>=2banks;protectpreviousrefinements',
          'maximum_changes':40,'cap_selection':'descending bank agreement,then mean human-anchor category margin;no review list',
          'guard':'primary>=20/34matchesandbalancedrecall>=.6333333333333333',
          'evaluation_limit':'development selection reuses34anchors;not independent final semantic Accuracy',
          'unselected_review_or_archive_read':False,'feature_sha256':hashes,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    similarities={};history=[];selections=[];predictions=[];near_list=[]
    with threadpool_limits(limits=4):
        for name,x in banks.items():
            sim=(x@x.T).astype('float32');similarities[name]=sim
            def setup(fit,query):
                sub=sim[np.ix_(fit,fit)].copy();np.fill_diagonal(sub,-np.inf)
                neighbors=np.argsort(-sub,axis=1,kind='stable')[:,:20]
                weights=np.maximum(np.take_along_axis(sub,neighbors,axis=1),0)**4
                adj=sparse.csr_matrix((weights.ravel(),(np.repeat(np.arange(len(fit)),20),neighbors.ravel())),shape=(len(fit),len(fit)))
                adj=(adj+adj.T)*.5;degree=np.asarray(adj.sum(1)).ravel();assert (degree>0).all()
                transition=sparse.diags(1/degree)@adj
                query_neighbors=np.argsort(-sim[np.ix_(query,fit)],axis=1,kind='stable')[:,:20] if len(query) else np.zeros((0,20),int)
                query_weights=np.maximum(np.take_along_axis(sim[np.ix_(query,fit)],query_neighbors,axis=1),0)**4 if len(query) else np.zeros((0,20))
                if len(query):query_weights/=np.maximum(query_weights.sum(1,keepdims=True),1e-12)
                rank={}
                for c in range(8):
                    candidates=np.flatnonzero(core[fit]&(y[fit]==c))
                    centrality=sim[np.ix_(fit[candidates],fit[candidates])].mean(1) if len(candidates) else np.empty(0)
                    order=np.lexsort((fit[candidates],-centrality,-reliable[fit[candidates]]))
                    used=set();selected=[]
                    for i in candidates[order]:
                        if groups[fit[i]] not in used:selected.append(i);used.add(groups[fit[i]])
                    rank[c]=np.array(selected,dtype=int)
                return transition,query_neighbors,query_weights,rank
            def propagate(fit,query,prepared,cfg):
                transition,qn,qw,rank=prepared;prior=np.zeros((len(fit),8),np.float64)
                for c in range(8):
                    selected=rank[c][:cfg['automatic_per_class']]
                    prior[selected,c]=cfg['automatic_mass']
                hp=human[fit];prior[hp]=np.eye(8)[y[fit][hp]]
                # Class-normalize restart mass so five human sources do not
                # intrinsically outweigh a class with one human source.
                prior/=np.maximum(prior.sum(0,keepdims=True),1e-12)
                state=prior.copy()
                for step in range(1500):
                    update=cfg['alpha']*(transition@state)+(1-cfg['alpha'])*prior
                    update[hp]=prior[hp];delta=float(np.abs(update-state).max());state=update
                    if delta<1e-7:break
                query_prob=(state[qn]*qw[:,:,None]).sum(1) if len(query) else np.empty((0,8))
                return state,query_prob
            prepared=[]
            for fold in range(5):
                fit=np.flatnonzero(folds!=fold);q=np.flatnonzero(folds[pos]==fold)
                assert not set(groups[fit])&set(groups[pos[q]])
                prepared.append((fit,q,setup(fit,pos[q])))
            best_score=-1;best=None
            for cfg in configs:
                hp=np.full(len(pos),-1)
                for fit,q,graph in prepared:
                    _,p=propagate(fit,pos[q],graph,cfg)
                    if len(q):hp[q]=p.argmax(1)
                assert (hp>=0).all();quality=float(balanced_accuracy_score(gold,hp));matches=int((hp==gold).sum())
                history.append({'bank':name,**cfg,'development_balanced_recall':quality,'anchor_matches':matches})
                if quality>best_score:best_score=quality;best=cfg;best_matches=matches
            fit=np.arange(n);state,_=propagate(fit,np.array([],int),setup(fit,np.array([],int)),best)
            predictions.append(state.argmax(1));selections.append({'bank':name,**best,'development_balanced_recall':best_score,'anchor_matches':best_matches})
            near_list.append(np.stack([sim[:,pos[gold==c]].max(1) if (gold==c).any() else np.full(n,-np.inf) for c in range(8)],axis=1))
            print('Sparse anchor graph family:',selections[-1],flush=True)
    winner=max(range(len(banks)),key=lambda i:selections[i]['development_balanced_recall']);chosen=selections[winner]
    pred=np.stack(predictions);primary=pred[winner];support=(pred==primary).sum(0);local=np.zeros(n,int);margins=[]
    for near in near_list:
        value=near[np.arange(n),primary].copy();other=near.copy();other[np.arange(n),primary]=-np.inf;margin=value-other.max(1)
        local+=(value>=.65)&(margin>=.01);margins.append(margin)
    margin=np.mean(margins,axis=0);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    protect=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    gate=(support>=3)&(local>=2)&~human&~protect&(primary!=current)
    proposed=int(gate.sum());rank=np.lexsort((np.arange(n),-margin,-support));allowed=rank[gate[rank]][:40]
    gate[:]=False;gate[allowed]=True
    accepted=chosen['anchor_matches']>=20 and chosen['development_balanced_recall']>=.6333333333333333
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'bounded_anchor_material_diffusion_snapshot',support[i]/4.,'multi_method_vote_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'bank_support':support,'local_support':local,'category_margin':margin,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,predictions=pred,accepted=gate)
    report=json.loads((NOTES/'json/robust_geometry_heads_frozen.json').read_text())
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  selected_heads=selections,primary_head=chosen,human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,
                  development_balanced_recall=chosen['development_balanced_recall'],human_diagnostic_protocol=plan['validation'],
                  validation_limit=plan['evaluation_limit'],plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),
                  accepted_by_development_guard=accepted,proposed_new_labels=proposed,changed_rows_vs_baseline=int(gate.sum()),
                  new_algorithmic_changes=int(gate.sum()),new_neural_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),
                  source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),
         'guard':plan['guard'],'primary_head_development_matches':chosen['anchor_matches'],'new_category_proposals':proposed,
         'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':main()

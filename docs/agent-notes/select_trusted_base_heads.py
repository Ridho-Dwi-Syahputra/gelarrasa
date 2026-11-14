"""Development-qualified experts rather than unweighted weak-bank vetoes.

No private review or archive targets are read. The34development sources are
reused for expert qualification; these diagnostics are not final accuracy.
"""
import json
import datetime
import numpy as np
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    prefix='base_trusted_primary';source_prefix='base_clean_propagation'
    source_file=NOTES/f'json/{source_prefix}_frozen.json';prior=json.loads(source_file.read_text())
    assert all(sha(ROOT/p)==h for p,h in prior['protected_sha256'].items())
    heads=prior['selected_knn_heads'];winner=max(range(len(heads)),key=lambda i:heads[i]['development_balanced_recall'])
    primary_head=heads[winner];best=primary_head['development_balanced_recall']
    trusted=[i for i,h in enumerate(heads) if h['development_balanced_recall']>=.6333333333333333 and h['development_balanced_recall']>=best-.1]
    assert winner in trusted
    graph_names=[g['bank'] for g in prior['graph_history']]
    trusted_graph=[i for i,b in enumerate(graph_names) if b in [heads[k]['bank'] for k in trusted]]
    plan={'method':'development_qualified_DINOv2Base_expert_consensus_v1','source_frozen_sha256':sha(source_file),
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'primary_head':primary_head,
          'qualification':'bank development balancedrecall>=.6333333333333333andwithin.1ofbest;atleast2qualifiedbanksand2correspondinggraphs',
          'trusted_banks':[heads[i]['bank'] for i in trusted],'trusted_graphs':[graph_names[i] for i in trusted_graph],
          'acceptance':'primary category agrees with ALLqualifiedbanks and ALLcorrespondinggraphs;candidate human-anchor cosine>=.65andclassmargin>=.01inALLqualifiedbanks;humanprotected',
          'previous_auto_refinements':'maychangeonlyifALLqualifiedbanks additionally havecosine>=.75andcategorymargin>=.03;no per-IDexceptions',
          'maximum_changes':40,'cap_selection':'descending minimum qualified-bank human-anchor margin,then minimum cosine;stablekeyorder tie',
          'guard':'primary>=20/34matchesandbalancedrecall>=.6333333333333333;atleast2qualifiedbanksand2graphs',
          'development_gate_limit':'graph includesallTRAINseeds;its full-graph gate is NOT evaluated asindependent validation;reported primary score isgroupedkNNdevelopmentonly',
          'private_review_or_archive_used':False,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str)
    votes=np.load(NOTES/f'.runtime/{source_prefix}_outputs.npz',allow_pickle=False);assert np.array_equal(votes['keys'],keys)
    pred=votes['knn_predictions'];graphs=votes['graph_predictions'];primary=pred[winner]
    human=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');idx=dict(zip(keys,range(len(keys))))
    pos=np.array([idx[k] for k in human.key]);gold=np.array([CLASSES.index(c) for c in human.assigned_class])
    matches=int((primary[pos]==gold).sum())
    assert matches==int(pd.read_csv(NOTES/f'table-csv/{source_prefix}_development.csv').query('bank==@primary_head["bank"]')
                           .sort_values('development_balanced_recall',ascending=False).iloc[0].anchor_matches)
    base=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False)
    vit=np.load(NOTES/'.runtime/waste_vit_train_features.npz',allow_pickle=False)
    det=np.load(NOTES/'.runtime/litter_object_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [base,vit,det])
    cls=unit(base['features'][:,:768]);vcls=unit(vit['features'][:,:768]);df=unit(det['features'][:,:2688])
    banks={'base_cls':cls,'base_local':unit(np.c_[cls,unit(base['features'][:,2304:])]),
           'base_vit':unit(np.c_[cls,vcls]),'base_vit_detector':unit(np.c_[cls,vcls,df])}
    cosine=[];margins=[]
    for bank in plan['trusted_banks']:
        sim=banks[bank]@banks[bank][pos].T
        near=np.stack([sim[:,gold==c].max(1) if (gold==c).any() else np.full(len(keys),-np.inf) for c in range(8)],axis=1)
        value=near[np.arange(len(keys)),primary].copy();near[np.arange(len(keys)),primary]=-np.inf
        cosine.append(value);margins.append(value-near.max(1))
    cosine=np.min(cosine,axis=0);margin=np.min(margins,axis=0)
    bank_support=(pred[trusted]==primary).sum(0);graph_support=(graphs[trusted_graph]==primary).sum(0)
    preserve=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    previous_allowed=(cosine>=.75)&(margin>=.03)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    gate=(bank_support==len(trusted))&(graph_support==len(trusted_graph))&(cosine>=.65)&(margin>=.01)
    gate&=~frame.human_reviewed.to_numpy(bool)&(~preserve|previous_allowed)&(primary!=current)
    proposals=int(gate.sum());rank=np.lexsort((np.arange(len(keys)),-cosine,-margin));allowed=rank[gate[rank]][:40]
    gate[:]=False;gate[allowed]=True
    accepted=matches>=20 and best>=.6333333333333333 and len(trusted)>=2 and len(trusted_graph)>=2
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',bank_support[i]/len(trusted),'multi_method_vote_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'qualified_bank_support':bank_support,'qualified_graph_support':graph_support,
                  'minimum_anchor_cosine':cosine,'minimum_anchor_class_margin':margin,'previous_auto_eligible':previous_allowed,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report=dict(prior)
    report.update(status='frozen_before_private_qualified_expert_audit',method=plan['method'],prediction_sha256=sha(dest),
                  primary_head=primary_head,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),
                  human_leave_source_group_out_matches=matches,development_balanced_recall=best,
                  accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),
                  new_vit_propagation_changes=int(gate.sum()),qualified_expert_banks=plan['trusted_banks'],
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),
                  source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),
                  validation_limit=plan['development_gate_limit'])
    assert all(sha(ROOT/p)==h for p,h in prior['protected_sha256'].items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),
         'guard':plan['guard'],'primary_head_development_matches':matches,'new_category_proposals':proposals,'accepted_changes':int(gate.sum()),
         'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','qualified_expert_banks','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':main()

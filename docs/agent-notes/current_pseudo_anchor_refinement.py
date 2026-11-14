"""Transfer from previously learned train pseudo-anchors, not cold labels only.

Pseudo-anchors depend on the34human sources already used upstream. Out-of-source
agreement here measures PROGRAM consistency, never independent human Accuracy.
No full-review or archived visual labels are read, filtered or fitted.
"""
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main(round_number=1):
    prefix='current_pseudo_anchor' if round_number==1 else f'current_pseudo_anchor_r{round_number}'
    protocol_path=NOTES/'json/current_pseudo_iteration_protocol.json'
    protocol=json.loads(protocol_path.read_text()) if round_number>1 else None
    prior_round=None
    if protocol:
        assert 1<round_number<=protocol['maximum_total_rounds']
        active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())
        expected='current_pseudo_anchor' if round_number==2 else f'current_pseudo_anchor_r{round_number-1}'
        assert active['source_frozen_prefix']==expected
        prior_round=json.loads((NOTES/f'json/{expected}_frozen.json').read_text())
        assert prior_round['proposed_new_labels']>=protocol['stop_when_proposals_below']
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'current_program_pseudo_anchor_neighborhood_transfer_v1','human_sources':34,'new_manual_labels':0,'test_rows':0,
          'iteration':round_number,'iteration_protocol_sha256':sha(protocol_path) if protocol else None,
          'targets':'currentmanifestcategoriesONLYforexistinghumanseedsandnonhistoricalprogramrefinements;vote-share>=.75,neighbor-share>=.9,neuralprobability>=.85,other>=.8;allremainpseudoexcept34human',
          'bank_fixed_before_fit':True,'banks':['base_cls','base_local','base_center','cnn_base','texture_base'],
          'knn':{'neighbors':5,'temperature':.07,'human_weight_multiplier':4.,'class_count_weight':'inverse square root ofanchorcount','exclude_entire_query_source_group':True},
          'gate':'atleast3/5banks agree,includingCNNortextureandoneDINO;minimumagreeingneighborprob>=.8;classcosinemargin>=.01in>=2banks;nearestsupportcos>=.65(.4centered)in>=2banks',
          'previous_pseudo_anchors':'changeonlyall5banksagreeandmeanprob>=.95;humanalwaysprotected',
          'guard':'>=80anchorrows,all8classes;outofsourcePROGRAMpseudoagreement>=.6andmacro>=.55;allpredictedclasses>=10,maxclass<=40percent',
          'maximum_changes':40,'cap':'descending agreeing-bank support,then mean classprobability,stablekeyorder',
          'evaluation_limit':'pseudo-anchorconsistencyonly;allpseudolabelsalreadydependon34humanupstream;NOindependenthumanholdoutorAccuracyreported',
          'independent_human_validation_performed':False,'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    groups=frame.parent_group.to_numpy(str);human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    assert human.sum()==34 and frame[human].assigned_class.value_counts().le(5).all()
    sources={}
    def take(name):
        path=NOTES/f'.runtime/{name}.npz';z=np.load(path,allow_pickle=False);assert np.array_equal(z['keys'],keys)
        sources[name]=sha(path);return z
    b=take('dinov2_base_train_features');c=take('convnext_train_features');t=take('convnext_texture_features')
    cls=unit(b['features'][:,:768]);local=unit(np.c_[cls,unit(b['features'][:,2304:])])
    banks={'base_cls':cls,'base_local':local,'base_center':unit(local-local.mean(0)),
           'cnn_base':unit(np.c_[unit(c['features']),cls]),'texture_base':unit(np.c_[unit(t['covariance']),cls])}
    old_source=frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    score=frame.category_score.to_numpy(float);kinds=frame.category_score_kind.to_numpy(str)
    threshold=np.full(n,.8);threshold[np.isin(kinds,['multi_method_vote_share','minimum_family_vote_share'])]=.75
    threshold[kinds=='weighted_neighbor_share']=.9;threshold[kinds=='uncalibrated_neural_probability']=.85
    anchor=human|(~old_source&(score>=threshold));pos=np.flatnonzero(anchor)
    labels=current[pos];class_counts=np.bincount(labels,minlength=8)
    base_weight=np.clip(frame.category_score.to_numpy(float)[pos],.1,1.)/np.sqrt(np.maximum(class_counts[labels],1));base_weight[human[pos]]*=4.
    predictions=[];probabilities=[];local_support=np.zeros(n,int);margin_support=[]
    with threadpool_limits(limits=4):
        for name,x in banks.items():
            sim=x@x[pos].T;sim[groups[:,None]==groups[pos][None,:]]=-np.inf
            nn=np.argsort(-sim,axis=1,kind='stable')[:,:5];value=np.take_along_axis(sim,nn,axis=1);assert np.isfinite(value).all()
            weight=np.exp((value-value.max(1,keepdims=True))/.07)*base_weight[nn]
            p=np.stack([(weight*(labels[nn]==k)).sum(1) for k in range(8)],axis=1);p/=p.sum(1,keepdims=True)
            predictions.append(p.argmax(1));probabilities.append(p)
            near=np.stack([sim[:,labels==k].max(1) if (labels==k).any() else np.full(n,-np.inf) for k in range(8)],axis=1);margin_support.append(near)
    pred=np.stack(predictions);p=np.stack(probabilities);votes=np.stack([(pred==k).sum(0) for k in range(8)],axis=1)
    candidate=votes.argmax(1);support=votes.max(1);agree=pred==candidate;target_probability=p[:,np.arange(n),candidate]
    min_probability=np.where(agree,target_probability,1.).min(0);mean_probability=target_probability.mean(0)
    local=np.zeros(n,int);cosine=np.zeros(n,int)
    for name,near in zip(banks,margin_support):
        values=near[np.arange(n),candidate].copy();others=near.copy();others[np.arange(n),candidate]=-np.inf
        local+=(values-others.max(1)>=.01);cosine+=(values>=(.4 if name=='base_center' else .65))
    gate=(support>=3)&agree[:3].any(0)&agree[3:].any(0)&(min_probability>=.8)&(local>=2)&(cosine>=2)&~human&(candidate!=current)
    gate&=~anchor|((support==5)&(mean_probability>=.95))
    pseudo=anchor&~human;consistency=float((candidate[pseudo]==current[pseudo]).mean()) if pseudo.any() else 0.
    recalls=[float((candidate[pseudo&(current==k)]==k).mean()) for k in range(8) if (pseudo&(current==k)).any()];macro=float(np.mean(recalls)) if recalls else 0.
    counts=np.bincount(candidate,minlength=8);accepted=len(pos)>=80 and (class_counts>0).all() and consistency>=.6 and macro>=.55 and (counts>=10).all() and counts.max()<=.4*n
    if prior_round:
        accepted=accepted and consistency>=prior_round['program_pseudo_outsource_agreement']-.02 and macro>=prior_round['program_pseudo_outsource_macro_agreement']-.02
    proposals=int(gate.sum());rank=np.lexsort((np.arange(n),-mean_probability,-support));allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[candidate[i]],'vit_knn_label_propagation_snapshot',min_probability[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[candidate],'feature_votes':support,'minimum_agreeing_probability':min_probability,'class_margin_banks':local,'cosine_support_banks':cosine,'is_pseudo_or_human_anchor':anchor,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    pd.DataFrame({'key':keys[pos],'assigned_class':np.array(CLASSES)[labels],'human':human[pos],'program_score':frame.category_score.to_numpy()[pos],
                  'source':frame.label_source.to_numpy()[pos],'outsource_predicted_class':np.array(CLASSES)[candidate[pos]]}).to_csv(NOTES/f'table-csv/{prefix}_training_pool.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,predictions=pred,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text())
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_current_pseudo_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head={'method':'five_bank_current_pseudo_anchor_knn','k':5,'human_weight':4.},primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),
                  selected_heads=[{'bank':name} for name in banks],selected_knn_heads=[],qualified_heads=[{'bank':name} for name in banks],
                  human_leave_source_group_out_matches=None,human_holdout_rows=0,development_balanced_recall=None,
                  human_diagnostic_protocol='notperformed;currentpseudolabelsalreadydependonall34human',confidence_calibration=None,
                  iteration=round_number,anchor_pool_rows=len(pos),anchor_pool_class_counts=dict(zip(CLASSES,class_counts.tolist())),program_pseudo_outsource_agreement=consistency,
                  program_pseudo_outsource_macro_agreement=macro,these_are_human_accuracy=False,
                  feature_sha256=sources,accepted_by_development_guard=bool(accepted),guard_protocol_type='programconsistencyproxy_notindependentvalidation',
                  proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),
                  source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['anchor_pool_rows','anchor_pool_class_counts','program_pseudo_outsource_agreement','program_pseudo_outsource_macro_agreement',
                                        'proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--round',type=int,default=1);main(parser.parse_args().round)

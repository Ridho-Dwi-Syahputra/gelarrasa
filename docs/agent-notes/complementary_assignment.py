"""Development-selected complementary train assignment, no private-review targets."""
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def main():
    prefix='complementary_assignment'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'complementary_grouped_neural_neighbor_assignment_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'components':'best34developmentqualifiedneuralbankandbestnormalizedneighborbank;bothsource-groupoutof-fold',
          'neural_weights':[0.,.25,.5,.75,1.],'selection':'highest34anchor balancedrecall;tiesmorematches;tiespreferweightclosestto.5',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoveragewith>=10coveredand>=.9empiricalselectiveprecision;tiesstricter',
          'acceptance':'learnedprobabilitythreshold,atleast1of2train-onlynormalizedgraphsagrees;allhumanprotected',
          'previous_auto_refinements':'samegloballylearnedthreshold;notpermanentgroundtruth;no perimageexceptions',
          'guard':'strictlyimprovebalancedrecallover.725,>=24/34matches,nonendpointmixture,andthresholdexists',
          'maximum_changes':40,'cap':'descending ensembleprobability,stablekeyorder',
          'evaluation_limit':'34anchorsreusedforcomponentselection,mixingandcalibration;notindependentfinalAccuracy;graphnotusedincalibrationevaluation',
          'unselected_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    kp=NOTES/'.runtime/normalized_base_outputs.npz';np_path=NOTES/'.runtime/base_neural_outputs.npz'
    k=np.load(kp,allow_pickle=False);neural=np.load(np_path,allow_pickle=False);assert np.array_equal(k['keys'],keys) and np.array_equal(neural['keys'],keys)
    ki=json.loads((NOTES/'json/normalized_base_frozen.json').read_text());ni=json.loads((NOTES/'json/base_neural_frozen.json').read_text())
    # normalized source selected_heads explicitly records the two current banks;
    # its inherited selected_knn_heads belongs to an earlier stage and is ignored.
    kheads=ki['selected_heads'];nheads=ni['selected_heads']
    kw=max(range(len(kheads)),key=lambda i:kheads[i]['development_balanced_recall']);nw=max(range(len(nheads)),key=lambda i:nheads[i]['development_balanced_recall'])
    pk,pn=k['probabilities'][kw],neural['probabilities'][nw]
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[x] for x in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    history=[];probabilities=[]
    for weight in plan['neural_weights']:
        p=(1-weight)*pk+weight*pn;pred=p[pos].argmax(1);quality=float(balanced_accuracy_score(gold,pred));matches=int((pred==gold).sum())
        history.append({'neural_weight':weight,'development_balanced_recall':quality,'anchor_matches':matches});probabilities.append(p)
    winner=max(range(len(history)),key=lambda i:(history[i]['development_balanced_recall'],history[i]['anchor_matches'],-abs(history[i]['neural_weight']-.5)))
    chosen=history[winner];p=probabilities[winner];primary=p.argmax(1);score=p.max(1);correct=primary[pos]==gold;calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());matches=int((covered&correct).sum())
        calibration.append({'threshold':t,'covered':count,'correct':matches,'precision':matches/count if count else None,'eligible':count>=10 and matches/count>=.9})
    eligible=[r for r in calibration if r['eligible']];cal=max(eligible,key=lambda r:(r['covered'],r['threshold'])) if eligible else None
    threshold=cal['threshold'] if cal else float('inf');human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    graph_support=(k['graphs']==primary).sum(0);gate=(score>=threshold)&(graph_support>=1)&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    accepted=cal is not None and chosen['development_balanced_recall']>.725 and chosen['anchor_matches']>=24 and chosen['neural_weight'] not in [0.,1.]
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'robust_neural_assignment_snapshot',score[i],'uncalibrated_neural_probability',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'ensemble_probability':score,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,accepted=gate)
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key');report=dict(ki)
    report.update(status='frozen_before_private_complementarity_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[kheads[kw],nheads[nw]],selected_knn_heads=[],
                  component_output_sha256={'knn':sha(kp),'neural':sha(np_path)},human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=chosen['development_balanced_recall'],
                  confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,
                  changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()),new_vit_propagation_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),
                  source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':main()

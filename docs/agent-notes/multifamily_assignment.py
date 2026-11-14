"""Three image-only train learners; bounded development-selected convex pooling."""
import json
import datetime
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump


def main():
    prefix='multifamily_assignment'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    weights=[(a/4,b/4,(4-a-b)/4) for a in range(5) for b in range(5-a)]
    plan={'method':'three_family_grouped_assignment_v1','components':['normalizedDINOBasekNN','DINOBaseGCEneural','ConvNeXtfeaturekNNmean'],
          'weights':weights,'selection':'maximum34anchorbalancedrecall;tie highermatches;tieprefermorecomponents',
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,'threshold_grid':[.7,.8,.9,.95,.98,.995],
          'threshold_selection':'maximum34developmentcoveragewith>=10coveredand>=.9empiricalselectiveprecision;tiesstricter',
          'acceptance':'learnedprobabilitythreshold,atleast1normalizedTRAINgraphandagreementatleast1CNNTRAINgraph;humanprotected',
          'guard':'balancedrecall>.725+1e-12and>=24/34matches,thresholdexists,atleast2nonzerofamilies',
          'maximum_changes':40,'cap':'descendingensembleprobability,stablekeyorder','previous_auto_refinements':'sameglobalthreshold,noperIDexception',
          'evaluation_limit':'34developmentanchorsreusedforcomponentselection,mixing,calibration;notindependentfinalAccuracy;graphnotusedincalibrationevaluation',
          'unselected_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    paths=[NOTES/'.runtime/normalized_base_outputs.npz',NOTES/'.runtime/base_neural_outputs.npz',NOTES/'.runtime/cnn_clean_propagation_outputs.npz']
    sources=[np.load(p,allow_pickle=False) for p in paths];assert all(np.array_equal(z['keys'],keys) for z in sources)
    ki=json.loads((NOTES/'json/normalized_base_frozen.json').read_text());ni=json.loads((NOTES/'json/base_neural_frozen.json').read_text())
    kw=max(range(len(ki['selected_heads'])),key=lambda i:ki['selected_heads'][i]['development_balanced_recall'])
    nw=max(range(len(ni['selected_heads'])),key=lambda i:ni['selected_heads'][i]['development_balanced_recall'])
    components=[sources[0]['probabilities'][kw],sources[1]['probabilities'][nw],sources[2]['probabilities'].mean(0)]
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    assert len(pos)==34 and s.assigned_class.value_counts().le(5).all()
    history=[]
    for w in weights:
        p=sum(v*x for v,x in zip(w,components));pred=p[pos].argmax(1)
        history.append({'weights':w,'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'anchor_matches':int((pred==gold).sum())})
    winner=max(range(len(history)),key=lambda i:(history[i]['development_balanced_recall'],history[i]['anchor_matches'],sum(v>0 for v in weights[i])))
    chosen=history[winner];p=sum(v*x for v,x in zip(weights[winner],components));primary=p.argmax(1);score=p.max(1);correct=primary[pos]==gold
    calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());matches=int((correct&covered).sum())
        calibration.append({'threshold':t,'covered':count,'correct':matches,'precision':matches/count if count else None,'eligible':count>=10 and matches/count>=.9})
    eligible=[r for r in calibration if r['eligible']];cal=max(eligible,key=lambda r:(r['covered'],r['threshold'])) if eligible else None
    threshold=cal['threshold'] if cal else float('inf');support0=(sources[0]['graphs']==primary).sum(0);support1=(sources[2]['graph_predictions']==primary).sum(0)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=(score>=threshold)&(support0>=1)&(support1>=1)&~frame.human_reviewed.to_numpy(bool)&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');allowed=rank[gate[rank]][:40];gate[:]=False;gate[allowed]=True
    accepted=cal is not None and chosen['development_balanced_recall']>.725+1e-12 and chosen['anchor_matches']>=24 and sum(v>0 for v in weights[winner])>=2
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    has_neural=weights[winner][1]>0
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[
        CLASSES[primary[i]],'robust_neural_assignment_snapshot' if has_neural else 'vit_knn_label_propagation_snapshot',score[i],
        'uncalibrated_neural_probability' if has_neural else 'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'ensemble_probability':score,'normalized_graph_support':support0,'cnn_graph_support':support1,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,accepted=gate)
    report=dict(ki);initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_multifamily_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,
                  primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'family':name,'weight':w} for name,w in zip(plan['components'],weights[winner]) if w>0],selected_knn_heads=[],
                  qualified_heads=[{'family':name,'weight':w} for name,w in zip(plan['components'],weights[winner]) if w>0],
                  component_output_sha256={p.name:sha(p) for p in paths},human_leave_source_group_out_matches=chosen['anchor_matches'],development_balanced_recall=chosen['development_balanced_recall'],
                  confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,
                  proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()),new_vit_propagation_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],
         'accepted_changes':int(gate.sum()),'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard']},indent=2),flush=True)


if __name__=='__main__':main()

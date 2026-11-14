"""Fixed coarse convex ablation on source-group OOF image-only learners.

Selection sees only34already-budgeted development sources. No full review or
archive labels are read. All graph fitting is train-only, not independent CV.
"""
import datetime
import json
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    prefix='object_complementarity'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    weights=[(a/4,b/4,c/4,(4-a-b-c)/4) for a in range(5) for b in range(5-a) for c in range(5-a-b)]
    names=['previousDINO_CNN24','object_zoom22','source_contrastive22','semantic_color24']
    plan={'method':'object_zoom_color_contrastive_grouped_complementarity_v1','components':names,'weights':weights,
          'human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'selection':'maximum34source-groupOOFmacrorecall,thenmatches,thenmorecomponents;fixedquartergrid35candidates',
          'threshold_grid':[.7,.8,.9,.95,.98,.995],'threshold_selection':'maxcoverage>=10at>=.9empiricalprecision;tiesstricter',
          'graph':'finalTRAINBaseROI,objectZoomROI,CNNBase;clamp34human;notusedincalibration',
          'acceptance':'learnedthresholdand>=2/3finalTRAINgraphsagree;protecthuman;noindividualexceptions',
          'guard':'>=25/34andmacro>.75+1e-12andthresholdexistsand>=2components',
          'maximum_changes':40,'cap':'descendingensembleprobability,stablekeyorder',
          'evaluation_limit':'34reuseddevelopmentnotindependentAccuracy;allcomponentsouterquerysourceexcludedbutselectionismultiplecomparison',
          'private_review_or_archive_read':False,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    paths=[NOTES/f'.runtime/{name}_outputs.npz' for name in ['multifamily_assignment','object_zoom_neighbors','source_contrastive','webinar_hybrid_neighbors']]
    sources=[np.load(p,allow_pickle=False) for p in paths];assert all(np.array_equal(z['keys'],keys) for z in sources)
    configs=[json.loads((NOTES/f'json/{name}_frozen.json').read_text()) for name in ['object_zoom_neighbors','source_contrastive','webinar_hybrid_neighbors']]
    indices=[max(range(len(c['selected_heads'])),key=lambda i:c['selected_heads'][i]['development_balanced_recall']) for c in configs]
    components=[sources[0]['probabilities']]+[z['probabilities'][i] for z,i in zip(sources[1:],indices)]
    assert all(p.shape==(n,8) and np.isfinite(p).all() and np.allclose(p.sum(1),1,atol=1e-5) for p in components)
    history=[]
    for w in weights:
        p=sum(v*x for v,x in zip(w,components));pred=p[pos].argmax(1)
        history.append({'weights':w,'development_balanced_recall':float(balanced_accuracy_score(gold,pred)),'anchor_matches':int((pred==gold).sum())})
    winner=max(range(len(history)),key=lambda i:(history[i]['development_balanced_recall'],history[i]['anchor_matches'],sum(v>0 for v in weights[i])))
    chosen=history[winner];p=sum(v*x for v,x in zip(weights[winner],components));primary=p.argmax(1);score=p.max(1)
    calibration=[]
    for t in plan['threshold_grid']:
        covered=score[pos]>=t;count=int(covered.sum());good=int((covered&(primary[pos]==gold)).sum());calibration.append({'threshold':t,'covered':count,'correct':good,'precision':good/count if count else None,'eligible':count>=10 and good/count>=.9})
    feasible=[r for r in calibration if r['eligible']];cal=max(feasible,key=lambda r:(r['covered'],r['threshold'])) if feasible else None
    accepted=bool(chosen['anchor_matches']>=25 and chosen['development_balanced_recall']>.75+1e-12 and cal is not None and sum(v>0 for v in weights[winner])>=2)
    b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);z=np.load(NOTES/'.runtime/dinov2_object_zoom_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [b,z,c])
    regional=unit(np.c_[unit(b['features'][:,:768]),unit(b['features'][:,2304:])]);zoom=unit(np.c_[unit(z['features'][:,:768]),unit(z['features'][:,2304:])]);cnn=unit(np.c_[unit(c['features']),unit(b['features'][:,:768])])
    with threadpool_limits(limits=3):graphs=np.stack([graph_prediction(unit(x-x.mean(0)),p,pos,gold) for x in [regional,zoom,cnn]])
    support=(graphs==primary).sum(0);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);human=frame.human_reviewed.to_numpy(bool)
    gate=~human&(primary!=current)&(score>=(cal['threshold'] if cal else np.inf))&(support>=2)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:40];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key');has_neural=weights[winner][2]>0
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'robust_neural_assignment_snapshot' if has_neural else 'vit_knn_label_propagation_snapshot',score[i],'uncalibrated_neural_probability' if has_neural else 'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'ensemble_probability':score,'graph_support':support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=p,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    selected=[{'family':name,'weight':w} for name,w in zip(names,weights[winner]) if w>0]
    report.update(status='frozen_before_private_object_complementarity_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=selected,qualified_heads=selected,selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],
        confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=int(gate.sum()) if has_neural else 0,new_vit_propagation_changes=int(gate.sum()) if not has_neural else 0,
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),validation_limit=plan['evaluation_limit'],component_output_sha256={p.name:sha(p) for p in paths})
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['guard'],'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','confidence_calibration','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline']},indent=2),flush=True)


if __name__=='__main__':main()

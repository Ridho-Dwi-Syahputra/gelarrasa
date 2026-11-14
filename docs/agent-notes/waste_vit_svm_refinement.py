"""Grouped non-VLM image-feature SVMs; only budgeted and automatic train targets.

The full participant review and visual archive are never read by this experiment.
Development anchors are reused for selection, so this is not final validation.
"""
import time
import json
import datetime
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main(use_base=False):
    start=time.time(); prefix='base_svm' if use_base else 'vit_svm'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True)
    keys=frame.parent_key.to_numpy(str);assert len(keys)==1904
    def take(path):
        z=np.load(path,allow_pickle=False); lookup=dict(zip(z['keys'],range(len(z['keys']))))
        return z['features'][[lookup[k] for k in keys]].astype('float32')
    vitpath=NOTES/'.runtime/waste_vit_train_features.npz'
    detpath=NOTES/'.runtime/litter_object_features.npz'
    assert sha(vitpath)==json.loads((NOTES/'json/waste_vit_feature_extraction.json').read_text())['feature_sha256']
    assert sha(detpath)==json.loads((NOTES/'json/litter_object_extraction.json').read_text())['feature_sha256']
    f=take(vitpath);d=unit(take(OLD/'output/embeddings/stage2_repr_v1/dinov2_features.npz'))
    det=unit(take(detpath)[:,:2688]); cls=unit(f[:,:768]); texture=unit(np.c_[unit(f[:,768:1536]),unit(f[:,1536:])])
    banks={'vit_cls':cls,'vit_context':unit(np.c_[cls,texture]),'vit_dino':unit(np.c_[cls,d]),'vit_dino_detector':unit(np.c_[cls,d,det])}
    if use_base:
        bp=NOTES/'.runtime/dinov2_base_train_features.npz'
        assert sha(bp)==json.loads((NOTES/'json/dinov2_base_feature_extraction.json').read_text())['feature_sha256']
        bf=take(bp);bc=unit(bf[:,:768]);local=unit(bf[:,2304:])
        banks={'base_cls':bc,'base_local':unit(np.c_[bc,local]),'base_vit':unit(np.c_[bc,cls]),'base_material':unit(np.c_[bc,unit(bf[:,768:1536]),local])}
    auto=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys]
    gf=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys]
    groups,folds=gf.group.to_numpy(),gf.fold.to_numpy()
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');idx=dict(zip(keys,range(len(keys))))
    pos=np.array([idx[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class])
    assert len(pos)==34 and seeds.assigned_class.value_counts().le(5).all()
    human=np.zeros(len(keys),bool);human[pos]=True
    y=np.array([CLASSES.index(c) for c in auto.automatic_class]);y[pos]=gold
    eligible=auto.automatic_anchor.to_numpy(bool)|human
    base=np.clip(auto.reliable_vote_count.to_numpy(float)/4,.25,1)
    configs=[{'kernel':kind,'gamma':g,'C':c,'human_weight':w}
             for kind,g in [('linear',None),('rbf',4.),('rbf',16.)] for c in [1.,10.] for w in [8.,32.]]
    minimum_matches=23 if use_base else 20;minimum_balanced=.725 if use_base else .6333333333333333
    plan={'method':'train_only_dinov2base_filtered_svm_consensus_v1' if use_base else 'train_only_waste_vit_grouped_svm_consensus_v1','train_rows':1904,'test_rows':0,
          'human_sources':34,'new_manual_labels':0,'targets':'cold automatic consensus core plus34existing participant seeds',
          'banks':{k:x.shape[1] for k,x in banks.items()},'configs':configs,'group_folds':5,
          'selection':'best balanced recall on34group-held-out development anchors per bank,then best bank asprimary;ties retain config order',
          'gate':'primary agrees with>=3/4SVMbanks;candidate has human-anchor cosine>=.65 and margin>=.01in>=2banks;protect previous refinements',
          'guard':f'primary matches>={minimum_matches}/34andbalancedrecall>={minimum_balanced};<=50newlabels',
          'automatic_filter':'foldfitonly;exclude core anchor with contradictory nearest human class atcosine>=.65andmargin>=.02;humanalwaysretained' if use_base else None,
          'evaluation_limit':'anchors reused for model development;not independent final accuracy',
          'unselected_review_or_archive_read':False,'protected_sha256':protected,
          'vit_feature_sha256':sha(vitpath),'detector_feature_sha256':sha(detpath),'base_feature_sha256':sha(bp) if use_base else None,
          'group_manifest_sha256':sha(NOTES/'table-csv/geometry_train_groups.csv'),'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    history=[];selections=[];predictions=[];holdouts=[];similarities=[]
    with threadpool_limits(limits=4):
        for name,x in banks.items():
            sim=(x@x.T).astype('float32');similarities.append(sim)
            def fit_indices(fold):
                fit=np.flatnonzero((folds!=fold)&eligible)
                if use_base:
                    anchors=fit[human[fit]]
                    near=np.stack([sim[np.ix_(fit,anchors[y[anchors]==c])].max(1) if (y[anchors]==c).any() else np.full(len(fit),-np.inf) for c in range(8)],axis=1)
                    cat=near.argmax(1);val=near[np.arange(len(fit)),cat].copy();near[np.arange(len(fit)),cat]=-np.inf
                    remove=(cat!=y[fit])&(val>=.65)&(val-near.max(1)>=.02)&~human[fit];fit=fit[~remove]
                return fit
            best_score=-1;best=None;best_h=None
            kernels={g:np.exp(-2*g*np.maximum(1-sim,0)) for g in [4.,16.]}
            for cfg in configs:
                kernel=sim if cfg['kernel']=='linear' else kernels[cfg['gamma']]
                hp=np.full(len(pos),-1)
                weights=base.copy();weights[human]=cfg['human_weight']
                for fold in range(5):
                    fit=fit_indices(fold);q=np.flatnonzero(folds[pos]==fold)
                    assert not set(groups[fit])&set(groups[pos[q]])
                    if not len(q):continue
                    model=SVC(kernel='precomputed',C=cfg['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                    model.fit(kernel[np.ix_(fit,fit)],y[fit],sample_weight=weights[fit])
                    hp[q]=model.predict(kernel[np.ix_(pos[q],fit)])
                assert (hp>=0).all()
                quality=float(balanced_accuracy_score(gold,hp));matches=int((hp==gold).sum())
                history.append({'bank':name,**cfg,'development_balanced_recall':quality,'anchor_matches':matches})
                if quality>best_score:best_score=quality;best=cfg;best_h=hp.copy()
            kernel=sim if best['kernel']=='linear' else kernels[best['gamma']]
            weights=base.copy();weights[human]=best['human_weight'];pred=np.full(len(keys),-1)
            for fold in range(5):
                fit=fit_indices(fold);q=np.flatnonzero(folds==fold)
                assert not set(groups[fit])&set(groups[q])
                model=SVC(kernel='precomputed',C=best['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                model.fit(kernel[np.ix_(fit,fit)],y[fit],sample_weight=weights[fit]);pred[q]=model.predict(kernel[np.ix_(q,fit)])
            assert np.array_equal(pred[pos],best_h)
            selections.append({'bank':name,**best,'development_balanced_recall':best_score,'anchor_matches':int((best_h==gold).sum())})
            predictions.append(pred);holdouts.append(best_h)
            print('Grouped SVM family selected:',selections[-1],flush=True)
    winner=max(range(len(banks)),key=lambda i:selections[i]['development_balanced_recall'])
    chosen=selections[winner];pred=np.stack(predictions);primary=pred[winner];support=(pred==primary).sum(0)
    local=np.zeros(len(keys),int)
    for sim in similarities:
        near=np.stack([sim[:,pos[gold==c]].max(1) if (gold==c).any() else np.full(len(keys),-np.inf) for c in range(8)],axis=1)
        value=near[np.arange(len(keys)),primary].copy();near[np.arange(len(keys)),primary]=-np.inf
        local+=(value>=.65)&(value-near.max(1)>=.01)
    current=np.array([CLASSES.index(c) for c in frame.assigned_class])
    protected_rows=~frame.label_source.isin(['iterative_program_refinement_snapshot','historical_routing_snapshot']).to_numpy()
    gate=(support>=3)&(local>=2)&~human&~protected_rows&(primary!=current)
    proposed=int(gate.sum());accepted=chosen['anchor_matches']>=minimum_matches and chosen['development_balanced_recall']>=minimum_balanced and proposed<=50
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):
        result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'vit_knn_label_propagation_snapshot',support[i]/4.,'multi_method_vote_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False)
    pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'head_support':support,'local_support':local,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,predictions=pred,anchor_positions=pos,anchor_gold=gold,holdout_predictions=np.stack(holdouts),accepted=gate)
    report=json.loads((NOTES/'json/robust_geometry_heads_frozen.json').read_text())
    initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,
                  selected_heads=selections,primary_head=chosen,human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,
                  development_balanced_recall=chosen['development_balanced_recall'],human_diagnostic_protocol='fivegeometrygroupfoldSVMdevelopment',
                  validation_limit=plan['evaluation_limit'],plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),
                  accepted_by_development_guard=accepted,proposed_new_labels=proposed,changed_rows_vs_baseline=int(gate.sum()),
                  new_vit_propagation_changes=int(gate.sum()),new_neural_changes=0,
                  cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),
                  source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),
         'guard':plan['guard'],'primary_head_development_matches':chosen['anchor_matches'],'new_category_proposals':proposed,
         'private_review_used_for_selection':False,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','proposed_new_labels','changed_rows_vs_baseline','accepted_by_development_guard','seconds']},indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--base',action='store_true');main(parser.parse_args().base)

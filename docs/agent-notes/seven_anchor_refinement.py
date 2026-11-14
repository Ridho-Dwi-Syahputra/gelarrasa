"""Train-only refinement from seven participant-reviewed anchors and frozen priors.

No archive/review-inventory/test labels are read. Graph assignments are
transductive on TRAIN, never independent validation accuracy.
"""
import json
import time
import sys
import argparse
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT, NOTES, OLD, CLASSES, sha, dump, unit


def main(cleaning=False, seed_file='table-csv/minimal_human_seed_proposal.csv', run_prefix=None):
    started=time.time()
    prefix=run_prefix or ('seven_anchor_cleaned' if cleaning else 'seven_anchor')
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    seed_path=(NOTES/seed_file).resolve();assert seed_path.is_relative_to(NOTES)
    seeds=pd.read_csv(seed_path,dtype={'image_id':str},usecols=['image_id','source_path','proposed_class','source'])
    assert 1<=len(seeds)<=40 and seeds.image_id.is_unique and seeds.proposed_class.value_counts().le(5).all()
    assert seeds.source.eq('user_review_of_displayed_train_image').all()
    plan={'method':'seven human anchors plus source-group-crossfit and train-only graph consensus',
          'human_seed_file_sha256':sha(seed_path),'manual_unique_source_images':len(seeds),'manual_class_counts':seeds.proposed_class.value_counts().to_dict(),
          'manual_sample_weight':32.,'lighting_pseudo_sample_weight':4.,'base_sample_weight':'.25+.75*historical_program_score',
          'source_group_folds':5,'human_anchor_groups_always_fit_for_other_groups':True,
          'graph_banks':['dino2','dino3_foreground','dino3_task8'],'graph_k':20,'graph_alpha':.98,'graph_max_iterations':1000,
          'extra_change_gate':'same category has >=4/5 classifier votes and >=2/3 graph votes, or >=3/5 and3/3; also cosine>=.6 to reviewed same-category anchor in>=2/5 banks',
          'only_change_to_human_supported_classes':True,'human_rows_fixed':True,'accepted_lighting_rows_fixed':True,
          'test_rows_in_fit':0,'archive_labels_read':False,'unselected_team_review_labels_read':False,
          'configuration_selection':'fixed before fitting, not selected by agreement with remaining reviewed/archive cases',
          'validation':'separate leave-source-group-out predictions for seven human anchors; tiny and not official accuracy',
          'conflicting_pseudo_prior_weight_multiplier':.05 if cleaning else 1.,
          'conflict_gate':'closest reviewed category agrees in>=3/5 banks and cosine>=.6 in>=2 banks; no hard relabel from this gate',
          'human_holdout_guidance':'heldout source group and its human label excluded from both fitting and conflict-guided sample weights',
          'protected_sha256':protected}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str})
    frame=m[m.split.eq('train')&~m.is_crop][['parent_key','parent_id','source_path','source_sha256','assigned_class','category_score']].rename(columns={'parent_key':'key','parent_id':'image_id'}).sort_values('image_id').reset_index(drop=True)
    targets=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str})
    frame=frame.merge(targets[['key','group','fold']],on='key',validate='one_to_one')
    assert len(frame)==1906 and frame.key.str.startswith('train:').all()
    idx=dict(zip(frame.key,range(len(frame))))
    for row in seeds.itertuples():
        pos=idx['train:'+row.image_id];assert frame.iloc[pos].source_path==row.source_path
        assert sha(ROOT/row.source_path)==frame.iloc[pos].source_sha256
    old_y=np.array([CLASSES.index(c) for c in frame.assigned_class])
    lamp_path=NOTES/'table-csv/lighting_specificity_selected.csv'
    lamp_info=json.loads((NOTES/'json/lighting_specificity_frozen.json').read_text());assert sha(lamp_path)==lamp_info['prediction_sha256']
    lamp=pd.read_csv(lamp_path,dtype={'image_id':str}).set_index('key').loc[frame.key]
    assert np.array_equal(lamp.baseline_class,frame.assigned_class)
    base=np.array([CLASSES.index(c) for c in lamp.predicted_class]);lamp_mask=base!=old_y
    seed_positions=np.array([idx['train:'+k] for k in seeds.image_id]);human_mask=np.zeros(len(frame),bool);human_mask[seed_positions]=True
    seed_labels=np.array([CLASSES.index(c) for c in seeds.proposed_class]);base[seed_positions]=seed_labels
    weights=.25+.75*np.clip(frame.category_score.to_numpy(),0,1);weights[lamp_mask]=4.;weights[human_mask]=32.
    def take(path):
        z=np.load(path,allow_pickle=False);lookup=dict(zip(z['keys'],range(len(z['keys']))));return z['features'][[lookup[k] for k in frame.key]].astype('float32')
    xd=unit(take(OLD/'output/embeddings/stage2_repr_v1/dinov2_features.npz'))
    patch=take(OLD/'output/embeddings/stage5_patch_v1/original224.npz');pw=np.repeat(np.array([np.sqrt(.5),.5,.5],dtype='float32'),384)
    xp=unit(patch*pw)
    v3=take(NOTES/'.runtime/dinov3_train_features.npz');task=take(NOTES/'.runtime/task8_teacher_features.npz')
    banks={'dino2':xd,'dino2_local':xp,'dino3_global':unit(v3[:,:768]),
           'dino3_foreground':unit(np.c_[v3[:,:384]*np.sqrt(.5),v3[:,768:1152]*.5,v3[:,1152:]*.5]),
           'dino3_task8':unit(np.c_[unit(v3[:,:768]),task])}
    champions=json.loads((NOTES/'json/noisy_program_frozen.json').read_text())['family_champions']
    groups=frame.group.to_numpy();folds=frame.fold.to_numpy();human_groups=set(groups[human_mask]);in_human_group=np.isin(groups,list(human_groups))
    graph_outputs=[];graph_info=[];model_outputs=[];loo_outputs=[];similarities={}
    with threadpool_limits(limits=4):
        similarities={name:x@x.T for name,x in banks.items()}
    conflict_cache={}
    def conflicts(available):
        key=tuple(int(v) for v in available)
        if key not in conflict_cache:
            votes=[];close=[]
            for sim in similarities.values():
                nearest=np.stack([sim[:,available[base[available]==c]].max(1) if np.any(base[available]==c) else np.full(len(frame),-np.inf) for c in range(8)],axis=1)
                votes.append(nearest.argmax(1));close.append(nearest)
            tally=np.stack([(np.stack(votes)==c).sum(0) for c in range(8)],axis=1);major=tally.argmax(1)
            nearby=sum((v[np.arange(len(frame)),major]>=.6) for v in close)
            mask=(tally.max(1)>=3)&(nearby>=2)&(major!=base)&~human_mask&~lamp_mask
            conflict_cache[key]=mask
        return conflict_cache[key]
    def predict(kernel,tr,va,config):
        fit_weights=weights[tr].copy()
        if cleaning:
            available=seed_positions[np.isin(seed_positions,tr)]
            fit_weights[conflicts(available)[tr]]*=.05
        if config['kind']=='svm':
            model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
            model.fit(kernel[np.ix_(tr,tr)],base[tr],sample_weight=fit_weights)
            return model.predict(kernel[np.ix_(va,tr)]).astype('int32')
        sim=kernel[np.ix_(va,tr)];nn=np.argsort(-sim,axis=1,kind='stable')[:,:config['neighbors']]
        value=np.take_along_axis(sim,nn,axis=1);w=np.exp((value-value.max(1,keepdims=True))/.07)*fit_weights[nn]
        return np.stack([(w*(base[tr][nn]==c)).sum(1) for c in range(8)],axis=1).argmax(1)
    with threadpool_limits(limits=4):
        for winner in champions:
            name=winner['bank'];config=winner['config'];sim=similarities[name]
            kernel=sim if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-sim,0))
            pred=np.full(len(frame),-1,dtype='int32');pred[human_mask]=base[human_mask]
            for f in range(5):
                tr=np.flatnonzero((folds!=f)|in_human_group);va=np.flatnonzero((folds==f)&~in_human_group)
                assert not set(groups[tr])&set(groups[va]);pred[va]=predict(kernel,tr,va,config)
            for g in set(groups[in_human_group&~human_mask]):
                tr=np.flatnonzero(groups!=g);va=np.flatnonzero((groups==g)&~human_mask)
                pred[va]=predict(kernel,tr,va,config)
            assert (pred>=0).all();model_outputs.append(pred)
            loo=[]
            for pos in seed_positions:
                tr=np.flatnonzero(groups!=groups[pos]);assert not np.any(groups[tr]==groups[pos])
                loo.append(int(predict(kernel,tr,np.array([pos]),config)[0]))
            loo_outputs.append(loo)
            print('Weighted crossfit and human holdout:',name,flush=True)
        for name in plan['graph_banks']:
            sim=similarities[name].copy();np.fill_diagonal(sim,-np.inf)
            nn=np.argsort(-sim,axis=1,kind='stable')[:,:20];val=np.take_along_axis(sim,nn,axis=1)
            edge=np.maximum(val-.2,0)**4
            a=sparse.csr_matrix((edge.ravel(),(np.repeat(np.arange(len(frame)),20),nn.ravel())),shape=(len(frame),len(frame)))
            a=(a+a.T)*.5;denom=np.asarray(a.sum(1)).ravel();assert (denom>0).all()
            p=sparse.diags(1/denom)@a
            certainty=np.clip(frame.category_score.to_numpy(),.3,.8)
            if cleaning:certainty[conflicts(seed_positions)]=.05
            prior=np.eye(8,dtype='float32')[base]*certainty[:,None]+(1-certainty[:,None])/8
            prior[lamp_mask]=np.eye(8)[base[lamp_mask]]*.9+.1/8
            prior[human_mask]=np.eye(8)[base[human_mask]]
            f=prior.copy()
            for step in range(1000):
                updated=.98*(p@f)+.02*prior
                updated[human_mask]=prior[human_mask];updated[lamp_mask&~human_mask]=prior[lamp_mask&~human_mask]
                delta=float(np.abs(updated-f).max());f=updated
                if delta<1e-6:break
            f/=f.sum(1,keepdims=True);graph_outputs.append(f.argmax(1))
            graph_info.append({'bank':name,'iterations':step+1,'max_delta':delta,'converged':delta<1e-6})
            print('Train-only graph:',name,'iterations',step+1,flush=True)
    model_outputs=np.stack(model_outputs);graph_outputs=np.stack(graph_outputs)
    tallies=np.stack([(model_outputs==c).sum(0) for c in range(8)],axis=1);majority=tallies.argmax(1);support=tallies.max(1)
    graph_support=(graph_outputs==majority).sum(0)
    locality=np.zeros(len(frame),dtype='int32')
    for name,sim in similarities.items():
        nearest=np.stack([sim[:,seed_positions[seed_labels==c]].max(1) if np.any(seed_labels==c) else np.full(len(frame),-np.inf) for c in range(8)],axis=1)
        locality+=(nearest[np.arange(len(frame)),majority]>=.6)
    gate=((support>=4)&(graph_support>=2)|((support>=3)&(graph_support==3)))&(locality>=2)
    gate&=np.isin(majority,seed_labels)&~human_mask&~lamp_mask&(majority!=base)
    final=base.copy();final[gate]=majority[gate]
    source=np.full(len(frame),'iterative_program_refinement_snapshot',dtype='U64')
    source[lamp_mask]='two_view_lighting_ontology_snapshot';source[gate]='seven_anchor_graph_consensus_snapshot';source[human_mask]='participant_reviewed_seed'
    result=frame[['key','image_id','source_path','source_sha256']].copy()
    result['baseline_class']=np.array(CLASSES)[old_y];result['predicted_class']=np.array(CLASSES)[final]
    result['human_reviewed']=human_mask;result['lighting_rule_changed']=lamp_mask;result['propagation_changed']=gate
    result['classifier_votes']=support;result['graph_votes_for_classifier_category']=graph_support;result['local_anchor_banks']=locality
    result['category_source']=source
    result['category_score']=frame.category_score.to_numpy()
    result['category_score_kind']='historical_uncalibrated_program_score'
    result.loc[lamp_mask,'category_score']=1.;result.loc[lamp_mask,'category_score_kind']='two_view_ontology_gate'
    result.loc[gate,'category_score']=support[gate]/5;result.loc[gate,'category_score_kind']='multi_method_vote_share'
    result.loc[human_mask,'category_score']=1.;result.loc[human_mask,'category_score_kind']='human_reviewed_category'
    result['ground_truth_official']=False
    output=NOTES/f'table-csv/{prefix}_selected.csv';result.to_csv(output,index=False)
    loo=np.stack(loo_outputs);lv=np.stack([(loo==c).sum(0) for c in range(8)],axis=1);lp=lv.argmax(1)
    loo_table=seeds[['image_id','proposed_class']].copy();loo_table['leave_source_group_out_class']=np.array(CLASSES)[lp];loo_table['family_votes']=lv.max(1)
    loo_table['matches_reviewed_category']=lp==seed_labels;loo_table.to_csv(NOTES/f'table-csv/{prefix}_human_holdout.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=frame.key.to_numpy(str),classifier_predictions=model_outputs,graph_predictions=graph_outputs,human_holdout_predictions=loo)
    report={'status':'frozen_before_posthoc_audit','method':prefix+'_crossfit_graph_and_lighting_v1',
            'manual_unique_source_images_used':len(seeds),'manual_class_counts':plan['manual_class_counts'],'test_rows_in_fit':0,
            'human_seed_sha256':sha(seed_path),'prediction_sha256':sha(output),'train_rows':1906,
            'accepted_lighting_rule_rows':int(lamp_mask.sum()),'accepted_graph_crossfit_rows':int(gate.sum()),
            'changed_rows_vs_baseline':int((final!=old_y).sum()),'graphs':graph_info,
            'downweighted_conflicting_pseudo_rows':int(conflicts(seed_positions).sum()) if cleaning else 0,
            'human_leave_source_group_out_matches':int((lp==seed_labels).sum()),'human_holdout_rows':len(seeds),
            'semantic_accuracy':None,'human_holdout_is_tiny_and_not_unbiased_final_validation':True,
            'classification_training_scope':'assignment heads only; not final competition classifier',
            'active_data_modified':False,'protected_sha256':protected,'source_code_sha256':sha(__file__),'seconds':time.time()-started}
    assert all(sha(ROOT/name)==value for name,value in protected.items())
    dump(NOTES/f'json/{prefix}_frozen.json',report)
    print(json.dumps(report,indent=2),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--clean-conflicts',action='store_true')
    parser.add_argument('--seed-file',default='table-csv/minimal_human_seed_proposal.csv');parser.add_argument('--prefix')
    args=parser.parse_args();main(args.clean_conflicts,args.seed_file,args.prefix)

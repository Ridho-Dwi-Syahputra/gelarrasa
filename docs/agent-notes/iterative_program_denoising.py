"""Fixed iterative automatic label denoising; no manual or archive targets.

Cross-fitting is a refinement mechanism, not independent accuracy evaluation:
targets after the first round depend on earlier full-train program outputs.
"""
import time
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump,unit
import json


def main():
    started=time.time()
    protected={name:sha(ROOT/name) for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    previous=json.loads((NOTES/'json/noisy_program_frozen.json').read_text())
    source=NOTES/'table-csv/noisy_program_selected.csv';assert sha(source)==previous['prediction_sha256']
    plan={'method':'iterative crossfit program-label denoising','input_program_label_sha256':sha(source),'manual_labels_consumed':0,'test_rows_in_fit':0,
          'max_total_rounds':4,'starting_round':1,'stop_changed_fraction':.01,'family_agreement_minimum':4,'feature_family_count':5,
          'model_configs':previous['family_champions'],'fixed_anchors':'initial program confidence>=.95, agrees with unique automatic CNN consensus and at least3 reliable source votes',
          'stopping_uses_reference_labels':False,'choose_round_by_archive_agreement':False,'selection_scores_are_not_independent_after_iterative_target_updates':True,
          'reference_archive_read':False,'researcher_has_seen_previous_private_audits':True,'protected_sha256':protected}
    dump(NOTES/'json/iterative_program_plan.json',plan)
    targets=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv',dtype={'image_id':str})
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});m=m[m.split.eq('train')&~m.is_crop]
    frame=targets.merge(m[['parent_key','assigned_class','category_score','source_sha256']],left_on='key',right_on='parent_key',validate='one_to_one')
    initial=np.array([CLASSES.index(c) for c in frame.assigned_class]);automatic=np.array([CLASSES.index(c) for c in frame.automatic_class])
    anchors=(frame.category_score.to_numpy()>=.95)&frame.automatic_anchor.to_numpy()&(frame.reliable_vote_count.to_numpy()>=3)&(automatic==initial)
    current=pd.read_csv(source,dtype={'image_id':str});assert np.array_equal(current.key.to_numpy(str),frame.key.to_numpy(str))
    labels=np.array([CLASSES.index(c) for c in current.predicted_class]);restored=int((labels[anchors]!=initial[anchors]).sum());labels[anchors]=initial[anchors]
    weights=.25+.75*np.clip(frame.category_score.to_numpy(),0,1)
    def load(folder,name='original224.npz'):return np.load(OLD/f'output/embeddings/{folder}/{name}',allow_pickle=False)
    def take(z):
        index={key:i for i,key in enumerate(z['keys'])};return z['features'][[index[k] for k in frame.key]].astype('float32')
    xd=unit(take(load('stage2_repr_v1','dinov2_features.npz')))
    patch=take(load('stage5_patch_v1'));pw=np.repeat(np.array([np.sqrt(.5),.5,.5],dtype='float32'),384);xp=unit(patch*pw)
    z=np.load(NOTES/'.runtime/dinov3_train_features.npz',allow_pickle=False);assert np.array_equal(z['keys'],frame.key.to_numpy(str));v=z['features']
    task=np.load(NOTES/'.runtime/task8_teacher_features.npz',allow_pickle=False);assert np.array_equal(task['keys'],frame.key.to_numpy(str))
    banks={'dino2':xd,'dino2_local':xp,'dino3_global':unit(v[:,:768]),
           'dino3_foreground':unit(np.c_[v[:,:384]*np.sqrt(.5),v[:,768:1152]*.5,v[:,1152:]*.5]),
           'dino3_task8':unit(np.c_[unit(v[:,:768]),task['features']])}
    folds=[(np.flatnonzero(frame.fold.to_numpy()!=i),np.flatnonzero(frame.fold.to_numpy()==i)) for i in range(5)]
    kernels={}
    with threadpool_limits(limits=4):
        for champion in previous['family_champions']:
            name=champion['bank'];config=champion['config'];sim=banks[name]@banks[name].T
            kernels[name]=sim if config['kind']=='knn' or config['kernel']=='linear' else np.exp(-2*config['gamma']*np.maximum(1-sim,0))
    history=[];snapshots=[]
    with threadpool_limits(limits=4):
        for round_id in range(2,5):
            outputs=[]
            for champion in previous['family_champions']:
                name=champion['bank'];config=champion['config'];kernel=kernels[name];pred=np.full(len(frame),-1,dtype='int32')
                for tr,va in folds:
                    assert not set(frame.group.iloc[tr])&set(frame.group.iloc[va])
                    if config['kind']=='svm':
                        model=SVC(kernel='precomputed',C=config['C'],class_weight='balanced',decision_function_shape='ovr',break_ties=True)
                        model.fit(kernel[np.ix_(tr,tr)],labels[tr],sample_weight=weights[tr]);pred[va]=model.predict(kernel[np.ix_(va,tr)])
                    else:
                        sim=kernel[np.ix_(va,tr)];nn=np.argsort(-sim,axis=1,kind='stable')[:,:config['neighbors']];values=np.take_along_axis(sim,nn,axis=1)
                        w=np.exp((values-values.max(1,keepdims=True))/.07)*weights[tr][nn]
                        scores=np.stack([(w*(labels[tr][nn]==c)).sum(1) for c in range(8)],axis=1);pred[va]=scores.argmax(1)
                assert (pred>=0).all();outputs.append(pred)
            tally=np.stack([(np.stack(outputs)==c).sum(0) for c in range(8)],axis=1);majority=tally.argmax(1);support=tally.max(1)
            change=(majority!=labels)&(support>=4)&~anchors
            updated=labels.copy();updated[change]=majority[change]
            assert np.array_equal(updated[anchors],initial[anchors]);counts=np.bincount(updated,minlength=8);assert (counts>=5).all()
            history.append({'round':round_id,'changes_from_previous_round':int(change.sum()),'changes_vs_initial_program':int((updated!=initial).sum()),'class_counts':dict(zip(CLASSES,counts.tolist()))})
            print('Automatic denoising round:',history[-1],flush=True)
            labels=updated;snapshots.append(labels.copy())
            if int(change.sum())<.01*len(frame):break
    result=frame[['key','image_id','source_sha256']].copy();result['baseline_class']=np.array(CLASSES)[initial];result['predicted_class']=np.array(CLASSES)[labels]
    result['method']='iterative_crossfit_program_denoising_v1';result['fixed_automatic_anchor']=anchors;result['ground_truth_verified']=False
    output=NOTES/'table-csv/iterative_program_selected.csv';result.to_csv(output,index=False)
    np.savez_compressed(NOTES/'.runtime/iterative_program_states.npz',keys=frame.key.to_numpy(str),round_predictions=np.stack(snapshots))
    report={'status':'frozen_before_reference_audit','manual_labels_consumed':0,'test_rows_in_fit':0,'train_rows':1906,'fixed_automatic_anchors':int(anchors.sum()),
            'initial_anchor_restorations':restored,'history':history,'selected_final_round':history[-1]['round'],'prediction_sha256':sha(output),
            'stop_reason':'changed_fraction_below_one_percent' if history[-1]['changes_from_previous_round']<.01*len(frame) else 'predeclared_maximum_rounds',
            'semantic_accuracy':None,'active_data_modified':False,'protected_sha256':protected,'source_code_sha256':sha(__file__),'seconds':time.time()-started}
    dump(NOTES/'json/iterative_program_frozen.json',report)
    assert all(sha(ROOT/name)==digest for name,digest in protected.items())
    print(json.dumps({k:v for k,v in report.items() if k!='protected_sha256'},indent=2),flush=True)


if __name__=='__main__':main()

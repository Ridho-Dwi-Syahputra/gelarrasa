"""Three-family, source-held-out augmented human prototype consensus.

Each source contributes at most one max-view similarity; eight views never count
as eight annotations. Private review is not loaded by this experiment.
"""
import datetime,json,time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit

def norm(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-12)

def main():
    start=time.time();prefix='augmented_prototype_consensus'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'three_family_augmented_genuine_source_prototypes_v1','human_sources':34,'views_per_source':8,'new_manual_labels':0,'test_rows_in_fit':0,
          'banks':['V2CLS_ROImaxview','CNNfullsourceT1T2maxview','V3CLS_PCA128whiteoriginal'],
          'source_vote':'maxviewcosinesimilarityperSOURCE,thenmaxperclass;noautomaticlabeltargets',
          'candidate':'atleast2of3banksagree;minmarginandminrawsimilarityonagreeingbanks;3bankconflictsabstain',
          'margin_grid':[.02,.05,.1],'raw_similarity_grid':[.45,.55,.65],
          'normalizer':'centeringfitouterTRAINonly;V3PCAwhiteningfitouterTRAINonly;all8querysourceviewsheldouttogether',
          'selection':'noninferiorwhole34withV3fallback;partialcoverage>=5,precision>=.9;maxmatches,macro,precision,coverage,strictermargin',
          'guard':'matches>=26andmacro>=.783333333333;covered>=5,precision>=.9',
          'maximum_changes':20,'cap':'descendingminimumclassmarginthenstablekeyorder',
          'score':'minimumcosineclassmargin;notprobabilityofcorrectlabel',
          'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,'protected_sha256':protected,
          'source_code_sha256':sha(__file__),'evaluation_limit':'34developmentanchorsreused;noindependentsemanticAccuracy;maxviewcanmatchbackground'}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34 and seeds.assigned_class.value_counts().le(5).all()
    geometry=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=geometry.fold.to_numpy(),geometry.group.to_numpy()
    v2=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);cnn=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False)
    aug=np.load(NOTES/'.runtime/human_source_aug_features.npz',allow_pickle=False);ca=np.load(NOTES/'.runtime/augmented_source_dictionary.npz',allow_pickle=False)
    for z in [v2,v3,cnn]:assert np.array_equal(z['keys'],keys)
    aidx={k:i for i,k in enumerate(aug['keys'])};cidx={k:i for i,k in enumerate(ca['keys'])};ar=np.array([aidx[k] for k in seeds.key]);cr=np.array([cidx[k] for k in seeds.key])
    original2=norm(np.concatenate([norm(v2['features'][:,:768]),norm(v2['features'][:,2304:3072])],axis=-1))
    views2=norm(np.concatenate([norm(aug['features'][ar,:,:768]),norm(aug['features'][ar,:,2304:3072])],axis=-1))
    originalcnn=norm(np.concatenate([norm(cnn['features'][:,0]),norm(cnn['features'][:,1])],axis=-1))
    viewscnn=norm(np.concatenate([norm(ca['features'][cr,:,0]),norm(ca['features'][cr,:,1])],axis=-1))
    original3=norm(v3['features'][:,:768]);views3=original3[pos,None,:]
    banks=[(original2,views2,False),(originalcnn,viewscnn,False),(original3,views3,True)]
    def infer(train,query,hs):
        predictions=[];margins=[];raw=[]
        for x,views,white in banks:
            if white:
                pca=PCA(n_components=128,svd_solver='randomized',random_state=20261108).fit(x[train]);scale=np.maximum(np.sqrt(pca.explained_variance_),1e-6)
                q=norm(pca.transform(x[query])/scale);r=norm(pca.transform(views[hs].reshape(-1,x.shape[1]))/scale)
            else:
                mean=x[train].mean(0);q=norm(x[query]-mean);r=norm(views[hs].reshape(-1,x.shape[1])-mean)
            sim=(q@r.T).reshape(len(query),len(hs),views.shape[1]).max(-1)
            rawsim=(x[query]@views[hs].reshape(-1,x.shape[1]).T).reshape(len(query),len(hs),views.shape[1]).max(-1)
            scores=np.stack([sim[:,gold[hs]==c].max(1) if (gold[hs]==c).any() else np.full(len(query),-1.) for c in range(8)],1)
            rawscore=np.stack([rawsim[:,gold[hs]==c].max(1) if (gold[hs]==c).any() else np.full(len(query),-1.) for c in range(8)],1)
            order=np.argsort(-scores,axis=1,kind='stable');pred=order[:,0];margin=scores[np.arange(len(query)),pred]-scores[np.arange(len(query)),order[:,1]]
            predictions.append(pred);margins.append(margin);raw.append(rawscore[np.arange(len(query)),pred])
        return np.stack(predictions),np.stack(margins),np.stack(raw)
    def route(parts,margin_threshold,similarity):
        pred,margin,raw=parts;votes=np.stack([(pred==c).sum(0) for c in range(8)],1);candidate=votes.argmax(1);agree=pred==candidate
        # The third dissenting bank does not veto a two-family agreement.
        minmargin=np.min(np.where(agree,margin,np.inf),axis=0);minraw=np.min(np.where(agree,raw,np.inf),axis=0)
        covered=(votes.max(1)>=2)&(minmargin>=margin_threshold)&(minraw>=similarity)
        return candidate,covered,minmargin,minraw
    old=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz',allow_pickle=False);info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);baseline=old['probabilities'][wi].argmax(1)
    with threadpool_limits(limits=2):
        oof=[np.empty((3,n),dtype=int),np.empty((3,n)),np.empty((3,n))]
        for f in range(5):
            train=np.flatnonzero(folds!=f);query=np.flatnonzero(folds==f);hs=np.flatnonzero(folds[pos]!=f);assert not set(groups[train])&set(groups[query])
            result=infer(train,query,hs)
            for target,value in zip(oof,result):target[:,query]=value
        development=[]
        for t in plan['margin_grid']:
            for similarity in plan['raw_similarity_grid']:
                pred,covered,margin,raw=route(oof,t,similarity);fallback=np.where(covered,pred,baseline);count=int(covered[pos].sum());correct=int((covered[pos]&(pred[pos]==gold)).sum());precision=correct/count if count else 0.
                matches=int((fallback[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,fallback[pos]));ok=matches>=26 and macro>=.783333333333-1e-12 and count>=5 and precision>=.9
                development.append({'margin':t,'similarity':similarity,'covered':count,'correct':correct,'precision':precision,'anchor_matches':matches,'development_balanced_recall':macro,'eligible':bool(ok)})
        choices=[r for r in development if r['eligible']];chosen=max(choices,key=lambda r:(r['anchor_matches'],r['development_balanced_recall'],r['precision'],r['covered'],r['margin'])) if choices else development[0];accepted=bool(choices)
        full=infer(np.arange(n),np.arange(n),np.arange(34));primary,covered,score,raw=route(full,chosen['margin'],chosen['similarity'])
    human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=covered&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'augmented_source_prototype_snapshot',score[i],'minimum_cosine_class_margin',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'covered':covered,'minimum_margin':score,'minimum_raw_similarity':raw,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,predictions=primary,covered=covered,score=score,accepted=gate,oof_predictions=oof[0],oof_margins=oof[1],oof_raw_similarity=oof[2])
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_augmented_prototype_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'source':name} for name in plan['banks']],qualified_heads=[{'source':name} for name in plan['banks']],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],confidence_calibration={'selected':chosen,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'chosen':chosen,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'development':development,'seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()

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
    start=time.time();prefix='local_descriptor_consensus'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'two_scale_genuine_seed_local_descriptor_consensus_v1','human_sources':34,'views_per_source':1,'reference':'https://arxiv.org/abs/1903.12290','adaptation':'frozenConvNeXtpatches,fixedprojection128,trainmeanremoval;notepisodicallytrainedDN4','new_manual_labels':0,'test_rows_in_fit':0,
          'banks':['fine7x7projected128','coarse7x7projected128'],
          'source_vote':'mean3nearestreferencepatchcosineperclass,softROIweightedquerymean;34genuineonly;refsweight>=.25max',
          'candidate':'bothpatchscalesagree;minimumclassmarginandminimumabsolutewinninglocalscore',
          'margin_grid':[.0,.01,.03,.05],'raw_similarity_grid':[.3,.5,.7],
          'normalizer':'fixedrandomprojectionnotdatafitted;patchmeanfitouterTRAINonly;allquerysourcegroupsheldoutfromreferences',
          'selection':'noninferiorwhole34withV3fallback;partialcoverage>=5,precision>=.9;maxmatches,macro,precision,coverage,strictermargin',
          'guard':'matches>=27andmacro>=.808333333333;covered>=5,precision>=.9',
          'maximum_changes':20,'cap':'descendingminimumclassmarginthenstablekeyorder',
          'score':'minimumcosineclassmargin;notprobabilityofcorrectlabel',
          'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,'protected_sha256':protected,
          'source_code_sha256':sha(__file__),'evaluation_limit':'34developmentanchorsreused;noindependentsemanticAccuracy;maxviewcanmatchbackground'}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys)
    seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seeds.key]);gold=np.array([CLASSES.index(c) for c in seeds.assigned_class]);assert len(pos)==34 and seeds.assigned_class.value_counts().le(5).all()
    geometry=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=geometry.fold.to_numpy(),geometry.group.to_numpy()
    index_path=NOTES/'.runtime/convnext_patch_token_index.json';index=json.loads(index_path.read_text());assert np.array_equal(index['keys'],keys)
    bags=[];weights=[]
    rng=np.random.default_rng(20261108)
    for name,dimension in [('fine',192),('coarse',768)]:
        record=index['files'][name];path=ROOT/record['path'];assert sha(path)==record['sha256'];raw=np.load(path,mmap_mode='r',allow_pickle=False)
        weightrecord=index['files'][name+'_weight'];wp=ROOT/weightrecord['path'];assert sha(wp)==weightrecord['sha256'];w=np.load(wp,allow_pickle=False).astype(np.float32)
        if name=='fine':
            raw=np.asarray(raw,dtype=np.float32).reshape(n,7,2,7,2,192).mean((2,4)).reshape(n,49,192)
            w=w.reshape(n,7,2,7,2).mean((2,4)).reshape(n,49)
        projection=rng.normal(size=(dimension,128)).astype(np.float32)/np.sqrt(128)
        with threadpool_limits(limits=2):bag=np.asarray(raw,dtype=np.float32)@projection
        bags.append(bag);weights.append(w)
    def infer(train,query,hs):
        predictions=[];margins=[];rawscores=[]
        for raw,w in zip(bags,weights):
            mean=np.average(raw[train].reshape(-1,128),axis=0,weights=w[train].ravel())
            x=norm(raw-mean);scores=np.full((len(query),8),-1.,np.float32)
            for category in range(8):
                source=pos[hs[gold[hs]==category]]
                if not len(source):continue
                keep=w[source]>=.25*np.maximum(w[source].max(1,keepdims=True),1e-6);references=x[source][keep];k=min(3,len(references))
                for first in range(0,len(query),32):
                    ids=query[first:first+32];sim=x[ids].reshape(-1,128)@references.T
                    local=np.partition(sim,-k,axis=1)[:,-k:].mean(1).reshape(len(ids),49)
                    scores[first:first+len(ids),category]=(local*w[ids]).sum(1)/np.maximum(w[ids].sum(1),1e-12)
            order=np.argsort(-scores,axis=1,kind='stable');pred=order[:,0];winning=scores[np.arange(len(query)),pred];margin=winning-scores[np.arange(len(query)),order[:,1]]
            predictions.append(pred);margins.append(margin);rawscores.append(winning)
        return np.stack(predictions),np.stack(margins),np.stack(rawscores)
    def route(parts,margin_threshold,similarity):
        pred,margin,raw=parts;votes=np.stack([(pred==c).sum(0) for c in range(8)],1);candidate=votes.argmax(1);agree=pred==candidate
        # The third dissenting bank does not veto a two-family agreement.
        minmargin=np.min(np.where(agree,margin,np.inf),axis=0);minraw=np.min(np.where(agree,raw,np.inf),axis=0)
        covered=(votes.max(1)>=2)&(minmargin>=margin_threshold)&(minraw>=similarity)
        return candidate,covered,minmargin,minraw
    old=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz',allow_pickle=False);info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);baseline=old['probabilities'][wi].argmax(1)
    with threadpool_limits(limits=2):
        oof=[np.empty((2,n),dtype=int),np.empty((2,n)),np.empty((2,n))]
        for f in range(5):
            train=np.flatnonzero(folds!=f);query=np.flatnonzero(folds==f);hs=np.flatnonzero(folds[pos]!=f);assert not set(groups[train])&set(groups[query])
            result=infer(train,query,hs)
            for target,value in zip(oof,result):target[:,query]=value
        development=[]
        for t in plan['margin_grid']:
            for similarity in plan['raw_similarity_grid']:
                pred,covered,margin,raw=route(oof,t,similarity);fallback=np.where(covered,pred,baseline);count=int(covered[pos].sum());correct=int((covered[pos]&(pred[pos]==gold)).sum());precision=correct/count if count else 0.
                matches=int((fallback[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,fallback[pos]));ok=matches>=27 and macro>=.808333333333-1e-12 and count>=5 and precision>=.9
                development.append({'margin':t,'similarity':similarity,'covered':count,'correct':correct,'precision':precision,'anchor_matches':matches,'development_balanced_recall':macro,'eligible':bool(ok)})
        choices=[r for r in development if r['eligible']];chosen=max(choices,key=lambda r:(r['anchor_matches'],r['development_balanced_recall'],r['precision'],r['covered'],r['margin'])) if choices else development[0];accepted=bool(choices)
        full=infer(np.arange(n),np.arange(n),np.arange(34));primary,covered,score,raw=route(full,chosen['margin'],chosen['similarity'])
    human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=covered&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'local_descriptor_seed_snapshot',score[i],'minimum_cosine_class_margin',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'covered':covered,'minimum_margin':score,'minimum_raw_similarity':raw,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,predictions=primary,covered=covered,score=score,accepted=gate,oof_predictions=oof[0],oof_margins=oof[1],oof_raw_similarity=oof[2])
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_augmented_prototype_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'source':name} for name in plan['banks']],qualified_heads=[{'source':name} for name in plan['banks']],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],confidence_calibration={'selected':chosen,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'chosen':chosen,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'development':development,'seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()

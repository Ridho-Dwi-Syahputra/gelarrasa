"""Train-only source-consensus clustering with existing human seed names.

Single-seed cluster labels are a cluster-assumption hypothesis, not human review
of the other members. All OOF query sources are excluded from cluster fitting.
No automatic categories, full review or visual archive serve as training labels.
"""
import datetime
import json
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from base_self_training import graph_prediction
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit


def main():
    start=time.time();prefix='single_anchor_cluster'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'single_genuine_anchor_multiview_cluster_expansion_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'targets':'same34humanONLY;noautomaticcategorytargets;clustergeometryunsupervisedonTRAIN',
          'banks':['source_dictionary','DINOregional','CNNdictionary'],'clusters':[32,64,128],'KMeans':'n_init3,max_iter150,seed20261108+fold',
          'cluster_qualification':'2to80members;>=1humanseedandallhumanseednamesidentical;seednotoutlier(beyondclusterQ75squaredradius)',
          'query_gate':'nearestcenterlabelqualified;within1.25*Q95squaredradius;secondcenterdistancegap>=.02;uncenteredcosinetohumanincluster>=.6',
          'consensus':'>=2of3bankswithsamecategory;conflictsabstain;atleastoneOOFcoveredcorrecthumanexistsforthatclass',
          'development_guard':'OOFconsensuscoverage>=5across>=2classes,allcoveredhumancorrect;whole34fallbackbaseline25nondecreasingmatchesandmacro',
          'selection':'largestcoveredhuman,thenmacro,thenmatches;listedclusterorderbreaksties',
          'final_gate':'selectedclusterconsensusandatleast1of3TRAINgraphsagree;humanprotected',
          'baseline':'bestsemantic_dictionary_neighborsgroupedOOF;fallbacknotclampedhumantruth',
          'maximum_changes':20,'cap':'descendingbankagreementthenminimumhuman-cosinesimilarity,stablekeyorder',
          'score_kind':'multi_method_vote_share=support/3;NOTprobabilityofcorrectness',
          'evaluation_limit':'34reuseddevelopmentandclasscoveragecalibration,notindependentAccuracy;single-seedclusterpurityunverified;graphcorroborationnotindependent',
          'private_review_or_archive_read':False,'researcher_has_seen_prior_audits':True,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    d=np.load(NOTES/'.runtime/semantic_dictionary_features.npz',allow_pickle=False);b=np.load(NOTES/'.runtime/dinov2_base_train_features.npz',allow_pickle=False);c=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False)
    assert all(np.array_equal(z['keys'],keys) for z in [d,b,c])
    dictionary=unit(np.concatenate([unit(d['features'][:,0]),unit(d['features'][:,1])],1));regional=unit(np.c_[unit(b['features'][:,:768]),unit(b['features'][:,2304:])]);cnn=unit(np.c_[unit(c['features']),unit(d['features'][:,0])]);banks=[dictionary,regional,cnn]
    z=np.load(NOTES/'.runtime/semantic_dictionary_neighbors_outputs.npz',allow_pickle=False);assert np.array_equal(z['keys'],keys)
    info=json.loads((NOTES/'json/semantic_dictionary_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);baseline=z['probabilities'][wi]
    base_matches=int((baseline[pos].argmax(1)==gold).sum());base_macro=float(balanced_accuracy_score(gold,baseline[pos].argmax(1)))
    history=[];training=[];outputs={}

    def fit_predict(raw,train,query,human_sources,k,seed):
        center=raw[train].mean(0);x=unit(raw-center);model=KMeans(n_clusters=k,n_init=3,max_iter=150,random_state=seed,algorithm='lloyd').fit(x[train])
        member=model.labels_;human_cluster=model.predict(x[pos[human_sources]]);distance=model.transform(x[query])**2;query_cluster=distance.argmin(1)
        local_dist=((x[train]-model.cluster_centers_[member])**2).sum(1);near=np.sort(distance,axis=1)[:,:2]
        pred=np.full(len(query),-1,int);quality=np.zeros(len(query));qualified=0
        for cluster in np.unique(human_cluster):
            sources=human_sources[human_cluster==cluster];labels=np.unique(gold[sources]);members=member==cluster;count=int(members.sum())
            if len(labels)!=1 or not 2<=count<=80:continue
            seed_dist=((x[pos[sources]]-model.cluster_centers_[cluster])**2).sum(1)
            if float(seed_dist.max())>float(np.quantile(local_dist[members],.75))+1e-6:continue
            qualified+=1;radius=max(float(np.quantile(local_dist[members],.95)),1e-6);rows=np.flatnonzero(query_cluster==cluster)
            if not len(rows):continue
            similarity=(raw[query[rows]]@raw[pos[sources]].T).max(1)
            accept=(distance[rows,cluster]<=1.25*radius)&(near[rows,1]-near[rows,0]>=.02)&(similarity>=.6)
            pred[rows[accept]]=labels[0];quality[rows[accept]]=similarity[accept]
        return pred,quality,qualified

    def consensus(predictions,quality):
        votes=np.stack([(predictions==k).sum(0) for k in range(8)],1);candidate=votes.argmax(1);support=votes.max(1);covered=support>=2
        q=np.where(predictions==candidate,quality,1.).min(0)
        return candidate,covered,support,q

    with threadpool_limits(limits=2):
        for clusters in plan['clusters']:
            predictions=np.full((3,n),-1,int);quality=np.zeros((3,n))
            for bank,raw in enumerate(banks):
                for fold in range(5):
                    train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);hs=np.flatnonzero(folds[pos]!=fold);assert not set(groups[train])&set(groups[query])
                    pred,q,qualified=fit_predict(raw,train,query,hs,clusters,20261108+fold)
                    predictions[bank,query]=pred;quality[bank,query]=q;training.append({'clusters':clusters,'bank':bank,'fold':fold,'qualified_clusters':qualified,'query_covered':int((pred>=0).sum())})
            candidate,covered,support,q=consensus(predictions,quality);allowed_classes=sorted(set(gold[covered[pos]&(candidate[pos]==gold)]));covered&=np.isin(candidate,allowed_classes)
            base=baseline.argmax(1).copy();base[covered]=candidate[covered];matches=int((base[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,base[pos]));count=int(covered[pos].sum());correct=int((covered[pos]&(candidate[pos]==gold)).sum())
            ok=count>=5 and correct==count and len(set(gold[covered[pos]]))>=2 and matches>=base_matches and macro>=base_macro-1e-12
            record={'clusters':clusters,'covered':count,'correct':correct,'classes_covered':len(set(gold[covered[pos]])),'anchor_matches':matches,'development_balanced_recall':macro,'allowed_classes':allowed_classes,'eligible':bool(ok)};history.append(record);outputs[clusters]=(predictions,quality)
            print('Single-anchor cluster development',record,'seconds',round(time.time()-start),flush=True)
        feasible=[r for r in history if r['eligible']];chosen=max(feasible,key=lambda r:(r['covered'],r['development_balanced_recall'],r['anchor_matches'])) if feasible else history[0];accepted=bool(feasible)
        predictions=[];quality=[]
        for raw in banks:
            pred,q,_=fit_predict(raw,np.arange(n),np.arange(n),np.arange(34),chosen['clusters'],20261108);predictions.append(pred);quality.append(q)
        predictions=np.stack(predictions);quality=np.stack(quality);primary,covered,support,rank_quality=consensus(predictions,quality);covered&=np.isin(primary,chosen['allowed_classes'])
        prior=baseline.copy();prior[covered]=.02*baseline[covered]+.98*np.eye(8)[primary[covered]]
        graphs=np.stack([graph_prediction(unit(raw-raw.mean(0)),prior,pos,gold) for raw in banks])
    graph_support=(graphs==primary).sum(0);human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=covered&(graph_support>=1)&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.lexsort((np.arange(n),-rank_quality,-support));ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'single_anchor_cluster_consensus_snapshot',support[i]/3,'multi_method_vote_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(training).to_csv(NOTES/f'table-csv/{prefix}_training.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'covered':covered,'bank_support':support,'human_similarity':rank_quality,'graph_support':graph_support,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=prior,predictions=predictions,graphs=graphs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_single_anchor_cluster_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'bank':b} for b in plan['banks']],qualified_heads=[{'bank':b} for b in plan['banks']],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],
        confidence_calibration={'selected':chosen,'independent_validation':False,'objective':'seedclustercoverageexpansionwithdevelopmentnoninferiority'},human_diagnostic_protocol=plan['evaluation_limit'],accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'guard':plan['development_guard'],'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({k:report[k] for k in ['primary_head','accepted_by_development_guard','proposed_new_labels','changed_rows_vs_baseline','seconds']},indent=2),flush=True)


if __name__=='__main__':main()

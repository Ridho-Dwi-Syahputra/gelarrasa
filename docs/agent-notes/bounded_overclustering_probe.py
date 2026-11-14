"""Train-only overclustering probe using only fourteen bounded human anchors."""
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,unit,sha,dump


def main():
    source=NOTES/'table-csv/bounded_consensus_selected.csv'
    fixed=json.loads((NOTES/'json/bounded_consensus_frozen.json').read_text());assert sha(source)==fixed['prediction_sha256']
    frame=pd.read_csv(source,dtype={'image_id':str});seeds=pd.read_csv(NOTES/'table-csv/bounded_human_seeds.csv',dtype={'image_id':str},usecols=['image_id','proposed_class'])
    assert len(frame)==1906 and len(seeds)==14 and seeds.proposed_class.value_counts().le(5).all()
    index=dict(zip(frame.image_id,range(len(frame))));seed_positions=np.array([index[k] for k in seeds.image_id]);seed_classes=np.array([CLASSES.index(c) for c in seeds.proposed_class])
    dump(NOTES/'json/bounded_overclustering_plan.json',{'manual_train_anchors':14,'test_rows_in_fit':0,
         'banks':['dino2','dino3_global','dino3_task8'],'k':[32,64,128],'PCA_dimensions':64,'random_state':20261107,
         'cluster_label_rule':'at least2 human anchors, all same category; otherwise no proposed label',
         'stability_rule':'at least4of9 configurations and2feature banks, no conflicting suggested categories',
         'source_candidate_sha256':sha(source),'unselected_review_labels_loaded':False,'archive_labels_loaded':False,
         'purpose':'diagnostic only, not automatically publishing extra labels'})
    def take(path):
        z=np.load(path,allow_pickle=False);lookup=dict(zip(z['keys'],range(len(z['keys']))));return unit(z['features'][[lookup[k] for k in frame.key]].astype('float32'))
    d=take(OLD/'output/embeddings/stage2_repr_v1/dinov2_features.npz')
    z=np.load(NOTES/'.runtime/dinov3_train_features.npz',allow_pickle=False);ix=dict(zip(z['keys'],range(len(z['keys']))));v=z['features'][[ix[k] for k in frame.key]]
    global_v=unit(v[:,:768]);task=take(NOTES/'.runtime/task8_teacher_features.npz')
    banks={'dino2':d,'dino3_global':global_v,'dino3_task8':unit(np.c_[global_v,task])}
    history=[];proposals=[];bank_names=[]
    with threadpool_limits(limits=4):
        for name,x in banks.items():
            pca=PCA(n_components=64,svd_solver='randomized',random_state=20261107)
            reduced=unit(pca.fit_transform(x).astype('float32'))
            for k in [32,64,128]:
                model=KMeans(n_clusters=k,n_init=3,max_iter=200,random_state=20261107,algorithm='lloyd')
                groups=model.fit_predict(reduced);proposal=np.full(len(frame),-1,dtype='int32')
                touched=0;conflicts=0;supported=0
                for group in range(k):
                    labels=seed_classes[groups[seed_positions]==group]
                    if len(labels):touched+=1
                    if len(set(labels))>1:conflicts+=1
                    if len(labels)>=2 and len(set(labels))==1:
                        proposal[groups==group]=int(labels[0]);supported+=1
                history.append({'bank':name,'k':k,'human_occupied_clusters':touched,'unanchored_clusters':k-touched,
                                'clusters_with_conflicting_human_categories':conflicts,'clusters_with_two_agreeing_human_anchors':supported,
                                'rows_with_supported_proposed_category':int((proposal>=0).sum())})
                proposals.append(proposal);bank_names.append(name)
                print('Overclustering:',name,k,history[-1],flush=True)
    proposed=np.stack(proposals);result=frame[['key','image_id','baseline_class','predicted_class']].rename(columns={'predicted_class':'published_candidate_class'}).copy()
    suggested=[];support=[];diversity=[]
    for i in range(len(frame)):
        choices=proposed[:,i];known=choices>=0;values=set(choices[known].tolist())
        count=int(known.sum());families=len({bank_names[j] for j in np.flatnonzero(known)})
        suggested.append(CLASSES[next(iter(values))] if len(values)==1 and count>=4 and families>=2 else '')
        support.append(count);diversity.append(families)
    result['clustering_proposal']=suggested;result['configuration_support']=support;result['feature_family_support']=diversity
    result['changed_candidate']=result.clustering_proposal.ne('')&result.clustering_proposal.ne(result.published_candidate_class)
    result.to_csv(NOTES/'table-csv/bounded_overclustering_proposals.csv',index=False)
    report={'status':'completed_diagnostic_only','manual_train_anchors':14,'test_rows_in_fit':0,'train_rows':1906,
            'configurations':history,'rows_with_stable_proposal':int(result.clustering_proposal.ne('').sum()),
            'rows_disagreeing_with_published_candidate':int(result.changed_candidate.sum()),
            'new_labels_published':0,'semantic_accuracy':None,'source_candidate_sha256':sha(source),
            'interpretation':'finer clusters do not provide missing category meaning; unsupported/mixed clusters are not automatically named from archive labels'}
    assert sha(source)==fixed['prediction_sha256'];dump(NOTES/'json/bounded_overclustering_result.json',report)
    print(json.dumps({k:v for k,v in report.items() if k!='configurations'},indent=2))


if __name__=='__main__':main()

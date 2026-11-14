"""Train-only group-held-out visual clusters and automatic category propagation.

The reference archive and test are not read. Cluster names arise only from
automatic train consensus plus the same 34 lawful human seed sources.
"""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];NOTES=ROOT/'docs/agent-notes'
CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def main():
    begin=time.time();plan={'method':'group_heldout_visual_subclusters_with_train_only_consensus_v1',
        'features':['DINOv3CLS','DINOv3CLS_plus_ConvNeXt'],
        'unsupervised':'PCA64 and MiniBatchKMeans fit on nonquery train groups only',
        'cluster_counts':[64,128,256],
        'category_evidence':'pre-human automatic anchors weight vote-count/4, 34 lawful manual seeds weight8',
        'prediction':'query nearest centroid; Dirichlet0.1 class smoothing; no per-image case rules',
        'test_rows':0,'review_archive_read':False,'manual_unique_sources':34,
        'guard':'34 source-held-out matches>=30, macro>=.9, plastic recall>=.8',
        'limitation':'automatic categories noisy, 34 reused development, cluster purity not ground truth'}
    (NOTES/'json/cluster_semantic_material_probe_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    m=pd.read_csv(ROOT/'data/processed/manifest.csv');f=m[(m.split=='train')&(~m.is_crop)&m.training_eligible].sort_values('parent_id').reset_index(drop=True);keys=f.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);human=np.zeros(n,bool);human[pos]=True
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy();auto=np.array([CLASSES.index(c) for c in a.automatic_class]);auto[pos]=gold;fitmask=a.automatic_anchor.to_numpy(bool)|human;weight=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.);weight[pos]=8.
    dz=np.load(NOTES/'.runtime/dinov3_base_train_features.npz');cz=np.load(NOTES/'.runtime/convnext_train_features.npz');assert np.array_equal(dz['keys'],keys) and np.array_equal(cz['keys'],keys);d=unit(dz['features'][:,:768]);c=unit(cz['features']);banks={'DINOv3CLS':d,'DINOv3CLS_ConvNeXt':unit(np.c_[d,c])}
    rows=[];saved={}
    with threadpool_limits(limits=4):
      for bank,x in banks.items():
        predictions={k:np.full(len(pos),-1,int) for k in plan['cluster_counts']}
        confidences={k:np.zeros(len(pos)) for k in plan['cluster_counts']}
        for fold in range(5):
          tr=np.flatnonzero(folds!=fold);q=np.flatnonzero(folds[pos]==fold);fit=tr[fitmask[tr]];assert not set(groups[fit])&set(groups[pos[q]])
          pca=PCA(n_components=64,svd_solver='randomized',random_state=20261108).fit(x[tr]);xt=unit(pca.transform(x[tr]));xq=unit(pca.transform(x[pos[q]]));xfit=xt[np.searchsorted(tr,fit)]
          for k in plan['cluster_counts']:
            km=MiniBatchKMeans(n_clusters=k,n_init=3,batch_size=256,max_iter=100,random_state=20261108).fit(xt)
            cluster=km.predict(xfit);votes=np.zeros((k,8),float)+.1
            for i,cat,w in zip(cluster,auto[fit],weight[fit]):votes[i,cat]+=w
            posterior=votes/votes.sum(1,keepdims=True);assign=km.predict(xq);probs=posterior[assign]
            predictions[k][q]=probs.argmax(1);confidences[k][q]=probs.max(1)
        for k in plan['cluster_counts']:
          pred=predictions[k];assert (pred>=0).all();matches=int((pred==gold).sum());macro=float(balanced_accuracy_score(gold,pred));plastic=float((pred[gold==4]==4).mean());rows.append({'bank':bank,'clusters':k,'matches':matches,'macro':macro,'plastic_recall':plastic,'mean_confidence':float(confidences[k].mean()),'guard':matches>=30 and macro>=.9 and plastic>=.8});print(rows[-1],round(time.time()-begin),flush=True)
          saved[f'{bank}_{k}_prediction']=pred;saved[f'{bank}_{k}_confidence']=confidences[k]
    np.savez_compressed(NOTES/'.runtime/cluster_semantic_material_probe_oof.npz',keys=s.key.to_numpy(str),gold=gold,**saved)
    (NOTES/'json/cluster_semantic_material_probe_result.json').write_text(json.dumps({'history':rows,'test_rows':0,'review_archive_read':False},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()

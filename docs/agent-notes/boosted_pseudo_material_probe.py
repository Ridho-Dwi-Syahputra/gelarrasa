"""Train-only nonlinear classifier on pre-human automatic anchors plus 34 seeds.

All fitted transforms and classifier trees exclude each query geometry group.
Only a diagnostic of lawful train seeds is produced; no active labels change.
"""
import json,time
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];NOTES=ROOT/'docs/agent-notes'
CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
def main():
    t=time.time();plan={'method':'group_heldout_nonlinear_pseudo_material_classifier_v1',
        'targets':'pre-human automatic consensus anchors; 34 genuine manual seeds override own auto label',
        'test_rows':0,'review_archive_used':False,'manual_unique_sources':34,
        'banks':['DINOv3CLS','DINOv3CLS+ConvNeXt','DINOv3ObjectMean'],
        'transform':'PCA32 fit only nonquery TRAIN + frozen YOLO 28 object statistics; no query in scaler',
        'classifier':'HistGradientBoosting max_iter120 max_leaf_nodes15 min_samples_leaf20 l2=1 learning_rate.06 random_state20261108',
        'weights':'automatic source vote count/4 floor.25; human8',
        'fold':'5 geometry groups; automatic pseudo sources in validation group excluded',
        'screen':'all34 OOF >=30 correct and macro>=.9, plastic recall>=.8',
        'limitation':'automatic categories noisy, 34 already reused for development, no independent semantic accuracy'}
    (NOTES/'json/boosted_pseudo_material_probe_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    m=pd.read_csv(ROOT/'data/processed/manifest.csv');f=m[(m.split=='train')&(~m.is_crop)&m.training_eligible].sort_values('parent_id').reset_index(drop=True);keys=f.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);human=np.zeros(n,bool);human[pos]=True
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy();target=np.array([CLASSES.index(c) for c in a.automatic_class]);target[pos]=gold;fitmask=a.automatic_anchor.to_numpy(bool)|human;weight=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.);weight[pos]=8.
    dino=np.load(NOTES/'.runtime/dinov3_base_train_features.npz');cnn=np.load(NOTES/'.runtime/convnext_train_features.npz');det=np.load(NOTES/'.runtime/litter_object_features.npz');assert all(np.array_equal(z['keys'],keys) for z in [dino,cnn,det])
    d=unit(dino['features'][:,:768]);c=unit(cnn['features']);o=unit(dino['features'][:,1536:2304]);y=np.sqrt(np.maximum(det['detection_stats'],0));banks={'CLS':d,'CLS_CNN':unit(np.c_[d,c]),'object':o};history=[]
    with threadpool_limits(limits=4):
      for name,x in banks.items():
        oof=np.full(len(pos),-1,int)
        for fold in range(5):
          train=np.flatnonzero(folds!=fold);fit=train[fitmask[train]];q=np.flatnonzero(folds[pos]==fold)
          assert not set(groups[fit])&set(groups[pos[q]])
          pca=PCA(n_components=32,svd_solver='randomized',random_state=20261108).fit(x[train]);xx=np.c_[pca.transform(x),y]
          model=HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,min_samples_leaf=20,l2_regularization=1,learning_rate=.06,random_state=20261108)
          model.fit(xx[fit],target[fit],sample_weight=weight[fit]);oof[q]=model.predict(xx[pos[q]])
        assert (oof>=0).all();matches=int((oof==gold).sum());macro=float(balanced_accuracy_score(gold,oof));plastic=float((oof[gold==4]==4).mean());history.append({'bank':name,'matches':matches,'macro':macro,'plastic_recall':plastic,'screen_passed':matches>=30 and macro>=.9 and plastic>=.8});print(name,history[-1],round(time.time()-t),flush=True)
    (NOTES/'json/boosted_pseudo_material_probe_result.json').write_text(json.dumps({'history':history,'source_groups':5,'test_rows':0},indent=2)+'\n',encoding='utf-8')

if __name__=='__main__':main()

"""Check whether source wrapper latent features separate the 34 lawful train labels.

No review/archive data or competition test data. This is diagnostic only unless
the source-held-out screen passes; no category file is modified here.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parents[2]
NOTES = ROOT/'docs/agent-notes'

def unit(x):
    return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def main():
    plan={'method':'source_wrapper_embedding_binary_plastic_screen',
          'references':'same34trainhumananchorsonly','external_images_loaded':False,
          'test_rows':0,'review_archive_read':False,
          'features':'frozen_dense6_512_from_pinned_nonVLM_source13_model',
          'fold':'five fixed geometry source groups; fit normalization and SVM on outer train only',
          'fixed_models':['linear_C1_balanced','rbf_gamma2_C1_balanced','rbf_gamma4_C1_balanced'],
          'screen':'OOF plastic-vs-other AUC>=.85 AND balancedaccuracy>=.8 at margin0',
          'limitation':'34 repeatedly used anchors; source training data overlap unknown'}
    (NOTES/'json/wrapper_embedding_anchor_cv_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    z=np.load(NOTES/'.runtime/wrapper_source_probe_34.npz');seeds=pd.read_csv(ROOT/'data/processed/manual_seeds.csv')
    assert np.array_equal(z['keys'],seeds.key.to_numpy(str)) and len(seeds)==34
    groups=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[seeds.key]
    fold=groups.fold.to_numpy();y=seeds.assigned_class.eq('plastic').to_numpy().astype(int)
    x=unit(z['embeddings'].astype('float64'));rows=[]
    for name,gamma in [('linear',None),('rbf2',2.),('rbf4',4.)]:
        oof=np.full(len(y),np.nan)
        for f in np.unique(fold):
            tr=fold!=f;va=~tr; assert y[tr].min()==0 and y[tr].max()==1
            center=x[tr].mean(0);a=unit(x[tr]-center);b=unit(x[va]-center)
            k=a@a.T;q=b@a.T
            if gamma is not None:k=np.exp(-2*gamma*np.maximum(1-k,0));q=np.exp(-2*gamma*np.maximum(1-q,0))
            m=SVC(kernel='precomputed',C=1.,class_weight='balanced').fit(k,y[tr])
            oof[va]=m.decision_function(q)
        auc=float(roc_auc_score(y,oof));bal=float(balanced_accuracy_score(y,oof>0));
        rows.append({'model':name,'auc':auc,'balanced_accuracy':bal,'plastic_recall':float((oof[y==1]>0).mean()),
                     'other_specificity':float((oof[y==0]<=0).mean()),'screen_passed':auc>=.85 and bal>=.8})
    result={'models':rows,'screen_passed':any(r['screen_passed'] for r in rows),'fit_sources':34,'test_rows':0}
    (NOTES/'json/wrapper_embedding_anchor_cv_result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()

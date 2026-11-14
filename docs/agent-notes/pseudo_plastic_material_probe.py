"""Binary train-only plastic model from pre-human automatic anchors and 34 seeds.

All fitting is restricted to other geometry source groups. The automatic
source table was produced before human seeds and contains no review archive.
This script screens a hypothesis; it does not modify active categories.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score,roc_auc_score
from sklearn.svm import SVC
from threadpoolctl import threadpool_limits

ROOT=Path(__file__).resolve().parents[2];NOTES=ROOT/'docs/agent-notes'
CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']

def unit(x):return x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)

def main():
    plan={'method':'prehuman_automatic_anchor_plus34_source_heldout_plastic_binary_v1',
          'auto_targets':'automatic_full_seed_targets.csv generated before human corrections',
          'auto_positive':'anchor AND automatic plastic AND pretrained detector dominant plastic >=.65 margin>=.05',
          'auto_negative':'anchor AND automatic nonplastic AND pretrained detector not dominant plastic',
          'fit':'other geometry groups only; manual seed overrides auto label, weight 8 vs automatic weight1',
          'features':['DINOv3CLS','DINOv3CLS_plus_YOLOstats','DINOv3ObjectMean_plus_YOLOstats'],
          'models':['linear_C1','rbf_gamma2_C1'], 'margins':[0,.2,.4],
          'guard':'when routed over baseline, >=30/34 matches, macro>=.9, >=2 human routes, 100% routed human precision',
          'test_rows':0,'archive_review_rows':0,'manual_sources':34,
          'limitation':'automatic targets may contain errors, repeated34 development; no independent semantic accuracy'}
    (NOTES/'json/pseudo_plastic_material_probe_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    m=pd.read_csv(ROOT/'data/processed/manifest.csv');f=m[(m.split=='train')&(~m.is_crop)&m.training_eligible].sort_values('parent_id').reset_index(drop=True);keys=f.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);human=np.zeros(n,bool);human[pos]=True
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=g.group.to_numpy(),g.fold.to_numpy()
    d=np.load(NOTES/'.runtime/litter_object_features.npz');v=np.load(NOTES/'.runtime/dinov3_base_train_features.npz');cnn=np.load(NOTES/'.runtime/convnext_train_features.npz');assert all(np.array_equal(z['keys'],keys) for z in [d,v,cnn]);det=d['detection_stats'][:,:5];dominant=(det[:,0]>=.65)&((det[:,0]-det[:,1:].max(1))>=.05)
    auto=a.automatic_anchor.to_numpy(bool);plastic=a.automatic_class.eq('plastic').to_numpy();core=auto&((plastic&dominant)|(~plastic&~dominant));core[human]=False;labels=plastic.astype(int);labels[pos]=(gold==4).astype(int)
    assert core.sum()>500 and (core&plastic).sum()>100
    cls=unit(v['features'][:,:768]);obj=unit(v['features'][:,1536:2304]);ys=unit(np.sqrt(np.maximum(d['detection_stats'],0)))
    banks={'CLS':cls,'CLS_YOLO':unit(np.c_[cls,ys]),'object_YOLO':unit(np.c_[obj,ys])}
    p=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz');aug=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz');vo=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz');info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);base=vo['probabilities'][wi].argmax(1);ap,am,ar=aug['oof_predictions'],aug['oof_margins'],aug['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac;acov=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45);base=np.where(acov,ac,base);rp=p['probabilities'];rc=rp[0].argmax(1);rs=np.min([q[np.arange(n),rc] for q in rp],0);base=np.where((rs>=.6)&~(acov&(ac!=rc)),rc,base);assert int((base[pos]==gold).sum())==28
    rows=[];outputs={}
    with threadpool_limits(limits=3):
      for bank,x in banks.items():
        for kind in ['linear','rbf2']:
          margin=np.full(len(pos),np.nan)
          for fold in range(5):
            query_seed=np.flatnonzero(folds[pos]==fold);fit=np.flatnonzero((folds!=fold)&(core|human));assert not set(groups[fit])&set(groups[pos[query_seed]])
            centers=x[folds!=fold].mean(0);xt=unit(x[fit]-centers);xq=unit(x[pos[query_seed]]-centers);kr=xt@xt.T;kq=xq@xt.T
            if kind=='rbf2':kr=np.exp(-4*np.maximum(1-kr,0));kq=np.exp(-4*np.maximum(1-kq,0))
            weight=np.where(human[fit],8.,1.);model=SVC(kernel='precomputed',C=1.,class_weight='balanced').fit(kr,labels[fit],sample_weight=weight)
            margin[query_seed]=model.decision_function(kq)
          assert np.isfinite(margin).all();outputs[f'{bank}_{kind}']=margin
          auc=float(roc_auc_score(gold==4,margin))
          for threshold in [0,.2,.4]:
            routed=(margin>=threshold)&(base[pos]!=4);pred=np.where(routed,4,base[pos]);matches=int((pred==gold).sum());macro=float(balanced_accuracy_score(gold,pred));covered=int(routed.sum());correct=int((routed&(pred==gold)).sum())
            rows.append({'bank':bank,'model':kind,'threshold':threshold,'auc':auc,'matches':matches,'macro':macro,'covered':covered,'correct':correct,'guard':matches>=30 and macro>=.9-1e-12 and covered>=2 and correct==covered})
    np.savez_compressed(NOTES/'.runtime/pseudo_plastic_material_probe_oof.npz',keys=s.key.to_numpy(str),gold=gold,baseline=base[pos],**outputs)
    out=pd.DataFrame(rows);out.to_csv(NOTES/'table-csv/pseudo_plastic_material_probe_development.csv',index=False);summary={'status':'screen_passed' if out.guard.any() else 'screen_failed','auto_positive':int((core&plastic).sum()),'auto_negative':int((core&~plastic).sum()),'best':out.sort_values(['matches','macro','correct'],ascending=False).iloc[0].to_dict(),'passing':int(out.guard.sum())};(NOTES/'json/pseudo_plastic_material_probe_result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary,indent=2));print(out.sort_values(['matches','macro'],ascending=False).head(10).to_string(index=False))

if __name__=='__main__':main()

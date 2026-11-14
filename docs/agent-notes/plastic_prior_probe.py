"""Train-only source-held-out prior probe for flexible plastic packaging.

Uses existing out-of-fold train probabilities. The review/archive table and all
test images are excluded. No publication occurs in this diagnostic script.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[2]
NOTES=ROOT/'docs/agent-notes'
CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']

def main():
    plan={'method':'class_prior_binary_consensus_probe_v1','inputs':'precomputed train-only OOF residual two banks',
          'manual_sources':34,'test_used':False,'review_archive_used':False,
          'hypothesis':'increased recall of flexible plastic materials without changing nonplastic consensus',
          'boost_log_odds':[0,.1,.2,.3,.4,.5,.6,.8,1.0,1.2,1.5,2.0],
          'minimum_source_probability':[.4,.5,.6,.7],
          'guard':'whole34 match>28 and macro>=.85; no active changes until further freeze/publisher',
          'limitation':'same34 reused in earlier development, not independent estimate'}
    (NOTES/'json/plastic_prior_probe_plan.json').write_text(json.dumps(plan,indent=2)+'\n',encoding='utf-8')
    m=pd.read_csv(ROOT/'data/processed/manifest.csv');frame=m[(m.split=='train')&(~m.is_crop)&m.training_eligible].sort_values('parent_id').reset_index(drop=True)
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');keys=frame.parent_key.to_numpy(str)
    lookup={k:i for i,k in enumerate(keys)};pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class]);assert len(pos)==34
    z=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz');assert np.array_equal(keys,z['keys'])
    rp=z['probabilities'];assert rp.shape==(2,1904,8)
    v=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz');assert np.array_equal(keys,v['keys'])
    info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);base=v['probabilities'][wi].argmax(1)
    aug=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz');ap,am,ar=aug['oof_predictions'],aug['oof_margins'],aug['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac;covered=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45)
    base=np.where(covered,ac,base);rc=rp[0].argmax(1);rs=np.min([p[np.arange(len(keys)),rc] for p in rp],0);base=np.where((rs>=.6)&~(covered&(ac!=rc)),rc,base)
    assert int((base[pos]==gold).sum())==28
    history=[]
    for boost in plan['boost_log_odds']:
        q=rp.copy().astype('float64');q[:,:,4]*=np.exp(boost);q/=q.sum(2,keepdims=True)
        both=(q.argmax(2)==4).all(0);score=q[:,:,4].min(0)
        for threshold in plan['minimum_source_probability']:
            routed=both&(score>=threshold)&(base!=4)
            pred=np.where(routed,4,base);matches=int((pred[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,pred[pos]));cov=int(routed[pos].sum());correct=int((routed[pos]&(pred[pos]==gold)).sum())
            history.append({'boost':boost,'threshold':threshold,'matches':matches,'macro':macro,'human_covered':cov,'human_correct':correct,'automatic_proposals':int(routed.sum()),'guard_passed':matches>28 and macro>=.85-1e-12})
    out=pd.DataFrame(history);out.to_csv(NOTES/'table-csv/plastic_prior_probe_development.csv',index=False)
    summary={'status':'screen_passed' if out.guard_passed.any() else 'screen_failed','best':out.sort_values(['matches','macro','human_correct','human_covered'],ascending=False).iloc[0].to_dict(),'passing':int(out.guard_passed.sum()),'test_rows':0,'manual_sources':34}
    (NOTES/'json/plastic_prior_probe_result.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summary,indent=2));print(out.sort_values(['matches','macro'],ascending=False).head(10).to_string(index=False))

if __name__=='__main__':main()

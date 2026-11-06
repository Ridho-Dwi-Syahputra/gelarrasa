"""Fixed structural routing: preserve stable groups, refine material mixtures.

Group diversity is estimated from fit-only CNN consensus, never archive labels
or assistant display mapping. Threshold is frozen before this run's audit.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np,pandas as pd,torch
from stage6_semantic_ontology import CLASSES
from stage7_waste_semantic_training import make_head
from stage15_second_cnn_refinement import load_second_features
from stage16_source_consensus import independent_anchors

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage17_hierarchical_routing_v1'
REPORT=ROOT/'reports/errors/stage17_hierarchical_routing_20261106'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')

def infer_head(path):
    c=torch.load(path,map_location='cpu',weights_only=True)
    keys,x,fkeys,xf=load_second_features(c['feature_branch']);h=make_head(c['kind'],c['feature_dimensions']);h.load_state_dict(c['state_dict']);h.eval()
    with torch.inference_mode():p=h(torch.from_numpy(x)).softmax(-1).numpy();pf=h(torch.from_numpy(xf)).softmax(-1).numpy()
    return keys,p,fkeys,pf

def coarse_groups(path):
    c=torch.load(path,map_location='cpu',weights_only=True);h=torch.nn.Linear(384,8);h.load_state_dict(c['state_dict']);h.eval()
    d=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    f=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    with torch.inference_mode():g=h(torch.from_numpy(d['features'])).argmax(1).numpy();gf=h(torch.from_numpy(f['features'])).argmax(1).numpy()
    return d['keys'],g,f['keys'],gf,c

def predict_saved_model(base,with_flip=False):
    meta=json.loads((base/'routing_model.json').read_text())
    for name,digest in meta['component_sha256'].items():assert sha(ROOT/name)==digest
    keys,a,fkeys,af=infer_head(ROOT/meta['stable_checkpoint']);keysb,b,fkeysb,bf=infer_head(ROOT/meta['refinement_checkpoint'])
    kg,g,kfg,gf,_=coarse_groups(ROOT/meta['coarse_checkpoint'])
    assert np.array_equal(keys,keysb) and np.array_equal(keys,kg) and np.array_equal(fkeys,fkeysb) and np.array_equal(fkeys,kfg)
    route=np.isin(g,meta['refinement_groups']);routef=np.isin(gf,meta['refinement_groups'])
    probability=np.where(route[:,None],b,a);flipped=np.where(routef[:,None],bf,af)
    if with_flip:return keys,probability,fkeys,flipped,route
    return keys,probability

def main():
    started=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    stable='output/groups/stage12_ontology_refinement_v1/selected_semantic_head.pt'
    refinement='output/groups/stage16_source_consensus_v1/selected_semantic_head.pt'
    coarse='output/groups/stage4_self_training_pilot_v1/selected_fit_head.pt'
    keys,groups,_,_,split=coarse_groups(ROOT/coarse);index=dict(zip(keys,range(len(keys))));fi=np.array([index[k] for k in split['fit_keys']]);vi=np.array([index[k] for k in split['heldout_keys']])
    ak,labels,clean,votes,_=independent_anchors();assert np.array_equal(keys,ak)
    records=[];refine=[]
    # Fixed threshold, not a search on archived labels or their mixed-cohort flag.
    for g in range(8):
        mask=(groups[fi]==g)&clean[fi];counts=np.bincount(labels[fi][mask],minlength=8);n=int(counts.sum());purity=float(counts.max()/max(n,1))
        use=n>=20 and purity<.6
        if use:refine.append(g)
        records.append(dict(anonymous_group=g,fit_consensus_anchors=n,dominant_semantic_share=purity,refinement_selected=use,**dict(zip(CLASSES,counts.tolist()))))
    pd.DataFrame(records).to_csv(REPORT/'fit_only_group_diversity.csv',index=False)
    meta=dict(method='fixed_hierarchical_material_routing',stable_checkpoint=stable,refinement_checkpoint=refinement,coarse_checkpoint=coarse,refinement_groups=refine,
        threshold=.6,minimum_consensus_anchors=20,component_sha256={p:sha(ROOT/p) for p in [stable,refinement,coarse]},
        reference_archive_read=False,assistant_display_mapping_read=False,reference_cohort_flag_read=False,reference_targets_or_filters=False,
        researcher_has_seen_previous_private_audits=True,fit_only_routing_statistics=True,semantic_accuracy=None,source_sha256=sha(__file__))
    dump(REPORT/'plan_before_inference.json',meta);dump(OUT/'routing_model.json',meta)
    keys,probability,fkeys,flipped,route=predict_saved_model(OUT,with_flip=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str});assert np.array_equal(keys,p.key.to_numpy(str))
    pred=probability.argmax(1);order=np.sort(probability,axis=1);result=p[['key','split','image_id','pixel_sha256','processed_path']].copy();result['predicted_class']=np.array(CLASSES)[pred]
    result['uncalibrated_probability']=probability.max(1);result['margin']=order[:,-1]-order[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    old=np.load(ROOT/'output/groups/stage7_waste_semantic_v1/named_probabilities.npz',allow_pickle=False)['prior_probabilities']
    result['cnn_prior_disagrees']=pred!=old.argmax(1);result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='fit_only_hierarchical_material_routing_v1';result['refinement_branch_used']=route;result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=keys,probabilities=probability,prior_probabilities=old)
    core=clean[vi];macro=float(np.mean([(pred[vi][core&(labels[vi]==c)]==c).mean() for c in range(8)]))
    lookup=dict(zip(fkeys,range(len(fkeys))));paired=[i for i in vi if keys[i] in lookup];fp=[lookup[keys[i]] for i in paired]
    flip=float((pred[paired]==flipped[fp].argmax(1)).mean())
    selected=dict(candidate='fixed_hierarchical_routing',macro_anchor_agreement=macro,flip=flip,routed_train=int(route[p.split.eq('train')].sum()),routing_groups=refine,
        note='Structural hypothesis, not a new proxy-score winner. Its audit cannot be called ground-truth accuracy.')
    dump(REPORT/'summary.json',dict(status='passed',selected=selected,train_counts=result[result.split.eq('train')].predicted_class.value_counts().to_dict(),train_needs_review=int(result[result.split.eq('train')].needs_review.sum()),mixed_output_names=0,reference_archive_read=False,semantic_accuracy=None,seconds=time.time()-started))
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),model_metadata_sha256=sha(OUT/'routing_model.json'),source_sha256=sha(__file__),protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),selection='fixed structural hypothesis, frozen before own private audit; not proxy-score or official accuracy winner'))
    print(json.dumps(selected,indent=2),flush=True)

if __name__=='__main__':main()

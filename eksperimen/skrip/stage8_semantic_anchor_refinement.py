"""Fit-only anonymous cluster/CNN ontology anchors, then named refinement.

Uses no archive labels or assistant display names. An eight-by-eight semantic
matching supplies a cluster interpretation from non-VLM CNN evidence. Clean
intersection seeds supervise new heads; individual rows are not manually moved.
"""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from threadpoolctl import threadpool_limits
from stage6_semantic_ontology import CLASSES
from stage7_waste_semantic_training import make_head,feature_blocks

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage8_semantic_anchor_v1'
REPORT=ROOT/'reports/errors/stage8_semantic_anchor_20261106'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')

def main():
    started=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    d=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    w=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/original224.npz',allow_pickle=False)
    assert np.array_equal(d['keys'],w['keys']) and np.array_equal(w['keys'],p.key.to_numpy(str));index=dict(zip(p.key,range(len(p))))
    anonymous=torch.load(ROOT/'output/groups/stage4_self_training_pilot_v1/selected_fit_head.pt',map_location='cpu',weights_only=True)
    fit_keys=anonymous['fit_keys'];val_keys=anonymous['heldout_keys'];fi=np.array([index[k] for k in fit_keys]);vi=np.array([index[k] for k in val_keys])
    coarse=torch.nn.Linear(384,8);coarse.load_state_dict(anonymous['state_dict']);coarse.eval()
    with torch.inference_mode():coarse_probability=coarse(torch.from_numpy(d['features'])).softmax(-1).numpy()
    groups=coarse_probability.argmax(1)
    generic=np.load(ROOT/'output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz',allow_pickle=False)
    waste=np.load(ROOT/'output/groups/stage7_waste_semantic_v1/named_probabilities.npz',allow_pickle=False)
    assert np.array_equal(generic['keys'],p.key.to_numpy(str)) and np.array_equal(waste['keys'],p.key.to_numpy(str))
    priors=[generic['prior_probabilities'],waste['prior_probabilities']]
    scores=np.zeros((8,8))
    # Aggregate source evidence on FIT only. No access to display_label_mapping.
    for g in range(8):
        members=fi[groups[fi]==g];assert len(members)>0
        scores[g]=np.max(np.stack([q[members].mean(0) for q in priors]),axis=0)
    left,right=linear_sum_assignment(-scores)
    correspondence=dict(zip(left.tolist(),right.tolist()))
    mapped=np.array([correspondence[int(g)] for g in groups])
    support=np.max(np.stack([q[np.arange(len(p)),mapped] for q in priors]),axis=0)
    any_agrees=np.any(np.stack([q.argmax(1)==mapped for q in priors]),axis=0)
    clean=(support>=.65)&any_agrees&(coarse_probability.max(1)>=.8)
    counts=np.bincount(mapped[fi][clean[fi]],minlength=8);val_clean=clean[vi]
    val_counts=np.bincount(mapped[vi][val_clean],minlength=8)
    assert (counts>=3).all(),counts
    assert (val_counts>0).all(),val_counts
    wf=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/hflip224.npz',allow_pickle=False)
    df=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    assert np.array_equal(wf['keys'],df['keys']);lookup=dict(zip(df['keys'],range(len(df['keys']))))
    tf=np.array([i for i,k in enumerate(fit_keys) if k in lookup]);tv=np.array([i for i,k in enumerate(val_keys) if k in lookup])
    fids=np.array([lookup[fit_keys[i]] for i in tf]);vids=np.array([lookup[val_keys[i]] for i in tv])
    with threadpool_limits(limits=4):
        geometry=feature_blocks(w['features'],d['features'],'hybrid')
        sim=geometry[fi]@geometry[fi].T;np.fill_diagonal(sim,-np.inf)
        nn=np.argsort(-sim,axis=1,kind='stable')[:,:10]
        nv=np.argsort(-(geometry[vi]@geometry[fi].T),axis=1,kind='stable')[:,:10]
    plan=dict(source_sha256=sha(__file__),task='semantic anchoring of fit-only anonymous clusters using CNN ontology evidence, then refinement',
        class_order=CLASSES,fit_keys=fit_keys,heldout_keys=val_keys,correspondence=correspondence,
        mapping_policy='Hungarian maximum aggregated source likelihood, one name per coarse group; does not impose predicted class counts.',
        aggregate_scores=scores.tolist(),seed_rule='coarse probability>=.8; at least one CNN source argmax agrees; its class support>=.65',
        fit_seed_counts=dict(zip(CLASSES,counts.tolist())),heldout_core_counts=dict(zip(CLASSES,val_counts.tolist())),
        anonymous_head_sha256=sha(ROOT/'output/groups/stage4_self_training_pilot_v1/selected_fit_head.pt'),
        source_priors=[sha(ROOT/'output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz'),sha(ROOT/'output/groups/stage7_waste_semantic_v1/named_probabilities.npz')],
        configurations=['waste','dino','hybrid'],heads=['linear','mlp'],epochs=180,lr=.003,
        selection='.5 macro heldout anchor agreement +.25 flip +.25 fixed hybrid-neighbor agreement; all8 occupied min5.',
        reference_archive_read=False,assistant_display_mapping_read=False,test_used_in_fit=False,
        researcher_has_seen_previous_private_audits=True,reference_targets_or_per_image_archive_filters_used=False,
        semantic_accuracy=None,pretraining_overlap_unknown=True)
    dump(REPORT/'plan_before_fit.json',plan)
    pd.DataFrame(scores,index=range(8),columns=CLASSES).to_csv(REPORT/'CNN_semantic_matching_scores.csv',index_label='anonymous_group')
    seeds=np.flatnonzero(clean[fi]);y=torch.from_numpy(mapped[fi[seeds]])
    table=p.iloc[fi[seeds]][['key','image_id']].copy();table['algorithmic_seed_class']=np.array(CLASSES)[y.numpy()];table.to_csv(OUT/'clean_fit_seeds.csv',index=False)
    weights=torch.from_numpy((len(seeds)/(8*np.bincount(y.numpy(),minlength=8))).astype('float32'))
    history=[];best=None
    def evaluate(head,branch):
        xx=feature_blocks(w['features'],d['features'],branch);fx=feature_blocks(wf['features'],df['features'],branch)
        head.eval()
        with torch.inference_mode():
            tr=head(torch.from_numpy(xx[fi])).argmax(1).numpy();v=head(torch.from_numpy(xx[vi])).argmax(1).numpy()
            flipped=head(torch.from_numpy(fx[vids])).argmax(1).numpy()
        macro=float(np.mean([(v[val_clean&(mapped[vi]==c)]==c).mean() for c in range(8)]))
        flip=float((v[tv]==flipped).mean());neighbor=float((v[:,None]==tr[nv]).mean());n=np.bincount(v,minlength=8)
        return dict(score=.5*macro+.25*flip+.25*neighbor,macro_anchor_agreement=macro,flip=flip,neighbor=neighbor,
            occupied=int((n>0).sum()),min_class=int(n.min()))
    previous=torch.load(ROOT/'output/groups/stage7_waste_semantic_v1/selected_semantic_head.pt',map_location='cpu',weights_only=True)
    head=make_head(previous['kind'],previous['feature_dimensions']);head.load_state_dict(previous['state_dict'])
    metric=evaluate(head,previous['feature_branch']);history.append(dict(candidate='previous_waste_student',epoch=previous['selected']['epoch'],**metric))
    if metric['occupied']==8 and metric['min_class']>=5:best=dict(candidate='previous_waste_student',kind=previous['kind'],branch=previous['feature_branch'],dimension=previous['feature_dimensions'],epoch=previous['selected']['epoch'],**metric,state=copy.deepcopy(head.state_dict()))
    for branch in ['waste','dino','hybrid']:
        xx=feature_blocks(w['features'],d['features'],branch);fx=feature_blocks(wf['features'],df['features'],branch)
        xt=torch.from_numpy(xx[fi]);xf=torch.from_numpy(fx[fids])
        for kind in ['linear','mlp']:
            torch.manual_seed(20261106);head=make_head(kind,xx.shape[1]);optimizer=torch.optim.Adam(head.parameters(),lr=.003,weight_decay=.0001)
            for epoch in range(1,181):
                head.train();logits=head(xt);q=logits.softmax(-1)
                ce=F.cross_entropy(logits[seeds],y,weight=weights)
                graph=-(q[:,None,:]*q[nn]).sum(-1).clamp_min(1e-8).log().mean()
                alternative=head(xf).softmax(-1);original=q[tf];mean=(original+alternative)/2
                js=.5*((original*(original.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean()+
                       (alternative*(alternative.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean())
                loss=ce+.03*graph+.2*js;assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step()
                if epoch%10==0:
                    metric=evaluate(head,branch);name=f'anchors_{branch}_{kind}';history.append(dict(candidate=name,epoch=epoch,**metric))
                    if metric['occupied']==8 and metric['min_class']>=5 and (best is None or metric['score']>best['score']+1e-7):best=dict(candidate=name,kind=kind,branch=branch,dimension=xx.shape[1],epoch=epoch,**metric,state=copy.deepcopy(head.state_dict()))
            print(f'anchor student {branch} {kind} complete',flush=True)
    assert best is not None
    state=best.pop('state');head=make_head(best['kind'],best['dimension']);head.load_state_dict(state);head.eval()
    xx=feature_blocks(w['features'],d['features'],best['branch'])
    with torch.inference_mode():probability=head(torch.from_numpy(xx)).softmax(-1).numpy()
    torch.save(dict(state_dict=state,kind=best['kind'],feature_dimensions=best['dimension'],feature_branch=best['branch'],class_order=CLASSES,
        selected=best,fit_keys=fit_keys,heldout_keys=val_keys,plan=plan),OUT/'selected_semantic_head.pt')
    pred=probability.argmax(1);order=np.sort(probability,axis=1)
    result=p[['key','split','image_id','pixel_sha256','processed_path']].copy()
    result['predicted_class']=np.array(CLASSES)[pred];result['uncalibrated_probability']=probability.max(1);result['margin']=order[:,-1]-order[:,-2]
    result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    result['cnn_prior_disagrees']=pred!=waste['prior_probabilities'].argmax(1)
    result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='NonVLM_CNN_fit_only_cluster_ontology_anchors_then_refinement_v1'
    result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=p.key.to_numpy(str),probabilities=probability,prior_probabilities=waste['prior_probabilities'])
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    summary=dict(status='passed',heads_trained=6,fit_images=len(fi),heldout_images=len(vi),seed_fit_images=len(seeds),selected=best,
        train_counts=result[result.split.eq('train')].predicted_class.value_counts().to_dict(),train_needs_review=int(result[result.split.eq('train')].needs_review.sum()),
        mixed_output_names=0,reference_archive_read=False,assistant_display_mapping_read=False,semantic_accuracy=None,seconds=time.time()-started)
    dump(REPORT/'summary.json',summary)
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),
        checkpoint_sha256=sha(OUT/'selected_semantic_head.pt'),summary_sha256=sha(REPORT/'summary.json'),source_sha256=sha(__file__),protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),
        selection='same heldout model-anchor/view/neighbor criterion before this candidate reference audit',checkpoint_sha256=sha(OUT/'selected_semantic_head.pt')))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

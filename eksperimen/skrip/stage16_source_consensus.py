"""Independent CNN consensus anchors, without any coarse-cluster targets.

Source class names come from pinned published metadata. Archive annotations
and assistant display mapping are never read. No fixed class-count quotas.
"""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np,pandas as pd,torch
from torch.nn import functional as F
from threadpoolctl import threadpool_limits
from stage6_semantic_ontology import CLASSES
from stage7_waste_semantic_training import make_head
from stage15_second_cnn_refinement import load_second_features

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage16_source_consensus_v1'
REPORT=ROOT/'reports/errors/stage16_source_consensus_20261106'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')

def independent_anchors():
    generic=np.load(ROOT/'output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz',allow_pickle=False)
    keys=generic['keys'];sources=[generic['prior_probabilities']];coverage=[np.ones(len(keys),dtype='float32')]
    for stage,path in [(7,'stage7_waste_cnn_v1'),(14,'stage14_second_waste_cnn_v1')]:
        z=np.load(ROOT/'output/embeddings'/path/'original224.npz',allow_pickle=False);assert np.array_equal(keys,z['keys'])
        folder='stage7_waste_cnn_20261106' if stage==7 else 'stage14_second_waste_cnn_20261106'
        names=json.loads((ROOT/'reports/errors'/folder/'checkpoint_and_ontology.json').read_text())['source_class_order']
        mapping={'battery':'electronic','biological':'foodwaste','brown-glass':'glass','green-glass':'glass','white-glass':'glass','glass':'glass','clothes':'textile','shoes':'textile',**{c:c for c in ['cardboard','paper','metal','plastic']}}
        evidence=np.zeros((len(keys),8),dtype='float32')
        for j,name in enumerate(names):
            if name in mapping:evidence[:,CLASSES.index(mapping[name])]+=z['probabilities'][:,j]
        cov=evidence.sum(1);sources.append(evidence/np.maximum(cov[:,None],1e-8));coverage.append(cov)
    expert=np.load(ROOT/'output/embeddings/stage11_ontology_cnn_v1/original224.npz',allow_pickle=False);assert np.array_equal(keys,expert['keys'])
    sources.append(expert['conditional']);coverage.append(expert['coverage'])
    sources=np.stack(sources);coverage=np.stack(coverage)
    thresholds=np.array([.0,.6,.6,.2])[:,None]
    reliable=(sources.max(2)>=.8)&(coverage>=thresholds)
    votes=np.zeros((len(keys),8),dtype='float32')
    for j in range(4):votes[np.flatnonzero(reliable[j]),sources[j,reliable[j]].argmax(1)]+=1
    labels=votes.argmax(1);ranked=np.sort(votes,axis=1)
    clean=(ranked[:,-1]>=2)&(ranked[:,-1]>ranked[:,-2])
    return keys,labels,clean,votes,sources

def main():
    started=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str});index=dict(zip(p.key,range(len(p))))
    split=json.loads((ROOT/'reports/errors/stage12_ontology_refinement_20261106/plan_before_fit.json').read_text())
    fit=split['fit_keys'];held=split['heldout_keys'];fi=np.array([index[k] for k in fit]);vi=np.array([index[k] for k in held])
    keys,labels,clean,votes,sources=independent_anchors();assert np.array_equal(keys,p.key.to_numpy(str))
    seed=np.flatnonzero(clean[fi]);core=clean[vi]
    counts=np.bincount(labels[fi[seed]],minlength=8);val_counts=np.bincount(labels[vi][core],minlength=8)
    assert (counts>=3).all() and (val_counts>0).all(),(counts,val_counts)
    branches=['dino','ontology_all','second_dino','second_all'];features={b:load_second_features(b) for b in branches}
    geometry=features['ontology_all'][1]
    with threadpool_limits(limits=4):
        sim=geometry[fi]@geometry[fi].T;np.fill_diagonal(sim,-np.inf);nn=np.argsort(-sim,axis=1,kind='stable')[:,:10]
        nv=np.argsort(-(geometry[vi]@geometry[fi].T),axis=1,kind='stable')[:,:10]
    flip_keys=features['ontology_all'][2];lookup=dict(zip(flip_keys,range(len(flip_keys))))
    tf=np.array([i for i,k in enumerate(fit) if k in lookup]);tv=np.array([i for i,k in enumerate(held) if k in lookup]);fids=np.array([lookup[fit[i]] for i in tf]);vids=np.array([lookup[held[i]] for i in tv])
    plan=dict(source_sha256=sha(__file__),fit_keys=fit,heldout_keys=held,fit_seed_counts=dict(zip(CLASSES,counts.tolist())),heldout_core_counts=dict(zip(CLASSES,val_counts.tolist())),
        anchor_policy='at least2 high-confidence independent image-only source families agree, no tied vote; generic/ResNet-waste/EfficientNet-waste/ImageNet22k-ontology; explicit raw coverage guards',
        coarse_cluster_targets_used=False,assistant_display_mapping_read=False,reference_archive_read=False,reference_targets_or_filters=False,
        researcher_has_seen_previous_private_audits=True,heldout_or_test_used_in_fit=False,semantic_accuracy=None,
        branches=branches,heads=['linear','mlp'],epochs=180,lr=.003,selection='.75 macro independent-source anchor agreement +.25 flip; neighbor logged only, not used to force coarse geometry',
        source_provenance=[sha(ROOT/'reports/errors'/v/'checkpoint_and_ontology.json') for v in ['stage7_waste_cnn_20261106','stage11_ontology_cnn_20261106','stage14_second_waste_cnn_20261106']])
    dump(REPORT/'plan_before_fit.json',plan)
    anchors=p.iloc[fi[seed]][['key']].copy();anchors['algorithmic_seed_class']=np.array(CLASSES)[labels[fi[seed]]];anchors.to_csv(OUT/'clean_fit_seeds.csv',index=False)
    y=torch.from_numpy(labels[fi[seed]]);weight=torch.from_numpy((len(seed)/(8*counts)).astype('float32'));history=[];best=None
    def metric(head,branch):
        _,x,_,xf=features[branch];head.eval()
        with torch.inference_mode():
            pred=head(torch.from_numpy(x)).argmax(1).numpy();flipped=head(torch.from_numpy(xf[vids])).argmax(1).numpy()
        v=pred[vi];macro=float(np.mean([(v[core&(labels[vi]==c)]==c).mean() for c in range(8)]));flip=float((v[tv]==flipped).mean());n=np.bincount(v,minlength=8)
        return dict(score=.75*macro+.25*flip,macro_anchor_agreement=macro,flip=flip,neighbor=float((v[:,None]==pred[fi][nv]).mean()),occupied=int((n>0).sum()),min_class=int(n.min()))
    previous=torch.load(ROOT/'output/groups/stage12_ontology_refinement_v1/selected_semantic_head.pt',map_location='cpu',weights_only=True)
    head=make_head(previous['kind'],previous['feature_dimensions']);head.load_state_dict(previous['state_dict']);m=metric(head,previous['feature_branch'])
    history.append(dict(candidate='previous_control',epoch=previous['selected']['epoch'],**m))
    best=dict(candidate='previous_control',kind=previous['kind'],branch=previous['feature_branch'],dimension=previous['feature_dimensions'],epoch=previous['selected']['epoch'],**m,state=copy.deepcopy(head.state_dict()))
    for branch in branches:
        _,x,_,xf=features[branch];xt=torch.from_numpy(x[fi]);xflip=torch.from_numpy(xf[fids])
        for kind in ['linear','mlp']:
            torch.manual_seed(20261106);head=make_head(kind,x.shape[1]);optimizer=torch.optim.Adam(head.parameters(),lr=.003,weight_decay=.0001)
            for epoch in range(1,181):
                head.train();logits=head(xt);q=logits.softmax(-1);ce=F.cross_entropy(logits[seed],y,weight=weight)
                alternative=head(xflip).softmax(-1);original=q[tf];mean=(original+alternative)/2
                js=.5*((original*(original.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean()+(alternative*(alternative.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean())
                # Neighbor loss deliberately small: lookalike materials need not share a class.
                graph=-(q[:,None,:]*q[nn]).sum(-1).clamp_min(1e-8).log().mean();loss=ce+.005*graph+.2*js
                assert torch.isfinite(loss);optimizer.zero_grad();loss.backward();optimizer.step()
                if epoch%10==0:
                    m=metric(head,branch);name=f'independent_{branch}_{kind}';history.append(dict(candidate=name,epoch=epoch,**m))
                    if m['occupied']==8 and m['min_class']>=5 and m['score']>best['score']+1e-7:best=dict(candidate=name,kind=kind,branch=branch,dimension=x.shape[1],epoch=epoch,**m,state=copy.deepcopy(head.state_dict()))
            print(f'independent anchors {branch} {kind} complete',flush=True)
    state=best.pop('state');head=make_head(best['kind'],best['dimension']);head.load_state_dict(state);head.eval();x=features[best['branch']][1]
    with torch.inference_mode():probability=head(torch.from_numpy(x)).softmax(-1).numpy()
    torch.save(dict(state_dict=state,kind=best['kind'],feature_dimensions=best['dimension'],feature_branch=best['branch'],feature_loader='second_cnn_v1',class_order=CLASSES,selected=best,fit_keys=fit,heldout_keys=held,plan=plan),OUT/'selected_semantic_head.pt')
    pred=probability.argmax(1);order=np.sort(probability,axis=1);result=p[['key','split','image_id','pixel_sha256','processed_path']].copy();result['predicted_class']=np.array(CLASSES)[pred]
    result['uncalibrated_probability']=probability.max(1);result['margin']=order[:,-1]-order[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    old_prior=np.load(ROOT/'output/groups/stage7_waste_semantic_v1/named_probabilities.npz',allow_pickle=False)['prior_probabilities']
    result['cnn_prior_disagrees']=pred!=old_prior.argmax(1);result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='independent_CNN_consensus_anchors_no_coarse_targets_v1';result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=keys,probabilities=probability,prior_probabilities=old_prior)
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    dump(REPORT/'summary.json',dict(status='passed',heads_trained=8,fit_images=len(fi),heldout_images=len(vi),seed_fit_images=len(seed),selected=best,train_counts=result[result.split.eq('train')].predicted_class.value_counts().to_dict(),train_needs_review=int(result[result.split.eq('train')].needs_review.sum()),mixed_output_names=0,reference_archive_read=False,semantic_accuracy=None,seconds=time.time()-started))
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),checkpoint_sha256=sha(OUT/'selected_semantic_head.pt'),source_sha256=sha(__file__),protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),selection='frozen independent-source agreement and view proxy before this candidate private audit'))
    print(json.dumps(best,indent=2),flush=True)

if __name__=='__main__':main()

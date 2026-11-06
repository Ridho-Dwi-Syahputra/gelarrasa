"""Fit-only graph diffusion of programmatic CNN anchors; no archive targets.

Inductive extension uses fit neighbors. Heldout/test points never enter graph
construction, seed generation for fitting, or distribution normalization.
"""
from pathlib import Path
import hashlib,json,time
import numpy as np
import pandas as pd
import torch
from scipy import sparse
from threadpoolctl import threadpool_limits
from stage6_semantic_ontology import CLASSES
from stage7_waste_semantic_training import make_head
from stage12_ontology_refinement import load_ontology_features

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage13_graph_semantic_v1'
REPORT=ROOT/'reports/errors/stage13_graph_semantic_20261106'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')

def anchors(profile):
    previous=json.loads((ROOT/'reports/errors/stage12_ontology_refinement_20261106/plan_before_fit.json').read_text())
    d=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    state=torch.load(ROOT/'output/groups/stage4_self_training_pilot_v1/selected_fit_head.pt',map_location='cpu',weights_only=True)
    h=torch.nn.Linear(384,8);h.load_state_dict(state['state_dict']);h.eval()
    with torch.inference_mode():cp=h(torch.from_numpy(d['features'])).softmax(-1).numpy()
    correspondence={int(k):v for k,v in previous['correspondence'].items()}
    mapped=np.array([correspondence[int(i)] for i in cp.argmax(1)])
    a=np.load(ROOT/'output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz',allow_pickle=False)['prior_probabilities']
    b=np.load(ROOT/'output/groups/stage7_waste_semantic_v1/named_probabilities.npz',allow_pickle=False)['prior_probabilities']
    clean=(np.maximum(a[np.arange(len(a)),mapped],b[np.arange(len(b)),mapped])>=.65)&((a.argmax(1)==mapped)|(b.argmax(1)==mapped))&(cp.max(1)>=.8)
    strong=(a.argmax(1)==b.argmax(1))&(a.max(1)>=.75)&(b.max(1)>=.9)
    waste=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/original224.npz',allow_pickle=False)
    names=json.loads((ROOT/'reports/errors/stage7_waste_cnn_20261106/checkpoint_and_ontology.json').read_text())['source_class_order']
    strong&=(b.argmax(1)!=CLASSES.index('electronic'))|(waste['probabilities'][:,names.index('battery')]>=.8)
    labels=mapped.copy();labels[strong]=b.argmax(1)[strong];clean|=strong
    e=np.load(ROOT/'output/embeddings/stage11_ontology_cnn_v1/original224.npz',allow_pickle=False)
    ec=e['conditional'].argmax(1);conf=e['conditional'].max(1)
    support=((ec==a.argmax(1))&(a.max(1)>=.65))|((ec==b.argmax(1))&(b.max(1)>=.8))
    extra=(e['coverage']>=.2)&(conf>=.8)&support
    extra|=(e['coverage']>=.5)&(conf>=.95)&((ec==mapped)|support)&((~strong)|(ec==labels))
    labels[extra]=ec[extra];clean|=extra
    return previous,labels,clean,b

def fit_diffusion(features,labels,clean,k,alpha):
    similarity=features@features.T;np.fill_diagonal(similarity,-np.inf)
    nn=np.argsort(-similarity,axis=1,kind='stable')[:,:k]
    row=np.repeat(np.arange(len(features)),k);value=similarity[row,nn.ravel()]
    value=np.exp((value-np.repeat(value.reshape(-1,k)[:,0],k))/.07)
    w=sparse.csr_matrix((value,(row,nn.ravel())),shape=(len(features),len(features)))
    w=w.maximum(w.T);degree=np.asarray(w.sum(1)).ravel()
    inv=sparse.diags(1/np.sqrt(np.maximum(degree,1e-12)));s=inv@w@inv
    y=np.zeros((len(features),8),dtype='float64');y[np.flatnonzero(clean),labels[clean]]=1
    # Class normalization comes from fit anchors only, never desired test counts.
    y/=np.sqrt(np.maximum(y.sum(0),1))[None,:]
    f=y.copy()
    for iteration in range(300):
        new=alpha*(s@f)+(1-alpha)*y
        if np.max(np.abs(new-f))<1e-8:f=new;break
        f=new
    return (f/np.maximum(f.sum(1,keepdims=True),1e-12)).astype('float32'),iteration+1

def extend(query,fit,distribution,k):
    sim=query@fit.T;nearest=np.argsort(-sim,axis=1,kind='stable')[:,:k]
    value=np.take_along_axis(sim,nearest,axis=1)
    weights=np.exp((value-value[:,:1])/.07);weights/=weights.sum(1,keepdims=True)
    return (distribution[nearest]*weights[:,:,None]).sum(1).astype('float32')

def predict_saved_model(base):
    meta=json.loads((base/'graph_model.json').read_text())
    if meta['method']=='neural_control':
        source=ROOT/meta['checkpoint'];assert sha(source)==meta['checkpoint_sha256']
        c=torch.load(source,map_location='cpu',weights_only=True)
        keys,x,_,_=load_ontology_features(c['feature_branch']);h=make_head(c['kind'],c['feature_dimensions']);h.load_state_dict(c['state_dict']);h.eval()
        with torch.inference_mode():return keys,h(torch.from_numpy(x)).softmax(-1).numpy()
    z=np.load(base/'graph_fit.npz',allow_pickle=False)
    keys,x,_,_=load_ontology_features(meta['branch'])
    return keys,extend(x,z['features'],z['distribution'],meta['k'])

def main():
    started=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str});index=dict(zip(p.key,range(len(p))))
    previous,labels,clean,prior=anchors(p)
    fi=np.array([index[k] for k in previous['fit_keys']]);vi=np.array([index[k] for k in previous['heldout_keys']]);core=clean[vi]
    seeds=p.iloc[fi[clean[fi]]][['key']].copy();seeds['algorithmic_seed_class']=np.array(CLASSES)[labels[fi[clean[fi]]]]
    prior_seeds=pd.read_csv(ROOT/'output/groups/stage12_ontology_refinement_v1/clean_fit_seeds.csv')[['key','algorithmic_seed_class']]
    assert seeds.reset_index(drop=True).equals(prior_seeds.reset_index(drop=True))
    branches=['dino','ontology_dino','ontology_all'];features={b:load_ontology_features(b) for b in branches}
    geometry=features['ontology_dino'][1]
    with threadpool_limits(limits=4):neighbors=np.argsort(-(geometry[vi]@geometry[fi].T),axis=1,kind='stable')[:,:10]
    flip_keys=features['ontology_dino'][2];lookup=dict(zip(flip_keys,range(len(flip_keys))))
    paired=np.array([i for i,k in enumerate(previous['heldout_keys']) if k in lookup]);fp=np.array([lookup[previous['heldout_keys'][i]] for i in paired])
    dump(REPORT/'plan_before_fit.json',dict(source_sha256=sha(__file__),branches=branches,k=[10,20,40],alpha=[.8,.95],fit=len(fi),heldout=len(vi),fit_seeds=len(seeds),
        seed_manifest_sha256=sha(ROOT/'output/groups/stage12_ontology_refinement_v1/clean_fit_seeds.csv'),reference_archive_read=False,reference_targets_or_filters=False,
        researcher_has_seen_previous_private_audits=True,heldout_or_test_in_graph=False,semantic_accuracy=None,
        selection='.5 macro programmatic-anchor agreement +.25 flip +.25 neighbor consistency, all8 min5; frozen before this candidate private audit',
        paper='https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html'))
    def metric(prob,flip):
        pred=prob.argmax(1);v=pred[vi]
        macro=float(np.mean([(v[core&(labels[vi]==c)]==c).mean() for c in range(8)]));n=np.bincount(v,minlength=8)
        flip_agree=float((v[paired]==flip.argmax(1)).mean());neighbor=float((v[:,None]==pred[fi][neighbors]).mean())
        return dict(score=.5*macro+.25*flip_agree+.25*neighbor,macro_anchor_agreement=macro,flip=flip_agree,neighbor=neighbor,occupied=int((n>0).sum()),min_class=int(n.min()))
    path=ROOT/'output/groups/stage12_ontology_refinement_v1/selected_semantic_head.pt';c=torch.load(path,map_location='cpu',weights_only=True)
    keys,x,fkeys,xf=load_ontology_features(c['feature_branch']);h=make_head(c['kind'],c['feature_dimensions']);h.load_state_dict(c['state_dict']);h.eval()
    with torch.inference_mode():base=h(torch.from_numpy(x)).softmax(-1).numpy();baseflip=h(torch.from_numpy(xf[fp])).softmax(-1).numpy()
    m=metric(base,baseflip);history=[dict(candidate='neural_control',**m)]
    best=dict(method='neural_control',checkpoint=str(path.relative_to(ROOT)),checkpoint_sha256=sha(path),**m);best_probability=base;selected_graph=None
    with threadpool_limits(limits=4):
        for branch in branches:
            keys,x,_,xf=features[branch];assert np.array_equal(keys,p.key.to_numpy(str))
            for k in [10,20,40]:
                for alpha in [.8,.95]:
                    distribution,steps=fit_diffusion(x[fi],labels[fi],clean[fi],k,alpha)
                    probability=extend(x,x[fi],distribution,k);flipped=extend(xf[fp],x[fi],distribution,k);m=metric(probability,flipped)
                    candidate=dict(method='graph_diffusion',branch=branch,k=k,alpha=alpha,iterations=steps,**m)
                    history.append(candidate)
                    if m['occupied']==8 and m['min_class']>=5 and m['score']>best['score']+1e-7:
                        best=candidate;best_probability=probability;selected_graph=(x[fi].copy(),distribution.copy())
            print('completed '+branch,flush=True)
    dump(OUT/'graph_model.json',best)
    if selected_graph is not None:np.savez_compressed(OUT/'graph_fit.npz',features=selected_graph[0],distribution=selected_graph[1])
    keys,reproduced=predict_saved_model(OUT);assert np.array_equal(keys,p.key.to_numpy(str)) and np.allclose(reproduced,best_probability,atol=1e-6)
    probability=best_probability;pred=probability.argmax(1);order=np.sort(probability,axis=1)
    result=p[['key','split','image_id','pixel_sha256','processed_path']].copy();result['predicted_class']=np.array(CLASSES)[pred]
    result['uncalibrated_probability']=probability.max(1);result['margin']=order[:,-1]-order[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    result['cnn_prior_disagrees']=pred!=prior.argmax(1);result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='fit_only_graph_semantic_diffusion_v1';result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=keys,probabilities=probability,prior_probabilities=prior)
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    dump(REPORT/'summary.json',dict(status='passed',graph_fits=18,selected=best,train_counts=result[result.split.eq('train')].predicted_class.value_counts().to_dict(),
        mixed_output_names=0,reference_archive_read=False,semantic_accuracy=None,seconds=time.time()-started))
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),model_metadata_sha256=sha(OUT/'graph_model.json'),source_sha256=sha(__file__),protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),selection='frozen model-only graph proxy before this candidate private audit'))
    print(json.dumps(best,indent=2),flush=True)

if __name__=='__main__':main()

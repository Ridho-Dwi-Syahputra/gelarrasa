"""Adapt recycling-domain CNN features to eight task categories on local fit.

Task-specific pretrained weights are disclosed. No external image dataset,
reference archive, old display labels, or VLM is read. CNN teacher confidence
is not ground truth; selection remains a development proxy.
"""
from pathlib import Path
import copy,hashlib,json,time
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from stage6_semantic_ontology import CLASSES,object_evidence

ROOT=Path(__file__).resolve().parents[2]
FEATURE=ROOT/'output/embeddings/stage7_waste_cnn_v1'
OUT=ROOT/'output/groups/stage7_waste_semantic_v1'
REPORT=ROOT/'reports/errors/stage7_waste_cnn_20261106'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,data):p.write_text(json.dumps(data,indent=2,default=str)+'\n',encoding='utf-8')

def make_head(kind,dimension):
    return torch.nn.Linear(dimension,8) if kind=='linear' else torch.nn.Sequential(torch.nn.Linear(dimension,128),torch.nn.GELU(),torch.nn.Linear(128,8))

def feature_blocks(waste,dino,branch):
    if branch=='waste':return waste
    if branch=='dino':return dino
    assert branch=='hybrid'
    return (np.concatenate([waste,dino],axis=1)/np.float32(np.sqrt(2))).astype('float32')

def main():
    started=time.time();OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True);torch.set_num_threads(4)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    w=np.load(FEATURE/'original224.npz',allow_pickle=False);d=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    assert np.array_equal(w['keys'],d['keys']) and np.array_equal(w['keys'],p.key.to_numpy(str))
    ontology=json.loads((REPORT/'checkpoint_and_ontology.json').read_text())
    source_names=ontology['source_class_order']
    correspondence={'battery':'electronic','biological':'foodwaste','brown-glass':'glass','cardboard':'cardboard','clothes':'textile',
                    'green-glass':'glass','metal':'metal','paper':'paper','plastic':'plastic','shoes':'textile','white-glass':'glass'}
    teacher=np.zeros((len(p),8),dtype='float32')
    for source,target in correspondence.items():teacher[:,CLASSES.index(target)]+=w['probabilities'][:,source_names.index(source)]
    # Unknown upstream trash is not assigned a fabricated material; the older
    # independent CNN material/object evidence supplies its soft remainder.
    old=np.load(ROOT/'output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz',allow_pickle=False)
    assert np.array_equal(old['keys'],w['keys'])
    teacher+=w['probabilities'][:,source_names.index('trash'),None]*old['prior_probabilities']
    object_data=np.load(ROOT/'output/embeddings/stage6_semantic_v1/resnet_objects_original224.npz',allow_pickle=False)
    object_names=json.loads((COMMON/'object_evidence_provenance.json').read_text())['categories']
    evidence,_=object_evidence(object_data['logits'],object_names)
    conditional=evidence/np.maximum(evidence.sum(1,keepdims=True),1e-8)
    coverage=evidence.max(1)/np.maximum(softmax(object_data['logits'],axis=1).max(1),1e-8)
    electronic=CLASSES.index('electronic')
    electronic_objects=(conditional[:,electronic]>=.6)&(coverage>=.5)&(evidence[:,electronic]>=.025)
    parent=np.zeros_like(teacher);parent[:,electronic]=1
    teacher[electronic_objects]=.9*parent[electronic_objects]+.1*teacher[electronic_objects]
    teacher=np.maximum(teacher,1e-8);teacher/=teacher.sum(1,keepdims=True)
    old_pointer=json.loads((COMMON/'selected_material_run.json').read_text())
    previous_path=ROOT/old_pointer['output'];previous=torch.load(previous_path/'selected_semantic_head.pt',map_location='cpu',weights_only=True)
    fit_keys=previous['fit_keys'];val_keys=previous['heldout_keys'];index=dict(zip(p.key,range(len(p))))
    fi=np.array([index[k] for k in fit_keys]);vi=np.array([index[k] for k in val_keys])
    split=pd.read_csv(ROOT/'reports/errors/stage2_20261105/neighbor_refinement/group_split.csv').set_index('key')
    assert not set(split.loc[fit_keys,'candidate_group'])&set(split.loc[val_keys,'candidate_group'])
    assert not set(p.iloc[fi].pixel_sha256)&set(p.iloc[vi].pixel_sha256)
    wf=np.load(FEATURE/'hflip224.npz',allow_pickle=False);df=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    assert np.array_equal(wf['keys'],df['keys']);lookup=dict(zip(wf['keys'],range(len(wf['keys']))))
    tf=np.array([i for i,k in enumerate(fit_keys) if k in lookup]);tv=np.array([i for i,k in enumerate(val_keys) if k in lookup])
    fids=np.array([lookup[fit_keys[i]] for i in tf]);vids=np.array([lookup[val_keys[i]] for i in tv])
    with threadpool_limits(limits=4):
        geometry=feature_blocks(w['features'],d['features'],'hybrid')
        similarity=geometry[fi]@geometry[fi].T;np.fill_diagonal(similarity,-np.inf)
        nn=np.argsort(-similarity,axis=1,kind='stable')[:,:10]
        nv=np.argsort(-(geometry[vi]@geometry[fi].T),axis=1,kind='stable')[:,:10]
    hard=teacher.argmax(1);core=teacher[vi].max(1)>=.7;counts=np.bincount(hard[vi][core],minlength=8)
    assert (counts>0).all(),counts
    plan=dict(source_sha256=sha(__file__),class_order=CLASSES,source_correspondence=correspondence,
        pretrained_source=ontology,fit_keys=fit_keys,heldout_keys=val_keys,epochs=180,lr=.003,
        branches=['waste','dino','hybrid'],heads=['linear','mlp'],view_loss=.2,graph_loss=.03,
        target='soft CNN evidence with confidence weights .2+max_probability**2; no LLM targets',
        selection='.5 macro CNN-core agreement +.25 flip +.25 fixed hybrid-neighbor agreement; all8 occupied min5 heldout.',
        core_counts=dict(zip(CLASSES,counts.tolist())),reference_archive_read=False,
        feature_sha256=sha(FEATURE/'original224.npz'),flip_feature_sha256=sha(FEATURE/'hflip224.npz'),
        pretrained_training_overlap_with_competition_unknown=True,semantic_accuracy=None)
    dump(REPORT/'training_plan_before_fit.json',plan)
    history=[];best=None
    def evaluate(head,branch):
        xx=feature_blocks(w['features'],d['features'],branch)
        fx=feature_blocks(wf['features'],df['features'],branch)
        head.eval()
        with torch.inference_mode():
            fit=head(torch.from_numpy(xx[fi])).argmax(1).numpy();val=head(torch.from_numpy(xx[vi])).argmax(1).numpy()
            flipped=head(torch.from_numpy(fx[vids])).argmax(1).numpy()
        macro=float(np.mean([(val[core&(hard[vi]==c)]==c).mean() for c in range(8)]))
        flip=float((val[tv]==flipped).mean());near=float((val[:,None]==fit[nv]).mean());n=np.bincount(val,minlength=8)
        return dict(score=.5*macro+.25*flip+.25*near,macro_CNN_core_agreement=macro,flip=flip,neighbor=near,
                    occupied=int((n>0).sum()),min_class=int(n.min()))
    head=make_head(previous['kind'],384);head.load_state_dict(previous['state_dict']);metric=evaluate(head,'dino')
    history.append(dict(candidate='previous_material_student',epoch=previous['selected']['epoch'],**metric))
    if metric['occupied']==8 and metric['min_class']>=5:
        best=dict(candidate='previous_material_student',kind=previous['kind'],branch='dino',dimension=384,epoch=previous['selected']['epoch'],**metric,state=copy.deepcopy(head.state_dict()))
    target=torch.from_numpy(teacher[fi]);weight=torch.from_numpy((.2+teacher[fi].max(1)**2).astype('float32'))
    for branch in ['waste','dino','hybrid']:
        xx=feature_blocks(w['features'],d['features'],branch);fx=feature_blocks(wf['features'],df['features'],branch)
        xt=torch.from_numpy(xx[fi]);xf=torch.from_numpy(fx[fids])
        for kind in ['linear','mlp']:
            torch.manual_seed(20261106);head=make_head(kind,xx.shape[1]);optimizer=torch.optim.Adam(head.parameters(),lr=.003,weight_decay=.0001)
            for epoch in range(1,181):
                head.train();logits=head(xt);q=logits.softmax(-1)
                ce=-(target*logits.log_softmax(-1)).sum(1)
                graph=-(q[:,None,:]*q[nn]).sum(-1).clamp_min(1e-8).log().mean()
                alt=head(xf).softmax(-1);orig=q[tf];mean=(orig+alt)/2
                js=.5*((orig*(orig.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean()+
                       (alt*(alt.clamp_min(1e-8).log()-mean.clamp_min(1e-8).log())).sum(1).mean())
                loss=(ce*weight).mean()+.03*graph+.2*js;assert torch.isfinite(loss)
                optimizer.zero_grad();loss.backward();optimizer.step()
                if epoch%10==0:
                    metric=evaluate(head,branch);name=f'{branch}_{kind}';history.append(dict(candidate=name,epoch=epoch,**metric))
                    if metric['occupied']==8 and metric['min_class']>=5 and (best is None or metric['score']>best['score']+1e-7):
                        best=dict(candidate=name,kind=kind,branch=branch,dimension=xx.shape[1],epoch=epoch,**metric,state=copy.deepcopy(head.state_dict()))
            print(f'waste-base student {branch} {kind} trained',flush=True)
    assert best is not None
    state=best.pop('state');head=make_head(best['kind'],best['dimension']);head.load_state_dict(state);head.eval()
    xx=feature_blocks(w['features'],d['features'],best['branch'])
    with torch.inference_mode():probability=head(torch.from_numpy(xx)).softmax(-1).numpy()
    torch.save(dict(state_dict=state,kind=best['kind'],feature_dimensions=best['dimension'],feature_branch=best['branch'],
        class_order=CLASSES,selected=best,fit_keys=fit_keys,heldout_keys=val_keys,plan=plan),OUT/'selected_semantic_head.pt')
    pred=probability.argmax(1);order=np.sort(probability,axis=1)
    result=p[['key','split','image_id','pixel_sha256','processed_path']].copy()
    result['predicted_class']=np.array(CLASSES)[pred];result['uncalibrated_probability']=probability.max(1)
    result['margin']=order[:,-1]-order[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    result['cnn_prior_disagrees']=pred!=hard
    result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='Task_specific_nonVLM_CNN_base_local_eight_class_head_v1'
    result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=p.key.to_numpy(str),probabilities=probability,prior_probabilities=teacher)
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    summary=dict(status='passed',heads_trained=6,selected=best,fit_images=len(fi),heldout_images=len(vi),
        train_counts=result[result.split.eq('train')].predicted_class.value_counts().to_dict(),
        train_needs_review=int(result[result.split.eq('train')].needs_review.sum()),mixed_output_names=0,
        reference_archive_read=False,external_image_dataset_downloaded=False,task_specific_pretrained_base_used=True,
        pretrained_training_overlap_unknown=True,semantic_accuracy=None,seconds=time.time()-started)
    dump(REPORT/'summary.json',summary)
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),
        checkpoint_sha256=sha(OUT/'selected_semantic_head.pt'),summary_sha256=sha(REPORT/'summary.json'),source_sha256=sha(__file__),
        protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),
        selection='same new waste-CNN-core/view/hybrid-neighbor development criterion for old control and six new heads; before any reference read',
        checkpoint_sha256=sha(OUT/'selected_semantic_head.pt')))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

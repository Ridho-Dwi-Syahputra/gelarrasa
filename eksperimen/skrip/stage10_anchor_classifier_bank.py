"""SVM/kNN ablation on the same algorithmic anchors, including local features."""
from pathlib import Path
import copy,hashlib,json,shutil,time
import joblib,numpy as np,pandas as pd,torch,sklearn
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from threadpoolctl import threadpool_limits
from stage7_waste_semantic_training import make_head,feature_blocks
from stage6_semantic_ontology import CLASSES

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'output/groups/stage10_anchor_bank_v1'
REPORT=ROOT/'reports/errors/stage10_anchor_bank_20261106'
COMMON=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,d):p.write_text(json.dumps(d,indent=2,default=str)+'\n',encoding='utf-8')

def load_features(branch):
    d=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    df=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_hflip224_probe.npz',allow_pickle=False)
    if branch in ['dino','hybrid']:
        w=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/original224.npz',allow_pickle=False)
        wf=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/hflip224.npz',allow_pickle=False)
        x=feature_blocks(w['features'],d['features'],branch);flip=feature_blocks(wf['features'],df['features'],branch)
    else:
        q=np.load(ROOT/'output/embeddings/stage5_patch_v1/original224.npz',allow_pickle=False)
        qf=np.load(ROOT/'output/embeddings/stage5_patch_v1/hflip224.npz',allow_pickle=False)
        assert np.array_equal(q['keys'],d['keys']) and np.array_equal(qf['keys'],df['keys'])
        weights=np.repeat(np.array([np.sqrt(.5),.5,.5],dtype='float32'),384)
        x=q['features']*weights;flip=qf['features']*weights
        if branch=='all':
            w=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/original224.npz',allow_pickle=False)
            wf=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/hflip224.npz',allow_pickle=False)
            x=np.concatenate([x,w['features']],axis=1)/np.float32(np.sqrt(2));flip=np.concatenate([flip,wf['features']],axis=1)/np.float32(np.sqrt(2))
    assert x.dtype==np.float32 and flip.dtype==np.float32
    return d['keys'],x,df['keys'],flip

def main():
    started=time.time();torch.set_num_threads(4);OUT.mkdir(parents=True,exist_ok=True);REPORT.mkdir(parents=True,exist_ok=True)
    p=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str});index=dict(zip(p.key,range(len(p))))
    source=ROOT/'output/groups/stage9_consensus_seed_v1'
    plan9=json.loads((ROOT/'reports/errors/stage9_consensus_seed_20261106/plan_before_fit.json').read_text())
    fit=plan9['fit_keys'];held=plan9['heldout_keys'];fi=np.array([index[k] for k in fit]);vi=np.array([index[k] for k in held])
    checkpoint=torch.load(source/'selected_semantic_head.pt',map_location='cpu',weights_only=True)
    d=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
    anon=torch.load(ROOT/'output/groups/stage4_self_training_pilot_v1/selected_fit_head.pt',map_location='cpu',weights_only=True)
    coarse=torch.nn.Linear(384,8);coarse.load_state_dict(anon['state_dict']);coarse.eval()
    with torch.inference_mode():cp=coarse(torch.from_numpy(d['features'])).softmax(-1).numpy()
    correspondence={int(k):v for k,v in plan9['correspondence'].items()};mapped=np.array([correspondence[int(k)] for k in cp.argmax(1)])
    a=np.load(ROOT/'output/groups/stage6_material_resolution_calibrated_v2/named_probabilities.npz',allow_pickle=False)['prior_probabilities']
    b=np.load(ROOT/'output/groups/stage7_waste_semantic_v1/named_probabilities.npz',allow_pickle=False)['prior_probabilities']
    support=np.maximum(a[np.arange(len(p)),mapped],b[np.arange(len(p)),mapped])
    clean=(support>=.65)&((a.argmax(1)==mapped)|(b.argmax(1)==mapped))&(cp.max(1)>=.8)
    strong=(a.argmax(1)==b.argmax(1))&(a.max(1)>=.75)&(b.max(1)>=.9)
    w=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/original224.npz',allow_pickle=False)
    names=json.loads((ROOT/'reports/errors/stage7_waste_cnn_20261106/checkpoint_and_ontology.json').read_text())['source_class_order']
    strong&=(b.argmax(1)!=CLASSES.index('electronic'))|(w['probabilities'][:,names.index('battery')]>=.8)
    labels=mapped.copy();labels[strong]=b.argmax(1)[strong];clean|=strong
    seeds=np.flatnonzero(clean[fi]);seed_keys=p.iloc[fi[seeds]].key.to_numpy(str)
    previous_seeds=pd.read_csv(source/'clean_fit_seeds.csv').key.to_numpy(str)
    assert np.array_equal(seed_keys,previous_seeds)
    y=labels[fi[seeds]];core=clean[vi]
    key,x,flip_keys,xf=load_features('hybrid');lookup=dict(zip(flip_keys,range(len(flip_keys))))
    paired=np.array([i for i,k in enumerate(held) if k in lookup]);fp=np.array([lookup[held[i]] for i in paired])
    with threadpool_limits(limits=4):nn=np.argsort(-(x[vi]@x[fi].T),axis=1,kind='stable')[:,:10]
    plan=dict(source_sha256=sha(__file__),sklearn=sklearn.__version__,fit_keys=fit,heldout_keys=held,seeds=len(seeds),
        seed_manifest_sha256=sha(source/'clean_fit_seeds.csv'),branches=['dino','global_local','hybrid','all'],
        classifiers=['SVC_RBF_C1','SVC_RBF_C10','SVC_RBF_C100','cosine_kNN3','cosine_kNN7','cosine_kNN15'],
        probability_policy='Score predict_proba.argmax consistently; SVC internal calibration sees fit anchors only, not ground truth.',
        source_consensus_seed_weight=2,selection='same macro-anchor/flip/neighbor score as stage9, no archive criterion',
        reference_archive_read=False,researcher_seen_prior_audits=True,test_in_fit=False,semantic_accuracy=None)
    dump(REPORT/'plan_before_fit.json',plan)
    def metric(probability,flipped):
        pred=probability.argmax(1);v=pred[vi]
        macro=float(np.mean([(v[core&(labels[vi]==c)]==c).mean() for c in range(8)]))
        flip=float((v[paired]==flipped.argmax(1)).mean());neighbor=float((v[:,None]==pred[fi][nn]).mean());n=np.bincount(v,minlength=8)
        return dict(score=.5*macro+.25*flip+.25*neighbor,macro_anchor_agreement=macro,flip=flip,neighbor=neighbor,occupied=int((n>0).sum()),min_class=int(n.min()))
    head=make_head(checkpoint['kind'],checkpoint['feature_dimensions']);head.load_state_dict(checkpoint['state_dict']);head.eval()
    _,oldx,_,oldf=load_features(checkpoint['feature_branch'])
    with torch.inference_mode():base=head(torch.from_numpy(oldx)).softmax(-1).numpy();baseflip=head(torch.from_numpy(oldf[fp])).softmax(-1).numpy()
    baseline=metric(base,baseflip);history=[dict(candidate='previous_anchor_mlp',**baseline)]
    best=dict(candidate='previous_anchor_mlp',branch=checkpoint['feature_branch'],**baseline);best_probability=base;best_classifier=None
    with threadpool_limits(limits=4):
        for branch in plan['branches']:
            _,x,_,xf=load_features(branch)
            configs=[('svc'+str(c),SVC(C=c,kernel='rbf',gamma='scale',class_weight='balanced',probability=True,random_state=20261106,cache_size=128)) for c in [1,10,100]]
            configs += [('knn'+str(k),KNeighborsClassifier(n_neighbors=k,weights='distance',metric='cosine',algorithm='brute',n_jobs=1)) for k in [3,7,15]]
            for name,clf in configs:
                if name.startswith('svc'):clf.fit(x[fi[seeds]],y,sample_weight=(1+strong[fi[seeds]].astype('float64')))
                else:clf.fit(x[fi[seeds]],y)
                assert np.array_equal(clf.classes_,np.arange(8))
                probability=clf.predict_proba(x).astype('float32');flipped=clf.predict_proba(xf[fp]).astype('float32')
                m=metric(probability,flipped);candidate=branch+'_'+name;history.append(dict(candidate=candidate,**m))
                if m['occupied']==8 and m['min_class']>=5 and m['score']>best['score']+1e-7:
                    best=dict(candidate=candidate,branch=branch,**m);best_probability=probability;best_classifier=copy.deepcopy(clf)
                print(candidate,round(m['score'],6),flush=True)
    if best_classifier is not None:
        joblib.dump(best_classifier,OUT/'classical_model.joblib');dump(OUT/'classical_model.json',dict(branch=best['branch'],class_order=CLASSES,model_sha256=sha(OUT/'classical_model.joblib')))
    else:shutil.copy2(source/'selected_semantic_head.pt',OUT/'selected_semantic_head.pt')
    probability=best_probability;pred=probability.argmax(1);order=np.sort(probability,axis=1)
    result=p[['key','split','image_id','pixel_sha256','processed_path']].copy();result['predicted_class']=np.array(CLASSES)[pred]
    result['uncalibrated_probability']=probability.max(1);result['margin']=order[:,-1]-order[:,-2];result['entropy']=-(probability*np.log(np.maximum(probability,1e-8))).sum(1)
    result['cnn_prior_disagrees']=pred!=b.argmax(1);result['needs_review']=result.uncalibrated_probability.lt(.65)|result.margin.lt(.25)|result.cnn_prior_disagrees
    result['ground_truth_verified']=False;result['method']='Algorithmic_semantic_anchors_classifier_bank_v1';result.to_csv(OUT/'named_assignments.csv',index=False)
    np.savez_compressed(OUT/'named_probabilities.npz',keys=p.key.to_numpy(str),probabilities=probability,prior_probabilities=b)
    pd.DataFrame(history).to_csv(REPORT/'development_metrics.csv',index=False)
    summary=dict(status='passed',classifiers_fit=24,selected=best,baseline=baseline,mixed_output_names=0,
        train_needs_review=int(result[result.split.eq('train')].needs_review.sum()),reference_archive_read=False,semantic_accuracy=None,seconds=time.time()-started)
    dump(REPORT/'summary.json',summary)
    protected=['configs/data_contract.json','output/groups/stage4_self_training_pilot_v1/candidate_assignments.csv','notebook/01_EDA_Preprocessing_Clustering.ipynb']
    dump(REPORT/'selection_frozen.json',dict(status='frozen_before_reference_audit',assignment_sha256=sha(OUT/'named_assignments.csv'),summary_sha256=sha(REPORT/'summary.json'),source_sha256=sha(__file__),protected={v:sha(ROOT/v) for v in protected}))
    dump(COMMON/'selected_material_run.json',dict(output=str(OUT.relative_to(ROOT)),report=str(REPORT.relative_to(ROOT)),selection='same model-anchor/view/neighbor development criterion before this candidate reference audit'))
    print(json.dumps(summary,indent=2),flush=True)

if __name__=='__main__':main()

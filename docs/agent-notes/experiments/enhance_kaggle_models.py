"""Add the established DINOv3/SVM/group-kNN route and train-only tree candidates."""
from pathlib import Path
import ast,json,textwrap

ROOT=Path(__file__).resolve().parents[3]
TARGET=ROOT/'final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb'
nb=json.loads(TARGET.read_text(encoding='utf-8'))
def md(s):return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(True)}
def code(s):return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':textwrap.dedent(s).strip().splitlines(True)}
if nb['metadata'].get('gelarrasa_pipeline',{}).get('version')=='active_processed_hybrid_v3':
    print('Hybrid v3 already built.');raise SystemExit(0)

train_src=''.join(nb['cells'][15]['source'])
split=train_src.index('oof_probs={}; proxy_summary=[]')
definitions=train_src[:split]
deep_cv=train_src[split:]
refit_src=''.join(nb['cells'][17]['source'])
refit_ast=ast.parse(refit_src)
submission_node=next(n for n in refit_ast.body if isinstance(n,ast.FunctionDef) and n.name=='write_submission')
submission_def=ast.get_source_segment(refit_src,submission_node)
definitions+='\n'+submission_def+'\n'

feature_code=r'''
import importlib.metadata,subprocess,joblib
DEPENDENCY_PINS={'timm':'1.0.27','xgboost':'3.1.2'}
for package,required in DEPENDENCY_PINS.items():
    try: current=importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError: current=None
    if current!=required:
        subprocess.check_call([sys.executable,'-m','pip','install','--quiet',f'{package}=={required}'])
import timm,xgboost,cv2
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import normalize
from skimage.feature import hog,local_binary_pattern
D3_REPOSITORY='timm/vit_base_patch16_dinov3.lvd1689m'
D3_REVISION='c6a5fb7d12bbd3cf3b0079253141c3332aaed7da'
D3_WEIGHT_SHA256='1f9ed8a2378d65e24bb710ba522ac9fa7be4e036d7aefb4384ce022833926332'
D3_RECIPE='dinov3base_256_letterbox_valid_foreground4x768_v1'
d3_weight=hf_hub_download(D3_REPOSITORY,'model.safetensors',revision=D3_REVISION)
assert file_sha256(d3_weight)==D3_WEIGHT_SHA256,'DINOv3 checkpoint SHA256 mismatch.'
feature_rows=pd.concat([train_frame,test_frame],ignore_index=True)
def native_view(row):
    im=rgb_source(row['source_path'])
    return im.crop(tuple(row['bbox'])) if row['is_crop'] else im
def d3_prepare(row,mean,std):
    im=native_view(row); width,height=im.size; scale=256/max(width,height)
    nw,nh=max(1,round(width*scale)),max(1,round(height*scale)); dx,dy=(256-nw)//2,(256-nh)//2
    canvas=np.empty((256,256,3),np.uint8);canvas[:]=[124,116,104]
    resized=np.asarray(im.resize((nw,nh),Image.Resampling.BICUBIC));canvas[dy:dy+nh,dx:dx+nw]=resized
    valid=np.zeros((256,256),np.float32);valid[dy:dy+nh,dx:dx+nw]=1.
    border=np.concatenate([resized[0],resized[-1],resized[:,0],resized[:,-1]],axis=0).astype(np.float32)
    center=np.median(border,axis=0);dispersion=np.linalg.norm(border-center,axis=1);foreground=valid.copy()
    if float(np.quantile(dispersion,.75))<30:
        distance=np.linalg.norm(resized.astype(np.float32)-center,axis=2)
        mask=(distance>max(25.,float(np.quantile(dispersion,.95))+10.)).astype(np.uint8)
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
        if .03<=float(mask.mean())<=.90:
            foreground[:]=0;foreground[dy:dy+nh,dx:dx+nw]=mask.astype(np.float32);foreground+=.05*valid
    valid_patch=cv2.resize(valid,(16,16),interpolation=cv2.INTER_AREA).reshape(-1)
    foreground_patch=cv2.resize(foreground,(16,16),interpolation=cv2.INTER_AREA).reshape(-1)
    tensor=(canvas.astype(np.float32)/255.-mean[None,None,:])/std[None,None,:]
    return torch.from_numpy(tensor.transpose(2,0,1).copy()),valid_patch,foreground_patch
def d3_descriptor_batch(encoder,prepared):
    images=torch.stack([a[0] for a in prepared]).to(DEVICE)
    tokens=encoder.forward_features(images);patches=tokens[:,encoder.num_prefix_tokens:]
    assert patches.shape[1:]==(256,768)
    valid=torch.as_tensor(np.stack([a[1] for a in prepared]),device=DEVICE);valid/=valid.sum(1,keepdim=True).clamp_min(1e-8)
    foreground=torch.as_tensor(np.stack([a[2] for a in prepared]),device=DEVICE);foreground/=foreground.sum(1,keepdim=True).clamp_min(1e-8)
    global_mean=(patches*valid[:,:,None]).sum(1);object_mean=(patches*foreground[:,:,None]).sum(1)
    object_std=(((patches-object_mean[:,None,:]).square()*foreground[:,:,None]).sum(1).clamp_min(0)).sqrt()
    return torch.cat([F.normalize(a,dim=1) for a in [tokens[:,0],global_mean,object_mean,object_std]],dim=1).cpu().numpy().astype(np.float32)
def hand_descriptor(row):
    im=native_view(row);rgb=np.asarray(im.resize((128,128),Image.Resampling.LANCZOS),dtype=np.uint8)
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV);lab=cv2.cvtColor(rgb,cv2.COLOR_RGB2LAB)
    blocks=[]
    def hist(values,bins,range_):
        h=np.histogram(values,bins=bins,range=range_)[0].astype(np.float32);return h/max(h.sum(),1.)
    for channel,bins,upper in [(0,16,180),(1,8,256),(2,8,256)]:blocks.append(hist(hsv[:,:,channel],bins,(0,upper)))
    joint=np.histogram2d(hsv[:,:,0].ravel(),hsv[:,:,1].ravel(),bins=(8,4),range=((0,180),(0,256)))[0].reshape(-1).astype(np.float32);blocks.append(joint/max(joint.sum(),1.))
    for channel in range(3):blocks.append(hist(lab[:,:,channel],8,(0,256)))
    for yy in [0,64]:
        for xx in [0,64]:
            patch=hsv[yy:yy+64,xx:xx+64]
            for channel,bins,upper in [(0,8,180),(1,4,256),(2,4,256)]:blocks.append(hist(patch[:,:,channel],bins,(0,upper)))
    arr=rgb.astype(np.float32)/255.;blocks.extend([arr.mean((0,1)),arr.std((0,1))])
    h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    fractions=[((h>=4)&(h<=30)&(s>=45)&(v>=25)&(v<=225)).mean(),(s<30).mean(),(s>120).mean(),(v<40).mean(),(v>220).mean(),s.std()/255.,v.std()/255.]
    blocks.append(np.array(fractions,np.float32))
    gray=np.asarray(im.convert('L').resize((64,64),Image.Resampling.LANCZOS),dtype=np.uint8)
    blocks.append(hog(gray.astype(np.float32)/255.,orientations=9,pixels_per_cell=(16,16),cells_per_block=(2,2),block_norm='L2-Hys').astype(np.float32))
    blocks.append(hist(local_binary_pattern(gray,8,1,method='uniform'),10,(0,10)))
    return np.concatenate(blocks).astype(np.float32)

cache=WORK_ROOT/'processed/dinov3_hybrid_features.npz'
feature_identity=hashlib.sha256((STATE_SHA256+D3_WEIGHT_SHA256+D3_RECIPE+'native_roi_hand_v1').encode()).hexdigest()
if cache.exists():
    with np.load(cache,allow_pickle=False) as zz:
        assert str(zz['identity'].item())==feature_identity
        d3_features=zz['dino'].copy();hand_features=zz['hand'].copy()
else:
    encoder=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0,pretrained_cfg_overlay={'file':str(d3_weight)}).eval().to(DEVICE)
    if hasattr(encoder,'rope') and hasattr(encoder.rope,'periods'):encoder.rope.periods=encoder.rope.periods.to(torch.bfloat16).to(torch.float32)
    cfg=timm.data.resolve_model_data_config(encoder);mean=np.asarray(cfg['mean'],np.float32);std=np.asarray(cfg['std'],np.float32)
    assert encoder.num_features==768
    records=feature_rows.to_dict('records');chunks=[];batch=8 if DEVICE.type=='cuda' else 2
    # FP32 preserves the established feature recipe; no local-label adaptation of this encoder.
    with torch.inference_mode():
        for start in range(0,len(records),batch):
            chunks.append(d3_descriptor_batch(encoder,[d3_prepare(row,mean,std) for row in records[start:start+batch]]))
            if start%256==0:print('DINOv3 descriptors:',min(start+batch,len(records)),'/',len(records),flush=True)
    d3_features=np.concatenate(chunks);hand_features=np.stack([hand_descriptor(row) for row in records])
    np.savez_compressed(cache,dino=d3_features,hand=hand_features,identity=np.array(feature_identity))
    del encoder;gc.collect();torch.cuda.empty_cache()
assert d3_features.shape==(len(feature_rows),3072) and np.isfinite(d3_features).all() and np.isfinite(hand_features).all()
dino_train=normalize(d3_features[:len(train_frame)]);dino_test=normalize(d3_features[len(train_frame):])
hand_train=hand_features[:len(train_frame)];hand_test=hand_features[len(train_frame):]
DINO_PROVENANCE={'repository':D3_REPOSITORY,'revision':D3_REVISION,'weight_sha256':D3_WEIGHT_SHA256,'recipe':D3_RECIPE,'feature_identity':feature_identity,'train_rows':len(train_frame),'test_rows':len(test_frame),'tree_features':'train-fold PCA of DINO + HSV/Lab/spatial color + HOG/LBP','native_view':'same processed sample membership/ROI/label, native resolution before model resize','timm':timm.__version__,'xgboost':xgboost.__version__}
(WORK_ROOT/'logs/dino_feature_provenance.json').write_text(json.dumps(DINO_PROVENANCE,indent=2),encoding='utf-8')
print('DINO:',dino_train.shape,'handcrafted:',hand_train.shape)
'''

classical_code=r'''
from collections import Counter
from scipy.special import softmax
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.decomposition import PCA
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from threadpoolctl import threadpool_limits
K_GRID=(1,3,5,7,11);NEIGHBOR_BLEND_GRID=(0.,.1,.2,.3,.4);TREE_COUNT=600;XGB_ROUNDS=360;INNER_SPLITS=3
TREE_RECIPES=[{'base':1.,'et':0.,'xgb':0.},{'base':.9,'et':.1,'xgb':0.},{'base':.9,'et':0.,'xgb':.1},{'base':.8,'et':.1,'xgb':.1},{'base':.8,'et':.2,'xgb':0.},{'base':.8,'et':0.,'xgb':.2},{'base':.6,'et':.2,'xgb':.2}]
def scores(y,prob):
    pred=prob.argmax(1)
    return {'agreement':float(accuracy_score(y,pred)),'macro_f1':float(f1_score(y,pred,labels=list(range(len(CLASSES))),average='macro',zero_division=0))}
def not_degraded(candidate,baseline):
    return candidate['agreement']>=baseline['agreement'] and candidate['macro_f1']>=baseline['macro_f1'] and (candidate['agreement']>baseline['agreement'] or candidate['macro_f1']>baseline['macro_f1'])
def group_neighbors(qx,fx,fy,fit_groups,k):
    sim=qx@fx.T;k=min(k,len(fx));ix=np.argsort(-sim,axis=1,kind='stable')[:,:k]
    top=np.take_along_axis(sim,ix,axis=1);count=Counter(fit_groups.tolist())
    source_weight=np.array([1./count[g] for g in fit_groups]);w=np.exp((top-top.max(1,keepdims=True))/.05)*source_weight[ix]
    pp=np.column_stack([(w*(fy[ix]==c)).sum(1) for c in range(len(CLASSES))]);return pp/np.maximum(pp.sum(1,keepdims=True),1e-12)
def fit_feature_models(x,h,y,g,qx,qh,seed):
    assert set(y)==set(range(len(CLASSES))),f'Classifier fit split is missing class IDs: {sorted(set(range(len(CLASSES)))-set(y))}'
    svm=OneVsRestClassifier(SVC(C=4.,kernel='rbf',gamma='scale'),n_jobs=1).fit(x,y)
    p_svm=softmax(svm.decision_function(qx),axis=1)
    # PCA is fitted inside the training partition, never on validation/test.
    pca=PCA(n_components=min(192,len(x)-1,x.shape[1]),svd_solver='randomized',random_state=seed)
    xt=np.column_stack([pca.fit_transform(x),h]).astype(np.float32);xq=np.column_stack([pca.transform(qx),qh]).astype(np.float32)
    counts=Counter(g.tolist());source_weights=np.array([1./counts[v] for v in g],np.float32)
    et=ExtraTreesClassifier(n_estimators=TREE_COUNT,min_samples_leaf=2,max_features=.6,class_weight='balanced',n_jobs=min(4,os.cpu_count() or 1),random_state=seed)
    et.fit(xt,y,sample_weight=source_weights)
    backend='cuda' if DEVICE.type=='cuda' else 'cpu'
    xgb=xgboost.XGBClassifier(n_estimators=XGB_ROUNDS,max_depth=4,learning_rate=.04,min_child_weight=2.,subsample=.85,colsample_bytree=.75,reg_lambda=8.,reg_alpha=.1,objective='multi:softprob',num_class=len(CLASSES),tree_method='hist',device=backend,max_bin=128,n_jobs=min(4,os.cpu_count() or 1),random_state=seed,eval_metric='mlogloss')
    weights=compute_sample_weight('balanced',y)*source_weights
    try:xgb.fit(xt,y,sample_weight=weights)
    except xgboost.core.XGBoostError as error:
        if backend=='cuda' and ('cuda' in str(error).lower() or 'gpu' in str(error).lower()):
            print('XGBoost CUDA fallback to CPU:',str(error)[:240],flush=True);xgb.set_params(device='cpu');xgb.fit(xt,y,sample_weight=weights)
        else:raise
    assert np.array_equal(et.classes_,np.arange(len(CLASSES))) and np.array_equal(xgb.classes_,np.arange(len(CLASSES)))
    predictions={'svm':p_svm,'et':et.predict_proba(xq),'xgb':xgb.predict_proba(xq),'neighbors':{k:group_neighbors(qx,x,y,g,k) for k in K_GRID}}
    return predictions,{'svm':svm,'pca':pca,'extra_trees':et,'xgboost':xgb}
def choose_knn_recipe(y,svm,neighbors):
    options=[]
    for k in K_GRID:
        for weight in NEIGHBOR_BLEND_GRID:
            metric=scores(y,(1-weight)*svm+weight*neighbors[k])
            options.append((metric['agreement'],metric['macro_f1'],-weight,-k,k,weight))
    best=max(options);return {'k':int(best[4]),'weight':float(best[5])}
def blend_trees(base,et,xgb,recipe):return recipe['base']*base+recipe['et']*et+recipe['xgb']*xgb
def choose_tree_recipe(y,base,et,xgb):
    baseline=scores(y,base);chosen=TREE_RECIPES[0];best=baseline
    for recipe in TREE_RECIPES[1:]:
        metric=scores(y,blend_trees(base,et,xgb,recipe))
        if not_degraded(metric,baseline) and (metric['agreement'],metric['macro_f1'])>(best['agreement'],best['macro_f1']):chosen=recipe;best=metric
    return dict(chosen)
def valid_inner_folds(y,g,seed):
    # Grouped stratification is approximate. Retry deterministic seeds solely for class coverage.
    for attempt in range(16):
        used_seed=seed+attempt
        cv=StratifiedGroupKFold(n_splits=INNER_SPLITS,shuffle=True,random_state=used_seed)
        candidate=list(cv.split(np.zeros(len(y)),y,g))
        if all(set(y[tr])==set(range(len(CLASSES))) for tr,va in candidate):
            for tr,va in candidate:assert set(g[tr]).isdisjoint(set(g[va]))
            return candidate,used_seed
    raise ValueError('Cannot form source-group inner folds with all classes in each fit split.')
def inner_classical_choice(indices,seed):
    y=pseudo[indices];g=groups[indices];n=len(indices)
    inner,inner_seed=valid_inner_folds(y,g,seed)
    sv=np.full((n,len(CLASSES)),np.nan);et=np.full_like(sv,np.nan);xg=np.full_like(sv,np.nan);neighbors={k:np.full_like(sv,np.nan) for k in K_GRID}
    with threadpool_limits(limits=4):
        for fold,(tr,va) in enumerate(inner):
            assert set(g[tr]).isdisjoint(set(g[va]))
            result,bundle=fit_feature_models(dino_train[indices[tr]],hand_train[indices[tr]],y[tr],g[tr],dino_train[indices[va]],hand_train[indices[va]],seed+fold)
            sv[va]=result['svm'];et[va]=result['et'];xg[va]=result['xgb']
            for k in K_GRID:neighbors[k][va]=result['neighbors'][k]
            del bundle;gc.collect()
    assert np.isfinite(sv).all() and np.isfinite(et).all() and np.isfinite(xg).all()
    neighbor_recipe=choose_knn_recipe(y,sv,neighbors)
    neighbor_recipe['inner_split_seed']=inner_seed
    base=(1-neighbor_recipe['weight'])*sv+neighbor_recipe['weight']*neighbors[neighbor_recipe['k']]
    tree_recipe=choose_tree_recipe(y,base,et,xg)
    return neighbor_recipe,tree_recipe

classic_names=['dino_svm','dino_groupknn','extra_trees','xgboost','dino_tree_hybrid']
classic_oof_probs={name:np.full((len(train_frame),len(CLASSES)),np.nan,np.float32) for name in classic_names};classic_choices=[]
with threadpool_limits(limits=4):
    for fold,(tr,va) in enumerate(folds):
        neighbor_recipe,tree_recipe=inner_classical_choice(tr,SEED+2000+fold)
        result,bundle=fit_feature_models(dino_train[tr],hand_train[tr],pseudo[tr],groups[tr],dino_train[va],hand_train[va],SEED+3000+fold)
        base=(1-neighbor_recipe['weight'])*result['svm']+neighbor_recipe['weight']*result['neighbors'][neighbor_recipe['k']]
        classic_oof_probs['dino_svm'][va]=result['svm'];classic_oof_probs['dino_groupknn'][va]=base
        classic_oof_probs['extra_trees'][va]=result['et'];classic_oof_probs['xgboost'][va]=result['xgb']
        classic_oof_probs['dino_tree_hybrid'][va]=blend_trees(base,result['et'],result['xgb'],tree_recipe)
        classic_choices.append({'fold':fold,'neighbor':neighbor_recipe,'tree':tree_recipe})
        print('Classic nested outer fold',fold,classic_choices[-1],flush=True);del bundle;gc.collect()
classic_summaries=[]
for name,pp in classic_oof_probs.items():
    assert np.isfinite(pp).all() and np.allclose(pp.sum(1),1,atol=1e-4)
    metric=scores(pseudo,pp);classic_summaries.append({'model':name,'proxy_agreement':metric['agreement'],'proxy_macro_f1':metric['macro_f1']})
    pd.DataFrame(pp,columns=CLASSES).assign(sample_id=train_ids,fold=fold_ids,target=train_frame.assigned_class).to_csv(WORK_ROOT/f'validation/oof_{name}.csv',index=False)
baseline_metric=scores(pseudo,classic_oof_probs['dino_groupknn']);hybrid_metric=scores(pseudo,classic_oof_probs['dino_tree_hybrid'])
CLASSIC_SELECTED_NAME='dino_tree_hybrid' if not_degraded(hybrid_metric,baseline_metric) else 'dino_groupknn'
full_neighbor,full_tree=inner_classical_choice(eligible_idx,SEED+4000)
with threadpool_limits(limits=4):
    result,bundle=fit_feature_models(dino_train,hand_train,pseudo,groups,dino_test,hand_test,SEED+5000)
base=(1-full_neighbor['weight'])*result['svm']+full_neighbor['weight']*result['neighbors'][full_neighbor['k']]
classic_final_probabilities={'dino_svm':result['svm'],'dino_groupknn':base,'extra_trees':result['et'],'xgboost':result['xgb'],'dino_tree_hybrid':blend_trees(base,result['et'],result['xgb'],full_tree)}
bundle['xgboost'].save_model(WORK_ROOT/'checkpoints/final_xgboost.json')
joblib.dump({k:v for k,v in bundle.items() if k!='xgboost'},WORK_ROOT/'checkpoints/final_dino_tree_models.joblib',compress=3)
np.savez_compressed(WORK_ROOT/'checkpoints/final_dino_knn_fit.npz',x=dino_train,y=pseudo,groups=groups,classes=np.asarray(CLASSES))
for name,pp in classic_final_probabilities.items():
    write_submission(pp,WORK_ROOT/f'predictions/submission_{name}.csv')
    pd.DataFrame(pp,columns=CLASSES).assign(image_id=test_ids).to_csv(WORK_ROOT/f'predictions/test_probabilities_{name}.csv',index=False)
write_submission(classic_final_probabilities[CLASSIC_SELECTED_NAME],WORK_ROOT/'DSC2026_Submission.csv')
classic_report={'selected_family':CLASSIC_SELECTED_NAME,'outer_folds':N_SPLITS,'inner_folds':INNER_SPLITS,'outer_choices':classic_choices,'full_train_neighbor':full_neighbor,'full_train_tree':full_tree,'tree_count':TREE_COUNT,'xgb_rounds':XGB_ROUNDS,'xgb_actual_device':json.loads(bundle['xgboost'].get_booster().save_config()).get('learner',{}).get('generic_param',{}).get('device'),'pca':'192 components max, fitted independently within every fit split','selection_rule':'train-only inner validation; retain group-kNN baseline unless tree hybrid is nondegraded on paired outer agreement and macro-F1','outer_metrics':classic_summaries,'data_state_sha256':STATE_SHA256,'test_rows_in_fit':0,'probability_note':'SVM softmax margins are ranking scores, not calibrated confidence'}
(WORK_ROOT/'logs/classical_selection.json').write_text(json.dumps(classic_report,indent=2),encoding='utf-8')
display(pd.DataFrame(classic_summaries));print('Classical submission already saved; selected:',CLASSIC_SELECTED_NAME)
del bundle;gc.collect();torch.cuda.empty_cache()
'''

deep_cv=deep_cv.replace('oof_probs={}; proxy_summary=[]','oof_probs=dict(classic_oof_probs); proxy_summary=list(classic_summaries)')
deep_cv=deep_cv.replace('oof_ensemble=np.mean(list(oof_probs.values()),axis=0)','oof_ensemble=np.mean([oof_probs[name] for name in MODEL_NAMES],axis=0)')
deep_cv+='''
fixed_hybrid_oof=.8*classic_oof_probs[CLASSIC_SELECTED_NAME]+.2*oof_ensemble
PROMOTE_NEURAL_BLEND=not_degraded(scores(pseudo,fixed_hybrid_oof),scores(pseudo,classic_oof_probs[CLASSIC_SELECTED_NAME]))
hybrid_metric=scores(pseudo,fixed_hybrid_oof)
proxy_summary.append({'model':'fixed_80classic_20neural','proxy_agreement':hybrid_metric['agreement'],'proxy_macro_f1':hybrid_metric['macro_f1']})
pd.DataFrame(proxy_summary).to_csv(WORK_ROOT/'validation/proxy_summary.csv',index=False)
print('Promote fixed global CNN blend:',PROMOTE_NEURAL_BLEND,'| baseline:',CLASSIC_SELECTED_NAME)
'''
refit_src=refit_src.replace("ensemble=np.mean(list(model_probabilities.values()),axis=0)","cnn_ensemble=np.mean(list(model_probabilities.values()),axis=0)\nwrite_submission(cnn_ensemble,WORK_ROOT/'predictions/submission_neural_ensemble.csv')\nhybrid_ensemble=.8*classic_final_probabilities[CLASSIC_SELECTED_NAME]+.2*cnn_ensemble\nwrite_submission(hybrid_ensemble,WORK_ROOT/'predictions/submission_classic_neural_hybrid.csv')\nensemble=hybrid_ensemble if PROMOTE_NEURAL_BLEND else classic_final_probabilities[CLASSIC_SELECTED_NAME]\nFINAL_ENSEMBLE_POLICY={'classic_family':CLASSIC_SELECTED_NAME,'cnn_blend_promoted':bool(PROMOTE_NEURAL_BLEND),'weights':{'classic':.8,'cnn_mean':.2} if PROMOTE_NEURAL_BLEND else {'classic':1.,'cnn_mean':0.},'guard':'paired grouped OOF agreement and macro-F1 must both be nondegraded; at least one strict gain; model-selection comparison is not an untouched estimate'}")
export_src=''.join(nb['cells'][19]['source'])
export_src=export_src.replace("'ensemble_policy':'global equal mean probabilities, fixed before test inference'","'ensemble_policy':FINAL_ENSEMBLE_POLICY,'classical_selection':classic_report,'dino_feature_provenance':DINO_PROVENANCE")

nb['cells'][15]=code(definitions)
nb['cells']=nb['cells'][:16]+[
md('''## 8. DINOv3 dan fitur warna/bentuk

Jalur yang sudah digunakan pada eksperimen sebelumnya: DINOv3 Base image-only, checkpoint/revisi/hash dibekukan, descriptor CLS + patch valid + foreground mean + foreground std. Anggota train, label, group dan ROI tetap persis processed aktif. View native ROI dipakai sebelum resize model agar resep fitur lama terjaga. Test hanya ditransformasi dengan encoder beku. HSV/Lab/spatial color, HOG dan LBP menambah informasi warna/bentuk bagi tree models.
'''),code(feature_code),
md('''## 9. SVM, group-kNN, Extra Trees, dan XGBoost

RBF-SVM dan kNN berbobot kelompok sumber menjadi baseline. Extra Trees dan XGBoost memakai PCA fitur DINO yang di-fit pada partisi train ditambah fitur warna/tekstur. Pilihan k, bobot tetangga, dan campuran tree dipilih lewat inner grouped validation; outer validation tidak ikut fitting atau pemilihan tersebut. Seluruh model mempertahankan assignment processed. Baseline, tree individual dan campurannya diekspor terpisah. CSV utama awal tersedia sebelum training CNN agar hasil klasik tidak hilang bila sesi berhenti kemudian.
'''),code(classical_code),
md('''## 10. Evaluasi CNN pada source fold yang sama

Empat CNN dilatih seperti jadwal sebelumnya. Ensemble CNN global dibandingkan dengan baseline klasik menggunakan campuran tetap 80% klasik dan 20% CNN. Campuran hanya dipromosikan bila agreement dan macro-F1 grouped OOF tidak turun dan sedikitnya satu naik. Ini perbandingan internal untuk seleksi model, bukan estimasi untouched final accuracy; tidak ada label atau aturan per ID test.
'''),code(deep_cv),
md('''## 11. Refit CNN, inferensi, dan final submission

Refit seluruh 1.886 sampel processed, satu backbone per giliran. Simpan CSV setiap model, ensemble neural, dan campuran klasik-neural. `DSC2026_Submission.csv` mengikuti kebijakan global yang ditetapkan dari train; model tambahan yang tidak lolos guard tidak menggantikan baseline klasik.
'''),code(refit_src),
md('''## 12. Ekspor log, kurva, hash, dan artefak

Manifest merekam label state, descriptor DINOv3, sumber bobot, seluruh evaluasi kandidat, recipe inner/outer, checkpoint, dan kebijakan submission. Semua keluaran disimpan di working dan paket ZIP.
'''),code(export_src)]
first=''.join(nb['cells'][0]['source'])
first+='\n\nVersi hybrid v3 menambahkan jalur DINOv3/SVM/group-kNN serta Extra Trees dan XGBoost. Kecocokan dengan CSV pembanding tidak digunakan oleh notebook untuk fitting atau seleksi. Hasil target test tidak dijamin; kesamaan processed tetap diperiksa sebelum training.\n'
nb['cells'][0]['source']=first.splitlines(True)
cfg=''.join(nb['cells'][1]['source']).replace("Pipeline: active_processed_port_v2","Pipeline: active_processed_hybrid_v3")
nb['cells'][1]['source']=cfg.splitlines(True)
nb['metadata']['gelarrasa_pipeline']['version']='active_processed_hybrid_v3'
for i,c in enumerate(nb['cells']):c['id']=f'gelarrasa-hybrid-{i:02d}'
TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
readme=TARGET.parent/'README.md'
readme.write_text(readme.read_text(encoding='utf-8')+'\n## Model hybrid v3\n\nDINOv3 Base empat blok + RBF-SVM dan group-weighted kNN dipulihkan sebagai baseline. Extra Trees 600 pohon dan XGBoost 360 boosting rounds diuji melalui nested grouped validation dengan PCA fold-train serta HSV/Lab/HOG/LBP. Kandidat individual selalu diekspor. CNN blend global tetap 80/20 hanya dipromosikan jika paired grouped OOF tidak turun pada agreement dan macro-F1. Notebook tidak membaca target CSV pembanding untuk fitting/seleksi, dan hasil selisih historis tidak dijamin sama dengan run baru. Cetakan versi: `active_processed_hybrid_v3`.\n',encoding='utf-8')
print('Enhanced target notebook:',TARGET,'cells',len(nb['cells']))

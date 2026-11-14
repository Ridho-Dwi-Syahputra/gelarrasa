"""Verify actual DINO recipe parity and exercise nested tree/classic notebook code on CPU."""
import ast,base64,gc,hashlib,json,os,tempfile,zlib
from pathlib import Path
import cv2,joblib,numpy as np,pandas as pd,torch,timm,xgboost
from PIL import Image,ImageOps
from torch.nn import functional as F
from huggingface_hub import hf_hub_download
from sklearn.preprocessing import normalize
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score
from skimage.feature import hog,local_binary_pattern

ROOT=Path(__file__).resolve().parents[3]
TARGET=ROOT/'final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb'
nb=json.loads(TARGET.read_text(encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code':compile(''.join(c['source']),f'cell{i}','exec')
scope=dict(globals());scope.update(DEVICE=torch.device('cpu'),SEED=2026)
scope['display']=lambda obj:None
state_tree=ast.parse(''.join(nb['cells'][7]['source']))
state_vars={n.targets[0].id:ast.literal_eval(n.value) for n in state_tree.body if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['FROZEN_STATE_B64','STATE_SHA256']}
state=json.loads(zlib.decompress(base64.b64decode(state_vars['FROZEN_STATE_B64'])))
scope.update(CLASSES=state['classes'],STATE_SHA256=state_vars['STATE_SHA256'])
rows=pd.DataFrame(state['rows']);tr=rows[rows.split.eq('train')&rows.training_eligible].reset_index(drop=True);te=rows[rows.split.eq('test')].reset_index(drop=True)
for frame in [tr,te]:
    frame['source_path']=[str(ROOT/f'data/Dataset DSC GelarRasa 2026/{split}'/file) for split,file in zip(frame.split,frame.source_file)]
for cell in [9,17]:
    tree=ast.parse(''.join(nb['cells'][cell]['source']))
    defs=[n for n in tree.body if isinstance(n,ast.FunctionDef)]
    exec(compile(ast.Module(body=defs,type_ignores=[]),f'defs{cell}','exec'),scope)
feature_src=''.join(nb['cells'][17]['source']);tree=ast.parse(feature_src)
for n in tree.body:
    if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id.startswith('D3_'):
        scope[n.targets[0].id]=ast.literal_eval(n.value)
torch.set_num_threads(2)
cache_file=Path(__file__).with_name('hybrid_local_verified_features.npz')
cache_identity=scope['STATE_SHA256']+scope['D3_WEIGHT_SHA256']+scope['D3_RECIPE']
if cache_file.exists():
    z=np.load(cache_file,allow_pickle=False)
    assert str(z['identity'].item())==cache_identity
    dino=z['dino'];hand=z['hand'];parity=float(z['parity'])
    print('Using locally validated feature cache',flush=True)
else:
    prior=np.load(ROOT/'data/processed/dinov3_full_ovr_features.npz',allow_pickle=False)
    keys=prior['keys'].astype(str);hashes=prior['source_sha256'].astype(str);features=prior['features'];idx={k:i for i,k in enumerate(keys)}
    prior_identity=hashlib.sha256((scope['D3_WEIGHT_SHA256']+scope['D3_RECIPE']+'|'.join(keys)+'|'.join(hashes)).encode()).hexdigest()
    assert prior_identity==str(prior['identity'].item())
    try:weight=hf_hub_download(scope['D3_REPOSITORY'],'model.safetensors',revision=scope['D3_REVISION'],local_files_only=True)
    except Exception:weight=hf_hub_download(scope['D3_REPOSITORY'],'model.safetensors',revision=scope['D3_REVISION'])
    assert scope['file_sha256'](weight)==scope['D3_WEIGHT_SHA256']
    encoder=timm.create_model('vit_base_patch16_dinov3.lvd1689m',pretrained=True,num_classes=0,pretrained_cfg_overlay={'file':str(weight)}).eval()
    encoder.rope.periods=encoder.rope.periods.to(torch.bfloat16).to(torch.float32)
    cfg=timm.data.resolve_model_data_config(encoder);mean=np.asarray(cfg['mean'],np.float32);std=np.asarray(cfg['std'],np.float32)
    probe=tr[~tr.is_crop].iloc[:2].to_dict('records')
    with torch.inference_mode():fresh=scope['d3_descriptor_batch'](encoder,[scope['d3_prepare'](r,mean,std) for r in probe])
    expected=np.stack([features[idx['train:'+r['parent_id']]] for r in probe]);parity=float(np.max(np.abs(fresh-expected)))
    assert parity<5e-5,f'DINO recipe parity failed: {parity}'
    print('Actual DINOv3 checkpoint/descriptor parity PASS; max abs diff',parity,flush=True)
    all_rows=pd.concat([tr,te],ignore_index=True);dino=np.zeros((len(all_rows),3072),np.float32)
    source_lookup={(s['split'],s['file_name']):s['sha256'] for s in state['sources']}
    for i,row in all_rows.iterrows():
        if row.is_crop:
            with torch.inference_mode():dino[i]=scope['d3_descriptor_batch'](encoder,[scope['d3_prepare'](row.to_dict(),mean,std)])[0]
        else:
            key=f'{row.split}:{row.parent_id}';j=idx[key]
            assert hashes[j]==source_lookup[(row.split,row.source_file)]
            dino[i]=features[j]
    del encoder;gc.collect()
    hand=np.stack([scope['hand_descriptor'](r) for r in all_rows.to_dict('records')])
    np.savez_compressed(cache_file,dino=dino,hand=hand,identity=np.array(cache_identity),parity=np.array(parity))
    print('DINO descriptors for all 7 crops and hand features ready',hand.shape,flush=True)

indices=[]
for name in state['classes']:
    indices.extend(tr[tr.assigned_class.eq(name)].drop_duplicates('parent_group').head(12).index)
indices=np.array(indices,dtype=int);sub=tr.iloc[indices].reset_index(drop=True)
scope.update(train_frame=sub,test_frame=te,dino_train=normalize(dino[indices]),dino_test=normalize(dino[len(tr):]),hand_train=hand[indices],hand_test=hand[len(tr):],pseudo=np.array([state['classes'].index(c) for c in sub.assigned_class]),groups=sub.parent_group.to_numpy(),train_ids=sub.sample_id.tolist(),test_ids=te.parent_id.map(lambda x:str(x).zfill(4)).tolist(),eligible_idx=np.arange(len(sub)),N_SPLITS=3)
scope['sample']=pd.read_csv(ROOT/'data/SampleSubmission_DSC.csv',dtype={'image_id':str});scope['sample']['image_id_norm']=scope['sample'].image_id.str.zfill(4)
scope['folds']=list(StratifiedGroupKFold(3,shuffle=True,random_state=2026).split(np.zeros(len(sub)),scope['pseudo'],scope['groups']))
fold_ids=np.full(len(sub),-1)
for fold,(_,va) in enumerate(scope['folds']):fold_ids[va]=fold
scope['fold_ids']=fold_ids
submission_def=next(n for n in ast.parse(''.join(nb['cells'][15]['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='write_submission')
exec(compile(ast.Module(body=[submission_def],type_ignores=[]),'submission','exec'),scope)
classical=''.join(nb['cells'][19]['source']);point=classical.index('classic_names=')
exec(compile(classical[:point],'classical_definitions','exec'),scope)
scope.update(TREE_COUNT=16,XGB_ROUNDS=8,INNER_SPLITS=2)
with tempfile.TemporaryDirectory(prefix='.hybrid_model_check_',dir=Path(__file__).parent) as td:
    temp=Path(td).resolve();assert temp.parent==Path(__file__).resolve().parent
    for d in ['validation','predictions','checkpoints','logs']:(temp/d).mkdir()
    scope['WORK_ROOT']=temp
    exec(compile(classical[point:],'classical_nested_smoke','exec'),scope)
    final=pd.read_csv(temp/'DSC2026_Submission.csv',dtype={'image_id':str})
    assert final.image_id.tolist()==scope['sample'].image_id.tolist() and len(final)==480
    for name,pp in scope['classic_final_probabilities'].items():assert pp.shape==(480,8) and np.isfinite(pp).all() and np.allclose(pp.sum(1),1,atol=1e-5)
    restored=joblib.load(temp/'checkpoints/final_dino_tree_models.joblib')
    restored_xgb=xgboost.XGBClassifier();restored_xgb.load_model(temp/'checkpoints/final_xgboost.json')
    xt=np.column_stack([restored['pca'].transform(scope['dino_test']),scope['hand_test']]).astype(np.float32)
    assert np.allclose(restored_xgb.predict_proba(xt),scope['classic_final_probabilities']['xgboost'],atol=1e-6)
    assert not scope['not_degraded']({'agreement':.8,'macro_f1':.9},{'agreement':.9,'macro_f1':.9})
    report={'notebook_sha256':hashlib.sha256(TARGET.read_bytes()).hexdigest(),'state_sha256':scope['STATE_SHA256'],'actual_dinov3_weight_sha256_verified':True,'native_descriptor_parity_max_abs_difference':parity,'native_crop_features_computed':7,'hand_feature_dimensions':int(hand.shape[1]),'nested_classifier_cpu_smoke_rows':len(sub),'outer_folds':3,'inner_folds':2,'smoke_tree_count':16,'smoke_xgb_rounds':8,'production_tree_count':600,'production_xgb_rounds':360,'candidate_probability_and_csv_checks':'pass','xgboost_serialization_roundtrip':'pass','baseline_non_degradation_guard':'pass','test_targets_used':False,'gpu_training_run':False,'timm':timm.__version__,'xgboost':xgboost.__version__}
Path(__file__).with_name('kaggle_hybrid_validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print('Nested SVM/group-kNN/ExtraTrees/XGBoost + CSV + serialized model: PASS',flush=True)

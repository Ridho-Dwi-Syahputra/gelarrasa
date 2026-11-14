"""Exercise real notebook cells against raw data and run CPU model/loop smoke checks."""
import ast,base64,gc,hashlib,json,os,random,sys,tempfile,time,zipfile,zlib
from pathlib import Path
import numpy as np,pandas as pd
from PIL import Image,ImageOps
import torch,torchvision
from torchvision import models,transforms
from torchvision.transforms import InterpolationMode
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torch.nn import functional as F
from sklearn.metrics import accuracy_score,f1_score,log_loss
from sklearn.model_selection import StratifiedGroupKFold
from scipy.fft import dctn
from skimage.metrics import structural_similarity

ROOT=Path(__file__).resolve().parents[3]
NOTEBOOK=ROOT/'final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb'
REPORT=Path(__file__).with_name('kaggle_processed_port_validation.json')
nb=json.loads(NOTEBOOK.read_text(encoding='utf-8'))
for i,c in enumerate(nb['cells']):
    if c['cell_type']=='code': compile(''.join(c['source']),f'cell_{i}','exec')
print('Notebook code compilation: PASS',flush=True)
start=time.time(); report={'notebook':str(NOTEBOOK),'syntax_valid':True,'gpu_training_executed':False}
torch.set_num_threads(2)
scope=dict(globals())
with tempfile.TemporaryDirectory(prefix='.kaggle_port_check_',dir=Path(__file__).parent) as td:
    temp=Path(td).resolve()
    assert temp.parent==Path(__file__).resolve().parent
    scope['WORK_ROOT']=temp
    for d in ['labels','eda','logs','checkpoints']: (temp/d).mkdir()
    src=''.join(nb['cells'][3]['source'])
    src=src.replace("SEARCH_ROOTS=[Path('/kaggle/input'),Path('/kaggle/working'),Path('/kaggle/temp'),Path('/content'),Path('/mnt/data')]",f'SEARCH_ROOTS=[Path({str(ROOT/"data")!r})]')
    exec(compile(src,'discovery','exec'),scope)
    exec(compile(''.join(nb['cells'][7]['source']),'snapshot','exec'),scope)
    print('Executing materialization on all raw sources...',flush=True)
    exec(compile(''.join(nb['cells'][9]['source']),'preprocessing','exec'),scope)
    original=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'sample_id':str,'parent_id':str},keep_default_na=False)
    regenerated=scope['processed']
    columns=['split','sample_id','parent_id','parent_group','assigned_class','is_crop','human_reviewed','training_eligible','label_source']
    old=original[columns].sort_values(['split','sample_id']).reset_index(drop=True)
    new=regenerated[columns].sort_values(['split','sample_id']).reset_index(drop=True)
    pd.testing.assert_frame_equal(old,new,check_dtype=False)
    assert len(scope['train_paths'])==1886
    report['processing']=scope['proof']; report['assignment_rows_equal_active_manifest']=True
    assert original[original.split.eq('train') & original.training_eligible].groupby('native_pixel_sha256').parent_group.nunique().le(1).all()
    for tr,va in StratifiedGroupKFold(n_splits=3,shuffle=True,random_state=2026).split(np.zeros(len(scope['train_frame'])),scope['pseudo'],scope['groups']):
        assert set(scope['groups'][tr]).isdisjoint(set(scope['groups'][va]))
        assert set(scope['pseudo'][tr])==set(range(8))
    report['full_population_source_group_split']='pass'
    print('Full materialization + label/group/eligibility parity: PASS',flush=True)
    # Load notebook definitions without executing network downloads or the full training run.
    for cell in [13,15]:
        tree=ast.parse(''.join(nb['cells'][cell]['source']))
        definitions=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.ClassDef))]
        exec(compile(ast.Module(body=definitions,type_ignores=[]),f'defs_{cell}','exec'),scope)
    scope.update(DEVICE=torch.device('cpu'),MEAN=(.485,.456,.406),STD=(.229,.224,.225),MODEL_IMAGE_SIZE=224,USE_MULTI_GPU=False,WORKERS=0,BATCH_SIZE=4,HEAD_EPOCHS=1,FINETUNE_EPOCHS=2,SEED=2026,CONFIG_HASH='local_cpu_loop_smoke',MODEL_NAMES=['resnet50','efficientnet_v2_s','convnext_small','swin_v2_t'])
    bits,pixels=scope['pixel_descriptor'](scope['test_paths'][0])
    assert bits.shape==(64,) and pixels.shape==(64,64)
    assert abs(structural_similarity(pixels,pixels,data_range=1.)-1.)<1e-6
    report['pixel_similarity_descriptors']='pass'
    export_cell=next(c for c in nb['cells'] if c['cell_type']=='code' and 'def write_submission(' in ''.join(c['source']))
    export_defs=[n for n in ast.parse(''.join(export_cell['source'])).body if isinstance(n,ast.FunctionDef) and n.name=='write_submission']
    exec(compile(ast.Module(body=export_defs,type_ignores=[]),'submission_definition','exec'),scope)
    probabilities=np.eye(8,dtype=np.float32)[np.arange(480)%8]
    out=scope['write_submission'](probabilities,temp/'smoke_submission.csv')
    assert out.image_id.tolist()==scope['sample'].image_id.tolist() and len(out)==480
    report['sample_order_and_480_row_csv']='pass'
    def seed_everything(seed): random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    scope['seed_everything']=seed_everything
    real_build=scope['build_model']; architectural=[]
    for name in scope['MODEL_NAMES']:
        m=real_build(name,False).eval()
        assert scope['get_head'](m,name).out_features==8
        with torch.inference_mode(): result=m(torch.zeros(1,3,224,224))
        assert tuple(result.shape)==(1,8) and torch.isfinite(result).all()
        architectural.append(name); del m,result; gc.collect()
        print('Architecture + lazy head + forward:',name,'PASS',flush=True)
    report['architecture_forward_pass']=architectural
    # Use actual processed images, one train and one held-out source per class.
    frame=scope['train_frame']; tr=[]; va=[]
    for cls in scope['CLASSES']:
        part=frame[frame.assigned_class.eq(cls)]
        first=part.index[0]; second=part[part.parent_group.ne(frame.loc[first,'parent_group'])].index[0]
        tr.append(first); va.append(second)
    scope['build_model']=lambda name,pretrained=True:real_build(name,False)
    model,history,checkpoint=scope['fit_one']('resnet50',np.array(tr),np.array(va),2,'cv_smoke')
    assert len(history)==2 and all('val_proxy_macro_f1' in row for row in history)
    del model; gc.collect(); report['head_and_finetune_validation_loop']='pass'
    print('Actual CPU head/full fine-tune + validation loop: PASS',flush=True)
    # Interrupt within the fine-tuning stage, then resume its optimizer and scheduler.
    original_epoch=scope['train_epoch']; calls=[0]
    def interrupt_epoch(*args,**kwargs):
        calls[0]+=1
        if calls[0]==3: raise RuntimeError('intentional interruption for resume proof')
        return original_epoch(*args,**kwargs)
    scope['train_epoch']=interrupt_epoch
    try: scope['fit_one']('resnet50',np.array(tr),None,3,'resume_smoke')
    except RuntimeError as e:
        assert 'intentional interruption' in str(e)
    else: raise AssertionError('interruption did not occur')
    scope['train_epoch']=original_epoch; gc.collect()
    model,history,checkpoint=scope['fit_one']('resnet50',np.array(tr),None,3,'resume_smoke')
    assert len(history)==3 and [r['epoch'] for r in history]==[1,2,3]
    assert not any('val_proxy_macro_f1' in row for row in history)
    report['within_finetune_optimizer_scheduler_resume']='pass'
    report['final_refit_has_no_fabricated_validation_metrics']=True
    del model; gc.collect()
    print('Within-stage optimizer/scheduler/scaler/RNG resume: PASS',flush=True)
report['seconds']=round(time.time()-start,2)
report['notebook_sha256']=hashlib.sha256(NOTEBOOK.read_bytes()).hexdigest()
REPORT.write_text(json.dumps(report,indent=2),encoding='utf-8')
print('REPORT:',REPORT,flush=True)

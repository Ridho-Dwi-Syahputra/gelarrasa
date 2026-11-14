"""Run the Kaggle preprocessing and train-only assignment locally on CPU."""
import gc
import hashlib
import json
import os
import tempfile
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[3]
NOTEBOOK = json.loads((ROOT/'final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb').read_text(encoding='utf-8'))
for i, cell in enumerate(NOTEBOOK['cells']):
    if cell['cell_type']=='code':
        compile(''.join(cell['source']), f'cell_{i}', 'exec')

RAW = ROOT/'data/Dataset DSC GelarRasa 2026'
IMAGE_EXTS={'.jpg','.jpeg','.png','.webp','.bmp'}
def files(split):
    return sorted((p for p in (RAW/split).iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS), key=lambda p:(p.stem,p.name))

with tempfile.TemporaryDirectory(prefix='kaggle_native_',dir=str(ROOT/'docs/agent-notes/experiments')) as temp:
    work=Path(temp)
    for d in ['eda','processed','labels','validation','checkpoints','predictions','logs','figures']:
        (work/d).mkdir(parents=True,exist_ok=True)
    ns={'os':os,'json':json,'hashlib':hashlib,'gc':gc,'Path':Path,'np':np,'pd':pd,
        'Image':Image,'ImageOps':ImageOps,'torch':torch,'plt':plt,'WORK_ROOT':work,
        'DEVICE':torch.device('cpu'),'SEED':2026,
        'CLASSES':['cardboard','electronic','foodwaste','glass','metal','paper','plastic','textile'],
        'raw_train_paths':files('train'),'raw_test_paths':files('test'),
        'sample':pd.read_csv(ROOT/'data/SampleSubmission_DSC.csv',dtype={'image_id':str})}
    ns['norm_id']=lambda value:str(value).strip().split('.')[0].zfill(4)
    ns['sample']['image_id_norm']=ns['sample'].image_id.map(ns['norm_id'])
    torch.set_num_threads(6)
    start=time.time()
    print('Executing native stage cell 7...',flush=True)
    exec(''.join(NOTEBOOK['cells'][7]['source']),ns)
    print('Executing clustering/materialization cell 9...',flush=True)
    exec(''.join(NOTEBOOK['cells'][9]['source']),ns)
    print('PASS:',len(ns['train_frame']),'eligible train,',len(ns['test_frame']),'test,',round(time.time()-start,1),'seconds',flush=True)
    print('Class counts:',ns['train_frame'].assigned_class.value_counts().to_dict(),flush=True)
    active=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'sample_id':str},keep_default_na=False)
    active=active[active.split.eq('train') & active.training_eligible][['sample_id','assigned_class']]
    joined=ns['train_frame'][['sample_id','assigned_class']].merge(active,on='sample_id',suffixes=('_new','_old'),validate='one_to_one')
    comparison={'shared_rows':len(joined),'same_class':int(joined.assigned_class_new.eq(joined.assigned_class_old).sum()),
        'different_class':int(joined.assigned_class_new.ne(joined.assigned_class_old).sum()),
        'class_counts':ns['train_frame'].assigned_class.value_counts().to_dict(),
        'source':'Read-only diagnostic after automatic assignment; old labels are not inputs to the Kaggle notebook.'}
    print('Postfreeze local comparison:',comparison,flush=True)
    (ROOT/'docs/agent-notes/experiments/kaggle_native_stage_diagnostic.json').write_text(json.dumps(comparison,indent=2),encoding='utf-8')

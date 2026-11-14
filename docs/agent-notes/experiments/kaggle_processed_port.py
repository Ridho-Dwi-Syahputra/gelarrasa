"""Build the Kaggle notebook from the active processed manifest, with explicit frozen provenance."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import base64
import hashlib
import json
import textwrap
import zlib
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[3]
TARGET = ROOT / 'final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb'
OLD = json.loads(TARGET.read_text(encoding='utf-8'))
LOCAL = pd.read_csv(ROOT/'data/processed/manifest.csv', dtype={'sample_id':str,'parent_id':str}, keep_default_na=False)
PROVENANCE = json.loads((ROOT/'data/processed/provenance.json').read_text(encoding='utf-8'))
CLASSES = ['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()

def pixel_hash(path):
    with Image.open(path) as im: return hashlib.sha256(im.convert('RGB').tobytes()).hexdigest()

def thumbnail_signature(path):
    import numpy as np
    with Image.open(path) as im:
        pixels=np.asarray(im.convert('RGB'),dtype=np.uint8)
    assert pixels.shape==(224,224,3)
    tiles=np.rint(pixels.reshape(16,14,16,14,3).mean(axis=(1,3))).astype(np.uint8)
    return base64.b64encode(zlib.compress(tiles.tobytes(),9)).decode('ascii')

def row_spec(row):
    return {
        'split':row['split'], 'sample_id':row['sample_id'], 'parent_id':row['parent_id'],
        'parent_group':row['parent_group'], 'source_file':Path(row['source_path']).name,
        'bbox':[int(row[k]) for k in ['x0','y0','x1','y1']], 'is_crop':bool(row['is_crop']),
        'region_name':row['region_name'], 'assigned_class':row['assigned_class'],
        'label_source':row['label_source'], 'human_reviewed':bool(row['human_reviewed']),
        'training_eligible':bool(row['training_eligible']),
        'expected_pixel_sha256':pixel_hash(ROOT/row['output_path']),
        'expected_thumbnail16_b64':thumbnail_signature(ROOT/row['output_path']),
        'deferral_reason':row['deferral_reason'],
    }

with ThreadPoolExecutor(max_workers=4) as pool:
    specs=list(pool.map(row_spec, LOCAL.to_dict('records')))

sources=[]
for split in ['train','test']:
    paths=sorted(p for p in (ROOT/f'data/Dataset DSC GelarRasa 2026/{split}').iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.webp','.bmp'})
    with ThreadPoolExecutor(max_workers=4) as pool:
        hashes=list(pool.map(digest,paths))
    sources.extend({'split':split,'file_name':p.name,'sha256':h} for p,h in zip(paths,hashes))

eligible=LOCAL[LOCAL.split.eq('train') & LOCAL.training_eligible]
state={
    'version':'active_processed_port_v3', 'classes':CLASSES,
    'origin':'Frozen operational output of the existing train-only assignment, cluster refinement and preprocessing workflow. Cluster/assignment fitting is NOT rerun by this portability notebook.',
    'source_processed_manifest_sha256':digest(ROOT/'data/processed/manifest.csv'),
    'source_preprocessing_bundle_sha256':digest(ROOT/'data/preprocessing_inputs.npz'),
    'policy':PROVENANCE['policy'], 'sources':sources, 'rows':specs,
    'expected_counts':{'raw_train':1920,'raw_test':480,'eligible_train':len(eligible),'eligible_full_frame':int((~eligible.is_crop).sum()),'eligible_crops':int(eligible.is_crop.sum()),'held_train_samples':int(((LOCAL.split=='train') & ~LOCAL.training_eligible).sum()),'processed_test':480},
    'class_counts':eligible.assigned_class.value_counts().reindex(CLASSES).astype(int).to_dict(),
    'human_seed_counts':eligible[eligible.human_reviewed].assigned_class.value_counts().reindex(CLASSES,fill_value=0).astype(int).to_dict(),
}
assert state['expected_counts']['eligible_train']==1886
assert all(not r['assigned_class'] for r in specs if r['split']=='test')
raw_state=json.dumps(state,ensure_ascii=False,sort_keys=True,separators=(',',':')).encode()
state_hash=hashlib.sha256(raw_state).hexdigest()
encoded=base64.b64encode(zlib.compress(raw_state,9)).decode()

def md(s): return {'cell_type':'markdown','metadata':{},'source':textwrap.dedent(s).strip().splitlines(True)}
def code(s): return {'cell_type':'code','execution_count':None,'metadata':{},'outputs':[],'source':textwrap.dedent(s).strip().splitlines(True)}

cells=[md('''
# GELAR RASA 2026 — Processed Train, GPU Training, Final Submission

Notebook ini mempertahankan **assignment kategori, source group, crop, dan pengecualian processed aktif** dari alur utama. Seluruh piksel dibuat ulang dari dataset GelarRasa yang terpasang di Kaggle. Snapshot metadata algoritmis disertakan secara eksplisit di dalam notebook; tidak ada file label atau cache lokal yang perlu dipasang.

**Asal label:** snapshot hasil proses train-only sebelumnya (termasuk 37 seed yang ditinjau tim). Notebook ini **tidak mengulang fitting seluruh metode clustering/pelabelan historis**. Ia memverifikasi hash sumber dan menerapkan ulang preprocessing. Hash piksel dibandingkan persis terlebih dahulu; perbedaan kecil lintas runtime diperiksa dengan ringkasan warna blok 16x16 sebelum training. Metadata test berisi identitas dan hash, tanpa target kategori.

Target port aktif: **1.879 foto utuh + 7 crop = 1.886 sampel train**, 27 turunan train ditahan, dan 480 gambar test. Angka 1.869 merupakan versi processed terdahulu. Training di notebook ini memakai seluruh 1.886 sampel eligible; tidak ada filter confidence atau pseudo-labeler baru yang mengganti assignment.

Keluaran: delapan folder processed train, processed test, manifest dan bukti kesamaan, EDA/preview, fold/OOF, riwayat serta checkpoint GPU, probabilitas per backbone, dan `DSC2026_Submission.csv`. Metrik OOF mengukur agreement terhadap assignment algoritmis yang dibekukan, bukan ground truth independen.
''')]

cells.append(code('''
import os,sys,json,time,random,hashlib,platform,zipfile,shutil,gc,base64,zlib
from pathlib import Path
from datetime import datetime,timezone
import numpy as np,pandas as pd
from PIL import Image,ImageOps,ImageFilter
import matplotlib.pyplot as plt
from IPython.display import display
import torch,torchvision
import importlib.metadata as package_metadata
SEED=2026
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
def seed_everything(value):
    random.seed(value); np.random.seed(value); torch.manual_seed(value)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(value)
seed_everything(SEED)
torch.backends.cudnn.benchmark=False
torch.backends.cudnn.deterministic=True
torch.use_deterministic_algorithms(True,warn_only=True)
DEVICE=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if DEVICE.type!='cuda': raise RuntimeError('Aktifkan GPU pada Session options Kaggle sebelum training.')
INPUT_ROOT=Path('/kaggle/input'); WORK_ROOT=Path('/kaggle/working/gelarrasa_run')
for d in ['eda','processed','labels','validation','checkpoints','predictions','logs','figures']:
    (WORK_ROOT/d).mkdir(parents=True,exist_ok=True)
GPU_INFO=[{'name':torch.cuda.get_device_name(i),'memory_GB':round(torch.cuda.get_device_properties(i).total_memory/1024**3,2)} for i in range(torch.cuda.device_count())]
ENVIRONMENT={'python':sys.version,'platform':platform.platform(),'torch':torch.__version__,'torchvision':torchvision.__version__,'pillow':package_metadata.version('Pillow'),'numpy':np.__version__,'pandas':pd.__version__,'cuda':torch.version.cuda,'gpu':GPU_INFO,'seed':SEED}
print('Pipeline: active_processed_port_v2')
print(json.dumps(ENVIRONMENT,indent=2))
(WORK_ROOT/'logs/environment.json').write_text(json.dumps(ENVIRONMENT,indent=2),encoding='utf-8')
'''))

cells.append(md('''## 1. Deteksi dan integritas input

Jalur Kaggle pada log sebelumnya didukung langsung. Fallback mencari pasangan train/test dan sample CSV yang dapat berada pada tingkat folder berbeda. Keunikan ID diperiksa per split; nomor train dan test boleh sama. Nilai kategori contoh pada sample CSV diabaikan.
'''))
discovery=''.join(OLD['cells'][3]['source'])
discovery=discovery.replace("return sorted([x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS],key=lambda x:(x.stem,x.as_posix()))", "direct=[x for x in folder.iterdir() if x.is_file() and x.suffix.lower() in IMAGE_EXTS]\n    files=direct or [x for x in folder.rglob('*') if x.is_file() and x.suffix.lower() in IMAGE_EXTS]\n    return sorted(files,key=lambda x:(x.stem,x.as_posix()))")
discovery=discovery.split("\nraw_train_paths=list(train_paths); raw_test_paths=list(test_paths)")[0]
discovery += "\nraw_train_paths=list(train_paths); raw_test_paths=list(test_paths)\nraw_train_ids=list(train_ids); raw_test_ids=list(test_ids)\n"
cells.append(code(discovery))
cells.append(md('''## 2. EDA sumber dan kualitas citra

Inventaris semua sumber sebelum transformasi: hash file/piksel, dimensi, rasio, ukuran byte dan indikator edge variance. Identitas gambar berasal dari pasangan split dan ID, bukan angka ID lintas split. Kesamaan piksel train-test merupakan audit deskriptif.
'''))
cells.append(code(''.join(OLD['cells'][5]['source'])))
cells.append(md('''## 3. Snapshot assignment dan kebijakan processed aktif

Blok terkompresi berikut hanya berisi metadata sumber, assignment train yang sudah dibekukan, source group, bounding box crop, status eligible, dan hash hasil processed. Tidak memuat piksel, target test, submission pembanding, atau fitur model historis. Hash payload dan hash semua sumber diverifikasi sebelum materialisasi. Snapshot berasal dari manifest operasional alur utama, bukan hasil classifier EfficientNet 37-seed yang dipakai pada versi Kaggle sebelumnya.
'''))
payload_code="import base64,zlib\nFROZEN_STATE_B64=(\n"+'\n'.join(repr(encoded[i:i+120]) for i in range(0,len(encoded),120))+"\n)\n"
payload_code+=f"STATE_SHA256='{state_hash}'\n"
payload_code+='''state_bytes=zlib.decompress(base64.b64decode(FROZEN_STATE_B64))
assert hashlib.sha256(state_bytes).hexdigest()==STATE_SHA256,'Snapshot metadata corrupted.'
PROCESS_STATE=json.loads(state_bytes)
CLASSES=PROCESS_STATE['classes']
print('Snapshot:',PROCESS_STATE['version'],'| SHA256:',STATE_SHA256)
print('Expected population:',PROCESS_STATE['expected_counts'])
print('Frozen assignment origins:',pd.Series([r['label_source'] for r in PROCESS_STATE['rows'] if r['split']=='train' and r['training_eligible']]).value_counts().to_dict())
(WORK_ROOT/'labels/processing_state.json').write_text(json.dumps(PROCESS_STATE,indent=2),encoding='utf-8')
'''
cells.append(code(payload_code))
cells.append(md('''## 4. Buat ulang processed dan periksa kesamaannya

Terapkan RGB/EXIF, crop native sesuai koordinat, lalu `ImageOps.contain` ke sisi maksimum 202 dan padding RGB `(127,127,127)` pada kanvas 224×224. Foto utuh/crop eligible disimpan pada delapan folder kelas; turunan yang ditahan terpisah di review. Semua test diproses tanpa label. Hash piksel setiap hasil dibandingkan; jika berbeda, ringkasan warna blok 16x16 harus tetap dekat. Perbedaan visual besar menghentikan proses sebelum training.
'''))
cells.append(code('''
def file_sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1<<20),b''): h.update(block)
    return h.hexdigest()
def rgb_source(path):
    with Image.open(path) as source:
        im=ImageOps.exif_transpose(source)
        if 'A' in im.getbands() or 'transparency' in im.info:
            rgba=im.convert('RGBA')
            im=Image.alpha_composite(Image.new('RGBA',rgba.size,(255,255,255,255)),rgba)
        return im.convert('RGB').copy()
def processed_canvas(im,policy):
    content=ImageOps.contain(im,(policy['max_content_side'],policy['max_content_side']),method=Image.Resampling.LANCZOS)
    size=policy['output_size']; out=Image.new('RGB',(size,size),tuple(policy['padding_rgb']))
    out.paste(content,((size-content.width)//2,(size-content.height)//2))
    return out
def thumbnail16(canvas):
    pixels=np.asarray(canvas,dtype=np.uint8)
    assert pixels.shape==(224,224,3),'Unexpected processed canvas size.'
    return np.rint(pixels.reshape(16,14,16,14,3).mean(axis=(1,3))).astype(np.uint8)
def validate_processed_pixels(canvas,spec):
    actual_sha=hashlib.sha256(canvas.tobytes()).hexdigest()
    exact=actual_sha==spec['expected_pixel_sha256']
    expected=np.frombuffer(zlib.decompress(base64.b64decode(spec['expected_thumbnail16_b64'])),dtype=np.uint8).reshape(16,16,3)
    actual=thumbnail16(canvas)
    error=np.abs(actual.astype(np.int16)-expected.astype(np.int16))
    mean_error=float(error.mean()); p99_error=float(np.percentile(error,99))
    assert exact or (mean_error<=2.0 and p99_error<=15.0), (
        f'Processed image differs materially: {spec["split"]}:{spec["sample_id"]}; '
        f'thumbnail MAE={mean_error:.3f}, p99={p99_error:.1f}')
    return actual_sha,exact,mean_error,p99_error
def materialize_processed(state,raw_train,raw_test,output_root):
    paths={'train':list(raw_train),'test':list(raw_test)}
    lookup={(split,p.name):p for split,pp in paths.items() for p in pp}
    expected={(r['split'],r['file_name']) for r in state['sources']}
    assert set(lookup)==expected,'Raw source names/count differ from frozen processed source dataset.'
    source_checks=[]
    for s in state['sources']:
        p=lookup[(s['split'],s['file_name'])]; sha=file_sha256(p)
        assert sha==s['sha256'],f'Raw file SHA256 mismatch: {s["split"]}/{s["file_name"]}'
        source_checks.append({**s,'actual_sha256':sha,'matches':True})
    for name in state['classes']: (output_root/'train'/name).mkdir(parents=True,exist_ok=True)
    out_rows=[]
    for spec in state['rows']:
        p=lookup[(spec['split'],spec['source_file'])]; im=rgb_source(p)
        x0,y0,x1,y1=spec['bbox']
        assert 0<=x0<x1<=im.width and 0<=y0<y1<=im.height,f'Invalid crop: {spec["sample_id"]}'
        region=im.crop((x0,y0,x1,y1)); canvas=processed_canvas(region,state['policy'])
        pixel_sha,pixel_exact,thumbnail_mae,thumbnail_p99=validate_processed_pixels(canvas,spec)
        if spec['split']=='test': dest=output_root/'test'/f'{spec["sample_id"]}.png'
        else:
            branch='train' if spec['training_eligible'] else 'review/excluded_train'
            dest=output_root/branch/spec['assigned_class']/f'{spec["sample_id"]}.png'
        dest.parent.mkdir(parents=True,exist_ok=True); canvas.save(dest,format='PNG')
        native=''
        if spec['is_crop']:
            native=output_root/'crops'/f'{spec["sample_id"]}.png'; native.parent.mkdir(parents=True,exist_ok=True); region.save(native,format='PNG')
        out_rows.append({**spec,'source_path':str(p),'output_path':str(dest),'native_crop_path':str(native),'actual_pixel_sha256':pixel_sha,'pixel_matches_frozen':pixel_exact,'thumbnail_mae':thumbnail_mae,'thumbnail_p99':thumbnail_p99})
    # Reserve source is retained outside all classifier folders.
    for parent in state['policy']['reserve_review']:
        matches=[p for p in raw_train if norm_id(p.stem)==parent]
        assert len(matches)==1
        dest=output_root/'review/reserved'/f'{parent}.png'; dest.parent.mkdir(parents=True,exist_ok=True); rgb_source(matches[0]).save(dest)
    return pd.DataFrame(out_rows),pd.DataFrame(source_checks)

PROCESSED_ROOT=WORK_ROOT/'processed'
processed,source_checks=materialize_processed(PROCESS_STATE,raw_train_paths,raw_test_paths,PROCESSED_ROOT)
processed.to_csv(PROCESSED_ROOT/'manifest.csv',index=False)
source_checks.to_csv(WORK_ROOT/'eda/source_integrity.csv',index=False)
train_frame=processed[processed.split.eq('train') & processed.training_eligible].reset_index(drop=True)
test_frame=processed[processed.split.eq('test')].reset_index(drop=True)
expected=PROCESS_STATE['expected_counts']
assert len(train_frame)==expected['eligible_train'] and len(test_frame)==expected['processed_test']
assert int(train_frame.is_crop.sum())==expected['eligible_crops']
assert train_frame.assigned_class.value_counts().to_dict()==PROCESS_STATE['class_counts']
assert not test_frame.assigned_class.astype(str).str.len().any()
reviewed=train_frame[train_frame.human_reviewed]
assert reviewed.assigned_class.value_counts().le(5).all() and len(reviewed)<=40
assert reviewed.parent_id.is_unique
assert train_frame.sample_id.is_unique and test_frame.sample_id.is_unique
train_paths=[Path(p) for p in train_frame.output_path]; test_paths=[Path(p) for p in test_frame.output_path]
train_ids=train_frame.sample_id.tolist(); test_ids=test_frame.parent_id.map(norm_id).tolist()
pseudo=np.array([CLASSES.index(c) for c in train_frame.assigned_class],dtype=np.int64)
eligible_idx=np.arange(len(train_frame)); groups=train_frame.parent_group.astype(str).to_numpy()
# Reuse the frozen source groups; all crops of a source inherit its group.
assert train_frame.groupby('parent_id').parent_group.nunique().le(1).all()
assert set(test_ids)==set(sample.image_id_norm) and len(test_ids)==len(sample)
train_frame.to_csv(WORK_ROOT/'labels/train_targets.csv',index=False)
proof={'state_sha256':STATE_SHA256,'source_files_verified':len(source_checks),'processed_pixels_verified':len(processed),'exact_pixel_matches':int(processed.pixel_matches_frozen.sum()),'thumbnail_tolerance_matches':int((~processed.pixel_matches_frozen).sum()),'max_thumbnail_mae':float(processed.thumbnail_mae.max()),'max_thumbnail_p99':float(processed.thumbnail_p99.max()),'eligible_train':len(train_frame),'eligible_full_frame':int((~train_frame.is_crop).sum()),'eligible_crops':int(train_frame.is_crop.sum()),'train_source_groups':int(train_frame.parent_group.nunique()),'class_counts':train_frame.assigned_class.value_counts().to_dict(),'labels_recomputed':False,'labels_origin':PROCESS_STATE['origin'],'all_pixels_match_active_processed':bool(processed.pixel_matches_frozen.all()),'all_images_passed_visual_integrity':True}
(WORK_ROOT/'processed/equivalence_report.json').write_text(json.dumps(proof,indent=2),encoding='utf-8')
print(json.dumps(proof,indent=2))
'''))

cells.append(md('''## 5. Distribusi akhir, preview crop, dan audit kesamaan

Hasil berikut menampilkan himpunan yang benar-benar dibaca model. Tujuh crop dipreview bersama gambar induknya; kelompok source tetap melekat pada crop. Histogram berasal dari manifest processed, bukan kategori sample submission.
'''))
cells.append(code('''
counts=train_frame.assigned_class.value_counts().reindex(CLASSES,fill_value=0)
counts.rename('train_samples').to_csv(WORK_ROOT/'labels/train_distribution.csv')
display(counts.to_frame('train_samples'))
fig,ax=plt.subplots(figsize=(10,4)); counts.plot.bar(ax=ax,rot=25,title='Distribusi train processed aktif'); ax.set_ylabel('sampel'); plt.tight_layout(); plt.savefig(WORK_ROOT/'figures/processed_train_distribution.png',dpi=150); plt.show(); plt.close(fig)
fig,axs=plt.subplots(2,4,figsize=(14,7))
for ax,category in zip(axs.flat,CLASSES):
    row=train_frame[train_frame.assigned_class.eq(category)].iloc[0]
    ax.imshow(rgb_source(row.output_path)); ax.set_title(f'{category} | {row.sample_id}'); ax.axis('off')
plt.tight_layout(); plt.savefig(WORK_ROOT/'figures/processed_class_preview.png',dpi=150); plt.show(); plt.close(fig)
crops=train_frame[train_frame.is_crop]
fig,axs=plt.subplots(len(crops),2,figsize=(8,3*len(crops)),squeeze=False)
for axes,row in zip(axs,crops.itertuples()):
    axes[0].imshow(rgb_source(row.source_path)); axes[0].set_title(f'Train source {row.parent_id}')
    axes[1].imshow(rgb_source(row.output_path)); axes[1].set_title(f'{row.sample_id}: {row.assigned_class}')
    for ax in axes: ax.axis('off')
plt.tight_layout(); plt.savefig(WORK_ROOT/'figures/crop_preprocessing_preview.png',dpi=130); plt.show(); plt.close(fig)
print('Low-confidence exclusion from previous Kaggle version is disabled; frozen eligible rows retained:',len(train_frame))
'''))

cells.append(md('''## 6. Kemiripan processed train-test (diagnostik)

Backbone beku EfficientNetV2-S hanya menghitung representasi deskriptif untuk pencarian kemiripan. Fitur test tidak dipakai fit normalisasi, PCA, target, sampler, atau model. Hasil nearest-train tidak mengubah satu pun assignment; CNN berikut memakai target processed yang sudah diverifikasi.
'''))
cells.append(code('''
from torch.utils.data import Dataset,DataLoader,WeightedRandomSampler
from torchvision import models,transforms
from torchvision.transforms import InterpolationMode
MEAN=(.485,.456,.406); STD=(.229,.224,.225); MODEL_IMAGE_SIZE=224
class ImageTensor(Dataset):
    def __init__(self,paths,augment=False):
        self.paths=list(paths); ops=[transforms.Resize((MODEL_IMAGE_SIZE,MODEL_IMAGE_SIZE),interpolation=InterpolationMode.BICUBIC)]
        if augment:
            ops += [transforms.RandomHorizontalFlip(.5),transforms.RandomAffine(8,translate=(.03,.03),scale=(.93,1.07),interpolation=InterpolationMode.BILINEAR,fill=(127,127,127)),transforms.RandomApply([transforms.ColorJitter(.12,.12,.12,.025)],p=.4)]
        self.tf=transforms.Compose(ops+[transforms.ToTensor(),transforms.Normalize(MEAN,STD)])
    def __len__(self): return len(self.paths)
    def __getitem__(self,i):
        with Image.open(self.paths[i]) as im: return self.tf(im.convert('RGB'))
try: embedding_net=models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1).to(DEVICE).eval()
except Exception as error: raise RuntimeError('Bobot pretrained gagal diunduh. Aktifkan Internet Kaggle; bobot acak tidak digunakan.') from error
def extract_embeddings(paths):
    loader=DataLoader(ImageTensor(paths),batch_size=24,shuffle=False,num_workers=2,pin_memory=True)
    chunks=[]
    with torch.inference_mode(),torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda',dtype=torch.float16):
        for xb in loader:
            xb=xb.to(DEVICE,non_blocking=True); z=embedding_net.avgpool(embedding_net.features(xb)).flatten(1)
            chunks.append(z.float().cpu().numpy())
    result=np.concatenate(chunks).astype('float32'); result/=np.maximum(np.linalg.norm(result,axis=1,keepdims=True),1e-12)
    return result
train_emb=extract_embeddings(train_paths); test_emb=extract_embeddings(test_paths)
np.savez_compressed(WORK_ROOT/'eda/processed_embeddings.npz',train=train_emb,test=test_emb,train_ids=np.array(train_ids),test_ids=np.array(test_ids))
similarity=test_emb@train_emb.T
nearest=np.argsort(-similarity,axis=1)[:,:3]
from scipy.fft import dctn
from skimage.metrics import structural_similarity
def pixel_descriptor(path):
    with Image.open(path) as im:
        gray=im.convert('L')
        small=np.asarray(gray.resize((32,32),Image.Resampling.LANCZOS),dtype=np.float32)
        low=dctn(small,type=2,norm='ortho')[:8,:8]; bits=(low>np.median(low)).reshape(-1)
        pix=np.asarray(gray.resize((64,64),Image.Resampling.LANCZOS),dtype=np.float32)/255.
    return bits,pix
train_desc=[pixel_descriptor(p) for p in train_paths]; test_desc=[pixel_descriptor(p) for p in test_paths]
train_phash=np.stack([p[0] for p in train_desc]); test_phash=np.stack([p[0] for p in test_desc])
train_pixels=np.stack([p[1] for p in train_desc]); test_pixels=np.stack([p[1] for p in test_desc])
neighbor_rows=[]
for q,nn in enumerate(nearest):
    for rank,j in enumerate(nn,1):
        neighbor_rows.append({'test_id':test_ids[q],'rank':rank,'train_sample_id':train_ids[j],'train_class':train_frame.iloc[j].assigned_class,'cosine':float(similarity[q,j]),'phash_hamming':int(np.count_nonzero(test_phash[q]!=train_phash[j])),'grayscale_mae_64':float(np.abs(test_pixels[q]-train_pixels[j]).mean()),'ssim_gray_64':float(structural_similarity(test_pixels[q],train_pixels[j],data_range=1.)),'train_group':groups[j],'train_native_width':int(train_frame.iloc[j].bbox[2]-train_frame.iloc[j].bbox[0]),'train_native_height':int(train_frame.iloc[j].bbox[3]-train_frame.iloc[j].bbox[1]),'test_native_width':int(test_frame.iloc[q].bbox[2]-test_frame.iloc[q].bbox[0]),'test_native_height':int(test_frame.iloc[q].bbox[3]-test_frame.iloc[q].bbox[1])})
pd.DataFrame(neighbor_rows).to_csv(WORK_ROOT/'eda/processed_train_test_neighbors.csv',index=False)
phash_nearest=[]
for q in range(len(test_paths)):
    distance=np.count_nonzero(train_phash!=test_phash[q],axis=1); j=int(distance.argmin())
    phash_nearest.append({'test_id':test_ids[q],'train_sample_id':train_ids[j],'phash_hamming':int(distance[j]),'embedding_cosine':float(similarity[q,j]),'candidate_near_duplicate':bool(distance[j]<=4)})
pd.DataFrame(phash_nearest).to_csv(WORK_ROOT/'eda/processed_phash_nearest.csv',index=False)
query_positions=np.argsort(-similarity.max(axis=1))[:8]
fig,axs=plt.subplots(len(query_positions),2,figsize=(8,3*len(query_positions)),squeeze=False)
for axes,q in zip(axs,query_positions):
    j=int(nearest[q,0]); axes[0].imshow(rgb_source(test_paths[q])); axes[0].set_title(f'Test {test_ids[q]} (no target)')
    axes[1].imshow(rgb_source(train_paths[j])); axes[1].set_title(f'Train {train_ids[j]}: {train_frame.iloc[j].assigned_class} | cos {similarity[q,j]:.3f}')
    for ax in axes: ax.axis('off')
plt.tight_layout(); plt.savefig(WORK_ROOT/'figures/processed_train_test_neighbors.png',dpi=130); plt.show(); plt.close(fig)
del embedding_net; gc.collect(); torch.cuda.empty_cache()
'''))

cells.append(md('''## 7. Source-group cross-validation dan GPU training

Semua training memakai assignment processed yang sudah dibekukan. Tiga fold dikelompokkan memakai source group yang sama seperti alur utama; crop dan induknya tidak dapat terpisah lintas fold. Setiap model menjalani 8 epoch/fold, dengan checkpoint terbaik berdasarkan macro-F1 proxy. Refit final memakai 3 epoch head + 18 epoch fine-tuning dan menyimpan bobot pada akhir jadwal tersebut. Final refit tidak memiliki validation set sehingga log final tidak mengklaim metrik validation.

Empat backbone memakai AMP dan kedua GPU jika tersedia, class-balanced sampler, MixUp pada piksel input, label smoothing, dan cosine learning rate. Tiap epoch menyimpan state optimizer/scheduler/scaler/RNG untuk resume pada working directory yang sama. Resume hanya berlaku bila fingerprint processed dan konfigurasi cocok.
'''))
cells.append(code('''
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score,f1_score,classification_report,confusion_matrix,log_loss
from torch.nn import functional as F
MODEL_NAMES=['resnet50','efficientnet_v2_s','convnext_small','swin_v2_t']
HEAD_EPOCHS=3; FINETUNE_EPOCHS=18; FOLD_EPOCHS=8; N_SPLITS=3; BATCH_SIZE=24; WORKERS=2
USE_MULTI_GPU=True; RUN_START=time.time()
TRAIN_CONFIG={'models':MODEL_NAMES,'head_epochs':HEAD_EPOCHS,'finetune_epochs':FINETUNE_EPOCHS,'fold_epochs':FOLD_EPOCHS,'folds':N_SPLITS,'batch_size':BATCH_SIZE,'model_size':MODEL_IMAGE_SIZE,'seed':SEED,'state_sha256':STATE_SHA256,'torch':str(torch.__version__),'torchvision':str(torchvision.__version__)}
CONFIG_HASH=hashlib.sha256(json.dumps(TRAIN_CONFIG,sort_keys=True).encode()).hexdigest()
(WORK_ROOT/'logs/train_config.json').write_text(json.dumps(TRAIN_CONFIG,indent=2),encoding='utf-8')
def core_model(model): return model.module if isinstance(model,torch.nn.DataParallel) else model
def get_head(model,name):
    m=core_model(model)
    if name=='resnet50': return m.fc
    if name=='efficientnet_v2_s': return m.classifier[1]
    if name=='convnext_small': return m.classifier[2]
    if name=='swin_v2_t': return m.head
    raise ValueError(f'Unknown model: {name}')
def build_model(name,pretrained=True):
    if name=='resnet50':
        m=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None); m.fc=torch.nn.Linear(m.fc.in_features,len(CLASSES))
    elif name=='efficientnet_v2_s':
        m=models.efficientnet_v2_s(weights=models.EfficientNet_V2_S_Weights.IMAGENET1K_V1 if pretrained else None); m.classifier[1]=torch.nn.Linear(m.classifier[1].in_features,len(CLASSES))
    elif name=='convnext_small':
        m=models.convnext_small(weights=models.ConvNeXt_Small_Weights.IMAGENET1K_V1 if pretrained else None); m.classifier[2]=torch.nn.Linear(m.classifier[2].in_features,len(CLASSES))
    elif name=='swin_v2_t':
        m=models.swin_v2_t(weights=models.Swin_V2_T_Weights.IMAGENET1K_V1 if pretrained else None); m.head=torch.nn.Linear(m.head.in_features,len(CLASSES))
    else: raise ValueError(name)
    return m
class LabeledDataset(Dataset):
    def __init__(self,indices,augment=False):
        self.indices=np.asarray(indices); self.base=ImageTensor([train_paths[j] for j in self.indices],augment)
    def __len__(self): return len(self.indices)
    def __getitem__(self,k):
        j=int(self.indices[k]); return self.base[k],int(pseudo[j]),train_ids[j]
def worker_seed(worker_id):
    value=torch.initial_seed()%2**32; np.random.seed(value); random.seed(value)
def make_train_loader(indices,seed):
    counts=np.bincount(pseudo[indices],minlength=len(CLASSES)); weights=1/np.maximum(counts,1)
    generator=torch.Generator().manual_seed(seed)
    sampler=WeightedRandomSampler(torch.tensor(weights[pseudo[indices]],dtype=torch.double),num_samples=len(indices),replacement=True,generator=generator)
    return DataLoader(LabeledDataset(indices,True),batch_size=BATCH_SIZE,sampler=sampler,num_workers=WORKERS,pin_memory=DEVICE.type=='cuda',worker_init_fn=worker_seed,generator=generator)
def predict_loader(model,loader):
    model.eval(); chunks=[]; labels=[]; ids=[]
    with torch.inference_mode(),torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda',dtype=torch.float16):
        for batch in loader:
            xb=batch[0] if isinstance(batch,(list,tuple)) else batch
            chunks.append(model(xb.to(DEVICE,non_blocking=True)).float().softmax(1).cpu().numpy())
            if isinstance(batch,(list,tuple)):
                labels.extend(batch[1].numpy().tolist()); ids.extend(list(batch[2]))
    return np.concatenate(chunks),labels,ids
def train_epoch(model,loader,opt,scaler,head_only,name):
    model.train()
    if head_only: model.eval(); get_head(model,name).train()
    loss_sum=0.; n=0
    for xb,yb,_ in loader:
        xb=xb.to(DEVICE,non_blocking=True); yb=yb.to(DEVICE); opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=DEVICE.type,enabled=DEVICE.type=='cuda',dtype=torch.float16):
            if np.random.random()<.3 and len(xb)>1:
                lam=float(np.random.beta(.15,.15)); perm=torch.randperm(len(xb),device=DEVICE)
                logits=model(lam*xb+(1-lam)*xb[perm])
                loss=lam*F.cross_entropy(logits,yb,label_smoothing=.025)+(1-lam)*F.cross_entropy(logits,yb[perm],label_smoothing=.025)
            else: loss=F.cross_entropy(model(xb),yb,label_smoothing=.025)
        scaler.scale(loss).backward(); scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),2.0); scaler.step(opt); scaler.update()
        loss_sum+=float(loss.detach().cpu())*len(yb); n+=len(yb)
    return loss_sum/max(n,1)
def load_checkpoint(path): return torch.load(path,map_location='cpu',weights_only=False)
def atomic_save(obj,path):
    tmp=path.with_suffix('.tmp'); torch.save(obj,tmp); os.replace(tmp,path)
def fit_one(name,tr_indices,va_indices,epochs,tag):
    seed=SEED+MODEL_NAMES.index(name)*100+sum(tag.encode()); seed_everything(seed)
    best_path=WORK_ROOT/f'checkpoints/{tag}_{name}_best.pt'
    final_path=WORK_ROOT/f'checkpoints/{tag}_{name}_final.pt'
    resume_path=WORK_ROOT/f'checkpoints/{tag}_{name}_resume.pt'
    done_path=WORK_ROOT/f'checkpoints/{tag}_{name}_done.json'
    chosen=best_path if va_indices is not None else final_path
    if done_path.exists() and chosen.exists():
        info=json.loads(done_path.read_text()); assert info['config_hash']==CONFIG_HASH,'Existing completed model has a different configuration.'
        ckpt=load_checkpoint(chosen); model=build_model(name,False); model.load_state_dict(ckpt['state']); model=model.to(DEVICE)
        if USE_MULTI_GPU and DEVICE.type=='cuda' and torch.cuda.device_count()>1: model=torch.nn.DataParallel(model)
        print('Reusing completed model:',tag,name); return model,pd.read_csv(WORK_ROOT/f'logs/{tag}_{name}_history.csv').to_dict('records'),chosen
    resume=load_checkpoint(resume_path) if resume_path.exists() else None
    if resume: assert resume['config_hash']==CONFIG_HASH,'Resume data/config mismatch.'
    model=build_model(name,pretrained=resume is None)
    if resume: model.load_state_dict(resume['state'])
    model=model.to(DEVICE)
    if USE_MULTI_GPU and DEVICE.type=='cuda' and torch.cuda.device_count()>1: model=torch.nn.DataParallel(model)
    scaler=torch.amp.GradScaler(DEVICE.type,enabled=DEVICE.type=='cuda')
    history=resume['history'] if resume else []; completed=int(resume['completed']) if resume else 0; best=float(resume['best']) if resume else -1.
    if resume:
        random.setstate(resume['rng_python']); np.random.set_state(resume['rng_numpy']); torch.set_rng_state(resume['rng_torch'])
        if DEVICE.type=='cuda': torch.cuda.set_rng_state_all(resume['rng_cuda'])
        scaler.load_state_dict(resume['scaler'])
    val_loader=DataLoader(LabeledDataset(va_indices),batch_size=BATCH_SIZE*2,shuffle=False,num_workers=WORKERS,pin_memory=DEVICE.type=='cuda') if va_indices is not None else None
    stages=[('head',min(HEAD_EPOCHS,epochs)),('finetune',max(0,epochs-HEAD_EPOCHS))]
    start=0
    for stage,count in stages:
        if count==0: continue
        stop=start+count
        if completed>=stop: start=stop; continue
        for param in model.parameters(): param.requires_grad=stage!='head'
        for param in get_head(model,name).parameters(): param.requires_grad=True
        head_params=list(get_head(model,name).parameters()); head_ids={id(p) for p in head_params}
        if stage=='head': param_groups=[{'params':head_params,'lr':1e-3}]
        else: param_groups=[{'params':[p for p in model.parameters() if id(p) not in head_ids],'lr':2e-5},{'params':head_params,'lr':1e-4}]
        opt=torch.optim.AdamW(param_groups,weight_decay=1e-4); scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=count,eta_min=1e-6)
        if resume and resume['stage']==stage and completed>start:
            opt.load_state_dict(resume['optimizer']); scheduler.load_state_dict(resume['scheduler'])
            # Optimizer tensors are moved explicitly for CPU-map-location resume.
            for values in opt.state.values():
                for key,value in values.items():
                    if torch.is_tensor(value): values[key]=value.to(DEVICE)
        for epoch in range(max(completed,start),stop):
            epoch_start=time.time(); loader=make_train_loader(tr_indices,seed+epoch)
            loss=train_epoch(model,loader,opt,scaler,stage=='head',name)
            row={'model':name,'tag':tag,'stage':stage,'epoch':epoch+1,'train_loss':loss,'lr_backbone_or_head':float(opt.param_groups[0]['lr']),'seconds':round(time.time()-epoch_start,2)}
            if val_loader is not None:
                pr,yy,_=predict_loader(model,val_loader); prediction=pr.argmax(1)
                row.update(val_proxy_agreement=float(accuracy_score(yy,prediction)),val_proxy_macro_f1=float(f1_score(yy,prediction,labels=list(range(len(CLASSES))),average='macro',zero_division=0)),val_proxy_logloss=float(log_loss(yy,pr,labels=list(range(len(CLASSES))))))
                if row['val_proxy_macro_f1']>best:
                    best=row['val_proxy_macro_f1']; atomic_save({'state':core_model(model).state_dict(),'name':name,'epoch':epoch+1,'config_hash':CONFIG_HASH,'classes':CLASSES},best_path)
            scheduler.step(); history.append(row); completed=epoch+1
            pd.DataFrame(history).to_csv(WORK_ROOT/f'logs/{tag}_{name}_history.csv',index=False)
            atomic_save({'state':core_model(model).state_dict(),'name':name,'stage':stage,'completed':completed,'best':best,'history':history,'optimizer':opt.state_dict(),'scheduler':scheduler.state_dict(),'scaler':scaler.state_dict(),'rng_python':random.getstate(),'rng_numpy':np.random.get_state(),'rng_torch':torch.get_rng_state(),'rng_cuda':torch.cuda.get_rng_state_all() if DEVICE.type=='cuda' else [],'config_hash':CONFIG_HASH},resume_path)
            print(row,flush=True)
        start=stop
    if va_indices is None: atomic_save({'state':core_model(model).state_dict(),'name':name,'epoch':completed,'config_hash':CONFIG_HASH,'classes':CLASSES},final_path)
    model.load_state_dict({('module.'+k if isinstance(model,torch.nn.DataParallel) else k):v for k,v in load_checkpoint(chosen)['state'].items()})
    done_path.write_text(json.dumps({'config_hash':CONFIG_HASH,'epochs':completed,'checkpoint':str(chosen)}),encoding='utf-8')
    if resume_path.exists(): resume_path.unlink()
    return model,history,chosen

splitter=StratifiedGroupKFold(n_splits=N_SPLITS,shuffle=True,random_state=SEED)
folds=list(splitter.split(np.zeros(len(train_frame)),pseudo,groups)); fold_ids=np.full(len(train_frame),-1)
for fold,(tr,va) in enumerate(folds):
    assert set(groups[tr]).isdisjoint(set(groups[va]))
    assert set(pseudo[tr])==set(range(len(CLASSES)))
    fold_ids[va]=fold
assert (fold_ids>=0).all()
fold_table=train_frame[['sample_id','parent_id','parent_group','assigned_class','is_crop']].copy(); fold_table['fold']=fold_ids; fold_table.to_csv(WORK_ROOT/'validation/folds.csv',index=False)
oof_probs={}; proxy_summary=[]
for name in MODEL_NAMES:
    oof=np.full((len(train_frame),len(CLASSES)),np.nan,dtype=np.float32)
    for fold,(tr,va) in enumerate(folds):
        model,history,checkpoint=fit_one(name,tr,va,FOLD_EPOCHS,f'fold{fold}')
        loader=DataLoader(LabeledDataset(va),batch_size=BATCH_SIZE*2,shuffle=False,num_workers=WORKERS,pin_memory=DEVICE.type=='cuda')
        pp,yy,ids=predict_loader(model,loader); assert ids==[train_ids[j] for j in va]; oof[va]=pp
        del model; gc.collect(); torch.cuda.empty_cache()
    assert np.isfinite(oof).all() and np.allclose(oof.sum(1),1,atol=1e-4)
    oof_probs[name]=oof; pred=oof.argmax(1)
    row={'model':name,'proxy_agreement':float(accuracy_score(pseudo,pred)),'proxy_macro_f1':float(f1_score(pseudo,pred,labels=list(range(len(CLASSES))),average='macro',zero_division=0))}; proxy_summary.append(row)
    pd.DataFrame(oof,columns=CLASSES).assign(sample_id=train_ids,fold=fold_ids,target=train_frame.assigned_class,predicted=[CLASSES[j] for j in pred]).to_csv(WORK_ROOT/f'validation/oof_{name}.csv',index=False)
    pd.DataFrame(confusion_matrix(pseudo,pred,labels=list(range(len(CLASSES)))),index=CLASSES,columns=CLASSES).to_csv(WORK_ROOT/f'validation/confusion_{name}.csv')
    pd.DataFrame(classification_report(pseudo,pred,labels=list(range(len(CLASSES))),target_names=CLASSES,output_dict=True,zero_division=0)).T.to_csv(WORK_ROOT/f'validation/report_{name}.csv')
    print(row)
oof_ensemble=np.mean(list(oof_probs.values()),axis=0)
proxy_summary.append({'model':'equal_probability_ensemble','proxy_agreement':float(accuracy_score(pseudo,oof_ensemble.argmax(1))),'proxy_macro_f1':float(f1_score(pseudo,oof_ensemble.argmax(1),labels=list(range(len(CLASSES))),average='macro',zero_division=0))})
pd.DataFrame(proxy_summary).to_csv(WORK_ROOT/'validation/proxy_summary.csv',index=False); display(pd.DataFrame(proxy_summary))
'''))

cells.append(md('''## 8. Refit seluruh processed train dan inferensi

Masing-masing backbone di-refit pada seluruh 1.886 sampel processed menggunakan jadwal 21 epoch. Model disimpan dan melakukan inferensi test, kemudian dibebaskan dari GPU sebelum model berikutnya. Ensemble memakai rata-rata probabilitas empat model secara global, tanpa aturan per ID test. Semua CSV individual dan CSV ensemble disusun mengikuti ID sample submission.
'''))
cells.append(code('''
FINAL_EPOCHS=HEAD_EPOCHS+FINETUNE_EPOCHS
model_probabilities={}; final_meta=[]
test_loader=DataLoader(ImageTensor(test_paths),batch_size=BATCH_SIZE*2,shuffle=False,num_workers=WORKERS,pin_memory=DEVICE.type=='cuda')
def write_submission(probabilities,path):
    assert probabilities.shape==(len(test_ids),len(CLASSES)) and np.isfinite(probabilities).all()
    assert np.allclose(probabilities.sum(1),1,atol=1e-4)
    pred=pd.DataFrame({'image_id_norm':test_ids,'waste_category':[CLASSES[j] for j in probabilities.argmax(1)]})
    out=sample[['image_id','image_id_norm']].merge(pred,on='image_id_norm',how='left',validate='one_to_one',sort=False)
    assert out.image_id.tolist()==sample.image_id.tolist()
    assert len(out)==480 and out.waste_category.notna().all() and set(out.waste_category)<=set(CLASSES)
    out=out[['image_id','waste_category']]; out.to_csv(path,index=False)
    return out
for name in MODEL_NAMES:
    model,history,checkpoint=fit_one(name,eligible_idx,None,FINAL_EPOCHS,'final')
    pp,_,_=predict_loader(model,test_loader)
    model_probabilities[name]=pp
    pd.DataFrame(pp,columns=CLASSES).assign(image_id=test_ids).to_csv(WORK_ROOT/f'predictions/test_probabilities_{name}.csv',index=False)
    write_submission(pp,WORK_ROOT/f'predictions/submission_{name}.csv')
    final_meta.append({'model':name,'checkpoint':str(checkpoint.relative_to(WORK_ROOT)),'checkpoint_sha256':file_sha256(checkpoint),'completed_epochs':len(history),'selection':'last epoch of fixed full-train refit; no validation set'})
    del model; gc.collect(); torch.cuda.empty_cache()
ensemble=np.mean(list(model_probabilities.values()),axis=0)
submission=write_submission(ensemble,WORK_ROOT/'DSC2026_Submission.csv')
pd.DataFrame(ensemble,columns=[f'prob_{c}' for c in CLASSES]).assign(image_id=test_ids).to_csv(WORK_ROOT/'predictions/test_probabilities_ensemble.csv',index=False)
display(submission.head(10)); print('Final CSV:',WORK_ROOT/'DSC2026_Submission.csv')
'''))

cells.append(md('''## 9. Ekspor log, kurva, hash, dan paket hasil

Manifest akhir menghubungkan snapshot processed, hash input/output, konfigurasi fold, versi runtime, checkpoint dan CSV. Keluaran grafik dari history menunjukkan loss training dan validation proxy hanya pada fold yang benar-benar memiliki validation set. Arsip unduhan menyertakan processed, laporan dan checkpoint final; checkpoint fold tetap tersedia di folder Output, tanpa digandakan di ZIP.
'''))
cells.append(code('''
for name in MODEL_NAMES:
    fig,axs=plt.subplots(1,2,figsize=(12,4))
    for fold in range(N_SPLITS):
        h=pd.read_csv(WORK_ROOT/f'logs/fold{fold}_{name}_history.csv')
        axs[0].plot(h.epoch,h.train_loss,label=f'fold {fold}')
        axs[1].plot(h.epoch,h.val_proxy_macro_f1,label=f'fold {fold}')
    axs[0].set_title(name+' train loss'); axs[1].set_title(name+' validation proxy macro-F1')
    for ax in axs: ax.set_xlabel('epoch'); ax.legend()
    plt.tight_layout(); plt.savefig(WORK_ROOT/f'figures/history_{name}.png',dpi=150); plt.show(); plt.close(fig)
run_manifest={'run_utc':datetime.now(timezone.utc).isoformat(),'environment':ENVIRONMENT,'processing_equivalence':proof,'training_config':TRAIN_CONFIG,'config_sha256':CONFIG_HASH,'proxy_validation':proxy_summary,'final_models':final_meta,'ensemble_policy':'global equal mean probabilities, fixed before test inference','submission_rows':len(submission),'submission_sha256':file_sha256(WORK_ROOT/'DSC2026_Submission.csv'),'submission_label_distribution':submission.waste_category.value_counts().to_dict(),'raw_test_used_for_fit':False,'assignment_fitting_repeated':False,'elapsed_training_seconds':round(time.time()-RUN_START,2)}
pretrained={'resnet50':models.ResNet50_Weights.IMAGENET1K_V2,'efficientnet_v2_s':models.EfficientNet_V2_S_Weights.IMAGENET1K_V1,'convnext_small':models.ConvNeXt_Small_Weights.IMAGENET1K_V1,'swin_v2_t':models.Swin_V2_T_Weights.IMAGENET1K_V1}
run_manifest['pretrained_weight_sources']={}
for name,weight in pretrained.items():
    cached=Path(torch.hub.get_dir())/'checkpoints'/weight.url.rsplit('/',1)[-1]
    run_manifest['pretrained_weight_sources'][name]={'enum':str(weight),'url':weight.url,'sha256':file_sha256(cached) if cached.exists() else None}
(WORK_ROOT/'logs/run_manifest.json').write_text(json.dumps(run_manifest,indent=2),encoding='utf-8')
csv=pd.read_csv(WORK_ROOT/'DSC2026_Submission.csv',dtype={'image_id':str})
assert list(csv.columns)==['image_id','waste_category'] and csv.image_id.tolist()==sample.image_id.tolist() and len(csv)==480
zip_path=Path('/kaggle/working/GelarRasa_GPU_run_artifacts.zip')
with zipfile.ZipFile(zip_path,'w',zipfile.ZIP_DEFLATED) as archive:
    for p in WORK_ROOT.rglob('*'):
        if not p.is_file() or p.name.endswith('_resume.pt'): continue
        if 'checkpoints' in p.parts and not p.name.startswith('final_'): continue
        archive.write(p,p.relative_to(WORK_ROOT.parent))
print('CSV:',WORK_ROOT/'DSC2026_Submission.csv')
print('Proof:',WORK_ROOT/'processed/equivalence_report.json')
print('ZIP:',zip_path)
'''))

for i,c in enumerate(cells): c['id']=f'gelarrasa-port-{i:02d}'
nb={'cells':cells,'metadata':{'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13'},'gelarrasa_pipeline':{'version':state['version'],'state_sha256':state_hash}},'nbformat':4,'nbformat_minor':5}
TARGET.write_text(json.dumps(nb,ensure_ascii=False,indent=1),encoding='utf-8')
(TARGET.parent/'README.md').write_text('''# GelarRasa GPU — processed aktif\n\nBuka `DSC2026_GelarRasa_GPU_OneRun.ipynb` di notebook Kaggle milik akun tim. Pasang dataset `arielzzzzzz/gelarrasa`, pilih GPU (T4 x2 didukung), aktifkan Internet untuk bobot torchvision, lalu Run All. Metadata notebook tidak menimpa pengaturan input/accelerator Kaggle.\n\nSeluruh piksel berasal dari dataset panitia yang terpasang. Notebook menyertakan snapshot metadata processed aktif: assignment train algoritmis, source group, 37 seed tim, koordinat crop, eligibility, dan hash. Snapshot itu adalah keluaran alur utama yang sudah dibekukan; fitting seluruh metode clustering historis tidak diulang. Tidak membaca file lokal, arsip anotasi, target test, atau submission pembanding. Kategori sample submission diabaikan.\n\nSebelum training, notebook membuat ulang gambar processed dan memverifikasi hash sumber. Hash piksel diperiksa persis; selisih kecil lintas lingkungan diperiksa lagi memakai ringkasan warna blok 16x16. Perbedaan visual besar menghentikan proses. Populasi aktif adalah 1.879 foto utuh + 7 crop = 1.886 sampel eligible, dengan 27 turunan train ditahan dan 480 test. Angka 1.869 adalah populasi historis, bukan manifest aktif yang digunakan pada port ini. Semua label/folder dan crop mengikuti manifest processed aktif; tidak ada filter confidence 37-seed baru.\n\nModel: ResNet-50, EfficientNetV2-S, ConvNeXt-Small, Swin-V2-Tiny. Tiga fold source-group, 8 epoch/fold; refit final 3 head + 18 fine-tune epoch per model. AMP dan kedua GPU dipakai jika tersedia. Model dilatih/inferensi berurutan agar memori GPU dibebaskan. Ensemble memakai rata-rata probabilitas global. Validasi merupakan agreement terhadap target algoritmis, bukan ground truth semantik independen.\n\nHasil: `/kaggle/working/gelarrasa_run/DSC2026_Submission.csv`, processed dalam delapan folder kelas, manifest/equivalence report, EDA/preview, source-group folds, OOF/report, log/checkpoint, probabilitas per model, serta `/kaggle/working/GelarRasa_GPU_run_artifacts.zip`. Resume membaca optimizer/scheduler/scaler/RNG bila working directory yang sama masih tersedia; sesi Kaggle baru tanpa checkpoint tidak melanjutkan run lama.\n\nDataset private harus dapat diakses penilai agar mereka bisa menjalankan notebook. Port ini tidak mengubah visibility. Verifikasi lokal mencakup rekonstruksi processed dan smoke check CPU; Run All T4 x2 tetap harus dilakukan di Kaggle.\n''',encoding='utf-8')
print('Notebook written:',TARGET,'| state SHA256',state_hash,'| eligible',len(eligible),'| payload bytes',len(encoded))
enhancement=Path(__file__).with_name('enhance_kaggle_models.py')
if enhancement.exists():
    import runpy
    runpy.run_path(str(enhancement),run_name='__main__')

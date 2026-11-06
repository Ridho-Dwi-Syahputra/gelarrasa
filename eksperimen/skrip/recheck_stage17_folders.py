"""Independent filesystem and pixel audit, never a source of training labels.

Compares the actual folders, not just existing agreement counters. Generates a
private comparison gallery without correcting categories or classifying images.
"""
from pathlib import Path
from collections import defaultdict
from datetime import datetime
from zoneinfo import ZoneInfo
import hashlib,html,json,os,time
import numpy as np
import pandas as pd
from PIL import Image,ImageOps,ImageDraw,ImageFont

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'data/processed/stage17_named_predictions_v1/73996d8db5'
REFERENCE=ROOT/'data/Dataset DSC GelarRasa 2026/train/label'
REPORT=ROOT/'reports/errors/stage17_hierarchical_routing_20261106'
CLASSES=['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']
EXTENSIONS={'.jpg','.jpeg','.png','.webp','.bmp','.tif','.tiff'}
def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def scan(folder):
    assert sorted(p.name for p in folder.iterdir() if p.is_dir())==sorted(CLASSES)
    found=defaultdict(list)
    for name in CLASSES:
        for path in sorted((folder/name).rglob('*')):
            if path.is_file() and path.suffix.lower() in EXTENSIONS:found[path.stem].append((name,path))
    duplicate={key:[str(p) for _,p in values] for key,values in found.items() if len(values)!=1}
    assert not duplicate,duplicate
    return {key:values[0] for key,values in found.items()}

def normalized_reference(path):
    # Independently implement documented stage1 EXIF + white-alpha policy.
    with Image.open(path) as source:
        image=ImageOps.exif_transpose(source)
        alpha='A' in image.getbands() or 'transparency' in image.info
        if alpha:
            rgba=image.convert('RGBA');background=Image.new('RGBA',rgba.size,(255,255,255,255))
            image=Image.alpha_composite(background,rgba).convert('RGB')
        else:image=image.convert('RGB')
        return np.asarray(image).copy(),alpha

def gallery_and_samples(table,private):
    gallery=private/'folder_comparison_gallery.html'
    parts=['<!doctype html><meta charset="utf-8"><title>Audit pembagian folder train</title><style>body{font:15px Arial;margin:24px;background:#fafafa}section{clear:both}figure{display:inline-block;vertical-align:top;width:170px;margin:6px;padding:8px;border:2px solid #187c42;background:white}.diff{border-color:#c33}img{width:160px;height:150px;object-fit:contain}figcaption{font-size:12px;line-height:1.5}</style>',
        f'<h1>Folder aktual: {int(table.agrees.sum())} sama, {int((~table.agrees).sum())} berbeda</h1><p>Sama berarti ID dan nama kategori cocok. Bukan verifikasi semantik semua gambar. Garis merah berarti label hasil berbeda dari label arsip.</p>']
    for name in CLASSES:
        subset=table[table.predicted_class.eq(name)].sort_values(['agrees','image_id'])
        parts.append(f'<section><h2>{name}: {len(subset)} gambar hasil</h2>')
        for row in subset.itertuples():
            target=os.path.relpath(ROOT/row.predicted_path,gallery.parent).replace('\\','/')
            badge='sama' if row.agrees else 'berbeda';border='' if row.agrees else 'diff'
            parts.append(f'<figure class="{border}"><a href="{html.escape(target)}"><img loading="lazy" src="{html.escape(target)}"></a><figcaption>{row.image_id} | {badge}<br>hasil: {row.predicted_class}<br>arsip: {row.archive_folder_class}</figcaption></figure>')
        parts.append('</section>')
    gallery.write_text('\n'.join(parts),encoding='utf-8')
    preview=ROOT/'output/runtime/folder_recheck_20261106';preview.mkdir(parents=True,exist_ok=True)
    font=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',19);small=ImageFont.truetype('C:/Windows/Fonts/arial.ttf',16)
    for sheet,names in enumerate([CLASSES[:4],CLASSES[4:]],1):
        canvas=Image.new('RGB',(1120,1220),'#f3f4f6');draw=ImageDraw.Draw(canvas)
        draw.text((15,10),'SAMPLE FOLDER AUDIT - existing labels only',font=font,fill='black')
        for row_index,name in enumerate(names):
            subset=table[table.predicted_class.eq(name)].sort_values('image_id')
            positions=sorted(set([0,len(subset)//3,2*len(subset)//3,len(subset)-1]))
            for column,position in enumerate(positions):
                row=subset.iloc[position];x=column*280;y=45+row_index*290
                with Image.open(ROOT/row.predicted_path) as im:thumb=ImageOps.contain(im.convert('RGB'),(250,215))
                canvas.paste(thumb,(x+(270-thumb.width)//2,y))
                draw.text((x+10,y+220),f'{row.image_id} result: {row.predicted_class}',font=small,fill='black')
                draw.text((x+10,y+244),f'archive: {row.archive_folder_class}',font=small,fill='#187c42' if row.agrees else '#c22')
        canvas.save(preview/f'sample_{sheet}.jpg',quality=88)
    return str(gallery.relative_to(ROOT))

def main():
    started=time.time();pred=scan(BASE/'train');reference=scan(REFERENCE)
    expected={f'{i:04d}' for i in range(1,1921)};assert set(pred)==set(reference)==expected
    m=pd.read_csv(BASE/'prediction_manifest.csv',dtype={'image_id':str});m=m[m.split.eq('train')].set_index('image_id');assert len(m)==1920 and m.index.is_unique
    profile=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str});profile=profile[profile.split.eq('train')].set_index('image_id')
    archive=pd.read_csv(ROOT/'docs/train_label_review/train_labels.csv',dtype={'image_id':str}).set_index('image_id')
    records=[];failures=[];historical_differences=[];alpha_ids=[]
    for number,image_id in enumerate(sorted(expected),1):
        pc,pp=pred[image_id];rc,rp=reference[image_id];row=m.loc[image_id];source=profile.loc[image_id]
        folder_ok=pc==row.predicted_class and pp.resolve()==(ROOT/row.predicted_path).resolve()
        digest=sha(pp);hash_ok=digest==row.sha256==source.processed_sha256
        archive_bytes_ok=sha(rp)==source.source_sha256
        with Image.open(pp) as im:pred_pixels=np.asarray(im.convert('RGB')).copy()
        ref_pixels,alpha=normalized_reference(rp)
        pixel_ok=pred_pixels.shape==ref_pixels.shape and np.array_equal(pred_pixels,ref_pixels)
        if alpha:alpha_ids.append(image_id)
        if not all([folder_ok,hash_ok,archive_bytes_ok,pixel_ok]):failures.append(dict(image_id=image_id,folder_ok=bool(folder_ok),hash_ok=bool(hash_ok),archive_bytes_ok=bool(archive_bytes_ok),pixel_ok=bool(pixel_ok)))
        if rc!=archive.loc[image_id].waste_category:historical_differences.append(dict(image_id=image_id,folder_label=rc,csv_label=archive.loc[image_id].waste_category))
        records.append(dict(image_id=image_id,predicted_class=pc,archive_folder_class=rc,agrees=pc==rc,predicted_path=str(pp.relative_to(ROOT)),reference_path=str(rp.relative_to(ROOT)),processed_sha256=digest,folder_manifest_ok=folder_ok,processed_hash_ok=hash_ok,archive_bytes_match_source=archive_bytes_ok,pixels_match_after_canonical_preprocessing=pixel_ok,alpha_normalized=alpha))
        if number%480==0:print(f'checked actual image pairs {number}/1920',flush=True)
    a=pd.DataFrame(records);private=BASE/'private_audit_only';private.mkdir(exist_ok=True)
    a.to_csv(private/'independent_folder_recheck_20261106.csv',index=False)
    confusion=pd.crosstab(a.archive_folder_class,a.predicted_class).reindex(index=CLASSES,columns=CLASSES,fill_value=0)
    confusion.to_csv(REPORT/'recheck_confusion_20261106.csv')
    counts=pd.DataFrame({'class':CLASSES,'reference_folder_images':[sum(v[0]==c for v in reference.values()) for c in CLASSES],'prediction_folder_images':[sum(v[0]==c for v in pred.values()) for c in CLASSES],'same_id_same_label':[int(confusion.loc[c,c]) for c in CLASSES]})
    counts['different_by_reference_class']=counts.reference_folder_images-counts.same_id_same_label
    counts.to_csv(REPORT/'recheck_class_counts_20261106.csv',index=False)
    old=pd.read_csv(ROOT/'data/processed/stage4_frozen_student_reference_audit_only/posthoc_reference_queue.csv',dtype={'image_id':str})
    old_mixed=set(old.loc[old.mixed_student,'image_id']);assert len(old_mixed)==570
    cohort=a[a.image_id.isin(old_mixed)]
    previous=json.loads((ROOT/'reports/errors/stage6_material_resolution_20261106/final_validation_and_private_audit.json').read_text())
    result=dict(status='passed' if not failures else 'failed',checked_at=datetime.now(ZoneInfo('Asia/Jakarta')).isoformat(),method='actual-directory enumeration, byte hashes, EXIF/white-alpha canonical pixels; no training or category changes',
        source_sha256=sha(Path(__file__)),prediction_folder=str(BASE/'train'),reference_folder=str(REFERENCE),prediction_images=len(pred),reference_images=len(reference),unique_ids_each=1920,missing_ids=0,extra_ids=0,duplicate_ids=0,
        same_id_same_folder_label=int(a.agrees.sum()),different_folder_label=int((~a.agrees).sum()),agreement_percent=100*float(a.agrees.mean()),
        image_pairs_match_after_canonical_preprocessing=int(a.pixels_match_after_canonical_preprocessing.sum()),reference_bytes_match_original_source=int(a.archive_bytes_match_source.sum()),prediction_hashes_match_original_processed=int(a.processed_hash_ok.sum()),folder_paths_match_manifest=int(a.folder_manifest_ok.sum()),
        alpha_normalized_image_ids=alpha_ids,failures=failures,former_mixed_images=len(cohort),former_mixed_agree=int(cohort.agrees.sum()),former_mixed_differ=int((~cohort.agrees).sum()),
        archive_folder_vs_historical_csv_differences=historical_differences,prior_report_count_reproduced=int(a.agrees.sum())==previous['archive_agrees'],
        all_images_visually_semantically_verified=False,semantic_accuracy=None,images_or_labels_changed=False,seconds=time.time()-started)
    result['local_gallery']=gallery_and_samples(a,private)
    (REPORT/'independent_folder_recheck_20261106.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True);print(counts.to_string(index=False),flush=True)
    assert not failures,failures
    assert result['same_id_same_folder_label']==1740 and result['different_folder_label']==180
    assert result['former_mixed_agree']==487 and result['former_mixed_differ']==83

if __name__=='__main__':main()

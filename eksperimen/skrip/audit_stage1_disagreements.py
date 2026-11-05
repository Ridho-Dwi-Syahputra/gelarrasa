"""Post-hoc archive comparison, deliberately isolated from model development.

Reads frozen cluster assignments and existing archive aliases; never fits models,
selects hyperparameters, names new clusters, or writes training labels. Agreement
with the annotation archive is not semantic accuracy or a public-error oracle.
"""
from __future__ import annotations

import hashlib
import html
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont, ImageOps
from scipy.stats import mannwhitneyu
from threadpoolctl import threadpool_limits


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding='utf-8')


def compare_numeric(frame, scope, keys):
    metrics = ['width', 'height', 'min_side', 'aspect_ratio', 'megapixels', 'brightness',
               'contrast', 'saturation', 'grayscale_entropy_bits', 'dark_fraction',
               'bright_fraction', 'laplacian_variance_256', 'edge_fraction', 'padding_fraction',
               'relative_margin', 'seed_agreement', 'centroid_distance',
               'nn10_cluster_cohesion', 'nn10_archive_agreement', 'nn1_cosine']
    rows = []
    for values, part in frame.groupby(keys, sort=True):
        if not isinstance(values, tuple):
            values = (values,)
        context = dict(zip(keys, values))
        for feature in metrics:
            bad = part.loc[part.disagrees_archive, feature].to_numpy(float)
            same = part.loc[~part.disagrees_archive, feature].to_numpy(float)
            if not len(bad) or not len(same):
                continue
            u = mannwhitneyu(bad, same, alternative='two-sided').statistic
            pooled = part[feature].to_numpy(float)
            iqr = float(np.quantile(pooled, .75) - np.quantile(pooled, .25))
            delta = float(np.median(bad) - np.median(same))
            rows.append(dict(scope=scope, **context, feature=feature,
                             n_disagrees=len(bad), n_agrees=len(same),
                             median_disagrees=float(np.median(bad)), median_agrees=float(np.median(same)),
                             median_difference=delta,
                             median_difference_over_pooled_iqr=delta/iqr if iqr > 0 else None,
                             rank_biserial=float(2*u/(len(bad)*len(same))-1),
                             interpretation='descriptive association, not causal or model-selection evidence'))
    return pd.DataFrame(rows)


def contact_sheet(root, rows, path, title):
    cols, tile_w, tile_h = 4, 304, 350
    canvas = Image.new('RGB', (cols*tile_w, 65+int(np.ceil(len(rows)/cols))*tile_h), 'white')
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 16)
        titlefont = ImageFont.truetype('C:/Windows/Fonts/arial.ttf', 20)
    except OSError:
        font = titlefont = ImageFont.load_default()
    draw.text((12, 12), title, fill='black', font=titlefont)
    draw.text((12, 39), 'Technical image inspection only; no category judgments or corrections.', fill='black', font=font)
    for i, row in enumerate(rows):
        x, y = i % cols*tile_w, 65+i//cols*tile_h
        with Image.open(root/row['processed_path']) as im:
            thumb = ImageOps.contain(im.convert('RGB'), (294, 288), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (x+(tile_w-thumb.width)//2, y+(288-thumb.height)//2))
        label = f"{row['key']}  {row['cluster_id']}\nmargin={row['relative_margin']:.3f} seed={row['seed_agreement']:.1f}\n{int(row['width'])}x{int(row['height'])} native px"
        draw.multiline_text((x+6, y+290), label, fill='black', font=font, spacing=2)
    canvas.save(path)


def choose_contacts(frame, disagrees, ascending):
    selected=[]
    for (cluster, split), part in frame[frame.disagrees_archive.eq(disagrees)].groupby(['cluster_id', 'split'], sort=True):
        selected.append(part.sort_values(['relative_margin','key'], ascending=[ascending,True]).iloc[0].to_dict())
    return selected


def gallery(root, frame, directory):
    cards=[]
    for row in frame[frame.disagrees_archive].sort_values(['split','cluster_id','relative_margin']).itertuples():
        preview=quote(Path(os.path.relpath(root/row.preview_path,directory)).as_posix(),safe='/.:')
        source=quote(Path(os.path.relpath(root/row.processed_path,directory)).as_posix(),safe='/.:')
        cards.append(f'<article data-split="{row.split}" data-cluster="{row.cluster_id}" data-key="{row.key}">'
            f'<a href="{source}"><img src="{preview}" loading="lazy" alt="{row.key}"></a>'
            f'<b>{row.key} · {row.cluster_id}</b><p>Archive: {html.escape(row.archive_label)}<br>'
            f'Frozen review alias: {html.escape(row.review_alias)}</p><small>'
            f'margin {row.relative_margin:.4f}; seed agreement {row.seed_agreement:.2f}<br>'
            f'NN10 cluster cohesion {row.nn10_cluster_cohesion:.2f}; cosine top1 {row.nn1_cosine:.3f}<br>'
            f'{int(row.width)}×{int(row.height)} native; padding {row.padding_fraction:.1%}</small></article>')
    text='''<!doctype html><meta charset="utf-8"><title>Post-hoc disagreement audit</title>
<style>body{font:15px system-ui;margin:22px;background:#f4f6f8}header{position:sticky;top:0;background:white;padding:12px;z-index:2}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px}article{background:white;padding:12px;border:1px solid #ddd;content-visibility:auto;contain-intrinsic-size:390px}img{width:100%;height:220px;object-fit:contain}small{display:block}p{margin:.5em 0}select,input{margin:5px;padding:5px}</style>
<header><b>POST-HOC ARCHIVE COMPARISON — not ground truth, not training labels</b>
<p>All 590 train and 171 test disagreements. Frozen alias mapping comes from train only. No test relabeling or tuning. Click an image for full resolution.</p>
<select id="split"><option value="">All splits</option><option>train</option><option>test</option></select>
<select id="cluster"><option value="">All clusters</option>'''+''.join(f'<option>cluster_{i:02d}</option>' for i in range(8))+'''</select>
<input id="find" placeholder="Find train:1254 or test:123"><span id="count"></span></header><main>'''+''.join(cards)+'''</main>
<script>function filter(){let n=0;document.querySelectorAll('article').forEach(a=>{let ok=(!split.value||a.dataset.split===split.value)&&(!cluster.value||a.dataset.cluster===cluster.value)&&a.dataset.key.includes(find.value);a.style.display=ok?'':'none';if(ok)n++;});document.getElementById('count').textContent=n+' shown';}const split=document.getElementById('split'),cluster=document.getElementById('cluster'),find=document.getElementById('find');[split,cluster,find].forEach(x=>x.addEventListener('input',filter));filter();</script>'''
    (directory/'all_disagreements.html').write_text(text,encoding='utf-8')


def run(root):
    root=Path(root).resolve()
    out=root/'reports/errors/stage2_20261105/diagnostics'
    out.mkdir(parents=True,exist_ok=True)
    paths=[root/'output/groups/stage1_v1/cluster_assignments.csv',
           root/'data/processed/stage1_named_review/review_alias_mapping.csv',
           root/'output/manifests/stage1_v1/image_profile.csv',
           root/'output/embeddings/stage1_v1/resnet50_features.npz',
           root/'docs/secret/submission_.csv',
           root/'notebook/01_EDA_Preprocessing_Clustering.ipynb']
    before={str(p.relative_to(root)):sha(p) for p in paths}
    cluster=pd.read_csv(paths[0],dtype={'image_id':str}).fillna('')
    source=json.loads(paths[0].with_name('cluster_provenance.json').read_text())
    assert sha(paths[0])==source['cluster_assignment_sha256']
    assert cluster.semantic_category.eq('').all()
    alias=pd.read_csv(paths[1])
    profile=pd.read_csv(paths[2],dtype={'image_id':str})
    metadata_cols=[c for c in profile.columns if c not in cluster.columns or c=='key']
    frame=cluster.merge(profile[metadata_cols],on='key',validate='one_to_one')
    frame=frame.merge(alias[['cluster_id','review_alias']],on='cluster_id',validate='many_to_one')
    rows=[]
    archive=root/'data/Dataset DSC GelarRasa 2026/train/label'
    for folder in sorted(archive.iterdir()):
        if folder.is_dir():
            rows.extend(dict(key='train:'+p.stem,archive_label=folder.name)
                        for p in folder.iterdir() if p.is_file() and p.suffix.lower() in
                        {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'})
    test=pd.read_csv(paths[4],dtype=str)
    assert sha(paths[4])==json.loads((root/'docs/secret/visual_review_99/validation.json').read_text())['sha256']
    rows.extend(dict(key='test:'+r.image_id,archive_label=r.waste_category) for r in test.itertuples())
    archive_labels=pd.DataFrame(rows)
    assert len(archive_labels)==2400 and archive_labels.key.is_unique
    assert set(archive_labels.key)==set(frame.key)
    frame=frame.merge(archive_labels,on='key',validate='one_to_one')
    frame['disagrees_archive']=frame.review_alias.ne(frame.archive_label)
    assert frame.groupby('split').disagrees_archive.sum().to_dict()=={'test':171,'train':590}
    # Read-only neighborhood diagnostics. No model fitting, selection, or labels exported to active data.
    arrays=np.load(paths[3]);assert arrays['keys'].tolist()==frame.key.tolist()
    x=arrays['features'].astype('float32')
    x=x/np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    train_idx=np.flatnonzero(frame.split.eq('train'))
    pixel=frame.pixel_sha256.to_numpy(); labels=frame.archive_label.to_numpy(); clusters=frame.cluster_id.to_numpy()
    with threadpool_limits(limits=2):
        cosine=x@x[train_idx].T
    neighbors=[]; cohesion=[]; archive_agreement=[]; top_cosine=[]
    for i,row in enumerate(frame.itertuples()):
        sim=cosine[i].copy()
        sim[pixel[train_idx]==pixel[i]]=-np.inf
        near=train_idx[np.argsort(-sim,kind='stable')[:10]]
        coh=float(np.mean(clusters[near]==clusters[i])); arch=float(np.mean(labels[near]==labels[i]))
        cohesion.append(coh); archive_agreement.append(arch); top_cosine.append(float(x[i]@x[near[0]]))
        for rank,j in enumerate(near,1):
            neighbors.append(dict(key=row.key,neighbor_key=frame.iloc[j].key,rank=rank,
                                  cosine=float(x[i]@x[j]),same_cluster=bool(clusters[i]==clusters[j]),
                                  same_archive_label=bool(labels[i]==labels[j])))
    frame['nn10_cluster_cohesion']=cohesion
    frame['nn10_archive_agreement']=archive_agreement
    frame['nn1_cosine']=top_cosine
    frame['small_margin']=frame.relative_margin.lt(.10)
    frame['seed_unstable']=frame.seed_agreement.lt(.80)
    frame['all_seeds_agree']=frame.seed_agreement.eq(1)
    frame['strong_primary_margin']=frame.relative_margin.ge(.10)
    frame.to_csv(out/'posthoc_image_diagnostics.csv',index=False)
    pd.DataFrame(neighbors).to_csv(out/'readonly_neighbor_diagnostics.csv',index=False)
    per_cluster=frame.groupby(['split','cluster_id','review_alias']).agg(
        images=('key','size'),disagreements=('disagrees_archive','sum'),
        small_margin=('small_margin','mean'),seed_unstable=('seed_unstable','mean'),
        nn_cluster_cohesion=('nn10_cluster_cohesion','mean')).reset_index()
    per_cluster['disagreement_fraction']=per_cluster.disagreements/per_cluster.images
    per_cluster.to_csv(out/'per_cluster_disagreement.csv',index=False)
    for split,data in frame.groupby('split'):
        pd.crosstab(data.archive_label,data.review_alias).to_csv(out/f'{split}_archive_vs_frozen_alias_crosstab.csv')
    global_stats=compare_numeric(frame,'split_only',['split'])
    within_stats=compare_numeric(frame,'within_split_and_cluster',['split','cluster_id'])
    global_stats.to_csv(out/'metadata_global_associations.csv',index=False)
    within_stats.to_csv(out/'metadata_within_cluster_associations.csv',index=False)
    binary=[]
    for (split,different),part in frame.groupby(['split','disagrees_archive']):
        binary.append(dict(split=split,disagrees_archive=bool(different),n=len(part),
                           small_margin_n=int(part.small_margin.sum()),
                           seed_unstable_n=int(part.seed_unstable.sum()),
                           all_seeds_agree_n=int(part.all_seeds_agree.sum()),
                           strong_margin_and_all_seeds_n=int((part.strong_primary_margin & part.all_seeds_agree).sum()),
                           low_resolution_n=int(part.min_side.lt(224).sum()),
                           high_padding_n=int(part.padding_fraction.gt(.5).sum()),
                           median_nn_cluster_cohesion=float(part.nn10_cluster_cohesion.median()),
                           median_nn_archive_agreement=float(part.nn10_archive_agreement.median()),
                           median_margin=float(part.relative_margin.median())))
    flags=pd.DataFrame(binary); flags.to_csv(out/'disagreement_diagnostic_slices.csv',index=False)
    selections={
        'disagreement_low_margin':choose_contacts(frame,True,True),
        'disagreement_high_margin':choose_contacts(frame,True,False),
        'agreement_controls':choose_contacts(frame,False,False)}
    selected=[]
    for name,records in selections.items():
        contact_sheet(root,records,out/f'{name}.png',name.replace('_',' ').title())
        selected.extend(dict(sheet=name,key=r['key'],cluster_id=r['cluster_id'],relative_margin=r['relative_margin']) for r in records)
    pd.DataFrame(selected).to_csv(out/'contact_sheet_selection.csv',index=False)
    gallery(root,frame,out)
    for relative,h in before.items():assert sha(root/relative)==h,relative
    summary=dict(total_images=2400,train_images=1920,test_images=480,
                 train_disagreements=590,test_disagreements=171,
                 labels_used_for_training=False,models_fit=0,mapping_changed=False,
                 number_of_neighbor_rows=len(neighbors),all_test_and_train_included=True,
                 gallery_disagreement_images=int(frame.disagrees_archive.sum()),
                 technical_contact_sheets={name:len(rows) for name,rows in selections.items()},
                 source_hashes_before_and_after=before,
                 disclaimer='Archive agreement is not ground truth, not accuracy, and not known public-error identity.')
    save_json(out/'audit_summary.json',summary)
    notes=['# Audit post-hoc ketidaksepakatan tahap 1','',
           'Label arsip adalah pembanding retrospektif, bukan ground truth. Mapping alias tetap berasal dari train; test tidak dipetakan ulang. Tidak ada model/parameter dipilih melalui hasil ini, tidak ada koreksi kategori, dan artefak aktif tidak berubah.','',
           '## Cakupan','',
           'Seluruh 1.920 train dan 480 test dianalisis: 590 train dan 171 test berbeda dari alias cluster. '
           'HTML memuat semua 761 kasus. Tiga contact sheet memuat sampel low-margin, high-margin, dan kontrol yang sesuai; tidak menampilkan label semantik pada gambar.','',
           '## Kestabilan bukan kebenaran','']
    for r in flags[flags.disagrees_archive].itertuples():
        notes.append(f'- {r.split}: {r.n} berbeda; {r.small_margin_n} margin<0,10; {r.seed_unstable_n} tidak stabil antar-seed; '
                     f'{r.all_seeds_agree_n} sepakat pada seluruh seed; {r.strong_margin_and_all_seeds_n} bahkan margin≥0,10 dan semua seed sepakat. '
                     f'Hanya {r.low_resolution_n} min-side<224 dan {r.high_padding_n} padding>50%.')
    notes+=['','Kasus dengan margin kuat dan seed konsisten menunjukkan bahwa hanya mengulang seed atau menghapus kasus rendah-confidence tidak mencakup seluruh ketidaksepakatan. Ini tidak membuktikan pihak mana salah. '
            'Ambang margin 0,10 merupakan irisan deskriptif tetap, bukan probabilitas benar yang terkalibrasi.','',
            '## Konsentrasi kasus','']
    for split,part in frame.groupby('split'):
        differences=part[part.disagrees_archive]
        dominant=differences.cluster_id.isin(['cluster_03','cluster_06']).sum()
        notes.append(f'- {split}: cluster_03 dan cluster_06 mencakup {dominant}/{len(differences)} '
                     f'({dominant/len(differences):.1%}) dari seluruh ketidaksepakatan.')
    notes+=['','Cluster_03 memuat delapan kategori arsip pada train; alias tetapnya hanya mencakup 78/269 train '
            'dan 7/90 test. Ini menunjukkan satu nama untuk satu cluster belum cukup mewakili isi arsip; '
            'mengganti satu alias tidak menyelesaikan semua pencampuran. Tidak dilakukan remapping.','',
            '## Metadata dan pengelompokan pembanding','',
            '`metadata_global_associations.csv` membandingkan seluruh kasus berbeda versus sesuai per split. '
            '`metadata_within_cluster_associations.csv` mengulangi pembandingan dalam cluster dan split yang sama untuk mengurangi confounding komposisi cluster. '
            'Rank-biserial positif berarti fitur cenderung lebih besar pada kelompok berbeda. Median/IQR dan ukuran sampel tersedia; '
            'tidak menganggap asosiasi sebagai penyebab, tidak melakukan pemilihan model berdasarkan label arsip.','',
            'Median resolusi test sama-sama min-side 416 piksel. Arah perbedaan ketajaman tidak konsisten antar-split: '
            'median Laplacian variance pada train berbeda/sesuai adalah 835,80/741,58, '
            'sedangkan test 561,89/880,28. Jadi tidak ada bukti bahwa blur atau resolusi rendah menjadi penjelasan tunggal.','',
            'Perbandingan dalam cluster juga menunjukkan confounding: pada test cluster_06 '
            '(46 berbeda, 41 sesuai), median padding berbeda/sesuai 0/0,25 dan brightness 0,826/0,682. '
            'Pada train cluster_05 (35 berbeda, 181 sesuai), entropy 6,952/5,419. '
            'Temuan ini mendukung pemeriksaan latar/format pengambilan per cluster; tidak membuktikan penyebab '
            'atau membenarkan crop/normalisasi otomatis. Kelompok dengan sedikit kasus perlu dibaca hati-hati.','',
            '## Tetangga fitur','',
            'Setiap gambar dicari 10 tetangga train pada embedding ResNet50 L2 asli. Seluruh canonical-pixel-identik dikeluarkan, termasuk self-match. '
            '`nn10_cluster_cohesion` mengukur proporsi tetangga dengan ID cluster yang sama. '
            '`nn10_archive_agreement` hanya diagnostik kesesuaian arsip, bukan ketepatan model. Semua baris disimpan dalam audit terpisah.','',
            'Median cohesion pada kasus berbeda adalah 0,8 train dan 0,9 test; kontrol sesuai masing-masing 1,0. '
            'Median kesesuaian arsip dengan tetangga juga masih 0,8 pada kedua split berbeda. '
            'Artinya banyak lingkungan lokal masih kompak meskipun satu alias cluster tidak cocok dengan arsip. '
            'Ini bukan pengukuran akurasi dan tidak dipakai memilih embedding/parameter.','',
            '## Batas pemeriksaan citra','',
            'Contact sheet dipilih mekanis menurut cluster/split dan margin. Ukuran objek, latar, occlusion, atau bentuk dapat ditinjau sebagai kondisi teknis; '
            'gambar tidak diberi kategori baru oleh agent. Galeri tidak membuktikan ketepatan material dan tidak mengenali satu kesalahan public dari skor 0,9930.','',
            '## Berkas','',
            '- `all_disagreements.html`: seluruh kasus dengan filter split/cluster/ID.',
            '- `posthoc_image_diagnostics.csv`: metadata, alias arsip, margin, seed, dan tetangga seluruh 2.400 gambar.',
            '- `per_cluster_disagreement.csv` dan crosstab: distribusi ketidaksepakatan, bukan confusion matrix ground truth.',
            '- `disagreement_low_margin.png`, `disagreement_high_margin.png`, `agreement_controls.png`: inspeksi teknis terpilih.',
            '- `audit_summary.json`: provenance dan bukti bahwa artefak aktif tidak berubah.']
    (out/'AUDIT_POSTHOC.md').write_text('\n'.join(notes)+'\n',encoding='utf-8')
    print(flags.to_string(index=False))
    print(per_cluster.to_string(index=False))
    print(json.dumps(summary,indent=2))
    return summary


if __name__=='__main__':
    run(Path(__file__).resolve().parents[2])

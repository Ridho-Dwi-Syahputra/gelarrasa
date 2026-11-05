"""Figures and review galleries from measured stage-1 artifacts only."""
from pathlib import Path
import html
import json
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageOps


def relative_url(path, base):
    from urllib.parse import quote
    return quote(Path(os.path.relpath(path, base)).as_posix(), safe='/.:')


def contact_sheet(ctx, rows, path, title, fields=('key',), columns=4):
    width, height = 224, 270
    canvas = Image.new('RGB', (columns * width, 48 + height * int(np.ceil(len(rows) / columns))), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((12, 15), title, fill='black')
    for index, row in enumerate(rows):
        with Image.open(ctx['work'] / row['preview_path']) as im:
            thumbnail = ImageOps.contain(im.convert('RGB'), (218, 218))
        x, y = (index % columns) * width, 48 + (index // columns) * height
        canvas.paste(thumbnail, (x + 3, y))
        label = ' | '.join(str(row.get(f, ''))[:32] for f in fields)
        draw.text((x + 4, y + 222), label[:105], fill='black')
    canvas.save(path)


def generate_figures(ctx, df, clusters):
    plt.rcParams.update({'figure.dpi': 130, 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'axes.grid': True, 'grid.alpha': .2})
    figures = ctx['report'] / 'figures'
    colors = {'train': '#176B87', 'test': '#C65125'}
    def save(fig, filename):
        fig.tight_layout()
        fig.savefig(figures / filename, bbox_inches='tight')
        plt.close(fig)
    def ecdf(ax, col, title):
        for split in ['train', 'test']:
            x = np.sort(df.loc[df.split.eq(split), col])
            ax.plot(x, np.arange(1, len(x)+1)/len(x), label=f'{split} (n={len(x)})', color=colors[split])
        ax.set_title(title)
        ax.set_ylabel('Cumulative fraction')
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for split in ['train', 'test']:
        d = df[df.split.eq(split)]
        axes[0,0].scatter(d.width, d.height, s=9, alpha=.3, label=split, color=colors[split])
    axes[0,0].set(xscale='log', yscale='log', xlabel='Width px', ylabel='Height px', title='Native image geometry')
    axes[0,0].legend()
    ecdf(axes[0,1], 'aspect_ratio', 'Aspect ratio: width / height')
    ecdf(axes[1,0], 'min_side', 'Native shortest side (px)')
    axes[1,0].axvline(224, color='black', ls='--', lw=1)
    ecdf(axes[1,1], 'padding_fraction', 'Fraction of padded area at 224 x 224')
    axes[1,1].legend()
    save(fig, '01_geometry.png')
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col, title in zip(axes.flat,
        ['brightness','contrast','saturation','grayscale_entropy_bits','laplacian_variance_256','bright_fraction'],
        ['Brightness (0-1)','Contrast (gray std / 255)','Mean saturation (0-1)',
         'Gray entropy (bits)','Laplacian variance at max-side 256','Bright pixel fraction (gray >= 247)']):
        ecdf(ax, col, title)
        if col == 'laplacian_variance_256': ax.set_xscale('symlog', linthresh=1)
    axes[0,0].legend()
    save(fig, '02_quality_indicators.png')
    drift = pd.read_csv(ctx['report'] / 'train_test_distribution.csv').sort_values('ks_statistic')
    fig, axes = plt.subplots(1,2,figsize=(13,6))
    axes[0].barh(drift.feature, drift.ks_statistic, color='#176B87')
    axes[0].set(xlabel='KS distance (0-1)', title='Train-test distribution difference')
    axes[1].barh(drift.feature, drift.js_distance_base2, color='#C65125')
    axes[1].set(xlabel='Jensen-Shannon distance (base 2)', title='Train-quantile bins, tail bins included')
    save(fig, '03_distribution_shift.png')
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    pd.crosstab(df.source_format, df.split, normalize='columns').plot.bar(ax=axes[0], color=[colors['test'],colors['train']])
    axes[0].set(title='File format proportions', ylabel='Fraction within split')
    df.groupby('split')[['mean_r','mean_g','mean_b']].mean().T.plot.bar(ax=axes[1])
    axes[1].set(title='Mean RGB intensity', ylabel='Intensity / 255')
    counts = df.groupby('split').apply(lambda d: pd.Series({'min_side<224':(d.min_side<224).mean(),
                  'padding>50%':(d.padding_fraction>.5).mean(),'grayscale':d.effective_grayscale.mean()}), include_groups=False)
    counts.T.plot.bar(ax=axes[2])
    axes[2].set(title='Technical slices (not quality labels)', ylabel='Fraction within split')
    save(fig, '04_format_and_slices.png')
    ph = pd.read_csv(ctx['report'] / 'test_train_phash_top10.csv', dtype={'test_id':str,'train_id':str})
    retr = pd.read_csv(ctx['report'] / 'test_train_embedding_top10.csv', dtype={'test_id':str,'train_id':str})
    sensitivity = pd.read_csv(ctx['report'] / 'phash_threshold_sensitivity.csv')
    fig, axes = plt.subplots(1,3,figsize=(14,4))
    for split, data in sensitivity.groupby('split_pair'):
        axes[0].plot(data.threshold, data.pair_count, marker='o', label=split)
    axes[0].set(yscale='symlog', xlabel='pHash bit-distance threshold', ylabel='Candidate pairs', title='Threshold sensitivity; not confirmed duplicates')
    axes[0].legend()
    axes[1].hist(ph.loc[ph['rank'].eq(1),'phash_distance'],bins=np.arange(0,66,2),color='#176B87')
    axes[1].set(title='Closest train per test: pHash',xlabel='Minimum original/flip Hamming distance',ylabel='Test images')
    axes[2].hist(retr.loc[retr['rank'].eq(1),'cosine_similarity'],bins=20,color='#C65125')
    axes[2].set(title='Closest train per test: CNN',xlabel='Cosine similarity',ylabel='Test images')
    save(fig,'05_similarity.png')
    projection = pd.read_csv(ctx['report'] / 'pca_navigation.csv')
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    cmap = plt.get_cmap('tab10')
    for k, (group, data) in enumerate(projection.groupby('cluster_id')):
        axes[0].scatter(data.pc1,data.pc2,s=9,alpha=.5,label=group,color=cmap(k))
    axes[0].legend(ncol=2,fontsize=8)
    axes[0].set(title='PCA navigation; distances may distort',xlabel='PC1',ylabel='PC2')
    for split, data in projection.groupby('split'):
        axes[1].scatter(data.pc1,data.pc2,s=9,alpha=.35,label=split,color=colors[split])
    axes[1].legend()
    axes[1].set(title='Same projection colored by split',xlabel='PC1',ylabel='PC2')
    save(fig,'06_cluster_navigation.png')
    fig, axes = plt.subplots(1,2,figsize=(13,5))
    pd.crosstab(clusters.cluster_id,clusters.split).plot.bar(ax=axes[0])
    axes[0].set(title='Anonymous group sizes',ylabel='Number of images')
    pd.crosstab(clusters.cluster_id,clusters.split,normalize='columns').plot.bar(ax=axes[1])
    axes[1].set(title='Within-split group proportions',ylabel='Fraction; not semantic class balance')
    save(fig,'07_cluster_populations.png')
    diag = pd.read_csv(ctx['report'] / 'cluster_diagnostics.csv')
    stab = pd.read_csv(ctx['report'] / 'cluster_stability.csv')
    names = sorted(set(stab.method_a) | set(stab.method_b))
    matrix = pd.DataFrame(np.eye(len(names)),index=names,columns=names)
    for r in stab.itertuples(): matrix.loc[r.method_a,r.method_b]=matrix.loc[r.method_b,r.method_a]=r.ari_train
    fig, axes = plt.subplots(1,2,figsize=(15,7))
    axes[0].barh(diag.method,diag.silhouette,color='#176B87')
    axes[0].set(title='Internal cohesion, not Accuracy',xlabel='Silhouette (own normalized PCA space)')
    im=axes[1].imshow(matrix, vmin=0,vmax=1,cmap='viridis')
    axes[1].set_xticks(range(len(names)), names, rotation=80,fontsize=7)
    axes[1].set_yticks(range(len(names)),names,fontsize=7)
    axes[1].set_title('Train assignment stability: adjusted Rand index')
    fig.colorbar(im,ax=axes[1],fraction=.04)
    save(fig,'08_cluster_diagnostics.png')
    from sklearn.metrics import roc_curve, roc_auc_score
    domain = pd.read_csv(ctx['report'] / 'domain_oof.csv')
    fig,axes=plt.subplots(1,2,figsize=(12,5))
    for col in ['metadata_oof_domain_probability','cnn_fixed_features_oof_domain_probability']:
        fpr,tpr,_=roc_curve(domain.domain_target,domain[col])
        auc=roc_auc_score(domain.domain_target,domain[col])
        axes[0].plot(fpr,tpr,label=f'{col.split("_oof")[0]} AUC={auc:.3f}')
    axes[0].plot([0,1],[0,1],'k--')
    axes[0].set(title='Grouped OOF train-vs-test discrimination',xlabel='False-positive rate',ylabel='True-positive rate')
    axes[0].legend(fontsize=8)
    axes[1].hist([clusters.loc[clusters.split.eq(s),'seed_agreement'] for s in ['train','test']],
                 bins=np.linspace(-.1,1.1,7),label=['train','test'])
    axes[1].legend()
    axes[1].set(title='Agreement across five seeds, not confidence',xlabel='Aligned assignment agreement',ylabel='Images')
    save(fig,'09_validation_and_stability.png')
    # Measured transform comparison: source full frame, letterbox, stretch, center crop.
    chosen = df.sort_values('aspect_ratio').head(4).index.tolist() + df.sort_values('aspect_ratio').tail(4).index.tolist()
    fig,axes=plt.subplots(len(chosen),4,figsize=(12,3*len(chosen)))
    for row, idx in enumerate(chosen):
        r=df.loc[idx]
        with Image.open(ctx['work']/r.processed_path) as im: original=im.convert('RGB').copy()
        variants=[original,Image.open(ctx['work']/r.preview_path),original.resize((224,224)),ImageOps.fit(original,(224,224))]
        for col,img in enumerate(variants):
            axes[row,col].imshow(img); axes[row,col].axis('off')
            axes[row,col].set_title(f'{r.key}\n'+['Full frame','Letterbox: selected','Stretch: geometry distortion','Center crop: illustrative only'][col],fontsize=8)
    save(fig,'10_preprocessing_comparison.png')
    saved=sorted(p.name for p in figures.glob('*.png'))
    return saved


def generate_galleries(ctx, df, clusters):
    folder=ctx['report']/'galleries'
    lookup=df.set_index('key').to_dict('index')
    merged=clusters.merge(df[['key','width','height','source_path']],on='key',validate='one_to_one')
    style='''<style>body{font:15px system-ui;margin:24px;background:#f5f7fa}a{color:#174a75}.grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(220px,1fr));gap:12px}.card{background:white;padding:10px;border:1px solid #ddd;content-visibility:auto;contain-intrinsic-size:310px}img{width:100%;height:210px;object-fit:contain}.pairs{display:grid;grid-template-columns:repeat(6,minmax(120px,1fr));gap:6px}section{padding:16px;background:white;margin-bottom:16px;content-visibility:auto;contain-intrinsic-size:610px}small{display:block}header{position:sticky;top:0;background:#fff;padding:8px;z-index:2}</style>'''
    pages=[]
    def thumb(key, text=''):
        r=lookup[key]
        src=relative_url(ctx['work']/r['preview_path'],folder)
        native=relative_url(ctx['work']/r['processed_path'],folder)
        return f'<div class="card"><a href="{native}"><img loading="lazy" src="{src}" alt="{key}"></a><b>{key}</b><small>{html.escape(text)}</small></div>'
    for (split, cluster), data in merged.groupby(['split','cluster_id']):
        filename=f'{split}_{cluster}.html'
        cards=[thumb(r.key,f'margin={r.relative_margin:.3f}; agreement={r.seed_agreement:.2f}; {r.review_reason}') for r in data.sort_values('centroid_distance').itertuples()]
        (folder/filename).write_text(f'<!doctype html><meta charset="utf-8">{style}<header><a href="index.html">Index</a> | {split} {cluster} | {len(data)} images | ordered nearest centroid first</header><p>Anonymous cluster; category name must be assigned by team. No per-image correction.</p><div class="grid">'+''.join(cards)+'</div>',encoding='utf-8')
        pages.append((filename,f'{split} {cluster}: {len(data)} images'))
        center=data.nsmallest(8,'centroid_distance')
        boundary=data.nsmallest(8,'relative_margin')
        selection=pd.concat([center,boundary]).drop_duplicates('key')
        records=[]
        for r in selection.to_dict('records'):
            r['margin']=f'margin {r["relative_margin"]:.3f}'
            records.append(r)
        contact_sheet(ctx,records,folder/f'{split}_{cluster}_review.png',f'{split} {cluster}: central and boundary examples',fields=('key','margin'))
    ph=pd.read_csv(ctx['report']/'test_train_phash_top10.csv',dtype={'test_id':str,'train_id':str})
    emb=pd.read_csv(ctx['report']/'test_train_embedding_top10.csv',dtype={'test_id':str,'train_id':str})
    sections=[]
    for test_id in df.loc[df.split.eq('test'),'image_id']:
        p=ph[ph.test_id.eq(test_id)].head(5)
        e=emb[emb.test_id.eq(test_id)].head(5)
        cards1=thumb('test:'+test_id,'QUERY')+''.join(thumb('train:'+r.train_id,f'pHash={r.phash_distance}; exact pixel={r.pixel_identical}') for r in p.itertuples())
        cards2=thumb('test:'+test_id,'QUERY')+''.join(thumb('train:'+r.train_id,f'CNN cosine={r.cosine_similarity:.4f}; {r.train_cluster}') for r in e.itertuples())
        sections.append(f'<section id="test-{test_id}"><h2>Test {test_id}</h2><p>Top 5 pHash, allowing horizontal/vertical flips</p><div class="pairs">{cards1}</div><p>Top 5 CNN cosine; similarity is not category truth</p><div class="pairs">{cards2}</div></section>')
    (folder/'all_test_neighbors.html').write_text(f'<!doctype html><meta charset="utf-8">{style}<header><a href="index.html">Index</a> | All 480 test images, both retrieval methods. Browser Find: Test ID.</header>'+''.join(sections),encoding='utf-8')
    pages.append(('all_test_neighbors.html','ALL 480 test images and nearest train candidates'))
    pairs=pd.read_csv(ctx['groups']/'phash_candidate_pairs.csv')
    sections=[]
    for r in pairs.itertuples():
        sections.append(f'<section><h3>{r.key_a} / {r.key_b}: distance {r.hamming}</h3><div class="pairs">'+thumb(r.key_a)+thumb(r.key_b)+f'</div><p>Pixel identical: {r.confirmed_pixel_identical}. Otherwise UNCONFIRMED similarity.</p></section>')
    (folder/'duplicate_candidates.html').write_text(f'<!doctype html><meta charset="utf-8">{style}<a href="index.html">Index</a>'+''.join(sections),encoding='utf-8')
    pages.append(('duplicate_candidates.html',f'{len(pairs)} perceptual duplicate candidates'))
    (folder/'index.html').write_text(f'<!doctype html><meta charset="utf-8">{style}<h1>GELAR RASA — Stage 1 review</h1><p>All images are available below. Algorithmic grouping, not semantic labels. Click a thumbnail for the native-resolution processed image.</p><ul>'+''.join(f'<li><a href="{p}">{html.escape(t)}</a></li>' for p,t in pages)+'</ul>',encoding='utf-8')
    return pages


def write_analysis(ctx, df, clusters):
    report=ctx['report']
    drift=pd.read_csv(report/'train_test_distribution.csv')
    diag=pd.read_csv(report/'cluster_diagnostics.csv')
    stab=pd.read_csv(report/'cluster_stability.csv')
    ph=pd.read_csv(report/'test_train_phash_top10.csv')
    emb=pd.read_csv(report/'test_train_embedding_top10.csv')
    domain=pd.read_csv(report/'domain_validation.csv',dtype={'fold':str})
    exact=pd.read_csv(ctx['groups']/'source_sha256_pairs.csv')
    pixel=pd.read_csv(ctx['groups']/'pixel_sha256_pairs.csv')
    candidate=pd.read_csv(ctx['groups']/'phash_candidate_pairs.csv')
    groupaudit=pd.read_csv(ctx['groups']/'candidate_group_audit.csv')
    top=ph[ph['rank'].eq(1)]
    lines=['# Tahap 1 — EDA, preprocessing, kemiripan dan clustering',
        '', 'Status: pipeline dijalankan pada seluruh sumber. Ini analisis tanpa target kategori resmi, bukan evaluasi Accuracy classifier.',
        '', '## 1. Pertanyaan dan cakupan',
        '', 'Apakah data lengkap dan dapat dibaca? Bagaimana train berbeda dari test? Apakah ada citra identik atau hampir sama? Seberapa stabil kelompok yang muncul tanpa label? Transform apa yang dapat dipertanggungjawabkan?',
        '', f'Train {int(df.split.eq("train").sum()):,}; test {int(df.split.eq("test").sum()):,}; gagal decode {int(df.decode_status.ne("ok").sum())}. ID test mengikuti sample. Kolom kategori contoh tidak digunakan. Direktori tambahan di bawah train tidak dihitung sebagai sampel.',
        '', '## 2. Integritas dan representasi',
        '', f'{len(df):,} citra telah didekode, dibuat RGB PNG native, dan diperiksa kesamaan piksel hasil simpan/baca. SHA-256 sumber diperiksa sebelum dan sesudah. EXIF diterapkan satu kali; alpha dikomposit putih. Tidak ada penghapusan citra atau koreksi kategori.',
        '', 'ICC dicatat sebagai metadata; belum dilakukan konversi profil warna ICC. Gamma/warna JPEG mengikuti decoder Pillow. Frame tambahan GIF/TIFF dicatat; pipeline ini memakai frame pertama. Lihat metadata sebelum memutuskan penanganan khusus.',
        '', '## 3. Geometri, kualitas, dan preprocessing', '']
    for split,d in df.groupby('split'):
        lines.append(f'- {split}: median ukuran {d.width.median():.0f}×{d.height.median():.0f}; min-side<224 {int(d.min_side.lt(224).sum())}/{len(d)}; padding median {d.padding_fraction.median():.1%}; grayscale efektif {int(d.effective_grayscale.sum())}/{len(d)}.')
    lines += ['', 'Native RGB dipertahankan agar resolusi model berikutnya masih dapat dipilih. Letterbox 224 hanya representasi eksplorasi backbone: seluruh frame dipertahankan, dengan padding warna rerata ImageNet. Resize meregangkan geometri, sedangkan center crop dapat menghilangkan tepi; keduanya hanya pembanding visual, bukan perubahan data aktif.',
        '', 'Blur, entropy, exposure, dan edge density adalah indikator: citra polos memang dapat memiliki entropy/ketajaman rendah. Ekor distribusi tidak otomatis berarti gambar rusak. Ambang persentil dipelajari dari train untuk antrean tinjauan, bukan seleksi/hapus data.',
        '', '## 4. Pergeseran train–test', '']
    for r in drift.head(5).itertuples():
        lines.append(f'- {r.feature}: KS={r.ks_statistic:.4f}, JS distance={r.js_distance_base2:.4f}, median train/test={r.train_median:.4g}/{r.test_median:.4g}, BH-q={r.bh_q_value:.3g}.')
    lines += ['', 'KS di sini jarak distribusi fitur, bukan metrik target kompetisi. P-value dikoreksi BH untuk banyak pengujian; pengulangan/kemiripan gambar tetap dapat melanggar asumsi independen. Effect size dan grafik lebih informatif daripada label signifikan saja. Bin JS memakai kuantil train dengan tail bins agar nilai test di luar rentang tidak hilang.', '']
    for r in domain[domain.fold.eq('pooled_oof')].itertuples():
        lines.append(f'- Domain classifier {r.feature_set}: grouped OOF AUC={r.roc_auc:.4f}. Target hanya asal split train/test; bukan kategori limbah atau Accuracy kompetisi.')
    lines += ['', 'StandardScaler dan regresi logistik domain fit di tiap train fold. Grup kandidat pHash tidak menyeberang fold. CNN sebagai fitur tetap tidak dilatih dengan domain target. Hasil ini mendeteksi keterbedaan, tidak menunjukkan penyebab kausal atau otomatis membenarkan reweighting.',
        '', '## 5. Duplikasi dan kemiripan',
        '', f'Pasangan byte-identik: {len(exact)}; pixel-identik setelah canonical RGB: {len(pixel)}. Kandidat pHash≤6: {len(candidate)}. Grup kandidat dengan risiko rantai/diameter>4: {int(groupaudit.chaining_risk.sum()) if len(groupaudit) else 0}. Tidak semua kandidat adalah foto/objek yang sama.',
        '', f'Dari {len(top)} test, {int(top.pixel_identical.sum())} memiliki train dengan pixel identik; {int(top.phash_distance.le(6).sum())} memiliki kandidat pHash≤6. Tidak ada penyalinan label maupun penggantian prediksi berdasarkan tetangga.',
        '', f'CNN nearest-train cosine median={emb.loc[emb["rank"].eq(1),"cosine_similarity"].median():.4f}. pHash menangkap struktur visual kasar; CNN dapat dekat karena bentuk, material, objek, maupun latar. Jarak/cosine bukan probabilitas benar. Seluruh 480 test tersedia dalam galeri tetangga.',
        '', '## 6. Clustering dan stabilitas',
        '', 'Primary telah ditetapkan sebelum perbandingan arsip: ResNet50 ImageNet1K V2, full-frame letterbox, L2 features, PCA64 fit pada train unik, renormalisasi, KMeans k=8 seed11 n_init10. Hanya satu wakil pixel-identik masuk fitting PCA/cluster. Test diproyeksikan dan diassign ke centroid train tanpa fit ulang.',
        '', 'Pembanding: lima seed CNN, dua seed fitur HSV+HOG, Ward8, serta k=6/10/12/16 sebagai sensitivitas eksplorasi. Ini belum pemilihan kategori atau izin memperlakukan subcluster sebagai label akhir. Silhouette antar representasi tidak otomatis sebanding dan tidak menentukan kebenaran semantik.', '']
    primary=diag[diag.method.eq('cnn_k8_seed11')].iloc[0]
    same=stab[stab.method_a.str.startswith('cnn_k8_') & stab.method_b.str.startswith('cnn_k8_')]
    lines += [f'Primary silhouette={primary.silhouette:.4f}; ukuran cluster train unik {int(primary.smallest_cluster)}–{int(primary.largest_cluster)}. ARI antar seed CNN: minimum {same.ari_train.min():.4f}, median {same.ari_train.median():.4f}.',
        '', f'Antrean tinjauan algoritmis: {int(clusters.review_reason.ne("").sum())} gambar. Alasan: margin centroid<0,10, agreement seed<0,80, atau jarak di atas p95 cluster train. Ini indikator ketidakstabilan, bukan bukti salah label; tidak ada perbaikan per gambar otomatis/manual.',
        '', 'Folder cluster_00…cluster_07 berisi gambar hasil preprocessing. Nama glass/paper/dan seterusnya belum dipetakan; anggota tim perlu menamai cluster dengan dasar yang dicatat. Satu cluster tidak dijamin sama dengan satu kelas; pemetaan kosong lebih jujur daripada memaksakan delapan kategori.',
        '', '## 7. Validasi yang dapat dan belum dapat disimpulkan',
        '', 'Draft 5-fold memakai group kandidat dan stratifikasi cluster anonim. Tidak ada grup kandidat menyeberang fold. Karena clustering fit pada seluruh train, agreement classifier terhadap target cluster global bukan validasi semantik independen. Pada tahap berikutnya, learned preprocessing/target construction perlu dibatasi ke train fold jika mengklaim out-of-sample; test Kaggle tetap tersembunyi.',
        '', 'Belum ada Accuracy kategori, confusion matrix kategori, atau classifier limbah yang dilatih/fine-tune. Backbone frozen hanya untuk eksplorasi. Jalur final pretrained memerlukan pelatihan/fine-tuning kompetisi sesuai aturan. Public sempurna pun tidak memastikan private sempurna.',
        '', '## 8. Pelajaran INFEST dan webinar',
        '', 'Penilaian INFEST menekankan keterkaitan temuan data→preprocessing→evaluasi, visualisasi bernarasi, serta notebook yang benar-benar dieksekusi. Temuan tersebut diadopsi; aturan dan Macro-F1 INFEST tidak dipindahkan ke GELAR RASA. Laporan penelaahan Excel terpisah memuat sumber sheet/sel.',
        '', 'Materi 1 h.11–14: representasi RGB, split, augmentasi dan baseline; h.17–18 fitur hybrid; h.20 transform hati-hati; h.22–29 evaluasi/learning curves. Materi 2 h.12–19 dan 33–34: grafik, denominator, konsistensi, integritas. Tidak mengasumsikan bahwa perbaikan visual otomatis memperbaiki klasifikasi.',
        '', '## 9. Keputusan dan langkah berikutnya',
        '', '1. Gunakan manifest sumber, bukan recursive glob seluruh train.',
        '2. Pertahankan native RGB serta koordinat/rasio letterbox; jangan denoise/CLAHE/crop massal tanpa ablasi.',
        '3. Tinjau grup near-duplicate dan risiko rantainya sebelum menjadikan group_id sebagai split final.',
        '4. Tim memetakan nama kategori cluster dan mencatat bila kelompok tidak koheren; jangan memaksakan koreksi satu per satu.',
        '5. Bekukan versi label dan split yang sah, lalu lakukan fine-tuning serta evaluasi tahap berikutnya. Jangan memakai skor internal clustering sebagai estimasi leaderboard.',
        '', '## Artefak dan reproduksi',
        '', '- `output/manifests/stage1_v1/`: sumber, metadata, SHA-256.',
        '- `data/processed/stage1_v1/`: native RGB, letterbox, folder cluster train/test, template mapping.',
        '- `output/groups/stage1_v1/`: exact groups, kandidat, cluster assignment dan provenance.',
        '- `reports/eda/stage1_20261105/galleries/index.html`: semua gambar dan tetangga.',
        '- `notebook/01_EDA_Preprocessing_Clustering.ipynb`: kode mandiri dan hasil eksekusi.',
        '', 'Keluaran processed diperlakukan immutable; cluster_views memakai hardlink bila filesystem mendukung, jadi mengedit file di satu view juga mengubah salinan native yang terkait. Untuk perubahan baru gunakan versi output baru.',
        '', 'Tanggal/asumsi aturan: 5 November 2026. Bantuan penulisan kode berdasarkan klarifikasi pengguna diperbolehkan selama notebook tidak memanggil LLM; notebook ini tidak memakai layanan inference eksternal atau VLM.']
    text='\n'.join(lines)+'\n'
    (report/'ANALISIS_TAHAP_1.md').write_text(text,encoding='utf-8')
    (ctx['processed']/'README.md').write_text('# Data preprocessing tahap 1\n\nLihat `reports/eda/stage1_20261105/ANALISIS_TAHAP_1.md` dan notebook.\n\n- `rgb_native/train`, `rgb_native/test`: full-resolution canonical RGB PNG.\n- `letterbox224/train`, `letterbox224/test`: full-frame input for exploratory CNN.\n- `cluster_views/train/cluster_00...07` dan `cluster_views/test/cluster_00...07`: kelompok algoritmis, belum nama kategori.\n- `cluster_category_mapping_TEMPLATE.csv`: diisi tim pada tingkat cluster. Tidak ada kategori yang ditebak oleh penulis kode.\n\nCluster views may be hardlinks. Treat as immutable. No class training targets or submission produced here.\n',encoding='utf-8')
    return text

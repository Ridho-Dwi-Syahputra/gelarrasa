"""Build the visual v3 main notebook; original executed v2 is backed up separately."""
from pathlib import Path
import hashlib
import nbformat

ROOT=Path(__file__).resolve().parents[2]
nb=nbformat.v4.new_notebook()
nb.metadata.kernelspec=dict(name='gelar-rasa-venv',display_name='Python (GELAR-RASA .venv)',language='python')
nb.metadata.gelar_pipeline_version='eda_visual_v3'
nb.metadata.execution_report_prefix='notebook_v3'
nb.metadata.execution_scope='Visual EDA, all-image region proposals, sampled crop probes, all-test pixel retrieval. Existing trained baseline reused after integrity checks unless REBUILD_BASELINE=True.'
nb.metadata.pipeline_sources={'stage3_visual_audit.py':hashlib.sha256((ROOT/'eksperimen/skrip/stage3_visual_audit.py').read_bytes()).hexdigest()}
cells=[]
def md(text):cells.append(nbformat.v4.new_markdown_cell(text))
def code(text):cells.append(nbformat.v4.new_code_cell(text))

md('''# GELAR RASA — EDA visual, preprocessing, crop, dan kemiripan piksel

Analisis **1.920 train dan 480 test**, dengan penekanan pada contoh gambar test, kualitas gambar, kandidat beberapa objek, serta hubungan test–train pada tingkat piksel dan ukuran native.

**Asal nama tampilan:** atas permintaan pengguna, asisten meninjau 128 contoh hasil model dan memberi interpretasi visual sementara pada tingkat kelompok. Ini EDA pribadi, bukan label resmi atau pemetaan kategori yang disahkan untuk kompetisi. Aturan yang diberikan pengguna melarang LLM/VLM untuk klasifikasi kompetisi; interpretasi ini dipisahkan dari target training, validasi, dan submission. Tidak ada panggilan LLM/VLM/API inference dalam notebook.

Delapan kelompok model belum terpisah sempurna menjadi delapan material. Kelompok campuran ditulis **plastic / glass** dan **plastic / textile / paper**, sehingga gambar kaca tidak dianggap tidak ada hanya karena belum memiliki kelompok terpisah. Jumlah per nama tampilan merupakan ukuran kelompok, bukan jumlah ground truth kategori.

Semua pekerjaan dijalankan pada `.venv` proyek. Output disimpan di notebook; agent menjalankan komputasi di latar belakang tanpa mengambil fokus layar.''')
code('''from pathlib import Path
import os, sys, json, hashlib, subprocess
from concurrent.futures import ThreadPoolExecutor
from IPython.display import display, Markdown, Image as DisplayImage
PROJECT_ROOT = Path(os.environ['GELAR_PROJECT_ROOT']).resolve() if 'GELAR_PROJECT_ROOT' in os.environ else next(p for p in [Path.cwd(),*Path.cwd().parents] if (p/'configs/data_contract.json').exists())
assert Path(sys.executable).resolve() == (PROJECT_ROOT/'.venv'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')).resolve(), 'Pilih Python (GELAR-RASA .venv)'
sys.path.insert(0,str(PROJECT_ROOT/'eksperimen/skrip'))
import stage3_visual_audit as audit
import pandas as pd
REBUILD_BASELINE = False
print('Interpreter:',sys.executable)
print('Rebuild baseline:',REBUILD_BASELINE)
print('Notebook v3: analisis visual baru, menggunakan head baseline yang sudah dilatih.')''')
md('''## 1. Input, baseline terlatih, dan integritas

Default memakai head DINOv2 yang sudah dilatih pada train pada run sebelumnya. Ini tidak mengklaim training baru. Jika artefak belum ada setelah clone, sel berikut menjalankan resep baseline dari data resmi lokal; `REBUILD_BASELINE=True` juga membangun ulangnya. Data, checkpoint, dan gambar tidak disertakan dalam Git.

Sebelum audit baru, hash 2.400 sumber serta PNG native diperiksa. Fitur harus cocok dengan checksum yang dipakai head, dan checkpoint harus mereproduksi seluruh assignment. Arsip label tidak dibaca.''')
code('''required = ['output/manifests/stage1_v1/image_profile.csv', 'output/groups/stage1_v1/groups.csv',
 'output/folds/stage1_v1/DRAFT_group_folds.csv', 'reports/eda/stage1_20261105/test_train_phash_top10.csv',
 'output/embeddings/stage2_repr_v1/dinov2_features.npz', 'output/groups/stage2_neighbor_v1/head.pt',
 'output/groups/stage2_neighbor_v1/assignments.csv']
if REBUILD_BASELINE or not all((PROJECT_ROOT/p).exists() for p in required):
    recipes=[('stage1_eda.py',[]),('stage2_visual_features.py',[]),('stage2_neighbor_refinement.py',[])]
    log=PROJECT_ROOT/'reports/reproducibility/baseline_v3_build.log'
    log.parent.mkdir(parents=True,exist_ok=True)
    with log.open('w',encoding='utf-8') as stream:
        for script,args in recipes:
            print('Menjalankan:',script,flush=True)
            environment=dict(os.environ,GELAR_RECOMPUTE_FEATURES='1' if REBUILD_BASELINE else '0')
            subprocess.run([sys.executable,str(PROJECT_ROOT/'eksperimen/skrip'/script),*args],cwd=PROJECT_ROOT,env=environment,stdout=stream,stderr=subprocess.STDOUT,check=True)
else:
    print('Artefak baseline tersedia; fitting tidak diulang untuk tugas EDA ini.')
ctx = audit.load_context()
def verify_source(row):
    assert audit.sha(PROJECT_ROOT/'data'/row.source_path)==row.source_sha256,row.key
    assert audit.sha(PROJECT_ROOT/row.processed_path)==row.processed_sha256,row.key
    return 1
with ThreadPoolExecutor(max_workers=4) as pool:
    checked=sum(pool.map(verify_source,list(ctx['df'].itertuples())))
print('Sumber + native PNG diperiksa:',checked)
display(ctx['df'].groupby('split').agg(gambar=('key','size'),piksel_unik=('pixel_sha256','nunique')))
display(ctx['df'][['key','width','height','source_format','display_label']].head(12))''')
md('''## 2. Distribusi label tampilan dan dimensi

Grafik memakai nama material, bukan nomor kelompok. Nama campuran dipertahankan. Tabel delapan kelas resmi disertakan untuk memperjelas bahwa jumlah kategori terverifikasi belum tersedia; nilai kosong tidak diubah menjadi nol.

Semua dimensi mengacu pada RGB native sesudah EXIF orientation, sebelum resize/padding. Ukuran yang sering muncul dapat mengindikasikan proses pengumpulan atau resize sumber, tetapi tidak membuktikan kelas maupun duplikat.''')
code('''distribution, official_counts, dimensions = audit.distribution_tables(ctx)
display(distribution)
display(DisplayImage(filename=str(audit.REPORT/'figures/distribution.jpg')))
display(official_counts)
display(dimensions.sort_values('jumlah',ascending=False).head(20))
size_names=ctx['df'].assign(ukuran=lambda d:d.width.astype(str)+'×'+d.height.astype(str))
dimension_labels=size_names.groupby(['split','ukuran','display_label']).size().rename('jumlah').reset_index()
dimension_labels.to_csv(audit.REPORT/'dimensions_by_display_name.csv',index=False)
display(dimension_labels.sort_values('jumlah',ascending=False).head(20))
display(DisplayImage(filename=str(audit.REPORT/'figures/dimensions_quality.jpg')))''')
md('''## 3. Preview train dan test

Sampel acak memakai seed tetap dan tidak dipilih berdasarkan label arsip. Preview test lebih banyak agar cakupan masalah test mudah diperiksa. Caption menampilkan ID gambar, dimensi native, dan nama tampilan sementara.''')
code('''for split,n in [('train',12),('test',24)]:
    rows=ctx['df'][ctx['df'].split.eq(split)].sample(n,random_state=20261106).sort_values('key')
    path=audit.contact_sheet(ctx,rows,f'{split}_random_preview',f'Preview {split}: sampel acak seed tetap')
    display(DisplayImage(filename=str(path)))
for split in ['train','test']:
    rows=pd.concat([g.sort_values('model_margin',ascending=False).head(2) for _,g in ctx['df'][ctx['df'].split.eq(split)].groupby('cluster_index')])
    path=audit.contact_sheet(ctx,rows,f'{split}_named_representatives',f'Contoh {split} per nama tampilan sementara')
    display(DisplayImage(filename=str(path)))''')
md('''## 4. Gambar berdekatan atau bersentuhan: kandidat region dan crop

Seluruh 2.400 gambar diperiksa. Perbedaan warna Lab terhadap tepi gambar membentuk hipotesis foreground, lalu connected components mencari region terpisah. Jika region menyatu, distance transform dan watershed dengan marker mengusulkan pemisahan.

**Region bukan kepastian objek:** satu benda dapat terpecah menjadi beberapa region; latar ramai juga dapat dianggap foreground. Watershed dapat menghasilkan oversegmentation. Crop hanya kandidat, bukan preprocessing massal yang sudah terbukti meningkatkan Accuracy. Semua koordinat dikembalikan ke piksel native dan hasil crop diuji persis terhadap potongan gambar induk.

Train disimpan di `data/processed/stage3_crop_candidates_v1/train_candidates/<ID>/`; crop test berada di `test_previews/`. Seluruh crop tetap membawa parent ID, hash sumber, candidate-group dan fold induk. Crop test tidak menjadi data train. Label induk tidak diwariskan karena dua region dapat berbeda material.

Dasar metode: [OpenCV watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html).''')
code('''regions, crop_manifest = audit.crop_audit(ctx)
display(regions.groupby('split').agg(gambar=('key','size'),kandidat_beberapa_region=('possible_multiple_regions','sum')))
display(crop_manifest.groupby(['split','proposal_method']).size().rename('jumlah_crop').reset_index())
display(crop_manifest[['parent_key','split','x0','y0','x1','y1','crop_width','crop_height','parent_fold','augmentation_candidate','label_status']].head(15))
previews,flag_counts = audit.make_previews(ctx)
display(flag_counts)
for name in ['test_low_sharpness','test_busy_scene','test_novel_appearance','test_ambiguous']:
    display(DisplayImage(filename=str(previews[name])))''')
md('''## 5. Preview masalah test dan dampak crop

Ketajaman memakai Laplacian variance pada ukuran terkontrol; permukaan polos bisa bernilai rendah tanpa blur. Tepi padat menandai tekstur/latar ramai, bukan label acak. Jarak embedding dan margin head memberi daftar kasus model ragu. Batas teknis berasal dari train, bukan tuning terhadap test.

Probe crop dipilih dengan seed tetap: maksimal 96 crop train dan 48 crop test. Head yang sama membandingkan crop dan flip-nya dengan prediksi gambar penuh. Perbedaan hasil menunjukkan sensitivitas konteks, bukan bukti crop lebih benar. Probabilitas belum dikalibrasi. Tidak ada crop yang dipromosikan menjadi target training.

Preview di bawah memperlihatkan sumber dengan kotak kandidat dan potongannya. Metode ini juga memiliki kegagalan pada benda panjang/tekstur atau objek yang menyentuh tepi.''')
code('''crop_probe = audit.crop_model_probe(ctx)
if not crop_probe.empty:
    display(crop_probe.groupby('split').agg(crop_diuji=('crop_key','size'),berbeda_dari_gambar_penuh=('crop_changed_model_group','sum'),stabil_flip=('crop_flip_stable','sum')))
    display(crop_probe[['parent_key','split','parent_display_label','crop_display_label','crop_probability_uncalibrated','crop_changed_model_group','crop_flip_stable']].head(16))
for split in ['train','test']:
    for method in ['connected_components','watershed_markers']:
        display(DisplayImage(filename=str(audit.crop_previews(ctx,split,method))))
conservative=crop_manifest[crop_manifest.augmentation_candidate]
print('Kandidat konservatif train:',len(conservative),'crop dari',conservative.parent_key.nunique(),'gambar induk; belum menjadi label training.')''')
md('''## 6. Kemiripan piksel seluruh test–train

Pencarian RGB MSE 64×64 dilakukan pada semua **480×1.920 pasangan**, termasuk horizontal flip. SSIM 128×128 kemudian dihitung pada gabungan kandidat RGB, flip, pHash, dan embedding. Ini pencarian SSIM pada shortlist, bukan SSIM menyeluruh untuk semua pasangan.

Skor setelah resize dapat tinggi karena latar serupa. Karena itu, tampilkan juga ukuran native, beda aspect ratio, hash byte, hash piksel, dan MAE native bila ukuran memungkinkan. Heatmap memperlihatkan selisih setelah alignment. Kesamaan ukuran sendiri bukan bukti gambar sama atau label sama.

[Definisi dan parameter SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity): `data_range=255` dinyatakan eksplisit. Resize full-frame untuk pengukuran dicatat sebagai transform yang dapat mendistorsi aspect ratio.''')
code('''pixel_pairs, pixel_best = audit.pixel_neighbors(ctx)
display(pixel_best[['test_key','train_key','test_display_label','train_display_label','ssim128_best','same_native_dimensions','pixel_identical','train_images_with_test_dimensions']].head(20))
display(pixel_best.groupby(['same_native_dimensions','pixel_identical']).agg(test=('test_key','size'),median_ssim=('ssim128_best','median')))
display(DisplayImage(filename=str(audit.similarity_previews(ctx))))
pixel_stats=json.loads((audit.REPORT/'pixel_method.json').read_text())
print('Test pixel identik dengan train:',pixel_stats['exact_pixel_test_count'])
print('Kandidat SSIM>=0.99:',pixel_stats['ssim_ge_099_candidates'])
print('Kandidat setelah saringan MSE/aspect ratio:',pixel_stats['conservative_similarity_candidates'])
print('Test dengan ukuran native tersedia di train:',pixel_stats['test_same_size_available'])
print('Saringan kemiripan adalah kandidat, bukan pembuktian kategori atau kesamaan sumber.')
display(Markdown('**Cakupan:** setiap test mendapat satu tetangga terbaik pada shortlist; ini bukan label test terverifikasi.'))''')
md('''## 7. Seluruh 480 test: galeri berhalaman

Galeri mencakup setiap test tepat sekali, dengan gambar sumber, tetangga train, dimensi, SSIM, dan flag teknis. Gambar tidak dihapus karena blur, ambigu, atau berbeda metode. File galeri dan gambar berada lokal; dataset/gambar terpisah tidak di-push.''')
code('''gallery = audit.all_test_gallery(ctx)
print('Galeri:',gallery)
display(Markdown('[Buka indeks galeri 480 test](../reports/eda/stage3_visual_20261106/galleries/index.html)'))
display(ctx['df'].groupby('split')[['flag_low_sharpness','flag_exposure','flag_novel_appearance','possible_multiple_regions']].sum())''')
md('''## 8. Keputusan preprocessing dan pemeriksaan akhir

1. Native RGB dan gambar penuh dipertahankan sebagai acuan. Tidak ada auto-drop atau koreksi label per gambar.
2. Crop region disimpan untuk eksperimen terkontrol. Connected components pada latar seragam lebih konservatif; watershed tetap hipotesis yang berisiko memotong satu benda menjadi dua.
3. Train/validasi harus dipisah berdasarkan kelompok induk sebelum menggunakan crop. Semua turunan satu sumber mengikuti fold yang sama.
4. Test tetap satu baris per gambar pada tugas single-label. Memilih salah satu objek test membutuhkan aturan/model yang telah divalidasi; jumlah objek tidak mengubah format submission.
5. Jumlah nama tampilan campuran menunjukkan keterbatasan partisi model. Menamai ulang saja belum menghasilkan classifier delapan material atau Accuracy kategori.

Sel terakhir memastikan sumber model tetap identik. Crop tidak mengubah bobot, label training, atau assignment baseline.''')
code('''summary = audit.finalize(ctx)
display(summary)
assert summary['test_images']==summary['tested_test_neighbors']==summary['all_test_preview_coverage']==480
assert not summary['training_labels_changed'] and not summary['test_crops_used_for_training']
print('RUN ALL EDA VISUAL SELESAI — preview, crop kandidat, dan analisis piksel tersimpan.')''')
nb.cells=cells
for i,cell in enumerate(nb.cells):cell['id']=hashlib.sha256(f'{i}:{cell.cell_type}:{cell.source}'.encode()).hexdigest()[:12]
nbformat.validate(nb)
path=ROOT/'notebook/01_EDA_Preprocessing_Clustering.ipynb';nbformat.write(nb,path)
print('Built',path,'with',sum(c.cell_type=='code' for c in cells),'code cells.')

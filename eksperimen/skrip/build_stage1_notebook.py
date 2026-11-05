"""Build the main EDA notebook: embedded stage1 and versioned stage2 modules."""
import hashlib
from pathlib import Path
import nbformat

root = Path(__file__).resolve().parents[2]
pipeline_path = root / 'eksperimen/skrip/stage1_eda.py'
report_path = root / 'eksperimen/skrip/stage1_report.py'
pipeline = pipeline_path.read_text(encoding='utf-8').split("\nif __name__ == '__main__':")[0]
report = report_path.read_text(encoding='utf-8')
nb = nbformat.v4.new_notebook()
nb.metadata.kernelspec = {'display_name':'Python (GELAR-RASA .venv)','language':'python','name':'gelar-rasa-venv'}
nb.metadata.language_info = {'name':'python','version':'3.10'}
stage2_paths = [root/'eksperimen/skrip'/name for name in [
    'stage2_visual_features.py', 'stage2_representation_ablation.py',
    'stage2_neighbor_refinement.py', 'stage2_evaluate_views.py',
    'materialize_stage2_clusters.py']]
nb.metadata.pipeline_sources = {p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in [pipeline_path,report_path,*stage2_paths]}
nb.metadata.gelar_pipeline_version = 'eda_clustering_v2'
cells=[]
def md(s): cells.append(nbformat.v4.new_markdown_cell(s))
def code(s): cells.append(nbformat.v4.new_code_cell(s))
md('''# GELAR RASA — EDA, Preprocessing, Kemiripan, dan Clustering

**Tujuan:** memahami 1.920 train tanpa target kategori resmi dan 480 test, menyusun preprocessing yang dapat dilacak, memeriksa kemiripan, serta menghasilkan kelompok anonim untuk ditelaah tim.

**Batas hasil:** cluster tidak sama dengan kelas. Tidak ada Accuracy kategori atau submission pada tahap ini. Nama kategori diberikan tim pada tingkat cluster setelah hasil diperiksa. Tidak ada pemanggilan layanan inference eksternal.

**Cara menjalankan:** pilih kernel **Python (GELAR-RASA .venv)** lalu jalankan dari atas ke bawah. Sel awal memeriksa interpreter `.venv` proyek. Data mengikuti struktur `data/Dataset DSC GelarRasa 2026/{train,test}` dan `data/SampleSubmission_DSC.csv`. Ubah `PROJECT_ROOT` bila perlu. Alur gabungan tahap 1–2 memakai output pada root proyek yang sama. Bobot ResNet50 ImageNet1K V2 dan DINOv2 image-only dimuat dari cache lokal atau unduhan checkpoint resmi. Tidak ada pemasangan paket otomatis di dalam notebook.

**Urutan:** integritas → metadata/transform → hash → distribusi → baseline ResNet → retrieval/domain → grafik/galeri → DINOv2 → ablasi representasi → training head tetangga → uji view → folder train/test → verifikasi. Default `RUN_STAGE2=True` dan `RECOMPUTE_DINO=True` menjalankan komputasi terbaru, bukan hanya membaca tabel hasil sebelumnya. Durasi CPU dapat mencapai puluhan menit; output disimpan per sel oleh runner proyek.''')
code('''from pathlib import Path
import os
import sys
import time
import json
import hashlib
import importlib.util
import importlib.metadata
from IPython.display import display, Markdown, Image as DisplayImage

PROJECT_ROOT = Path(os.environ['GELAR_PROJECT_ROOT']).resolve() if 'GELAR_PROJECT_ROOT' in os.environ else next(p for p in [Path.cwd(), *Path.cwd().parents] if (p/'configs/data_contract.json').exists())
RUN_STAGE2 = True
RECOMPUTE_DINO = True
expected_python = PROJECT_ROOT / '.venv' / ('Scripts/python.exe' if os.name == 'nt' else 'bin/python')
assert os.path.normcase(str(Path(sys.executable).resolve())) == os.path.normcase(str(expected_python.resolve())), (
    f'Pilih kernel Python (GELAR-RASA .venv). Interpreter aktif: {sys.executable}')
if RUN_STAGE2:
    assert Path(os.environ.get('GELAR_WORK_ROOT', str(PROJECT_ROOT))).resolve() == PROJECT_ROOT, 'Tahap 2 memerlukan output pada PROJECT_ROOT.'
    assert Path(os.environ.get('GELAR_DATA_ROOT', str(PROJECT_ROOT/'data'))).resolve() == PROJECT_ROOT/'data', 'Alur gabungan memakai data proyek.'
NOTEBOOK_STARTED = time.time()
print('Project:', PROJECT_ROOT)
print('Interpreter:', sys.executable)
print('Stage 2:', RUN_STAGE2, '| Recompute DINO:', RECOMPUTE_DINO)
print('Python:', sys.version.split()[0])
print({name:importlib.metadata.version(name) for name in ['numpy','pandas','torch','torchvision','timm','scikit-learn','Pillow','nbclient','ipykernel']})''')
md('''## 1. Metode yang dapat direproduksi

Kode fungsi lengkap disertakan agar notebook tidak bergantung pada berkas skrip yang tersembunyi. Seed, transform, lingkungan, dan bobot dicatat. Membaca fungsi saja belum berarti hasil telah dihitung; sel berikutnya mengeksekusi tiap tahap.

Inspirasi kualitas penulisan berasal dari umpan balik INFEST: hubungan temuan data dengan preprocessing, visualisasi disertai interpretasi, dan hasil eksekusi tersimpan. Aturan dan metrik INFEST tidak dipindahkan ke kompetisi ini.''')
code(pipeline)
code(report)
code("ctx = setup(PROJECT_ROOT)\ndisplay(pd.DataFrame([CONFIG]))\nprint((ctx['report']/'environment.json').read_text())")
md('''## 2. Cakupan input dan identitas gambar

Train dipindai hanya pada tingkat folder sumber. Setiap stem ID harus unik. Seluruh test harus cocok dengan ID dan urutan sample submission; isi contoh kategori tidak diperlakukan sebagai target. SHA-256 mengunci sumber untuk pemeriksaan perubahan sesudah pipeline.''')
code("sources = inventory(ctx)\ndisplay(sources.groupby('split').agg(images=('key','size'), total_bytes=('byte_size','sum')))\ndisplay(sources.head())")
md('''## 3. Decode, warna, orientasi, kualitas, dan preprocessing

Seluruh gambar didekode penuh. EXIF orientation diterapkan sekali, alpha dikomposit pada putih, lalu RGB uint8 disimpan sebagai PNG native tanpa resize. Kesamaan piksel sebelum dan sesudah simpan diperiksa untuk setiap gambar. Letterbox 224 mempertahankan frame penuh untuk ekstraksi fitur; native tetap tersedia untuk resolusi lain.

Brightness/contrast memakai gray 0–1; entropy dalam bit; Laplacian variance pada gambar dengan sisi maksimum 256 tanpa padding. Nilai-nilai ini indikator struktur/kualitas, bukan label buruk. Tidak ada penghapusan citra berdasarkan ambang atau penerapan CLAHE/denoise massal.''')
code("profile = profile_and_preprocess(ctx, sources)\ndisplay(profile.groupby('split')[['width','height','aspect_ratio','brightness','contrast','saturation','padding_fraction']].describe().T)\ndisplay(profile.groupby(['split','source_format','source_mode']).size().rename('n').reset_index())")
md('''## 4. Exact duplicate, pHash, dan risiko kelompok berantai

SHA-256 byte dan hash canonical pixel membuktikan dua jenis kesamaan yang berbeda. pHash 64 bit dibanding dengan jarak Hamming, juga terhadap horizontal/vertical flip. Ambang 6 hanya menghasilkan kandidat. Ambang 4 membentuk kelompok konservatif untuk draft validasi; diameter kelompok dicatat untuk mendeteksi rantai A≈B≈C yang belum tentu A≈C.

Semua train–test, train–train, dan test–test diperiksa. Tidak ada penggantian label atau prediksi dari pasangan mirip.''')
code("groups = audit_duplicates(ctx, profile)\nfor filename in ['source_sha256_pairs.csv','pixel_sha256_pairs.csv','candidate_group_audit.csv']:\n    table=pd.read_csv(ctx['groups']/filename)\n    print(filename, len(table))\n    display(table.head(20))\ndisplay(pd.read_csv(ctx['report']/'phash_threshold_sensitivity.csv'))")
md('''## 5. Distribusi train–test dan daftar tinjauan teknis

KS, Wasserstein, dan Jensen–Shannon mengukur aspek perbedaan distribusi. Bin kuantil ditentukan dari train dan memiliki tail bins; nilai test di luar rentang tetap dihitung. P-value KS disertai koreksi Benjamini–Hochberg, namun ketergantungan gambar tetap membatasi interpretasi signifikansi.

Antrean teknis memakai ekor 1%/99% train dan native min-side<224. Flag ini tidak menghapus gambar atau menilai kategori.''')
code("drift = distribution_audit(ctx, profile)\ndisplay(drift)\nprint('Technical flags:', len(pd.read_csv(ctx['report']/'technical_review_queue.csv')))")
md('''## 6. Representasi visual: CNN dan pembanding klasik

ResNet50 ImageNet1K V2 dipakai sebagai extractor **non-VLM**, dengan layer klasifikasi dilepas. Tidak ada target kategori maupun teks prompt. Input memakai letterbox dan normalisasi ImageNet; berbeda dari default center crop, sehingga perubahan ini dicatat. Vector 2048 dimensi dinormalisasi L2.

Pembanding HSV histogram + HOG membantu menguji apakah struktur cluster hanya mengikuti warna/tekstur. Descriptor klasik dihitung pada resize langsung 128×128, sehingga rasio aspek dapat berubah; ini keterbatasan pembanding fitur, bukan transform untuk PNG native atau input CNN utama. Perbedaan cluster kedua metode tidak dapat diatribusikan hanya ke arsitektur karena representasi dan transformnya juga berbeda. Frozen extractor ini untuk EDA, belum model kompetisi yang telah di-fine-tune.''')
code("cnn_features = extract_features(ctx, profile)\nprint('Feature matrix:', cnn_features.shape)\nprint((ctx['embeddings']/'feature_provenance.json').read_text())")
md('''## 7. Clustering tanpa target kategori

Primary ditetapkan sebelum evaluasi: PCA64 fit pada satu wakil tiap pixel-identik train, normalisasi ulang, KMeans k=8 seed11 n_init10. Test hanya ditransform dan ditempatkan ke centroid train. Lima seed CNN, dua seed klasik, Ward8, serta k=6/10/12/16 menjadi pembanding diagnostik.

Silhouette, Calinski–Harabasz, Davies–Bouldin, dan ARI mengukur sifat internal/stabilitas, **bukan Accuracy**. ARI mengabaikan permutasi ID; agreement per gambar memakai matching antar seed pada train. Margin centroid dan jarak outlier memberi antrean tinjauan algoritmis.

Folder `cluster_00`–`cluster_07` memuat kelompok gambar hasil preprocessing. Template nama kategori dibiarkan kosong untuk pemetaan oleh anggota tim; delapan cluster tidak menjamin delapan kelas semantik terpisah.''')
code("clusters = fit_clusters(ctx, profile, cnn_features, groups)\ndisplay(pd.read_csv(ctx['report']/'cluster_diagnostics.csv'))\ndisplay(pd.crosstab(clusters.cluster_id,clusters.split))\ndisplay(pd.read_csv(ctx['report']/'cluster_stability.csv').head(15))\nprint('Review queue:', clusters.review_reason.ne('').sum())")
md('''## 8. Kemiripan setiap test dengan train

Top-10 pHash dan top-10 CNN disimpan untuk semua 480 test. Galeri menampilkan lima tetangga dari masing-masing metode. Retrieval train–train mengecualikan seluruh pasangan canonical-pixel-identik agar tetangga tidak sekadar salinan sendiri. Cosine tinggi dapat berasal dari latar/warna/objek umum; bukan probabilitas kategori benar.''')
code("embedding_retrieval(ctx, profile, cnn_features, clusters)\ndisplay(pd.read_csv(ctx['report']/'test_coverage_review.csv',dtype={'nearest_train_id':str}).head(20))")
md('''## 9. Diagnostik validasi dan batas klaim

Regresi logistik domain menjawab apakah gambar berasal dari train atau test. Lima fold memakai grup kandidat; scaler dan classifier fit di train fold saja. Metadata serta CNN diuji terpisah; OOF AUC domain bukan metrik kategori limbah.

Draft fold kategori hanya rancangan yang menjaga grup dan distribusi cluster. Karena target cluster dibangun dari seluruh train, agreement terhadapnya tidak boleh disebut Accuracy semantik independen. Belum ada label resmi untuk confusion matrix kategori. Test tidak digunakan untuk fitting PCA/KMeans primary.''')
code("validation_diagnostics(ctx, profile, cnn_features, clusters, groups)\ndisplay(pd.read_csv(ctx['report']/'domain_validation.csv'))\ndisplay(pd.read_csv(ctx['work']/'output/folds'/VERSION/'DRAFT_group_folds.csv',dtype={'image_id':str}).groupby(['fold','cluster_id']).size().unstack(fill_value=0))")
md('''## 10. Bukti visual dan interpretasi

Grafik berikut dihitung dari artefak run yang sama. Galeri lengkap menyediakan semua gambar untuk penelaahan tim. Sebelum/sesudah preprocessing membandingkan full frame, letterbox yang dipilih, stretch, dan center crop sebagai ilustrasi risiko. Pembanding tidak mengganti data aktif.''')
code("figure_files = generate_figures(ctx, profile, clusters)\ngallery_pages = generate_galleries(ctx, profile, clusters)\nprint('Figures:',len(figure_files),'Gallery pages:',len(gallery_pages))\nprint('Open:',ctx['report']/'galleries/index.html')")
for name,explanation in [
 ('01_geometry.png','Geometri dan padding menentukan berapa banyak area representasi diisi gambar. Padding besar dan upsampling perlu diperiksa sebelum memilih resolusi training.'),
 ('02_quality_indicators.png','Kurva menunjukkan sebaran penuh, bukan hanya mean. Ketajaman dan entropy dipengaruhi tekstur asli; jangan menyebut semua nilai rendah sebagai data rusak.'),
 ('03_distribution_shift.png','Urutkan prioritas dari besarnya perbedaan fitur dan contoh aktual. Statistik saja belum menjelaskan penyebab atau membuktikan solusi augmentasi tertentu.'),
 ('04_format_and_slices.png','Format, warna, dan geometri dapat menjadi petunjuk sumber data. Standarisasi RGB membantu input konsisten tanpa memaksakan grayscale.'),
 ('05_similarity.png','Kenaikan jumlah pasangan pada ambang longgar menunjukkan sensitivitas threshold. Kandidat perlu ditelaah dan tidak menjadi jalur penyalinan label.'),
 ('06_cluster_navigation.png','PCA dua dimensi hanya navigasi. Cluster dihitung dalam representasi berdimensi lebih tinggi, bukan dari plot ini.'),
 ('07_cluster_populations.png','Ukuran kelompok tidak sama dengan jumlah kelas yang benar. Perbedaan komposisi train-test memberi pertanyaan cakupan, bukan prior label yang dipaksakan.'),
 ('08_cluster_diagnostics.png','Stabilitas tinggi berarti metode konsisten terhadap seed, bukan kategori pasti benar. Nilai internal berbeda ruang fitur tidak bisa dijadikan perlombaan Accuracy.'),
 ('09_validation_and_stability.png','AUC domain mengukur keterbedaan train-test; agreement seed mengukur ketidakstabilan cluster. Keduanya bukan confidence klasifikasi material.'),
 ('10_preprocessing_comparison.png','Letterbox mempertahankan frame penuh. Stretch mengubah bentuk; crop memotong area. Dampak ke Accuracy baru dapat diuji pada tahap model berikutnya.')]:
    md('### '+name.split('_',1)[1].replace('.png','').replace('_',' ').title()+'\n\n'+explanation)
    code(f"display(DisplayImage(filename=str(ctx['report']/'figures/{name}')))")
md('''## 11. Analisis hasil aktual dan keputusan

Ringkasan berikut dibuat dari tabel run, sehingga angka, rujukan, keterbatasan, dan keputusan preprocessing konsisten dengan keluaran yang sedang diperiksa.''')
code("analysis_text = write_analysis(ctx, profile, clusters)\ndisplay(Markdown(analysis_text))")
md('''## 12. Pemeriksaan akhir

Sel akhir memeriksa ulang byte sumber, cakupan ID, grup exact, dan status kategori. Berhasil menjalankan tahap 1 berarti integritas dan artefak analisis tersedia; bukan sertifikasi label atau jaminan leaderboard. Native RGB dan cluster views diperlakukan immutable; sebagian views merupakan hardlink.''')
code("summary = finalize(ctx, profile, clusters, groups)\ndisplay(summary)\nprint('STAGE 1 COMPLETE — EDA and unsupervised groups, semantic mapping pending.')")
md('''## 13. Perluasan yang diuji: representasi dan clustering terlatih

Baseline di atas tetap menjadi kontrol. Tahap berikut membandingkan ResNet50 dan DINOv2 image-only dalam empat ruang: tanpa PCA, PCA64, PCA128, dan PCA256. Semua PCA/KMeans fit pada pixel-unik train; test hanya mendapat transform dan assignment. Tiga seed per konfigurasi mengukur kestabilan, bukan memilih label kategori.

Kode tahap 2 berada di `eksperimen/skrip/` dengan SHA-256 dicetak sebelum eksekusi. Modul dimuat dalam namespace tersendiri; argumen dan environment dipulihkan setelah pemanggilan agar kernel tidak tercampur. Tidak ada API inference atau target kategori yang dibaca.

Referensi metode: [DINOv2](https://arxiv.org/html/2304.07193v2) untuk fitur image-only dan [SCAN](https://arxiv.org/html/2005.12320v2) untuk konsistensi tetangga. Implementasi head di sini merupakan adaptasi terbatas, bukan reproduksi keseluruhan pipeline paper.''')
code('''STAGE2_REPORT = PROJECT_ROOT / 'reports/errors/stage2_20261105'
STAGE2_FEATURES = PROJECT_ROOT / 'output/embeddings/stage2_repr_v1'
stage2_runtime_records = []

def run_stage2_module(filename, arguments=(), environment=None):
    if not RUN_STAGE2:
        print('Dilewati oleh RUN_STAGE2=False:', filename)
        return None
    script = PROJECT_ROOT / 'eksperimen/skrip' / filename
    source_digest = hashlib.sha256(script.read_bytes()).hexdigest()
    print(f'Menjalankan {filename} | SHA-256 {source_digest}', flush=True)
    old_argv = sys.argv[:]
    changes = environment or {}
    old_environment = {key:os.environ.get(key) for key in changes}
    started = time.time()
    module_name = 'gelar_notebook_' + script.stem
    previous_module = sys.modules.get(module_name)
    try:
        sys.argv = [str(script), *map(str, arguments)]
        os.environ.update({key:str(value) for key,value in changes.items()})
        spec = importlib.util.spec_from_file_location(module_name, script)
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
        module.main()
    finally:
        sys.argv = old_argv
        for key,value in old_environment.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        if previous_module is None:
            sys.modules.pop(module_name, None)
        else:
            sys.modules[module_name] = previous_module
    record = dict(script=filename, source_sha256=source_digest,
                  seconds=round(time.time()-started, 3), status='completed')
    stage2_runtime_records.append(record)
    print(record, flush=True)
    return module

if RUN_STAGE2:
    STAGE2_REPORT.mkdir(parents=True, exist_ok=True)
    for name in ['stage2_visual_features.py','stage2_representation_ablation.py',
                 'stage2_neighbor_refinement.py','stage2_evaluate_views.py',
                 'materialize_stage2_clusters.py']:
        path = PROJECT_ROOT/'eksperimen/skrip'/name
        assert path.exists(), path
    print('Modul tahap 2 tersedia; fitting tetap train-only.')''')
md('''## 14. DINOv2 dan probe perubahan view

DINOv2 ViT-S/14 menghasilkan fitur CLS 384 dimensi. Checkpoint, revision dan hash dicatat; model tidak memakai encoder teks. Notebook memakai full-frame letterbox224 dan normalisasi yang sama dengan tahap 1. Resolusi224 ini ablation komputasi; bukan klaim mengikuti seluruh resep checkpoint default518.

Sebanyak256 train pixel-unik dipilih acak sebelum melihat hasil view, lalu diuji horizontal flip224 dan full-frame336. Probe ini membantu mengukur sensitivitas representasi; semua2400 gambar tetap mempunyai fitur224. `RECOMPUTE_DINO=True` menghitung ulang semua fitur dan probe. Bila diubah keFalse, log akan menyatakan penggunaan cache; hasil itu tidak disebut ekstraksi baru.''')
code('''if RUN_STAGE2:
    run_stage2_module('stage2_visual_features.py', environment={
        'GELAR_RECOMPUTE_FEATURES':'1' if RECOMPUTE_DINO else '0'})
    checkpoint_info = json.loads((STAGE2_FEATURES/'dino_checkpoint.json').read_text(encoding='utf-8'))
    feature_check = json.loads((STAGE2_FEATURES/'feature_validation.json').read_text(encoding='utf-8'))
    with np.load(STAGE2_FEATURES/'dinov2_features.npz', allow_pickle=False) as feature_archive:
        assert np.array_equal(feature_archive['keys'], profile.key.to_numpy(str))
        print('DINO features:', feature_archive['features'].shape)
    display(pd.DataFrame([checkpoint_info]))
    display(feature_check)''')
md('''## 15. Ablasi PCA, seed, dan sensitivitas kelompok train

Empat branch memakai konfigurasi KMeans identik: k8, seed11/29/47, n_init10. PCA seed11 berbeda dari seed PCA baseline tahap1; baseline asli tetap dipertahankan sebagai pembanding tersendiri. Retensi top10 tetangga mengecualikan semua pasangan pixel-identik. Retensi tanpa PCA bernilai1 menurut definisi, bukan bukti kualitas kategori.

Silhouette dibandingkan pada ruang asli yang sama untuk masing-masing backbone. Skor pada ruang PCA sendiri tidak boleh langsung dibandingkan lintas dimensi atau backbone. Refit setelah mengeluarkan20% kelompok train mengukur sensitivitas terhadap pengurangan data; referensi full-train pernah melihat kelompok itu sehingga hasilnya bukan OOF Accuracy.''')
code('''if RUN_STAGE2:
    run_stage2_module('stage2_representation_ablation.py', ['--prefix','resnet50'])
    run_stage2_module('stage2_representation_ablation.py', [
        '--features', STAGE2_FEATURES/'dinov2_features.npz', '--prefix','dinov2'])
    ablation_metrics = []
    ablation_stability = []
    for backbone in ['resnet50','dinov2']:
        folder = STAGE2_REPORT/'ablation'/backbone
        ablation_metrics.append(pd.read_csv(folder/'metrics_by_seed.csv').assign(backbone=backbone))
        ablation_stability.append(pd.read_csv(folder/'stability_summary.csv').assign(backbone=backbone))
    ablation_metrics = pd.concat(ablation_metrics, ignore_index=True)
    ablation_stability = pd.concat(ablation_stability, ignore_index=True)
    display(ablation_metrics.groupby(['backbone','branch'])[
        ['variance_retained','knn10_retention_train','silhouette_original',
         'neighbor_cohesion_original_train','min_cluster','max_cluster']].mean())
    display(ablation_stability[['backbone','branch','seed_ari_min','seed_ari_mean',
        'group_removed_holdout_ari_vs_full','group_removed_test_ari_vs_full']])''')
code('''if RUN_STAGE2:
    import matplotlib.pyplot as plt
    branch_order = ['nopca','pca64','pca128','pca256']
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for backbone, color in [('resnet50','#277da8'),('dinov2','#e8871e')]:
        means = ablation_metrics[ablation_metrics.backbone.eq(backbone)].groupby('branch').mean(numeric_only=True).reindex(branch_order)
        stable = ablation_stability[ablation_stability.backbone.eq(backbone)].set_index('branch').reindex(branch_order)
        axes[0].plot(branch_order, means.knn10_retention_train, marker='o', color=color, label=backbone)
        axes[1].plot(branch_order, stable.seed_ari_min, marker='o', color=color, label=backbone)
        axes[2].plot(branch_order, means.silhouette_original, marker='o', color=color, label=backbone)
    axes[0].set_title('Retensi tetangga di ruang asli')
    axes[1].set_title('ARI minimum antarseed')
    axes[2].set_title('Silhouette: bandingkan dalam backbone')
    for ax in axes:
        ax.grid(alpha=.2); ax.legend()
    fig.suptitle('Diagnostik geometris; bukan Accuracy kategori')
    fig.tight_layout()
    figure_path = STAGE2_REPORT/'notebook_representation_ablation.png'
    fig.savefig(figure_path, dpi=150); plt.close(fig)
    display(DisplayImage(filename=str(figure_path)))''')
md('''## 16. Melatih head konsistensi tetangga pada train

Head linear belajar pada fitur DINOv2 frozen: prediksi gambar dan10 tetangga train didorong konsisten, dengan entropy marginal untuk mencegah satu cluster mendominasi. Inisialisasi memakai centroid KMeans. Entropy membawa asumsi keseimbangan; distribusi kategori sebenarnya belum diketahui.

Enam kandidat memakai seed11/29/47 dan entropy weight0,5/2,0, maksimal120 epoch. Model dipilih melalui loss held-out train dengan bobot entropy pembanding tetap2, serta delapan kelompok tetap terisi. Graph held-out hanya menunjuk fit-train. Test tidak ikut fitting atau pemilihan. Setelah dipilih, head direfit pada semua train pixel-unik.

Ini adaptasi loss SCAN dengan full-batch graph dan backbone frozen. Tidak ada image augmentation saat training head atau self-labeling lanjutan. Konsistensi tetangga dan loss bukan Accuracy material.''')
code('''if RUN_STAGE2:
    run_stage2_module('stage2_neighbor_refinement.py')
    head_report = STAGE2_REPORT/'neighbor_refinement'
    head_result = json.loads((head_report/'result.json').read_text(encoding='utf-8'))
    assert head_result['status'] == 'complete' and head_result['refit_no_collapsed_cluster']
    candidate_scores = pd.read_csv(head_report/'candidate_validation.csv')
    learning_curves = pd.read_csv(head_report/'learning_curves.csv')
    display(candidate_scores)
    display(head_result)
    chosen_curve = learning_curves[
        learning_curves.seed.eq(head_result['selected_seed']) &
        learning_curves.entropy_weight.eq(head_result['entropy_weight'])]
    display(chosen_curve[chosen_curve.epoch.isin([0, head_result['epochs']])][[
        'epoch','train_hard_neighbor_agreement','validation_hard_neighbor_agreement',
        'validation_selection_score','validation_occupied_clusters']])''')
code('''if RUN_STAGE2:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for (seed, weight), curve in learning_curves.groupby(['seed','entropy_weight']):
        label = f'seed{seed}, entropy={weight}'
        axes[0].plot(curve.epoch, curve.validation_selection_score, label=label)
        axes[1].plot(curve.epoch, curve.validation_hard_neighbor_agreement, label=label)
    axes[0].set_title('Loss selection held-out (lebih kecil)')
    axes[1].set_title('Agreement tetangga held-out (bukan Accuracy)')
    for ax in axes:
        ax.set_xlabel('Epoch'); ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    figure_path = STAGE2_REPORT/'notebook_head_learning_curves.png'
    fig.savefig(figure_path, dpi=150); plt.close(fig)
    display(DisplayImage(filename=str(figure_path)))''')
md('''## 17. Kestabilan view dan tetangga test terbaru

Setiap model yang telah dibekukan menginferensi kembali256 probe train pada flip dan resolusi336. Konsistensi yang tinggi berarti assignment tidak berubah, tetapi assignment yang stabil masih dapat tidak sesuai kategori. Perubahan view pun dapat lebih baik atau lebih buruk; dataset ini belum menyediakan target kategori resmi untuk membedakannya.

Seluruh480test juga mendapat lima tetangga train DINOv2 dengan exact pixel duplicates dikeluarkan. Retrieval dipakai untuk pemeriksaan struktur dan cakupan, bukan menyalin kategori atau memilih jawaban test.''')
code('''if RUN_STAGE2:
    run_stage2_module('stage2_evaluate_views.py')
    view_metrics = pd.read_csv(STAGE2_REPORT/'view_robustness.csv')
    display(view_metrics[['method','view','n','unchanged_count','unchanged_fraction','ari']])
    display(DisplayImage(filename=str(STAGE2_REPORT/'view_consistency.png')))
    display(pd.read_csv(STAGE2_REPORT/'shared_space_silhouette.csv').pivot(
        index='method',columns='reference_space',values='silhouette'))
    test_neighbors = pd.read_csv(STAGE2_REPORT/'dinov2_test_train_top5.csv')
    assert test_neighbors.test_key.nunique() == 480
    display(test_neighbors.head(15))
    print('Galeri seluruh test:', STAGE2_REPORT/'galleries/dinov2_test_neighbors.html')''')
md('''## 18. Membentuk folder seluruh train dan test

Assignment head yang dibekukan diterapkan pada semua gambar. PNG native hasil preprocessing disalin secara independen ke `data/processed/stage2_neighbor_v1/{train,test}/cluster_00..07`. Seluruh hash salinan diverifikasi; tidak ada gambar yang hilang karena confidence rendah.

Nomor cluster tahap2 tidak sama artinya dengan nomor cluster tahap1. Template penamaan kategori tersedia untuk tim pada tingkat cluster. Folder hasil ini adalah kandidat pengelompokan algoritmis, belum label material yang terverifikasi.''')
code('''if RUN_STAGE2:
    run_stage2_module('materialize_stage2_clusters.py')
    folder_root = PROJECT_ROOT/'data/processed/stage2_neighbor_v1'
    cluster_manifest = pd.read_csv(folder_root/'cluster_manifest.csv', dtype={'image_id':str})
    folder_counts = pd.read_csv(folder_root/'folder_counts.csv')
    display(folder_counts)
    display(cluster_manifest[['key','cluster_id','folder_path','png_sha256']].head(12))
    print('Train:', folder_root/'train')
    print('Test:', folder_root/'test')
    print('Galeri lengkap:', STAGE2_REPORT/'galleries/neighbor_refinement_clusters.html')''')
md('''## 19. Contoh setiap cluster: train dan test

Untuk setiap cluster ditampilkan dua skor head tertinggi dan dua terendah. Pemilihan ini mekanis untuk memperlihatkan variasi dan batas kelompok, bukan anotasi kategori. Skor head belum terkalibrasi; angkanya tidak boleh dibaca sebagai peluang kategori benar.

Gambar berikut mempertahankan full-frame. Seluruh anggota, bukan hanya contoh, tersedia di galeri dan folder. Saat menafsirkan, periksa ukuran objek, kilauan, pola latar, teks, occlusion, dan banyaknya objek dalam frame.''')
code('''if RUN_STAGE2:
    from PIL import Image as PILImage, ImageDraw, ImageFont, ImageOps
    review_frame = profile[['key','preview_path']].merge(cluster_manifest, on='key', validate='one_to_one')
    try:
        contact_font = ImageFont.truetype('arial.ttf', 17)
    except OSError:
        contact_font = ImageFont.load_default()
    for split in ['train','test']:
        for page in range(2):
            sheet = PILImage.new('RGB', (960, 1130), 'white')
            draw = ImageDraw.Draw(sheet)
            draw.text((10,8), f'{split.upper()} - cluster {page*4:02d} sampai {page*4+3:02d}; contoh mekanis', fill='black', font=contact_font)
            for row_slot, cluster in enumerate(range(page*4, page*4+4)):
                subset = review_frame[review_frame.split.eq(split) & review_frame.cluster_index.eq(cluster)].sort_values('uncalibrated_head_probability', ascending=False)
                examples = pd.concat([subset.head(2),subset.tail(2)]).drop_duplicates('key')
                for col_slot, image_row in enumerate(examples.itertuples()):
                    x0 = col_slot*240+5; y0 = row_slot*270+42
                    with PILImage.open(PROJECT_ROOT/image_row.preview_path) as image_source:
                        tile = ImageOps.contain(image_source.convert('RGB'), (228,228))
                    sheet.paste(tile, (x0+(228-tile.width)//2,y0+(228-tile.height)//2))
                    draw.text((x0,y0+230),f'{image_row.key}  cluster_{cluster:02d}',fill='black',font=contact_font)
                    draw.text((x0,y0+250),f'head={image_row.uncalibrated_head_probability:.3f}',fill='black',font=contact_font)
            target = STAGE2_REPORT/f'notebook_{split}_cluster_examples_{page+1}.png'
            sheet.save(target)
            display(DisplayImage(filename=str(target)))''')
md('''## 20. Verifikasi hasil run gabungan dan keputusan

Sel akhir menguji cakupan ID, pemisahan kelompok fit/held-out, assignment pixel-identik, kesesuaian manifest folder, serta reproduksi probabilitas dari checkpoint head. Hasil ini menguji integritas dan eksekusi; tidak mengesahkan label kategori atau membuktikan skor leaderboard.

Keputusan saat ini: mempertahankan native RGB dan full-frame, membandingkan representasi secara terkontrol, serta memakai head terlatih sebagai kandidat pengelompokan dengan KMeans tetap sebagai kontrol. Pengembangan berikutnya dapat menguji fine-tuning image-only dan konsistensi multi-view pada train resmi. Pemilihan final membutuhkan bukti kategori yang sah; confidence atau silhouette saja tidak cukup.''')
code('''notebook_checks = dict(stage1_complete=True, stage2_enabled=RUN_STAGE2,
                       semantic_accuracy=None, submission_created=False)
if RUN_STAGE2:
    import torch
    assignment_path = PROJECT_ROOT/'output/groups/stage2_neighbor_v1/assignments.csv'
    head_assignments = pd.read_csv(assignment_path, dtype={'image_id':str})
    assert head_assignments.key.is_unique and set(head_assignments.key) == set(profile.key)
    assert head_assignments.groupby('split').size().to_dict() == {'test':480,'train':1920}
    assert cluster_manifest.key.is_unique and set(cluster_manifest.key) == set(profile.key)
    aligned_manifest = cluster_manifest.set_index('key').loc[head_assignments.key]
    assert np.array_equal(aligned_manifest.cluster_index, head_assignments.cluster_index)
    exact = profile[['key','pixel_sha256']].merge(head_assignments[['key','cluster_index']],on='key',validate='one_to_one')
    assert exact.groupby('pixel_sha256').cluster_index.nunique().max() == 1
    split_table = pd.read_csv(STAGE2_REPORT/'neighbor_refinement/group_split.csv')
    fit_groups = set(split_table.loc[split_table.role.eq('fit_train'),'candidate_group'])
    holdout_groups = set(split_table.loc[split_table.role.eq('heldout_train'),'candidate_group'])
    assert not fit_groups & holdout_groups
    assert split_table.split.eq('train').all()
    checkpoint = torch.load(PROJECT_ROOT/'output/groups/stage2_neighbor_v1/head.pt',map_location='cpu',weights_only=True)
    trained_head = torch.nn.Linear(checkpoint['input_dimension'],8)
    trained_head.load_state_dict(checkpoint['state_dict']); trained_head.eval()
    with np.load(STAGE2_FEATURES/'dinov2_features.npz',allow_pickle=False) as feature_archive:
        assert np.array_equal(feature_archive['keys'],head_assignments.key.to_numpy(str))
        with torch.inference_mode():
            reproduced = trained_head(torch.from_numpy(feature_archive['features'])).softmax(-1).numpy()
    with np.load(PROJECT_ROOT/'output/groups/stage2_neighbor_v1/predictions.npz',allow_pickle=False) as saved_prediction:
        assert np.array_equal(saved_prediction['keys'],head_assignments.key.to_numpy(str))
        assert np.allclose(reproduced,saved_prediction['probabilities'],atol=1e-6)
    assert np.allclose(reproduced.sum(1),1,atol=1e-5) and np.isfinite(reproduced).all()
    assert np.array_equal(reproduced.argmax(1),head_assignments.cluster_index)
    for split,expected in [('train',1920),('test',480)]:
        assert len(list((folder_root/split).glob('*/*.png'))) == expected
        assert len([p for p in (folder_root/split).iterdir() if p.is_dir()]) == 8
    notebook_checks.update(stage2_complete=True, train_images=1920, test_images=480,
        head_probability_rows_verified=2400, group_overlap=0,
        pixel_identical_assignments_consistent=True, category_mapping='pending_team_mapping',
        assignment_sha256=hashlib.sha256(assignment_path.read_bytes()).hexdigest(),
        stage2_runtime_records=stage2_runtime_records)
    display(pd.DataFrame(stage2_runtime_records))
notebook_checks['elapsed_seconds'] = round(time.time()-NOTEBOOK_STARTED,3)
notebook_check_path = PROJECT_ROOT/'reports/reproducibility/notebook_v2_in_kernel_checks.json'
notebook_check_path.parent.mkdir(parents=True,exist_ok=True)
notebook_check_path.write_text(json.dumps(notebook_checks,indent=2),encoding='utf-8')
display(notebook_checks)
print('NOTEBOOK COMPUTATION COMPLETE - outputs tersimpan, Accuracy kategori belum diukur.')''')
nb.cells=cells
for index, cell in enumerate(nb.cells):
    cell['id'] = hashlib.sha256(f'{index}:{cell.cell_type}:{cell.source}'.encode('utf-8')).hexdigest()[:12]
path=root/'notebook/01_EDA_Preprocessing_Clustering.ipynb'
nbformat.validate(nb)
nbformat.write(nb,path)
print(path, 'cells',len(cells),'code',sum(c.cell_type=='code' for c in cells))

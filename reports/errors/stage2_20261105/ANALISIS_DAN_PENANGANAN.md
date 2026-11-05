# Analisis perbedaan dan penanganan citra — 5 November 2026

**Status dokumen:** riwayat baseline tahap 2. Notebook utama sekarang EDA visual v3; hasil terbaru ada di `reports/eda/stage3_visual_20261106/ANALISIS_VISUAL.md` dan bukti `reports/reproducibility/notebook_v3_delivery_validation.json`.

## Hasil utama

Masalah yang perlu ditangani adalah kesesuaian kelompok visual dengan kategori material. Mengulang seed, memperbesar PCA, atau menghapus gambar buram belum cukup. Eksperimen baru menghasilkan kandidat **DINOv2 + head konsistensi tetangga** yang lebih tahan terhadap perubahan resolusi pada sampel train. Ini bukti perbaikan kestabilan tertentu, bukan bukti Accuracy kategori mendekati 100%.

Seluruh **1.920 train dan 480 test** sekarang juga tersedia dalam delapan folder algoritmis baru di [stage2_neighbor_v1](../../../data/processed/stage2_neighbor_v1/README.md). Model mengelompokkan train dan menginferensi test; tidak ada pemindahan anggota per gambar. Nomor cluster baru tidak mewakili nomor atau nama cluster tahap 1.

## Cakupan pemeriksaan

- Audit program mencakup seluruh 2.400 gambar, termasuk 590 perbedaan train dan 171 perbedaan test terhadap arsip. Perbedaan dengan arsip bukan jumlah kesalahan yang diketahui.
- Pemeriksaan visual teknis dilakukan pada lima contact sheet: 46 tampilan awal dan 32 tampilan tambahan khusus train; beberapa ID berulang. Tidak diklaim bahwa semua 761 kasus ditinjau visual satu per satu pada sesi ini. Galeri mencakup seluruh kasus untuk peninjauan lanjut.
- Tiga sheet awal memisahkan margin rendah, margin relatif tinggi, dan kontrol. Dua sheet tambahan mengurutkan kasus train berdasarkan ketidaksepakatan algoritma, konsistensi tetangga, dan perubahan view.
- Arsip anotasi hanya digunakan dalam audit terpisah. Fitting, pemilihan hyperparameter, head, template kategori, dan folder baru tidak membaca kategori arsip. Skor public 0,9930 merupakan bukti screenshot pengguna atas submission lama; private dan identitas kesalahan tidak diketahui.

## Mengapa train perlu diperbaiki pada tingkat metode

| Diagnosis tahap 1 | Train | Test |
|---|---:|---:|
| Berbeda dari arsip | 590/1.920 | 171/480 |
| Berbeda, margin >=0,10, seluruh seed sepakat | 333 | 102 |
| Berbeda, seed agreement <0,80 | 11 | 1 |
| Perbedaan dalam cluster lama 03 dan 06 | 335 | 129 |

Artinya, fokus hanya pada confidence rendah akan melewatkan banyak kasus. Cluster lama 03 sendiri memuat delapan kategori arsip; dua komposisi terbesarnya hampir seimbang. Mengganti satu nama untuk seluruh cluster itu tidak menyelesaikan pencampuran. Angka ini diagnosis retrospektif, bukan sumber label baru.

Blur juga bukan penjelasan tunggal: median Laplacian variance train berbeda/sesuai adalah 835,80/741,58, sedangkan test 561,89/880,28. Arah asosiasi berbalik antar-split. Pada test cluster lama 06, median padding berbeda/sesuai 0/0,25 dan brightness 0,826/0,682. Komposisi dan gaya gambar perlu diperiksa bersama; asosiasi ini tidak membuktikan sebab.

### Temuan visual teknis pada train

| Contoh ID | Yang benar-benar tampak | Penanganan yang layak diuji |
|---|---|---|
| 0699 | Pengambilan sangat dekat, tulisan besar, sebagian bentuk terpotong | Pertahankan full-frame; crop tambahan dapat menghilangkan konteks |
| 1154 | Latar berpola garis kuat di belakang objek | Uji konsistensi fitur antar-view dan ketergantungan latar |
| 1896, 1605 | Objek relatif kecil dengan ruang latar luas; 1605 berlatar garis | Resolusi lebih tinggi dan pooling patch sebagai pembanding, bukan crop manual |
| 1084 | Komposisi seperti poster dengan teks, tangan, dan beberapa elemen | Periksa pengaruh teks/latar; jangan menganggap semua foto objek tunggal |
| 1302, 0420 | Detail terlihat kasar/buram dan objek bertumpuk | Jangan membuang massal; ukur stabilitas pada view yang mempertahankan cakupan |
| 1048 | Tangan menutupi sebagian objek dan kontras tepi rendah | Hindari segmentasi/background removal otomatis tanpa audit kegagalan |
| 1388 | Kilauan dan pembungkus mendominasi tampilan permukaan | Pertahankan warna/tekstur; jangan memakai grayscale atau sharpening secara universal |
| 0504 | Beberapa unsur kecil di lingkungan luar yang kompleks | Global embedding saja mungkin kurang; bandingkan representasi patch dan full-frame |

Tabel ini tidak menetapkan atau mengoreksi kategori gambar. Observasi teknis tersebut mendukung eksperimen terkontrol, bukan alasan mengganti label satu per satu.

## Eksperimen yang benar-benar dijalankan

### 1. Ablasi PCA dan representasi

Dua backbone: ResNet50 V2 tahap 1 dan **DINOv2 ViT-S/14 image-only**. Setiap backbone diuji tanpa PCA, PCA64, PCA128, PCA256; KMeans8 dengan seed 11/29/47 dan n_init=10. PCA dan cluster hanya fit pada 1.919 pixel-unik train. Test hanya ditransform dan diassign. Seluruh branch disimpan, tidak dipilih memakai arsip.

| Backbone/branch | Varian dipertahankan | Tetangga top10 asli dipertahankan, train | ARI minimum antar-seed |
|---|---:|---:|---:|
| ResNet tanpa PCA | 100% | 100% menurut definisi | 0,9418 |
| ResNet PCA64 | 58,35% | 72,97% | 0,8776 |
| ResNet PCA128 | 70,43% | 80,79% | 0,9444 |
| ResNet PCA256 | 82,30% | 86,11% | 0,9591 |
| DINOv2 tanpa PCA | 100% | 100% menurut definisi | 0,7045 |
| DINOv2 PCA64 | 69,79% | 79,58% | 0,8736 |
| DINOv2 PCA128 | 84,68% | 87,51% | 0,9640 |
| DINOv2 PCA256 | 96,51% | 91,83% | 0,9345 |

PCA seed ablasi adalah 11; tahap 1 memakai PCA seed 20261105. Karena itu hasil PCA64 ablasi bukan identik dengan partisi tahap 1. Baseline tahap 1 juga direproduksi persis secara terpisah dalam pengujian view.

Silhouette ResNet dalam ruang asli yang sama hampir tidak berubah: mean sekitar 0,0709–0,0714. Jadi menyelamatkan lebih banyak varian belum menunjukkan pemisahan kategori lebih baik. Silhouette antar-backbone tidak boleh dibandingkan langsung. Tabel `shared_space_silhouette.csv` menunjukkan masing-masing partisi cenderung lebih baik pada ruang representasinya sendiri; belum ada pemenang semantik.

Refit setelah mengeluarkan 20% kelompok train juga diuji. Perbandingan terhadap partisi full-train adalah sensitivitas pengelompokan, **bukan OOF Accuracy**; referensi full-train pernah melihat gambar yang dikeluarkan.

DINOv2 memakai checkpoint pinned `4610ca143709d58a633b6397a74412c2c3842454`, SHA-256 `04d27f3400d059fc0cfd7d17dd1909a75bf3ea8fb3eeb48b97cb99e57ee20081`. Fitur CLS 384 dimensi diekstrak lokal/CPU. Input224 merupakan eksperimen hemat komputasi dengan letterbox, bukan klaim mengikuti seluruh resep default checkpoint518. Backbone belum di-fine-tune pada kompetisi.

### 2. Head konsistensi tetangga yang dilatih pada train

Implementasi terinspirasi persamaan SCAN: minimisasi `mean[-log(dot(p_i,p_neighbor))] - lambda*entropy(mean(p))`. Graph berisi 10 tetangga train; semua pixel-identik dikeluarkan dari pencarian tetangga. Fitur backbone frozen. Head linear diinisialisasi dari centroid KMeans, lalu dilatih dengan Adam. Ini adaptasi kecil, bukan reproduksi SCAN penuh: tidak ada augmentasi training atau tahap self-labeling paper.

Enam kandidat: seed 11/29/47 x entropy weight 0,5/2,0; maksimal120 epoch, evaluasi setiap10 epoch. Split 80/20 memakai kandidat kelompok duplikat; graph validation hanya menunjuk fit-train. Pemilihan memakai loss validation tanpa label dengan koefisien entropy pembanding tetap2, serta syarat delapan cluster terisi. Tidak ada target kategori, arsip, atau test dalam pemilihan.

Kandidat terpilih secara kriteria tersebut: **seed47, weight2, epoch60**. Konsistensi tetangga keras pada held-out train naik **86,82% -> 90,03%** dibanding inisialisasinya. Head kemudian direfit pada semua 1.919 train unik. Distribusi delapan cluster tetap terisi. Angka90,03% adalah agreement internal graph, bukan persentase kategori benar. Entropy juga membawa bias distribusi mendekati seimbang; ukuran kelas sebenarnya belum diketahui.

### 3. Uji view pada 256 train acak yang ditetapkan sebelumnya

| Metode | Cluster tetap setelah horizontal flip | Cluster tetap saat224 ->336 |
|---|---:|---:|
| Pipeline tahap 1 asli | 252/256 (98,44%) | 241/256 (94,14%) |
| DINOv2 PCA128 KMeans | 253/256 (98,83%) | 244/256 (95,31%) |
| DINOv2 + head tetangga | 252/256 (98,44%) | 253/256 (98,83%) |

Head memperbaiki satu ukuran kestabilan terhadap resolusi, tetapi tidak meningkatkan semua ukuran. Model cluster tetap dilatih pada224;336 hanya view diagnostik. Pergantian cluster dapat benar atau salah. Sampel berasal dari train, bukan set kategori berlabel yang independen. **98,83% ini bukan Accuracy.**

## Penanganan 590 kasus train sekarang

Assignment baru dibekukan sebelum menyambung kembali daftar perbedaan lama:

- 283/590 berpindah kelompok setelah nomor cluster lama-baru disejajarkan tanpa nama kategori. Ini bukan283 koreksi yang terbukti benar.
- 236/590 masih memenuhi sedikitnya satu flag struktural: algoritma pembanding berbeda, kurang dari70% tetangga mendukung cluster, probabilitas head <0,8, atau view berubah.
- 167/590 berbeda dari partisi pembanding DINO PCA128;103/590 memiliki dukungan tetangga <0,7. Irisan flag bertumpang tindih.
- 72 dari590 masuk probe view;5 berubah pada sedikitnya satu view. Kasus yang tidak diuji view tidak dihitung sebagai stabil.
- Pada seluruh train,373/1.920 ditandai. Seluruh480test tetap tercakup;97test ditandai. Gambar tidak dihapus karena flag.

Antrean ini menentukan wilayah metode yang perlu diperiksa. Ia tidak menghasilkan kategori baru dari anotasi lama. Semua data train tetap di folder; training lanjutan dapat membandingkan pembobotan konsistensi dengan bobot seragam, dengan cakupan dan efek per-cluster tercatat.

## Jalan lanjut yang paling beralasan

1. **Gunakan hasil head sebagai kandidat pengelompokan, pertahankan KMeans sebagai kontrol.** Lanjutkan pengujian konsistensi pada group holdout yang baru; jangan memilih hanya berdasarkan silhouette atau confidence.
2. **Uji fine-tuning image-only dengan pasangan full-frame yang mempertahankan objek**, termasuk variasi resolusi. Semua pasangan/training berasal dari train resmi. Kegagalan pada objek kecil dan latar sibuk menjadi hipotesis, bukan target per-gambar.
3. **Uji auxiliary overclustering pada training** sebagai pembanding agar variasi dalam satu kategori tidak dipaksa menjadi satu pola visual. Head akhir tetap8. Jangan langsung menganggap setiap subcluster punya kategori tertentu; belum diterapkan dalam run ini.
4. **Penamaan dilakukan untuk seluruh cluster oleh tim dengan provenance.** Bila satu cluster campur, perbaiki algoritma dan ulangi assignment secara menyeluruh. Jangan memperbaiki anggota satu per satu dari arsip.
5. **Validasi kategori harus dipisahkan dari validasi algoritma.** Kestabilan, neighbor agreement, dan kecocokan target turunan menguji konsistensi. Untuk menilai Accuracy kategori, diperlukan dasar penilaian yang sah menurut aturan. Public30% tidak mengungkap jawaban setiap gambar atau menjamin private70%.

Ini urutan eksperimen yang didukung temuan saat ini. Belum ada bukti untuk menyebut satu metode sebagai clustering material terbaik atau menjanjikan skor1,0000. Model baru tidak otomatis dipromosikan menjadi dataset kategori/submission aktif.

## Riset primer yang digunakan

- [DINOv2](https://arxiv.org/html/2304.07193v2): dibaca pengantar, bagian fitur image-only, resolusi dan evaluasi frozen/linear; mendukung percobaan backbone visual. Skor benchmark paper tidak ditransfer ke lomba. [Kartu checkpoint timm](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) diperiksa untuk dimensi, normalisasi, resolusi dan lisensi.
- [SCAN, ECCV2020](https://arxiv.org/html/2005.12320v2): bagian2.1–2.3 dan loss resmi diperiksa. Implementasi lokal memakai ide konsistensi-neighbor dan entropy, dengan adaptasi yang dijelaskan di atas; tidak menyalin pipeline repo atau mengklaim reproduksi utuh.
- [IIC, ICCV2019](https://arxiv.org/html/1807.06653v4): bagian metode dan auxiliary overclustering dibaca sebagai arah lanjutan. Head bantu mempelajari variasi yang lebih rinci; head utama tetap untuk jumlah kategori akhir. Belum dijalankan.
- [Backgrounds Challenge](https://arxiv.org/abs/2006.09994): abstrak/dokumentasi mendukung audit ketergantungan latar. Penyebab kesalahan pada dataset ini tetap hipotesis, bukan sudah dibuktikan oleh paper tersebut.
- [DECMCV, preprint2025](https://arxiv.org/pdf/2503.02241): metode dan tabel evaluasi dibaca. Angka93,78% TrashNet disertai pembuangan37% data. Ide konsensus berguna untuk diagnostik; angka tersebut bukan bukti accuracy seluruh test atau izin menghilangkan kasus sulit.

Akses sumber:5November2026. Tidak ada dataset eksternal, VLM, atau layanan inference pihak ketiga digunakan dalam run. Bantuan kode mengikuti klarifikasi pengguna; bukan kutipan izin baru panitia.

## Artefak dan reproduksi

- [Folder train/test](../../../data/processed/stage2_neighbor_v1/README.md), [jumlah tiap folder](../../../data/processed/stage2_neighbor_v1/folder_counts.csv), [manifest](../../../data/processed/stage2_neighbor_v1/cluster_manifest.csv).
- [Galeri seluruh train/test hasil head](galleries/neighbor_refinement_clusters.html), [galeri pembanding DINO PCA128](galleries/dinov2_pca128_clusters.html), [lima tetangga untuk seluruh480test](galleries/dinov2_test_neighbors.html).
- [Audit perbedaan lama](diagnostics/AUDIT_POSTHOC.md), [galeri semua761perbedaan](diagnostics/all_disagreements.html), [tindak lanjut per-ID](diagnostics/old_disagreements_structural_followup.csv).
- [Antrean struktural seluruh data](algorithmic_review_queue_all.csv), [hasil view](view_robustness.csv), [validasi akhir](final_validation.json).

Jalankan dari root proyek, berurutan:

```powershell
python eksperimen/skrip/stage2_visual_features.py
python eksperimen/skrip/stage2_representation_ablation.py --prefix resnet50
python eksperimen/skrip/stage2_representation_ablation.py --features output/embeddings/stage2_repr_v1/dinov2_features.npz --prefix dinov2
python eksperimen/skrip/stage2_neighbor_refinement.py
python eksperimen/skrip/stage2_evaluate_views.py
python eksperimen/skrip/stage2_training_review_queue.py
python eksperimen/skrip/materialize_stage2_clusters.py
python eksperimen/skrip/validate_stage2.py
```

`audit_stage1_disagreements.py` menghasilkan audit arsip terpisah yang diperlukan untuk skrip tindak lanjut. Audit tersebut bukan input ke tiga skrip fitting pertama. Snapshot notebook/output sebelum perluasan disimpan pada `reports/reproducibility/backups/20261105_before_notebook_v2/`. Notebook utama kini menggabungkan tahap 1+2 dan sudah lulus Run All 35/35 sel kode, 0 error, 1.264,14 detik dari `.venv`. Bukti terbaru: `reports/reproducibility/notebook_v2_delivery_validation.json`, termasuk 2.400 hash raw terhadap backup dan enam eksekusi modul. `final_validation.json` di folder laporan ini tetap merupakan bukti run awal dengan hash artefak saat itu; gunakan bukti baru untuk notebook gabungan. Audit arsip tetap terpisah dari notebook.

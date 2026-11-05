# Tahap 1 — EDA, preprocessing, kemiripan dan clustering

Status: pipeline dijalankan pada seluruh sumber. Ini analisis tanpa target kategori resmi, bukan evaluasi Accuracy classifier.

## 1. Pertanyaan dan cakupan

Apakah data lengkap dan dapat dibaca? Bagaimana train berbeda dari test? Apakah ada citra identik atau hampir sama? Seberapa stabil kelompok yang muncul tanpa label? Transform apa yang dapat dipertanggungjawabkan?

Train 1,920; test 480; gagal decode 0. ID test mengikuti sample. Kolom kategori contoh tidak digunakan. Direktori tambahan di bawah train tidak dihitung sebagai sampel.

## 2. Integritas dan representasi

2,400 citra telah didekode, dibuat RGB PNG native, dan diperiksa kesamaan piksel hasil simpan/baca. SHA-256 sumber diperiksa sebelum dan sesudah. EXIF diterapkan satu kali; alpha dikomposit putih. Tidak ada penghapusan citra atau koreksi kategori.

ICC dicatat sebagai metadata; belum dilakukan konversi profil warna ICC. Gamma/warna JPEG mengikuti decoder Pillow. Frame tambahan GIF/TIFF dicatat; pipeline ini memakai frame pertama. Lihat metadata sebelum memutuskan penanganan khusus.

## 3. Geometri, kualitas, dan preprocessing

- test: median ukuran 416×416; min-side<224 25/480; padding median 0.0%; grayscale efektif 13/480.
- train: median ukuran 509×486; min-side<224 119/1920; padding median 0.0%; grayscale efektif 19/1920.

Native RGB dipertahankan agar resolusi model berikutnya masih dapat dipilih. Letterbox 224 hanya representasi eksplorasi backbone: seluruh frame dipertahankan, dengan padding warna rerata ImageNet. Resize meregangkan geometri, sedangkan center crop dapat menghilangkan tepi; keduanya hanya pembanding visual, bukan perubahan data aktif.

Blur, entropy, exposure, dan edge density adalah indikator: citra polos memang dapat memiliki entropy/ketajaman rendah. Ekor distribusi tidak otomatis berarti gambar rusak. Ambang persentil dipelajari dari train untuk antrean tinjauan, bukan seleksi/hapus data.

## 4. Pergeseran train–test

- megapixels: KS=0.1693, JS distance=0.2882, median train/test=0.2136/0.1731, BH-q=7e-09.
- height: KS=0.1599, JS distance=0.2116, median train/test=485.5/416, BH-q=3.85e-08.
- min_side: KS=0.1172, JS distance=0.2882, median train/test=416/416, BH-q=0.000246.
- bright_fraction: KS=0.1005, JS distance=0.1194, median train/test=0.002563/0.005814, BH-q=0.00286.
- width: KS=0.0995, JS distance=0.2514, median train/test=509/416, BH-q=0.00286.

KS di sini jarak distribusi fitur, bukan metrik target kompetisi. P-value dikoreksi BH untuk banyak pengujian; pengulangan/kemiripan gambar tetap dapat melanggar asumsi independen. Effect size dan grafik lebih informatif daripada label signifikan saja. Bin JS memakai kuantil train dengan tail bins agar nilai test di luar rentang tidak hilang.

- Domain classifier metadata: grouped OOF AUC=0.6006. Target hanya asal split train/test; bukan kategori limbah atau Accuracy kompetisi.
- Domain classifier cnn_fixed_features: grouped OOF AUC=0.6578. Target hanya asal split train/test; bukan kategori limbah atau Accuracy kompetisi.

StandardScaler dan regresi logistik domain fit di tiap train fold. Grup kandidat pHash tidak menyeberang fold. CNN sebagai fitur tetap tidak dilatih dengan domain target. Hasil ini mendeteksi keterbedaan, tidak menunjukkan penyebab kausal atau otomatis membenarkan reweighting.

## 5. Duplikasi dan kemiripan

Pasangan byte-identik: 4; pixel-identik setelah canonical RGB: 4. Kandidat pHash≤6: 108. Grup kandidat dengan risiko rantai/diameter>4: 0. Tidak semua kandidat adalah foto/objek yang sama.

Dari 480 test, 3 memiliki train dengan pixel identik; 32 memiliki kandidat pHash≤6. Tidak ada penyalinan label maupun penggantian prediksi berdasarkan tetangga.

CNN nearest-train cosine median=0.6744. pHash menangkap struktur visual kasar; CNN dapat dekat karena bentuk, material, objek, maupun latar. Jarak/cosine bukan probabilitas benar. Seluruh 480 test tersedia dalam galeri tetangga.

## 6. Clustering dan stabilitas

Primary telah ditetapkan sebelum perbandingan arsip: ResNet50 ImageNet1K V2, full-frame letterbox, L2 features, PCA64 fit pada train unik, renormalisasi, KMeans k=8 seed11 n_init10. Hanya satu wakil pixel-identik masuk fitting PCA/cluster. Test diproyeksikan dan diassign ke centroid train tanpa fit ulang.

Pembanding: lima seed CNN, dua seed fitur HSV+HOG, Ward8, serta k=6/10/12/16 sebagai sensitivitas eksplorasi. Ini belum pemilihan kategori atau izin memperlakukan subcluster sebagai label akhir. Silhouette antar representasi tidak otomatis sebanding dan tidak menentukan kebenaran semantik.

Primary silhouette=0.1336; ukuran cluster train unik 169–302. ARI antar seed CNN: minimum 0.9755, median 0.9850.

Antrean tinjauan algoritmis: 612 gambar. Alasan: margin centroid<0,10, agreement seed<0,80, atau jarak di atas p95 cluster train. Ini indikator ketidakstabilan, bukan bukti salah label; tidak ada perbaikan per gambar otomatis/manual.

Folder cluster_00…cluster_07 berisi gambar hasil preprocessing. Nama glass/paper/dan seterusnya belum dipetakan; anggota tim perlu menamai cluster dengan dasar yang dicatat. Satu cluster tidak dijamin sama dengan satu kelas; pemetaan kosong lebih jujur daripada memaksakan delapan kategori.

## 7. Validasi yang dapat dan belum dapat disimpulkan

Draft 5-fold memakai group kandidat dan stratifikasi cluster anonim. Tidak ada grup kandidat menyeberang fold. Karena clustering fit pada seluruh train, agreement classifier terhadap target cluster global bukan validasi semantik independen. Pada tahap berikutnya, learned preprocessing/target construction perlu dibatasi ke train fold jika mengklaim out-of-sample; test Kaggle tetap tersembunyi.

Belum ada Accuracy kategori, confusion matrix kategori, atau classifier limbah yang dilatih/fine-tune. Backbone frozen hanya untuk eksplorasi. Jalur final pretrained memerlukan pelatihan/fine-tuning kompetisi sesuai aturan. Public sempurna pun tidak memastikan private sempurna.

## 8. Pelajaran INFEST dan webinar

Penilaian INFEST menekankan keterkaitan temuan data→preprocessing→evaluasi, visualisasi bernarasi, serta notebook yang benar-benar dieksekusi. Temuan tersebut diadopsi; aturan dan Macro-F1 INFEST tidak dipindahkan ke GELAR RASA. Laporan penelaahan Excel terpisah memuat sumber sheet/sel.

Materi 1 h.11–14: representasi RGB, split, augmentasi dan baseline; h.17–18 fitur hybrid; h.20 transform hati-hati; h.22–29 evaluasi/learning curves. Materi 2 h.12–19 dan 33–34: grafik, denominator, konsistensi, integritas. Tidak mengasumsikan bahwa perbaikan visual otomatis memperbaiki klasifikasi.

## 9. Keputusan dan langkah berikutnya

1. Gunakan manifest sumber, bukan recursive glob seluruh train.
2. Pertahankan native RGB serta koordinat/rasio letterbox; jangan denoise/CLAHE/crop massal tanpa ablasi.
3. Tinjau grup near-duplicate dan risiko rantainya sebelum menjadikan group_id sebagai split final.
4. Tim memetakan nama kategori cluster dan mencatat bila kelompok tidak koheren; jangan memaksakan koreksi satu per satu.
5. Bekukan versi label dan split yang sah, lalu lakukan fine-tuning serta evaluasi tahap berikutnya. Jangan memakai skor internal clustering sebagai estimasi leaderboard.

## Artefak dan reproduksi

- `output/manifests/stage1_v1/`: sumber, metadata, SHA-256.
- `data/processed/stage1_v1/`: native RGB, letterbox, folder cluster train/test, template mapping.
- `output/groups/stage1_v1/`: exact groups, kandidat, cluster assignment dan provenance.
- `reports/eda/stage1_20261105/galleries/index.html`: semua gambar dan tetangga.
- `notebook/01_EDA_Preprocessing_Clustering.ipynb`: kode mandiri dan hasil eksekusi.

Keluaran processed diperlakukan immutable; cluster_views memakai hardlink bila filesystem mendukung, jadi mengedit file di satu view juga mengubah salinan native yang terkait. Untuk perubahan baru gunakan versi output baru.

Tanggal/asumsi aturan: 5 November 2026. Bantuan penulisan kode berdasarkan klarifikasi pengguna diperbolehkan selama notebook tidak memanggil LLM; notebook ini tidak memakai layanan inference eksternal atau VLM.

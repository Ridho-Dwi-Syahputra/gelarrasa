# Penyempurnaan cluster train, 9 November 2026

Pembaruan berikutnya: audit kini memperhitungkan **13 pengecualian eksplisit peserta**, bukan hanya arsip mentah. Dari 102 selisih mentah sebelumnya, 13 memang sengaja berbeda sesuai review, sehingga antrean yang diminta peserta adalah 89. Aturan geometri sumber dan tiga classifier yang menahan seluruh source group memulihkan 0064 menjadi cardboard: 191 SIFT matches, 183 inliers, inlier fraction 0,958, convex hull sekitar 0,768 pada kedua gambar, dan enam dari enam prediksi cardboard. Koreksi itu dihitung ulang di notebook; antrean menjadi **88** dan selisih arsip mentah menjadi 101. Seluruh 109 review tegas eligible tetap terjaga. Notebook terbaru lulus 19/19 sel tanpa error dan validator lulus.

Uji baru soft Dawid–Skene mempelajari kekeliruan empat sumber otomatis lama dengan 35 acuan manusia. Delapan konfigurasi dievaluasi memakai lima fold kelompok; yang terpilih memberi 11/35 cocok, macro recall 0,3458, sehingga gagal guard 29/35 dan 0,8. Tidak ada hasil eksperimen itu diterapkan. Dasar metode: [Snorkel](https://arxiv.org/abs/1711.10160) dan [Dawid–Skene](https://doi.org/10.2307/2346806). Model sumber saling berkorelasi dan acuan kecil; memperkirakan confusion matrix saja belum menyelesaikan ambiguitas material. Lihat `json/weak_source_em_frozen.json`, `json/geometric_group_label_frozen.json`, dan `json/current_reference_with_exceptions.json`.

## Hasil aktif

Peserta mengonfirmasi sendiri foto train 1039 sebagai **foodwaste**. Koreksi ini dicatat sebagai acuan manusia ke-35; kategori foodwaste kini memakai **2 dari 5** jatah. Foto tersebut dan 1.886 foto utuh eligible lainnya diekstraksi menjadi embedding DINOv2 dari data train saja. KMeans 32 kelompok dijalankan pada lima seed tetap. Foto yang berada satu kelompok dengan 1039 pada minimal empat pengulangan membentuk inti stabil 72 foto. Tidak ada acuan manusia lain yang bertentangan dalam inti ini.

Foto 1039 diubah sebagai acuan manusia; 70 foto lain mendapat pseudo-label foodwaste melalui aturan cluster. Sel notebook mengulang fit KMeans, memindahkan PNG hasil ke folder kategori, memperbarui manifest, lalu melatih tiga classifier pada label baru. Data test tidak masuk fit cluster atau classifier. Arsip anotasi visual lama tidak masuk target, fitur, atau penentuan anggota kelompok.

Perbandingan arsip dilakukan **sesudah kandidat dibekukan**. Dari 1.887 foto utuh eligible yang beririsan dengan arsip, ketidakcocokan turun dari **173 ke 102**: 71 cocok kembali dan nol ketidakcocokan baru. Angka ini adalah agreement dengan pembanding historis, bukan Accuracy semantik resmi. Arsip sendiri pernah memiliki kasus yang ditolak peserta; menyamakan seluruh 102 secara paksa bukan tujuan validasi model.

Contoh konkret: dari delapan ID yang sebelumnya kamu nyatakan sudah benar menurut program, **0260, 0267, dan 0396** masih berbeda dari arsip. Menyamakan ketiganya ke arsip justru membatalkan keputusan reviewmu. Karena itu, sisa 102 tidak boleh diperlakukan sebagai 102 target koreksi otomatis.

## Percobaan yang tidak dipakai

| Kandidat | Perubahan yang diusulkan | Audit pembanding setelah pembekuan | Keputusan |
| --- | ---: | --- | --- |
| Semua 35 acuan, KMeans DINOv2 | 206 | 3 pulih, 199 regresi; total 298 berbeda | Ditolak karena kelas seperti cardboard/electronic banyak masuk textile |
| Acuan plastic 0321, fitur objek | 28 | 0 pulih, 28 regresi; total 130 berbeda | Ditolak; kemiripan objek tidak membuktikan bahan glass/plastic |
| Pasangan acuan kelas sama, dua bank fitur | 0 | Tidak diperlukan | Tidak diterapkan; gate terlalu konservatif untuk tambahan perubahan |

Eksperimen 0321 menunjukkan batas metode secara konkret: cluster stabil dapat menggabungkan botol dari bahan berbeda. Karena itu, aturan satu acuan 1039 tidak diperluas otomatis ke semua acuan.

## Verifikasi

- Notebook [DSC2026_Notebook.ipynb](../../final-submit/DSC2026_Notebook.ipynb) selesai **dua kali berturut-turut**, masing-masing 18 sel kode, nol error. Tiga CSV hasil kedua run memiliki hash identik.
- Validator proyek lulus: 2.400 sumber, 2.393 gambar processed, 35 acuan manusia dengan batas lima per kelas, 116 edge geometri dalam group yang sama, 480 ID per CSV, dan kategori folder sesuai manifest.
- Foto 1039 kini `foodwaste`, `label_source=participant_reviewed_seed`. Sebanyak 70 foto lain memiliki `label_source=train_only_cluster_anchor_1039_v1`.
- Sisa **102** adalah selisih terhadap arsip, belum 102 kesalahan panitia yang terbukti. Tidak ada label test yang diketahui.

## Batas reproduksi paket

Tahap cluster 1039 dihitung ulang oleh notebook. Pipeline sebelumnya masih bergantung pada `data/preprocessing_inputs.npz`, snapshot assignment train historis yang diperiksa hash. Agar rerun panitia benar-benar mandiri, dataset dan snapshot tersebut harus tersedia melalui sumber yang bisa diakses panitia; dua Run All lokal belum membuktikan akses eksternal itu.

# Sumber primer

Ditinjau 4–5 November 2026. Status baca dibedakan agar abstrak tidak dianggap audit penuh paper/kode. Dokumentasi `stable` dapat berubah; versi implementasi dikunci saat digunakan. Tidak ada dataset/bobot eksternal yang diunduh. Aturan aktif 5 November dapat mengecualikan metode paper dari kandidat kompetisi.

| ID | Sumber | Jenis dan cakupan baca | Relevansi |
|---|---|---|---|
| S01 | [Towards Accurate and Efficient Waste Image Classification, Nguyen dkk.](https://arxiv.org/html/2510.21833v1) | Preprint 2025, bagian metode/hasil/koreksi label ditinjau | CNN features + classifier klasik; audit label |
| S02 | [Versi jurnal, Ain Shams Engineering Journal 17(4), 104062](https://www.sciencedirect.com/science/article/pii/S2090447926000894) | Jurnal 2026, abstrak/highlights | Versi terbit terkait S01; angka tidak identik, jangan digabung |
| S03 | [RealWaste, Single dkk., Information 14(12), 633](https://www.mdpi.com/2078-2489/14/12/633) | Jurnal 2023, cuplikan abstrak terindeks; akses penuh 429 saat peninjauan | Perbedaan foto bersih dan limbah di lokasi nyata |
| S04 | [TrashNet, Gary Thung dan Mindy Yang](https://github.com/garythung/trashnet) | Repo penulis, deskripsi dataset | Enam kelas, tidak sama langsung dengan delapan kelas kompetisi |
| S05 | [TACO, Proenca dan Simoes](https://arxiv.org/abs/2003.06975) | Preprint 2020, abstrak; [repo penulis](https://github.com/pedropro/TACO) | Deteksi/segmentasi sampah dalam konteks, berbeda target dari klasifikasi |
| S06 | [EfficientNet, Tan dan Le, ICML 2019](https://proceedings.mlr.press/v97/tan19a.html) | Paper konferensi, abstrak | Baseline CNN dan trade-off resolusi/komputasi |
| S07 | [EfficientNetV2, Tan dan Le, ICML 2021](https://proceedings.mlr.press/v139/tan21a.html) | Paper konferensi, abstrak | Alternatif CNN efisien; progressive learning sebagai opsi |
| S08 | [A ConvNet for the 2020s, Liu dkk.](https://arxiv.org/abs/2201.03545) | Paper CVPR 2022, abstrak arXiv; CVF fetch 403 | ConvNeXt sebagai pembanding |
| S09 | [DINOv2, Oquab dkk.](https://arxiv.org/abs/2304.07193) | Paper, abstrak versi 2024 | Embedding frozen untuk linear probe/retrieval bila bobot boleh |
| S10 | [CLIP, Radford dkk.](https://arxiv.org/abs/2103.00020) | Paper 2021, abstrak | Pembanding semantik bersyarat, bukan ground truth |
| S11 | [Confident Learning, Northcutt, Jiang, Chuang, JAIR 70](https://dspace.mit.edu/entities/publication/39b0f8e0-b673-42c6-99f6-0af829bef7f5) | Jurnal 2021, abstrak publikasi institusi penulis | Peringkat dugaan masalah label |
| S12 | [Leakage and the reproducibility crisis, Kapoor dan Narayanan, Patterns 4(9)](https://pmc.ncbi.nlm.nih.gov/articles/PMC10499856/) | Jurnal 2023, bagian taksonomi/ringkasan | Pencegahan kontaminasi evaluasi |
| S13 | [mixup, Zhang dkk.](https://arxiv.org/abs/1710.09412) | Paper ICLR 2018, abstrak | Interpolasi gambar dan target |
| S14 | [CutMix, Yun dkk.](https://arxiv.org/abs/1905.04899) | Paper ICCV 2019, abstrak | Pertukaran patch dan target berbobot area |
| S15 | [Grad-CAM, Selvaraju dkk.](https://arxiv.org/abs/1610.02391) | Paper, abstrak | Diagnostik lokasi perhatian |
| S16 | [FixMatch, Sohn dkk.](https://arxiv.org/abs/2001.07685) | Paper NeurIPS 2020, abstrak | Konsistensi weak/strong augmentation dan pseudo-label |
| S17 | [Calibration of Modern Neural Networks, Guo dkk.](https://proceedings.mlr.press/v70/guo17a.html) | Paper ICML 2017, abstrak | Confidence dan temperature scaling |
| S18 | [PyTorch transfer learning](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html) | Dokumentasi resmi | Freeze/head/fine-tune |
| S19 | [timm feature extraction](https://huggingface.co/docs/timm/feature_extraction) | Dokumentasi resmi | Ekstraksi embedding sesuai API |
| S20 | [Torchvision transforms](https://docs.pytorch.org/vision/stable/transforms.html) | Dokumentasi resmi | Transform dan dtype |
| S21 | [Ultralytics classification](https://docs.ultralytics.com/tasks/classify/) | Dokumentasi resmi, bagian klasifikasi dan crop | YOLO `-cls`, risiko crop pada aspect ratio ekstrem |
| S22 | [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) | Dokumentasi resmi | Kelompok non-overlap dan stratifikasi mendekati |
| S23 | [cleanlab data-centric workflows](https://docs.cleanlab.ai/stable/tutorials/indepth_overview.html) | Dokumentasi resmi | Audit label memakai probabilitas di luar training sampel |
| S24 | [Nested versus non-nested CV](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html) | Dokumentasi resmi, tujuan outer/inner CV | Repeat CV tidak menghapus bias seleksi model |
| S25 | [StackingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.StackingClassifier.html) | Dokumentasi resmi, fungsi CV internal dan prefit | CV pembentuk meta-training berbeda dari evaluasi prosedur lengkap |
| S26 | [cleanlab image Datalab](https://docs.cleanlab.ai/stable/tutorials/datalab/image.html) | Dokumentasi resmi, input OOF/features dan jenis issue | Prioritas tinjauan label, duplikat, outlier, citra gelap; tidak otomatis relabel |
| S27 | [FiftyOne image embeddings](https://docs.voxel51.com/tutorials/image_embeddings.html) | Tutorial resmi, ekstraksi dan proyeksi | Galeri/neighbor dan navigasi domain; proyeksi bukan label |
| S28 | [Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html) | Dokumentasi resmi, exif_transpose dan pilihan resize | Orientasi dan perbandingan pad/contain/fit |
| S29 | [scikit-image data types](https://scikit-image.org/docs/stable/user_guide/data_types.html) | Dokumentasi resmi, dtype/range dan konversi | Hindari float 0–255 dianggap float 0–1 atau normalisasi dua kali |
| S30 | [scikit-image HOG](https://scikit-image.org/docs/stable/auto_examples/features_detection/plot_hog.html) | Dokumentasi resmi, deskriptor gradien | Baseline tekstur/bentuk; cocokkan channel_axis dengan versi |
| S31 | [Kaggle docker-python](https://github.com/Kaggle/docker-python) | Repo resmi, README/struktur runtime | Persiapan lingkungan Kaggle; bukan bukti GPU/kuota akun saat ini |
| S32 | [Kaggle kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md) | Dokumentasi resmi, files/output/status dan versi | Mengarsipkan hasil notebook; tidak ada upload/run dalam tugas ini |
| S33 | [Voxel51 skills](https://github.com/voxel51/fiftyone-skills) dan [AREX-Skill](https://github.com/VectorSpaceLab/AREX-Skill) | Snapshot commit, entrypoint dan bagian terkait; bukan audit semua kode | Tujuh paket tambahan untuk kurasi, preprocessing, label quality, dan evaluasi |
| S34 | [Handbook resmi](<../../rules/HANDBOOK PRELIMINARY ROUND DSC GELAR RASA 2026.pdf>) | Seluruh 8 halaman diekstrak; timeline/rubrik diperiksa visual | Bobot 70/20/10, lima finalis, format/nama, tenggat, larangan metode |
| S35 | [Q&A Padlet panitia](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk) | Seluruh 19 posting/31 komentar snapshot; 17 jawaban resmi dibedakan | Larangan label/koreksi per gambar, VLM, pretrained non-VLM, reset 07.00 WIB |
| S36 | [Template laporan resmi](https://docs.google.com/document/d/1mtwAMvd3a8Zn7zr4fo9gZtsT3zWDdr_7bgvwA77xrEE/edit) | Ekspor TXT dibaca; DOCX/PDF diarsipkan | BAB I–IV; 10 halaman inti; A4/TNR12/1,15/justify/APA7 |
| S37 | [Scientific writing](https://github.com/K-Dense-AI/scientific-agent-skills/tree/a25cf9ec00e2745d03f8704acf6f6919694ccd9a/skills/scientific-writing) dan [scientific visualization](https://github.com/K-Dense-AI/scientific-agent-skills/tree/a25cf9ec00e2745d03f8704acf6f6919694ccd9a/skills/scientific-visualization) | Entry point/bagian terkait ditinjau; subtree MIT dipin commit; helper belum dijalankan | Bukti klaim, konsistensi angka, grafik dan pemeriksaan export yang disesuaikan rubrik |

Untuk reproduksi persis satu metode, baca full text, suplemen, dan kode sumber terlebih dahulu. Sumber di atas cukup untuk menyusun hipotesis dan skill; tidak membuktikan metode mana terbaik pada data kita.

Tambahan audit 5 November 2026:

| ID | Sumber primer | Cakupan baca | Penerapan lokal/batas |
|---|---|---|---|
| S38 | [DINOv2 paper](https://arxiv.org/html/2304.07193v2) dan [repo resmi](https://github.com/facebookresearch/dinov2) | Pengantar image-only, resolusi, frozen/linear evaluation; README backbone | ViT-S/14 feature extraction lokal; DINO.txt tidak dipakai; bukan bukti Accuracy lomba |
| S39 | [timm ViT-S DINOv2 model card](https://huggingface.co/timm/vit_small_patch14_dinov2.lvd142m) | Model details, feature extraction, normalisasi/resolusi/lisensi | Checkpoint pin4610ca143709d58a633b6397a74412c2c3842454, Apache2; SHA dicatat; resolusi224/336 adalah ablation |
| S40 | [SCAN, ECCV2020](https://arxiv.org/html/2005.12320v2), [loss resmi](https://raw.githubusercontent.com/wvangansbeke/Unsupervised-Classification/master/losses/losses.py) | Bagian2.1–2.3, evaluasi, persamaan2 dan SCANLoss | Implementasi lokal terinspirasi neighbor loss; frozen head/KMeans init/full-batch, bukan reproduksi lengkap; kode repo tidak disalin |
| S41 | [IIC, ICCV2019](https://arxiv.org/html/1807.06653v4) | Metode, auxiliary overclustering, eksperimen | Usulan head bantu untuk variasi intra-kategori; belum dijalankan |
| S42 | [Backgrounds Challenge](https://arxiv.org/abs/2006.09994), [repo](https://github.com/MadryLab/backgrounds_challenge) | Abstrak dan dokumentasi | Dasar hipotesis pengaruh latar; bukan bukti sebab error lokal |
| S43 | [DECMCV waste clustering, preprint2025](https://arxiv.org/pdf/2503.02241) | Metode3.1–3.2, tabel2 dan evaluasi | Angka93,78% disertai37% data dikeluarkan; ide consensus untuk review, bukan menghilangkan test sulit |

Eksperimen aktual dan batas klaim tersedia di `reports/errors/stage2_20261105/ANALISIS_DAN_PENANGANAN.md` dari root proyek. Artikel mendukung hipotesis; tidak membuktikan skor1,0000 pada semua480test.

Catatan akses: halaman Kaggle Notebooks tidak memberikan isi yang dapat dibaca pada penelusuran ini, sehingga panduan runtime memakai repo resmi S31–S32. Halaman reproducibility Albumentations tidak berhasil diakses; contoh dari snapshot AREX tetap perlu dicocokkan dengan dokumentasi versi saat implementasi. S01 mempunyai ketidakselarasan narasi bagian 4.4 dan Tabel 11 yang dicatat di [tinjauan riset](RESEARCH_REVIEW.md); angka headline tidak dijadikan target yang dijamin.

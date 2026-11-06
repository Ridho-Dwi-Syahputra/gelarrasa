# GELAR RASA 2026 ? analisis citra limbah

Dataset resmi lokal: 1.920 train tanpa label kategori resmi dan 480 test. Delapan kategori: glass, paper, electronic, metal, plastic, cardboard, textile, foodwaste. Metrik kompetisi: Accuracy.

## Pembagian kategori terbaru — 6 November 2026

Model bertingkat kini memberi satu prediksi kelas untuk seluruh 1.920 train dan 480 test. Pada 570 train yang sebelumnya campuran, 487 cocok dengan arsip pembanding dan 83 berbeda; total seluruh train 1.740 cocok / 180 berbeda. Ini agreement pribadi, **bukan Accuracy resmi atau bukti semua label benar**. Arsip tidak menjadi target atau filter per gambar. Pengembangan iteratif telah melihat audit terdahulu, sedangkan keputusan setiap run dibekukan sebelum auditnya.

- [Notebook preview model kategori](notebook/02_Material_Resolution_Review.ipynb): empat sel dijalankan, preview tiap kelas dan antrean tinjau; memakai hasil training yang sudah dibekukan.
- [Hasil seluruh percobaan dan batas interpretasi](reports/errors/stage6_material_resolution_20261106/RESULTS_AND_LIMITS.md).
- [Bukti model/folder dan perbandingan terbaru](reports/errors/stage6_material_resolution_20261106/final_validation_and_private_audit.json).
- Folder lokal: `data/processed/stage17_named_predictions_v1/73996d8db5/{train,test}/{glass,paper,electronic,metal,plastic,cardboard,textile,foodwaste}/`.

Tanda tinjau 570 train saat ini memiliki arti berbeda dari 570 kelompok campuran lama: 126 softmax/margin rendah dan 444 hanya konflik sumber CNN. Semua gambar punya kategori; tanda bukan jumlah salah terverifikasi. Data, model, gambar dan cache tetap lokal; label aktif belum dipromosikan.

## Notebook EDA utama dan riwayat

[Notebook utama](notebook/01_EDA_Preprocessing_Clustering.ipynb) kini versi **EDA visual v3**: preview train/test, nama material sementara, distribusi kelompok, kasus kualitas/ambigu, kandidat crop, kemiripan piksel dan dimensi native. Sepuluh sel kode memanggil modul yang dapat ditelusuri; head baseline yang sudah dilatih dipakai ulang setelah verifikasi hash. Audit visual dijalankan baru, tanpa mengklaim fitting ulang baseline.

- [Analisis dan keputusan preprocessing](reports/eda/stage3_visual_20261106/ANALISIS_VISUAL.md).
- [Bukti Run All v3](reports/reproducibility/notebook_v3_execution.json) dan [pemeriksaan integritas](reports/reproducibility/notebook_v3_delivery_validation.json).
- [Seluruh 480 test dan tetangga train ? galeri lokal](reports/eda/stage3_visual_20261106/galleries/index.html).
- [Panduan menjalankan notebook di latar belakang](docs/MENJALANKAN_NOTEBOOK.md).

Temuan: tiga test identik piksel dengan train; 29 kandidat SSIM >=0,99; 439 test memiliki ukuran native yang juga tersedia pada train. Crop menghasilkan 2.171 kandidat train dan 627 preview test, dengan fold induk dipertahankan. Sebagian crop mengubah hasil model dan watershed dapat membelah satu benda; kandidat belum dipromosikan menjadi label/training.

**Asal nama:** interpretasi visual sementara oleh asisten, diminta pengguna untuk EDA pribadi. Dua kelompok bercampur plastic/glass dan plastic/textile/paper. Nama bukan target training/validasi/submission kompetisi dan tidak membuktikan Accuracy. Asal dan batas penggunaan direkam di [konfigurasi tampilan](configs/display_label_mapping.json). Kontrak label aktif tetap belum dipromosikan. Arsip anotasi lama tidak dibaca dalam jalur audit ini.

## Audit perbandingan dan persiapan self-training

Atas klarifikasi pengguna, arsip label dipakai sebagai pembanding post-hoc saja. [Audit](reports/errors/stage4_seed_audit_20261106/ANALISIS_SELF_TRAINING.md) memisahkan 216 perbedaan dari 570 kelompok campuran yang belum bisa dinilai tunggal. Perbandingan bukan Accuracy resmi dan tidak mengubah pilihan seed.

Filter model tanpa membaca arsip menghasilkan 1.233 kandidat inti, 686 cadangan ambigu dan satu baris duplikat yang tetap disimpan. Audit seed tersebut adalah persiapan. Pilot terpisah kemudian melatih dua teacher dan empat student/head dalam dua putaran, tanpa label arsip sebagai input. Lihat [ringkasan audit seed](reports/errors/stage4_seed_audit_20261106/summary.json).

Folder lokal `data/processed/stage3_named_display_review/{train,test}/` menampilkan nama material dari head terbaru untuk inspeksi pribadi. 0046 kini berada pada `train/paper/0046.png`; `stage1_named_review` adalah hasil lama. Nama tampilan dan antrean audit bukan input training kompetisi.

Pilot self-training kini tercakup pada sel terakhir notebook. [Hasil development dan batas interpretasi](reports/errors/stage4_self_training_20261106/summary.json) memakai 1.538 train-fit dan 381 heldout kelompok; backbone tetap frozen. Kandidat dasar tidak diganti dan Accuracy kategori belum diukur.

Uji lanjutan fitur lokal dan 32 prototipe sudah selesai: 18 fitting dan 26 kondisi pembanding. Student lama tetap terpilih; kelompok lebih rinci juga memecah kelas, sehingga belum menjadi pengganti. Lihat [hasil dan keputusan stage5](reports/errors/stage5_material_structure_20261106/RESULTS_AND_DECISION.md). Notebook utama tetap versi yang sudah lulus 10/10.

## Isi repo

Kode, notebook dengan preview tertanam, konfigurasi, serta dokumen/tabel kecil diunggah. Dataset, gambar terpisah, crop, embedding/checkpoint, backup dan file besar tetap lokal. Jangan melanjutkan rencana upload aset lama yang dibatalkan pengguna. Semua komputasi agent berjalan di latar belakang tanpa membuka atau mengambil fokus jendela.

Untuk menjalankan setelah clone, siapkan dataset resmi di `data/Dataset DSC GelarRasa 2026/{train,test}/` dan sample di `data/SampleSubmission_DSC.csv`, buat `.venv` dari `requirements-lock.txt`, lalu ikuti panduan notebook. Artefak baseline akan dibangun jika belum tersedia. Materi webinar dan dokumen besar tetap tersedia pada komputer kerja, termasuk ekstrak teks kecil yang disertakan repo.

## Referensi proyek

1. [Pustaka 22 skill khusus + 17 paket snapshot](docs/skills/README.md) dan [pemilih skill per prompt](docs/skills/custom/gelar-rasa-skill-router/SKILL.md).
2. [Aturan dan rubrik](docs/skills/references/RULES_AND_RUBRIC.md), [rancangan laporan](docs/skills/references/REPORT_BLUEPRINT.md), [rencana EDA/preprocessing](docs/skills/references/EDA_PREPROCESS_PLAN.md).
3. [Materi webinar](docs/skills/references/WEBINAR_MAP.md) dan [riset metode](docs/skills/references/RESEARCH_REVIEW.md).
4. [Struktur folder](docs/STRUKTUR_PROYEK.md), [ledger eksperimen](docs/eksperimen.md), dan [kontrak artefak](docs/skills/references/ARTIFACT_CONTRACTS.md).
5. [Persiapan GPU Kaggle](docs/skills/custom/kaggle-gpu-workflow/SKILL.md); belum meluncurkan sesi GPU/upload dataset Kaggle.

## Riwayat baseline

Versi v2 telah lulus Run All 35/35 sel pada 5 November dan melatih head image-only pada train. Snapshot sebelum perubahan visual v3 disimpan lokal di `reports/reproducibility/backups/20261106_before_visual_v3/`. [Analisis tahap 2](reports/errors/stage2_20261105/ANALISIS_DAN_PENANGANAN.md) dan [tahap 1](reports/eda/stage1_20261105/ANALISIS_TAHAP_1.md) merupakan riwayat; bukti v3 di atas berlaku untuk notebook utama saat ini.

Arsip visual lama `submission_99` kini `docs/secret/submission_.csv`; public 0,9930 berasal dari screenshot pengguna. Arsip bukan label validasi atau hasil classifier baru. Belum membuat submission baru pada pekerjaan EDA visual ini.

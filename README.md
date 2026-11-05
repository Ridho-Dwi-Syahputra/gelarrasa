# GELAR RASA 2026 — Klasifikasi citra limbah

Klasifikasi satu gambar ke delapan kategori: glass, paper, electronic, metal, plastic, cardboard, textile, foodwaste. Data resmi lokal terdiri atas 1.920 gambar train dan 480 test; metrik kompetisi **Accuracy**.

Raw train tidak memiliki label resmi. Label visual asisten yang pernah dibuat di `train/label/` dan `docs/train_label_review/` kini **arsip yang dikecualikan dari jalur kompetisi**, setelah Q&A resmi melarang label/koreksi per gambar manual. Label aktif hasil metode peserta belum tersedia. Raw tetap dipertahankan.

## Mulai di sini

Notebook utama kini memuat tahap 1 dan tahap 2 dalam **66 sel, 35 sel kode**. Run All dari `.venv` lulus **35/35, 0 error**, 1.152,76 detik pada CPU. Fitur dan fitting dihitung ulang. [Bukti eksekusi dan integritas](reports/reproducibility/notebook_v2_delivery_validation.json) memverifikasi enam eksekusi modul tahap 2 serta 2.400 hash raw terhadap backup sebelum run. [Petunjuk membuka notebook](docs/MENJALANKAN_NOTEBOOK.md) menjelaskan kernel **Python (GELAR-RASA .venv)** dan pemulihan PDF webinar setelah clone.

Pembaruan tahap 2, 5 November: [analisis perbedaan dan eksperimen baru](reports/errors/stage2_20261105/ANALISIS_DAN_PENANGANAN.md), [delapan folder train/test baru](data/processed/stage2_neighbor_v1/README.md), dan [galeri seluruh hasil head clustering](reports/errors/stage2_20261105/galleries/neighbor_refinement_clusters.html) tersedia. DINOv2 image-only, delapan konfigurasi representasi, enam kandidat head konsistensi tetangga, serta uji view sudah dijalankan. Head dilatih pada train tanpa target kategori; nama cluster tetap anonim. Kestabilan perubahan resolusi pada 256 train naik dari 241/256 menjadi 253/256; **ini bukan Accuracy kategori**. [Validasi awal tahap 2](reports/errors/stage2_20261105/final_validation.json) adalah bukti run sebelumnya; validasi notebook gabungan terbaru tersedia pada tautan bukti eksekusi di atas. Belum membuat submission baru atau mengganti mapping kategori aktif.

Tahap 1 kini memiliki [notebook EDA/preprocessing/clustering](notebook/01_EDA_Preprocessing_Clustering.ipynb), [analisis hasil](reports/eda/stage1_20261105/ANALISIS_TAHAP_1.md), dan [galeri seluruh citra serta tetangga setiap test](reports/eda/stage1_20261105/galleries/index.html). Gambar turunan berada di [data/processed/stage1_v1](data/processed/stage1_v1/README.md): native RGB, letterbox, dan delapan folder cluster untuk tiap split. Cluster masih anonim; mapping kategori oleh tim belum tersedia. Status Run All gabungan terbaru tersimpan di `reports/reproducibility/notebook_v2_execution.json`; berkas `notebook_execution.json` tahap 1 merupakan riwayat run baseline 25 sel.

[Folder peninjauan bernama kategori](data/processed/stage1_named_review/README.md) menyediakan struktur `train/<kategori>/<cluster_id>` dan `test/<kategori>/<cluster_id>`. Nama adalah alias dari komposisi arsip lama, bukan label material terverifikasi. Keanggotaan cluster dibekukan; folder peninjauan ini dikecualikan dari training/submission aktif.

1. [Pustaka skill — 22 khusus proyek + 17 paket snapshot eksternal, 240 file vendor](docs/skills/README.md) dan [pemilih skill per prompt](docs/skills/custom/gelar-rasa-skill-router/SKILL.md).
2. [Aturan/Q&A dan rubrik aktif](docs/skills/references/RULES_AND_RUBRIC.md), [rancangan laporan](docs/skills/references/REPORT_BLUEPRINT.md), serta [rencana EDA/preprocessing](docs/skills/references/EDA_PREPROCESS_PLAN.md).
3. [Materi webinar](docs/skills/references/WEBINAR_MAP.md) dan [riset/metode kandidat](docs/skills/references/RESEARCH_REVIEW.md).
4. [Tahapan kerja](docs/skills/references/EXPERIMENT_PLAYBOOK.md) dan [kontrak artefak evaluasi](docs/skills/references/ARTIFACT_CONTRACTS.md).
5. [Persiapan GPU Kaggle](docs/skills/custom/kaggle-gpu-workflow/SKILL.md), [struktur folder](docs/STRUKTUR_PROYEK.md), dan [ledger eksperimen](docs/eksperimen.md).

Status 5 November 2026: aturan, 19 posting/31 komentar Q&A, template laporan resmi, pustaka skill, serta tahap 1 tersedia. Clustering algoritmis dan diagnostik OOF train-vs-test telah dijalankan; **belum ada classifier kategori limbah yang di-fine-tune atau Accuracy kategori**. Kaggle bernilai 70, laporan 20, notebook 10; tenggat 14 November 2026 23.59 WIB. Alur label harus berasal dari metode peserta, bukan anotasi AI lama. VLM termasuk feature extraction dilarang; klarifikasi pengguna mengizinkan bantuan kode tanpa pemanggilan LLM di notebook.

Tahap 1 berjalan lokal/CPU, menggunakan bobot non-VLM ResNet50 ImageNet1K V2 yang tersedia di cache. Notebook menyertakan kode pipeline lengkap dan output Run All. GPU Kaggle tetap opsi berikutnya; belum ada upload dataset atau sesi GPU Kaggle yang diluncurkan.

Training peserta nanti membaca manifest baru dengan provenance yang sesuai aturan. CSV arsip visual awal `submission_99.csv` sekarang berada di `docs/secret/submission_.csv` dengan hash identik. Screenshot pengguna menunjukkan public Accuracy 0,9930; [catatan skor dan batas interpretasi](reports/public_scores/submission_99_user_report_20261105.json) tersimpan. Arsip tidak menjadi target training/validasi atau pemetaan kategori aktif. Data dan label tidak diubah oleh pencatatan skor.

Pemeriksaan paket: `python docs/skills/tools/validate_collection.py --verify-images --no-write`. Opsi `--verify-images` memeriksa hash byte train asli dan salinan; `--no-write` menjaga laporan tersimpan tetap sama. Tanpa `--no-write`, hasil menimpa `docs/skills/sources/validation.json`. Pemeriksaan ini tidak melakukan decode gambar, membuktikan label benar, atau mengukur Accuracy; lihat [cakupan validator](docs/skills/README.md#pemeriksaan-paket).

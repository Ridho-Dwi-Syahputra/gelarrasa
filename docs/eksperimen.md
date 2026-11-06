# Ledger GELAR RASA

## Pemisahan material dengan CNN dan pembagian bertingkat — 6 November 2026

- Rangkaian stage6–17 benar-benar dijalankan: CNN objek/material, fusi terkalibrasi, seleksi seed, pilot adaptasi blok ViT, dua pretrained CNN khusus limbah, ConvNeXt22k/ontologi, head linear/MLP,24 SVM/kNN,18 konfigurasi difusi graf, dan pembagian bertingkat. Detail, kontrol negatif dan sumber: `reports/errors/stage6_material_resolution_20261106/RESULTS_AND_LIMITS.md`.
- Pilot blok akhir dua learning rate/empat epoch per kondisi menunjukkan gradien dan update; kontrol epoch0 tetap menang. Jangan menyebut backbone model akhir di-fine-tune berdasarkan pilot yang ditolak.
- Kesepakatan kuat CNN tanpa target dari cluster awal meningkatkan pemisahan570 menjadi 487 cocok / 83 berbeda, tetapi merusak kohort stabil. Routing stage17 memakai statistik kategori dari fit-only consensus: minimal 20 anchor dan proporsi dominan<0,6 memilih refinement; sisanya mempertahankan stage12. Dua kelompok terpilih secara programatis; tidak membaca mapping asisten atau flag campuran arsip.
- Pembanding akhir setelah freeze: **1.740 cocok, 180 berbeda**, dari keseluruhan 1.920 train. Pecahan570 lama:487 cocok / 83 berbeda. Awal164 perbedaan mengecualikan570 yang belum dapat dibandingkan, sehingga denominator awal tidak setara dengan180 sekarang.
- Tanda tinjau akhir570 train terdiri 126 softmax/margin rendah dan 444 konflik sumber saja. Itu kriteria berbeda dari 570 kelompok campuran lama. Test 25 rendah+101 konflik sumber saja; semua 480 test punya prediksi. Tidak menghapus citra ambigu atau memaksakan jumlah kelas.
- Materialisasi/hash/reproduksi probabilitas lulus untuk 2.400 gambar. Folder lokal: `data/processed/stage17_named_predictions_v1/73996d8db5/`. Hardlink read-only mengarah ke salinan independen stage7, bukan raw/native sumber; sumber asli dipertahankan.
- Notebook pendamping 02 lulus 4/4 sel, 0 error, 3 figure preview tertanam, tanpa mengulang training atau membaca arsip. Notebook 01 tetap v3 utuh dan bukti10/10 historisnya masih berlaku. Kode, laporan dan notebook kecil disinkronkan; gambar/model/cache tetap lokal.
- Arsip tidak menjadi target/filter per gambar atau pemetaan; keputusan kandidat setiap run dibekukan sebelum auditnya. Peneliti telah melihat audit sebelumnya: pengembangan metode bersifat adaptif, bukan validasi buta. Agreement arsip bukan Accuracy resmi; tidak ada promosi label aktif, submission baru, atau klaim sempurna. Pretraining overlap tidak diketahui; lisensi/model dataset CNN14 tidak dinyatakan pembuat. Sumber dan keterbatasan tetap terbuka.

## Uji representasi lokal dan overclustering - 6 November 2026

- Atas permintaan mencari cara agar kelompok mendekati material pada pembanding, diuji fitur CLS, mean/std patch non-padding dan hybrid. Referensi tidak menjadi input fitting atau seleksi.
- 18 fitting KMeans (k8/32, tiga jenis fitur, tiga seed), penggabungan Ward pada dua jenis fitur, dan dua kontrol memberi26 kondisi. Fit1538/development381 terpisah kelompok; views87 heldout. Dua puluh empat model assignment direproduksi pada2400 gambar.
- Student lama tetap menang pada proxy development0,968946; hybrid32-Ward8-seed47 kandidat baru terdekat0,966736. Backbone frozen, bukan fine-tuning representasi.
- Audit privat setelah freeze: homogeneity student0,6919 vs hybrid32 0,7946, tetapi completeness turun0,6961 ke0,4909; ini fragmentasi, bukan Accuracy yang meningkat. ARI student0,6041 tetap di atas kandidat tetap yang diaudit.
- Tidak mengubah baseline, label aktif, notebook atau jumlah perbedaan student yang dilaporkan sebelumnya. Arah belum diuji berikutnya: joint adaptation encoder/head dengan pseudo-label algoritmis terpercaya, bukan arsip LLM.
- Bukti: `reports/errors/stage5_material_structure_20261106/RESULTS_AND_DECISION.md`, `validation.json`, dan `private_reference_audit.json`. Fitur/model tetap lokal; kode dan laporan kecil disinkronkan ke repo.


## Pilot teacher/student dua putaran ? 6 November 2026

- Klarifikasi pengguna: arsip sebagai pembanding; kasus berbeda disimpan untuk ditinjau lagi. Training terpisah hanya membaca fitur image-only, kelompok sumber dan pseudo-group model. Nama tampilan maupun arsip tidak dibaca oleh kode training.
- Dua teacher dilatih dari awal pada 1.538 train-fit. Sebanyak 381 heldout kelompok terpisah; nol overlap kelompok/pixel. View flip tersedia untuk 169 fit dan 87 heldout dari probe 256 train yang dipilih sebelumnya.
- Saringan model menghasilkan 1.110 seed fit dan menyimpan 428 cadangan ambigu. Tiga kondisi student awal diuji; putaran kedua mencoba memasukkan kembali empat contoh yang kini konsisten menurut model/tetangga/flip. Total empat head student, backbone tetap frozen.
- Kontrol teacher: neighbor agreement 0,900262 dan flip86/87. Student terpilih seed-CE/graph epoch20: neighbor agreement0,901837 dan flip87/87. Putaran kedua tidak mengalahkan kondisi terpilih; berhenti pada batas dua putaran, bukan mengejar kesamaan arsip.
- Setelah pembelajaran/pemilihan selesai, audit private pada cohort sama 1.346 kasus bernama tunggal menunjukkan kesesuaian arsip1180?1182, dua pergeseran menuju arsip dan nol menjauh. Sisanya574 campuran tidak dihitung sebagai label tunggal. Ini bukan Accuracy resmi atau alasan memilih model.
- 0046 tetap pada tampilan paper pada teacher dan student. Baseline/kontrak label aktif tidak diganti. Semua model/seed/cadangan tetap lokal.
- Kode: `stage4_self_training_pilot.py`; verifikasi dan perbandingan pasca-freeze: `audit_frozen_student.py`. Laporan: `reports/errors/stage4_self_training_20261106/`. Pilot juga dieksekusi pada sel terakhir notebook utama.

## Audit usulan self-training dan kasus 0046 ? 6 November 2026

- Pengguna mengusulkan referensi arsip LLM dan iterasi pembersihan/training. Pengguna kemudian memperjelas penggunaan sebagai pembanding. Dibuat antrean audit terpisah tanpa mengubah mask training; belum melatih student baru.
- Kasus 0046 diperiksa: alias lama electronic memiliki margin 0,037; head terbaru sudah pada kelompok bertema paper sebelum permintaan ini. Tidak ada override per gambar.
- Perbandingan post-hoc 1.920 train: dari 1.350 anggota kelompok nama tunggal, 1.134 sejalan arsip dan 216 berbeda; 570 anggota kelompok campuran tidak dinilai sebagai label tunggal. Ini perbandingan terhadap arsip asisten, bukan Accuracy resmi. Assignment sebelum/sesudah identik.
- Audit seleksi independen dari arsip: 1.233 seed kandidat, 686 cadangan ambigu, satu duplikat disimpan. Confidence/margin/tetangga/kesepakatan model dihitung pada train, tanpa label arsip atau nama tampilan. Tidak menghapus data atau mempromosikan label kategori.
- Tampilan hasil terbaru bernama material dibuat pada `data/processed/stage3_named_display_review/`, memakai tautan ke salinan model tahap 2 tanpa memindahkan anggota. 0046 tersedia pada folder paper. Antrean referensi berada di `data/processed/stage4_reference_audit_only/`; hash seed, assignment dan kontrak tetap sama.
- Laporan: `reports/errors/stage4_seed_audit_20261106/ANALISIS_SELF_TRAINING.md`; manifest lokal: `data/processed/stage4_seed_audit_v1/`. Teacher sebelumnya refit seluruh train, sehingga audit ini bukan evaluasi holdout independen.

## EDA visual v3 ? 6 November 2026

- Permintaan: preview train/test, distribusi label tampilan, crop gambar berdekatan, contoh kasus teknis, kemiripan piksel dan ukuran. Skill: image-eda-duplicates, image-preprocessing-audit, grouped-image-validation. Seluruh pekerjaan berjalan di latar belakang.
- Pengguna meminta nama dari pengamatan asisten. Sebanyak 128 contoh hasil model ditinjau; interpretasi dicatat eksplisit untuk EDA pribadi saja, bukan label training/validasi/submission. Dua kelompok campuran tetap ditulis campuran; label aktif pada kontrak tidak dipromosikan.
- Notebook utama dibangun menjadi v3 dengan sembilan sel kode modular. Baseline/head v2 dipakai ulang setelah verifikasi fitur/checkpoint/sumber; tidak mengklaim refit model pada tugas ini. Snapshot v2 dipertahankan lokal.
- Semua 2.400 gambar dianalisis regionnya. Dihasilkan 2.171 crop train dan 627 crop test untuk preview; 174 crop dari 68 train memenuhi saringan konservatif. Semua kandidat belum dipromosikan ke training dan mewarisi kelompok/fold induk.
- Probe crop: 13/96 train dan 6/48 test berubah kelompok dari gambar penuh; 93/96 train dan 48/48 test stabil terhadap flip. Ini sensitivitas view, bukan Accuracy/koreksi benar.
- Pencarian RGB MSE semua 480?1.920 pasangan dan SSIM shortlist menemukan tiga test identik piksel, 29 kandidat SSIM>=0,99, 19 kandidat setelah saringan MSE/aspect ratio. Ukuran 439 test muncul pada train; ukuran sama sendiri tidak membuktikan duplikat/kelas.
- Preview connected components dan watershed dipisahkan, caption dibungkus agar tidak bertumpuk, dan skor MSE diberi alignment yang sama dengan tampilan heatmap. Galeri 15 halaman mencakup seluruh 480 test.
- Laporan: `reports/eda/stage3_visual_20261106/ANALISIS_VISUAL.md`. Bukti terbaru: `notebook_v3_execution.json` dan `notebook_v3_delivery_validation.json` pada `reports/reproducibility/`.
- Dataset/gambar terpisah/crop/model/file besar tetap lokal. Notebook berpreview tertanam, kode, konfigurasi dan dokumen kecil saja yang dikirim ke repo.

## Run All ulang di latar belakang ? selesai 5 November 2026, 23.58 WIB

- Pengguna meminta tidak mengambil fokus layar. Kontrol UI dihentikan, VS Code dikembalikan ke latar belakang dan pengaturan aksesibilitas sementara dipulihkan persis ke byte awal. Preferensi ini disimpan pada `AGENTS.md`.
- Runner `.venv` menjalankan ulang seluruh tahap 1+2, dengan DINO dihitung ulang. Waktu mulai `2026-11-05T23:37:10.507189+07:00`, selesai `2026-11-05T23:58:14.626989+07:00`, durasi 1264.14 detik.
- Hasil: 35/35 sel kode, 177 output tersimpan, tanpa error. Enam eksekusi modul tahap 2 terverifikasi; 2.400 hash sumber cocok dengan backup, 2.400 probabilitas head direproduksi dan tidak ada overlap kelompok fit/holdout.
- Notebook SHA-256: `1e1a712b2c10a3bb21dca5395330e193d96a3f3d9faaa2027503d913354753b8`. Bukti terbaru berada di `reports/reproducibility/notebook_v2_delivery_validation.json`. Pratinjau HTML lokal diperbarui dari output tersimpan, tanpa menjalankan training lagi.
- Dataset, gambar terpisah, model/embedding dan file besar tetap lokal. Notebook dengan grafik tertanam, kode dan laporan kecil saja yang dikirim ke Git.

## Pembatasan isi repo dan pemeriksaan tampilan notebook ? 5 November 2026

- Pengguna mengubah lingkup push: dataset, gambar terpisah dan file besar tidak perlu diunggah. Upload 15 batch aset tersisa dibatalkan; commit aset yang belum terpublikasi hanya dipertahankan pada ref backup lokal. File lokal tidak dihapus.
- Repo melacak kode, notebook beroutput, dan dokumen kecil. Seluruh `data/`, `output/`, backup, potongan PDF dan ekstensi gambar/model/arsip dikecualikan. Batas pemeriksaan file lain 5 MiB; notebook utama dikecualikan dari batas ini agar 176 output tersimpan tetap tersedia.
- Pemeriksaan langsung notebook pada disk: 35/35 sel memiliki execution_count 1?35, 176 output, 14 sel memuat gambar, nol error, dan tidak ada flag yang menyembunyikan output. Tidak dilakukan training ulang karena hasil Run All lengkap sudah tersimpan.
- Pratinjau HTML lokal dibuat dari output tersebut melalui `eksperimen/skrip/preview_executed_notebook.py`, tanpa mengeksekusi model. Lokasi: `reports/reproducibility/notebook_preview/01_EDA_Preprocessing_Clustering.html`.

## Notebook gabungan dan lingkungan lokal ? 5 November 2026

- Skill: reproducible-image-submission dan report-package-audit; investigate-first untuk kegagalan proses Git.
- Notebook utama diperluas menjadi 66 sel, 35 sel kode; kernel `gelar-rasa-venv`, dependensi terisolasi di `.venv`, versi lengkap pada `requirements-lock.txt`. `pip check` lulus.
- Run All penuh lulus 35/35, tanpa error, 1.152,76 detik pada CPU. `RECOMPUTE_DINO=True`: fitur, PCA/KMeans dan enam kandidat head dihitung/dilatih ulang; checkpoint backbone dapat memakai cache. Percobaan pertama terhenti karena lock Windows saat penyimpanan; laporan kegagalan dipertahankan, penyimpanan kini mencoba ulang secara atomik.
- Semua 1.920 train dan 480 test tersedia pada delapan folder anonim per split. Materializer memeriksa setiap hash salinan. Sel akhir mereproduksi 2.400 probabilitas dari checkpoint, memeriksa cakupan ID, pixel-identik serta overlap kelompok fit/holdout nol.
- Pemeriksaan pascarun: 2.400 hash raw cocok dengan manifest pada backup sebelum perubahan; enam hash kode modul cocok dengan catatan runtime; notebook menyimpan urutan eksekusi 1?35. Bukti baru: `reports/reproducibility/notebook_v2_execution.json`, `notebook_v2_in_kernel_checks.json`, `notebook_v2_delivery_validation.json`.
- Snapshot sebelum perluasan disimpan di `reports/reproducibility/backups/20261105_before_notebook_v2/`. Bukti `final_validation.json` tahap 1/2 lama merupakan riwayat dan bukan bukti hash notebook gabungan yang baru.
- Repo private tujuan: `Ridho-Dwi-Syahputra/gelarrasa`. Kuota LFS akun menolak upload. Aset disimpan memakai Git biasa; PDF webinar 940.599.610 byte dibagi 20 bagian 45 MiB, dipulihkan dan diverifikasi SHA-256 oleh `restore_large_files.py`. PDF asli lokal dipertahankan.
- Tidak ada Accuracy kategori terukur, promosi mapping kategori, atau submission baru pada pekerjaan ini.

## Run tahap 2 — stage2_repr_v1 dan stage2_neighbor_v1, 5 November 2026

- Permintaan: analisis perbedaan train/test dengan arsip, cari penanganan sesuai aturan, dan kelompokkan train juga. Skill yang dipakai: image-error-analysis, image-preprocessing-audit, vision-research-review; router/aturan diperiksa.
- Audit post-hoc seluruh data: 590 train + 171 test berbeda; 333 train + 102 test tetap margin>=0,10 dan seluruh seed sepakat. Lima contact sheet diperiksa secara teknis; tidak ada kategori per gambar dikoreksi.
- Eksperimen representasi: ResNet50 dan DINOv2 ViT-S/14 image-only; tanpa PCA, PCA64/128/256; masing-masing tiga seed KMeans8. Semua fitting hanya 1.919 train pixel-unik. Delapan refit penghapusan kelompok train mengukur sensitivitas partisi, bukan Accuracy.
- DINOv2 checkpoint pinned/revision/SHA tersimpan di `output/embeddings/stage2_repr_v1/dino_checkpoint.json`. Seluruh 2.400 embedding diekstrak lokal/CPU; tambahan hflip224 dan full-frame336 pada 256 train acak yang dipilih sebelum pengujian view. Tidak ada VLM, API inference, atau unduhan dataset eksternal.
- Enam head SCAN-inspired dilatih pada fitur DINO frozen dengan split train candidate-group 80/20. Pilihan menggunakan loss tanpa label dan delapan cluster terisi: seed47/entropy2/epoch60; refit seluruh train unik. Ini adaptasi head, bukan reproduksi SCAN penuh atau fine-tuning backbone.
- Konsistensi tetangga held-out train kandidat terpilih: 0,868241 -> 0,900262. Konsistensi cluster saat224 ->336 pada probe train: baseline tahap1 241/256, head baru253/256. Kedua metrik **bukan Accuracy kategori**.
- Semua train dikelompokkan: `data/processed/stage2_neighbor_v1/train/cluster_00..07`, jumlah 241/252/227/182/330/241/240/207. Test memakai model tetap, 480 gambar. Seluruh PNG merupakan salinan independen dari RGB native, tanpa reassignment manual.
- Tindak lanjut audit setelah assignment dibekukan: 283/590 kasus train lama berpindah setelah alignment cluster; ini bukan jumlah koreksi benar. Sebanyak236/590 masih punya flag struktural. Seluruh train373/1.920 ditandai, test97/480; tidak ada penghapusan gambar. Tidak ada pemilihan model atau nama cluster memakai arsip.
- Validasi lulus: 2.400 hash raw, 2.400 hash salinan folder, 19.200 prediksi KMeans dan 2.400 prediksi head direproduksi, nol overlap kelompok fit/holdout, pixel-identik mendapat assignment sama, 9.202 tautan galeri valid. Raw, stage1 notebook/assignment, arsip submission, dan kontrak aktif tetap sama.
- Artefak/analisis: `reports/errors/stage2_20261105/ANALISIS_DAN_PENANGANAN.md`; hasil pemeriksaan `final_validation.json`; kode `eksperimen/skrip/stage2_*.py` dan `materialize_stage2_clusters.py`.
- Keputusan: head merupakan kandidat pengelompokan yang layak diuji lanjut; belum pemenang semantik. Mapping kategori aktif tetap kosong; Accuracy kategori dan fine-tuning backbone belum tersedia. Tahap 2 kemudian lulus Run All pada notebook gabungan seperti dicatat di atas; belum membuat submission baru. Tidak ada klaim mencapai1,0000.

## Status 5 November 2026

- Arsip anotasi visual train lama: 1.920/1.920 dengan 51 kasus meragukan; **dikecualikan dari jalur kompetisi** setelah Q&A terbaru. Integritas arsip tetap diperiksa, tetapi tidak digunakan training/validasi/nama cluster. Label aktif baru belum tersedia.
- Arsip anotasi visual test dahulu `submission_99.csv`, sekarang `docs/secret/submission_.csv`; SHA-256 identik dengan catatan akhir arsip. Screenshot pengguna menunjukkan public Accuracy **0,9930**, pada sekitar 30% test menurut pengguna. Bukti berasal dari screenshot percakapan, bukan pengambilan langsung dari Kaggle. Private score dan identitas kesalahan belum diketahui; arsip tetap tidak dipakai sebagai label validasi model.
- Persiapan pengetahuan: 22 skill khusus dan 17 paket snapshot eksternal (240 file vendor). Ditambah skill pemilih, laporan, grafik, audit paket, dan rancangan clustering peserta. Jumlah snapshot tidak menghitung ulang subskill bersarang.
- [Rubrik label lokal](skills/references/LABEL_TAXONOMY.md), [rencana EDA/preprocessing](skills/references/EDA_PREPROCESS_PLAN.md), dan [kontrak artefak](skills/references/ARTIFACT_CONTRACTS.md) tersedia sebagai panduan; penulisannya tidak mengubah label atau menyelesaikan audit visual baru.
- Persiapan mencakup skill preprocessing, domain shift, diagnostik training, dan [GPU Kaggle](skills/custom/kaggle-gpu-workflow/SKILL.md). Notebook tahap 1 kini tersedia dan dijalankan lokal/CPU; belum ada staging/upload dataset atau sesi GPU Kaggle.
- **Training model GELAR RASA pada tahap pengumpulan skill: belum dijalankan.** Belum ada OOF, best model, atau rekomendasi submission hasil training.
- [Aturan aktif](skills/references/RULES_AND_RUBRIC.md) kini tersedia: handbook 8 halaman, seluruh 19 posting/31 komentar Padlet dalam snapshot, serta template laporan PDF/DOCX/TXT. Belum mengisi form, membuat salinan di akun Google, atau mengunggah paket.
- Rencana lama yang memakai anotasi manual sebagai label_v1 dibatalkan sebagai jalur kompetisi; bukan sekadar mengganti nama label tersebut menjadi hasil cluster. Pengembangan model oleh peserta mengikuti batas LLM pada aturan.

## Antrean awal

| Prioritas | Run yang direncanakan | Hipotesis | Prasyarat | Status |
|---|---|---|---|---|
| P0 | exp001_audit_manifest_labels | Asal raw dan label aktif dapat dibedakan dari arsip manual | Manifest raw dan aturan aktif | Direvisi; persiapan |
| P0 | exp002_eda_preprocessing_audit | Decode, warna, orientasi, dan detail material perlu diperiksa sebelum pemodelan | Sumber asli; rencana EDA; galeri sebelum/sesudah | Direncanakan; persiapan |
| P1 | exp003_duplicate_domain_folds | Kelompok sumber dan cakupan domain memberi pembanding yang lebih dapat dipercaya | EDA, konfirmasi kandidat duplikat, label terlacak | Direncanakan; persiapan |
| P2 | exp004_participant_clustering | Metode peserta dapat membentuk target turunan dengan provenance | Fitur non-VLM, aturan, mapping nama cluster oleh tim | Rancangan konseptual; belum dijalankan |
| P2 | exp005_frozen_embedding_linear | Representasi pretrained dapat membantu data kecil | Aturan bobot jelas; fold sama; tahap training diminta | Direncanakan |
| P3 | exp006_effb0_finetune | Fine-tuning memperbaiki error baseline frozen | Baseline, review residual, dan runtime siap | Direncanakan |
| P4 | exp007_hybrid_ablation | Fitur klasik memberi informasi tambahan pada embedding | OOF komponen setara | Direncanakan |

Angka di antrean dapat berubah sebelum run pertama; setelah run dimulai, ID tidak dipakai ulang. Setiap perubahan metode, label, atau split dicatat. Rincian strategi: [EXPERIMENT_PLAYBOOK](skills/references/EXPERIMENT_PLAYBOOK.md).

Lingkup aktif: memahami aturan dan menyiapkan skills, EDA teknis serta laporan sesuai rubrik. Label/koreksi citra oleh agent tidak dilanjutkan. Persiapan Kaggle berupa panduan input-output/config; runtime GPU diperiksa saat peserta akan menjalankan alur yang sesuai aturan. Tabel bukan bukti training/hasil model.

## Run tahap 1 — stage1_v1, 5 November 2026

Permintaan pengguna: EDA rinci, preprocessing, clustering train/test, pHash/retrieval, belajar dari penilaian INFEST dan materi webinar. Pengguna mengklarifikasi bahwa bantuan kode boleh selama notebook tidak memanggil LLM. Semua kategori tetap anonim sampai mapping cluster oleh tim tersedia.

- Sumber: 1.920 train + 480 test. Seluruh gambar didekode; canonical RGB PNG native dan letterbox224 disimpan, tidak ada silent exclusion.
- Audit: byte/pixel SHA-256, pHash/dHash/wHash, kandidat flip pHash, distribusi train–test, metadata kualitas, dan galeri semua gambar.
- Fitur: frozen ResNet50 IMAGENET1K_V2, tanpa VLM; pembanding HSV/HOG. PCA64 dan KMeans fit pada 1.919 pixel-unik train, test hanya ditransform/assign.
- Primary clustering ditetapkan sebelum comparison: k8 seed11; 5 seed CNN, 2 seed klasik, Ward8, sensitivitas k6/10/12/16. Test tidak masuk fit cluster. Cluster dan silhouette bukan label/Accuracy resmi.
- Validasi: OOF classifier domain train-vs-test dengan preprocessing fold-local; draft 5-fold berkelompok untuk train, bukan validasi kategori independen.
- Artefak: `notebook/01_EDA_Preprocessing_Clustering.ipynb`, `reports/eda/stage1_20261105/`, `output/{manifests,groups,embeddings,folds}/stage1_v1/`, `data/processed/stage1_v1/`.
- Run All pertama lulus 25 sel kode. Koreksi simetri pHash flip diikuti Run All penuh ulang: 25/25 sel kode, 0 error, 375,29 detik pada CPU. Status dan durasi ada pada `notebook_execution.json`. Sesudahnya hanya satu paragraf markdown diperjelas tentang resize128 descriptor klasik; kode terjalankan dan output dipertahankan.
- Validasi akhir lulus: 2.400 hash sumber, 4.800 hash turunan, 2.400 isi cluster views, 4.819 target tautan galeri, 480 bagian tetangga test, dan nol group overlap pada draft folds. Enam unit test preprocessing/hash juga lulus.
- Perbandingan terhadap folder arsip dilakukan sesudah assignment dibekukan; SHA-256 assignment sebelum/sesudah sama (`709ca688c5ff1e56d74ac7275cec75f19829e9cc7efb8d52d9306c78387bd903`). Detail perbandingan dilaporkan di chat sesuai permintaan; tidak dimasukkan ke notebook atau dipakai untuk nama/anggota cluster.
- Belum dilakukan: mapping nama kategori oleh tim, fine-tuning classifier limbah, Accuracy semantik, submission. Perbandingan arsip yang diminta hanya setelah assignment dibekukan, tidak untuk menyetel/menamai cluster.

Validator paket memakai `--verify-images` untuk memeriksa hash 1.920 sumber train dan 1.920 salinan terhadap manifest awal; ini tidak memeriksa decode atau ketepatan label. `--no-write` mencegah penggantian `docs/skills/sources/validation.json`. Hasil pemeriksaan struktur/integritas dicatat terpisah dari hasil model, dan tidak diberi skor Accuracy.

## Format catatan hasil

Catatan skor 5 November: `reports/public_scores/submission_99_user_report_20261105.json`. Angka 0,9930 menunjukkan kesesuaian sangat tinggi pada subset public, tetapi tidak menetapkan Accuracy semua test atau label train. Perbandingan cluster dengan arsip test tetap 309 sama/171 berbeda; angka berbeda bukan jumlah kesalahan LLM. Public Accuracy clustering belum diukur. Nama cluster hasil peninjauan berasal dari komposisi arsip train, bukan pemetaan kategori independen. Keputusan metodologis: jangan memperlakukan cluster k8 sekarang sebagai label material siap-training atau menganggap ketidaksepakatan sebagai izin menimpa label/hasil; status arsip dan aturan tetap berlaku.

`run_id | parent | hypothesis | change | label_version | fold_version | seed | accuracy | macro_f1 | errors | runtime | artifact_path | decision`

Isi skor hanya dari eksekusi nyata; nilai belum tersedia ditulis `null`/belum diukur. Public score berasal dari hasil kompetisi yang terverifikasi, bukan estimasi CV.

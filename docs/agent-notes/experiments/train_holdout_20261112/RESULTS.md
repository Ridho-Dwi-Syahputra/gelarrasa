# Rerun train holdout ? 12 November 2026

## Keputusan data

- Dipertahankan dari daftar tahap pertama: 0018, 0086, 0120, 0604, 0637, 0856, 0890, 1129. Kelas tahap pertama dipakai; 0086 menjadi seed elektronik baru.
- Label manual yang dikonfirmasi tim: 0200 dan 0265 sebagai foodwaste.
- Sebelas ID ditahan dari fit classifier: 0079, 0160, 0183, 0189, 0225, 0400, 1018, 1115, 1142, 1157, 1436.
- Untuk menjaga batas lima acuan per kelas, seed elektronik 0147 diganti 0086 dan 0147 ikut ditahan. Tiga dari sebelas ID (1018, 1142, 1436) sebelumnya sudah tertahan.
- Ledger sekarang 37 seed: {"cardboard": 5, "electronic": 5, "foodwaste": 4, "glass": 5, "metal": 3, "paper": 5, "plastic": 5, "textile": 5}.

## Training dan evaluasi

Notebook Run All selesai: 24 sel kode, nol error. Classifier memakai 1879 foto train utuh; tujuh crop tidak masuk fit. Test hanya untuk inferensi; test rows used in fit = 0.

DINOv3 Base OVR RBF-SVM: grouped OOF agreement 0.9548, balanced accuracy 0.9550, macro-F1 0.9554. Ini agreement dengan target train turunan, bukan accuracy semantik resmi.

Kalibrasi classwise lebih rendah pada nested OOF (agreement 0.9516 vs 0.9548), maka CSV eksperimennya diarsipkan dan bukan kandidat utama. Skor leaderboard kandidat baru belum diketahui.

## Berkas

- Kandidat baru: `final-submit/output/submission_008_dinov3_full_ovr_svm.csv` (480 ID unik).
- Submission 009 lama dipertahankan utuh: `final-submit/output/submission_009.csv`.
- Diagnostik OOF, laporan, dan kandidat lebih lemah dipindah ke folder `experiment_outputs/` ini. File sementara notebook juga ada di `notebook_scratch/`; output lama lengkap ada di `prior_output/` dan `interrupted_full_run/`.
- Tidak ada file submission yang dikirim ke Kaggle.

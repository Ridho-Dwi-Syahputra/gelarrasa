---
name: grouped-image-validation
description: Membuat fold citra berdasarkan kelompok sumber dan menyimpan OOF lengkap untuk membandingkan model pada Accuracy tanpa kontaminasi duplikat, preprocessing, atau anotasi test.
---

# Validasi gambar

Baca [konteks](../../references/PROJECT_CONTEXT.md). Bekukan manifest, label_version, class_order, group_id, dan seed sebelum perbandingan. Gunakan [StratifiedGroupKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) bila kelompok tersedia; stratifikasi tidak selalu sempurna ketika kelompok besar. Periksa cukup kelompok tiap kelas; turunkan jumlah fold bila perlu. Lima fold kandidat awal, bukan kewajiban 10/20 fold.

Assert tidak ada group di train dan validation bersamaan, setiap ID mendapat tepat satu prediksi OOF per repeat, dan seluruh kelas direpresentasikan sesuai kelayakan data. Buat split sebelum augmentasi. Fit scaler, PCA, feature selector, classifier, dan setiap backbone supervised hanya pada fold train. Backbone pretrained frozen boleh dibagi cache bila sama sekali tidak diadaptasi pada label lokal; simpan tag dan transform.

Tuning/stacking/calibration memakai inner split dalam outer evaluation. Jangan menggunakan validation untuk pseudo-label training lalu mengklaim OOF independen. Early stopping membuat skor fold dipakai untuk seleksi; jangan menyebutnya final untouched test. Repeat pada ID yang sama mengukur stabilitas, bukan evaluasi independen setelah tuning. Untuk menilai prosedur seleksi gunakan nested CV atau holdout kelompok yang benar-benar belum dipakai; ungkap bila desain metode sebelumnya sudah dipengaruhi seluruh data.

Ikuti [kontrak artefak dan evaluasi bertingkat](../../references/ARTIFACT_CONTRACTS.md), terutama ketika backbone, kalibrasi, atau meta-model ikut dilatih. Bekukan ID/label evaluasi dan catat versi, bukan hanya seed splitter.

Simpan probabilitas delapan kelas, prediksi, target turunan dari metode peserta, fold, group, serta provenance pembentuk target per ID di `final-submit/output/` (berkas OOF ketika tahap pelatihan sudah dijalankan). Nyatakan apakah pembentukan target melihat semua train. Laporkan agreement pooled/per fold, macro-F1 dan disagreement dengan keterbatasannya; tanpa ground truth independen itu bukan bukti Accuracy semantik. Jangan memakai anotasi asisten lama atau menghapus kasus sulit agar skor naik.

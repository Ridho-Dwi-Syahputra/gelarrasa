# Eksperimen classifier train-only, 10 November 2026

Tujuan: mencari representasi/classifier yang lebih kuat tanpa membaca `submission_.csv` sebagai target, fitur, atau input notebook. Pembandingan dengan file itu dilakukan hanya setelah kandidat dibekukan. Semua OOF memakai pemisahan `parent_group`; labelnya adalah target otomatis pada train, bukan ground truth semantik resmi.

## Hasil

| Metode | OOF agreement | Catatan |
|---|---:|---|
| DINOv2 + HSV SVM | 0.973783 | Kandidat baseline, `submission_004_dino_hsv_svm.csv` |
| DINOv3 CLS SVM | 0.986624 | Kandidat `submission_006_dinov3_cls_svm.csv`; nested model selection vs baseline 0.988229 |
| DINOv3 full descriptors + multiclass RBF SVM | 0.987694 | Kandidat `submission_007_dinov3_full_svm.csv` |
| DINOv3 full descriptors + OVR RBF SVM | 0.989834 | Kandidat `submission_008_dinov3_full_ovr_svm.csv`; balanced agreement 0.990190, macro F1 0.990287 |

DINOv3 full descriptors memakai CLS, valid-patch mean, object-patch mean, dan object-patch standard deviation. Lokalisasi YOLO pada test hanya membentuk fitur saat inferensi; tidak ada test target masuk ke classifier. Bobot image-only checkpoint dicatat di JSON kandidat.

WRS berbasis inverse class frequency, inverse source-group frequency, gabungan class/group, serta bobot tambahan untuk contoh yang sulit tidak mengubah prediksi OOF pada SVM. MLP dengan WeightedRandomSampler tidak mengalahkan baseline DINOv2+HSV. LinearSVC dan LogisticRegression pada DINOv3 juga di bawah OVR RBF.

## Audit pascapembekuan

Perbandingan diagnostik setelah model dan CSV dibekukan: kandidat 004 sama dengan 466/480 baris file lama; 006 sama 471/480; 007 sama 470/480; 008 sama 472/480. Jadi OVR mengurangi perbedaan dari 14 menjadi 8, tetapi belum identik. File lama tidak dipakai memilih fitur, hyperparameter, atau label kandidat.

OOF di atas menunjukkan agreement terhadap label otomatis train. Itu tidak menjamin accuracy LB/private. Kandidat 008 belum memperoleh skor Kaggle. Tidak ada submission yang dikirim. Notebook final belum diperbarui untuk menghasilkan kandidat 006-008; CSV/JSON ini adalah artefak eksperimen terpisah.

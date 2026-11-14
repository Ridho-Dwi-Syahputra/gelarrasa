# Audit hasil Kaggle tahap pertama

Target tetap: `docs/submission_ (1).csv`. Semua pembandingan dijoin berdasarkan ID test, bukan urutan listing folder. Analisis ini tidak mengubah label, model, atau submission.

Ensemble: **270 berbeda, 210 sama dari 480 baris** (43.75% sama). CSV final cocok dengan argmax rata-rata probabilitas empat model.

| Model | Sama | Berbeda |
|---|---:|---:|
| resnet50 | 217 | 263 |
| efficientnet_v2_s | 211 | 269 |
| convnext_small | 213 | 267 |
| swin_v2_t | 210 | 270 |
| ensemble | 210 | 270 |

| Kelas target | Jumlah target | Sama | Berbeda | Jumlah prediksi ensemble |
|---|---:|---:|---:|---:|
| cardboard | 60 | 49 | 11 | 93 |
| electronic | 60 | 8 | 52 | 50 |
| foodwaste | 60 | 26 | 34 | 35 |
| glass | 60 | 25 | 35 | 42 |
| metal | 60 | 2 | 58 | 11 |
| paper | 61 | 21 | 40 | 53 |
| plastic | 60 | 50 | 10 | 136 |
| textile | 59 | 29 | 30 | 60 |

## Temuan penyebab

1. Run lama membuat pseudo-label baru dari 37 seed EfficientNetV2-S, memilih 1.690 dari 1.920 foto, dan membuang 230 dengan ambang confidence. Ia tidak menggunakan processed aktif.
2. Dari 1879 foto utuh processed aktif, 1171 memperoleh label berbeda pada run lama. Pada 1650 foto yang sama-sama benar-benar dipakai, 988 label berbeda dan hanya 662 sama.
3. Filter confidence lama membuang 229 foto utuh eligible aktif, sementara 40 foto di luar full-frame aktif masuk training. Tujuh crop aktif tidak digunakan.
4. Pseudo-target lama sangat tidak seimbang: plastic 521, cardboard 344, metal 39. Prediksi test menjadi plastic 136 dan metal 11. Pergeseran ini didukung jumlah label dan prediksi, bukan sekadar dugaan batas GPU.
5. Semua backbone menyelesaikan 21 epoch refit (3 head + 18 fine-tune). Run bukan berhenti setelah tiga epoch.
6. Pada log refit lama, kolom val_proxy_agreement sama persis dengan train_proxy_agreement di seluruh epoch; kode lama menggunakan hasil train ketika tidak ada validation loader. Validasi fold terpisah berada pada proxy_summary.csv, dengan agreement terhadap pseudo-target sekitar 61,48?66,98%.
7. Pada 199 gambar test, semua empat prediksi individual berbeda dari target. Pada 71 perbedaan ensemble, setidaknya satu model individual cocok target. Ini menunjukkan masalah target training bersama, bukan hanya pilihan satu backbone.

## Berkas detail

- `submission_differences.csv`: seluruh 270 ID yang berbeda, kedua kategori, prediksi individual, confidence, dan jumlah model yang cocok target.
- `all_test_predictions.csv`: seluruh 480 ID.
- `old_train_vs_active_processed.csv`: audit label dan eligibility per foto train.
- `target_vs_ensemble_confusion.csv` dan `confusion_pairs.csv`: pasangan kategori yang tertukar.
- `model_comparison.csv`, `class_comparison.csv`, `training_history_summary.csv`, `summary.json`: ringkasan dan provenance.

Notebook port terbaru memakai snapshot processed aktif 1.879 foto utuh + 7 crop, direkonstruksi dari raw dan diverifikasi hash piksel. Audit run lama ini tidak memengaruhi pemilihan label atau prediksi test pada notebook baru.

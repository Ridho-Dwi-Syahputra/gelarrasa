# Training CPU dari foto utuh eligible — 9 November 2026

## Ruang lingkup

Model classifier dilatih pada **1.869 foto utuh train eligible**. Tujuh crop tidak dimasukkan ke fit. Sebanyak 37 foto yang ditahan tetap di luar fit. Data test berisi 480 gambar dan hanya dipakai untuk ekstraksi fitur/inferensi. Semua classifier berjalan di CPU.

## Model dan validasi

Fitur beku DINOv2-small diproses dengan pipeline train yang sudah ada; classifier dibandingkan memakai lima fold `StratifiedGroupKFold` berdasarkan source group:

| Model | OOF agreement dengan target train | Macro-F1 | Balanced accuracy |
|---|---:|---:|---:|
| Logistic regression | 0,96041 | 0,96081 | 0,96037 |
| SVM RBF | **0,96843** | **0,96900** | **0,96889** |
| kNN-7 cosine | 0,95452 | 0,95518 | 0,95543 |

Target train berasal dari proses pelabelan yang telah dijelaskan sebelumnya. Karena itu, metrik ini mengukur reproduksi target turunan, **bukan Accuracy semantik resmi**. SVM dipilih sebagai kandidat lokal berdasarkan OOF train, bukan berdasarkan isi file prediksi test lama atau leaderboard.

## Hasil

Notebook membuat tiga CSV, masing-masing 480 ID unik dengan kolom `image_id,waste_category`:

- `final-submit/output/submission_001_dino_logistic.csv`
- `final-submit/output/submission_002_dino_svm_rbf.csv` — kandidat utama saat ini
- `final-submit/output/submission_003_dino_knn7.csv`

File yang disebut peserta sebagai `output/submission/_.csv` tidak ada pada path tersebut. File pembanding yang ada di folder output bernama `final-submit/output/submission_.csv`; isi file itu tidak dipakai dalam fit, validasi, pemilihan model, atau perubahan prediksi.

## Reproduksi

Run All notebook utama selesai melalui builder latar belakang: 20 sel kode, 0 error, 19 visual, 162,7 detik. Validator proyek dengan `--verify-images` lulus dan memeriksa 4.793 hash sumber/hasil. Rincian hash dan laporan output ada di `final-submit/output/modeling_report.json` serta `docs/agent-notes/execution.json`.

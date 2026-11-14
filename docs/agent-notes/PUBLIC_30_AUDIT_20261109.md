# Audit skor public 30% dan tiga student DINOv2

Tanggal: 9 November 2026 WIB. Angka public berasal dari tangkapan layar peserta, bukan API. Empat CSV memiliki 480 ID dalam urutan identik. `final-submit/output/submission_.csv` byte-identik dengan arsip visual terdahulu `arsip/sebelum_rapi_20261107/docs/secret/submission_.csv`; file ini hanya pembanding posthoc, tidak dipakai sebagai target training, seleksi bobot, atau koreksi test.

| Berkas | Public screenshot | Jika 144 public, benar | Berbeda dari `submission_.csv` pada 480 test |
| --- | ---: | ---: | ---: |
| `submission_.csv` | 0,9930 | 143/144 | — |
| logistic DINOv2 | 0,9027 | 130/144 | 45 |
| SVM RBF DINOv2 | 0,9097 | 131/144 | 41 |
| kNN7 DINOv2 | 0,9166 | 132/144 | 42 |

Jumlah benar 144 adalah inferensi dari persentase public 30% dari 480 dan tampilan skor, bukan daftar ID public. Tiga student sepakat satu sama lain tetapi berbeda dari arsip pada 22 test. Pola dominan dari perbandingan CSV: 12 `foodwaste` arsip berubah menjadi `textile` pada ketiga model; beberapa `plastic` menjadi `glass`; `metal` menjadi `electronic`. Ketiga model setuju dengan arsip pada test 017/042/201 yang identik piksel setelah preprocessing dengan train.

Audit train eligible full-frame (1.887 foto) terhadap arsip pembanding menunjukkan 1.714 cocok dan 173 berbeda. Dari 173, **73** adalah arsip `foodwaste` → target program `textile` (55 kelompok sumber), dan **28** arsip `plastic` → program `glass` (28 kelompok). Pada 73 kasus pertama, semua memiliki flag `needs_review=True`, tetapi 71 skornya ≥0,9 dan median skor 0,997; skor assignment bukan probabilitas kebenaran. Train 1039, 0894, 1421 menampilkan adegan piring/sisa makanan dan dekat secara embedding dengan beberapa test dalam kelompok yang sama secara visual, tetapi semuanya mempunyai target aktif `textile`. Banyak gambar ini memuat pakaian pekerja di latar; atribut latar bisa menjadi shortcut. Untuk `plastic` → `glass`, galeri audit memperlihatkan botol atau kemasan bening. Ini menjelaskan mekanisme kesalahan student, tetapi arsip bukan ground truth resmi.

OOF 5 fold pada pseudo-label membalik ranking public: SVM agreement 0,9340 > kNN 0,9219, sedangkan public kNN 0,9166 > SVM 0,9097. OOF ini mengukur kemampuan meniru target otomatis dan terkontaminasi oleh pembentukan target seluruh train; jangan gunakan sebagai estimasi Accuracy sebenarnya.

## Dampak GPU

Notebook sekarang membekukan DINOv2 ViT-S (`eval` + `inference_mode`) lalu melatih logistic/SVM/kNN pada fitur 384 dimensi. Memindahkan eksekusi yang sama dari CPU ke GPU mempercepat ekstraksi fitur, tetapi tidak memperbaiki target pseudo yang sistematis salah. GPU baru berpotensi membantu bila dipakai untuk eksperimen **train-only** yang berbeda: audit kelompok sumber/objek, perbaikan assignment otomatis berbasis bukti train, lalu fine-tuning backbone non-VLM dengan augmentasi yang tidak mengubah material, validasi kelompok, dan ablation terhadap baseline frozen. Jangan memakai arsip visual atau public feedback sebagai target, test untuk fit, VLM, atau data eksternal. Fine-tuning sebelum target diperbaiki dapat memperkuat shortcut latar.

Prioritas berikut: (1) diagnosis kelompok train beradegan sisa makanan yang masuk `textile` dan botol bening yang masuk `glass` memakai fitur/cluster train-only; (2) uji koreksi otomatis pada fold kelompok serta audit 34 acuan manusia tanpa menambah label melampaui 5 per kelas; (3) baru port ke Kaggle GPU dan bandingkan backbone frozen vs fine-tuning. Perubahan yang hanya menaikkan pseudo-label agreement tidak cukup untuk dipilih.

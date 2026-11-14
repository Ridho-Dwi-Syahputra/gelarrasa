# GelarRasa GPU — processed aktif

Buka `DSC2026_GelarRasa_GPU_OneRun.ipynb` di notebook Kaggle milik akun tim. Pasang dataset `arielzzzzzz/gelarrasa`, pilih GPU (T4 x2 didukung), aktifkan Internet untuk bobot torchvision, lalu Run All. Metadata notebook tidak menimpa pengaturan input/accelerator Kaggle.

Seluruh piksel berasal dari dataset panitia yang terpasang. Notebook menyertakan snapshot metadata processed aktif: assignment train algoritmis, source group, 37 seed tim, koordinat crop, eligibility, dan hash. Snapshot itu adalah keluaran alur utama yang sudah dibekukan; fitting seluruh metode clustering historis tidak diulang. Tidak membaca file lokal, arsip anotasi, target test, atau submission pembanding. Kategori sample submission diabaikan.

Sebelum training, notebook membuat ulang gambar processed dan memverifikasi hash sumber. Hash piksel diperiksa persis; selisih kecil lintas lingkungan diperiksa lagi memakai ringkasan warna blok 16x16. Perbedaan visual besar menghentikan proses. Populasi setelah pembaruan holdout adalah 1.888 foto utuh + 7 crop = 1.895 sampel eligible, dengan 18 sumber train ditahan dan 480 test. Angka 1.869 adalah populasi historis, bukan manifest aktif yang digunakan pada port ini. Semua label/folder dan crop mengikuti manifest processed aktif; tidak ada filter confidence 37-seed baru.

Model: ResNet-50, EfficientNetV2-S, ConvNeXt-Small, Swin-V2-Tiny. Tiga fold source-group, 8 epoch/fold; refit final 3 head + 18 fine-tune epoch per model. AMP dan kedua GPU dipakai jika tersedia. Model dilatih/inferensi berurutan agar memori GPU dibebaskan. Ensemble memakai rata-rata probabilitas global. Validasi merupakan agreement terhadap target algoritmis, bukan ground truth semantik independen.

Hasil: `/kaggle/working/gelarrasa_run/DSC2026_Submission.csv`, processed dalam delapan folder kelas, manifest/equivalence report, EDA/preview, source-group folds, OOF/report, log/checkpoint, probabilitas per model, serta `/kaggle/working/GelarRasa_GPU_run_artifacts.zip`. Resume membaca optimizer/scheduler/scaler/RNG bila working directory yang sama masih tersedia; sesi Kaggle baru tanpa checkpoint tidak melanjutkan run lama.

Dataset private harus dapat diakses penilai agar mereka bisa menjalankan notebook. Port ini tidak mengubah visibility. Verifikasi lokal mencakup rekonstruksi processed dan smoke check CPU; Run All T4 x2 tetap harus dilakukan di Kaggle.

## Model hybrid v3

DINOv3 Base empat blok + RBF-SVM dan group-weighted kNN dipulihkan sebagai baseline. Extra Trees 600 pohon dan XGBoost 360 boosting rounds diuji melalui nested grouped validation dengan PCA fold-train serta HSV/Lab/HOG/LBP. Kandidat individual selalu diekspor. CNN blend global tetap 80/20 hanya dipromosikan jika paired grouped OOF tidak turun pada agreement dan macro-F1. Notebook tidak membaca target CSV pembanding untuk fitting/seleksi, dan hasil selisih historis tidak dijamin sama dengan run baru. Cetakan versi: `active_processed_hybrid_v3`.

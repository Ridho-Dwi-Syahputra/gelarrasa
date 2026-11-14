# Kaggle hybrid v3

Artefak: final-submit/ekperimen-kaggle/DSC2026_GelarRasa_GPU_OneRun.ipynb.

Pembanding historis final-submit/output/submission_014_train_group_knn.csv dan final_submission.csv masing-masing berbeda 10/480 baris terhadap docs/submission_ (1).csv saat audit setelah pembekuan. Jalurnya DINOv3 Base descriptor empat blok, RBF-SVM dan inverse-source-group kNN. Jumlah itu tidak dijanjikan untuk run baru: populasi model lama dan port aktif tidak identik pada tahap pemilihan model, walaupun processed aktif dipertahankan secara utuh oleh port.

Perbaikan dari run Kaggle pertama:
- Hentikan pseudo-label baru 37-seed dan filter confidence yang mengubah 988 label pada 1.650 foto bersama.
- Bekukan seluruh 1.886 target processed aktif serta tujuh crop, pengecualian dan kelompok sumber. Verifikasi 2.400 hash sumber dan 2.393 hasil processed.
- Pisahkan log validation fold dari refit seluruh train; perbaiki MixUp input dan pemilihan classifier head.
- Pulihkan DINOv3 empat blok dengan checkpoint/revisi/hash dan resep native ROI yang sama. Kesamaan descriptor dua sumber tervalidasi dengan selisih maksimum 0.0; fitur tujuh crop dihitung.
- Tambahkan Extra Trees 600 pohon dan XGBoost 360 boosting rounds. PCA hanya di-fit di fit split; warna HSV/Lab/spasial, HOG dan LBP berasal dari piksel sampel. Tidak mengubah label.
- Gunakan inner source-group validation untuk k, bobot tetangga dan blend tree; laporkan outer validation terpisah. Seed split dapat diulang deterministik hanya untuk memastikan semua kelas tersedia dalam fit, bukan untuk mencari skor.
- Simpan setiap kandidat dan pertahankan baseline bila penambahan model tidak memenuhi guard paired OOF. Campuran CNN menggunakan bobot global tetap 80/20, tanpa aturan ID test.

Verifikasi: kode semua sel valid; rekonstruksi processed penuh cocok; empat arsitektur forward; loop head/fine-tune; resume optimizer/scheduler/scaler/RNG; nested SVM/kNN/ExtraTrees/XGBoost pada CPU dengan 96 baris, pohon/round smoke diperkecil; round-trip model XGBoost; format dan urutan 480 baris CSV. Lihat kedua JSON verifikasi pada folder ini. Run GPU produksi T4 x2 dan skor submission baru belum diverifikasi.

Pipeline marker: active_processed_hybrid_v3. Snapshot data marker: active_processed_port_v2 (data tetap sama). Tidak ada target CSV pembanding yang dibaca oleh notebook untuk fitting atau seleksi.

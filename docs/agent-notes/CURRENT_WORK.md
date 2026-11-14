# Status kerja aktif

17berbeda/109sesuai dari126review, prefix `frozen_residual_continuation`. Manual34tetap; tidak memakai test/anotasi arsip sebagai target. Status nol: False.

Tidak ada ekstraksi DINO atau publisher yang perlu diulang setelah status publication passed. Cache1904DINOv3Base lengkap. Lihat SOURCE_METHOD_REFINEMENT_RESULT.md dan json/current_review_status.json untuk bukti aktif. Notebook13/13,0error,16visual,4793hash; RunAll assignment snapshot+crop, belumrefitsemuaheads/classifier/CSV. Jangan mengklaim seluruh train/test benar.

Riset lanjutan harus memakai34acuan yang sama dan train-only, freezeplan sebelumfit, guard sebelumaudit, lalu publisher/RunAll/validator, baru audit. Jangan memilih perubahan perID dari sisa review atau menghapus foto sulit. Kuota metal2/food4tidak dapat dipindah ke kelas lain. Tidak ada spawnagent/GUI/push/uploadtanpaotorisasi.

## Fine-tuning Tahap 2 dan batas sumber ? 10 November 2026

Notebook final tidak membaca `submission_.csv`, tabel label test terpisah, atau `SampleSubmission_DSC.csv`. ID submission dibentuk dari nama file di folder test. Gambar train/test yang dibaca berasal dari `data/Dataset DSC GelarRasa 2026`; tidak ada gambar dari dataset eksternal. Bagian akhir notebook berjudul ?Fine-tuning Tahap 2?; pemilihan pseudo-label berjalan otomatis dari disagreement model dan seluruh baris selain query terpilih tetap dari kandidat HSV. Run All terbaru 22/22 sel tanpa error; output kandidat `submission_005_dino_hsv_svm_finetune_stage2.csv`, diagnostik `test_finetune_stage2_diagnostics.csv`.

Dependency turunan yang masih dibaca: `data/preprocessing_inputs.npz`, snapshot assignment/group/reference feature dari eksperimen train terdahulu; bukan gambar eksternal tetapi tidak dibangun ulang oleh notebook saat ini. Encoder memakai checkpoint pretrained DINOv2 yang diizinkan aturan kompetisi, tersimpan/cache dari Hugging Face. Karena itu klaim tepatnya: sumber gambar kompetisi hanya folder dataset dan tidak ada label submission test dari luar; preprocessing snapshot dan bobot model tetap dependency tambahan. Untuk tafsir literal ?hanya dataset + artefak yang dibuat notebook pada Run All?, pipeline preprocessing/assignment dan pemuatan model perlu dibangun ulang tanpa dependency tersebut.

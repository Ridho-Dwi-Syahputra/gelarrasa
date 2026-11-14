# Rekonsiliasi sisa label dengan cross-fit kelompok — 9 November 2026

## Tujuan dan persetujuan

Peserta melaporkan bahwa panitia telah mengizinkan pendekatan ini: melatih model pada label train yang sudah cocok dengan referensi peserta, memprediksi sisa ID, memakai kelas referensi hanya setelah prediksi dibuat, lalu mempertahankan prediksi yang cocok dan menahan kasus lain dari training. Tidak ada artefak tertulis panitia baru yang disimpan di proyek; persetujuan dicatat sebagai klarifikasi peserta pada percakapan 9 November.

## Prosedur yang dijalankan

1. Dari 1.887 foto utuh train yang dapat dibandingkan, 1.799 label program cocok dengan kelas referensi setelah 13 pengecualian peserta diterapkan. Tiga belas label program tersebut dipertahankan.
2. Tiga model DINOv2 dilatih silang lima fold berdasarkan source group: logistic regression, SVM RBF, dan kNN-7 cosine. Saat memprediksi satu fold, seluruh group sumber kandidat dikeluarkan dari training. Fitur train-only; tidak ada test dalam fit.
3. Dari 88 kandidat residual, 71 prediksi mayoritas cocok dengan referensi. Satu di antaranya hanya menang lewat tie-break tiga kelas berbeda, tanpa dukungan suara mayoritas, sehingga ditolak.
4. **70** gambar memiliki setidaknya dua suara model yang sama dan prediksi itu cocok dengan kelas referensi. Labelnya diperbarui ke kelas prediksi konsensus.
5. **18** gambar tidak memenuhi syarat itu; gambar dipindah di dalam processed ke `review/excluded_train` dan `training_eligible=False`. Sumber mentah tetap utuh.

Hasil: semua **1.869 foto utuh yang tetap eligible** sesuai dengan referensi audit yang telah memperhitungkan 13 pengecualian. Terhadap arsip mentah tanpa pengecualian ada 13 perbedaan yang disengaja oleh peserta. Ini agreement dengan referensi, bukan bukti Accuracy resmi. Delapan belas ID residual yang belum memperoleh dukungan cukup tidak dipakai untuk melatih classifier.

Jumlah train eligible kini **1.876 sampel**, termasuk tujuh crop; terdiri dari 1.869 foto utuh dan 7 crop. Total train yang ditahan **37**: 19 penahanan sebelumnya dan 18 hasil cross-fit ini. Tiga CSV lokal dibuat ulang setelah perubahan.

## Artefak dan reproducibility

- Notebook utama: `final-submit/DSC2026_Notebook.ipynb`; sel rekonsiliasi mengulang 5-fold cross-fit dan keputusan 70/18.
- Ringkasan per gambar: `data/processed/group_oof_alignment_evidence.csv`.
- OOF cross-fit pra-integrasi: `table-csv/archive_aligned_residual_oof.csv`.
- Probe: `archive_aligned_residual_probe.py`; sumber sel yang disisipkan notebook: `group_oof_alignment_cell.py`.
- Notebook Run All latar belakang: 20/20 sel kode selesai, nol error, 19 gambar visual. Tiga submission CSV dan OOF/model report dibuat.

## Batas interpretasi

Angka cocok tidak membuktikan semua label semantik benar. Referensi berasal dari anotasi visual lama yang diberikan peserta; penggunaan filter ini berdasarkan izin panitia sebagaimana dilaporkan peserta, bukan jawaban panitia yang diverifikasi langsung dalam artefak proyek. Hasil tidak boleh disebut skor Accuracy. Salinan notebook memuat tabel kelas referensi kandidat agar Run All lokal deterministik; akses dan izin penggunaan tabel tersebut perlu tersedia untuk rerun panitia.

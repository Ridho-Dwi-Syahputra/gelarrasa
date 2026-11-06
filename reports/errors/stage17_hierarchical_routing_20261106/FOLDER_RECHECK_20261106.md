# Pemeriksaan ulang folder aktual — 6 November 2026

Seluruh 1.920 pasangan gambar dihitung ulang langsung dari direktori, bukan dari angka audit sebelumnya atau kolom prediksi saja. Terhadap folder `data/Dataset DSC GelarRasa 2026/train/label`, hasilnya tetap **1.740 ID dengan nama kategori sama dan 180 berbeda**. Pada 570 anggota kelompok campuran lama: 487 sama dan 83 berbeda.

## Pembagian folder hasil yang benar-benar ada

| Folder hasil | Jumlah gambar hasil | Sama ID dan kategori dengan folder arsip |
|---|---:|---:|
| glass | 204 | 193 |
| paper | 243 | 184 |
| electronic | 208 | 194 |
| metal | 250 | 227 |
| plastic | 276 | 250 |
| cardboard | 247 | 237 |
| textile | 225 | 210 |
| foodwaste | 267 | 245 |
| **Total** | **1.920** | **1.740** |

Tidak ada ID hilang, ekstra, atau berulang di folder hasil maupun folder arsip. Seluruh 1.920 penempatan sesuai manifest; hash file keluaran cocok dengan native hasil preprocessing. Seluruh 1.920 file arsip cocok byte-nya dengan sumber awal dan menghasilkan piksel yang sama setelah normalisasi EXIF/transparansi yang terdokumentasi.

## Perbedaan antara dua sumber arsip

Folder arsip dan CSV lama `docs/train_label_review/train_labels.csv` tidak sepenuhnya sama: **0026** berada di folder `metal`, sedangkan CSV lama mencatat `cardboard`. Model memprediksi `metal`. Akibatnya, perbandingan dengan **CSV lama adalah 1.739 sama / 181 berbeda**. Angka 1.740 yang sebelumnya dilaporkan benar untuk folder arsip saat ini. Penyebab perubahan kategori antara kedua sumber belum ditetapkan; tidak ada sumber yang diubah dalam pemeriksaan ini.

## Kasus transparansi 1568

Pemeriksaan awal menggunakan konversi RGB langsung dan berhenti pada ID1568. File berekstensi JPG itu sebenarnya PNG RGBA. Preprocessing yang sudah ada menggabungkan transparansi dengan latar putih; konversi RGB langsung pada auditor pertama mengabaikan alpha. Setelah pemeriksaan menerapkan kebijakan yang sama secara independen, seluruh 1.920 pasangan lolos. Ini koreksi metode audit, bukan perubahan foto, label, atau pipeline preprocessing.

## Preview dan batas klaim

Dua contact sheet menampilkan 32 anggota deterministik: empat posisi ID terurut per folder hasil. Preview dilihat untuk memeriksa tampilan pembagian dan caption yang berasal dari label yang sudah ada, tanpa anotasi atau koreksi kategori baru. Galeri lokal memuat seluruh 1.920 gambar, dengan label hasil dan arsip serta penanda perbedaan. Contoh perbedaan yang tetap ada pada kedua sumber: 1332 hasil `electronic`, arsip `textile`; 1885 hasil `foodwaste`, arsip `paper`; 1918 hasil `plastic`, arsip `electronic`.

Kesamaan dengan arsip **bukan verifikasi bahwa 1.740 gambar semuanya memiliki kategori semantik yang benar** dan bukan Accuracy resmi kompetisi. Seluruh perbandingan tetap audit pribadi, bukan target training atau koreksi model.

Artefak:

- [Bukti hitung ulang dan pemeriksaan isi gambar](independent_folder_recheck_20261106.json).
- [Jumlah per kelas](recheck_class_counts_20261106.csv) dan [matriks perbandingan](recheck_confusion_20261106.csv).
- Galeri lengkap lokal: `data/processed/stage17_named_predictions_v1/73996d8db5/private_audit_only/folder_comparison_gallery.html`.
- Daftar per gambar lokal: `data/processed/stage17_named_predictions_v1/73996d8db5/private_audit_only/independent_folder_recheck_20261106.csv`.
- Kode pemeriksaan: `eksperimen/skrip/recheck_stage17_folders.py`. Tidak melatih model, memindah gambar, mengganti kategori, atau memanggil LLM/VLM.

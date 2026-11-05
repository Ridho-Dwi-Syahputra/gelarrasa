# Label visual train GELAR RASA

Selesai: seluruh 1.920 gambar train dibuka satu per satu dan diberi label berdasarkan pengamatan visual asisten. Tidak menggunakan classifier otomatis untuk menentukan label. Pratinjau per gambar berukuran maksimum 384 piksel; gambar ambigu dibuka pada resolusi asli. Sebanyak 52 gambar ditinjau ulang pada resolusi asli.

Gambar berlabel merupakan salinan pada `data/Dataset DSC GelarRasa 2026/train/label/<kategori>/`. Seluruh gambar asli tetap berada di lokasi semula. SHA-256 setiap gambar asli dan salinannya telah dicocokkan dengan manifest awal.

| Kategori | Jumlah |
|---|---:|
| glass | 212 |
| paper | 215 |
| electronic | 211 |
| metal | 238 |
| plastic | 304 |
| cardboard | 273 |
| textile | 219 |
| foodwaste | 248 |
| **Total** | **1920** |

## Dasar label dan batas kepastian

Label mengikuti objek limbah utama yang terlihat serta materialnya. Pada gambar campuran, objek utama menjadi dasar keputusan; bahan pembungkus sekunder dicatat jika relevan. Label tidak dipaksakan mengikuti jumlah kelas contoh pada materi. Ini adalah anotasi visual, bukan label resmi panitia atau jaminan kebenaran.

Keyakinan subjektif: 1.869 high, 49 medium, 2 low. Nilai ini bukan probabilitas terkalibrasi. Semua gambar sudah ditempatkan pada satu dari delapan folder, termasuk label sementara yang meragukan. Dua prioritas pemeriksaan manusia: `0376` (glass; gambar bergaya ilustrasi/still-life) dan `1890` (electronic; bentuk perangkat atau lemari penyimpanan belum dapat dipastikan). Daftar lengkap 51 label yang masih meragukan tersedia pada `labels_perlu_tinjauan.csv`.

## Berkas audit

- `train_labels.csv`: label akhir setiap gambar.
- `train_labels_audit.csv`: label, alasan visual, tingkat keyakinan, lokasi, dan hash.
- `original_resolution_second_review.csv`: 52 gambar yang telah ditinjau ulang pada resolusi asli.
- `corrections.jsonl`: riwayat koreksi, termasuk 1156 plastic -> paper dan 1176 plastic -> cardboard.
- `decisions_first_pass.jsonl`: catatan sebelum koreksi akhir.
- `validation.json` dan `final_audit.json`: pemeriksaan kelengkapan serta keutuhan berkas.

Satu kelompok gambar identik secara byte ditemukan; label dalam kelompok tersebut konsisten. Kemiripan visual lain belum diaudit secara menyeluruh. Saat menyiapkan validasi model, hindari menempatkan gambar duplikat pada train dan validation sekaligus.

Artefak ini hanya untuk pelabelan train; tidak mengubah submission test.

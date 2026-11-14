# Pembersihan 17 sumber train yang disengketakan

Pada 8 November 2026, peserta meminta 17 sumber train yang masih berbeda dari review visual tim dikeluarkan dari pelatihan classifier. Ini adalah **eksklusi sumber**, bukan penggantian kategori atau tambahan label manusia. Foto asli dan PNG hasil preprocessing tetap ada agar keputusan bisa diaudit. PNG yang ditahan ditempatkan di `data/processed/review/excluded_train`, di luar delapan folder `data/processed/train`. Manifest `data/processed/manifest.csv` adalah daftar sampel untuk loader: pilih `split == train` dan `training_eligible == True`.

ID sumber yang dikeluarkan: 0058, 0265, 0310, 0498, 0559, 0964, 0978, 1018, 1142, 1281, 1436, 1515, 1531, 1657, 1848, 1904, 1913. Dua sumber yang sudah ditahan sebelumnya, 1662 dan 1885, tetap ditahan. Daftar 17 tersimpan dalam `data/preprocessing_inputs.npz` sebagai `train_cleaning_exclusion_keys`, sedangkan `deferred_train_keys` berisi seluruh 19 sumber yang tidak boleh dipakai untuk fit classifier. Tidak ada kategori review yang ditulis sebagai target baru.

Dasar aturan: [jawaban panitia tentang noise data](https://padlet.com/gelarrasa/qna-data-science-competition-gelar-rasa-2026-s023u77x9kd3f0kznkvk/wish/PVKBQO0wN9pkZj5x), 5 November 2026 pukul 23:18 WIB, mengizinkan penghapusan data ambigu/noisy pada tahap cleaning. Jawaban itu tidak secara eksplisit membahas penghitungan kuota koreksi label; oleh karena itu jumlah label manusia tetap 34 dan tidak bertambah. Pemilihan 17 ID berasal dari arahan peserta, bukan dari kemiripan dengan test.

## Cakupan data test sebagai audit terpisah

Pencarian DINOv2 beku terhadap 480 test menunjukkan tidak satu pun test memiliki salah satu dari 17 sumber ini sebagai tetangga train **nomor satu** di antara 1.906 gambar train. Tidak ada pasangan piksel identik di 51 pasangan top-3 yang diperiksa, dan tidak ada top-1 pHash berjarak <=12/64; cosine top-1 maksimum 0,699. Ini tidak membuktikan bahwa jenis objek serupa tidak ada di test. Beberapa tampilan kemasan masih tampak serupa, misalnya train 0559/1515 dengan test 112 serta train 1531 dengan test 085. Audit test ini **tidak** dipakai memilih mask eksklusi, target, atau hyperparameter training.

Galeri: [17 pasangan visual](remaining_visual_review/remaining17_test_neighbors.html). Tabel: [pasangan top-3](table-csv/remaining17_train_test_similarity.csv).

# Analisis visual tahap 3 ? 6 November 2026

## Cakupan dan asal nama

Audit mencakup seluruh 1.920 train dan 480 test. Preview notebook menampilkan contoh train, lebih banyak contoh test, kasus ketajaman rendah, tekstur/latar ramai, penampilan jauh dari train, dan hasil model ambigu. Galeri 15 halaman mencakup seluruh 480 test, satu kali per gambar.

Atas permintaan pengguna, asisten meninjau 128 contoh train yang dipilih dari kelompok hasil model: enam probabilitas tertinggi, enam terendah, dan empat acak per kelompok. Nama adalah interpretasi visual untuk EDA pribadi. Ini bukan ground truth atau pemetaan kategori yang disahkan untuk kompetisi. Aturan yang diberikan pengguna melarang klasifikasi kompetisi menggunakan LLM/VLM; konfigurasi menandai nama tampilan tidak boleh dijadikan target training/validasi/submission. Tidak ada arsip anotasi lama yang dibaca dan tidak ada panggilan LLM/API inference dalam notebook.

Enam kelompok memiliki tema dominan electronic, cardboard, foodwaste, textile, metal, dan paper. Dua kelompok lain ditulis campuran plastic/glass dan plastic/textile/paper. Contoh probabilitas tinggi pun menunjukkan campuran, sehingga mengganti delapan nomor dengan delapan nama tunggal akan memberikan kesan keliru bahwa kaca sudah terpisah dari plastik. Tabel nama tampilan menghitung keanggotaan model; jumlah kelas resmi tetap tidak tersedia.

## Keputusan cropping

Metode memakai jarak warna Lab dari statistik tepi gambar, connected components, dan hipotesis watershed berbasis distance transform untuk region yang bersentuhan. Tidak ada klik/mask/koreksi per gambar. Koordinat dikembalikan ke piksel native dan setiap PNG crop diuji terhadap potongan sumber.

Hasil screening: 890 train dan 246 test memiliki setidaknya dua region yang memenuhi batas geometri. Ini bukan bukti jumlah objek atau label berbeda. Terdapat 2.171 crop train dan 627 crop test untuk preview. Hanya 174 crop dari 68 gambar train memenuhi saringan konservatif connected-components pada latar relatif seragam; sisanya termasuk hipotesis pemisahan yang lebih berisiko. Semua tetap berstatus kandidat tanpa label dan belum dipromosikan ke training.

Preview dipisahkan antara region terpisah dan watershed. Watershed tampak dapat membelah benda memanjang atau satu objek bertekstur menjadi potongan. Karena itu, crop massal dan pewarisan label induk tidak diterapkan. Kotak region hanya usulan algoritma, bukan anotasi objek.

Probe 96 crop train dan 48 crop test memakai DINOv2 serta head baseline yang sama, tanpa melatih ulang bobot. Sebanyak 13/96 train dan 6/48 test berubah kelompok dibanding gambar penuh; 93/96 train dan 48/48 test stabil terhadap flip. Perubahan kelompok bukan jumlah koreksi benar; konsistensi juga bukan Accuracy. Gambar multiobjek berbeda material justru dapat membuat label induk tidak berlaku untuk setiap crop.

Setiap turunan mencatat parent ID, hash sumber, kelompok sumber, fold induk, bbox, ukuran dan hash crop. Semua turunan satu kelompok mengikuti fold yang sama. Crop test berada di folder preview tersendiri dan tidak menjadi tambahan train. Untuk eksperimen classifier berikutnya, pilih transform pada train/validasi yang sesuai aturan dan bekukan keputusan sebelum inferensi test. Tugas test tetap satu label per gambar penuh.

## Piksel dan ukuran

RGB MSE 64?64 membandingkan seluruh 480?1.920 pasangan test?train, langsung dan horizontal flip. SSIM grayscale 128?128 kemudian dihitung pada gabungan shortlist RGB, flip, pHash dan embedding. SSIM tidak dihitung untuk seluruh pasangan; skor tetangga terbaik hanya terbaik pada shortlist. Resize full-frame dapat mengubah aspect ratio, sehingga ukuran native dan selisih rasio turut disimpan. MAE native, fraksi piksel RGB identik dan selisih kanal maksimum dihitung setelah alignment hanya jika ukuran native sama.

Terdapat tiga pasangan hash piksel identik:

- test:017 ? train:1046, 800?533.
- test:042 ? train:1821, 640?640.
- test:201 ? train:1135, 674?900.

Sebanyak 29 test memiliki SSIM shortlist >=0,99. Dari jumlah itu, 19 memenuhi saringan tambahan MSE RGB <=0,0002 pada alignment terpilih dan selisih log aspect ratio <=0,03. Ambang ini hanya saringan kandidat; bukan kalibrasi probabilitas duplikat dan bukan pembuktian label. Preview menunjukkan pasangan yang sangat dekat sesudah flip atau resize meskipun hash byte/piksel native berbeda.

Ukuran 439/480 test muncul pada setidaknya satu train. Resolusi 416?416 mencakup 240/480 test (50%) dan 579/1.920 train (30,16%); 640?640 mencakup 136/480 test (28,33%) dan 700/1.920 train (36,46%). Ini indikasi perbedaan komposisi sumber/resize yang perlu diperhatikan. Ukuran sama tidak menghubungkan dua gambar sebagai sumber yang sama. Tabel tambahan menyilangkan ukuran dengan nama tampilan sementara untuk meninjau bias sumber; tidak dipakai sebagai jawaban label.

## Artefak dan batas interpretasi

- Notebook utama: `notebook/01_EDA_Preprocessing_Clustering.ipynb` (v3).
- Ringkasan: `summary.json`; threshold hanya dari train: `quality_thresholds.json`.
- Seluruh pasangan kandidat: `pixel_ssim_candidate_pairs.csv`; hasil per test: `all_test_best_pixel_neighbors.csv`.
- Ketercakupan test: `gallery_coverage.json`; galeri: `galleries/index.html`.
- Manifest crop: `data/processed/stage3_crop_candidates_v1/crop_manifest.csv`.
- Probe view crop: `crop_model_view_probe.csv`; tabel dimensi: `dimensions_by_split.csv` dan `dimensions_by_display_name.csv`.
- Bukti Run All dan pemeriksaan akhir: `reports/reproducibility/notebook_v3_execution.json` dan `notebook_v3_delivery_validation.json`.

Raw, kontrak label aktif, checkpoint, embedding dan assignment baseline dipertahankan. Tidak ada Accuracy kategori atau submission baru. Dataset, gambar terpisah, crop, embedding dan file besar tetap lokal sesuai permintaan pengguna; notebook menyimpan preview tertanam.

## Referensi metode

- [Dokumentasi OpenCV watershed](https://docs.opencv.org/4.x/d3/db4/tutorial_py_watershed.html) membahas pemisahan objek bersentuhan dan risiko oversegmentation.
- [Dokumentasi scikit-image SSIM](https://scikit-image.org/docs/stable/api/skimage.metrics.html#skimage.metrics.structural_similarity) menjelaskan kebutuhan data range serta parameter perbandingan struktur.

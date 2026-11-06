# Pemisahan kelompok campuran — 6 November 2026

Tujuan pengguna adalah mendekatkan pengelompokan algoritmis pada arsip pembanding, khususnya 570 train yang sebelumnya hanya mendapat nama kelompok campuran. Hasil baru memberikan satu prediksi kategori per gambar, tetapi tidak menganggap nama folder sebagai bukti semua isinya benar.

## Pengujian yang benar-benar dilakukan

1. CNN objek ImageNet ResNet50 dan CNN material MINC GoogLeNet: dua skema fusi dan head lokal. Fusi mentah gagal karena skala probabilitas 1.000 kelas objek tidak sebanding dengan 23 kelas bahan. Versi bersyarat mempertahankan cakupan bukti; kategori elektronik/kardus memakai bukti objek yang relevan.
2. Seleksi seed konservatif dari kesepakatan sumber, batas jumlah seed, serta head linear/MLP. Tidak memakai arsip anotasi untuk label, filter per gambar, ataupun pemetaan kategori.
3. Pilot pelatihan blok terakhir DINOv2: dua learning rate, empat epoch per konfigurasi, 512 fit dan 381 heldout; gradien dan perubahan bobot dibuktikan. Kontrol epoch nol menang. Pilot tidak dipakai untuk mengklaim backbone akhir sudah di-fine-tune.
4. CNN ResNet50 khusus limbah sebagai basis kedua, kemudian enam head lokal pada fitur CNN, DINO dan gabungannya. Hasil awal lebih buruk pada pembanding; eksperimen negatif tetap dicatat.
5. Interpretasi kelompok anonim melalui agregat bukti CNN pada fit, pemetaan Hungarian satu-ke-satu, kemudian anchor yang diperluas oleh kesepakatan sumber kuat. Pemetaan tidak memaksakan jumlah prediksi per kategori.
6. Bank 24 SVM/kNN dengan fitur global/lokal. Proksi internal meningkat, tetapi perbandingan arsip memburuk: bukti bahwa proksi tidak sama dengan ketepatan material.
7. ConvNeXt-Tiny ImageNet22k dengan ontologi publik WordNet dan enam head lokal. Model ini menambah informasi objek dan meningkatkan hasil perbandingan. Data WordNet adalah metadata leksikal eksternal yang dinyatakan terbuka, bukan citra pelatihan tambahan. Tidak ada klaim bahwa semua interpretasi aturan untuk eksperimen ini sudah disahkan panitia.
8. Delapan belas konfigurasi difusi graf fit-only. Heldout dan test tidak masuk graf. Tidak mengalahkan kontrol pada kriteria yang dibekukan, sehingga kontrol dipertahankan.
9. EfficientNetV2-S khusus limbah sebagai sumber CNN kedua. Enam head dan 205 fit-anchor tambahan tidak mengalahkan kontrol stage12 pada proksi yang sama. Bobot tersedia publik, tetapi lisensi/dataset sumber tidak dinyatakan pembuat; tidak mengunggah atau mendistribusikan bobotnya.
10. Delapan classifier dengan target murni kesepakatan sedikitnya dua keluarga sumber CNN, tanpa target dari cluster awal. Akurasi terhadap target turunan pada heldout naik, dan 487 dari 570 kelompok lama cocok dengan arsip; tetapi 123 anggota stabil lama menjadi berbeda. Hasil negatif pada kohort stabil tidak disembunyikan.
11. Pembagian bertingkat: dari fit saja, ukur proporsi kategori dominan menurut anchor CNN pada setiap kelompok anonim. Kelompok dengan sedikitnya 20 anchor dan proporsi dominan<0,6 memakai classifier stage16; sisanya memakai stage12. Aturan tunggal ditulis sebelum audit run ini, tidak mencari threshold pada arsip atau membaca flag kohort campuran. Statistik fit memilih dua kelompok dan merutekan 570 train. Model bertingkat adalah hipotesis struktural; bukan pemenang proksi stage16 yang sama dan belum pemenang Accuracy resmi.

Semua backbone adalah image-only non-VLM. Pelatihan classifier lokal menggunakan train kompetisi, dengan target turunan dari model/struktur graf. Tidak mengunduh dataset gambar eksternal, menjalankan kode remote, atau memanggil LLM dalam notebook. Bobot CNN khusus limbah telah dilatih pihak lain; bukan model yang dilatih dari nol oleh tim. Irisan pretraining dengan data lomba belum diketahui dan hasil bukan evaluasi independen terhadap ground truth.

## Hasil pembanding setelah prediksi dibekukan

| Tahap | Total train berbeda dari arsip | Dari 570 lama: cocok | Dari 570 lama: berbeda |
|---|---:|---:|---:|
| Student awal, sebelum pemisahan | 164 pada 1.350 yang bisa dibandingkan | belum dapat dibandingkan | belum dapat dibandingkan |
| CNN limbah + head, stage7 | 547 | 416 | 154 |
| Anchor kelompok stabil, stage8 | 400 | 306 | 264 |
| Anchor kesepakatan kuat, stage9 | 260 | 430 | 140 |
| Bank SVM/kNN, stage10 | 273 | 431 | 139 |
| CNN22k + ontologi + head, stage12 | 226 | 441 | 129 |
| Difusi graf, stage13: kontrol stage12 dipertahankan | 226 | 441 | 129 |
| CNN limbah kedua, stage15: kontrol stage12 dipertahankan | 226 | 441 | 129 |
| Konsensus tanpa target cluster awal, stage16 | 254 | 487 | 83 |
| Pembagian bertingkat, stage17 | **180** | **487** | **83** |

Angka awal 164 **tidak setara** dengan total 226: awalnya 570 gambar dikeluarkan dari perbandingan kategori tunggal. Pada stage12 seluruh 1.920 train dapat dibandingkan; 84 kesalahan lama menjadi cocok, 17 yang tadinya cocok menjadi berbeda, dan 129 dari kelompok campuran masih berbeda. Persamaan cek: `(164 - 84 + 17) + 129 = 226`.

Pada hasil bertingkat stage17, kohort stabil mempertahankan angka stage12 dan kohort campuran mendapat pemisahan stage16: `(164 - 84 + 17) + 83 = 180`. Seluruh1.920 train memiliki prediksi; **1.740 cocok dan 180 berbeda**. Perbaikan agreement pembanding bukan bukti label sempurna. Folder lokal terkini: `data/processed/stage17_named_predictions_v1/73996d8db5/{train,test}/{nama_kategori}/`. Gambar 0046 diprediksi paper dan tidak ditandai perlu tinjau.

Arsip tetap pembanding pribadi, bukan ground truth resmi, target validasi, atau ukuran Accuracy kompetisi. Peneliti telah melihat audit terdahulu dan pengembangan metode berlangsung iteratif, sehingga rangkaian eksperimen ini tidak buta terhadap pembanding. Dalam setiap run, pilihan checkpoint menggunakan proksi anchor/view/tetangga yang ditetapkan sebelum audit run tersebut. Klaim tidak memakai arsip untuk pemilihan hanya berlaku untuk pemilihan kandidat di dalam run, bukan klaim bahwa seluruh riset dilakukan tanpa mengetahui feedback lama.

## Interpretasi folder dan ketidakpastian

- Setiap train/test punya satu prediksi pada salah satu dari delapan nama resmi; tidak ada lagi keluaran bernama `campuran`.
- `mixed_output_names=0` berarti cakupan prediksi lengkap. Itu **tidak** berarti nol gambar ambigu, nol kesalahan, atau ketepatan semantik 100%.
- Stage12 masih menandai 605 train dan 138 test untuk tinjau; tanda merupakan gabungan margin kecil, softmax rendah, atau konflik dengan CNN sumber. Sebagian tanda berasal dari kekeliruan sumber CNN; jumlahnya bukan jumlah salah terverifikasi.
- Stage17 menandai570 train untuk tinjau:126 memiliki softmax/margin rendah dan 444 hanya konflik sumber CNN. Angka570 ini kebetulan sama dengan ukuran kohort campuran lama, tetapi merujuk kriteria dan anggota yang berbeda. Bukan570 gambar yang masih tidak mempunyai kategori. Test:25 keyakinan/margin rendah dan 101 konflik sumber saja. Lihat [rincian alasan](../stage17_hierarchical_routing_20261106/review_reason_breakdown.csv).
- Tidak menghapus gambar ambigu. Semua ID asli dipertahankan; duplikat piksel mengikuti kebijakan split sebelumnya.
- Versi materialisasi baru memakai hardlink read-only ke salinan independen stage7 untuk menghemat disk. Bukan hardlink ke raw/native sumber. Jangan mengedit gambar versi prediksi in-place; buat salinan jika akan melakukan augmentasi.
- Kontrak label aktif dan notebook EDA utama tidak diganti. Kandidat baru tetap eksperimen, bukan pengesahan dataset berlabel atau rekomendasi submission sempurna.

## Bukti dan reproduksi

Pointer model saat ini: [selected_material_run.json](selected_material_run.json). Bukti materialisasi, hash, reproduksi probabilitas dan audit terbaru: [final_validation_and_private_audit.json](final_validation_and_private_audit.json). Setiap run juga menyimpan rencana sebelum fitting, perkembangan metrik, checkpoint lokal dan pembekuan assignment sebelum pembacaan arsip.

Ringkasan sumber dan dependensi terperinci: [METHOD_AND_REPRODUCTION](../stage7_waste_cnn_20261106/METHOD_AND_REPRODUCTION.md). Tahap6 MINC membutuhkan OpenCV4 terisolasi karena binding Caffe tidak tersedia pada OpenCV5 utama. Corpus WordNet berada pada cache pengguna, bukan dataset lomba. Checkpoint, embedding dan gambar tidak diunggah ke Git.

Urutan baru: `stage6_*` → `stage7_*` → `stage8_*` → `stage9_*` → `stage10_*` → `stage11_*` → `stage12_*` → `stage13_*`. Ikuti metadata checkpoint yang dipilih; tidak perlu mengulang eksperimen negatif untuk sekadar melihat preview. `stage6_finalize_predictions.py` memverifikasi ulang probabilitas, hash dan penempatan seluruh 2.400 gambar, lalu melakukan audit pribadi terpisah. Notebook pendamping dibuat/dijalankan oleh `stage6_build_review_notebook.py` dan tidak membaca audit arsip.

Lanjutan aktual: `stage14_second_waste_cnn.py` → `stage15_second_cnn_refinement.py` → `stage16_source_consensus.py` → `stage17_hierarchical_routing.py` → finalizer → pembuat notebook. Stage17 memakai checkpoint12 dan16, sehingga seluruh eksperimen negatif tidak diperlukan untuk inferensi apabila checkpoint dan fitur tersebut sudah tersedia. [Notebook pendamping](../../../notebook/02_Material_Resolution_Review.ipynb) mempunyai empat sel kode dengan preview tertanam; bukti eksekusi ada di `reports/reproducibility/notebook_material_resolution_execution.json`. Notebook 01 tetap utuh sebagai EDA historis yang sudah lulus 10/10.

## Landasan metode dan sumber primer

- [SCAN — penulis dan implementasi resmi](https://github.com/wvangansbeke/Unsupervised-Classification): representasi dan clustering head dipisahkan; konsistensi tetangga bukan jaminan kesamaan bahan.
- [SPICE — paper penulis](https://arxiv.org/abs/2103.09382): prototype dan pseudo-label yang andal, bukan menerima semua argmax sebagai label bersih.
- [Learning with Local and Global Consistency](https://proceedings.neurips.cc/paper/2003/hash/87682805257e619d49b8e0dfdc14affa-Abstract.html): inspirasi difusi graf; implementasi lokal menambah ekstensi induktif fit-only dan bukan klaim reproduksi seluruh eksperimen paper.
- [MINC — Cornell](https://opensurfaces.cs.cornell.edu/publications/minc/): bobot CNN bahan, kelas dan preprocessing.
- [ConvNeXt — penulis](https://github.com/facebookresearch/ConvNeXt) dan [checkpoint timm ImageNet22k](https://huggingface.co/timm/convnext_tiny.fb_in22k): arsitektur dan bobot image-only.
- [ResNet50 limbah — pembuat checkpoint](https://huggingface.co/kendrickfff/my_resnet50_garbage_classificationv1.2): sumber bobot khusus domain; urutan kelas diverifikasi dari aplikasi pembuat yang dipin, tidak dieksekusi.
- [EfficientNetV2 limbah — handler pembuat](https://huggingface.co/attilaultzindur/garbage_classifier_effnetv2s_ft/blob/35f28e0f432e4d874d88bb214da6cb96553f10fd/handler.py): eksperimen sumber kedua; lisensi dan dataset pretraining tidak dinyatakan pada model card, keterbatasan provenance dicatat.

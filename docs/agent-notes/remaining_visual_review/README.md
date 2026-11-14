# 150 perbedaan kategori: visualisasi dan cakupan test

Seluruh gambar berada pada remaining_01.png sampai remaining_08.png, urut ID numerik. Tujuh lembar pertama masing-masing20foto; lembar terakhir10foto. Judul menampilkan kategori program aktif dan kategori arsip pembanding. Keduanya bukan ground truth resmi.

index.html menampilkan150train beserta masing-masing3test terdekat dalam embedding DINOv2. Buka file tersebut untuk melihat foto asli dengan ukuran lebih besar; galeri tidak dibuka otomatis oleh script. train_test_examples.png memuat delapan contoh pasangan, termasuk satu kedekatan lemah untuk menunjukkan batas pencarian.

## Hasil aktual

- Membandingkan150×480=72.000pasangan. Tidak ada piksel hasil decode yang identik; tidak ada pasangan dengan minimum pHash≤4/64, termasuk flip horizontal/vertikal.
- Tujuh train memiliki tetangga test dengan cosine≥0,80; empat≥0,85; tidak ada≥0,90. Batas ini deskriptif, bukan probabilitas atau hasil kalibrasi kelas.
- Dari480test,20memiliki tetangga train terdekat di dalam kelompok150. Dua puluh test itu berhubungan dengan15IDtrain:0141,0289,0343,0352,0512,0531,0539,0714,0867,1112,1228,1399,1476,1566,1817.
- Jika seluruh150dikeluarkan dari pencarian, kedekatan tetangga test turun≥0,05pada17test dan≥0,10pada10test. Ini penurunan cakupan representasi, bukan estimasi penurunan Accuracy.
- Contoh:0343menjadi tetangga terdekat test076/165;1817untuk153/409;0531untuk380;1112untuk209/452/456;1476untuk225.
- Dimensi asli pasangan embedding terdekat sama pada54/150kasus. Kesamaan ukuran tidak membuktikan kesamaan objek atau bahan.
- pHash tidak boleh dipakai sendiri: pasangan1497–070berjarak8/64tetapi cosine hanya0,060. SSIM pada letterbox juga dapat didominasi latar/padding.

## Informasi baru dan implikasi

Tidak semua perbedaan merupakan kasus tanpa padanan test. Menghapus150sekaligus berpotensi mengurangi variasi yang dicakup train. Kategori program pada gambar yang dipertahankan tetap belum terverifikasi; keberadaan tetangga test tidak memperbaiki kebenaran kategori train.

Perbedaan bersifat terstruktur:27kasus program paper versus arsip cardboard,27paper versus plastic, dan14metal versus paper. Ini menunjukkan bagian representasi/assignment yang patut dievaluasi secara umum pada train, bukan daftar label koreksi. Hipotesis berikutnya dapat menguji representasi lokal/tekstur dan ketahanan terhadap latar, dengan pemilihan metode serta evaluasi hanya dari train. Tidak menetapkan kelas test, menyalin kelas arsip, atau memakai daftar150untuk target/filter pelatihan.

Tidak menemukan tetangga test yang dekat juga tidak membuktikan suatu train tidak berguna: citra tersebut masih dapat menambah variasi dan informasi material. Keputusan cleaning harus mempertimbangkan kualitas kategori dan validasi train, bukan sekadar keberadaan objek serupa di test.

## Bukti

Script: ../show_remaining_train_test.py. Ringkasan: ../json/remaining_visual_review.json. Tabel450pasangan: ../table-csv/remaining_train_test_top3.csv. TabelpHash: ../table-csv/remaining_train_test_phash.csv. Cakupan seluruh480test: ../table-csv/remaining_test_feature_coverage.csv.

630gambar sumber diverifikasi hash; manifest aktif, input snapshot, dan notebook tidak berubah. Tidak ada model fit, label baru, atau jatah manusia yang dipakai. Jatah tetap0/40.

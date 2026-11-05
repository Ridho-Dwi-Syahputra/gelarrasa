---
name: image-domain-shift-audit
description: Membandingkan distribusi train dan test citra tanpa memakai label test, termasuk ukuran, kualitas, sumber visual, embedding, serta potensi shortcut sebelum menetapkan preprocessing.
---

# Pergeseran distribusi dan cakupan

Gunakan manifest seluruh 1.920 train dan 480 test; jumlah dipastikan dari kontrak, bukan jumlah file yang kebetulan berhasil dibaca. Profilkan dimensi, aspect ratio, format/mode, brightness, contrast, blur proxy, dan source_type. Catat definisi indikator serta denominator; threshold blur bergantung resolusi dan tekstur, bukan label buruk universal.

Ambil sampel teknis dari ekor distribusi/cluster program untuk menilai cakupan input. Jangan meminta LLM memberi kategori citra atau memakai kelas anotasi lama sebagai rujukan. Bandingkan sumber, bukan hanya histogram gabungan. Distribusi kelas test tidak diketahui; label visual lama dikecualikan dari analisis kompetisi.

Embedding frozen dengan tag/transform yang sama dapat membantu mencari domain yang belum tercakup. Gunakan PCA/UMAP/t-SNE sebagai navigasi, bukan pembuktian kelas terpisah. Uji tetangga pada embedding asli; jarak di proyeksi 2D dapat menyesatkan. [FiftyOne embedding tutorial](https://docs.voxel51.com/tutorials/image_embeddings.html).

Opsional: classifier domain train-vs-test dengan held-out/group split dan pelaporan AUC khusus **diagnostik domain**, bukan metrik target kompetisi. Akurasi domain tinggi menunjukkan keterbedaan, tidak membuktikan penyebab atau langsung membenarkan importance weighting. Jangan memasukkan label test semu ke analisis ini. Learned preprocessing untuk evaluasi model tetap fit train fold saja; analisis joint train/test diberi label transductive/deskriptif dan tidak dinilai sebagai bukti generalisasi.

Keluaran: tabel slice, galeri, hipotesis transform/sampling yang dapat diuji, dan catatan coverage yang belum diketahui. Jangan menghapus kelompok sulit atau memaksa distribusi label menjadi seimbang agar profil terlihat bagus.

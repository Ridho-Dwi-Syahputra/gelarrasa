---
name: image-error-analysis
description: Membedah kesalahan klasifikasi limbah melalui OOF, pasangan kelas, galeri citra, dan audit latar atau perhatian model sebelum memilih perbaikan.
---

# Analisis kesalahan per gambar

Gunakan probabilitas OOF dari versi target-program/fold sama; anotasi visual asisten lama tidak boleh menjadi target. Sebutkan bahwa agreement terhadap label cluster bukan Accuracy semantik resmi. Bila target yang layak tersedia, tampilkan jumlah disagreement, metrik per kelas dan normalisasi confusion matrix. Jangan memakai KS/AUC biner proyek lama.

Buat diagnostik disagreement, confidence dan kualitas input berdasarkan program. Tinjau mapping/loader/crop, ukuran, blur, background sebagai kemungkinan penyebab. Agent tidak menetapkan label citra dan tidak mengoreksi kategori per gambar. Temuan teknis tidak otomatis membuktikan material yang sebenarnya.

Prioritas pasangan: glass–plastic; paper–cardboard; metal–electronic; textile–paper pada objek menyerupai pakaian. Bandingkan full-frame/crop, resolusi, dan background diagnostic pada ID sama; setiap perubahan harus punya hipotesis.

[Grad-CAM](https://arxiv.org/abs/1610.02391) membantu memilih area untuk diperiksa. Simpan kelas yang dijelaskan dan layer yang dipakai. Heatmap bukan segmentasi objek, bukti material, atau jaminan tidak ada shortcut. Embedding neighbors juga hanya petunjuk.

Untuk dua model, simpan empat kelompok correctness (keduanya benar, hanya A, hanya B, keduanya salah). Ini lebih berguna untuk memilih ensemble daripada korelasi probabilitas saja. Keluaran: `reports/errors/<run_id>/`, tabel per ID, akar masalah yang didukung bukti, dan eksperimen pembeda berikutnya. Jangan menutupi kesalahan dengan mengubah target tanpa riwayat.

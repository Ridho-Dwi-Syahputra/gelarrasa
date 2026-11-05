---
name: classical-image-baseline
description: Membuat baseline citra limbah dari HOG, histogram warna, dan tekstur dengan SVM atau model linear untuk mengukur manfaat fitur klasik sebelum model besar.
---

# Fitur klasik dan SVM

Ikuti [Materi 1 hlm. 14](../../references/WEBINAR_MAP.md) dan fold yang telah dikunci. Gunakan citra RGB yang berorientasi benar. OpenCV membaca BGR: konversi sekali, jangan menukar channel dua kali. Definisikan resize/aspect handling dan normalisasi histogram secara eksplisit.

Bangun tiga baseline: HOG grayscale saja, histogram warna HSV/Lab saja, serta gabungan. LBP dapat ditambah sebagai ablation tekstur bila membingungkan paper/cardboard/textile. Kandidat awal HOG 9 orientations, cell 8×8 atau 16×16; histogram 16–32 bins/channel. Ini titik mulai uji, bukan parameter terbaik.

Gunakan Pipeline StandardScaler → classifier; fit per fold train. Bandingkan linear classifier dengan RBF SVM. Tuning C/gamma di inner validation. Jangan melakukan seleksi fitur atau normalisasi statistik pada seluruh data sebelum CV. Simpan urutan ID/dimensi fitur/konfigurasi dan hash manifest pada cache.

Untuk OOF hard predictions, probabilitas SVM tidak wajib. Jika akan ensemble, gunakan kalibrasi dengan inner split yang menjaga group; jangan menerima CV internal acak yang memisahkan duplikat.

Keluaran: OOF, confusion matrix, runtime, dan contoh gagal. Baseline ini menunjukkan apa yang dapat dijelaskan warna/tekstur; skor rendah bukan alasan menghapusnya dari catatan. Rujukan implementasi umum: scikit-learn vendor, disesuaikan dengan [grouped validation](../grouped-image-validation/SKILL.md).

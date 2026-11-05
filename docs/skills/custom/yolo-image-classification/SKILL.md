---
name: yolo-image-classification
description: Menguji YOLO classification untuk label tingkat gambar dan menilai kapan deteksi atau segmentasi layak ditambahkan pada gambar limbah multiobjek.
---

# YOLO sesuai target

Target GELAR RASA adalah satu label per gambar, tanpa box/mask. Gunakan varian **classification (`-cls`)** yang tersedia pada versi Ultralytics yang dipasang; catat versi dan nama checkpoint, jangan mengambil nama model terbaru dari contoh vendor tanpa pemeriksaan. Lihat [dokumentasi resmi](https://docs.ultralytics.com/tasks/classify/).

Buat folder fold train/val dari manifest yang telah dikunci melalui copy/link staging, bukan memindahkan raw. Verifikasi `names/class_to_idx` dan urutan probabilitas terhadap kontrak. Jangan menjalankan split otomatis yang mengabaikan group.

Audit transform default: dokumentasi memperingatkan RandomResizedCrop/CenterCrop dapat memotong bagian penting pada aspect ratio ekstrem. Bandingkan full-frame padding/resize yang sesuai dengan square-crop pada contoh limbah dan validation. Accuracy top-1 adalah metrik utama, bukan mAP.

Deteksi/segmentasi baru masuk jika error menunjukkan objek relevan kecil atau campuran dan tersedia anotasi box/mask yang memadai. Label image-level tidak berubah menjadi box ground truth hanya dengan memberi seluruh gambar sebuah kotak. Bila memakai detector pretrained untuk proposal crop, cek aturan bobot, dokumentasikan kegagalan crop, dan pertahankan full-frame fallback. Detektor kelas umum tidak otomatis memahami semua material limbah.

Simpan konfigurasi, mapping, OOF, checkpoint, dan perbandingan dengan CNN baseline. Catat lisensi implementasi/bobot saat packaging. Skill ini tidak memerlukan upload dataset ke Roboflow atau training deteksi untuk sekadar klasifikasi.

---
name: vision-augmentation
description: Memilih preprocessing, augmentasi citra, MixUp/CutMix, dan TTA yang mempertahankan informasi material limbah serta dapat dievaluasi terpisah.
---

# Transform yang menjaga makna

Baca [webinar hlm. 11–13 dan 20](../../references/WEBINAR_MAP.md). Selalu split sebelum augmentasi; varian satu sumber berada dalam group sama. Raw tetap utuh; simpan seed dan konfigurasi transform, bukan menimpa gambar.

Baseline: EXIF transpose → RGB → resize/pad atau crop terkontrol → tensor → normalisasi checkpoint. Bandingkan 224 dengan 320/384 hanya setelah mengamati hilangnya detail. Tampilkan pasangan sebelum/sesudah untuk tiap kelas, termasuk kaca transparan, foil, sobekan kertas, dan tekstur kain.

Awali horizontal flip, rotasi kecil, perubahan brightness/contrast ringan. Uji efek color jitter terhadap petunjuk material. Vertical flip, grayscale, blur berat, background removal, atau crop ekstrem bukan default. Jika crop membuang objek utama, label lama dapat menjadi salah.

MixUp: gabungkan input dan target dengan koefisien sama. CutMix: bobot target mengikuti area patch aktual setelah clipping. Uji keduanya terpisah; jangan menumpuk semua augmentasi sekaligus. Rujukan: [MixUp](https://arxiv.org/abs/1710.09412), [CutMix](https://arxiv.org/abs/1905.04899), [Torchvision](https://docs.pytorch.org/vision/stable/transforms.html).

Validation standar deterministik tanpa augmentasi acak. TTA merupakan eksperimen inferensi terpisah, misalnya original+horizontal flip; rata-ratakan probabilitas dengan class order sama. Evaluasi paired terhadap single-view dan catat biaya. Jangan memasukkan variant TTA sebagai sampel validation independen.

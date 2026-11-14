---
name: image-preprocessing-audit
description: Menyiapkan dan membuktikan preprocessing gambar limbah menjaga orientasi, warna, rentang piksel, detail material, dan keterlacakan ke sumber sebelum training.
---

# Preprocessing citra yang dapat diperiksa

Mulai dari [rencana EDA/preprocessing](../../references/EDA_PREPROCESS_PLAN.md). Gunakan sumber asli; preview 384-pixel hanya untuk tinjauan. Decode seluruh gambar, bukan hanya membaca header. Catat kegagalan per ID dan jangan diam-diam menghilangkan file yang gagal.

Terapkan orientasi EXIF satu kali. Catat mode L/P/RGBA/CMYK bila ada, konversi RGB konsisten, dan untuk alpha tentukan warna compositing sebelum membuang channel. Periksa ukuran setelah orientasi. Byte hash dan hash pixel decode mempunyai makna berbeda; perubahan orientasi metadata bukan alasan mengganti label.

Tentukan dtype/range pada setiap batas library. `uint8` 0–255, float 0–1, dan tensor ternormalisasi bukan representasi yang dapat dipertukarkan begitu saja. `astype(float)` tidak membagi piksel dengan 255. Hindari normalisasi dua kali serta pertukaran BGR/RGB ganda. [Pillow ImageOps](https://pillow.readthedocs.io/en/stable/reference/ImageOps.html), [scikit-image dtype](https://scikit-image.org/docs/stable/user_guide/data_types.html).

Bandingkan full-frame aspect-preserving pad dengan resize langsung dan crop terkontrol. Lihat tepi botol, tekstur kain, serta lapisan karton; tetapkan interpolation/antialias dan warna padding. Crop manual/segmentasi harus mencatat koordinat dan alasan. Jangan menerapkan denoise, sharpening, CLAHE, grayscale, atau background removal massal hanya karena namanya preprocessing.

Simpan `source_id, source_hash, transform_id, transform_config, output_path, output_hash`. Cache harus invalid ketika sumber/transform berubah. Turunan berada di `data/processed/` (satu versi aktif; versi sebelumnya diarsipkan); jangan menimpa raw. Dataset validation/test menggunakan transform deterministik yang sesuai checkpoint. Keluaran: audit seluruh file, galeri sebelum/sesudah per kondisi, dan keputusan transform yang didukung bukti.

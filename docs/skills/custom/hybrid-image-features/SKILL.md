---
name: hybrid-image-features
description: Menguji embedding CNN atau ViT dengan fitur HOG, warna, dan classifier klasik melalui ablasi yang menjaga pemisahan fold dan skala tiap blok fitur.
---

# Hybrid fitur visual

Baca [peta webinar](../../references/WEBINAR_MAP.md) dan [tinjauan hybrid](../../references/RESEARCH_REVIEW.md). Gunakan [timm feature extraction](https://huggingface.co/docs/timm/feature_extraction) untuk memastikan bentuk keluaran; dimensi embedding berbeda antar backbone, jangan hardcode 2048 untuk semuanya.

Bandingkan embedding-only, handcrafted-only, concat, serta decision-level averaging pada fold sama. StandardScaler per fitur yang fit di train fold menyetarakan skala koordinat, tetapi tidak otomatis menyetarakan kontribusi blok dengan jumlah dimensi berbeda. Untuk concat uji normalisasi L2 per sampel tiap blok dengan denominator `max(norm, epsilon)` (vektor nol tetap nol), lalu bobot blok eksplisit. Bobot, PCA/feature selection/C/gamma dipilih di inner split. Catat urutan operasi karena scaling setelah L2 dapat mengubah keseimbangan kembali. DINOv2 frozen dapat menjadi pembanding embedding jika aturan bobot jelas; bukan kewajiban.

Cache frozen pretrained features dapat dihitung deterministik per ID tanpa label. Jika backbone di-fine-tune memakai label lokal, kedua sisi fold (train dan validation) harus diekstrak oleh encoder yang sama, dilatih hanya pada fold train. Jangan mencampur embedding dari encoder fold berbeda menjadi satu matriks classifier global: sumbu fiturnya belum tentu sepadan. Dilarang mengekstrak embedding semua train dari backbone full-data lalu menyebut CV downstream sebagai OOF bebas leakage. Cache seperti itu hanya sesuai refit/inferensi, atau analisis in-sample yang diberi label jelas. Lihat [kontrak embedding](../../references/ARTIFACT_CONTRACTS.md).

Classifier kandidat: logistic regression/linear SVM dahulu, RBF SVM bila diperlukan. KNN menjadi diagnostik neighborhood; exclude self dan kelompok sumber yang sama dalam evaluasi. Jangan mengganti label test dengan nearest neighbor yang hanya mirip kelas.

Keluaran: fitur dan provenance, OOF tiap komponen, tabel ablasi, analisis gambar yang diperbaiki/diperburuk, serta biaya. Hybrid dipertahankan hanya bila manfaat bertahan pada evaluasi setara, bukan karena paper mencapai angka tinggi di benchmark lain.

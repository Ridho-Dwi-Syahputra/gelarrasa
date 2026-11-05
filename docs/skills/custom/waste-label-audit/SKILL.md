---
name: waste-label-audit
description: Memeriksa asal label cluster, mapping kategori oleh peserta, dan konsistensi artefak limbah tanpa melabel atau mengoreksi gambar kompetisi secara manual oleh LLM.
---

# Audit provenance label limbah

Gunakan [aturan aktif](../../references/RULES_AND_RUBRIC.md) dan [rancangan clustering](../unsupervised-waste-labeling/SKILL.md). [Definisi kategori](../../references/LABEL_TAXONOMY.md) hanya kamus material untuk peserta, bukan instruksi agent mengklasifikasikan citra.

Baca [konteks](../../references/PROJECT_CONTEXT.md). Label visual asisten lama tidak memenuhi jalur pelabelan yang dijelaskan Q&A dan tidak boleh digunakan sebagai target, seed, mapping cluster, atau evaluasi kompetisi. Pertahankan arsip; jangan menyembunyikan provenance atau menjadikan persetujuan manusia setelah melihat label AI sebagai bukti asal algoritmis. Jika pengguna meminta perbandingan arsip, bekukan assignment dahulu, lakukan terpisah dari notebook kompetisi, sebut sumber sebenarnya dan laporkan sebagai agreement dua metode tanpa ground truth; tidak memperbaiki per gambar atau menyetel metode dari hasil tersebut.

Periksa hubungan raw ID → feature run → cluster assignment → mapping nama oleh tim → label program. Versikan mapping, parameter dan checksum. Perubahan anggota harus berasal dari metode yang diizinkan dan tercatat, bukan pemindahan gambar manual. Pertanyaan subcluster/koreksi otomatis yang belum dijawab tetap unresolved.

Rekam `image_id, cluster_id, assigned_class, method_run_id, mapping_version, label_version`. Periksa coverage ID, kategori sah, mapping konsisten dan exclusion mask yang memiliki alasan metode. Confidence/proximity bersifat diagnostik; jangan memaksakan kuota kelas atau memakai label sample test.

Metode seperti cleanlab dapat menandai disagreement jika prediksi/provenance sesuai; penandaan bukan izin mengubah label per gambar atau pembuktian kelas yang benar. Pembuatan/pelatihan model oleh peserta tunduk pada aturan LLM yang dijelaskan pada sumber aktif.

Keluaran: audit asal-usul, mapping/coverage yang gagal, dan status label aktif. Tidak ada keputusan kategori per gambar oleh agent dalam skill ini.

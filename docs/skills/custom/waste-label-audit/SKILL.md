---
name: waste-label-audit
description: Memeriksa asal label cluster, mapping kategori oleh peserta, dan konsistensi artefak limbah tanpa melabel atau mengoreksi gambar kompetisi secara manual oleh LLM.
---

# Audit provenance label limbah

Gunakan [aturan aktif](../../references/RULES_AND_RUBRIC.md) dan [rancangan clustering](../unsupervised-waste-labeling/SKILL.md). [Definisi kategori](../../references/LABEL_TAXONOMY.md) hanya kamus material untuk peserta, bukan instruksi agent mengklasifikasikan citra.

Baca [konteks](../../references/PROJECT_CONTEXT.md). Label visual asisten lama tidak memenuhi jalur pelabelan yang dijelaskan Q&A dan tidak boleh digunakan sebagai target, seed, mapping cluster, atau evaluasi kompetisi. Pertahankan arsip; jangan menyembunyikan provenance atau menjadikan persetujuan manusia setelah melihat label AI sebagai bukti asal algoritmis. Jika pengguna meminta perbandingan arsip, bekukan assignment dahulu, lakukan terpisah dari notebook kompetisi, sebut sumber sebenarnya dan laporkan sebagai agreement dua metode tanpa ground truth; tidak memperbaiki per gambar atau menyetel metode dari hasil tersebut.

Periksa hubungan raw ID → feature run → cluster assignment → mapping nama oleh tim → label program. Versikan mapping, parameter dan checksum. Q&A 7 November membolehkan jumlah cluster/subcluster fleksibel dan metode pseudo-label otomatis pada train-only; jumlah cluster tidak perlu sama dengan jumlah kelas. Perubahan anggota harus dihasilkan oleh metode tersebut, bukan pemindahan gambar manual. Label manusia yang menjadi seed tetap dibatasi lima per kelas, dan setiap koreksi algoritmis harus mempunyai provenance serta validasi yang tidak menggunakan test.

Rekam `image_id, cluster_id, assigned_class, method_run_id, mapping_version, label_version`. Periksa coverage ID, kategori sah, mapping konsisten dan exclusion mask yang memiliki alasan metode. Confidence/proximity bersifat diagnostik; jangan memaksakan kuota kelas atau memakai label sample test.

Metode seperti cleanlab dapat menandai disagreement jika prediksi/provenance sesuai; penandaan bukan izin mengubah label per gambar atau pembuktian kelas yang benar. Pembuatan/pelatihan model oleh peserta tunduk pada aturan LLM yang dijelaskan pada sumber aktif.

Keluaran: audit asal-usul, mapping/coverage yang gagal, dan status label aktif. Tidak ada keputusan kategori per gambar oleh agent dalam skill ini.

Pembaruan 7 November 2026 terverifikasi pada handbook terbaru h.7-8: label manusia train maksimal5gambar/kelas (maksimal40ID sumber unik), test tidak boleh dilabeli manual atau masuk training. Baca pembaruan RULES_AND_RUBRIC sebelum aturan historis di atas; larangan total label manusia train lama sudah diganti batas ini. Label visual asisten lama tetap tidak menjadi seed manusia. Galeri kandidat adalah usulan program, belum label tim sampai diperiksa dan dikonfirmasi.

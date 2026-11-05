---
name: report-package-audit
description: Memeriksa konsistensi laporan PDF, notebook, CSV, bukti eksperimen, format nama, dan kelengkapan paket pengumpulan GELAR-RASA tanpa mengirim berkas secara otomatis.
---

# Pemeriksaan paket laporan

Gunakan [ketentuan aktif](../../references/RULES_AND_RUBRIC.md) dan salinan template resmi. Periksa nama tim/ketua dari informasi GELAR-RASA yang tersedia, jangan mengambil nama tim lomba BINUS/LOGIKA. Tiga nama resmi berbeda untuk Submission.csv, Notebook.ipynb, Laporan.pdf; jangan hanya mengganti ekstensi file.

Untuk PDF, tentukan batas halaman inti BAB I–IV terpisah dari cover/daftar/pustaka/lampiran. Cek A4, Times New Roman 12 pt, spasi 1,15, justify pada sumber dokumen dan hasil export. Font metadata saja tidak membuktikan seluruh format benar. Render seluruh halaman untuk inspeksi tabel pecah, caption terpotong, glyph hilang, nomor/daftar tidak sinkron, atau halaman kosong tak sengaja.

Untuk isi, hubungkan setiap hasil utama ke artefak nyata dan versi yang sama pada notebook. Cek sitasi↔pustaka dua arah, DOI/URL sumber, serta APA 7. Jangan memakai sumber yang belum dibaca sebagai pembuktian. Nyatakan angka unavailable, jangan membulatkan menjadi 100% tanpa denominator.

Untuk notebook, pisahkan inspection-only, smoke test, dan Run All. Bandingkan output prediksi dengan CSV yang dinyatakan; provenance label/checkpoint harus konsisten dengan aturan. Tidak memakai label visual asisten lama atau file test manual. Tidak menjalankan training saat tugas hanya review laporan.

Simpan manifest paket beserta hash, daftar temuan dan status selesai/pending. Verifikasi akses sharing saat diminta menyiapkan pengumpulan. Mengunduh template bukan membuat salinan Google Docs dalam akun; mengemas file bukan mengirim form. Hasil pemeriksaan tidak mengesahkan metode yang melanggar aturan.

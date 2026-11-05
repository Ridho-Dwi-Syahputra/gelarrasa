# PDF webinar lengkap

PDF asli disimpan tanpa perubahan dalam potongan 45 MiB karena kuota Git LFS akun tidak tersedia. Semua byte dan urutan bagian dicatat di `manifest.json`.

Dari root repo, jalankan `python eksperimen/skrip/restore_large_files.py` untuk mengembalikan `docs/materi/PPT Webinar Gelar Rasa 2026.pdf`. Untuk hanya memverifikasi bagian: tambahkan `--verify`. Skrip memeriksa SHA-256 setiap bagian dan PDF gabungan serta tidak menimpa PDF lain.

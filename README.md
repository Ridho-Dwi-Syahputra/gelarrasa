# GELAR RASA 2026

**Buka satu notebook:** [final-submit/DSC2026_Notebook.ipynb](final-submit/DSC2026_Notebook.ipynb).

```text
GELAR-RASA/
|-- data/
|   |-- Dataset DSC GelarRasa 2026/   data asli
|   `-- processed/                   train (8 kelas), test, crops, review
|-- docs/                            materi, rules, skills, agent-notes
|-- final-submit/
|   |-- DSC2026_Notebook.ipynb
|   |-- output/                      model dan CSV submission kelak
|   `-- reports/                     PNG dan galeri notebook
`-- arsip/                           seluruh versi lama
```

Sekarang fokus **EDA dan preprocessing**: 1.913 sampel train dalam delapan folder, 1.911 eligible, 480 test dan tujuh native crop. Status aktif: **17 berbeda / 109 sesuai dari 126 review tegas; 34 sumber manusia tetap, maksimal lima per kelas.** Prefix `frozen_residual_continuation`. Dua sumber 1662/1885 tetap ditahan; raw dan piksel test tidak diubah. Notebook lulus 13 sel kode, nol error, 16 visual dan 4.793 pemeriksaan hash. Keselarasan review bukan Accuracy panitia. Perubahan di luar review belum terverifikasi. Run All mematerialisasikan snapshot assignment dan menghitung crop; belum mengulang seluruh fit assignment heads atau membuat classifier/CSV final.

- [Panduan menjalankan dan struktur](docs/skills/project/PROJECT.md).
- [Skills dan aturan kerja agen](docs/skills/AGENTS.md).
- [Materi webinar](docs/materi/).
- [Catatan audit, cache dan eksperimen](docs/agent-notes/).
- [Bukti penerapan webinar](docs/agent-notes/webinar_coverage.csv).
- [Hasil eksekusi notebook](docs/agent-notes/execution.json).
- [Metode, eksperimen dan batas hasil penyempurnaan otomatis](docs/agent-notes/AUTOMATIC_REFINEMENT.md).
- [Status kategori aktif dan bukti](docs/agent-notes/CURRENT_LABEL_STATUS.md).
- [Galeri kasus tindak lanjut](docs/agent-notes/remaining_visual_review/latest_review.html).

Data asli dipertahankan. Versi lama dipindahkan ke arsip, bukan dihapus. Data, gambar terpisah, model, output dan arsip besar tetap lokal.

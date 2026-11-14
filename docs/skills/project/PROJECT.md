# Panduan proyek

Buka **final-submit/DSC2026_Notebook.ipynb**. Jalankan sel berurutan dengan kernel Python GELAR-RASA. Seluruh kode EDA/preprocessing ada di notebook; cache model opsional untuk tinjauan hasil lama.

Struktur aktif:

- data/Dataset DSC GelarRasa 2026: data asli dan arsip anotasi yang dipertahankan.
- data/processed/train: 1.913 citra dalam delapan folder kategori algoritmis.
- data/processed/test: citra test hasil preprocessing.
- data/processed/crops: native crop untuk audit sebelum resize.
- data/processed/review: cadangan tinjauan.
- data/processed/manifest.csv: parent, koordinat, group, partisi dan hash.
- final-submit/output: hasil model dan CSV submission saat tahap modeling tersedia; sementara kosong.
- docs/agent-notes: cache, fitur, draft split, audit, eksperimen, webinar dan bukti eksekusi.
- final-submit/reports: gambar PNG dan galeri hasil notebook.
- docs: materi, aturan dan skills.
- arsip: seluruh versi lama yang tidak diperlukan untuk alur aktif.

Untuk membangun dan menjalankan ulang notebook dari root proyek:

```powershell
.\.venv\Scripts\python.exe docs/skills/tools/build_notebook.py
```

Kernel builder bernama gelar-rasa-venv. Eksekusi tidak membuka VS Code/browser. Pastikan data resmi dan SampleSubmission_DSC.csv tersedia pada data. Output diperbarui di lokasi yang sama; kebijakan baru perlu mengarsipkan versi data aktif sebelumnya terlebih dahulu agar tidak menyisakan file lama.

Eksekusi 7 November 2026: 13 dari 13 sel kode selesai tanpa error, dengan 16 visual tertanam; data processed berhasil diregenerasi dari folder kosong. Bukti lengkap ada di docs/agent-notes/execution.json. Validator memeriksa 4.793 hash gambar sumber dan turunan; hasilnya docs/agent-notes/skills_validation.json.

Fokus sekarang EDA/preprocessing. Materi tentang training/evaluasi/submission ditandai tahap berikutnya pada notebook dan webinar_coverage.csv. Kategori crop dihitung terpisah memakai DINOv2 dan voting tetangga fit-train. Nama folder adalah assignment program yang belum menjadi ground truth independen. Tidak ada klaim Accuracy dari proyeksi fitur atau confidence model.

Riwayat pemindahan tersimpan di arsip/sebelum_rapi_20261107/migration_manifest.json. Untuk rollback gunakan pasangan archived_path dan old_path setelah memastikan tidak menimpa keluaran baru. Snapshot identitas file diverifikasi saat pemindahan. Dataset asli tidak dipindah.

Input operasional notebook: raw train/test, sample submission, serta data/preprocessing_inputs.npz. Notebook tidak merujuk folder docs atau layanan model bahasa. Catatan administrasi dikumpulkan builder setelah notebook selesai.

Pembaruan terbaru7November2026:tahap material/sourcegroup memberi5perubahan algoritmis,tanpa tambahan manual;total73sejak snapshot150. Budget34/40tetap,max5/kelas,sisa metal2/foodwaste4.126reviewtegas:81cocok/45berbeda,sebelumnya48berbeda.0432glass/1510plastic/1817paperpulih.116koneksigeometri kini satu groupdanpartisi;group sumber bukan kategori.1913traintersimpan,1911eligible,480testtetap,1662/1885ditahan. Baca docs/agent-notes/MATERIAL_REFINEMENT_RESULT.md dan json/material_followup_delivery.json. Notebook mematerialisasikan snapshot,belumseluruhfitrefinementatauclassifierfinal.
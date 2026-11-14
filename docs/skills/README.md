# Pustaka skill GELAR RASA

Diperbarui 7 November 2026. Fokus: klasifikasi satu citra limbah menjadi satu dari delapan kelas serta laporan sesuai rubrik. Paket ini berisi **22 skill khusus proyek dan 17 paket snapshot eksternal (240 file)**. Subskill bersarang tidak dihitung sebagai paket tambahan. Ini koleksi terkurasi, bukan klaim mencakup seluruh skill di internet.

Setiap tugas dimulai dengan [pemilih skill](custom/gelar-rasa-skill-router/SKILL.md) dan [aturan/rubrik aktif](references/RULES_AND_RUBRIC.md). Handbook terbaru (7 November, halaman 7-8) mengizinkan maksimal lima label manusia per kelas train, melarang label manual test/training dengan test, dan tetap melarang VLM; anotasi visual asisten lama kini arsip yang dikecualikan dari jalur kompetisi. Label aktif hasil metode peserta belum tersedia. Lanjutkan ke [konteks](references/PROJECT_CONTEXT.md), [rencana EDA/preprocessing](references/EDA_PREPROCESS_PLAN.md), atau [rancangan laporan](references/REPORT_BLUEPRINT.md) sesuai tugas. Semua metode merupakan panduan, bukan hasil eksperimen yang sudah terjadi.

## Pilih skill sesuai pekerjaan

| Kebutuhan | Skill lokal |
|---|---|
| Mengecek aturan dan memilih skill setiap prompt proyek | [gelar-rasa-skill-router](custom/gelar-rasa-skill-router/SKILL.md) |
| Menentukan langkah berikut dari bukti | [gelar-rasa-strategy](custom/gelar-rasa-strategy/SKILL.md) |
| Mencari dan menilai jurnal/repo | [vision-research-review](custom/vision-research-review/SKILL.md) |
| Menelaah pelabelan melalui clustering peserta | [unsupervised-waste-labeling](custom/unsupervised-waste-labeling/SKILL.md) |
| Memeriksa provenance label cluster dan mapping | [waste-label-audit](custom/waste-label-audit/SKILL.md) |
| EDA gambar, sumber, dan duplikat | [image-eda-duplicates](custom/image-eda-duplicates/SKILL.md) |
| Memeriksa decode, orientasi, warna, dan transform gambar | [image-preprocessing-audit](custom/image-preprocessing-audit/SKILL.md) |
| Membandingkan cakupan/domain train dan test tanpa label test | [image-domain-shift-audit](custom/image-domain-shift-audit/SKILL.md) |
| Menyusun fold dan evaluasi yang dapat dipercaya | [grouped-image-validation](custom/grouped-image-validation/SKILL.md) |
| HOG, warna, tekstur, SVM | [classical-image-baseline](custom/classical-image-baseline/SKILL.md) |
| CNN, transfer learning, fine-tuning | [cnn-transfer-learning](custom/cnn-transfer-learning/SKILL.md) |
| Memeriksa loop training, kurva, throughput, dan resume | [vision-training-diagnostics](custom/vision-training-diagnostics/SKILL.md) |
| Embedding CNN/ViT + fitur klasik | [hybrid-image-features](custom/hybrid-image-features/SKILL.md) |
| YOLO classification dan keputusan perlu deteksi | [yolo-image-classification](custom/yolo-image-classification/SKILL.md) |
| Resize, augmentasi, MixUp/CutMix, TTA | [vision-augmentation](custom/vision-augmentation/SKILL.md) |
| Confusion matrix, kesalahan material, Grad-CAM | [image-error-analysis](custom/image-error-analysis/SKILL.md) |
| Ensemble dan pseudo-label bersyarat | [image-ensemble-pseudolabel](custom/image-ensemble-pseudolabel/SKILL.md) |
| Notebook yang bisa diulang dan CSV submission | [reproducible-image-submission](custom/reproducible-image-submission/SKILL.md) |
| Menyiapkan path, preflight, checkpoint, dan output untuk GPU Kaggle | [kaggle-gpu-workflow](custom/kaggle-gpu-workflow/SKILL.md) |
| Kerangka laporan, bukti, dan interpretasi menurut rubrik | [competition-analysis-report](custom/competition-analysis-report/SKILL.md) |
| Grafik/tabel/diagram analisis yang dapat ditelusuri | [analytical-figures](custom/analytical-figures/SKILL.md) |
| Audit PDF, notebook, CSV, format dan konsistensi paket | [report-package-audit](custom/report-package-audit/SKILL.md) |

## Sumber pendukung

- [Aturan, Q&A, bobot nilai, tenggat, dan dampak pada label lama](references/RULES_AND_RUBRIC.md).
- [Blueprint laporan dan bukti untuk setiap kriteria](references/REPORT_BLUEPRINT.md).
- [Template resmi PDF](../rules/sources/official_report_template_20261105.pdf) dan [DOCX](../rules/sources/official_report_template_20261105.docx).
- [Riset primer dan implikasi eksperimen](references/RESEARCH_REVIEW.md).
- [Katalog sumber resmi](references/SOURCE_REGISTER.md).
- [Seleksi dan tinjauan skill eksternal](references/UPSTREAM_REVIEW.md).
- [Rubrik anotasi lokal dan batas antar kelas](references/LABEL_TAXONOMY.md).
- [Rencana EDA/preprocessing beserta artefak dan syarat selesai](references/EDA_PREPROCESS_PLAN.md).
- [Kontrak artefak, nested validation, dan strategi refit](references/ARTIFACT_CONTRACTS.md).
- [Hasil penyempurnaan, pengujian, dan batas kesiapan](references/REVIEW_IMPROVEMENTS.md).
- [Versi, lisensi, SHA-256 setiap file unduhan](sources/vendor_manifest.json).
- [Struktur proyek](project/PROJECT.md).

`custom/` adalah petunjuk kerja yang ditulis untuk dataset ini. `vendor/` menyimpan subtree terpilih secara utuh pada commit yang dicatat; ini bukan klaim bahwa seluruh dependensi transitif di luar subtree sudah disalin. Instruksi vendor adalah referensi yang harus disesuaikan dengan proyek. Tidak ada pemasangan global, peluncuran training, upload data, atau layanan berbayar dari pengumpulan ini. Keterpasangan sebagai file lokal tidak berarti otomatis muncul dalam katalog skill setiap aplikasi; agent dapat membacanya lewat path dari indeks ini.

Metadata, hashing, audit label, dan sebagian besar EDA dapat dimulai di lokal/CPU. Panduan Kaggle menyiapkan opsi GPU untuk embedding/training setelah input, label, dan split siap; tersedianya skill tidak berarti notebook sudah dibuat, sesi GPU sudah berjalan, atau data sudah diunggah.

Contoh pemakaian: “Baca `docs/skills/custom/image-eda-duplicates/SKILL.md`, lalu lakukan audit sesuai kontrak data.” Baca hanya skill yang relevan, bukan seluruh vendor sekaligus.

## Pemeriksaan paket

Jalankan `python docs/skills/tools/validate_collection.py --verify-images` dari root proyek setelah Run All. Pemeriksaan mencakup tautan skill, checksum sumber/vendor, struktur aktif, bukti eksekusi notebook, cakupan sumber dan turunan, pemisahan group, serta hash 2.400 gambar asli dan 2.393 gambar praproses. Tidak membaca anotasi arsip atau menjalankan training.

Hasil disimpan pada `docs/agent-notes/skills_validation.json`. Opsi `--no-write` hanya mencetak hasil. Tanpa `--verify-images`, pemeriksaan data hanya memeriksa struktur dan identitas berkas yang tercatat. Keberhasilan pemeriksaan tidak membuktikan ketepatan kategori atau Accuracy.

Validator dan fixture lama disimpan bersama pada `arsip/sebelum_rapi_20261107/docs/skills/tools/`; tidak dipakai untuk kontrak aktif.

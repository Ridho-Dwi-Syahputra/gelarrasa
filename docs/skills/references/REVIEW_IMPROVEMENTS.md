# Hasil penyempurnaan koleksi — 4 November 2026

Catatan ini menyimpan hasil pemeriksaan historis 4 November. **Aturan dan arah label telah direvisi pada 5 November** dalam [aturan aktif](RULES_AND_RUBRIC.md); saran tinjauan/perbaikan label manual di bawah bukan lagi rencana kompetisi. Inventaris terbaru 22 skill lokal/17 paket vendor ada di indeks, sedangkan angka 17/15/191 berikut mencatat keadaan saat tinjauan lama, bukan hasil terbaru.

Lingkup: memperkuat skill dan persiapan pipeline dengan prioritas EDA/preprocessing. Tidak ada training, relabeling, inferensi model, atau upload Kaggle dalam pekerjaan ini.

## Perubahan yang sudah dibuat

| Temuan/kebutuhan | Perbaikan |
|---|---|
| Audit label perlu konvensi yang dapat diperiksa | [Rubrik lokal delapan kelas](LABEL_TAXONOMY.md), penanganan objek campuran, uncertainty, dan versi taxonomy |
| EDA/preprocessing sebelumnya terlalu tersebar | [Rencana bertahap](EDA_PREPROCESS_PLAN.md), skill audit preprocessing dan domain shift |
| Referensi tambahan perlu sesuai tugas data citra | Empat paket Voxel51 dan tiga paket AREX; total 15 paket/191 file, commit dan checksum disimpan |
| Repeat CV dapat disalahartikan sebagai evaluasi independen | Pisahkan stabilitas dari evaluasi seleksi; jelaskan nested/holdout dan batas bias desain |
| Stacking/hybrid dapat terkontaminasi meski memakai istilah OOF | [Kontrak artefak](ARTIFACT_CONTRACTS.md) merinci inner-OOF hanya di outer-train dan encoder fold yang konsisten |
| Urutan label folder berbeda dari probabilitas model | Mapping berdasarkan nama kelas; sample CSV tidak dipakai menebak urutan probabilitas |
| StandardScaler bukan penyeimbang otomatis antar blok fitur | Bedakan scaling koordinat, L2 per blok, bobot, dan urutan operasi |
| Full-train refit belum punya kebijakan durasi | Tentukan fold ensemble atau refit dengan epoch/step terpilih dari development; test bukan early stopping |
| Kemungkinan memakai GPU Kaggle | [Skill porting](../custom/kaggle-gpu-workflow/SKILL.md): path, preflight, versi runtime, checkpoint/resume, dan arsip output |
| Pipeline training perlu diagnosis sebelum sweep | Skill vision-training-diagnostics untuk batch sanity check, kurva, effective batch, throughput dan resume |

## Bukti pemeriksaan

- Validator resmi skill-creator: **17/17** entrypoint lokal lulus. Hasil: [skill_creator_validation.json](../sources/skill_creator_validation.json).
- Pengujian kontrak data: **16 tes lulus**, dijalankan melalui `python -B docs/skills/tools/test_validate_collection.py`. Fixture sementara memakai byte sintetis; ini menguji ID, path, kecocokan CSV, hash, dan perilaku CLI, bukan decode gambar.
- Validator koleksi memeriksa tautan lokal, checksum vendor/materi, cakupan label/sample dan jalur salinan. `--verify-images` menambahkan hash train asli/salinan; hasil aktual, waktu, dan jumlah pemeriksaan ada di [validation.json](../sources/validation.json).
- `--no-write` tersedia untuk pemeriksaan tanpa mengganti laporan. Python `-O` ditolak agar structural assertions tidak diam-diam dilewati. Kegagalan normal menulis status gagal, bukan meninggalkan laporan sukses yang menyesatkan.
- Tinjauan independen dilakukan terhadap metodologi ML dan tiga skenario penggunaan: hybrid embeddings, ensemble mapping, serta preparation-only. Perbaikan di atas menangani temuan konkret dari tinjauan tersebut. Ini bukan pembuktian bahwa setiap kemungkinan workflow sudah diuji.

## Batas kesiapan

Jumlah skill dan hash yang lulus tidak membuktikan akurasi anotasi atau hasil kompetisi. Dekode/quality profiling menyeluruh, tinjauan kembali 51 label meragukan, near-duplicate grouping, fold, notebook training, dan eksperimen model tetap pekerjaan berikutnya. Snapshot vendor belum diuji sebagai software terpasang; status cakupan sumber dan keterbatasan paper dicatat dalam [tinjauan sumber](UPSTREAM_REVIEW.md) dan [riset](RESEARCH_REVIEW.md).

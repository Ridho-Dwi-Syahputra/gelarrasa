---
name: kaggle-gpu-workflow
description: Menyiapkan pipeline citra lokal agar dapat dijalankan di notebook Kaggle dengan konfigurasi path, pemeriksaan GPU, checkpoint, dan penyimpanan hasil tanpa memulai training atau publikasi secara otomatis.
---

# Persiapan GPU Kaggle

Baca [aturan aktif](../../references/RULES_AND_RUBRIC.md) sebelum menyiapkan paket. Jangan memasukkan label visual asisten lama atau komponen VLM. Label aktif perlu asal-usul metode peserta; pretrained non-VLM memerlukan training/fine-tuning peserta pada data kompetisi. Skill infrastruktur ini tidak memberi pengecualian atas larangan penggunaan LLM untuk membuat model, klasifikasi, atau prediksi.

Gunakan saat porting atau mempersiapkan eksekusi, bukan menganggap GPU diperlukan untuk seluruh EDA. Metadata, hashing, audit label, dan pHash biasanya dapat dimulai lokal/CPU; ekstraksi embedding dan training dapat dipindah setelah manifest/label/split siap. [Lingkungan resmi Kaggle](https://github.com/Kaggle/docker-python), [dokumentasi kernels CLI](https://github.com/Kaggle/kaggle-cli/blob/main/docs/kernels.md).

Pakai `DATA_ROOT` input dan `WORK_ROOT` output yang dikonfigurasi. Pada notebook Kaggle, petakan input yang dipasang di `/kaggle/input` dan tulis hasil di `/kaggle/working`; jangan menaruh path Windows absolut di pipeline. Cocokkan isi manifest/hash dan ID setelah staging. Sertakan satu versi sumber, label CSV, audit/taxonomy, group/fold, config, serta kode; jangan mengunggah raw dan salinan label sebagai dua dataset berbeda.

Preflight runtime mencatat Python, torch/torchvision/CUDA, GPU yang benar-benar tersedia, RAM/VRAM, ruang disk, internet, dan lokasi input. Jangan hardcode jenis GPU, jatah mingguan, atau durasi sesi dari artikel lama. Periksa akun/runtime aktual ketika akan berjalan. Jangan reinstall torch secara buta; uji kompatibilitas pasangan torch/torchvision yang ada.

Internet-off memerlukan wheel/bobot yang telah disediakan secara sah. Jangan membuat asumsi unduhan otomatis berhasil. Bila pretrained dipakai, sertakan tag/checksum dan processor. Mulai satu batch train/validation untuk smoke test setelah training diotorisasi, lalu estimasi durasi. AMP dan DataLoader workers ditentukan dari perangkat; CPU fallback dicatat, bukan disebut run GPU.

Checkpoint best dan resume terpisah; resume memuat optimizer/scheduler/scaler/RNG, epoch/step, config dan hash data. Simpan berkala ke working dan persistenkan melalui mekanisme output versi notebook yang tersedia sebelum sesi berakhir. File working tidak dianggap arsip permanen hanya karena pernah ditulis. Uji resume pada run kecil.

Keluaran tahap persiapan: peta input-output, environment/preflight report, manifest paket, dan command/notebook siap diuji. Upload dataset, publish notebook, memulai sesi GPU, serta submit kompetisi tidak terjadi hanya karena membaca skill ini; ikuti permintaan pengguna yang berlaku.

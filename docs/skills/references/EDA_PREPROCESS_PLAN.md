# Rencana EDA dan preprocessing prioritas

Status 5 November: panduan persiapan direvisi setelah [aturan/Q&A](RULES_AND_RUBRIC.md). Anotasi visual lama dikecualikan; label aktif hasil metode peserta belum tersedia. Pemeriksaan berikut belum boleh ditulis selesai sebelum artefaknya tersedia. EDA teknis tidak memerlukan agent memberi kelas pada citra.

| Urutan | Pemeriksaan | Artefak konkret | Syarat selesai |
|---|---|---|---|
| 1 | Cakupan ID, manifest, original vs copy, decode seluruh file | `reports/eda/integrity.csv` | 1.920/480 ID tercakup; semua error decode eksplisit; tidak ada silent skip |
| 2 | Mode/EXIF/dtype/size/aspect/quality | `reports/eda/image_profile.csv` dan galeri ekor distribusi | Definisi indikator dan seluruh ID tercatat; metadata buruk tidak otomatis dibuang |
| 3 | Asal label baru dari metode peserta | feature/clustering run, cluster assignment, mapping tim, label_version | Tidak menggunakan label visual lama; tidak ada pelabelan/koreksi gambar manual; mapping kategori cluster terlacak |
| 4 | Exact/pixel/perceptual duplicate dan satu sumber | pasangan konfirmasi + group_id | Exact pair lama terjaga; kandidat near-duplicate punya status review, bukan dianggap fakta |
| 5 | Cakupan train–test/source slices | tabel slice dan tetangga/galeri | Tidak memakai label sample sebagai kebenaran; bias sumber dicatat |
| 6 | Ablasi preprocessing terkontrol | config transform + before/after + mapping sumber | Orientasi/warna/range benar; objek tidak hilang; raw tetap identik |
| 7 | Split lalu smoke test loader | manifest/fold/transform hash, batch audit | Tidak ada group overlap; class mapping konsisten; fold memadai tiap kelas |

## Cara memilih perbaikan

Pisahkan isu yang dapat dibuktikan tanpa model (file rusak, ID salah, warna terbalik) dari hipotesis yang membutuhkan evaluasi (CLAHE, crop, denoise, resolusi tinggi). Beri setiap hipotesis kontrol tanpa transform. Transform destruktif yang tampak “lebih bagus” bagi manusia belum tentu membantu klasifikasi material.

Material yang mirip, latar dominan, tekstur dan ilustrasi merupakan hipotesis tantangan representasi. Periksa kualitas input dan perubahan transform tanpa meminta LLM/VLM menentukan kategori gambar. Jangan memulai dengan seluruh kombinasi transform; gunakan pembanding yang menjawab isu tertentu.

## Pembagian lokal dan Kaggle

Lokal/CPU: inventaris, hash, decode, pHash, metadata dan galeri teknis. GPU Kaggle bila dibutuhkan peserta: embedding non-VLM dan training. Pembuatan model peserta tetap mengikuti aturan penggunaan LLM. Bawa manifest/label-program/split yang sama; satu fold pilot mengukur biaya/diagnosis, bukan memilih fold termudah sebagai skor akhir.

## Gerbang menuju modeling

Yang diperlukan adalah label/source dari metode yang sesuai aturan, mapping konsisten, unresolved transparan, kelompok utama diperiksa, serta pipeline input benar. Agreement terhadap label cluster bukan semantic ground truth. Jangan mengganti denominator agar skor terlihat tinggi atau kembali memakai 51 anotasi lama sebagai target audit.

Peserta menilai rancangan [clustering](../custom/unsupervised-waste-labeling/SKILL.md), lalu training/fine-tuning pada data kompetisi sesuai [playbook](EXPERIMENT_PLAYBOOK.md). Panduan Kaggle tidak berarti GPU telah dipakai; metadata yang lulus tidak membuktikan label benar atau metode memenuhi seluruh aturan.

---
name: unsupervised-waste-labeling
description: Menelaah rancangan pelabelan train tanpa label melalui representasi non-VLM dan clustering peserta, dengan penamaan cluster, provenance, serta evaluasi yang tidak memakai anotasi manual lama.
---

# Rancangan pelabelan melalui metode peserta

Baca [Q&A dan batas metode](../../references/RULES_AND_RUBRIC.md). Klarifikasi pengguna 5 November mengizinkan bantuan penulisan kode selama notebook tidak memanggil LLM. Implementasi algoritma lokal dapat dikerjakan sesuai otorisasi itu; agent tidak memberi kategori dari pengamatan citra. Dataset tidak mempunyai label resmi. Jangan mengambil `train/label`, `train_labels.csv`, atau anotasi test sebagai target, nama cluster, seed maupun ukuran kualitas kompetisi. Perbandingan arsip yang diminta eksplisit hanya laporan terpisah sesudah assignment dibekukan; tidak mengubah algoritma, mapping, atau prediksi.

Rancangan yang perlu diuji peserta: manifest raw → preprocessing konsisten → fitur klasik atau representasi pretrained **non-VLM** yang memenuhi aturan → algoritma clustering → mapping nama cluster oleh anggota tim → training/fine-tuning lokal pada data kompetisi. Model pretrained jadi yang langsung menebak test tidak memenuhi jawaban terbaru. Simpan asal bobot, transform, fitur, seed, parameter cluster, dan assignment setiap ID.

Clustering memisahkan pola representasi, tidak otomatis kategori material. Bandingkan metode/normalisasi pada stabilitas, distribusi ukuran cluster, tetangga dan struktur sumber; silhouette tinggi tidak membuktikan Accuracy material tinggi. Latar, warna, atau bentuk bisa mendominasi. PCA/UMAP untuk proyeksi berbeda dari preprocessing clustering; rekam keduanya.

Penamaan cluster oleh tim diperbolehkan; agent tidak menetapkan nama kelas dari melihat citra kompetisi. Koreksi/pemindahan anggota satu per satu secara manual dilarang. Jangan membuat cluster satu gambar untuk menyamarkan manual labeling. Batas subclustering/jumlah cluster dan koreksi otomatis belum dijawab dalam snapshot; jangan mengklaim strategi tersebut sudah disetujui.

Simpan `image_id, cluster_id, assigned_class, method_run_id, mapping_version` dari program dan mapping tim. Label aktif baru harus punya provenance terpisah; raw/arsip lama tetap utuh. Jangan menamai ulang label manual sebagai pseudo-label algoritmis.

Dalam evaluasi, jelaskan apakah clustering melihat seluruh train atau hanya subset training. Agreement classifier terhadap label cluster mengukur reproduksi target turunan, bukan kebenaran semantik. Nilai public tidak mengungkap label private. Hasil skill: rancangan metode, kontrak artefak, daftar ketidakpastian, dan bukti yang harus dihasilkan peserta; bukan prediksi atau skor yang belum dijalankan.

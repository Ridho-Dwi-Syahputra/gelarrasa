# Perbaikan assignment train setelah pembaruan aturan

Aturan resmi terbaru diunduh dari tautan handbook panitia, diperiksa 7 November 2026. PDF terbaru 9 halaman; halaman7 memperbolehkan maksimal5label manusia per kelas train, melarang label manual test/training dengan test. Halaman8 tetap melarang layanan inference/LLM/VLM dan mensyaratkan satu notebook serta dataset yang dapat diakses panitia. Snapshot: docs/rules/sources/handbook_latest_20261107.pdf. Asal label AI historis tidak berubah menjadi label manusia karena pembaruan ini.

## Eksperimen yang benar-benar dijalankan

Empat metode fixed: prototype DINO, ridge DINO+CNN limbah, graph DINO dan graph DINO+CNN limbah. Seluruh graph/fit memakai train saja:1524fit,382heldout. Acuan awal40 adalah konsensus otomatis program (5/class), bukan hand labeling. Pemilihan model memakai agreement pada257heldout automatic consensus anchors; ini bukan Accuracy ground truth.

Kandidat terpilih sebelum audit arsip: ridge DINO+CNN. Agreement proxy97,7718%, dibanding baseline97,6668%. Pada audit terpisah sesudah pembekuan, perbedaan arsip menjadi323, dibanding174baseline. Dari174perbedaan lama,50berubah menjadi cocok tetapi199perbedaan baru muncul. Hasil ini belum layak menggantikan data aktif. Nilai proxy dari sumber otomatis belum menjadi alat pemilih yang cukup untuk kasus sulit; keterwakilan40acuan, bias sumber dan validitas label perlu dievaluasi dengan acuan tim.

Notebook aktif, input snapshot dan manifest train dipastikan tidak berubah;174perbedaan masih status aktif. Tidak ada hasil eksperimen yang disalin ke folder kelas, tidak ada label arsip yang menjadi target/filter, dan tidak ada data test dalam fit.

## Acuan manusia yang masih diperlukan

Galeri seed_review.html memuat40kandidat dari konsensus dan keragaman fitur, diurutkan ID menaik. Dropdown awal kosong. Anggota tim perlu melihat foto dan memberikan kelasnya; usulan program tidak boleh otomatis disebut label manusia. Gambar ambigu boleh dilewati. Ekspor membutuhkan nama pemeriksa, minimal satu acuan setiap kelas dan maksimal5/class. SetiapIDsumber hanya dihitung sekali dan budget tidak direset pada fold atau augmentasi. Riwayat tinjauan sebelumnya tetap disimpan; catatan ini tidak mengklaim menghapus atau mereset tindakan manual sebelumnya.

Setelah label tim tersedia: validasi ID/hash/kuota, evaluasi silang pada acuan manusia dengan seluruh group validation dikeluarkan dari graph/training, lalu fit pada train dengan seed tersebut dan inferensi gambar lainnya. Metode calon adalah nearest-neighbor/graph dan classifier yang dilatih dengan acuan terbatas, bukan menyalin174label arsip. Seluruh perubahan global dibekukan sebelum audit arsip berikutnya. Tidak ada janji bahwa40acuan akan otomatis menghilangkan seluruh perbedaan atau memberi skor sempurna.

FixMatch adalah opsi lanjutan untuk fine-tuning dengan augmentasi lemah/kuat dan pseudo-label yang dibatasi confidence. Belum dijalankan;40label manusia belum tersedia. LaplacianShot bersifat transduktif pada query pada paper aslinya; penerapan lomba harus membatasi graph ke train. Test kompetisi hanya untuk inferensi setelah model/parameter dibekukan.

## Sumber primer yang dibaca

- Learning with Local and Global Consistency: https://www.microsoft.com/en-us/research/publication/learning-local-global-consistency/
- Laplacian Regularized Few-Shot Learning: https://proceedings.mlr.press/v119/ziko20a.html
- FixMatch: https://research.google/pubs/fixmatch-simplifying-semi-supervised-learning-with-consistency-and-confidence/

Berkas pembekuan: json/fewshot_selection_frozen.json; audit terpisah: json/fewshot_selected_archive_audit.json. Seluruh angka adalah hasil eksperimen lokal, bukan skor Kaggle atau sertifikasi kepatuhan panitia.

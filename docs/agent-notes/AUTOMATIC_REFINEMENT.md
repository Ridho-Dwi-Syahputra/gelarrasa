# Perbaikan kategori otomatis, 7 November 2026

Catatan ini merekam tahap historis tanpa acuan manusia. Status terbaru setelah review pengguna dan aktivasi14acuan ada pada [BOUNDED_REFINEMENT_RESULT.md](BOUNDED_REFINEMENT_RESULT.md); angka150di bawah bukan jumlah kesalahan aktif.

Arahan terbaru: simpan seluruh jatah label manusia. Semua eksperimen pada catatan ini memakai **0/40 label manusia** dan **0 foto test dalam fit**. Usulan galeri 40 foto tidak dianggap telah diperiksa manusia.

## Yang berubah pada pendekatan

Memakai hanya konsensus pretrained yang sangat yakin membuang cakupan subtipe: konsensus awal hanya menyediakan 123 acuan otomatis foodwaste. Sumber label yang lebih sempit dapat tampak konsisten pada acuannya sendiri tetapi mengubah kategori yang sebenarnya sudah masuk akal.

Pendekatan baru mempertahankan seluruh 1.906 kategori foto utuh dari program routing sebagai **target bising**, bukan menggantinya dengan arsip visual. Lima keluarga fitur membuat prediksi dengan lima fold yang memisahkan source group. Penggantian kategori memerlukan sedikitnya empat keluarga yang sepakat. Keluarga tersebut memakai DINOv2 global/lokal, DINOv3 global/foreground, dan DINOv3 bersama representasi CNN delapan kelas. Ini lima keluarga representasi, **bukan lima sumber ground truth yang independen**.

Satu konfigurasi per keluarga dipilih dari 45 konfigurasi memakai konsistensi flip dan agreement dengan inti kategori program yang didukung sumber lain. Angka tersebut bukan Accuracy/AUC/OOF semantik resmi. Fold memisahkan source group, tetapi target historis sendiri berasal dari proses fit terdahulu.

Iterasi berikut memakai konfigurasi yang sama, mempertahankan 589 jangkar otomatis kuat, dan berhenti bila perubahan kurang dari satu persen atau mencapai empat putaran. Putaran kedua mengubah 17 kategori dari putaran pertama sehingga memenuhi aturan berhenti. Batas dan aturan dibekukan sebelum membaca arsip pembanding. Iterasi lanjutan tidak merupakan evaluasi holdout independen karena targetnya bergantung keluaran putaran sebelumnya.

## Audit terpisah setelah prediksi dibekukan

Audit ini membandingkan prediksi dengan arsip label visual terdahulu, bukan label resmi. Arsip tidak masuk fitur, target, pemetaan nama, filtering, atau pemilihan konfigurasi dalam eksperimen ini. Peneliti telah mengetahui hasil audit historis; pengaruh pengetahuan tersebut tidak dapat dihapus secara retrospektif.

| Eksperimen | Perbedaan terhadap arsip, dari 1.906 foto utuh |
|---|---:|
| Routing aktif sebelum eksperimen | 174 |
| 48 kandidat, seluruh acuan konsensus otomatis | 277 |
| Gate konsensus konservatif | 174; tidak menerima perubahan |
| Penyesuaian prior kelas dan split visual | 267 |
| Representasi DINOv3, 32 konfigurasi | 264 |
| Perluasan ontologi WordNet secara umum | 175 |
| Teacher CNN delapan kelas langsung dengan gate flip/confidence | 223 |
| Denoising kategori program, empat dari lima keluarga, putaran pertama | 150 |
| Iterasi dengan jangkar tetap dan penghentian berdasarkan kestabilan | **150** |

Hasil final dibekukan pada `json/iterative_program_frozen.json`, hash prediksi `d7765ffaeb2e1aecaf9762418319509fe8873e03aff72a351748fb821f5b0076`. Audit final: **42 perbedaan lama menjadi cocok, 18 perbedaan baru muncul**, sehingga berkurang 24 secara bersih. Delapan kategori lain berubah tetapi tetap berbeda. Total 68 kategori program berubah.

Tidak memilih putaran atau perubahan per gambar berdasarkan arsip. Nilai 150 yang sama pada kedua putaran tidak dijadikan aturan penghentian. Hasil audit tidak membuktikan 68 perubahan tersebut benar secara semantik.

## Publikasi dan reproduksi

`json/program_refinement_publication.json` merekam status publikasi aktual. Script publikasi menyimpan versi aktif sebelumnya, memperbarui snapshot input beserta provenance, memulai dari folder processed kosong, dan menjalankan notebook utama. Jika Run All atau pemeriksaan integritas gagal, versi sebelumnya dipulihkan.

Notebook menghasilkan delapan folder train, test, native crop, review, manifest, fitur, dan provenance. Foto utuh membaca snapshot program yang sudah dibekukan; **notebook preprocessing belum melatih ulang seluruh proses refinement**. Script eksperimen, konfigurasi, fitur, dan input awal disimpan untuk replay refinement; replay harus memakai manifest routing awal pada backup publikasi, bukan manifest aktif yang sudah diperbarui. Snapshot test tidak diubah. Kategori tujuh crop diinferensikan ulang dari referensi train; jangan menghitungnya sebagai ketidakcocokan kategori foto induk.

Publikasi lulus: notebook13/13sel kode,0error,16visual;4.793hashgambar diverifikasi. Audit aktif ada pada `json/published_refinement_audit.json`. Jumlah saat ini1.756cocok dan150berbeda dari1.906foto utuh. Ketujuh crop tetap berkategori electronic(3),foodwaste(1),paper(1),plastic(1),cardboard(1). Daftar150perbedaan numerik menaik ada pada `table-csv/remaining_program_archive_differences.csv`, hanya untuk diagnostik.

Skor foto yang diubah disimpan sebagai batas bawah dukungan empat dari lima keluarga, bukan probabilitas terkalibrasi. Seluruh kategori tetap ditandai belum diverifikasi. Raw dan folder arsip label tidak diubah.

Lingkup tetap EDA/preprocessing, belum classifier final atau submission. Hasil ini tidak menjamin Accuracy sempurna atau kelulusan verifikasi kompetisi. Representasi CNN menggunakan bobot pretrained sebagai base; classifier final peserta, evaluasi independen, akses data panitia, dan seluruh alur training-submission dalam satu notebook masih perlu diselesaikan.

## Sumber representasi

- [DINOv3, repositori resmi Meta](https://github.com/facebookresearch/dinov3), dan [kartu encoder yang digunakan](https://huggingface.co/timm/vit_small_patch16_dinov3.lvd1689m). Revision dan checksum ada pada `json/dinov3_descriptor_provenance.json`.
- [CNN delapan kelas](https://huggingface.co/C0mpl1cated3/trash_classifier_v2), revision `ac54e811e3a5d505b57bf6dc1495ed5cd695c036`, dibuat 27 Agustus 2026 menurut metadata sumber. Overlap data pretraining dengan data panitia tidak diketahui. Tidak mengunduh dataset gambar tambahan atau menjalankan aplikasi/API sumber.
- [Dokumentasi resmi AdaptiveConcatPool2d FastAI](https://docs.fast.ai/layers.html#adaptiveconcatpool2d), untuk memeriksa susunan fitur max lalu mean. Bobot diekstrak sebagai tensor tanpa menjalankan callable pada serialized export; bukti pada `json/task8_tensor_extraction.json`.

> **Snapshot historis sebelum revisi review terbaru.** Status aktif: [REVIEW_REVISION_RESULT.md](REVIEW_REVISION_RESULT.md),34acuan dan48perbedaan dari126review tegas.

# Hasil perbaikan berbatas acuan manusia

Status aktual7November2026: perubahan telah diterapkan ke processed dan notebook utama telah dijalankan ulang. Ini perbaikan assignment/preprocessing, bukan classifier/submission final atau bukti Accuracy sempurna.

## Hasil yang diterapkan

- **42 kategori foto utuh berubah** terhadap snapshot saat tugas ini dimulai. Sebanyak11berasal dari koreksi pada acuan manusia, dan31dari aturan/prediksi program. Acuan manusia berjumlah14; tiga lainnya memang sudah sekelas dengan program.
- Asal program baru:14aturan ontology pencahayaan dengan dukungan spesifik pada dua view;11propagasi yang stabil pada dua pembobotan prior;6hasil clustering yang juga didukung graph dan mayoritas classifier.
- Dua usulan lampu yang hanya memiliki satu dukungan spesifik dikembalikan ke kategori program sebelumnya dan diberi flag review. Sepuluh usulan sensitif terhadap pembobotan tidak diterapkan. Ini pengendalian ketidakpastian, bukan penetapan kelas benar oleh asisten.
- Kesembilan preferensi tegas pengguna yang mendukung program lama tetap dipertahankan.
- **1662 dan1885 ditahan** sesuai jawaban pengguna: bukan acuan manusia, `training_eligible=False`, tidak menjadi referensi penilaian crop. File sumber dan preview tetap disimpan. Penerapan penahanan datang setelah fit assignment heads; tidak mengklaim kedua foto tidak pernah hadir dalam eksperimen historis.

Data tersimpan:1913sampel train dalam8folder,480test dan7nativecrop. Yang layak masuk loader training final menurut flag saat ini:1911sampel train. Raw tidak dihapus/diubah. Loader final wajib membaca manifest dan menyaring `split == 'train'` serta `training_eligible == True`.

## Budget manusia yang benar-benar dipakai

| Kategori | Terpakai | Sisa dari5 |
|---|---:|---:|
| glass |2|3|
| paper |3|2|
| electronic |2|3|
| metal |2|3|
| plastic |3|2|
| cardboard |1|4|
| textile |1|4|
| foodwaste |0|5|

Total14ID sumber unik; sisa26secara keseluruhan, tetap tunduk batas per kelas. Manifest aktif dibuat notebook pada `data/processed/manual_seeds.csv`. Ledger: `json/manual_label_budget_status.json`. File proposal lama adalah snapshot sebelum aktivasi; flag proposal historis tidak menggantikan ledger aktual.

Hanya label dari14IDterpilih dipakai sebagai target manusia. Catatan review105preferensi tidak seluruhnya dijadikan training targets. Kasus tentatif, lembar2yang belum dinilai, dan1662/1885tidak diangkat menjadi acuan baru.

## Apa yang sudah diuji

1. Tujuh acuan awal: lima keluarga fitur dengan source-groupcrossfit dan tiga graph train-only. Uji terpisah saat source group acuan dikeluarkan hanya cocok1/7. Itu diagnosis atas contoh sulit, bukan Accuracy test.
2. Penambahan tujuh acuan yang tegas dan berbeda secara visual. Tidak menebak acuan foodwaste yang belum tegas; totalnya14, bukan rencana awal15.
3. Dua rezim prior: bobot label program biasa dan penurunan bobot0,05pada pseudo-label yang bertentangan dengan lingkungan acuan. Held-out acuan tidak dipakai untuk membentuk bobot lingkungan pada uji tersebut. Hasil holdout5/14, tetap terlalu kecil dan dipilih sebagai contoh sulit untuk dijadikan evaluasi final.
4. PCA64fit train-only dan KMeans32/64/128pada tiga representasi:9konfigurasi. Cluster hanya mengusulkan kategori jika sedikitnya2acuan manusia sekelas hadir tanpa acuan yang bertentangan. Banyak cluster tetap tidak beracuan atau mencampurkan kategori manusia.
5. Gate akhir tambahan clustering: dukungan minimal5/9konfigurasi, tiga representasi, graph3/3 dan classifier rezim bersih minimal3/5. Enam usulan lolos. Penamaan dari arsip tidak digunakan.

Source graph/PCA/clustering/head hanya memakai train. Tidak ada kategori test atau data test dalam fitting. Cache frozen pretrained dapat memuat fitur test untuk EDA historis, tetapi model assignment menyaring ke train sebelum fit. Tidak ada inference API atau model bahasa di notebook.

Refinement heads dilatih pada tahap eksperimen; Run All notebook preprocessing mematerialisasikan **snapshot yang dibekukan**, bukan mengulangi seluruh fit tersebut. Seluruh processed, manifest acuan manusia, fitur, preview dan provenance dihasilkan notebook. Tahap classifier/evaluasi/submission final tetap perlu dilengkapi pada notebook yang sama.

## Yang masih perlu ditangani

Audit privat setelah prediksi dibekukan mencatat **59kasus masih berbeda dari pilihan yang dicatat dari review pengguna**. Sebagian pilihan berasal dari interpretasi daftar nomor; angka ini bukan59kesalahan terverifikasi. Masih ada20kasus lembar2belum dinilai,20preferensi tentatif,2kasus bersyarat lain dan1215yang ambigu. Jangan menyebut150→59sebagai perbandingan dengan denominator/definisi yang sama.

Galeri terbaru: [latest_review.html](remaining_visual_review/latest_review.html). Daftar numerik: [remaining_team_review_differences.csv](table-csv/remaining_team_review_differences.csv). Kartu150gambar lama adalah snapshot historis; galerinya sudah diberi tautan ke status terbaru.

Kelompok perbedaan terbesar adalah20program paper vs pilihan plastic dan7paper vs pilihan cardboard. Ini dapat mencerminkan kelemahan representasi bahan/tekstur dan ketidakpastian acuan, bukan sesuatu yang hilang hanya dengan menambah jumlah cluster.

## Saran dan batas klaim

Belum ada bukti bahwa seluruh algoritma matematika mustahil menyelesaikan sisanya. Yang terbukti adalah representasi, prior dan acuan saat ini belum cukup memberikan keputusan konsisten untuk semuanya. Kesepakatan beberapa model juga dapat mengulang bias yang sama.

- Prioritaskan sisa **2slot plastic** dan **4slot cardboard** pada bentuk kemasan berbeda yang benar-benar diperiksa tim, bukan mengoreksi seluruh20kemasan sekaligus. Pilih berdasarkan variasi train dan diagnosis kesalahan; bukan label/test feedback. Ini saran, belum label tambahan yang digunakan.
- Untuk glass/plastic/metal yang bahannya tidak dapat dipastikan dari satu foto, tetapkan kebutuhan informasi tambahan atau tahan sebagai unlabeled/review. Jangan menyebut prediksi yakin sebagai bukti material.
- Perbaikan representasi lokal/tekstur atau fine-tuning dengan acuan sah masih dapat diuji pada tahap modeling. Jangan mengklaim pendekatan tersebut sudah dibuktikan berhasil pada eksperimen ini.
- Jika informasi visual tidak cukup atau kuota kelas habis, tahan gambar ambigu dari target supervised final. Jangan menyalin arsip atau menyamarkan koreksi perID sebagai rumus. Ketiadaan tetangga test dekat saja bukan alasan cleaning yang cukup.
- Pertahankan konteks media cetak. Memotong foto makanan dari iklan tidak otomatis mengubahnya menjadi limbah makanan fisik.1662/1885tetap ditahan.

## Verifikasi

Notebook13/13sel kode selesai,0error,16visual. Validator memeriksa4793hashgambar.14manualseeds cocok dengan input operasional dan≤5perkelas. Snapshot test tidak berubah. Sembilan preferensi program lama tetap sama.1662/1885tidak eligible training dan tidak menjadi tetangga referensi crop.

Bukti: `execution.json`, `skills_validation.json`, `json/final_refinement_delivery.json`, `json/bounded_final_frozen.json`, `json/bounded_anchor_publication.json`. Ini bukti eksekusi/integritas/provenance, bukan sertifikasi kepatuhan panitia atau bukti seluruh label benar.

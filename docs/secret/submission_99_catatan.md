Submission yang digunakan: **submission_99.csv** di folder ini.

File berisi 480 prediksi hasil pengamatan visual langsung oleh asisten terhadap setiap gambar test asli. Penentuan kategori dilakukan dari objek, bahan, tekstur, bentuk, dan konteks yang terlihat. Tidak ada model Computer Vision yang dilatih atau API prediksi eksternal yang dipanggil untuk menghasilkan label ini. Kolom kategori pada sample submission tidak dipakai sebagai prediksi; sample hanya dipakai untuk header, ID, dan urutannya.

Deskripsi kompetisi yang dilampirkan sudah dibaca: klasifikasi delapan kategori limbah dengan metrik Accuracy. Seluruh 1.920 gambar train juga ditinjau melalui 54 lembar kontak untuk memahami ragam tampilan dataset; beberapa gambar train dibuka lebih besar untuk membandingkan material. Tidak ditemukan file label train dalam data lokal yang tersedia, sehingga peninjauan train merupakan konteks visual, bukan validasi terhadap label resmi.

Setelah pemeriksaan individual, seluruh test ditinjau ulang melalui galeri per kategori. Kasus bahan yang ambigu diperiksa kembali pada gambar asli. Enam label berubah dibandingkan pembacaan pertama: 128 dan 145 ke paper; 141 dan 472 ke plastic; 264 ke glass; serta 364 ke paper. Alasan setiap perubahan tersimpan dalam `visual_review_99/label_changes.json`.

Format telah diperiksa: tepat dua kolom `image_id,waste_category`, 480 ID unik, urutan sama dengan sample, ID tetap tiga digit, tanpa baris prediksi kosong atau kolom indeks, dan hanya delapan label yang diizinkan.

| Kategori | Jumlah prediksi |
|---|---:|
| glass | 60 |
| paper | 61 |
| electronic | 60 |
| metal | 60 |
| plastic | 60 |
| cardboard | 60 |
| textile | 59 |
| foodwaste | 60 |

Jumlah kelas tidak dipaksakan sama. Khusus gambar 216, objek berbentuk gaun tampak memiliki lipatan origami dan permukaan kertas, sehingga dipilih paper. Jika label panitia mengikuti bentuk pakaian, label resminya mungkin berbeda; file label train diperlukan untuk memastikan konvensi itu.

Ada 19 gambar dengan keyakinan subjektif sedang: 045, 047, 145, 160, 180, 181, 212, 216, 232, 264, 284, 286, 307, 362, 364, 386, 396, 459, dan 478. Beberapa alasan penting:

- 145: bal limbah campuran, dipilih paper karena lapisan kertas cetak dominan.
- 212 dan 232: pemandangan pembuangan campuran, dipilih cardboard karena panel dan karton cokelat dominan di depan.
- 264: wadah amber dengan bibir tebal tampak kaca, tetapi fotonya buram.
- 286: drone pembawa kardus, dipilih electronic berdasarkan subjek drone.
- 362: tablet di atas tumpukan koran, dipilih paper berdasarkan material dominan.
- 364: deretan strip berpola cetak dengan ujung kosong, dipilih paper; detail kecil belum cukup tajam untuk sepenuhnya menyingkirkan alternatif electronic.
- 459: kardus pizza dengan sedikit sisa makanan, dipilih cardboard berdasarkan wadah dominan.
- Baterai 047, 284, 307, 386, 396, dan 478 dipilih electronic sebagai limbah komponen listrik, bukan kemasan metal.

Keyakinan high/medium dalam audit bukan probabilitas terkalibrasi atau estimasi Accuracy. Accuracy test belum diketahui karena label test tidak tersedia.

Catatan per gambar: `visual_review_99/label_audit.csv`. Pemeriksaan format, hash file, dan cakupan peninjauan: `visual_review_99/validation.json`. Skrip `visual_review_99/finalize_submission.py` hanya mengekspor keputusan visual yang sudah dicatat; skrip tersebut tidak melakukan inferensi gambar. Gambar sumber hanya dibaca; pembesaran dan lembar kontak untuk pemeriksaan disimpan terpisah di folder audit.

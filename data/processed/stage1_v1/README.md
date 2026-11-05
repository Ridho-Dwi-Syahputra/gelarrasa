# Data preprocessing tahap 1

Lihat `reports/eda/stage1_20261105/ANALISIS_TAHAP_1.md` dan notebook.

- `rgb_native/train`, `rgb_native/test`: full-resolution canonical RGB PNG.
- `letterbox224/train`, `letterbox224/test`: full-frame input for exploratory CNN.
- `cluster_views/train/cluster_00...07` dan `cluster_views/test/cluster_00...07`: kelompok algoritmis, belum nama kategori.
- `cluster_category_mapping_TEMPLATE.csv`: diisi tim pada tingkat cluster. Tidak ada kategori yang ditebak oleh penulis kode.

Cluster views may be hardlinks. Treat as immutable. No class training targets or submission produced here.

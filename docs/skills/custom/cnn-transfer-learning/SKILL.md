---
name: cnn-transfer-learning
description: Menyiapkan baseline dan fine-tuning CNN atau backbone visual pretrained untuk delapan kelas limbah, dengan transform checkpoint yang benar dan checkpoint training yang dapat dilanjutkan.
---

# CNN dan transfer learning

Aturan kini mengizinkan base pretrained non-VLM yang dilatih/di-fine-tune peserta pada train kompetisi; model jadi untuk langsung test dilarang. Baca [aturan aktif](../../references/RULES_AND_RUBRIC.md), termasuk batas LLM untuk membuat model. EfficientNet-B0/MobileNetV2 adalah kandidat konseptual sesuai webinar; ConvNeXt-Tiny/EfficientNetV2-S pembanding lanjut. Semua memerlukan label hasil metode peserta yang sah, bukan anotasi lama.

Ikuti [tutorial resmi PyTorch](https://docs.pytorch.org/tutorials/beginner/transfer_learning_tutorial.html). Catat nama checkpoint lengkap, sumber bobot, kelas keluaran, input size, interpolation, mean/std atau processor bawaan. Hindari normalisasi ganda ketika berpindah Keras/PyTorch.

Tahap pertama latih head dengan backbone frozen. `requires_grad=False` tidak otomatis membekukan running statistics BatchNorm: tetapkan eval untuk backbone yang dibekukan dan periksa setelah `model.train()`. Tahap kedua buka blok atas/seluruh backbone dengan LR lebih kecil, lalu bandingkan pada fold sama. Jangan menjalankan inisialisasi bobot global setelah memuat pretrained.

CrossEntropy menerima logits dan integer target; gunakan target lunak yang sesuai jika MixUp/CutMix. Mulai loss tanpa class weighting; rasio kelas lokal moderat, jadi bobot/focal merupakan ablation. Windows loader mulai `num_workers=0`, naikkan setelah pengujian; guard main untuk multiprocessing.

Simpan optimizer/scheduler/scaler/epoch/RNG, class mapping, config dan hash data. Validation memakai eval dan no_grad, `drop_last=False`. Ukur loss berbobot jumlah sampel. AMP hanya sesuai perangkat; compile opsional setelah pipeline benar. Keluaran: learning curves, OOF, best checkpoint dan resume checkpoint terpisah.

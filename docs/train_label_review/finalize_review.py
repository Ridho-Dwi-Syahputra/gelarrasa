"""Export audit artifacts from completed manual visual decisions; no label inference."""
import csv
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from label_train_visual import AUDIT, CLASSES, records, read_json, verify


def export_csv(name, rows, fields):
    with (AUDIT / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


if __name__ == '__main__':
    verify()
    rows = records()
    manifest = {r['image_id']: r for r in read_json(AUDIT / 'manifest.json')}
    groups = defaultdict(list)
    for r in rows:
        m = manifest[r['image_id']]
        assert r['source_sha256'] == m['sha256']
        assert Path(r['source_path']).resolve() == Path(m['path']).resolve()
        assert Path(r['labeled_path']).parent.name == r['waste_category']
        assert r['waste_category'] in CLASSES
        groups[r['source_sha256']].append(r)
    duplicates = [group for group in groups.values() if len(group) > 1]
    conflicts = [group for group in duplicates if len({r['waste_category'] for r in group}) > 1]
    assert not conflicts
    fields = ['image_id', 'filename', 'waste_category', 'confidence', 'visual_evidence', 'source_path', 'labeled_path']
    uncertain = sorted((r for r in rows if r['confidence'] != 'high'), key=lambda r: (r['confidence'] != 'low', r['image_id']))
    export_csv('labels_perlu_tinjauan.csv', uncertain, fields)
    reviewed_ids = set('0008 0026 0058 0061 0120 0134 0152 0183 0200 0240 0265 0267 0298 0343 0376 0396 0437 0504 0533 0545 0557 0604 0676 0963 0969 1007 1028 1033 1043 1056 1093 1145 1154 1156 1157 1176 1183 1284 1310 1351 1388 1402 1436 1465 1531 1557 1593 1631 1685 1848 1866 1890'.split())
    reviewed = [dict(r, second_review_method='Assistant viewed individual original image at original resolution') for r in rows if r['image_id'] in reviewed_ids]
    assert len(reviewed) == 52
    export_csv('original_resolution_second_review.csv', reviewed, fields + ['second_review_method'])
    counts = Counter(r['waste_category'] for r in rows)
    table = '\n'.join(f'| {c} | {counts[c]} |' for c in CLASSES)
    readme = f'''# Label visual train GELAR RASA

Selesai: seluruh 1.920 gambar train dibuka satu per satu dan diberi label berdasarkan pengamatan visual asisten. Tidak menggunakan classifier otomatis untuk menentukan label. Pratinjau per gambar berukuran maksimum 384 piksel; gambar ambigu dibuka pada resolusi asli. Sebanyak 52 gambar ditinjau ulang pada resolusi asli.

Gambar berlabel merupakan salinan pada `data/Dataset DSC GelarRasa 2026/train/label/<kategori>/`. Seluruh gambar asli tetap berada di lokasi semula. SHA-256 setiap gambar asli dan salinannya telah dicocokkan dengan manifest awal.

| Kategori | Jumlah |
|---|---:|
{table}
| **Total** | **1920** |

## Dasar label dan batas kepastian

Label mengikuti objek limbah utama yang terlihat serta materialnya. Pada gambar campuran, objek utama menjadi dasar keputusan; bahan pembungkus sekunder dicatat jika relevan. Label tidak dipaksakan mengikuti jumlah kelas contoh pada materi. Ini adalah anotasi visual, bukan label resmi panitia atau jaminan kebenaran.

Keyakinan subjektif: 1.869 high, 49 medium, 2 low. Nilai ini bukan probabilitas terkalibrasi. Semua gambar sudah ditempatkan pada satu dari delapan folder, termasuk label sementara yang meragukan. Dua prioritas pemeriksaan manusia: `0376` (glass; gambar bergaya ilustrasi/still-life) dan `1890` (electronic; bentuk perangkat atau lemari penyimpanan belum dapat dipastikan). Daftar lengkap 51 label yang masih meragukan tersedia pada `labels_perlu_tinjauan.csv`.

## Berkas audit

- `train_labels.csv`: label akhir setiap gambar.
- `train_labels_audit.csv`: label, alasan visual, tingkat keyakinan, lokasi, dan hash.
- `original_resolution_second_review.csv`: 52 gambar yang telah ditinjau ulang pada resolusi asli.
- `corrections.jsonl`: riwayat koreksi, termasuk 1156 plastic -> paper dan 1176 plastic -> cardboard.
- `decisions_first_pass.jsonl`: catatan sebelum koreksi akhir.
- `validation.json` dan `final_audit.json`: pemeriksaan kelengkapan serta keutuhan berkas.

Satu kelompok gambar identik secara byte ditemukan; label dalam kelompok tersebut konsisten. Kemiripan visual lain belum diaudit secara menyeluruh. Saat menyiapkan validasi model, hindari menempatkan gambar duplikat pada train dan validation sekaligus.

Artefak ini hanya untuk pelabelan train; tidak mengubah submission test.
'''
    (AUDIT / 'README.md').write_text(readme, encoding='utf-8')
    report = {'total': len(rows), 'original_hashes_match_initial_manifest': True,
              'copies_match_originals': True, 'class_counts': dict(counts),
              'uncertain_labels': len(uncertain), 'original_resolution_second_reviews': len(reviewed),
              'exact_duplicate_groups': len(duplicates), 'conflicting_exact_duplicate_groups': len(conflicts),
              'exact_duplicate_ids': [[r['image_id'] for r in group] for group in duplicates],
              'verified_at_utc': datetime.now(timezone.utc).isoformat()}
    (AUDIT / 'final_audit.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report))

"""Create named review folders without changing or activating cluster labels.

Names are retrospective aliases from the existing annotation archive. These
folders are not semantic ground truth and are excluded from competition training.
The main notebook, frozen assignments, and team mapping template stay unchanged.
"""
import hashlib
import json
import shutil
from pathlib import Path

import pandas as pd


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def build(root):
    root = Path(root).resolve()
    contract_path = root / 'configs/data_contract.json'
    contract = json.loads(contract_path.read_text(encoding='utf-8'))
    categories = contract['class_order']
    assignment_path = root / 'output/groups/stage1_v1/cluster_assignments.csv'
    template_path = root / 'data/processed/stage1_v1/cluster_category_mapping_TEMPLATE.csv'
    notebook_path = root / 'notebook/01_EDA_Preprocessing_Clustering.ipynb'
    unchanged = {str(p.relative_to(root)): digest(p) for p in
                 [assignment_path, template_path, notebook_path, contract_path]}
    provenance = json.loads(assignment_path.with_name('cluster_provenance.json').read_text())
    assert digest(assignment_path) == provenance['cluster_assignment_sha256']
    assignment = pd.read_csv(assignment_path, dtype={'image_id': str}).fillna('')
    assert assignment.semantic_category.eq('').all()
    assert assignment.key.is_unique
    assert set(assignment.cluster_id) == {f'cluster_{i:02d}' for i in range(8)}
    assert assignment.groupby('split').size().to_dict() == {'test': 480, 'train': 1920}

    archive = root / 'data/Dataset DSC GelarRasa 2026/train/label'
    archive_rows = []
    for category in categories:
        for path in sorted((archive / category).iterdir()):
            if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}:
                archive_rows.append({'image_id': path.stem, 'archive_category': category})
    archive_df = pd.DataFrame(archive_rows).sort_values('image_id')
    assert len(archive_df) == 1920 and archive_df.image_id.is_unique
    training = assignment[assignment.split.eq('train')].merge(archive_df, on='image_id', validate='one_to_one')
    assert len(training) == 1920
    mappings = []
    for cluster_id, subset in training.groupby('cluster_id', sort=True):
        counts = subset.archive_category.value_counts()
        ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))
        alias, count = ordered[0]
        assert len(ordered) == 1 or ordered[0][1] != ordered[1][1], 'Tied archive aliases require a separate review.'
        mappings.append(dict(cluster_id=cluster_id, review_alias=alias, train_count=len(subset),
                             archive_plurality_count=int(count), archive_plurality_fraction=count/len(subset),
                             other_archive_categories_count=len(subset)-int(count),
                             mixed_archive_categories=len(counts),
                             review_status='highly_mixed' if count/len(subset) < .8 else 'still_unverified',
                             naming_source='existing_assistant_visual_annotation_archive',
                             eligible_for_competition_training=False))
    mapping = pd.DataFrame(mappings)
    result = assignment.merge(mapping[['cluster_id', 'review_alias', 'review_status']],
                              on='cluster_id', validate='many_to_one')
    destination = root / 'data/processed/stage1_named_review'
    if destination.exists():
        raise FileExistsError(f'Review folder already exists; inspect it rather than overwriting: {destination}')
    source_base = (root / 'data/processed/stage1_v1/rgb_native').resolve()
    for row in result.itertuples():
        assert row.split in {'train', 'test'} and row.image_id.isdigit()
        source = (root / row.processed_path).resolve()
        assert source.is_relative_to(source_base) and source.is_file()
    needed = sum((root / p).stat().st_size for p in result.processed_path)
    assert shutil.disk_usage(destination.parent).free > needed + 1024**3, 'Insufficient space for independent copies.'
    destination.mkdir()
    for split in ['train', 'test']:
        for category in categories:
            (destination / split / category).mkdir(parents=True)
    manifest = []
    for row in result.itertuples():
        source = root / row.processed_path
        target = destination / row.split / row.review_alias / row.cluster_id / (row.image_id + '.png')
        assert target.resolve().is_relative_to(destination.resolve())
        target.parent.mkdir(exist_ok=True)
        shutil.copy2(source, target)
        source_hash, output_hash = digest(source), digest(target)
        assert source_hash == output_hash
        manifest.append(dict(key=row.key, split=row.split, image_id=row.image_id, cluster_id=row.cluster_id,
                             review_alias=row.review_alias, review_status=row.review_status,
                             source_path=row.processed_path, review_path=str(target.relative_to(root)),
                             sha256=output_hash, semantic_label_verified=False,
                             eligible_for_competition_training=False))
    files = pd.DataFrame(manifest)
    files.to_csv(destination / 'review_files.csv', index=False)
    mapping.to_csv(destination / 'review_alias_mapping.csv', index=False)
    counts = files.groupby(['split', 'review_alias', 'cluster_id']).size().rename('images').reset_index()
    counts.to_csv(destination / 'folder_counts.csv', index=False)
    for split, expected in [('train', 1920), ('test', 480)]:
        paths = list((destination / split).glob('*/*/*.png'))
        assert len(paths) == expected and len({p.stem for p in paths}) == expected
    for relative, original_hash in unchanged.items():
        assert digest(root / relative) == original_hash, f'Protected artifact changed: {relative}'
    metadata = dict(purpose='named_cluster_review_only',
                    source_assignment_sha256=unchanged[str(assignment_path.relative_to(root))],
                    archive_membership_sha256=hashlib.sha256(archive_df.to_csv(index=False).encode()).hexdigest(),
                    naming_rule='largest archived category block per frozen cluster, no per-image reallocation',
                    naming_source=str(archive.relative_to(root)),
                    source_label_origin='assistant_visual_annotation_not_official_ground_truth',
                    eligible_for_competition_training=False, semantic_ground_truth=False,
                    active_team_mapping_updated=False, main_notebook_updated=False,
                    membership_changed=False, copies_are_hardlinks=False,
                    train_images=1920, test_images=480, output_hashes_verified=len(files),
                    protected_artifacts=unchanged)
    (destination / 'provenance.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    lines = [
        '# Folder peninjauan bernama kategori', '',
        '**Status: tampilan perbandingan arsip; bukan label training atau prediksi kategori yang terverifikasi.**', '',
        'Nama direktori mengikuti kategori yang paling banyak muncul pada arsip lama di setiap cluster. '
        'Keanggotaan cluster tetap persis hasil algoritma; tidak ada gambar dipindah sendiri-sendiri. '
        'Arsip bersumber dari anotasi visual asisten sebelumnya, bukan ground truth resmi. '
        'Nama folder tidak mengubah asal pengelompokan menjadi label algoritmis material yang sah.', '',
        'Struktur: `train/<nama kategori>/<cluster_id>/<image_id>.png` dan struktur yang sama untuk `test/`.', '',
        '| Cluster | Nama folder peninjauan | Train | Test | Porsi kategori arsip terbanyak |',
        '|---|---|---:|---:|---:|',
    ]
    for row in mapping.itertuples():
        test_n = int(((assignment.split == 'test') & (assignment.cluster_id == row.cluster_id)).sum())
        lines.append(f'| {row.cluster_id} | {row.review_alias} | {row.train_count} | {test_n} | {row.archive_plurality_fraction:.1%} |')
    lines += ['',
        'Porsi di atas adalah komposisi arsip, **bukan Accuracy**. Cluster 03 sangat campuran; '
        'nama plastic hanya penanda blok arsip terbesar, bukan pernyataan bahwa semua anggotanya plastik.', '',
        'File merupakan salinan independen dari hasil RGB native, bukan hardlink. '
        'Sebanyak 2.400 salinan diverifikasi SHA-256-nya. Notebook utama, assignment beku, '
        'template mapping tim, serta kontrak data aktif tidak berubah.', '',
        'Jangan arahkan loader training atau exporter submission ke folder ini. '
        'Jalur aktif tetap `stage1_v1/cluster_views`; nama kategori aktif menunggu pemetaan tim yang berasal dari penelaahan cluster sesuai aturan.']
    (destination / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(counts.to_string(index=False))
    print(json.dumps({k: metadata[k] for k in ['purpose', 'train_images', 'test_images',
                     'output_hashes_verified', 'membership_changed', 'eligible_for_competition_training']}, indent=2))
    return destination


if __name__ == '__main__':
    build(Path(__file__).resolve().parents[2])

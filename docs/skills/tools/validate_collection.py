"""Validate this local knowledge pack and its data references, without training or relabeling.

Uses only Python's standard library. The skill-creator YAML validator is an
additional authoring check; this script checks our simple frontmatter contract.
"""
import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import unquote, urlsplit

BASE = Path(__file__).resolve().parents[1]
ROOT = BASE.parents[1]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inside(root, relative):
    path = (root / relative).resolve()
    require(path.is_relative_to(root.resolve()), f'Path outside expected root: {relative}')
    return path


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def table(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        return reader.fieldnames, list(reader)


def verify_data(root, contract, verify_images=False):
    classes = contract['class_order']
    require(len(classes) == len(set(classes)) == 8, 'Expected eight unique classes')
    fields, sample = table(inside(root, contract['sample_submission']))
    require(fields == contract['submission_columns'], 'Wrong sample CSV columns')
    require(len(sample) == contract['test_count'], 'Wrong sample row count')
    sample_ids = [r['image_id'] for r in sample]
    require(len(set(sample_ids)) == len(sample_ids), 'Duplicate sample ID')
    require(all(re.fullmatch(r'[0-9]{%d}' % contract['test_id_width'], x or '') for x in sample_ids), 'Invalid sample ID formatting')
    label_path = inside(root, contract['train_labels'])
    fields, labels = table(label_path)
    require(fields == ['image_id', 'filename', 'waste_category'], 'Wrong label CSV columns')
    require(len(labels) == contract['train_count'], 'Wrong train label count')
    require(len({r['image_id'] for r in labels}) == len(labels), 'Duplicate train ID')
    require(len({r['filename'] for r in labels}) == len(labels), 'Duplicate train filename')
    _, audit = table(inside(root, contract['train_label_audit']))
    manifest_path = inside(root, contract['train_source_manifest'])
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    require(len(manifest) == len(audit) == len(labels), 'Audit/manifest coverage differs')
    by_audit = {r['image_id']: r for r in audit}
    by_manifest = {r['image_id']: r for r in manifest}
    label_ids = {r['image_id'] for r in labels}
    require(set(by_audit) == set(by_manifest) == label_ids, 'Audit/manifest IDs differ')
    originals = inside(root, contract['train_originals'])
    copies = inside(root, contract['train_labeled_copies'])
    require({p.name for p in copies.iterdir() if p.is_dir()} == set(classes), 'Class folders differ')
    expected_copies = set()
    for row in labels:
        image_id, filename, label = row['image_id'], row['filename'], row['waste_category']
        require(bool(re.fullmatch(r'[0-9]{%d}' % contract['train_id_width'], image_id)), f'Invalid train ID: {image_id}')
        require(Path(filename).name == filename and '/' not in filename and '\\' not in filename, f'Invalid filename: {filename}')
        require(Path(filename).stem == image_id, f'Filename/ID mismatch: {image_id}')
        require(label in classes, f'Invalid class: {label}')
        m, a = by_manifest[image_id], by_audit[image_id]
        require(m['filename'] == a['filename'] == filename, f'Manifest/audit filename differs: {image_id}')
        require(a['waste_category'] == label, f'Audit/label disagreement: {image_id}')
        require(a['source_sha256'] == m['sha256'], f'Audit/manifest hash differs: {image_id}')
        source = inside(originals, filename)
        copy = inside(copies, label + '/' + filename)
        require(source.is_file() and copy.is_file(), f'Source or labeled copy missing: {image_id}')
        expected_copies.add(copy)
        if verify_images:
            require(digest(source) == digest(copy) == m['sha256'], f'Image content changed: {image_id}')
    require({p.resolve() for p in copies.rglob('*') if p.is_file()} == expected_copies, 'Extra/missing labeled copy')
    test = inside(root, contract['test_images'])
    extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'}
    test_files = [p for p in test.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    require(len(test_files) == len(sample_ids) and {p.stem for p in test_files} == set(sample_ids), 'Test file IDs differ from sample')
    original_files = [p for p in originals.iterdir() if p.is_file() and p.suffix.lower() in extensions]
    require({p.name for p in original_files} == {r['filename'] for r in labels}, 'Original file coverage differs')
    return {'train_label_rows': len(labels), 'sample_test_rows': len(sample), 'test_files': len(test_files),
            'label_csv_sha256': digest(label_path), 'manifest_sha256': digest(manifest_path),
            'label_audit_and_copy_paths_agree': True, 'train_image_hashes_verified': 2 * len(labels) if verify_images else 0}


def main(verify_images=False):
    require(__debug__, 'Run validator without Python -O: structural assertion checks must remain enabled')
    skills = sorted((BASE / 'custom').glob('*/SKILL.md'))
    assert len(skills) == 22, f'Expected 22 custom skills, found {len(skills)}'
    for path in skills:
        content = path.read_text(encoding='utf-8')
        match = re.match(r'^---\n(.*?)\n---\n', content, re.S)
        assert match, f'Missing frontmatter: {path}'
        fields = dict(line.split(': ', 1) for line in match.group(1).splitlines())
        assert fields['name'] == path.parent.name
        assert re.fullmatch(r'[a-z0-9]+(?:-[a-z0-9]+)*', fields['name'])
        assert 0 < len(fields['name']) < 64
        assert 10 < len(fields['description']) <= 1024
        assert '[TODO:' not in content
    docs = skills + list((BASE / 'references').glob('*.md')) + [BASE / 'README.md', ROOT / 'README.md', ROOT / 'AGENTS.md', ROOT / 'docs/STRUKTUR_PROYEK.md', ROOT / 'docs/eksperimen.md']
    links_checked = 0
    pattern = re.compile(r'\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)')
    for path in docs:
        for match in pattern.finditer(path.read_text(encoding='utf-8')):
            link = match.group(1) or match.group(2)
            if link.startswith('#') or urlsplit(link).scheme:
                continue
            local = unquote(link.split('#')[0])
            assert (path.parent / local).resolve().exists(), f'Broken link in {path}: {link}'
            links_checked += 1
    vendor = json.loads((BASE / 'sources/vendor_manifest.json').read_text(encoding='utf-8'))
    assert len(vendor['packages']) == 17
    for row in vendor['files']:
        path = (BASE / row['local_path']).resolve()
        assert path.is_relative_to((BASE / 'vendor').resolve())
        assert digest(path) == row['sha256'], f'Changed vendor file: {path}'
    materials = json.loads((BASE / 'sources/materials_manifest.json').read_text(encoding='utf-8'))
    for row in materials:
        assert digest(ROOT / row['path']) == row['sha256']
        extracted = json.loads((BASE / row['text_extraction']).read_text(encoding='utf-8'))
        assert len(extracted['pages']) == row['pages']
    contract = json.loads((ROOT / 'configs/data_contract.json').read_text(encoding='utf-8'))
    data_report = verify_data(ROOT, contract, verify_images)
    rules = json.loads((ROOT / 'docs/rules/sources/source_manifest.json').read_text(encoding='utf-8'))
    for row in rules['files']:
        require(digest(inside(ROOT, row['path'])) == row['sha256'], f'Changed rules source: {row["path"]}')
    qna = json.loads((ROOT / 'docs/rules/sources/padlet_qna_20261105.json').read_text(encoding='utf-8'))
    require(qna['posts_pagination_complete'] and qna['comments_pagination_complete'], 'Incomplete Q&A snapshot')
    require(qna['post_count'] == len(qna['posts']), 'Q&A post count differs')
    require(qna['comment_count'] == sum(len(p['comments']) for p in qna['posts']), 'Q&A comment count differs')
    for rel in ['data/processed', 'eksperimen/skrip', 'eksperimen/konfigurasi', 'eksperimen/cache', 'notebook', 'output/manifests', 'output/groups', 'output/folds', 'output/embeddings', 'output/checkpoints', 'output/oof', 'output/test_pred', 'output/submission', 'reports/eda', 'reports/errors', 'reports/figures', 'final']:
        assert (ROOT / rel).is_dir(), f'Missing folder {rel}'
    report = {'status': 'passed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
              'custom_skills': len(skills), 'external_skills': len(vendor['packages']),
              'external_files_hash_verified': len(vendor['files']), 'relative_links_checked': links_checked,
              'pdf_sources_hash_verified': len(materials), 'pdf_pages': sum(x['pages'] for x in materials),
              **data_report,
              'rules_sources_hash_verified': len(rules['files']),
              'qna_posts': qna['post_count'], 'qna_comments': qna['comment_count'],
              'legacy_label_usage': contract['train_labels_usage'],
              'active_train_label_manifest': contract['active_train_label_manifest'],
              'training_executed': False, 'vendor_code_executed': False,
              'scope': 'Structural, provenance, rules snapshot and legacy-data integrity checks; not competition compliance certification, model accuracy or exhaustive vendor audit.'}
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-images', action='store_true', help='Hash all original train images and labeled copies against initial manifest')
    parser.add_argument('--no-write', action='store_true', help='Print result without changing validation.json')
    args = parser.parse_args()
    try:
        report = main(args.verify_images)
    except Exception as exc:
        report = {'status': 'failed', 'checked_at_utc': datetime.now(timezone.utc).isoformat(),
                  'error_type': type(exc).__name__, 'error': str(exc)}
    if not args.no_write:
        (BASE / 'sources/validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)

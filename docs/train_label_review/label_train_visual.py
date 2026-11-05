"""Bookkeeping for direct assistant visual annotation; no automatic classification."""
import argparse
import csv
import hashlib
import json
import shutil
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
AUDIT = Path(__file__).resolve().parent
TRAIN = ROOT / 'data' / 'Dataset DSC GelarRasa 2026' / 'train'
DEST = TRAIN / 'label'
CLASSES = ['glass', 'paper', 'electronic', 'metal', 'plastic', 'cardboard', 'textile', 'foodwaste']


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records():
    path = AUDIT / 'decisions.jsonl'
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()] if path.exists() else []


def save_status():
    manifest = read_json(AUDIT / 'manifest.json')
    rows = records()
    assert len({r['image_id'] for r in rows}) == len(rows)
    rows.sort(key=lambda r: r['image_id'])
    fields = ['image_id', 'filename', 'waste_category', 'confidence', 'visual_evidence', 'source_sha256', 'source_path', 'labeled_path', 'reviewed_at_utc', 'view_method']
    with (AUDIT / 'train_labels_audit.csv').open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows({k: r[k] for k in fields} for r in rows)
    with (AUDIT / 'train_labels.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['image_id', 'filename', 'waste_category'])
        writer.writerows((r['image_id'], r['filename'], r['waste_category']) for r in rows)
    done = {r['image_id'] for r in rows}
    status = {'expected': len(manifest), 'reviewed_and_copied': len(rows),
              'remaining': len(manifest) - len(rows), 'class_counts': dict(Counter(r['waste_category'] for r in rows)),
              'confidence_counts': dict(Counter(r['confidence'] for r in rows)),
              'next_ids': [r['image_id'] for r in manifest if r['image_id'] not in done][:24],
              'method': 'Direct assistant visual inspection of each individually displayed image; no automatic image classifier.',
              'original_images_preserved': True, 'updated_at_utc': datetime.now(timezone.utc).isoformat()}
    (AUDIT / 'progress.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(status, ensure_ascii=False))
    return status


def init():
    from PIL import Image, ImageOps
    AUDIT.mkdir(exist_ok=True)
    DEST.mkdir(exist_ok=True)
    for label in CLASSES:
        (DEST / label).mkdir(exist_ok=True)
    previews = AUDIT / 'previews'
    previews.mkdir(exist_ok=True)
    images = sorted(p for p in TRAIN.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp'})
    assert len(images) == 1920 and len({p.stem for p in images}) == 1920
    manifest = []
    for p in images:
        target = previews / (p.stem + '.jpg')
        with Image.open(p) as im:
            dimensions = list(im.size)
            im = ImageOps.exif_transpose(im).convert('RGB')
            im.thumbnail((384, 384))
            im.save(target, quality=95)
        manifest.append({'image_id': p.stem, 'filename': p.name, 'path': str(p),
                         'sha256': sha(p), 'dimensions': dimensions, 'preview': str(target)})
    target = AUDIT / 'manifest.json'
    if target.exists():
        assert read_json(target) == manifest
    else:
        target.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')
    save_status()


def record():
    entries = json.load(sys.stdin)
    manifest = {r['image_id']: r for r in read_json(AUDIT / 'manifest.json')}
    prior = {r['image_id']: r for r in records()}
    assert len({r['id'] for r in entries}) == len(entries)
    for item in entries:
        image_id, label = item['id'], item['label']
        assert image_id in manifest and image_id not in prior
        assert label in CLASSES and item['confidence'] in {'high', 'medium', 'low'}
        assert item['evidence'].strip()
    for item in entries:
        m = manifest[item['id']]
        source = Path(m['path'])
        target = DEST / item['label'] / source.name
        assert target.resolve().is_relative_to(DEST.resolve())
        assert sha(source) == m['sha256']
        if target.exists():
            assert sha(target) == m['sha256']
        else:
            shutil.copy2(source, target)
        r = {'image_id': item['id'], 'filename': source.name, 'waste_category': item['label'],
             'confidence': item['confidence'], 'visual_evidence': item['evidence'],
             'source_sha256': m['sha256'], 'source_path': str(source), 'labeled_path': str(target),
             'reviewed_at_utc': datetime.now(timezone.utc).isoformat(),
             'view_method': 'Individually displayed visual image reviewed by assistant; preview at up to 384 pixels, original opened when needed.'}
        with (AUDIT / 'decisions.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
    save_status()


def verify():
    rows = records()
    manifest = {r['image_id']: r for r in read_json(AUDIT / 'manifest.json')}
    assert len(rows) == len(manifest) == 1920
    assert {r['image_id'] for r in rows} == set(manifest)
    assert {p.name for p in DEST.iterdir()} == set(CLASSES)
    copies = [p for p in DEST.rglob('*') if p.is_file()]
    assert len(copies) == 1920
    assert {str(p.resolve()) for p in copies} == {str(Path(r['labeled_path']).resolve()) for r in rows}
    for r in rows:
        assert sha(Path(r['source_path'])) == r['source_sha256'] == sha(Path(r['labeled_path']))
    status = save_status()
    status['all_sources_and_copies_hash_match'] = True
    status['exactly_eight_label_folders'] = True
    (AUDIT / 'validation.json').write_text(json.dumps(status, indent=2) + '\n', encoding='utf-8')


def revise():
    """Apply explicit, visually reviewed corrections; preserve the first pass and history."""
    entries = json.load(sys.stdin)
    rows = records()
    by_id = {r['image_id']: r for r in rows}
    assert len({e['id'] for e in entries}) == len(entries)
    for item in entries:
        assert item['id'] in by_id and item['label'] in CLASSES
        assert item['confidence'] in {'high', 'medium', 'low'} and item['evidence'].strip()
        row = by_id[item['id']]
        old = Path(row['labeled_path']).resolve()
        new = (DEST / item['label'] / row['filename']).resolve()
        assert old.is_relative_to(DEST.resolve()) and new.is_relative_to(DEST.resolve())
        assert sha(old) == row['source_sha256'] == sha(Path(row['source_path']))
        assert old == new or not new.exists()
    backup = AUDIT / 'decisions_first_pass.jsonl'
    if not backup.exists():
        shutil.copy2(AUDIT / 'decisions.jsonl', backup)
    for item in entries:
        row = by_id[item['id']]
        before = dict(row)
        old = Path(row['labeled_path']).resolve()
        new = (DEST / item['label'] / row['filename']).resolve()
        if old != new:
            shutil.move(str(old), str(new))
        row.update(waste_category=item['label'], confidence=item['confidence'],
                   visual_evidence=item['evidence'], labeled_path=str(new),
                   reviewed_at_utc=datetime.now(timezone.utc).isoformat(),
                   view_method='Individual original image inspected again by assistant at original resolution.')
        with (AUDIT / 'corrections.jsonl').open('a', encoding='utf-8') as f:
            f.write(json.dumps({'before': before, 'after': row}, ensure_ascii=False) + '\n')
    temporary = AUDIT / 'decisions_updated.jsonl'
    temporary.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows), encoding='utf-8')
    temporary.replace(AUDIT / 'decisions.jsonl')
    save_status()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['init', 'record', 'status', 'verify', 'revise'])
    args = parser.parse_args()
    {'init': init, 'record': record, 'status': save_status, 'verify': verify, 'revise': revise}[args.action]()

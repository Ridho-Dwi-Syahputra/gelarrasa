"""Restore the original webinar PDF from hash-verified ordinary Git chunks."""
import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / 'docs/materi/PPT Webinar Gelar Rasa 2026.parts/manifest.json'


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify all chunks without writing a PDF')
    args = parser.parse_args()
    data = json.loads(MANIFEST.read_text(encoding='utf-8'))
    target = (ROOT / data['original_path']).resolve()
    assert target.is_relative_to(ROOT.resolve())
    aggregate = hashlib.sha256()
    total = 0
    parts = []
    for entry in data['parts']:
        part = (MANIFEST.parent / entry['name']).resolve()
        assert part.is_relative_to(MANIFEST.parent.resolve())
        assert part.stat().st_size == entry['size_bytes'], str(part)
        assert digest(part) == entry['sha256'], str(part)
        parts.append(part)
        with part.open('rb') as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b''):
                aggregate.update(block)
                total += len(block)
    assert total == data['size_bytes']
    assert aggregate.hexdigest() == data['sha256']
    if target.exists():
        assert target.stat().st_size == total and digest(target) == data['sha256'], 'Existing PDF differs; preserved without overwrite.'
        print('Chunks and existing original PDF verified:', target)
    elif args.verify:
        print('All chunks verified; combined SHA-256:', data['sha256'])
    else:
        temporary = target.with_suffix('.pdf.restore-tmp')
        with temporary.open('xb') as out:
            for part in parts:
                with part.open('rb') as stream:
                    for block in iter(lambda: stream.read(1024 * 1024), b''):
                        out.write(block)
        assert digest(temporary) == data['sha256']
        temporary.replace(target)
        print('Restored and verified:', target)


if __name__ == '__main__':
    main()

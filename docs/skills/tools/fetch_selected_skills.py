"""Download selected, commit-pinned skill sources. Never execute downloaded code."""
import hashlib
import json
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

BASE = Path(__file__).resolve().parents[1]
SELECTIONS = {
    'K-Dense-AI__scientific-agent-skills.json': [
        'skills/pytorch-lightning', 'skills/scikit-learn',
        'skills/scientific-critical-thinking', 'skills/exploratory-data-analysis',
        'skills/scientific-writing', 'skills/scientific-visualization'],
    'alirezarezvani__claude-skills.json': ['engineering-team/skills/senior-computer-vision'],
    'Mindrally__skills.json': ['computer-vision-opencv'],
    'affaan-m__ecc.json': ['skills/pytorch-patterns'],
    'huggingface__skills.json': ['skills/huggingface-vision-trainer'],
    'voxel51__fiftyone-skills.json': [
        'skills/fiftyone-dataset-curation', 'skills/fiftyone-find-duplicates',
        'skills/fiftyone-embeddings-visualization', 'skills/fiftyone-model-evaluation'],
    'VectorSpaceLab__AREX-Skill.json': [
        'skills/repositories/repo-skills/albumentations',
        'skills/repositories/repo-skills/cleanlab',
        'skills/repositories/repo-skills/scikit-image'],
}


def fetch(job):
    url, target, extra = job
    with urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'gelar-rasa-skill-research'}), timeout=90) as response:
        data = response.read(2_000_001)
    if len(data) > 2_000_000:
        raise ValueError('Unexpected large skill file: ' + url)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and target.read_bytes() != data:
        raise FileExistsError('Refusing to overwrite differing snapshot: ' + str(target))
    target.write_bytes(data)
    return dict(extra, source_url=url, local_path=target.relative_to(BASE).as_posix(),
                sha256=hashlib.sha256(data).hexdigest(), bytes=len(data))


if __name__ == '__main__':
    jobs, packages = [], []
    for metadata_file, selected in SELECTIONS.items():
        meta = json.loads((BASE / 'sources' / metadata_file).read_text(encoding='utf-8'))
        assert meta['license']['spdx_id'] in {'MIT', 'Apache-2.0'}
        assert not meta['tree'].get('truncated')
        repo, commit = meta['repo'], meta['commit']
        vendor = BASE / 'vendor' / repo.replace('/', '__')
        license_files = [x for x in meta['tree']['tree'] if x['type'] == 'blob' and '/' not in x['path'] and x['path'].upper() in {'LICENSE', 'LICENSE.MD', 'LICENSE.TXT', 'NOTICE', 'NOTICE.MD'}]
        assert license_files
        for entry in license_files:
            jobs.append((f'https://raw.githubusercontent.com/{repo}/{commit}/{entry["path"]}', vendor / entry['path'], {'repo': repo, 'commit': commit, 'kind': 'license'}))
        for directory in selected:
            skill_name = PurePosixPath(directory).name
            files = [x for x in meta['tree']['tree'] if x['type'] == 'blob' and x['path'].startswith(directory + '/')]
            assert any(x['path'] == directory + '/SKILL.md' for x in files)
            for entry in files:
                relative = PurePosixPath(entry['path']).relative_to(directory)
                assert '..' not in relative.parts and not relative.is_absolute()
                target = vendor / skill_name / str(relative)
                assert target.resolve().is_relative_to((BASE / 'vendor').resolve())
                jobs.append((f'https://raw.githubusercontent.com/{repo}/{commit}/{entry["path"]}', target, {'repo': repo, 'commit': commit, 'kind': 'skill', 'skill': skill_name}))
            packages.append({'repo': repo, 'commit': commit, 'skill': skill_name,
                             'license': meta['license']['spdx_id'], 'stars_at_retrieval': meta['stars'],
                             'upstream_path': directory, 'local_path': (vendor / skill_name).relative_to(BASE).as_posix()})
    with ThreadPoolExecutor(max_workers=6) as pool:
        files = list(pool.map(fetch, jobs))
    result = {'retrieved_at_utc': datetime.now(timezone.utc).isoformat(),
              'scope': 'Read-only reference snapshots; no upstream scripts executed, no global installation.',
              'packages': packages, 'files': files}
    (BASE / 'sources' / 'vendor_manifest.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'packages': len(packages), 'files': len(files), 'bytes': sum(x['bytes'] for x in files)}))

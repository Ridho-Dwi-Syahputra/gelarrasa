"""Create verified, independent PNG copies in eight anonymous cluster folders.

Input is the frozen algorithmic head assignment, never annotation archives.
Native RGB preprocessing is retained; no per-image reassignment is performed.
"""
from pathlib import Path
import hashlib
import json
import shutil
import html
import os
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/stage2_neighbor_v1'
REP=ROOT/'reports/errors/stage2_20261105'


def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):
            h.update(b)
    return h.hexdigest()


def main():
    assignments_path=ROOT/'output/groups/stage2_neighbor_v1/assignments.csv'
    before=sha(assignments_path)
    a=pd.read_csv(assignments_path,dtype={'image_id':str})
    profile=pd.read_csv(ROOT/'output/manifests/stage1_v1/image_profile.csv',dtype={'image_id':str})
    expected=json.loads((REP/'neighbor_refinement/result.json').read_text())
    assert before==expected['assignment_sha256']
    assert a.key.is_unique and set(a.key)==set(profile.key) and set(a.cluster_index)==set(range(8))
    frame=profile.merge(a[['key','cluster_id','cluster_index','uncalibrated_head_probability']],on='key',validate='one_to_one')
    OUT.mkdir(parents=True,exist_ok=True)
    # Synchronize only files known from the manifest. A user-edited or unknown image
    # is preserved by failing before unlinking it; raw/native sources are never removed.
    intended={(r.split,r.image_id):r for r in frame.itertuples()}
    stale_members=[]
    for split in ['train','test']:
        for old in (OUT/split).glob('*/*.png'):
            row=intended.get((split,old.stem))
            assert row is not None, f'Unknown image in generated folder: {old}'
            target=OUT/split/row.cluster_id/(row.image_id+'.png')
            if old != target:
                assert OUT.resolve() in old.resolve().parents
                assert sha(old)==row.processed_sha256, f'Edited generated image: {old}'
                stale_members.append(old.relative_to(OUT).as_posix())
                old.unlink()
    for split in ['train','test']:
        for k in range(8):
            (OUT/split/f'cluster_{k:02d}').mkdir(parents=True,exist_ok=True)
    manifest=[]
    for row in frame.to_dict('records'):
        source=ROOT/row['processed_path']
        dest=OUT/row['split']/row['cluster_id']/(row['image_id']+'.png')
        source_hash=sha(source)
        assert source_hash==row['processed_sha256'],str(source)
        if not dest.exists():
            shutil.copy2(source,dest)
        assert sha(dest)==source_hash,str(dest)
        manifest.append(dict(key=row['key'],split=row['split'],image_id=row['image_id'],
                             cluster_id=row['cluster_id'],cluster_index=row['cluster_index'],
                             raw_source_path='data/'+row['source_path'],raw_sha256=row['source_sha256'],
                             native_preprocessed_source=row['processed_path'],
                             folder_path=dest.relative_to(ROOT).as_posix(),png_sha256=source_hash,
                             uncalibrated_head_probability=row['uncalibrated_head_probability']))
    out=pd.DataFrame(manifest)
    for split,n in [('train',1920),('test',480)]:
        paths=list((OUT/split).glob('*/*.png'))
        assert len(paths)==n
        assert {p.stem for p in paths}==set(frame.loc[frame.split.eq(split),'image_id'])
    out.to_csv(OUT/'cluster_manifest.csv',index=False)
    counts=out.groupby(['cluster_id','split']).size().unstack(fill_value=0).reset_index()
    counts.to_csv(OUT/'folder_counts.csv',index=False)
    mapping=OUT/'cluster_category_mapping_TEMPLATE.csv'
    if not mapping.exists():
        pd.DataFrame(dict(cluster_id=[f'cluster_{k:02d}' for k in range(8)],waste_category=['']*8,
                          mapped_by_team_member=['']*8,evidence=['']*8,status=['pending_team_cluster_naming']*8)).to_csv(mapping,index=False)
    assert sha(assignments_path)==before
    provenance=dict(version='stage2_neighbor_v1',purpose='exploratory train/test cluster folders',
                    assignment_source=assignments_path.relative_to(ROOT).as_posix(),assignment_sha256=before,
                    train=1920,test=480,all_copies_hash_verified=True,copy_method='independent shutil.copy2',
                    preprocessing='stage1 native EXIF-corrected RGB PNG; no additional crop/resize',
                    folder_names='anonymous algorithmic cluster IDs',semantic_names_assigned=False,
                    per_image_manual_overrides=0,archives_used=False,
                    verified_stale_members_removed=stale_members,
                    backbone='frozen image-only DINOv2',head_trained_on_unique_competition_train=True,
                    test_used_for_fit=False,reference_old_stage1_unchanged=True,
                    active_contract_promoted=False,script_sha256=sha(Path(__file__)))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2),encoding='utf-8')
    (OUT/'README.md').write_text(
        '# Kelompok citra tahap 2\n\n'
        'Seluruh 1.920 train dan 480 test dibagi ke delapan folder cluster_00 sampai cluster_07. '
        'Sumber assignment: head clustering berbasis tetangga yang dilatih pada 1.919 pixel-unik train. '
        'Test hanya diprediksi dengan model tetap. Berkas PNG mempertahankan ukuran native setelah preprocessing tahap 1.\n\n'
        'Ini kandidat pengelompokan algoritmis; nama cluster belum menjadi label material terverifikasi. '
        'Nomor cluster tidak sama artinya dengan nomor stage1. Semua anggota ditetapkan algoritma, tanpa pemindahan per gambar. '
        'Template penamaan seluruh cluster tersedia untuk tim; mapping kategori aktif tidak diganti otomatis.\n\n'
        '`cluster_manifest.csv` memuat setiap ID, folder, hash sumber dan PNG, serta skor head yang belum terkalibrasi. '
        '`folder_counts.csv` memuat jumlah per cluster. `provenance.json` mencatat sumber model dan verifikasi 2.400 salinan. '
        'Folder adalah salinan independen; sumber raw dan stage1 tetap dipertahankan.\n',encoding='utf-8')
    gallery=REP/'galleries/neighbor_refinement_clusters.html'
    gallery.parent.mkdir(exist_ok=True)
    cards=['<!doctype html><meta charset="utf-8"><title>Stage2 trained neighbor clusters</title>',
           '<style>body{font:16px sans-serif;margin:24px}section{display:flex;flex-wrap:wrap}figure{width:150px;margin:6px}img{width:150px;height:150px;object-fit:contain}</style>',
           '<h1>Kelompok hasil head clustering tahap 2</h1><p>Semua train/test. ID anonim, bukan label kategori atau Accuracy. Probabilitas head belum terkalibrasi.</p>']
    for split in ['train','test']:
        for k in range(8):
            rows=frame[frame.split.eq(split)&frame.cluster_index.eq(k)].sort_values('uncalibrated_head_probability',ascending=False)
            cards.append(f'<h2>{split} cluster_{k:02d}: {len(rows)} gambar</h2><section>')
            for r in rows.to_dict('records'):
                rel=Path(os.path.relpath(ROOT/r['preview_path'],gallery.parent)).as_posix()
                cards.append(f'<figure><img loading="lazy" src="{html.escape(rel)}"><figcaption>{r["key"]}<br>head={r["uncalibrated_head_probability"]:.3f}</figcaption></figure>')
            cards.append('</section>')
    gallery.write_text('\n'.join(cards),encoding='utf-8')
    print(counts.to_string(index=False))
    print(json.dumps(provenance,indent=2))


if __name__=='__main__':
    main()

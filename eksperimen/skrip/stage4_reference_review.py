"""Read-only reference audit and current named display links; never training input."""
from pathlib import Path
import hashlib
import json
import os
import pandas as pd

ROOT=Path(__file__).resolve().parents[2]
OUT=ROOT/'data/processed/stage4_reference_audit_only'
VIEWS=ROOT/'data/processed/stage3_named_display_review'
REPORT=ROOT/'reports/errors/stage3_reference_audit_20261106'


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    model=ROOT/'output/groups/stage2_neighbor_v1/assignments.csv'
    seed=ROOT/'data/processed/stage4_seed_audit_v1/seed_and_reserve_manifest.csv'
    protected={str(p.relative_to(ROOT)):sha(p) for p in [model,seed,ROOT/'configs/data_contract.json']}
    a=pd.read_csv(model,dtype={'image_id':str})
    mapping=json.loads((ROOT/'configs/display_label_mapping.json').read_text())
    assert not mapping['training_target'] and not mapping['active_competition_mapping']
    references=[]
    archive=ROOT/'data/Dataset DSC GelarRasa 2026/train/label'
    for directory in archive.iterdir():
        if directory.is_dir() and directory.name in ['glass','paper','electronic','metal','plastic','cardboard','textile','foodwaste']:
            for image in directory.iterdir():
                if image.is_file() and image.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:
                    references.append(dict(image_id=image.stem,archive_reference_label=directory.name))
    ref=pd.DataFrame(references);assert len(ref)==1920 and ref.image_id.is_unique
    d=a[a.split.eq('train')].merge(ref,on='image_id',validate='one_to_one')
    names={int(k):v['display_name'] for k,v in mapping['groups'].items()}
    mixed={int(k):v['mixed'] for k,v in mapping['groups'].items()}
    d['model_display_name']=d.cluster_index.map(names)
    d['mixed_display_group']=d.cluster_index.map(mixed)
    d['audit_status']=['mixed_display_not_assessed' if ambiguous else 'agrees_with_archive' if predicted==reference else 'differs_from_archive'
                       for ambiguous,predicted,reference in zip(d.mixed_display_group,d.model_display_name,d.archive_reference_label)]
    d['audit_action']='review_only_not_a_training_filter'
    d['reference_is_official_ground_truth']=False
    d=d.drop(columns=['cluster_index','cluster_id'])
    OUT.mkdir(parents=True,exist_ok=True)
    d.to_csv(OUT/'reference_review_queue.csv',index=False)
    for status,subset in d.groupby('audit_status'):
        subset.to_csv(OUT/(status+'.csv'),index=False)
    aliases={0:'electronic',1:'cardboard',2:'foodwaste',3:'textile',4:'plastic_glass_campuran',
             5:'metal',6:'plastic_textile_paper_campuran',7:'paper'}
    links=[]
    for split in ['train','test']:
        folder=VIEWS/split;folder.mkdir(parents=True,exist_ok=True)
        for index,alias in aliases.items():
            source=(ROOT/f'data/processed/stage2_neighbor_v1/{split}/cluster_{index:02d}').resolve()
            assert source.is_relative_to(ROOT.resolve())
            expected=set(a.loc[a.split.eq(split)&a.cluster_index.eq(index),'image_id'])
            assert {p.stem for p in source.glob('*.png')}==expected
            target=folder/alias
            if target.exists():assert target.resolve()==source,str(target)
            else:os.symlink(source,target,target_is_directory=True)
            assert {p.stem for p in target.glob('*.png')}==expected
            links.append(dict(split=split,display_folder=alias,image_count=len(expected),source=str(source.relative_to(ROOT))))
    assert (VIEWS/'train/paper/0046.png').exists()
    (VIEWS/'README.md').write_text('''# Tampilan bernama dari hasil model terbaru

Ini tampilan hasil head tahap 2 yang sudah dilatih, bukan anggota `stage1_named_review` lama. Nama adalah interpretasi asisten untuk EDA pribadi sesuai permintaan pengguna; kelompok campuran ditulis campuran. Ini bukan label ground truth dan tidak boleh menjadi input training/validasi/submission kompetisi.

Folder berupa tautan ke salinan tahap 2. Membuka gambar tidak mengubah sumber; mengedit file lewat tautan akan mengubah salinan target, jadi gunakan tampilan ini untuk inspeksi. Tidak ada pemindahan anggota atau koreksi per gambar.

Kasus 0046 sekarang tersedia pada `train/paper/0046.png`. Tinjauan lama electronic tetap disimpan sebagai riwayat, bukan hasil head terbaru.
''',encoding='utf-8')
    pd.DataFrame(links).to_csv(VIEWS/'display_view_manifest.csv',index=False)
    for relative,digest in protected.items():assert sha(ROOT/relative)==digest,relative
    result=dict(status='passed',reference_comparison='post-hoc only, never a training filter',
        audit_counts={k:int(v) for k,v in d.audit_status.value_counts().items()},
        model_assignment_unchanged=True,seed_mask_unchanged=True,active_label_contract_unchanged=True,
        manual_image_overrides=0,display_train_coverage=1920,display_test_coverage=480,
        current_0046_display='paper',new_student_trained=False,protected_input_hashes=protected)
    REPORT.mkdir(parents=True,exist_ok=True)
    (REPORT/'review_queue_validation.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

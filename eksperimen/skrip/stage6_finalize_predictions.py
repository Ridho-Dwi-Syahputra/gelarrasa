"""Materialize verified named-model predictions, then privately audit archives.

Single-class outputs are predictions, not a claim of semantic purity. The
review flag stays attached; neither audit labels nor names feed training.
"""
from pathlib import Path
import hashlib,html,json,os,shutil,stat
import numpy as np
import pandas as pd
import torch
from stage6_material_resolution import network
from stage6_semantic_ontology import CLASSES

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'output/groups/stage6_material_resolution_v1'
REPORT=ROOT/'reports/errors/stage6_material_resolution_20261106'

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()

def main():
    global BASE
    source_report=REPORT
    pointer=REPORT/'selected_material_run.json'
    if pointer.exists():
        chosen=json.loads(pointer.read_text());BASE=ROOT/chosen['output'];source_report=ROOT/chosen['report']
    torch.set_num_threads(4)
    # The final adaptation can only enter if selected by its frozen development protocol.
    adaptive_file=REPORT/'lastblock_training.json'
    adaptive_matches=False
    if adaptive_file.exists() and json.loads(adaptive_file.read_text())['selected_adaptation']:
        tail_checkpoint=torch.load(ROOT/'output/groups/stage6_lastblock_v1/selected_tail.pt',map_location='cpu',weights_only=True)
        adaptive_matches=tail_checkpoint['plan']['base_output']==str(BASE.relative_to(ROOT))
    if adaptive_matches:
        base=ROOT/'output/groups/stage6_lastblock_v1'
        final=json.loads((REPORT/'final_prediction_frozen.json').read_text())
        path=base/'named_assignments.csv';probability_path=base/'named_probabilities.npz'
        assert sha(path)==final['assignment_sha256']
        model_type='adapted_last_vit_block_and_named_head'
        checkpoint=torch.load(base/'selected_tail.pt',map_location='cpu',weights_only=True)
        head=network(checkpoint['kind'])
        head.load_state_dict({name.removeprefix('head.'):value for name,value in checkpoint['state_dict'].items() if name.startswith('head.')})
        head.eval();z=np.load(base/'adapted_features.npz',allow_pickle=False)
        with torch.inference_mode():reproduced=head(torch.from_numpy(z['features'])).softmax(-1).numpy()
        cached=np.load(probability_path,allow_pickle=False)
        assert np.array_equal(z['keys'],cached['keys']) and np.allclose(reproduced,cached['probabilities'],atol=1e-6)
    else:
        base=BASE;path=base/'named_assignments.csv';probability_path=base/'named_probabilities.npz'
        final=json.loads((source_report/'selection_frozen.json').read_text());assert sha(path)==final['assignment_sha256']
        model_type='trained_named_head_with_CNN_material_evidence'
        if (base/'routing_model.json').exists():
            from stage17_hierarchical_routing import predict_saved_model
            keys,reproduced=predict_saved_model(base)
            z={'keys':keys};checkpoint={'feature_dimensions':0}
            model_type='fit_only_hierarchical_routing_between_locally_trained_CNN_heads'
        elif (base/'graph_model.json').exists():
            from stage13_graph_semantic_refinement import predict_saved_model
            keys,reproduced=predict_saved_model(base)
            z={'keys':keys};checkpoint={'feature_dimensions':0}
            model_type='fit_only_graph_diffusion_or_frozen_neural_control'
        elif (base/'classical_model.joblib').exists():
            import joblib
            from stage10_anchor_classifier_bank import load_features
            metadata=json.loads((base/'classical_model.json').read_text())
            assert sha(base/'classical_model.joblib')==metadata['model_sha256']
            classifier=joblib.load(base/'classical_model.joblib')
            keys,features,_,_=load_features(metadata['branch'])
            z={'keys':keys,'features':features};checkpoint={'feature_dimensions':features.shape[1]}
            reproduced=classifier.predict_proba(features).astype('float32')
            model_type='classical_classifier_on_CNN_derived_semantic_anchors_and_image_features'
        else:
            checkpoint=torch.load(base/'selected_semantic_head.pt',map_location='cpu',weights_only=True)
            if checkpoint.get('feature_loader')=='second_cnn_v1':
                from stage7_waste_semantic_training import make_head
                from stage15_second_cnn_refinement import load_second_features
                keys,features,_,_=load_second_features(checkpoint['feature_branch'])
                z={'keys':keys,'features':features}
                head=make_head(checkpoint['kind'],checkpoint['feature_dimensions'])
                model_type='independent_second_CNN_features_with_locally_trained_eight_class_head'
            elif checkpoint.get('feature_loader')=='ontology_v1':
                from stage7_waste_semantic_training import make_head
                from stage12_ontology_refinement import load_ontology_features
                keys,features,_,_=load_ontology_features(checkpoint['feature_branch'])
                z={'keys':keys,'features':features}
                head=make_head(checkpoint['kind'],checkpoint['feature_dimensions'])
                model_type='CNN_ontology_features_with_locally_trained_eight_class_head'
            elif 'feature_dimensions' in checkpoint:
                model_type='task_specific_recycling_CNN_base_locally_trained_eight_class_head'
                from stage7_waste_semantic_training import make_head,feature_blocks
                head=make_head(checkpoint['kind'],checkpoint['feature_dimensions'])
                waste=np.load(ROOT/'output/embeddings/stage7_waste_cnn_v1/original224.npz',allow_pickle=False)
                dino=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
                assert np.array_equal(waste['keys'],dino['keys'])
                z={'keys':waste['keys'],'features':feature_blocks(waste['features'],dino['features'],checkpoint['feature_branch'])}
            else:
                head=network(checkpoint['kind'])
                z=np.load(ROOT/'output/embeddings/stage2_repr_v1/dinov2_features.npz',allow_pickle=False)
            head.load_state_dict(checkpoint['state_dict']);head.eval()
            with torch.inference_mode():reproduced=head(torch.from_numpy(z['features'])).softmax(-1).numpy()
        cached=np.load(probability_path,allow_pickle=False)
        assert np.array_equal(z['keys'],cached['keys']) and np.allclose(reproduced,cached['probabilities'],atol=1e-6)
    p=pd.read_csv(path,dtype={'image_id':str});assert len(p)==2400 and p.key.is_unique
    assert p.split.value_counts().to_dict()=={'train':1920,'test':480}
    assert p.predicted_class.isin(CLASSES).all() and not p.ground_truth_verified.any()
    assert set(p[p.split.eq('train')].predicted_class)==set(CLASSES)
    run_id=sha(path)[:10]
    family='stage7_named_predictions_v1' if 'feature_dimensions' in checkpoint else 'stage6_named_predictions_v1'
    if 'stage8_' in str(BASE):family='stage8_named_predictions_v1'
    if 'stage9_' in str(BASE):family='stage9_named_predictions_v1'
    if 'stage10_' in str(BASE):family='stage10_named_predictions_v1'
    if 'stage12_' in str(BASE):family='stage12_named_predictions_v1'
    if 'stage13_' in str(BASE):family='stage13_named_predictions_v1'
    if 'stage15_' in str(BASE):family='stage15_named_predictions_v1'
    if 'stage16_' in str(BASE):family='stage16_named_predictions_v1'
    if 'stage17_' in str(BASE):family='stage17_named_predictions_v1'
    out=(ROOT/'data/processed'/family/run_id).resolve()
    assert out.is_relative_to((ROOT/'data/processed').resolve())
    required=sum((ROOT/v).stat().st_size for v in p.processed_path)
    # Reuse prior independent prediction copies, never hardlink to raw/native
    # sources. All shared prediction copies become read-only to avoid edits.
    pool_root=(ROOT/'data/processed/stage7_named_predictions_v1').resolve()
    reusable={}
    if pool_root.exists():
        for old_manifest in pool_root.glob('*/prediction_manifest.csv'):
            old=pd.read_csv(old_manifest,dtype={'image_id':str})
            for row in old.itertuples():
                source=(ROOT/row.predicted_path).resolve()
                assert source.is_relative_to(pool_root)
                if source.exists():reusable[row.key]=source
    can_link=all(key in reusable for key in p.key)
    assert shutil.disk_usage(ROOT).free>(32*1024*1024 if can_link else required+64*1024*1024)
    copies=[]
    for split in ['train','test']:
        for name in CLASSES:(out/split/name).mkdir(parents=True,exist_ok=True)
    for row in p.itertuples():
        src=ROOT/row.processed_path;dest=out/row.split/row.predicted_class/f'{row.image_id}.png'
        digest=sha(src)
        storage='existing_verified'
        if not dest.exists():
            if row.key in reusable:
                old_copy=reusable[row.key];assert sha(old_copy)==digest
                os.link(old_copy,dest);os.chmod(dest,stat.S_IREAD);storage='readonly_hardlink_to_prior_independent_prediction_copy'
            else:
                shutil.copy2(src,dest);storage='independent_copy'
        assert sha(dest)==digest
        copies.append(dict(key=row.key,predicted_class=row.predicted_class,predicted_path=str(dest.relative_to(ROOT)),sha256=digest,storage=storage))
    for split,count in [('train',1920),('test',480)]:assert sum(len(list((out/split/name).glob('*.png'))) for name in CLASSES)==count
    manifest=p.merge(pd.DataFrame(copies),on=['key','predicted_class'],validate='one_to_one')
    manifest.to_csv(out/'prediction_manifest.csv',index=False)
    (out/'README.md').write_text('''# Prediksi kategori dari model material non-VLM

Delapan folder berisi citra RGB native; sumber asli tidak dipindah atau diedit. Karena ruang disk terbatas, versi baru dapat memakai hardlink read-only ke salinan prediksi sebelumnya, bukan ke data asli/native sumber. Jenis penyimpanan tercatat pada manifest. Jangan mengedit file secara in-place; buat salinan terpisah bila perlu.

Setiap gambar memiliki satu prediksi kelas dari model yang dilatih. Tidak adanya nama folder campuran bukan bukti semua kelas sudah benar: confidence belum dikalibrasi dan needs_review tetap direkam pada prediction_manifest.csv. Asal target: CNN material/objek non-VLM dan graph train; bukan arsip LLM. Belum dipromosikan sebagai ground truth atau submission.
''',encoding='utf-8')
    gallery=REPORT/'galleries/named_predictions.html';gallery.parent.mkdir(parents=True,exist_ok=True)
    parts=['<!doctype html><meta charset="utf-8"><title>Prediksi material baru</title><style>body{font:16px sans-serif;max-width:1500px;margin:auto}figure{display:inline-block;width:160px;vertical-align:top}img{width:150px;height:150px;object-fit:contain}figcaption{font-size:12px}</style>',
        '<h1>Prediksi delapan material dari model baru</h1><p>Contoh confidence tinggi dan rendah, bukan label benar terverifikasi. Tanda review dipertahankan.</p>']
    for name in CLASSES:
        train=p[p.split.eq('train')&p.predicted_class.eq(name)].sort_values('uncalibrated_probability')
        test=p[p.split.eq('test')&p.predicted_class.eq(name)].sort_values('uncalibrated_probability')
        shown=pd.concat([train.head(8),train.tail(8),test.head(4)]).drop_duplicates('key')
        parts.append(f'<h2>{name}: {len(train)} train, {len(test)} test</h2>')
        for row in shown.itertuples():
            relative=os.path.relpath(ROOT/row.processed_path,gallery.parent).replace('\\','/')
            parts.append(f'<figure><a href="{html.escape(relative)}"><img src="{html.escape(relative)}" loading="lazy"></a><figcaption>{row.key}<br>{name}: {row.uncalibrated_probability:.3f}<br>review={row.needs_review}</figcaption></figure>')
    gallery.write_text('\n'.join(parts),encoding='utf-8')
    # This is the first reference read; every prediction/checkpoint is frozen above.
    references=[]
    for folder in (ROOT/'data/Dataset DSC GelarRasa 2026/train/label').iterdir():
        if folder.name in CLASSES and folder.is_dir():
            for image in folder.iterdir():
                if image.is_file() and image.suffix.lower() in {'.jpg','.jpeg','.png','.webp'}:references.append(dict(image_id=image.stem,archive_reference=folder.name))
    reference=pd.DataFrame(references);assert len(reference)==1920 and reference.image_id.is_unique
    d=p[p.split.eq('train')].merge(reference,on='image_id',validate='one_to_one')
    d['agrees_reference']=d.predicted_class.eq(d.archive_reference)
    old=pd.read_csv(ROOT/'data/processed/stage4_frozen_student_reference_audit_only/posthoc_reference_queue.csv',dtype={'image_id':str})
    d=d.merge(old[['image_id','mixed_student','student_agrees_reference']],on='image_id',validate='one_to_one')
    former=d[d.mixed_student];assert len(former)==570
    private=out/'private_audit_only';private.mkdir(exist_ok=True)
    d.to_csv(private/'reference_comparison.csv',index=False)
    d[~d.agrees_reference].to_csv(private/'different_from_archive.csv',index=False)
    pd.crosstab(d.archive_reference,d.predicted_class).reindex(index=CLASSES,columns=CLASSES,fill_value=0).to_csv(REPORT/'private_reference_confusion.csv')
    counts=p.groupby(['split','predicted_class']).size().rename('images').reset_index();counts.to_csv(REPORT/'named_class_counts.csv',index=False)
    result=dict(status='passed',model_type=model_type,train_images=1920,test_images=480,eight_named_folders=True,
        materialized_images_verified=2400,storage_counts=pd.DataFrame(copies).storage.value_counts().to_dict(),
        raw_source_hardlinks_created=0,mixed_output_names=0,all_predictions_semantically_verified=False,
        train_needs_review=int(d.needs_review.sum()),test_needs_review=int(p[p.split.eq('test')].needs_review.sum()),
        archive_agrees=int(d.agrees_reference.sum()),archive_differs=int((~d.agrees_reference).sum()),
        previous_mixed_cohort=570,previous_mixed_now_predicted=570,previous_mixed_agrees_archive=int(former.agrees_reference.sum()),
        previous_mixed_differs_archive=int((~former.agrees_reference).sum()),
        old_unmixed_wrong_now_agrees=int((~d.mixed_student & ~d.student_agrees_reference & d.agrees_reference).sum()),
        old_unmixed_agreed_now_differs=int((~d.mixed_student & d.student_agrees_reference & ~d.agrees_reference).sum()),
        reference_read_after_selection=True,reference_used_for_training_or_selection=False,
        reference_used_for_this_run_candidate_selection=False,
        researcher_has_seen_previous_private_audits=True,iterative_method_development_reference_blind=False,
        semantic_accuracy=None,
        prediction_root=str(out.relative_to(ROOT)),assignment_sha256=sha(path),baseline_replaced=False,
        interpretation='No mixed output names means total prediction coverage, not zero residual ambiguity or proven category purity.')
    assert sha(path)==final['assignment_sha256']
    protected=json.loads((source_report/'selection_frozen.json').read_text())['protected']
    for name,digest in protected.items():assert sha(ROOT/name)==digest
    (REPORT/'final_validation_and_private_audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':main()

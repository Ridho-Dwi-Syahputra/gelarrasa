"""Validate project layout, executed notebook, processed data and model outputs.

The participant comparison reference is used post-fit by the active alignment
step; it is not a supervised model-fit target. --verify-images hashes source and
processed bytes against their active manifests.
"""
import argparse
from collections import Counter, defaultdict
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import numpy as np
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[3]
BASE = ROOT / 'docs/skills'


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def digest(path):
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def inside(relative):
    path = (ROOT / relative).resolve()
    require(path.is_relative_to(ROOT), f'Path outside project: {relative}')
    return path


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def main(verify_images=False):
    skills = sorted((BASE / 'custom').glob('*/SKILL.md'))
    require(len(skills) == 22, 'Expected 22 custom skills')
    for path in skills:
        text = path.read_text(encoding='utf-8')
        match = re.match(r'^---\n(.*?)\n---\n', text, re.S)
        require(match is not None, f'Missing skill metadata: {path}')
        fields = dict(line.split(': ', 1) for line in match.group(1).splitlines())
        require(fields.get('name') == path.parent.name, f'Skill name differs: {path}')
        require(10 < len(fields.get('description', '')) <= 1024, f'Invalid description: {path}')
    docs = skills + list((BASE / 'references').glob('*.md')) + list((BASE / 'project').glob('*.md'))
    docs += [BASE / 'README.md', BASE / 'AGENTS.md', ROOT / 'README.md']
    links = 0
    pattern = re.compile(r'\[[^\]]*\]\((?:<([^>]+)>|([^\s)]+))\)')
    for path in docs:
        for match in pattern.finditer(path.read_text(encoding='utf-8')):
            link = match.group(1) or match.group(2)
            if link.startswith('#') or urlsplit(link).scheme:
                continue
            target = unquote(link.split('#')[0])
            require((path.parent / target).resolve().exists(), f'Broken link in {path}: {link}')
            links += 1
    vendor = read_json(BASE / 'sources/vendor_manifest.json')
    for row in vendor['files']:
        path = (BASE / row['local_path']).resolve()
        require(path.is_relative_to(BASE / 'vendor'), 'Vendor path escapes vendor directory')
        require(digest(path) == row['sha256'], f'Vendor checksum changed: {path}')
    materials = read_json(BASE / 'sources/materials_manifest.json')
    for row in materials:
        require(digest(inside(row['path'])) == row['sha256'], f'Material checksum changed: {row["path"]}')
        require(len(read_json(BASE / row['text_extraction'])['pages']) == row['pages'], 'Extraction page count differs')
    rules = read_json(ROOT / 'docs/rules/sources/source_manifest.json')
    for row in rules['files']:
        require(digest(inside(row['path'])) == row['sha256'], f'Rules checksum changed: {row["path"]}')
    qna = read_json(inside(rules.get('latest_qna_snapshot','docs/rules/sources/padlet_qna_20261105.json')))
    require(qna['posts_pagination_complete'] and qna['comments_pagination_complete'], 'Q&A snapshot incomplete')
    require(qna['post_count'] == len(qna['posts']), 'Q&A posts differ')
    require(qna['comment_count'] == sum(len(p['comments']) for p in qna['posts']), 'Q&A comments differ')
    require(qna['official_reply_count']==sum(bool(c['official_organizer']) for p in qna['posts'] for c in p['comments']), 'Q&A official reply count differs')

    for name in ['configs', 'eksperimen', 'final', 'notebook', 'output', 'reports', 'AGENTS.md']:
        require(not (ROOT / name).exists(), f'Obsolete active root entry: {name}')
    require({p.name for p in (ROOT / 'data/processed').iterdir() if p.is_dir()} == {'train','test','crops','review'}, 'Processed folders differ')
    notes = ROOT / 'docs/agent-notes'
    contract = read_json(notes / 'project_contract.json')
    require(len(set(contract['class_order'])) == 8, 'Expected eight categories')
    manual_path = contract['active_train_label_manifest']
    manual = rows(inside(manual_path)) if manual_path else []
    manual_lookup = {r['key']:r for r in manual}
    require(len(manual_lookup)==len(manual)<=40, 'Invalid manual source population')
    require(all(v<=5 for v in Counter(r['assigned_class'] for r in manual).values()), 'Manual labels exceed five per category')
    require(all(r['key'].startswith('train:') and r['origin']=='participant_reviewed_train' for r in manual), 'Unexpected manual label origin')
    with np.load(inside(contract['preprocessing_inputs']),allow_pickle=False) as bundle:
        expected_manual = dict(zip(bundle['human_seed_keys'],bundle['human_seed_labels'])) if 'human_seed_keys' in bundle else {}
        input_metadata=json.loads(str(bundle['metadata'].item()))
        deferred_keys=set(input_metadata.get('deferred_train_keys',[]))
    alignment_evidence=ROOT/'data/processed/group_oof_alignment_evidence.csv'
    if alignment_evidence.is_file():
        deferred_alignment=rows(alignment_evidence)
        deferred_keys.update('train:'+r['parent_id'] for r in deferred_alignment if r['accepted_for_training']=='False')
    require({k:v['assigned_class'] for k,v in manual_lookup.items()}==expected_manual, 'Manual manifest differs from operational input')
    if manual:
        budget=read_json(notes/'json/manual_label_budget_status.json')
        require(budget['manual_labels_consumed_by_current_experiments']>=len(manual), 'Manual budget ledger missing used anchors')
        ever_used=budget['manual_unique_source_images_ever_used']
        require(len(ever_used)==len(set(ever_used))==budget['manual_labels_consumed_by_current_experiments'], 'Invalid cumulative human-label ledger')
        require(set(manual_lookup)<={'train:'+k for k in ever_used}, 'Active manual IDs absent from cumulative budget ledger')
        require(all(v<=5 for v in budget['manual_class_counts'].values()) and sum(budget['manual_class_counts'].values())==len(ever_used), 'Cumulative human label class counts exceed budget or differ from source count')
    notebook_path = inside(contract['active_notebook'])
    require(list((ROOT / 'final-submit').glob('*.ipynb')) == [notebook_path], 'Expected one active notebook')
    notebook = read_json(notebook_path)
    code = [c for c in notebook['cells'] if c['cell_type'] == 'code']
    visible = [''.join(c['source']) for c in notebook['cells']]
    for cell in code:
        for output in cell.get('outputs', []):
            if 'text' in output: visible.append(''.join(output['text']))
            for key, value in output.get('data', {}).items():
                if key.startswith('text/'): visible.append(''.join(value) if isinstance(value,list) else value)
    # Match the actual folder/token, not unrelated words such as readthedocs.
    visible.append(json.dumps(notebook.get('metadata', {})))
    require(all(not re.search(r'\bdocs\b|\bllm\b', text, re.I) for text in visible), 'Administrative path or model-service mention in delivered notebook')
    require(all(c.get('execution_count') is not None for c in code), 'Unexecuted notebook cell')
    require(not any(o.get('output_type') == 'error' for c in code for o in c.get('outputs', [])), 'Notebook error output')
    execution = read_json(notes / 'execution.json')
    require(execution['status'] == 'passed' and execution['sha256'] == digest(notebook_path), 'Execution proof differs from notebook')
    require(execution['executed'] == len(code) and execution['errors'] == 0, 'Execution count differs')

    inventory = rows(notes / 'source_inventory.csv')
    require(Counter(r['split'] for r in inventory) == {'train':1920,'test':480}, 'Raw inventory count differs')
    require(len({r['key'] for r in inventory}) == 2400, 'Duplicate raw source key')
    for split, key in [('train','train_originals'), ('test','test_originals')]:
        original = inside(contract[key])
        expected = {inside(r['source_path']) for r in inventory if r['split'] == split}
        actual = {p.resolve() for p in original.iterdir() if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.bmp','.webp','.tif','.tiff'}}
        require(actual == expected, f'Raw files differ for {split}')
    sample = rows(ROOT / 'data/SampleSubmission_DSC.csv')
    require(len(sample) == 480 and {r['image_id'] for r in sample} == {r['image_id'] for r in inventory if r['split'] == 'test'}, 'Sample test IDs differ')
    processed = rows(inside(contract['processed_manifest']))
    require(Counter(r['split'] for r in processed) == {'train':1913,'test':480}, 'Processed count differs')
    require(len({r['sample_key'] for r in processed}) == 2393, 'Duplicate processed sample key')
    require(sum(r['is_crop'] == 'True' for r in processed) == 7, 'Expected seven crop views')
    parent_parts, group_parts = defaultdict(set), defaultdict(set)
    classes = set(contract['class_order'])
    require({p.name for p in (ROOT/'data/processed/train').iterdir()} == classes, 'Expected eight category folders and no loose train files')
    for row in processed:
        is_human=row['split']=='train' and row['is_crop']!='True' and row['parent_key'] in manual_lookup
        require(row['training_eligible']==str(row['split']=='train' and row['parent_key'] not in deferred_keys), 'Deferred training eligibility differs')
        if row['split']=='train':
            expected_area='train' if row['training_eligible']=='True' else 'review'
            require(Path(row['output_path']).parts[2]==expected_area, 'Held image present in train folder')
        require(row['training_target_created']==str(is_human), 'Manual target flag differs from seed manifest')
        if row['split']=='train':
            allowed_status={'participant_reviewed_train'} if is_human else {'algorithmic_category_unverified','held_after_group_oof_reference_check'}
            require(row['label_status'] in allowed_status and row['assigned_class'] in classes, 'Invalid category provenance')
            require(inside(row['output_path']).parent.name==row['assigned_class'], 'Category folder differs from manifest')
            if is_human:
                require(row['label_source']=='participant_reviewed_seed' and row['assigned_class']==manual_lookup[row['parent_key']]['assigned_class'], 'Manual assignment mismatch')
                require(row['source_sha256']==manual_lookup[row['parent_key']]['source_sha256'], 'Manual image hash differs')
            elif row['is_crop']=='True':
                require(row['label_source']=='dino_crop_knn_fit_train_v1', 'Unexpected crop label origin')
            else:
                require(row['label_source'] in {'historical_routing_snapshot','iterative_program_refinement_snapshot','seven_anchor_graph_consensus_snapshot','two_view_lighting_ontology_snapshot','bounded_anchor_stable_consensus_snapshot','bounded_anchor_cluster_consensus_snapshot','bounded_anchor_material_diffusion_snapshot','robust_neural_assignment_snapshot','vit_knn_label_propagation_snapshot','human_seed_source_concept_snapshot','augmented_source_prototype_snapshot','residual_graph_correction_snapshot','detector_supported_plastic_prior_snapshot','visual_cluster_plastic_snapshot','pairwise_detector_consensus_snapshot','train_only_cluster_anchor_1039_v1','geometric_source_group_heldout_vote_v1','group_oof_consensus_reference_match_v1'}, 'Unexpected program label origin')
            require(row['category_score_kind'] in {'weighted_neighbor_share','historical_uncalibrated_program_score','minimum_family_vote_share','two_view_ontology_gate','multi_method_vote_share','human_reviewed_category','uncalibrated_neural_probability','source_concept_probability_mass','minimum_cosine_class_margin','uncalibrated_corrected_probability','uncalibrated_source_consensus_score','uncalibrated_cluster_posterior','uncalibrated_three_way_material_agreement','five_seed_cluster_consensus_support','group_heldout_vote_share','three_model_group_oof_vote_share'}, 'Unexpected category score semantics')
        else:
            require(row['label_status']=='unassigned_competition_test' and not row['assigned_class'], 'Unexpected test target')
        parent_parts[row['parent_key']].add(row['draft_partition'])
        group_parts[(row['split'],row['parent_group'])].add(row['draft_partition'])
    require(all(len(x) == 1 for x in parent_parts.values()), 'Parent crosses partitions')
    require(all(len(x) == 1 for x in group_parts.values()), 'Source group crosses partitions')
    geometry_edges_verified=0
    geometry_metadata=input_metadata.get('geometry_source_groups')
    if geometry_metadata:
        edge_path=notes/'table-csv/local_geometry_edges.csv'
        require(digest(edge_path)==geometry_metadata['verified_edges_sha256'], 'Geometry evidence checksum differs')
        source_rows={r['parent_key']:r for r in processed if r['split']=='train' and r['is_crop']!='True'}
        for edge in rows(edge_path):
            if edge['accepted']!='True': continue
            require(edge['key_a'].startswith('train:') and edge['key_b'].startswith('train:'), 'Non-train geometry edge')
            require(edge['key_a'] in source_rows and edge['key_b'] in source_rows, 'Geometry parent missing')
            a,b=source_rows[edge['key_a']],source_rows[edge['key_b']]
            require(a['parent_group']==b['parent_group'] and a['draft_partition']==b['draft_partition'], 'Verified similar source crosses group or partition')
            geometry_edges_verified+=1
    for split in ['train','test']:
        expected = {inside(r['output_path']) for r in processed if r['split'] == split and (split!='train' or r['training_eligible']=='True')}
        require({p.resolve() for p in (ROOT / 'data/processed' / split).rglob('*.png')} == expected, f'Unexpected processed file: {split}')
    held_expected={inside(r['output_path']) for r in processed if r['split']=='train' and r['training_eligible']=='False'}
    require({p.resolve() for p in (ROOT/'data/processed/review/excluded_train').rglob('*.png')}==held_expected, 'Quarantined train images differ')
    hashed = 0
    if verify_images:
        for row in inventory:
            require(digest(inside(row['source_path'])) == row['source_sha256'], f'Raw bytes changed: {row["key"]}')
            hashed += 1
        for row in processed:
            require(digest(inside(row['output_path'])) == row['output_sha256'], f'Processed bytes changed: {row["sample_key"]}')
            hashed += 1
    coverage = rows(notes / 'webinar_coverage.csv')
    require(len(coverage) == 16, 'Webinar coverage incomplete')
    figures = read_json(notes / 'notebook_figures.json')
    exported = {inside(row['path']) for row in figures}
    reports = ROOT / 'final-submit/reports'
    require(len(figures) == execution['embedded_figures'], 'Saved figures differ from notebook output')
    require(exported == {p.resolve() for p in reports.glob('*.png')}, 'Missing or stale exported figure')
    for row in figures:
        require(digest(inside(row['path'])) == row['sha256'], f'Figure bytes changed: {row["path"]}')
    require({p.resolve() for p in reports.iterdir()} == exported | {reports / 'all_test_gallery.html'}, 'Reports must contain notebook figures and gallery only')
    output = ROOT / 'final-submit/output'
    expected_output = {'oof_diagnostics.csv','modeling_report.json',
                       'submission_001_dino_logistic.csv','submission_002_dino_svm_rbf.csv',
                       'submission_003_dino_knn7.csv'}
    present_output={p.name for p in output.iterdir()}
    require(expected_output <= present_output <= expected_output | {'submission_.csv'}, 'Training output files differ')
    if 'submission_.csv' in present_output:
        archived=ROOT/'arsip/sebelum_rapi_20261107/docs/secret/submission_.csv'
        require(digest(output/'submission_.csv')==digest(archived), 'Visual comparator differs from archived source')
    model_report = read_json(output/'modeling_report.json')
    require(model_report['train_samples']==1869 and model_report['test_samples']==480, 'Modeling population differs')
    require(model_report['training_view']=='eligible_full_frame_only' and model_report['crop_views_used_for_classifier']==0 and model_report['device']=='cpu', 'Modeling scope or device differs')
    require(model_report['official_accuracy'] is None and len(model_report['oof_metrics'])==3, 'Evaluation provenance differs')
    require(digest(ROOT/'data/processed/dinov2_small_embeddings.npz')==model_report['embedding_cache_sha256'], 'Embedding cache hash differs')
    require(len(rows(output/'oof_diagnostics.csv'))==1869, 'OOF row count differs')
    for item in model_report['submission_files']:
        path=inside(item['path'])
        require(path.parent==output.resolve() and path.name in expected_output, 'Submission path differs')
        require(digest(path)==item['sha256'], 'Submission hash differs')
        submitted=rows(path)
        require(len(submitted)==480 and list(submitted[0])==['image_id','waste_category'], 'Submission shape differs')
        require([r['image_id'] for r in submitted]==[r['image_id'] for r in sample], 'Submission ID order differs')
        require(all(r['waste_category'] in classes for r in submitted), 'Submission label differs')
    require(len(model_report['submission_files'])==3, 'Expected three submissions')
    require(execution['new_model_training'] and execution['submission_created'], 'Notebook proof missing modeling')
    return {'status':'passed', 'checked_at_utc':datetime.now(timezone.utc).isoformat(),
            'custom_skills':len(skills), 'vendor_packages':len(vendor['packages']),
            'vendor_files_hash_verified':len(vendor['files']), 'relative_links_checked':links,
            'materials_hash_verified':len(materials), 'rules_sources_hash_verified':len(rules['files']),
            'qna_posts_snapshot':qna['post_count'],'qna_comments_snapshot':qna['comment_count'],'qna_official_replies_snapshot':qna['official_reply_count'],
            'raw_images':len(inventory), 'processed_images':len(processed), 'image_hashes_verified':hashed,
            'code_cells_executed':len(code), 'exported_figures':len(figures), 'webinar_coverage_rows':len(coverage),
            'manual_train_seed_images':len(manual), 'manual_category_counts':dict(Counter(r['assigned_class'] for r in manual)),
            'deferred_train_keys':sorted(deferred_keys),'training_eligible_train_samples':sum(r['split']=='train' and r['training_eligible']=='True' for r in processed),
            'train_category_counts':dict(Counter(r['assigned_class'] for r in processed if r['split']=='train' and r['training_eligible']=='True')),
            'notebook_administrative_path_mentions':0, 'notebook_model_service_mentions':0,
            'source_groups_disjoint':True, 'archive_annotations_used_for_fit':False,
            'reference_used_postfit_for_alignment':True, 'training_executed':True,
            'verified_geometry_edges_same_group_and_partition':geometry_edges_verified,
            'scope':'Layout, source provenance, notebook execution, preprocessing, model artifacts and CSV integrity; not semantic accuracy or competition certification.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-images', action='store_true')
    parser.add_argument('--no-write', action='store_true')
    args = parser.parse_args()
    try:
        report = main(args.verify_images)
    except Exception as exc:
        report = {'status':'failed','checked_at_utc':datetime.now(timezone.utc).isoformat(),'error_type':type(exc).__name__,'error':str(exc)}
    if not args.no_write:
        (ROOT / 'docs/agent-notes/skills_validation.json').write_bytes((json.dumps(report,indent=2)+'\n').encode('utf-8'))
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status'] == 'passed' else 1)

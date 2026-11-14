"""Record team review separately; show conditional cases without changing labels."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from show_remaining_train_test import read_image, ROOT, NOTES, HOME


def main():
    protected = {name: hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    frame = pd.read_csv(NOTES/'table-csv/remaining_program_archive_differences.csv', dtype={'parent_id':str}).sort_values('parent_id').reset_index(drop=True)
    frame['page'] = frame.index//20+1
    frame['team_preference'] = 'not_reviewed'
    frame['interpretation'] = 'none'
    def choose(ids, preference, interpretation='explicit_preference'):
        normalized = {f'{int(v):04d}' for v in ids}
        assert normalized <= set(frame.parent_id)
        mask = frame.parent_id.isin(normalized)
        frame.loc[mask,'team_preference'] = preference
        frame.loc[mask,'interpretation'] = interpretation
    choose([141,251,256,72,85,168,217,147,56,58,237], 'archive', 'user_declares_program_wrong_in_displayed_comparison')
    choose(frame.loc[frame.page.eq(1)&frame.team_preference.eq('not_reviewed'),'parent_id'], 'program_tentative', 'user_says_other_cases_fifty_fifty')
    choose([787,777,539,559,562,594,636,714,785,597,774], 'archive', 'interpreted_from_list_contrasted_with_remaining_program_correct')
    choose([676], 'conditional_review_or_exclusion', 'user_requests_test_coverage_check_first')
    choose(frame.loc[frame.page.eq(3)&frame.team_preference.eq('not_reviewed'),'parent_id'], 'program_tentative', 'user_says_program_seems_correct')
    choose([815,821,826,862,867,868,889,956,930,932,955,964,969,978,1018,1050], 'archive', 'interpreted_from_list_contrasted_with_remaining_program_correct')
    choose([1027], 'archive')
    choose(frame.loc[frame.page.eq(4)&frame.team_preference.eq('not_reviewed'),'parent_id'], 'program_tentative', 'user_says_program_maybe_correct')
    for page, program, conditional, ambiguous in [
        (5,[1154,1180],[],[1215]), (6,[1284,1368,1337],[],[]),
        (7,[1548,1637,1663],[1650,1662],[]), (8,[1892],[1885],[])]:
        choose(frame.loc[frame.page.eq(page),'parent_id'], 'archive', 'user_confirms_archive_for_remaining_page_members')
        choose(program,'program')
        choose(conditional,'conditional_review_or_exclusion','user_requests_test_coverage_check_first')
        choose(ambiguous,'ambiguous','user_is_uncertain_between_program_and_cardboard')
    frame['proposed_class'] = ''
    frame.loc[frame.team_preference.eq('archive'),'proposed_class'] = frame.archive_class
    frame.loc[frame.team_preference.eq('program'),'proposed_class'] = frame.assigned_class
    frame['activated_for_training'] = False
    frame['source'] = 'user_visual_review_of_displayed_train_images_20261107'
    frame.to_csv(NOTES/'table-csv/team_review_inventory.csv',index=False)
    counts = frame.loc[frame.proposed_class.ne(''),'proposed_class'].value_counts().to_dict()
    report = {'scope':'review inventory, not an approved training label manifest','manual_training_labels_activated':0,
              'review_status_counts':frame.team_preference.value_counts().to_dict(),'proposed_categories_if_all_used':counts,
              'classes_exceeding_five_if_all_used':{k:v for k,v in counts.items() if v>5},
              'page_2_not_reviewed_by_user':True,'active_data_modified':False,'protected_sha256':protected}
    (NOTES/'json/team_review_inventory.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    pairs = pd.read_csv(NOTES/'table-csv/remaining_train_test_top3.csv',dtype={'train_id':str,'test_id':str})
    query_ids = ['0147','0594','0676','1650','1662','1885']
    fig,axes = plt.subplots(6,4,figsize=(14.4,20.4),dpi=110)
    for ax in axes.flat: ax.axis('off')
    for ri,key in enumerate(query_ids):
        original = frame[frame.parent_id.eq(key)].iloc[0]
        axes[ri,0].imshow(read_image(original.source_path))
        axes[ri,0].set_title(f'Train {key}\nprogram {original.assigned_class}; arsip {original.archive_class}',fontsize=10)
        for ci,row in enumerate(pairs[pairs.train_id.eq(key)].sort_values('rank').itertuples(),1):
            axes[ri,ci].imshow(read_image(row.test_path))
            axes[ri,ci].set_title(f'Test {row.test_id}\ncosine {row.cosine:.3f}; pHash {row.phash_min_64bit}',fontsize=10)
    fig.suptitle('Kasus yang diminta tim: train dan tiga tetangga DINOv2\nTetangga terdekat belum tentu serupa; test tanpa kategori',fontsize=14)
    fig.tight_layout(rect=(0,0,1,.96));fig.savefig(HOME/'conditional_cases_test_neighbors.png',facecolor='white');plt.close(fig)
    profile = pd.read_csv(NOTES/'source_inventory.csv',dtype={'image_id':str}).set_index('key')
    candidates = ['test:254','test:364','test:440']
    fig,axes = plt.subplots(1,3,figsize=(12,4.6),dpi=130)
    for ax,key in zip(axes,candidates):
        ax.imshow(read_image(profile.at[key,'source_path']));ax.axis('off');ax.set_title(key.replace(':',' '),fontsize=12)
    fig.suptitle('Kandidat pencarian kosakata lampu dari CNN 21k\nBelum label kelas test; tidak digunakan untuk training',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.88));fig.savefig(HOME/'lighting_test_candidates.png',facecolor='white');plt.close(fig)
    assert all(hashlib.sha256((ROOT/k).read_bytes()).hexdigest()==v for k,v in protected.items())
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

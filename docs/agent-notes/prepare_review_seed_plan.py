"""Prepare a bounded human-review proposal; do not activate or fit labels."""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from show_remaining_train_test import ROOT, NOTES, HOME, OLD, read_image


def main():
    review=pd.read_csv(NOTES/'table-csv/team_review_inventory.csv',dtype={'parent_id':str}).set_index('parent_id')
    ids=['0056','0072','0085','0141','0147','0251','0256']
    proposal=review.loc[ids,['source_path','assigned_class','archive_class','team_preference','interpretation','proposed_class']].copy()
    assert proposal.team_preference.eq('archive').all()
    proposal['source']='user_review_of_displayed_train_image'
    proposal['activated_for_training']=False
    counts=proposal.proposed_class.value_counts()
    assert counts.le(5).all() and len(proposal)==7
    proposal.to_csv(NOTES/'table-csv/minimal_human_seed_proposal.csv',index_label='image_id')
    report={'status':'proposal_only_not_activated','proposed_unique_source_images':7,'proposed_class_counts':counts.to_dict(),
            'manual_training_labels_activated':0,'remaining_if_proposal_is_later_activated':33,
            'selection_reason':'explicitly reviewed priority cases; texture, flat panes, light source and packaging; no test target labels',
            'full_review_inventory_is_not_training_data':True}
    (NOTES/'json/minimal_human_seed_proposal.json').write_bytes((json.dumps(report,indent=2)+'\n').encode())
    path=ROOT/'arsip/sebelum_rapi_20261107/output/embeddings/stage11_ontology_cnn_v1/original224.npz'
    z=np.load(path,allow_pickle=False);x=z['features'].astype('float32');x/=np.maximum(np.linalg.norm(x,axis=1,keepdims=True),1e-12)
    idx=dict(zip(z['keys'],range(len(x))));test_keys=[k for k in z['keys'] if k.startswith('test:')]
    profile=pd.read_csv(NOTES/'source_inventory.csv',dtype={'image_id':str}).set_index('key')
    fig,axes=plt.subplots(2,4,figsize=(14.4,7.8),dpi=115)
    for ax in axes.flat:ax.axis('off')
    rows=[]
    for ri,key in enumerate(['1662','1885']):
        axes[ri,0].imshow(read_image(review.at[key,'source_path']))
        axes[ri,0].set_title(f'Train {key}\nobjek/isi cetakan perlu dibedakan',fontsize=10)
        sims=x[[idx[k] for k in test_keys]]@x[idx['train:'+key]]
        for ci,j in enumerate(np.argsort(-sims)[:3],1):
            test_key=test_keys[int(j)];axes[ri,ci].imshow(read_image(profile.at[test_key,'source_path']))
            axes[ri,ci].set_title(f'{test_key.replace(":"," ")}\nCNN cosine {sims[j]:.3f}',fontsize=10)
            rows.append({'train_id':key,'test_id':test_key.split(':')[1],'rank':ci,'representation':'ConvNeXt21k768','cosine':float(sims[j]),'diagnostic_only':True})
    fig.suptitle('Pandangan tambahan kasus 1662/1885 dengan representasi CNN\nTest tanpa kategori; tidak ada crop atau pemindahan label otomatis',fontsize=13)
    fig.tight_layout(rect=(0,0,1,.9));fig.savefig(HOME/'printed_case_cnn_neighbors.png',facecolor='white');plt.close(fig)
    pd.DataFrame(rows).to_csv(NOTES/'table-csv/printed_case_cnn_neighbors.csv',index=False)
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()

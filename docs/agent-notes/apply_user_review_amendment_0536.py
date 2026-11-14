"""Record the user's explicit0536audit amendment; NEVER alter model targets."""
import datetime,json
import pandas as pd
from fewshot_train_only_probe import ROOT,NOTES,sha,dump

def main():
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb','data/processed/manual_seeds.csv']}
    path=NOTES/'table-csv/team_review_revision.csv';before=sha(path);table=pd.read_csv(path,dtype={'parent_id':str}).fillna('');mask=table.parent_id.eq('0536');assert int(mask.sum())==1
    old=table.loc[mask,'proposed_class'].iloc[0];assert not table.loc[mask,'activated_for_training'].any()
    backup=NOTES/f'.runtime/team_review_before_0536_{before[:12]}.csv'
    if not backup.exists():backup.write_bytes(path.read_bytes())
    table.loc[mask,'proposed_class']='cardboard'
    table.loc[mask,'team_preference']='participant_latest_explicit_cardboard'
    table.loc[mask,'interpretation']='user_explicit_audit_amendment_not_training_target'
    table.loc[mask,'review_revision']='participant_explicit_0536_cardboard_20261108'
    table.loc[mask,'source']='user_chat_20261108_explicit_0536_cardboard'
    table.to_csv(path,index=False)
    assert all(sha(ROOT/p)==h for p,h in protected.items())
    dump(NOTES/'json/user_review_amendment_0536.json',{'image_id':'0536','old_audit_class':old,'new_audit_class':'cardboard','authority':'explicit_user_chat_20261108','scope':'audit_reference_only','manual_seed_added':False,'model_or_processed_labels_changed':False,'before_review_sha256':before,'after_review_sha256':sha(path),'backup':str(backup.relative_to(ROOT)),'protected_sha256_unchanged':protected,'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'image_id':'0536','old_audit_class':old,'new_audit_class':'cardboard','model_unchanged':True}),flush=True)

if __name__=='__main__':main()

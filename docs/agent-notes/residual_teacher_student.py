"""Train-only residual teachers propose corrected pseudo examples for grouped kNN.

All teacher graphs, pseudo targets, PCA and learner references are rebuilt on the
outer TRAIN partition. Query groups never enter any of those steps. The only
human category targets are the same34previously consumed genuine sources.
"""
import datetime,json,time
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,CLASSES,sha,dump,unit

def main():
    start=time.time();prefix='residual_teacher_student'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'fold_contained_residual_teacher_and_corrective_pseudo_kNN_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'input':'auto-onlycoldcoreplus34genuine;frozenDINOv3BaseCLSandfullsourceposteriorimageonly',
          'teacher_graph':'trainonlyk20symcos^4;clampgenuineerror;CLSalpha.7scale4smooth.5;sourcealpha.95scale4nosmooth',
          'teacher_base':'auto-onlycoldcorek7;excludeownsourcegroup;allgenuinegroupsremovedfromcoldcore',
          'pseudo_gate':'bothteacherbanksagree;minclassshare>=.7;predictionsdifferfromcoldautomaticclass;humanprotected',
          'pseudo_budget':'max20perclass,max60total,max1persourcegroup,descendingconfidence',
          'student':'twofeaturekNN7,T.07;genuineweight32,pseudoweight2,coldcoreoriginalweight;twoarmscontrol/corrective',
          'normalizer':'CLS_PCA128whiteandfullsource_center;fitouterTRAINonly',
          'selection':'best34sourcegroupheldoutmacrothenmatches;bothbanksqualifiedwithin.1macroofbest;thresholdgrid.7/.8/.9/.95',
          'guard':'global>=29/34macro>=.85andselective>=12precision>=.9;OR>=28macro>=.85andselective>=16precision1',
          'maximum_changes':20,'cap':'descendingminimumqualifiedbankshare,stablekeyorder','private_review_or_archive_read':False,'researcher_seen_prior_audits':True,
          'evaluation_limit':'34reuseddevelopment,samefoldsnotindependentAccuracy;graphnotindependentOOFvalidation',
          'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True);frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    seed=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in seed.key]);gold=np.array([CLASSES.index(c) for c in seed.assigned_class]);assert len(pos)==34 and seed.assigned_class.value_counts().le(5).all()
    geo=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];groups,folds=geo.group.to_numpy(),geo.fold.to_numpy()
    a=pd.read_csv(NOTES/'table-csv/automatic_full_seed_targets.csv').set_index('key').loc[keys];y=np.array([CLASSES.index(c) for c in a.automatic_class]);reliable=np.clip(a.reliable_vote_count.to_numpy(float)/4,.25,1.);core=a.automatic_anchor.to_numpy(bool)&~np.isin(groups,groups[pos]);human=np.zeros(n,bool);human[pos]=True
    v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);src=np.load(NOTES/'.runtime/full_source_distribution_features.npz',allow_pickle=False);assert np.array_equal(v3['keys'],keys) and np.array_equal(src['keys'],keys)
    banks={'CLS':unit(v3['features'][:,:768]),'source':unit(np.c_[unit(src['features'][:,0]),unit(src['features'][:,1])])};names=list(banks)
    arms=['control','corrective'];outputs=np.zeros((len(arms),2,n,8),np.float32);pool_history=[]

    def normalized(raw,fit,name):
        if name=='CLS':
            pca=PCA(n_components=128,svd_solver='randomized',random_state=20261108).fit(raw[fit]);return unit(pca.transform(raw)/np.maximum(np.sqrt(pca.explained_variance_),1e-6))
        return unit(raw-raw[fit].mean(0))
    def predict(x,refs,query,targets,weights):
        sim=x[query]@x[refs].T;sim[groups[query,None]==groups[refs][None,:]]=-np.inf
        neighbors=np.argsort(-sim,axis=1,kind='stable')[:,:7];v=np.take_along_axis(sim,neighbors,axis=1);assert np.isfinite(v).all()
        w=np.exp((v-v.max(1,keepdims=True))/.07)*weights[refs][neighbors]
        p=np.stack([(w*(targets[refs][neighbors]==c)).sum(1) for c in range(8)],axis=1);return p/np.maximum(p.sum(1,keepdims=True),1e-12)
    def teacher(x,train,query,hs,name):
        refs=train[core[train]];hp=pos[hs];local=dict(zip(train,range(len(train))));hlocal=np.array([local[i] for i in hp]);prior=predict(x,refs,train,y,reliable);qp=predict(x,refs,query,y,reliable)
        sim=x[train]@x[train].T;sim[groups[train,None]==groups[train][None,:]]=-np.inf;near=np.argsort(-sim,axis=1,kind='stable')[:,:20];edge=np.maximum(np.take_along_axis(sim,near,1),0)**4
        adj=sparse.csr_matrix((edge.ravel(),(np.repeat(np.arange(len(train)),20),near.ravel())),shape=(len(train),len(train)));adj=(adj+adj.T)*.5;degree=np.asarray(adj.sum(1)).ravel();t=sparse.diags(1/np.maximum(degree,1e-12))@adj
        qsim=x[query]@x[train].T;qn=np.argsort(-qsim,axis=1,kind='stable')[:,:20];qw=np.maximum(np.take_along_axis(qsim,qn,1),0)**4;qw/=np.maximum(qw.sum(1,keepdims=True),1e-12)
        error=np.zeros_like(prior);error[hlocal]=np.eye(8)[gold[hs]]-prior[hlocal];alpha=.7 if name=='CLS' else .95;state=error.copy()
        for _ in range(250):
            update=alpha*(t@state)+(1-alpha)*error;update[hlocal]=error[hlocal];delta=np.max(np.abs(update-state));state=update
            if delta<1e-6:break
        trainpred=np.maximum(prior+4*state,1e-8);trainpred/=trainpred.sum(1,keepdims=True)
        queryerror=(state[qn]*qw[:,:,None]).sum(1);qpred=np.maximum(qp+4*queryerror,1e-8);qpred/=qpred.sum(1,keepdims=True)
        if name=='CLS':
            p=trainpred.copy();p[hlocal]=np.eye(8)[gold[hs]];smooth=p.copy()
            for _ in range(150):
                update=.5*(t@smooth)+.5*p;update[hlocal]=p[hlocal];delta=np.max(np.abs(update-smooth));smooth=update
                if delta<1e-6:break
            trainpred=smooth;qpred=.5*qpred+.5*(smooth[qn]*qw[:,:,None]).sum(1)
        return trainpred,qpred

    with threadpool_limits(limits=3):
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);hs=np.flatnonzero(folds[pos]!=fold);assert not set(groups[train])&set(groups[query])
            x={name:normalized(raw,train,name) for name,raw in banks.items()};teach=[]
            for name in names:teach.append(teacher(x[name],train,query,hs,name)[0]);print('teacher fold',fold,name,flush=True)
            first,second=teach;pred=first.argmax(1);agree=pred==second.argmax(1);conf=np.minimum(first[np.arange(len(train)),pred],second[np.arange(len(train)),pred]);candidate=agree&(conf>=.7)&(pred!=y[train])&~human[train]
            order=np.argsort(-conf,kind='stable');chosen=[];counts=np.zeros(8,int);seen=set()
            for local in order:
                if not candidate[local]:continue
                i=train[local];cls=pred[local]
                if counts[cls]>=20 or groups[i] in seen:continue
                chosen.append(i);counts[cls]+=1;seen.add(groups[i])
                if len(chosen)>=60:break
            pool_history.append({'fold':fold,'candidate':int(candidate.sum()),'selected':len(chosen),'class_counts':dict(zip(CLASSES,counts.tolist()))})
            pseudo=np.array(chosen,int);targets=y.copy();targets[pseudo]=pred[[np.where(train==i)[0][0] for i in pseudo]];targets[pos[hs]]=gold[hs]
            weight=reliable.copy();weight[pseudo]=2.;weight[pos[hs]]=32.
            for arm_no,arm in enumerate(arms):
                refs=np.unique(np.r_[train[core[train]],pos[hs],pseudo if arm=='corrective' else np.empty(0,int)])
                for bank_no,name in enumerate(names):outputs[arm_no,bank_no,query]=predict(x[name],refs,query,targets,weight)
    history=[]
    for arm_no,arm in enumerate(arms):
        for bank_no,name in enumerate(names):
            pr=outputs[arm_no,bank_no];p=pr[pos].argmax(1);history.append({'arm':arm,'bank':name,'arm_no':arm_no,'bank_no':bank_no,'anchor_matches':int((p==gold).sum()),'development_balanced_recall':float(balanced_accuracy_score(gold,p))})
    chosen=max(history,key=lambda r:(r['development_balanced_recall'],r['anchor_matches'],r['arm']=='corrective'));primary=outputs[chosen['arm_no'],chosen['bank_no']].argmax(1)
    qualified=[r['bank_no'] for r in history if r['arm']==chosen['arm'] and r['development_balanced_recall']>=chosen['development_balanced_recall']-.1 and r['development_balanced_recall']>=.65]
    score=np.min([outputs[chosen['arm_no'],j][np.arange(n),primary] for j in qualified],axis=0) if qualified else np.zeros(n)
    calibration=[]
    for threshold in [.7,.8,.9,.95]:
        selected=score[pos]>=threshold;count=int(selected.sum());correct=int((selected&(primary[pos]==gold)).sum());precision=correct/count if count else 0.;calibration.append({'threshold':threshold,'covered':count,'correct':correct,'precision':precision,'eligible':count>=12 and precision>=.9})
    choices=[v for v in calibration if v['eligible']];cal=max(choices,key=lambda v:(v['covered'],v['threshold'])) if choices else None
    accepted=bool(cal is not None and ((chosen['anchor_matches']>=29 and chosen['development_balanced_recall']>=.85 and cal['covered']>=12 and cal['precision']>=.9) or (chosen['anchor_matches']>=28 and chosen['development_balanced_recall']>=.85 and cal['covered']>=16 and cal['precision']==1.)))
    current=np.array([CLASSES.index(c) for c in frame.assigned_class]);gate=(score>=cal['threshold'] if cal else np.zeros(n,bool))&~human&(primary!=current);proposals=int(gate.sum());order=np.argsort(-score,kind='stable');selected=order[gate[order]][:20];gate[:]=False;gate[selected]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'residual_teacher_student_snapshot',score[i],'weighted_neighbor_share',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(history).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(pool_history).to_csv(NOTES/f'table-csv/{prefix}_pseudo_pool.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':np.array(CLASSES)[primary],'minimum_neighbor_share':score,'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,probabilities=outputs,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_residual_student_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=history,qualified_heads=[history[chosen['arm_no']*2+j] for j in qualified],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],confidence_calibration={'selected':cal,'history':calibration,'independent_validation':False},accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=int(gate.sum()),cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'])
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report);dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'chosen':chosen,'qualified':qualified,'calibration':calibration,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'pool':pool_history,'seconds':time.time()-start},indent=2),flush=True)

if __name__=='__main__':main()

"""Human-seeded source-concept label propagation with a visual prototype check.

Object/material source class INDICES are associated automatically with the same
34human targets. No source class names, hard-coded material rules, private review,
annotation archive, or test input. CV is development, not independent Accuracy.
"""
import datetime
import json
import time
import numpy as np
import pandas as pd
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score
from safetensors import safe_open
from huggingface_hub import hf_hub_download
from threadpoolctl import threadpool_limits
from fewshot_train_only_probe import ROOT,NOTES,OLD,CLASSES,sha,dump,unit
from augmented_source_concepts import load_augmented_source_concepts


def source_topk(keys):
    cache=NOTES/'.runtime/source_object_top10.npz'
    if cache.exists():
        z=np.load(cache,allow_pickle=False);assert np.array_equal(z['keys'],keys);return z['indices'],z['probabilities'],sha(cache)
    z=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);assert np.array_equal(z['keys'],keys)
    source=json.loads((OLD/'reports/errors/stage11_ontology_cnn_20261106/checkpoint_and_ontology.json').read_text())
    weight=hf_hub_download(source['repository'],'model.safetensors',revision=source['revision'],local_files_only=True);assert sha(weight)==source['checkpoint_sha256']
    with safe_open(weight,framework='np') as state:
        gamma=state.get_tensor('head.norm.weight').astype(np.float64);beta=state.get_tensor('head.norm.bias').astype(np.float64);fc=state.get_tensor('head.fc.weight');bias=state.get_tensor('head.fc.bias')
    indices=[];probabilities=[]
    with threadpool_limits(limits=2):
        for first in range(0,len(keys),32):
            u=z['features'][first:first+32].astype(np.float64);aa=((u/gamma)**2).sum(1);bb=-2*(u*beta/gamma**2).sum(1);cc=((beta/gamma)**2).sum()-768
            scale=(-bb+np.sqrt(bb*bb-4*aa*cc))/(2*aa);p=softmax((u*scale[:,None]).astype(np.float32)@fc.T+bias,axis=1)
            idx=np.argpartition(-p,9,axis=1)[:,:10];order=np.argsort(-np.take_along_axis(p,idx,1),axis=1);idx=np.take_along_axis(idx,order,1)
            indices.append(idx);probabilities.append(np.take_along_axis(p,idx,1))
    indices=np.concatenate(indices);probabilities=np.concatenate(probabilities)
    np.savez_compressed(cache,keys=keys,indices=indices,probabilities=probabilities,source_sha256=z['source_sha256'])
    return indices,probabilities,sha(cache)


def main():
    start=time.time();prefix='augmented_concept_seed'
    protected={p:sha(ROOT/p) for p in ['data/processed/manifest.csv','data/preprocessing_inputs.npz','final-submit/DSC2026_Notebook.ipynb']}
    plan={'method':'genuine_source_multiview_concept_association_v1','human_sources':34,'new_manual_labels':0,'test_rows_in_fit':0,
          'targets':'same34humanONLY;noautomaticcategorytargets;sourceconceptnamesNEVERmanuallymapped',
          'source_concepts':'objectCNN21841andMINC23numericindices;associateconceptwithlabelsobservedin34sources',
          'configs':['object','material','object_then_material'],'object_topk':10,'object_reference_views':8,'object_ref_min_probability':.1,'object_query_min_probability':.03,'object_min_mapped_mass':.1,
          'material_topk':3,'material_ref_min_probability':.2,'material_query_min_probability':.2,'material_min_mapped_mass':.4,
          'class_fraction_min':.9,'pure_visual_similarity_min':{'object':.45,'material':.65},
          'conflicted_concept':'per-sourcemaxviewcos+.1logmaximumreferenceconceptmass;classmargin.03;querycos.55;sourcevotesnotviewcounts',
          'visual_check':'queryCNNunitcosagainstmaximumover8viewsofmatchedgenuinesource;purefloor.45;conflictclassmargin.03andcosfloor.55',
          'normalizer':'PCA128whiteningfitouterTRAINonly;allquerygroupsheldoutfromreferencesandstatistics',
          'guard':'whole34fallbackprototype-plus-residual28nondecreasingmatchesandmacro.85;partialcovered>=5andprecision>=.9',
          'selection':'maxwhole34matches,thenmacro,thenpartialprecision,thencoverage;listedconfigorderbreaksties',
          'purpose':'known-conceptfew-shotannotation,notguaranteedrecognitionofunseenmaterialwithinsameconcept',
          'maximum_changes':20,'cap':'descendingwinningmappedsourceprobabilitymass,stablekeyorder',
          'score':'winningmappedsourceprobabilityMASS,notconditionalshareandnotmaterialcorrectnessprobability',
          'evaluation_limit':'34reuseddevelopment,smallpartialcoverage;sourceconceptcancontainmixedmaterials;allnewpseudolabelsunverified',
          'private_review_or_archive_read':False,'researcher_seen_prior_audits':True,'protected_sha256':protected,'source_code_sha256':sha(__file__)}
    dump(NOTES/f'json/{prefix}_plan.json',plan)
    m=pd.read_csv(ROOT/'data/processed/manifest.csv',dtype={'parent_id':str});complete=m[m.split.eq('train')&~m.is_crop].sort_values('parent_id').reset_index(drop=True)
    frame=complete[complete.training_eligible].reset_index(drop=True);keys=frame.parent_key.to_numpy(str);n=len(keys);assert n==1904
    s=pd.read_csv(ROOT/'data/processed/manual_seeds.csv');assert len(s)==34 and s.assigned_class.value_counts().le(5).all();lookup=dict(zip(keys,range(n)));pos=np.array([lookup[k] for k in s.key]);gold=np.array([CLASSES.index(c) for c in s.assigned_class])
    g=pd.read_csv(NOTES/'table-csv/geometry_train_groups.csv').set_index('key').loc[keys];folds,groups=g.fold.to_numpy(),g.group.to_numpy()
    v3=np.load(NOTES/'.runtime/dinov3_base_train_features.npz',allow_pickle=False);material=np.load(NOTES/'.runtime/minc_train_features.npz',allow_pickle=False);assert np.array_equal(v3['keys'],keys) and np.array_equal(material['keys'],keys)
    visual=unit(v3['features'][:,:768]);material_visual=unit(material['features'][:,1:].mean(1));object_idx,object_prob,source_sha=source_topk(keys)
    material_prob=material['probabilities'][:,1:].mean(1);material_idx=np.argsort(-material_prob,axis=1)[:,:3];material_top=np.take_along_axis(material_prob,material_idx,1)
    banks={'object':(object_idx[:,:10],object_prob[:,:10],.03,.1,visual,.45),'material':(material_idx,material_top,.2,.4,material_visual,.65)}
    old=np.load(NOTES/'.runtime/dinov3_base_neighbors_outputs.npz',allow_pickle=False);assert np.array_equal(old['keys'],keys)
    info=json.loads((NOTES/'json/dinov3_base_neighbors_frozen.json').read_text());wi=max(range(len(info['selected_heads'])),key=lambda i:info['selected_heads'][i]['development_balanced_recall']);baseline=old['probabilities'][wi]
    baseclass=baseline.argmax(1)
    augmodel=np.load(NOTES/'.runtime/augmented_prototype_consensus_outputs.npz',allow_pickle=False);res=np.load(NOTES/'.runtime/residual_graph_correction_outputs.npz',allow_pickle=False)
    for bank in [augmodel,res]:assert np.array_equal(bank['keys'],keys)
    ap,am,ar=augmodel['oof_predictions'],augmodel['oof_margins'],augmodel['oof_raw_similarity'];votes=np.stack([(ap==c).sum(0) for c in range(8)],1);ac=votes.argmax(1);agree=ap==ac
    acov=(votes.max(1)>=2)&(np.min(np.where(agree,am,np.inf),0)>=.1)&(np.min(np.where(agree,ar,np.inf),0)>=.45)
    baseclass=np.where(acov,ac,baseclass);ri=json.loads((NOTES/'json/residual_graph_correction_frozen.json').read_text());rwi=max(range(len(ri['selected_heads'])),key=lambda i:(ri['selected_heads'][i]['development_balanced_recall'],ri['selected_heads'][i]['anchor_matches']));rp=res['probabilities'];rc=rp[rwi].argmax(1);rs=np.min([p[np.arange(n),rc] for p in rp],0);route=(rs>=.6)&~(acov&(ac!=rc));baseclass=np.where(route,rc,baseclass);baseline=np.eye(8)[baseclass]
    baseline_matches=int((baseclass[pos]==gold).sum());baseline_macro=float(balanced_accuracy_score(gold,baseclass[pos]));assert baseline_matches==28
    augidx,augprob,augcnn,augsha=load_augmented_source_concepts(s.key.to_numpy(str));cnn=np.load(NOTES/'.runtime/convnext_train_features.npz',allow_pickle=False);assert np.array_equal(cnn['keys'],keys);cnnquery=unit(cnn['features'])
    oof={name:{'class':np.full(n,-1,int),'mass':np.zeros(n),'share':np.zeros(n),'source_concepts':np.zeros(n,int)} for name in banks};mapping_records=[]

    def run_bank(name,train,query,hs,whitened):
        idx,prob,minprob,minmass,pure_visual,floor=banks[name]
        refs={}
        concept_strength={}
        for h in hs:
            indices=augidx[h].ravel() if name=='object' else idx[pos[h]]
            masses=augprob[h].ravel() if name=='object' else prob[pos[h]]
            for j,p in zip(indices,masses):
                threshold=.1 if name=='object' else minprob
                if p>=threshold:
                    refs.setdefault(int(j),set()).add(int(h));concept_strength[(int(j),int(h))]=max(float(p),concept_strength.get((int(j),int(h)),0.))
        refs={j:sorted(hs) for j,hs in refs.items()}
        raw_similarity=visual[query]@visual[pos[hs]].T;white_similarity=whitened[query]@whitened[pos[hs]].T
        nearest=hs[white_similarity.argmax(1)];nearest_class=gold[nearest]
        scores=np.zeros((len(query),8));used=np.zeros(len(query),int)
        for local,q in enumerate(query):
            for j,mass in zip(idx[q],prob[q]):
                if mass<minprob or int(j) not in refs:continue
                sources=np.array(refs[int(j)],int);labels=np.unique(gold[sources]);chosen=None
                if name=='object':
                    local_similarity=(augcnn[sources]@cnnquery[q]).max(1)
                    if len(labels)==1:
                        if local_similarity.max()>=.45:chosen=int(labels[0])
                    else:
                        strengths=np.array([concept_strength[(int(j),int(h))] for h in sources]);scores_by_source=local_similarity+.1*np.log(np.maximum(strengths,1e-12))
                        class_score=np.array([scores_by_source[gold[sources]==c].max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])
                        if class_score[order[0]]-class_score[order[1]]>=.03 and local_similarity[gold[sources]==candidate].max()>=.55:chosen=candidate
                elif len(labels)==1:
                    if float((pure_visual[q]@pure_visual[pos[sources]].T).max())>=floor:chosen=int(labels[0])
                else:
                    class_score=np.array([(whitened[q]@whitened[pos[sources[gold[sources]==c]]].T).max() for c in labels]);order=np.argsort(-class_score,kind='stable');candidate=int(labels[order[0]])
                    refs_class=sources[gold[sources]==candidate]
                    if class_score[order[0]]-class_score[order[1]]>=.05 and float((visual[q]@visual[pos[refs_class]].T).max())>=.65:chosen=candidate
                if chosen is not None:scores[local,chosen]+=float(mass);used[local]+=1
        total=scores.sum(1);candidate=scores.argmax(1);winning=scores[np.arange(len(query)),candidate];fraction=winning/np.maximum(total,1e-12)
        covered=(total>=minmass)&(fraction>=.9)
        candidate[~covered]=-1
        return {'class':candidate,'mass':winning,'share':fraction,'source_concepts':used},refs

    def choose(name,parts):
        if name!='object_then_material':return parts[name]
        mask=parts['object']['class']>=0
        return {k:np.where(mask,parts['object'][k],parts['material'][k]) for k in parts['object']}

    with threadpool_limits(limits=2):
        for fold in range(5):
            train=np.flatnonzero(folds!=fold);query=np.flatnonzero(folds==fold);hs=np.flatnonzero(folds[pos]!=fold);assert not set(groups[train])&set(groups[query])
            pca=PCA(n_components=128,svd_solver='randomized',random_state=20261108+fold).fit(visual[train]);white=unit(pca.transform(visual)/np.maximum(np.sqrt(pca.explained_variance_),1e-6))
            for name in banks:
                result,_=run_bank(name,train,query,hs,white)
                for k in oof[name]:oof[name][k][query]=result[k]
        development=[]
        for config in plan['configs']:
            result=choose(config,oof);covered=result['class'][pos]>=0;count=int(covered.sum());correct=int((covered&(result['class'][pos]==gold)).sum());precision=correct/count if count else 0.
            pred=baseline.argmax(1).copy();mask=result['class']>=0;pred[mask]=result['class'][mask];matches=int((pred[pos]==gold).sum());macro=float(balanced_accuracy_score(gold,pred[pos]))
            ok=count>=5 and precision>=.9 and matches>=baseline_matches and macro>=baseline_macro-1e-12
            development.append({'config':config,'covered':count,'correct':correct,'precision':precision,'anchor_matches':matches,'development_balanced_recall':macro,'eligible':bool(ok)})
        feasible=[r for r in development if r['eligible']];chosen=max(feasible,key=lambda r:(r['anchor_matches'],r['development_balanced_recall'],r['precision'],r['covered'])) if feasible else development[0];accepted=bool(feasible)
        pca=PCA(n_components=128,svd_solver='randomized',random_state=20261108).fit(visual);white=unit(pca.transform(visual)/np.maximum(np.sqrt(pca.explained_variance_),1e-6));parts={}
        for name in banks:
            parts[name],refs=run_bank(name,np.arange(n),np.arange(n),np.arange(34),white)
            for j,hs in refs.items():mapping_records.append({'bank':name,'source_concept_index':j,'human_reference_keys':'|'.join(keys[pos[hs]]),'human_classes':'|'.join(sorted(set(np.array(CLASSES)[gold[hs]]))),'human_references':len(hs)})
        final=choose(chosen['config'],parts)
    human=frame.human_reviewed.to_numpy(bool);current=np.array([CLASSES.index(c) for c in frame.assigned_class]);primary=final['class'];score=final['mass'];gate=(primary>=0)&~human&(primary!=current)
    proposals=int(gate.sum());rank=np.argsort(-score,kind='stable');ids=rank[gate[rank]][:20];gate[:]=False;gate[ids]=True
    if not accepted:gate[:]=False
    result=complete[['parent_key','parent_id','source_path','source_sha256']].rename(columns={'parent_key':'key','parent_id':'image_id'}).copy()
    for dest,src in [('baseline_class','assigned_class'),('predicted_class','assigned_class'),('category_source','label_source'),('category_score','category_score'),('category_score_kind','category_score_kind'),('human_reviewed','human_reviewed'),('needs_review','needs_review')]:result[dest]=complete[src]
    result=result.set_index('key')
    for i in np.flatnonzero(gate):result.loc[keys[i],['predicted_class','category_source','category_score','category_score_kind','needs_review']]=[CLASSES[primary[i]],'human_seed_source_concept_snapshot',score[i],'source_concept_probability_mass',True]
    dest=NOTES/f'table-csv/{prefix}_selected.csv';result.reset_index().to_csv(dest,index=False);pd.DataFrame(development).to_csv(NOTES/f'table-csv/{prefix}_development.csv',index=False);pd.DataFrame(mapping_records).to_csv(NOTES/f'table-csv/{prefix}_learned_mapping.csv',index=False)
    pd.DataFrame({'key':keys,'candidate_class':[CLASSES[c] if c>=0 else '' for c in primary],'winning_mapped_source_mass':score,'conditional_class_fraction':final['share'],'source_concepts':final['source_concepts'],'accepted':gate}).to_csv(NOTES/f'table-csv/{prefix}_evidence.csv',index=False)
    np.savez_compressed(NOTES/f'.runtime/{prefix}_outputs.npz',keys=keys,predictions=primary,source_mass=score,accepted=gate)
    active=json.loads((NOTES/'json/bounded_anchor_publication.json').read_text())['source_frozen_prefix'];report=json.loads((NOTES/f'json/{active}_frozen.json').read_text());initial=pd.read_csv(NOTES/'table-csv/bounded_consensus_selected.csv').set_index('key')
    report.update(status='frozen_before_private_source_concept_seed_audit',method=plan['method'],prediction_sha256=sha(dest),protected_sha256=protected,primary_head=chosen,primary_plan_sha256=sha(NOTES/f'json/{prefix}_plan.json'),selected_heads=[{'source':chosen['config']}],qualified_heads=[{'source':chosen['config']}],selected_knn_heads=[],human_leave_source_group_out_matches=chosen['anchor_matches'],human_holdout_rows=34,development_balanced_recall=chosen['development_balanced_recall'],
        confidence_calibration={'selected':chosen,'independent_validation':False,'objective':'knownconceptprototypepropagation'},human_diagnostic_protocol=plan['evaluation_limit'],accepted_by_development_guard=accepted,proposed_new_labels=proposals,changed_rows_vs_baseline=int(gate.sum()),new_neural_changes=0,new_vit_propagation_changes=0,
        cumulative_changes_vs_initial_program=int(result.predicted_class.ne(initial.loc[result.index,'baseline_class']).sum()),source_counts=result.category_source.value_counts().to_dict(),source_code_sha256=sha(__file__),seconds=time.time()-start,validation_limit=plan['evaluation_limit'],source_concept_cache_sha256=source_sha,augmented_concept_cache_sha256=augsha)
    assert all(sha(ROOT/p)==h for p,h in protected.items());dump(NOTES/f'json/{prefix}_frozen.json',report)
    dump(NOTES/f'json/{prefix}_publication_decision.json',{'decision':'publish' if accepted else 'reject','prediction_sha256':sha(dest),'accepted_changes':int(gate.sum()),'guard':plan['guard'],'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()})
    print(json.dumps({'development':development,'chosen':chosen,'accepted':accepted,'proposals':proposals,'changes':int(gate.sum()),'seconds':time.time()-start},indent=2),flush=True)


if __name__=='__main__':main()

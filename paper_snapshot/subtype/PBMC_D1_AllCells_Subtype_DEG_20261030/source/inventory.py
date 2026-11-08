"""Repeat frozen Task99 K34 screen on all cells; inventory PCA and B2 symmetrically."""
import importlib.util,itertools
from collections import Counter
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score,silhouette_score
from sklearn.cluster import KMeans
from scipy.optimize import linear_sum_assignment
from analysis_common import *

def main():
    cells,labels=context();k=frozen.k_extra(len(set(labels)));assert k==34
    (RUN/'tables').mkdir(exist_ok=True);(RUN/'partitions').mkdir(exist_ok=True)
    candidates=[];members=[];all_compositions=[];representations={};partitions={}
    pd.DataFrame({'barcode':cells,'reference_label':labels}).to_csv(RUN/'input/cell_order_reference.csv',index=False)
    for method in METHODS:
        methodrows=[];seed_parts={}
        for seed in SEEDS:
            x=np.load(BASE/method/f'seed_{seed}/target_embedding_standardized.npy').astype(np.float64)
            assert x.shape[0]==9347
            representations[method,seed]=x
            print('CLUSTER',method,seed,flush=True)
            cl=frozen.kmeans(x,k,seed,n_init=8)
            np.save(RUN/'partitions'/f'{method}_{seed}_K34.npy',cl)
            splits=frozen.find_splits(cl,labels,'D1',seed,k)
            comp=pd.crosstab(pd.Series(cl,name='cluster'),pd.Series(labels,name='reference_label')).stack().reset_index(name='cells')
            comp['method']=method;comp['seed']=seed;all_compositions.append(comp)
            rows=[]
            for r in splits:
                r=dict(r,method=method,candidate_id=f'D1_ALL_{method}_{slug(r["reference_cell_type"])}_C1')
                part=partition(r,cl,labels,cells)
                r.update(n_A_reference=int((part=='A').sum()),n_B_reference=int((part=='B').sum()),
                         subclass_total=int((labels==r['reference_cell_type']).sum()))
                r['selected_fraction']=(r['n_A_reference']+r['n_B_reference'])/r['subclass_total']
                rows.append(r);seed_parts[r['candidate_id'],seed]=part
            methodrows.extend(rows)
            print('SPLITS',method,seed,[(r['reference_cell_type'],r['n_A_reference'],r['n_B_reference']) for r in rows],flush=True)
        # The first fixed seed where a candidate exists is its anchor. No marker/P-value selection.
        for cid in sorted(set(r['candidate_id'] for r in methodrows)):
            rr=[r for r in methodrows if r['candidate_id']==cid]
            anchor=min(int(r['seed']) for r in rr);a=seed_parts[cid,anchor]
            for r in rr:
                seed=int(r['seed']);b=seed_parts[cid,seed]
                aligned,j=align(a,b)
                if not b.equals(aligned):
                    r['cluster_a'],r['cluster_b']=r['cluster_b'],r['cluster_a']
                    r['n_A_reference'],r['n_B_reference']=r['n_B_reference'],r['n_A_reference']
                    r['purity_a'],r['purity_b']=r['purity_b'],r['purity_a']
                r.update(anchor_seed=anchor,orientation_rule='max barcode group Jaccard to lowest available fixed seed',anchor_group_Jaccard=j)
                partitions[cid,seed]=aligned
                for barcode,group in aligned.items():members.append(dict(candidate_id=cid,method=method,seed=seed,barcode=barcode,group=group,reference_subclass=r['reference_cell_type']))
            candidates.extend(rr)
    screen=pd.DataFrame(candidates);screen.to_csv(RUN/'tables/candidate_screen_by_seed.csv',index=False)
    pd.DataFrame(members).to_csv(RUN/'tables/candidate_membership.csv.gz',index=False,compression='gzip')
    pd.concat(all_compositions).to_csv(RUN/'tables/cluster_composition.csv',index=False)
    stability=[];inventory=[]
    for cid,rr in screen.groupby('candidate_id',sort=False):
        pair_rows=[]
        for a,b in itertools.combinations(sorted(rr.seed.unique()),2):
            aa=partitions[cid,int(a)];bb=partitions[cid,int(b)]
            common=sorted(set(aa.index)&set(bb.index));union=set(aa.index)|set(bb.index)
            two=len(common)>1 and aa.loc[common].nunique()==bb.loc[common].nunique()==2
            row=dict(candidate_id=cid,method=rr.method.iloc[0],seed_a=a,seed_b=b,
               member_Jaccard=len(common)/len(union),group_Jaccard=align(aa,bb)[1],shared_cells=len(common),
               partition_ARI=adjusted_rand_score(aa.loc[common],bb.loc[common]) if two else np.nan,
               partition_NMI=normalized_mutual_info_score(aa.loc[common],bb.loc[common]) if two else np.nan,
               valid_two_group_overlap=two)
            pair_rows.append(row);stability.append(row)
        ps=pd.DataFrame(pair_rows)
        rec=dict(candidate_id=cid,method=rr.method.iloc[0],reference_subclass=rr.reference_cell_type.iloc[0],
             seeds=';'.join(map(str,sorted(rr.seed.unique()))),n_seeds=rr.seed.nunique(),anchor_seed=int(rr.anchor_seed.iloc[0]),
             n_A_by_seed=';'.join(f'{int(r.seed)}:{int(r.n_A_reference)}' for r in rr.itertuples()),
             n_B_by_seed=';'.join(f'{int(r.seed)}:{int(r.n_B_reference)}' for r in rr.itertuples()),
             mean_selected_fraction=float(rr.selected_fraction.mean()),
             mean_member_Jaccard=float(ps.member_Jaccard.mean()) if len(ps) else np.nan,
             mean_group_Jaccard=float(ps.group_Jaccard.mean()) if len(ps) else np.nan,
             mean_partition_ARI=float(ps.partition_ARI.mean()) if len(ps) else np.nan,
             mean_partition_NMI=float(ps.partition_NMI.mean()) if len(ps) else np.nan,
             min_group_cells=int(min(rr.n_A_reference.min(),rr.n_B_reference.min())),
             paired_ATAC=True,RNA_counts='full 36601 genes',DEG_status='PENDING',GO_Reactome_status='PENDING',
             risk='posthoc label-assisted screen; one target cohort; seeds are not biological replicates')
        inventory.append(rec)
    inv=pd.DataFrame(inventory);inv.to_csv(RUN/'candidate_inventory.csv',index=False)
    pd.DataFrame(stability).to_csv(RUN/'tables/cross_seed_candidate_stability.csv',index=False)
    # Record every subclass, including failures and historical D1 candidates, for each method and seed.
    complete=[]
    for method in METHODS:
        for seed in SEEDS:
            for label in sorted(set(labels)):
                hit=screen[(screen.method==method)&(screen.seed==seed)&(screen.reference_cell_type==label)]
                complete.append(dict(method=method,seed=seed,reference_subclass=label,cells=int((labels==label).sum()),
                                     status='CANDIDATE' if len(hit) else 'NO_SPLIT_PASSED_FIXED_SCREEN'))
    pd.DataFrame(complete).to_csv(RUN/'tables/all_subclass_screen_status.csv',index=False)
    # Blind Leiden at the already-fixed resolution is a sensitivity check, never used to pick K or replace failed K34 splits.
    sensitivity=[]
    for method in METHODS:
        for seed in SEEDS:
            p=pd.read_csv(BASE/'figures'/method/f'seed_{seed}/plot_data.csv').set_index('barcode').loc[cells]
            splits=frozen.find_splits(p.blind_leiden.to_numpy(),labels,'D1',seed,p.blind_leiden.nunique())
            sensitivity.extend([dict(method=method,algorithm='saved blind Leiden resolution 1.0',**r) for r in splits])
    pd.DataFrame(sensitivity).to_csv(RUN/'tables/leiden_sensitivity_candidates.csv',index=False)
    # Each fixed candidate is tested in every representation on exactly its recorded A/B cells.
    recover=[];agreement=[]
    for (cid,seed),part in partitions.items():
        origin=screen[(screen.candidate_id==cid)&(screen.seed==seed)].iloc[0]
        idx=pd.Index(cells).get_indexer(part.index);assert np.all(idx>=0)
        truth=(part=='B').astype(int).to_numpy()
        for method in METHODS:
            x=representations[method,seed][idx]
            cl=KMeans(n_clusters=2,n_init=100,random_state=seed,max_iter=1000).fit_predict(x)
            tab=np.zeros((2,2),int);np.add.at(tab,(truth,cl),1)
            a,b=linear_sum_assignment(-tab);mp=dict(zip(b,a));pred=np.asarray([mp[v] for v in cl])
            recover.append(dict(candidate_id=cid,origin_method=origin.method,evaluated_method=method,seed=seed,
                reference_subclass=origin.reference_cell_type,cells=len(idx),n_A=int((truth==0).sum()),n_B=int((truth==1).sum()),
                ARI=adjusted_rand_score(truth,cl),NMI=normalized_mutual_info_score(truth,cl),
                matched_accuracy=float(np.mean(pred==truth)),
                candidate_label_silhouette=float(silhouette_score(x,truth)),
                scope='posthoc binary KMeans within same fixed candidate members; no cell reselection; not independent discovery'))
        for method in METHODS:
            if method==origin.method:continue
            other=screen[(screen.method==method)&(screen.seed==seed)&(screen.reference_cell_type==origin.reference_cell_type)]
            if not len(other):
                agreement.append(dict(candidate_id=cid,seed=seed,compared_method=method,status='NO_INDEPENDENT_SPLIT'));continue
            othercid=other.candidate_id.iloc[0];bb=partitions[othercid,seed]
            common=sorted(set(part.index)&set(bb.index));union=set(part.index)|set(bb.index)
            valid=len(common)>1 and part.loc[common].nunique()==bb.loc[common].nunique()==2
            agreement.append(dict(candidate_id=cid,seed=seed,compared_method=method,status='MATCHED_BY_BARCODES',
                member_Jaccard=len(common)/len(union),shared_cells=len(common),
                partition_ARI=adjusted_rand_score(part.loc[common],bb.loc[common]) if valid else np.nan,
                partition_NMI=normalized_mutual_info_score(part.loc[common],bb.loc[common]) if valid else np.nan))
    pd.DataFrame(recover).to_csv(RUN/'tables/fixed_candidate_recovery_all_methods.csv',index=False)
    pd.DataFrame(agreement).to_csv(RUN/'tables/independent_split_cross_method_agreement.csv',index=False)
    write(RUN/'audit/INVENTORY_COMPLETE.json',dict(status='COMPLETE',candidates=len(inv),seed_candidates=len(screen),
        fixed_k=34,n_init=8,min_cluster_size=30,min_majority_purity=.6,top_two_by_cluster_size=True,
        model_training=False,cell_subsampling=False,reference_labels_used_only_posthoc=True,
        anchoring='lowest available fixed seed; alignment by barcode overlap only',
        hashes={str(p.relative_to(RUN)):sha(p) for p in [RUN/'source/frozen_task99_analysis.py',RUN/'candidate_inventory.csv',RUN/'tables/candidate_membership.csv.gz']}))
    print('INVENTORY_COMPLETE',len(inv),'candidates',len(screen),'seed candidates',flush=True)

if __name__=='__main__':main()

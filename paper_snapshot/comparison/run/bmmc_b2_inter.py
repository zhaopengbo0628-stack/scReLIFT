"""BMMC Site inter: source Full B2 checkpoint -> target RNA-only evaluation."""
import json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, h5py, anndata as ad, torch
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
    adjusted_mutual_info_score, accuracy_score, f1_score, balanced_accuracy_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from scipy import sparse
from scipy.optimize import linear_sum_assignment

ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT),str(ROOT/'main/run/s_mode'),str(ROOT/'main/run/d_mode'),str(ROOT/'main/core/regulatory/common/legacy_support')]
from main.core.regulatory.s_mode.single_source import normalize_counts
import importlib.util
spec=importlib.util.spec_from_file_location('b2',ROOT/'main/core/run_b2_strict_lodo.py')
b=importlib.util.module_from_spec(spec); spec.loader.exec_module(b)
CACHE=ROOT/'data/cache/s_mode/BMMC/bmmc_site_intra_v1_20261024'
H5=ROOT/'data/raw/BMMC/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'
OUT=ROOT/'comparison/result/BMMC_inter/b2full'
SEEDS=[20020130,20020131,20020132]
SITES=['BMMC_Site1','BMMC_Site2','BMMC_Site3','BMMC_Site4']
PAIRS=[('BMMC_Site1','BMMC_Site2'),('BMMC_Site2','BMMC_Site3'),('BMMC_Site3','BMMC_Site4'),('BMMC_Site4','BMMC_Site1')]


def score(z,labels):
    names,y=np.unique(labels,return_inverse=True); scaled=StandardScaler().fit_transform(z)
    pred=KMeans(n_clusters=len(names),init='k-means++',n_init=100,random_state=20020130,max_iter=1000,tol=1e-4,algorithm='lloyd').fit_predict(scaled)
    table=np.zeros((len(names),len(names)),np.int64); np.add.at(table,(y,pred),1)
    a,c=linear_sum_assignment(-table); mapping=dict(zip(c,a)); matched=np.array([mapping[int(k)] for k in pred])
    nn=NearestNeighbors(n_neighbors=min(16,len(y))).fit(scaled).kneighbors(scaled,return_distance=False)
    purity=np.mean([np.mean(y[row[row!=i][:15]]==y[i]) for i,row in enumerate(nn)])
    return dict(ARI=float(adjusted_rand_score(y,pred)),NMI=float(normalized_mutual_info_score(y,pred)),
        AMI=float(adjusted_mutual_info_score(y,pred)),ACC=float(accuracy_score(y,matched)),
        Macro_F1=float(f1_score(y,matched,average='macro',zero_division=0)),
        Balanced_ACC=float(balanced_accuracy_score(y,matched)),Weighted_F1=float(f1_score(y,matched,average='weighted',zero_division=0)),
        Purity_at_15=float(purity),Silhouette=float(silhouette_score(scaled,y)),cells=len(y),classes=len(names))

a=ad.read_h5ad(H5,backed='r')
obs_names=a.obs_names.astype(str).to_numpy(); var=a.var.copy()
all_genes=var['gene_id'].astype(str).to_numpy()
del a
with h5py.File(H5,'r') as f:
    sh=tuple(int(x) for x in f['layers/counts'].attrs['shape'])
    M=sparse.csr_matrix((f['layers/counts/data'][:],f['layers/counts/indices'][:],f['layers/counts/indptr'][:]),shape=sh)
row_lookup={c:i for i,c in enumerate(obs_names)}; gene_pos={g:i for i,g in enumerate(all_genes)}
for source,target in PAIRS:
    sc=CACHE/source; tc=CACHE/target
    src_genes=np.load(sc/'gene_ids.npy',allow_pickle=True).astype(str)
    state=np.load(sc/'rna_state.npz'); mean,std=state['mean'],state['std']
    tgt_cells=np.load(tc/'barcodes.npy',allow_pickle=True).astype(str); rows=np.array([row_lookup[c] for c in tgt_cells])
    test=np.load(tc/'test_index.npy'); labels=np.load(tc/'labels.npy',allow_pickle=True).astype(str)[test]
    raw=M[rows][:, :].tocsr(); log=normalize_counts(raw[test])
    cols=np.array([gene_pos[g] for g in src_genes if g in gene_pos],int); hit=np.array([g in gene_pos for g in src_genes])
    values=np.zeros((len(test),len(src_genes)),np.float32); values[:,hit]=log[:,cols].toarray()
    x=((values-mean)/(std+1e-6)).astype(np.float32)
    for seed in SEEDS:
        unit=OUT/f'D{SITES.index(source)+1}_to_D{SITES.index(target)+1}'/f'seed_{seed}'
        ck=ROOT/'result_contain/97_bmmc_b2full_rnaonly_3seed_intra_20261024/models'/source/f'seed_{seed}'/'checkpoints/student/best.pt'
        model=b.Encoder(2000); model.load_state_dict(torch.load(ck,map_location='cpu',weights_only=False)['model']); model.eval().requires_grad_(False)
        with torch.inference_mode():
            z=np.concatenate([model(torch.from_numpy(x[i:i+512])).numpy() for i in range(0,len(x),512)])
        assert z.shape==(len(x),64) and np.isfinite(z).all()
        unit.mkdir(parents=True,exist_ok=True); np.save(unit/'embedding.npy',z)
        m=score(z,labels); (unit/'metrics.json').write_text(json.dumps(m,indent=2)+'\n')
        pd.DataFrame(dict(barcode=tgt_cells[test],reference_label=labels)).to_csv(unit/'cell_order.csv',index=False)
        (unit/'manifest.json').write_text(json.dumps(dict(status='COMPLETE',source=source,target=target,seed=seed,metrics=m,
            checkpoint_sha256=__import__('hashlib').sha256(ck.read_bytes()).hexdigest(),cells=len(x),classes=int(len(set(labels))),
            target_atac_used=False,input='target RNA mapped to source 2000 genes; missing target genes zero; source train mean/std'),indent=2)+'\n')
        print('INTER',source,target,seed,m['ARI'],m['NMI'],flush=True)

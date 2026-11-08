import importlib.util
from pathlib import Path
import numpy as np, pandas as pd
S=Path('/home/u202522802045/share/zpb/result_contain/AD_LODO_BIOLOGY_VALIDATION_20261029')
spec=importlib.util.spec_from_file_location('bv',S/'biology_validation.py'); bv=importlib.util.module_from_spec(spec); spec.loader.exec_module(bv)
meta=pd.read_csv(bv.CELLS,low_memory=False); f=bv.load_gene_features()
deg=pd.read_csv(S/'03_pseudobulk_deg/CELLTYPE_DEG_ALL.csv'); ora=pd.read_csv(S/'04_go_reactome/GO_BP_REACTOME_ALL.csv')
edges=pd.read_csv(S/'05_regulatory_evidence/STABLE_REGULATORY_EDGES_WITH_SUPPORT.csv'); ov=pd.read_csv(S/'05_regulatory_evidence/DEG_STABLE_EDGE_OVERLAP.csv')
zf=np.load(S/'02_umap_clustering/Full_B2_cell_embeddings.npy',mmap_mode='r'); zr=np.load(S/'02_umap_clustering/RNA_only_cell_embeddings.npy',mmap_mode='r')
u1=pd.read_csv(S/'02_umap_clustering/Full_B2_CELL_UMAP.csv');u2=pd.read_csv(S/'02_umap_clustering/RNA_only_CELL_UMAP.csv')
cluster=u1.cluster.to_numpy(); emb=[('Full_B2',u1[['UMAP1','UMAP2']].to_numpy()),('RNA_only',u2[['UMAP1','UMAP2']].to_numpy())]
bv.plots(meta,deg,ora,zf,zr,cluster,emb); bv.write_docs(meta,deg,ora,edges,ov)
bv.write_json(S/'logs/RUN_COMPLETE.json',{'status':'COMPLETE','fold':'fold1','seed':20020132,'cohort':'GSE214979','cells':len(meta),'genes':len(f),'deg_method':'logCPM Welch fallback','gpu_used':False,'target_atac_used':False})
print('RESUME_COMPLETE')

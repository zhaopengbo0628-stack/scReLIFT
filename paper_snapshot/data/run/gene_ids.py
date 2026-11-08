import h5py, numpy as np
b='/home/u202522802045/share/zpb/data/raw/BMMC/GSE194122/GSE194122_openproblems_neurips2021_multiome_BMMC_processed.h5ad'
with h5py.File(b,'r') as f:
    print('BMMC var index',f['var/_index'][:8])
    print('BMMC gene_id codes',f['var/gene_id'][:8])
    print('BMMC categories',f['var/__categories/gene_id'][:8])
p='/home/u202522802045/share/zpb/final_dmode_b2full_updated_pbmc002/validated_prepared/D2/gene_ids.npy'
print('PBMC exists',__import__('pathlib').Path(p).exists())
if __import__('pathlib').Path(p).exists():
    g=np.load(p,allow_pickle=True); print('PBMC',g.shape,g[:8])

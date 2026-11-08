from common import *
import h5py
from scipy.io import mmread
p=SHARED/'data/cache/d_mode/PBMC/legacy_full_b2_corrected/_historical_dependencies'
with h5py.File(SHARED/'data/raw/PBMC/PBMC-MO-003/pbmc_unsorted_10k_filtered_feature_bc_matrix.h5') as h:
    ft=h['matrix/features/feature_type'][:].astype(str)=='Gene Expression'
    ids=h['matrix/features/id'][:].astype(str)[ft]
lookup=dict(zip(ids,range(len(ids))))
oldids=np.load(p/'validated_prepared/D2/gene_ids.npy').astype(str)
selected=np.asarray([lookup[g] for g in oldids])
old=np.load(p/'validated_prepared/D2/rna.npy')
x=mmread(RUN/'input/D2_author/rna_counts_genes_by_cells.mtx').T.tocsr()
for dtype in [np.float32,np.float64,np.int32,np.int64]:
    new=normalize_log1p(x.astype(dtype))[:,selected].toarray().astype(np.float32)
    print(dtype,'maxerr',np.max(np.abs(new-old)),'neq',np.count_nonzero(new!=old),'zero_mismatch',np.count_nonzero((new==0)!=(old==0)))

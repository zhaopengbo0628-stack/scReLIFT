"""Full-library QC RNA view; CPU only, runs after other plotting completes."""
import time
from common import *

def main():
    while not (RUN/'audit/POSTPROCESS_STATUS.json').exists(): time.sleep(30)
    from postprocess import render
    p=SHARED/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common/D1'
    x=sparse.load_npz(p/'rna_counts.npz').tocsr()
    assert x.shape==(9347,36601)
    # No new QC filtering, no HVG, no PCA; same inherited paired/QC cohort.
    render('RNA_QC_fullgene_log1p_no_PCA',20020130,normalize_log1p(x).astype(np.float32),metric='cosine')
    write(RUN/'audit/QC_RNA_PLOT_STATUS.json',dict(status='COMPLETE',cells=9347,genes=36601,
          new_qc_applied=False,new_hvg_applied=False,pca_applied=False,
          normalization='unchanged full-library CP10K + log1p',metric='cosine',
          role='posthoc full-library RNA visualization; not used for Student fitting or selection'))

if __name__=='__main__': main()

"""Verify actual completed artifacts, checkpoint epoch and full barcode coverage."""
import pandas as pd
import torch
from common import *

def main():
    cells=(RUN/'artifacts/D1_barcodes.tsv').read_text().splitlines()
    assert len(cells)==len(set(cells))==9347
    rows=[]
    for method in ['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2']:
        for seed in SEEDS:
            unit=RUN/method/f'seed_{seed}'
            metrics=json.loads((unit/'metrics.json').read_text())
            x=np.load(unit/'target_embedding_raw.npy')
            assert x.shape[0]==9347 and np.isfinite(x).all()
            assert sha(unit/'target_embedding_raw.npy')==metrics['target_embedding_sha256']
            assignments=pd.read_csv(unit/'cluster_assignments.csv')
            assert assignments.barcode.tolist()==cells
            assert int((assignments.reference_label=='CD8 TEM').sum())==622
            row=dict(method=method,seed=seed,cells=9347,finite=True,barcode_order_exact=True,
                     cd8_tem=622,embedding_sha256=sha(unit/'target_embedding_raw.npy'))
            if method=='FullB2':
                manifest=json.loads((unit/'manifest.json').read_text());assert manifest['status']=='COMPLETE'
                for filename,expected in manifest['hashes'].items():
                    assert sha(unit/filename)==expected, f'Artifact changed: {unit/filename}'
                for stage in ['teacher','student']:
                    ck=torch.load(unit/f'checkpoints/{stage}/latest.pt',map_location='cpu',weights_only=False)
                    assert ck['epoch']==100 and all(torch.isfinite(v).all() for v in ck['model'].values())
                assert x.shape[1]==64
                assert manifest['target_used_for_training'] is False and manifest['target_used_for_checkpoint_selection'] is False
                row.update(teacher_epoch=100,student_epoch=100,manifest_hashes_match=True)
            else:
                assert x.shape[1]==50
            rows.append(row)
    pd.DataFrame(rows).to_csv(RUN/'audit/RESULT_INTEGRITY.csv',index=False)
    q=pd.read_csv(RUN/'JOB_QUEUE.csv');assert len(q)==4 and q.Status.eq('COMPLETED').all()
    write(RUN/'audit/RESULT_INTEGRITY.json',dict(status='PASS',units=9,full_b2_seeds=SEEDS,
        source_all_cells=16835,target_all_cells=9347,cd8_tem_cells=622,
        no_target_training=True,all_checkpoints_final_epoch_100=True,all_artifact_hashes_verified=True))
    print('RESULT_INTEGRITY_PASS',flush=True)

if __name__=='__main__': main()

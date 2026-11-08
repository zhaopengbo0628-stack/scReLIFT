import json
import tempfile
import unittest
from pathlib import Path

import anndata as ad
import h5py
import numpy as np
import pandas as pd
from scipy import sparse

from stage_seaad import extract, sha, stage


class StageSeaadTest(unittest.TestCase):
    def test_paired_raw_counts_and_metadata(self):
        with tempfile.TemporaryDirectory() as tmp:
            raw = Path(tmp)/'raw'; raw.mkdir()
            output = Path(tmp)/'benchmark'
            obs = pd.DataFrame({'method': ['10xMulti']*4, 'Donor ID': ['d1','d1','d2','d2'],
                'Class': ['Neuron']*4, 'Subclass': ['Sst']*4, 'Supertype': ['Sst_1']*4,
                'Age at Death': [80,80,85,85], 'Sex': ['F','F','M','M'],
                'Cognitive Status': ['No dementia']*2+['Dementia']*2}, index=['a','b','c','d'])
            counts = sparse.csr_matrix(np.array([[1,2],[3,4],[5,6],[7,8]], dtype=float))
            rna = ad.AnnData(counts.copy(), obs=obs, var=pd.DataFrame(
                {'gene_ids':['ENSG1','ENSG2'], 'genome':['GRCh38']*2}, index=['g1','g2']))
            rna.layers['UMIs'] = counts.copy()
            rna.X.data[:] = 0.5
            atac = ad.AnnData(counts[[2,0,3,1]], obs=obs.iloc[[2,0,3,1]].copy(),
                var=pd.DataFrame(index=['chr1:1-10','chr2:20-30']))
            for name, data in [('rna.h5ad',rna),('atac.h5ad',atac)]:
                data.write_h5ad(raw/name)
                (raw/(name+'.download.json')).write_text(json.dumps(dict(status='COMPLETE',
                    size_bytes=(raw/name).stat().st_size,sha256=sha(raw/name))))
            (raw/'PAIRING_REPORT.json').write_text(json.dumps(dict(rna='rna.h5ad',atac='atac.h5ad',
                permitted_full_download=True, identity_mismatches={'Donor ID':0},paired_cells=3)))
            pd.DataFrame({'cell_id':['d','a','c'], 'rna_row':[3,0,2], 'atac_row':[2,1,0],
                'identity_consistent':[True]*3}).to_csv(raw/'PAIRED_CELLS.csv',index=False)
            stage(raw,output,'synthetic_test')
            for name in ('rna','atac'):
                with h5py.File(output/(name+'_counts.h5'),'r') as f:
                    g=f['X']; result=sparse.csr_matrix((g['data'][:],g['indices'][:],g['indptr'][:]),
                        shape=tuple(g.attrs['shape']))
                    np.testing.assert_array_equal(result.toarray(),counts[[0,2,3]].toarray())
            self.assertEqual((output/'barcodes.txt').read_text().splitlines(),['a','c','d'])
            self.assertEqual(json.loads((output/'manifest.json').read_text())['status'],'COMPLETE')
            self.assertFalse((output/'donor_split.csv').exists())

    def test_nonfinite_counts_stop(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'bad.h5ad'
            ad.AnnData(sparse.csr_matrix([[1.0,np.nan]])).write_h5ad(path)
            with h5py.File(path,'r') as f:
                with self.assertRaisesRegex(ValueError,'Nonfinite'):
                    extract(f['X'],np.array([0]),Path(tmp)/'out.h5')


if __name__ == '__main__':
    unittest.main()

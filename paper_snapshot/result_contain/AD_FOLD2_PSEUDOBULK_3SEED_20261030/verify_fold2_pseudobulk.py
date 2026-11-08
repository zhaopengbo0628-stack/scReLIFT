from pathlib import Path
import hashlib, json
import numpy as np
import pandas as pd
import h5py
from scipy.stats import t as t_dist
from statsmodels.stats.multitest import multipletests

ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/AD_FOLD2_PSEUDOBULK_3SEED_20261030'
BENCH=ROOT/'result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04'
z=np.load(OUT/'PSEUDOBULK_RAW_COUNTS.npz')
summary=pd.read_csv(OUT/'ALL_ANALYSES_SUMMARY.csv')
assert len(summary)==24 and not summary.duplicated(['scope','cell_type']).any()
assert set(summary.scope)=={'true_label_reference','seed_20020130','seed_20020131','seed_20020132'}
meta=pd.read_csv(BENCH/'preprocessing/cohorts/GSE268609/cells.csv.gz',dtype={'sample_group_id':str})
coverage=pd.read_csv(OUT/'DONOR_CELLTYPE_COVERAGE.csv',dtype={'unit_id':str})
unit='28'
pair=coverage[(coverage.unit_id==unit)&(coverage.cell_type!='ALL_CELLS')].sort_values('n_cells',ascending=False).iloc[0]
mapping={'Excitatory':['mGC','CA_neurons','CA2-4_neurons'],'Inhibitory':['GABA_neurons'],
         'Astrocytes':['Astrocytes'],'Oligodendrocytes':['mOli'],'OPC':['OPCs']}
rows=meta.loc[meta.sample_group_id.eq(unit)&meta.Cluster.isin(mapping[pair.cell_type]),'count_row'].to_numpy(int)
assert len(rows)==pair.n_cells and len(rows)>0
with h5py.File(meta.count_file.iloc[0],'r') as f:
    g=f['X']
    pointers=g['indptr'][:]
    gene_var=f['var/id']
    if isinstance(gene_var,h5py.Dataset):
        native_ids=gene_var.asstr()[:]
    else:
        native_ids=gene_var['categories'].asstr()[:][gene_var['codes'][:]]
    raw=np.zeros(len(native_ids),np.int64)
    for row in rows:
        lo,hi=int(pointers[row]),int(pointers[row+1])
        np.add.at(raw,g['indices'][lo:hi],g['data'][lo:hi].astype(np.int64))
lookup={g:i for i,g in enumerate(native_ids)}
raw=raw[np.array([lookup[g] for g in z['genes']])]
idx=int(np.where(z['units']==unit)[0][0]);typ=int(np.where(z['cell_types']==pair.cell_type)[0][0])
assert np.array_equal(raw,z['counts'][typ,idx]), 'independent direct HDF5-row aggregation mismatch'

scope='seed_20020131'; celltype='OPC'
idx=int(np.where(z['cell_types']==celltype)[0][0])
units=z['units']
labels=pd.read_csv(OUT/scope/'GROUP_ASSIGNMENTS.csv',dtype={'unit_id':str}).set_index('unit_id').loc[units,'label'].to_numpy(int)
x=z['counts'][idx]
keep=(z['n_cells'][idx]>=20)&(x.sum(1)>0)
x=x[keep]; labels=labels[keep]
cpm=x/x.sum(1,keepdims=True)*1e6
logged=np.log1p(cpm)
pos,neg=logged[labels==1],logged[labels==0]
n1,n0=len(pos),len(neg)
v1,v0=pos.var(0,ddof=1)/n1,neg.var(0,ddof=1)/n0
with np.errstate(divide='ignore',invalid='ignore'):
    t=(pos.mean(0)-neg.mean(0))/np.sqrt(v1+v0)
    df=(v1+v0)**2/(v1*v1/(n1-1)+v0*v0/(n0-1))
    manual=2*t_dist.sf(np.abs(t),df)
manual=np.nan_to_num(manual,nan=1.,posinf=1.,neginf=1.)
q=multipletests(manual,method='fdr_bh')[1]
tab=pd.read_csv(OUT/scope/celltype/'DEG_all_tested.csv.gz').set_index('gene_id').loc[z['genes']]
assert np.allclose(manual,tab.pvalue.to_numpy(),rtol=1e-8,atol=1e-12)
assert np.allclose(q,tab.FDR.to_numpy(),rtol=1e-8,atol=1e-12)
for row in summary.to_dict('records'):
    dest=OUT/row['scope']/row['cell_type']
    terms=pd.read_csv(dest/'ENRICHMENT_significant.csv')
    if row['status']=='NOT_EVALUABLE':
        assert len(terms)==0 and pd.isna(row['significant_genes'])
        continue
    deg=pd.read_csv(dest/'DEG_significant.csv')
    assert len(deg)==row['significant_genes']
    for lib,prefix in [('GO_BP','GO'),('Reactome','Reactome')]:
        for direction in ['AD_up','Control_up']:
            count=int(((terms.library==lib)&(terms.direction==direction)).sum())
            assert count==row[prefix+'_'+direction]
    if len(terms):assert (terms.FDR<.05).all() and (terms.overlap>=3).all()
before=json.loads((OUT/'PROTECTED_INPUT_HASHES_BEFORE.json').read_text())
for p,digest in before.items():
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    assert h.hexdigest()==digest
report=dict(status='PASS',analyses_verified=len(summary),
            independent_HDF5_count_replay={'unit_id':unit,'cell_type':str(pair.cell_type),'cells':len(rows),'gene_vector_exact':True},
            independent_Welch_formula_and_statsmodels_BH={'scope':scope,'cell_type':celltype,'genes':len(tab),
            'max_abs_p_error':float(np.max(np.abs(manual-tab.pvalue.to_numpy()))),
            'max_abs_FDR_error':float(np.max(np.abs(q-tab.FDR.to_numpy())))},
            DEG_and_ORA_counts_verified=True,protected_inputs_unchanged=True)
(OUT/'INDEPENDENT_VERIFICATION.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))

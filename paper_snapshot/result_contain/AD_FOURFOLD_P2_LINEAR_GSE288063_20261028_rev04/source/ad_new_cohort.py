"""Author-QC paired-count adapter for the fixed 10-donor hippocampus cohort."""
from pathlib import Path
import ast, importlib.util, json
import h5py
import numpy as np
import pandas as pd

ROOT = Path('/home/u202522802045/share/zpb')
BUNDLE = ROOT/'data/raw/AD/GSE288063_bundle_10donor'
NAME = 'GSE288063_GSE278576'
EXPECTED = {'ad13':5344,'ad1551':3760,'ad22':3935,'ad24':5018,'ad671':6393,
            'hc1153':2550,'hc1216':7955,'hc19':4825,'hc26':8917,'hc40':6302}

def broad_types(frame):
    out=np.full(len(frame),'Unknown',object)
    out[frame['class'].eq('Excitatory_Neuron')]='Excitatory'
    out[frame['class'].eq('Inhibitory_Neuron')]='Inhibitory'
    mp={'Astro':'Astrocytes','Oligo':'Oligodendrocytes','OPC':'OPC',
        'Microglia':'Microglia/PVM','Macro':'Microglia/PVM','Endo':'Endothelial','VLMC':'Pericytes/VLMC'}
    for label,value in mp.items():out[frame.subclass.eq(label)]=value
    return out.astype(str)

def first_three(legacy):
    """Reuse frozen readers/splits without opening the excluded retired_legacy_cohort."""
    tree=ast.parse(Path(legacy.__file__).read_text())
    fun=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='source_defs')
    cut=next(i for i,n in enumerate(fun.body) if isinstance(n,ast.Assign)
             and any(isinstance(t,ast.Name) and t.id=='pm' for t in n.targets))
    ret=ast.parse("return genes, {'GSE214979':(r1,a1,c1,pk1,tr1,va1,'donor_id'), 'GSE268609':(r2,a2,s2,pk2,tr2,va2,'sample_group_id'), 'SEAAD_MTG':(r3,a3,s3,pk3,tr3,va3,'Donor ID')}").body[0]
    fun.body=fun.body[:cut]+[ret]
    ns=dict(vars(legacy))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[fun],type_ignores=[])),legacy.__file__,'exec'),ns)
    return ns['source_defs']()

def new_definition(legacy,genes):
    report=json.loads((BUNDLE/'verification_report_rev04.json').read_text())
    assert report['status']=='PASS' and report['all_metadata_barcodes_observed'] and report['all_paired']
    manifest=json.loads((BUNDLE/'download_manifest.json').read_text())
    assert manifest['status']=='COMPLETE'
    author=pd.read_csv(manifest['metadata_path'],sep='\t')
    parts_r,parts_a,metas,native=[],[],[],[]
    for item in manifest['files']:
        p=Path(item['path']);lib=item['donor']
        if 'ad22rep1' in p.name:lib='ad22_rep1'
        if 'ad22rep2' in p.name:lib='ad22_rep2'
        assert legacy.sha(p)==item['sha256'],p
        ann=author[author.library.eq(lib)].copy()
        assert len(ann)>0 and ann.cell_barcode.is_unique and set(ann.donor)=={item['donor']}
        with h5py.File(p,'r') as h:
            m=h['matrix'];ft=m['features/feature_type'][:].astype(str)
            ids=m['features/id'][:].astype(str);bc=m['barcodes'][:].astype(str)
            assert np.array_equal(ids[ft=='Gene Expression'],genes)
            rows=np.flatnonzero(np.isin(bc,ann.cell_barcode.to_numpy(dtype=bc.dtype)))
            assert len(rows)==len(ann)
            ann=ann.set_index('cell_barcode').loc[bc[rows]].reset_index()
            pk=ids[ft=='Peaks'];native.append(pk)
            parts_r.append(legacy.TenX(p,rows,np.flatnonzero(ft=='Gene Expression')))
            parts_a.append(legacy.TenX(p,rows,np.flatnonzero(ft=='Peaks')))
        ann['barcode']=lib+':'+ann.cell_barcode.astype(str)
        ann['library_id']=lib;ann['donor_id']=ann.donor.astype(str)
        ann['disease_status']=ann.group.map({'AD':'AD','Normal':'Control'})
        assert ann.disease_status.notna().all()
        ann['count_row']=rows;ann['count_file']=str(p)
        ann['broad_type']=broad_types(ann);ann['qc_status']='author_metadata_barcode_subset'
        metas.append(ann)
    meta=pd.concat(metas,ignore_index=True)
    assert meta.barcode.is_unique and len(meta)==54999
    assert meta.groupby('donor_id').size().to_dict()==EXPECTED
    assert meta.groupby('donor_id').disease_status.nunique().eq(1).all()
    peaks=np.unique(np.concatenate(native))
    maps=[np.searchsorted(peaks,p) for p in native]
    rna=legacy.Concat(parts_r);atac=legacy.Concat(parts_a,maps,len(peaks))
    tr,va=legacy.split(meta,'donor_id','disease_status',NAME)
    return rna,atac,meta,peaks,tr,va,'donor_id'

def source_defs(legacy):
    genes,defs=first_three(legacy)
    defs[NAME]=new_definition(legacy,genes)
    return genes,defs

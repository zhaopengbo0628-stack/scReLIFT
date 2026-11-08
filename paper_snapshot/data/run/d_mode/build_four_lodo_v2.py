"""Build four LODO B_D folds from the current mouse Sources. No training."""
import json, sys, time, re
from dataclasses import asdict
from pathlib import Path
import numpy as np
from scipy import sparse
ROOT=Path('/home/u202522802045/share/zpb')
sys.path[:0]=[str(ROOT/'main/run/d_mode'),str(ROOT/'main/core/regulatory/common/legacy_support')]
import prepare_updated_b2_sources as frozen
import v5d.regulatory_graph as rg
import v5d.harmonize as hz
_PEAK=re.compile(r"^(?P<chrom>.+?)(?:[:_]|-)(?P<start>\d+)-(?P<end>\d+)$")
def _parse_peak(value):
    m=_PEAK.match(str(value))
    if not m: raise ValueError(f'Invalid peak coordinate: {value}')
    c=m.group('chrom'); c=c[3:] if c.lower().startswith('chr') else c
    c='MT' if c=='M' else c
    return c,int(m.group('start')),int(m.group('end'))
rg.PEAK_PATTERN=_PEAK; rg.parse_peak=_parse_peak; hz.parse_peak=_parse_peak; frozen.parse_peak=_parse_peak
from v5d.dataset_spec import DatasetSpec
from v5d.harmonize import prepare_harmonized
from v5d.data import V5DData
from v5d.dmode_reliable_selector import DModeReliabilityConfig, build_source_only_b
from v5d.regulatory_graph import read_tss
CACHE=ROOT/'data/cache/d_mode/Mouse-brain/four_dataset_v2_20261024'
TARGETS=['D3','D4','Gao','ISSAAC']
GTF=ROOT/'data/reference/genome_annotation/Mus_musculus/GRCm38/gencode_M25/gencode.vM25.chr_patch_hapl_scaff.annotation.gtf.gz'
def build(target):
    assert target in TARGETS
    sources=[k for k in TARGETS if k!=target]
    folder=CACHE/'folds'/f'target_{target}'/'source_artifacts'
    if (folder/'manifest.json').exists():
        print('VERIFIED',target,flush=True); return
    if folder.exists() and any(folder.iterdir()):
        raise RuntimeError(f'incomplete: {folder}')
    started=time.perf_counter(); folder.mkdir(parents=True)
    specs=[DatasetSpec(k,CACHE/'source_prepared'/k) for k in sources]
    prepare_harmonized(specs,folder/'harmonized',feature_reference_names=sources)
    genes=np.load(folder/'harmonized/common_gene_ids.npy').astype(str)
    peaks=np.load(folder/'harmonized/canonical_peak_names.npy').astype(str)
    tss=read_tss(GTF)
    missing=[g for g in genes if g not in tss]
    if missing: raise RuntimeError(f'genes without TSS: {missing[:5]}')
    graph=frozen.graph_fast(genes,peaks,tss); graph['peak_names']=peaks
    np.savez_compressed(folder/'candidate_graph.npz',**graph)
    np.save(folder/'gene_ids.npy',genes); np.save(folder/'peak_names.npy',peaks)
    prep=time.perf_counter()-started
    datasets=[V5DData(folder/'harmonized',k) for k in sources]
    tick=time.perf_counter()
    selection=build_source_only_b(datasets,graph,genes,DModeReliabilityConfig(),20020130)
    sel=time.perf_counter()-tick
    assert selection['B_D'].nnz>0
    sparse.save_npz(folder/'B_D.npz',selection['B_D'])
    for name,value in selection.items():
        if isinstance(value,np.ndarray): np.save(folder/f'{name}.npy',value)
        elif isinstance(value,list) and value and isinstance(value[0],np.ndarray):
            np.savez_compressed(folder/f'{name}.npz',**dict(zip(sources,value)))
    l1=np.asarray(abs(selection['B_D']).sum(0)).ravel()
    manifest=dict(status='LODO_B_D_COMPLETE',target=target,sources=sources,
        gene_count=len(genes),peak_count=len(peaks),candidate_edges=int(graph['candidate_mask'].sum()),
        B_edges=int(selection['B_D'].nnz),retained_genes=int((l1>0).sum()),retained_genes_nonzero=int(np.count_nonzero(np.asarray(selection['B_D'].getnnz(0)).ravel())),
        source_cells={d.name:int(d.rna.shape[0]) for d in datasets},source_train={d.name:int(d.train.size) for d in datasets},
        source_val={d.name:int(d.val.size) for d in datasets},source_test={d.name:int(d.test.size) for d in datasets},
        config=asdict(DModeReliabilityConfig()),graph_window=250000,graph_candidates=50,
        preparation_seconds=prep,selector_seconds=sel,
        target_atac_loaded=False,target_labels_loaded=False,target_hvg_fitted=False,
        note='B_D built from three Sources; target held out; no model training')
    (folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(manifest,indent=2),flush=True)
if __name__=='__main__':
    for t in TARGETS:
        print('BUILD',t,flush=True); build(t)

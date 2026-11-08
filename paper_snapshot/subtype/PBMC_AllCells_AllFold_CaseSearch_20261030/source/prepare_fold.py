"""Rebuild source-only harmonization, regulatory B_D, F and scaler for the selected target."""
import time
from dataclasses import asdict
import pandas as pd
from common import *
from v5d.dataset_spec import DatasetSpec
from v5d.harmonize import prepare_harmonized
from v5d.regulatory_graph import read_tss
from v5d.dmode_reliable_selector import DModeReliabilityConfig,build_source_only_b
from v5d.dmode_teacher import fit_dmode_field
from v5d.data import V5DData


def main():
    cfg=json.loads((RUN/'fold_config.json').read_text());sources=cfg['sources'];target=cfg['target'];art=RUN/'artifacts'
    if (art/'PREPARATION_COMPLETE.json').exists():
        m=json.loads((art/'PREPARATION_COMPLETE.json').read_text())
        for f,h in m['hashes'].items():assert sha(RUN/f)==h
        print('PREPARATION_REUSED',target,flush=True);return
    tss=read_tss(Path('/home/u202522802045/glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz'))
    specs=[DatasetSpec(k,art/'native'/k) for k in sources]
    prepare_harmonized(specs,art/'harmonized',feature_reference_names=sources,held_out_target_name=target)
    genes=np.load(art/'harmonized/common_gene_ids.npy').astype(str);peaks=np.load(art/'harmonized/canonical_peak_names.npy').astype(str)
    graph=graph_fast(genes,peaks,tss);graph['peak_names']=peaks;np.savez_compressed(art/'candidate_graph.npz',**graph)
    datasets=[V5DData(art/'harmonized',k) for k in sources]
    config=DModeReliabilityConfig();started=time.perf_counter()
    print('BUILD_B_D',target,len(genes),len(peaks),flush=True)
    result=build_source_only_b(datasets,graph,genes,config,20020130)
    sparse.save_npz(art/'B_D.npz',result['B_D'])
    for key,value in result.items():
        if isinstance(value,np.ndarray):np.save(art/f'{key}.npy',value)
        elif isinstance(value,list) and value and isinstance(value[0],np.ndarray):np.savez_compressed(art/f'{key}.npz',**{s.name:v for s,v in zip(datasets,value)})
    write(art/'regulatory_manifest.json',dict(config=asdict(config),selection=result['manifest'],seconds=time.perf_counter()-started,target_loaded=False))
    for data in datasets:
        field,state,manifest=fit_dmode_field(data.atac,result['B_D'],data.train)
        np.save(art/f'{data.name}_F.npy',field);np.savez_compressed(art/f'{data.name}_field_state.npz',**state)
        write(art/f'{data.name}_field_manifest.json',manifest)
    blocks=[np.asarray(d.rna,dtype=np.float32) for d in datasets]
    means=np.stack([x.mean(0) for x in blocks]);mu=means.mean(0).astype(np.float32)
    std=np.sqrt(np.mean(np.stack([x.var(0) for x in blocks])+(means-mu)**2,axis=0)).astype(np.float32)
    np.savez_compressed(art/'source_scaler.npz',mean=mu,std=std);np.save(art/'gene_ids.npy',genes)
    # Target is first read after all Source-only feature/template/scaler fitting has completed.
    registry=json.loads(Path(cfg['registry']).read_text());d=registry[target]
    x=sparse.load_npz(d['rna']).tocsr();ids=pd.read_csv(d['gene_table'],sep='\t').iloc[:,0].astype(str).to_numpy()
    cells=Path(d['barcodes']).read_text().splitlines();lookup={g:i for i,g in enumerate(ids)}
    selected=np.asarray([lookup[g] for g in genes])
    values=normalize_log1p(x)[:,selected].toarray().astype(np.float32)
    np.save(art/f'{target}_rna_all.npy',values);(art/f'{target}_barcodes.tsv').write_text('\n'.join(cells)+'\n')
    write(art/'PREPARATION_COMPLETE.json',dict(status='COMPLETE',sources=sources,target=target,
        source_cells=cfg['source_cells'],target_cells=len(cells),gene_count=len(genes),edges=int(result['B_D'].nnz),
        source_all_cells=True,target_training=False,target_atac_loaded=False,
        hashes={str(p.relative_to(RUN)):sha(p) for p in art.rglob('*') if p.is_file()}))
    print('PREPARATION_COMPLETE',target,len(genes),flush=True)


if __name__=='__main__':main()

"""Rebuild FC cache from the already filtered benchmark, preserving old artifacts."""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

ROOT = Path('/home/u202522802045/share/zpb')
sys.path[:0] = [str(ROOT), str(ROOT/'data/run/s_mode'), str(ROOT/'main/core/regulatory/common/legacy_support'), str(ROOT/'comparison/run/gao_intra')]
from common import read, write, sha
from build_fc_labeled import graph_fast, GTF
from main.core.regulatory.s_mode.single_source import normalize_counts, select_hvg, build_bs, build_field
from v5d.regulatory_graph import read_tss

BENCH = ROOT/'data/benchmark/Mouse-brain/FC_author_labeled_v1_20261023/FC'
OLD = ROOT/'data/cache/s_mode/Mouse-brain/FC_author_labeled_v1_20261023/FC'
OUT = ROOT/'data/cache/s_mode/Mouse-brain/FC_author_labeled_aligned_v2_20261023/FC'
TASK = ROOT/'result_contain/88_mousebrain_comparison_intra_4datasets_3seed_20261023'


def main():
    started = time.perf_counter()
    assert not OUT.exists(), str(OUT)
    assert read(TASK/'FC_CELL_ALIGNMENT_AUDIT.json')['old_results_valid_for_comparison'] is False
    for p,h in read(OLD/'manifest.json')['hashes'].items():
        assert sha(p) == h, p
    meta = pd.read_csv(BENCH/'metadata.csv', keep_default_na=False)
    assert np.array_equal(np.load(BENCH/'barcodes.npy'), meta.barcode.to_numpy(str))
    rna = sparse.load_npz(BENCH/'rna_counts.npz').tocsr()
    atac = sparse.load_npz(BENCH/'atac_counts.npz').tocsr().astype(bool).astype(np.float32)
    assert rna.shape[0] == atac.shape[0] == len(meta) == 9821
    indices = {s:np.load(OLD/f'{s}_index.npy') for s in ['train','val','test']}
    assert np.array_equal(np.sort(np.concatenate(list(indices.values()))), np.arange(len(meta)))
    for s,ix in indices.items():
        assert (meta.iloc[ix].split == s).all()
    full = normalize_counts(rna)
    genes = np.load(BENCH/'gene_ids.npy').astype(str)
    peaks = np.load(BENCH/'peak_names.npy').astype(str)
    mapping = pd.read_csv(BENCH/'identity_map/gene_mapping.tsv', sep='\t', keep_default_na=False)
    assert mapping.input_feature.tolist() == genes.tolist()
    tss = read_tss(GTF)
    eligible = np.flatnonzero(mapping.status.eq('unique').to_numpy() & mapping.ensembl_gene_id.isin(tss).to_numpy())
    selected,_ = select_hvg(full[indices['train']][:,eligible], 2000)
    columns = eligible[selected]
    values = full[:,columns].toarray().astype(np.float32)
    graph = graph_fast(genes[columns], peaks, tss); graph['peak_names'] = peaks
    tick = time.perf_counter()
    bs,diagnostics = build_bs(values[indices['train']], atac[indices['train']], graph)
    selector_seconds = time.perf_counter()-tick
    paired = np.r_[indices['train'],indices['val']]
    tick = time.perf_counter()
    field,state,report = build_field(atac[paired], bs, np.arange(len(indices['train'])))
    field_seconds = time.perf_counter()-tick
    OUT.mkdir(parents=True)
    for s,ix in indices.items():
        np.save(OUT/f'rna_{s}.npy', values[ix]); np.save(OUT/f'{s}_index.npy', ix)
    np.save(OUT/'F_train.npy', field[:len(indices['train'])]); np.save(OUT/'F_val.npy', field[len(indices['train']):])
    np.save(OUT/'gene_ids.npy', genes[columns]); np.save(OUT/'peak_names.npy', peaks)
    np.savez_compressed(OUT/'rna_state.npz', mean=values[indices['train']].mean(0), std=values[indices['train']].std(0), fit_cell_index=indices['train'])
    np.savez_compressed(OUT/'field_state.npz', **state)
    np.savez_compressed(OUT/'candidate_graph.npz', **graph)
    np.savez_compressed(OUT/'regulatory_stability.npz', **diagnostics)
    sparse.save_npz(OUT/'B_S.npz', bs)
    (OUT/'peak_availability.npz').symlink_to(BENCH/'peak_availability.npz')
    pd.DataFrame(dict(hvg_rank=np.arange(2000), ensembl_gene_id=genes[columns], native_column=columns)).to_csv(OUT/'hvg.csv', index=False)
    write(OUT/'manifest.json', dict(status='S_MODE_INTRA_CACHE_COMPLETE', dataset='FC', cells=len(meta), genes=2000,
        split_counts={s:len(ix) for s,ix in indices.items()}, replaces_invalid_cache=str(OLD),
        correction='Normalize the filtered 9821-row benchmark; fit HVG, RNA state and B/F on its frozen train cells',
        new_QC=False, split_changed=False, labels_used_for_fitting=False, test_used_for_fitting=False,
        selector_seconds=selector_seconds, field_seconds=field_seconds, preparation_seconds=time.perf_counter()-started,
        field_report=report, code_sha256=sha(__file__),
        hashes={str(p):sha(p) for p in [BENCH/'rna_counts.npz', BENCH/'atac_counts.npz', BENCH/'metadata.csv', BENCH/'barcodes.npy', BENCH/'gene_ids.npy']}))
    print('FC_ALIGNED_CACHE_READY', OUT, flush=True)


if __name__ == '__main__':
    main()

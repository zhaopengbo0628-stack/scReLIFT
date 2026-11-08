"""Apply the existing SOURCE-only coordinate mapping to published dictionaries, no counts."""
import ast
import hashlib
import importlib.util
import json
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path('E:/my_research')
OUT = Path(__file__).resolve().parent.parent
DEST = OUT/'regulatory_edges'
OLD = ROOT/'result_contain/AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout'


if __name__ == '__main__':
    if (DEST/'canonical_coordinate_audit.json').exists():
        raise FileExistsError('Preserve completed audit')
    graph_path = OUT/'source_snapshot/regulatory_graph_frozen.py'
    spec = importlib.util.spec_from_file_location('graph_frozen', graph_path)
    graph = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(graph)
    path = OUT/'source_snapshot/harmonize_frozen.py'
    tree = ast.parse(path.read_text())
    functions = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in {
        '_canonical_peaks_from_sources', '_overlap_mapping'}]
    assert len(functions) == 2
    namespace = dict(np=np, parse_peak=graph.parse_peak, Path=Path)
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), 'exec'), namespace)
    names = {}
    inputs = {}
    for name, version in [('GSE214979','paired_v1'), ('GSE268609','paired_counts_v1')]:
        src = ROOT/'data/benchmark/AD'/name/version/'post_qc/atac_counts.h5ad'
        with h5py.File(src) as f:
            v = f['var']; names[name] = v[v.attrs['_index']].asstr()[:]
        inputs[name] = str(src)
    src = OLD/'clinical_evidence/SEAAD_official_atac_var.csv'
    names['SEAAD_MTG'] = pd.read_csv(src)['_index'].to_numpy(str)
    inputs['SEAAD_MTG'] = str(src)
    roots = []
    for name, peaks in names.items():
        assert len(np.unique(peaks)) == len(peaks)
        folder = DEST/name
        folder.mkdir(exist_ok=False)
        np.save(folder/'peak_names.npy', peaks)
        roots.append(folder)
    canonical = namespace['_canonical_peaks_from_sources'](roots)
    np.save(DEST/'canonical_peak_names.npy', canonical)
    rows, available = [], []
    for name, peaks in names.items():
        mapping, stats = namespace['_overlap_mapping'](peaks, canonical)
        assert (mapping >= 0).all()
        np.save(DEST/name/'native_to_canonical.npy', mapping)
        present = np.bincount(mapping, minlength=len(canonical)) > 0
        available.append(present)
        rows.append(dict(cohort=name, **stats))
    present = np.stack(available)
    widths = np.array([graph.parse_peak(p)[2]-graph.parse_peak(p)[1] for p in canonical])
    pd.DataFrame(rows).to_csv(DEST/'canonical_mapping_by_source.csv', index=False)
    pd.DataFrame(dict(canonical_peak=canonical, width_bp=widths,
        source_coverage=present.sum(0), **{name: p for name,p in zip(names,present)})).to_csv(
            DEST/'canonical_region_availability.csv.gz', index=False)
    report = dict(status='COORDINATE_DICTIONARY_AUDIT_ONLY_NOT_STABLE_EDGES',
        method='Existing harmonize.py merged union, strict half-open overlap, binary OR aggregation',
        canonical_regions=len(canonical), three_source_measurable_regions=int(present.all(0).sum()),
        width_median_bp=float(np.median(widths)), width_p99_bp=float(np.quantile(widths,.99)),
        width_max_bp=int(widths.max()), regions_over_100kb=int((widths>100000).sum()),
        inputs=inputs, harmonize_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        regulatory_graph_sha256=hashlib.sha256(graph_path.read_bytes()).hexdigest(),
        source_expression_or_counts_read=False, target_used=False, stable_edges_computed=False,
        biological_coordinate_provenance='prior_GRCh38_evidence; original_annotation_release_for_SOURCE2_SEAAD_unverified',
        limitations=['Published feature dictionaries may include validation donors; no expression-driven selection performed.',
                    'Merged regions can span multiple native peaks; this is binary accessibility, not summed fragment counts.',
                    'Coordinate availability does not establish reproducible regulatory relationships.'])
    (DEST/'canonical_coordinate_audit.json').write_text(json.dumps(report, indent=2)+'\n')
    print(json.dumps(report, indent=2))

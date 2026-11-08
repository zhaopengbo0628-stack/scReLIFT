"""Read-only, filename-only search; does not load or modify shared matrices."""
import os
from pathlib import Path

base = Path('/home/u2021082/share/zpb')
for root in [Path('/home/u2021082/zpb'), Path('/home/u202522802045'), base]:
    print('ROOT', root, root.exists(), 'resolved', root.resolve())
for sub in ['data/benchmark/PBMC/PBMC-MO-002/legacy_corrected_refs_v1/native_access',
            'DATA/PBMC-MO-002__10X__GranSorted__10k__Multiome',
            'result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_prepared/D2']:
    p = base/sub
    print('PATH', p, 'exists', p.exists(), 'link', p.is_symlink())
    if p.is_dir():
        for child in p.iterdir():
            print('CHILD', child.name, child.is_symlink(), child.resolve(), child.exists())
            if child.is_dir() and child.name in ['raw', 'processed']:
                print('FILES', [(q.name, q.stat().st_size) for q in child.iterdir() if q.is_file()])

seen = set()
for top in [base, Path('/home/u202522802045/glue_156478'), Path('/home/u2021082/zpb')]:
    if not top.is_dir():
        continue
    for root, dirs, files in os.walk(top, followlinks=False):
        dirs[:] = [d for d in dirs if d not in ['.git', '__pycache__', 'node_modules', '.venv', 'venv', 'site-packages']]
        for name in files:
            path = Path(root)/name
            lower = str(path).lower()
            keep = ('gse19698' in lower or 'hda' in name.lower() or 'gencode.v32' in name.lower()
                    or (('pbmc-mo-002' in lower or '/d2/' in lower) and name in
                        ['rna_counts.npz', 'atac.npz', 'barcodes.npy', 'barcodes.tsv', 'gene_names.npy', 'peak_names.npy']))
            if keep and str(path) not in seen:
                seen.add(str(path))
                print('MATCH', path, path.stat().st_size if path.exists() else 'BROKEN_LINK', flush=True)

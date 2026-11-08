import csv
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

ROOT = Path('/home/u2021082/zpb/main/core/regulatory/pbmc_biology_20261027')
MOTIF = ROOT / 'motif/JASPAR2026_CORE_vertebrates.meme'
FASTA = Path('/tmp/zpb_fimo/hg38.fa')
FIMO = Path('/home/u2021082/.local/share/mamba/pkgs/https/conda.anaconda.org/bioconda/linux-64/meme-5.5.9-pl5321he99cc7f_1/bin/fimo')

def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out

def run_one(fold, kind):
    bed = ROOT / f'motif/{fold}_{kind}.bed'
    seq = ROOT / f'motif/{fold}_{kind}.fa'
    out = ROOT / f'motif/fimo_{fold}_{kind}'
    if not seq.exists():
        subprocess.run(['bedtools', 'getfasta', '-fi', str(FASTA), '-bed', str(bed), '-name', '-fo', str(seq)], check=True)
    if not (out / 'fimo.tsv').exists():
        out.mkdir(exist_ok=True)
        subprocess.run([str(FIMO), '--oc', str(out), '--thresh', '1e-4', '--max-stored-scores', '10000000', str(MOTIF), str(seq)], check=True)
    hits = set()
    with open(out / 'fimo.tsv', errors='replace') as fh:
        for row in csv.DictReader((x for x in fh if not x.startswith('#')), delimiter='\t'):
            if row.get('motif_id') and row.get('sequence_name'):
                sequence_id = row['sequence_name'].split('::', 1)[0]
                hits.add((row['motif_id'], sequence_id))
    return hits

def main():
    rows = []
    for fold in [f'target_D{i}' for i in range(1, 5)]:
        selected = run_one(fold, 'selected')
        candidate = run_one(fold, 'candidate')
        sel_names = {line.rstrip('\n').split('\t')[3] for line in (ROOT / f'motif/{fold}_selected.bed').open()}
        cand_names = {line.rstrip('\n').split('\t')[3] for line in (ROOT / f'motif/{fold}_candidate.bed').open()}
        non_names = cand_names - sel_names
        motifs = sorted({m for m, _ in candidate | selected})
        for motif in motifs:
            a = len({s for m, s in selected if m == motif})
            b = len(sel_names) - a
            c = len({s for m, s in candidate if m == motif} & non_names)
            d = len(non_names) - c
            _, p = fisher_exact([[a, b], [c, d]], alternative='greater')
            rows.append(dict(fold=fold, motif_id=motif, selected_peaks=len(sel_names), background_peaks=len(non_names), selected_hit_peaks=a, background_hit_peaks=c, odds_ratio=float((a*d)/(max(1,b*c))), p_value=float(p)))
    df = pd.DataFrame(rows)
    df['fdr_bh'] = np.nan
    for _, ix in df.groupby('fold').groups.items():
        df.loc[ix, 'fdr_bh'] = bh(df.loc[ix, 'p_value'])
    df.sort_values(['fold', 'fdr_bh', 'p_value']).to_csv(ROOT / 'motif/motif_enrichment_fimo.csv', index=False)
    df[df.fdr_bh <= .05].to_csv(ROOT / 'motif/motif_enrichment_fimo_fdr05.csv', index=False)
    print(df.groupby('fold').size().to_string())

if __name__ == '__main__':
    main()

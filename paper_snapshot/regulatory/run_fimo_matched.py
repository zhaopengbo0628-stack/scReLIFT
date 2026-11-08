import csv
import math
import re
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/u2021082/zpb/main/core/regulatory/pbmc_biology_20261027')
MOTIF = ROOT / 'motif/JASPAR2026_CORE_vertebrates.meme'
FASTA = Path('/tmp/zpb_fimo/hg38.fa')
FIMO = Path('/home/u2021082/.local/share/mamba/pkgs/https/conda.anaconda.org/bioconda/linux-64/meme-5.5.9-pl5321he99cc7f_1/bin/fimo')
SEED = 20020130
N_NULL = 200
N_CHUNKS = 8

def bh(p):
    p = np.asarray(p, float)
    out = np.full(len(p), np.nan)
    ok = np.isfinite(p)
    if not ok.any(): return out
    ii = np.flatnonzero(ok)
    order = ii[np.argsort(p[ii])]
    q = np.minimum.accumulate((p[order] * len(order) / np.arange(1, len(order)+1))[::-1])[::-1]
    out[order] = np.minimum(q, 1.0)
    return out

def bed_records(path):
    rows = []
    with open(path) as fh:
        for line in fh:
            c, s, e, name = line.rstrip('\n').split('\t')[:4]
            peak_index = int(name.rsplit('_', 1)[-1])
            rows.append((name, peak_index, c, int(s), int(e), int(e)-int(s)))
    return pd.DataFrame(rows, columns=['name','peak_index','chrom','start','end','length'])

def prepare_chunks():
    chunk_dir = ROOT / 'motif/fimo_chunks'; chunk_dir.mkdir(exist_ok=True)
    chunks = sorted(chunk_dir.glob('chunk_*.meme'))
    if len(chunks) == N_CHUNKS:
        return chunks
    text = MOTIF.read_text()
    header, body = text.split('MOTIF ', 1)
    motifs = ['MOTIF ' + x for x in body.split('\nMOTIF ') if x.strip()]
    for p in chunk_dir.glob('chunk_*.meme'): p.unlink()
    chunks = []
    for i in range(N_CHUNKS):
        part = motifs[i::N_CHUNKS]
        p = chunk_dir / f'chunk_{i:02d}.meme'
        p.write_text(header + '\n'.join(part) + '\n')
        chunks.append(p)
    return chunks

def run_scan(fold, chunks):
    bed = ROOT / f'motif/{fold}_candidate.bed'
    fa = ROOT / f'motif/{fold}_candidate.fa'
    if not fa.exists():
        subprocess.run(['bedtools','getfasta','-fi',str(FASTA),'-bed',str(bed),'-name','-fo',str(fa)], check=True)
    hits = set()
    def one(i_p):
        i, motif = i_p
        raw = ROOT / f'motif/fimo_{fold}_candidate_chunk{i:02d}.tsv'
        err = ROOT / f'motif/fimo_{fold}_candidate_chunk{i:02d}.stderr.log'
        if not raw.exists() or raw.stat().st_size < 1000:
            with open(raw, 'w') as out, open(err, 'w') as ef:
                subprocess.run([str(FIMO),'--text','--no-qvalue','--skip-matched-sequence','--thresh','1e-4','--max-stored-scores','10000000',str(motif),str(fa)], stdout=out, stderr=ef, check=True)
        local = set()
        with open(raw, errors='replace') as fh:
            for line in fh:
                if not line or line.startswith('#') or line.startswith('Using motif'):
                    continue
                parts = line.rstrip('\n').split('\t')
                if len(parts) >= 8 and parts[0] != 'motif_id' and re.match(r'^MA\d+', parts[0]):
                    seq_id = parts[2].split('::', 1)[0]
                    local.add((parts[0], int(seq_id.rsplit('_', 1)[-1])))
        return local
    with ThreadPoolExecutor(max_workers=N_CHUNKS) as ex:
        for local in ex.map(one, enumerate(chunks)):
            hits.update(local)
    return hits

def matched_samples(cand, selected, rng):
    bg = cand[~cand.peak_index.isin(set(selected.peak_index))].copy()
    cand_bins = pd.qcut(cand.length, q=min(5, max(1, cand.length.nunique())), labels=False, duplicates='drop')
    bg['len_bin'] = cand_bins.reindex(bg.index).fillna(0).astype(int)
    sel = selected.copy()
    bin_by_peak = dict(zip(cand.peak_index, cand_bins.astype(int)))
    sel['len_bin'] = sel.peak_index.map(bin_by_peak).fillna(0).astype(int)
    pools = {(c, b): g.peak_index.to_numpy() for (c, b), g in bg.groupby(['chrom','len_bin'])}
    pools_chr = {c: g.peak_index.to_numpy() for c, g in bg.groupby('chrom')}
    all_bg = bg.peak_index.to_numpy()
    samples = []
    for _ in range(N_NULL):
        chosen = []
        used = set()
        for (chrom, b), g in sel.groupby(['chrom','len_bin']):
            n = len(g)
            pool = np.asarray([x for x in pools.get((chrom,b), np.array([], dtype=object)) if x not in used])
            if len(pool) < n:
                pool = np.asarray([x for x in pools_chr.get(chrom, np.array([], dtype=object)) if x not in used])
            if len(pool) < n:
                pool = np.asarray([x for x in all_bg if x not in used])
            if len(pool) == 0:
                continue
            replace = len(pool) < n
            add = rng.choice(pool, size=n, replace=replace)
            chosen.extend(add.tolist()); used.update(add.tolist())
        samples.append(chosen)
    return samples, bg

def analyze_fold(fold, chunks):
    cand = bed_records(ROOT / f'motif/{fold}_candidate.bed')
    sel = bed_records(ROOT / f'motif/{fold}_selected.bed')
    hitset = run_scan(fold, chunks)
    by_motif = {}
    for motif, seq in hitset:
        by_motif.setdefault(motif, set()).add(seq)
    rng = np.random.RandomState(SEED)
    samples, bg = matched_samples(cand, sel, rng)
    rows = []
    motif_ids = re.findall(r'^MOTIF\s+(MA\S+)', MOTIF.read_text(), flags=re.M)
    for motif in sorted(set(motif_ids)):
        selected_ids = set(sel.peak_index)
        bg_ids = set(bg.peak_index)
        selected_hits = len(by_motif.get(motif, set()) & selected_ids)
        obs = selected_hits / max(1, len(sel))
        null_rates = np.array([len(by_motif.get(motif, set()) & set(s)) / max(1, len(s)) for s in samples], float)
        null_mean = float(np.mean(null_rates)) if len(null_rates) else np.nan
        # Fisher test against all non-selected candidate peaks is retained as a
        # secondary descriptive statistic; primary enrichment uses matched-null p.
        bg_hits = len(by_motif.get(motif, set()) & bg_ids)
        _, fisher_p = fisher_exact([[selected_hits, len(sel)-selected_hits], [bg_hits, len(bg)-bg_hits]], alternative='greater')
        rows.append(dict(fold=fold, motif_id=motif, selected_peaks=len(sel), background_peaks=len(bg), selected_hit_peaks=selected_hits, background_hit_peaks=bg_hits, selected_hit_rate=obs, background_hit_rate=bg_hits/max(1,len(bg)), matched_null_mean=null_mean, matched_null_sd=float(np.std(null_rates, ddof=1)) if len(null_rates)>1 else np.nan, fold_change=obs/null_mean if null_mean>0 else np.nan, enrichment_p=float((1+np.sum(null_rates >= obs))/(len(null_rates)+1)) if len(null_rates) else np.nan, fisher_p=float(fisher_p)))
    return pd.DataFrame(rows)

def main():
    out = ROOT / 'motif'; out.mkdir(exist_ok=True)
    chunks = prepare_chunks()
    all_df = []
    for fold in [f'target_D{i}' for i in range(1,5)]:
        df = analyze_fold(fold, chunks)
        all_df.append(df)
        df.to_csv(out / f'motif_enrichment_{fold}.csv', index=False)
    res = pd.concat(all_df, ignore_index=True)
    res['fdr_bh'] = np.nan
    for fold, ix in res.groupby('fold').groups.items():
        res.loc[ix, 'fdr_bh'] = bh(res.loc[ix, 'enrichment_p'].to_numpy())
    res = res.sort_values(['fold','fdr_bh','enrichment_p'])
    res.to_csv(out / 'motif_enrichment_fimo.csv', index=False)
    res[res.fdr_bh <= .05].to_csv(out / 'motif_enrichment_fimo_fdr05.csv', index=False)
    summary = res.groupby('fold').agg(motifs_tested=('motif_id','nunique'), significant_fdr05=('fdr_bh', lambda s: int((s<=.05).sum())), min_fdr=('fdr_bh','min')).reset_index()
    rep = res[res.fdr_bh <= .05].groupby('motif_id').agg(folds=('fold','nunique'), min_fdr=('fdr_bh','min'), mean_fold_change=('fold_change','mean')).reset_index().sort_values(['folds','min_fdr'], ascending=[False,True])
    summary.to_csv(out / 'motif_per_fold_summary.csv', index=False)
    rep.to_csv(out / 'motif_recurrent_summary.csv', index=False)
    top = res.sort_values(['fold','fdr_bh']).groupby('fold').head(15)
    fig, ax = plt.subplots(figsize=(10,6)); top2 = top.copy(); top2['label'] = top2['fold'].str.replace('target_','') + ':' + top2['motif_id']; top2 = top2.sort_values('fold_change')
    ax.barh(top2['label'], top2['fold_change'], color='#4472C4'); ax.axvline(1, color='black', lw=.8); ax.set_xlabel('Selected / matched-null hit-rate fold change'); ax.set_ylabel('Fold:motif'); fig.tight_layout(); fig.savefig(out/'motif_enrichment_dotplot.png', dpi=220); fig.savefig(out/'motif_enrichment_dotplot.svg'); plt.close(fig)
    print(summary.to_string(index=False))

if __name__ == '__main__': main()

"""Recompute motif enrichment from frozen B; retain sampling multiplicity."""
import argparse
import concurrent.futures
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import fisher_exact

SHARED = Path('/eaas/default/groups/public_cluster/home/share/zpb')
ARTIFACTS = SHARED / 'result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts'
WORK = Path('/home/u202522802045/share/zpb/regulatory/motif_null_corrected_20261101')
FIMO = Path('/home/u202522802045/share/zpb/regulatory/tools/meme-5.5.9/package/bin/fimo')
BEDTOOLS = Path('/home/u202522802045/envs/glue_156478/bin/bedtools')
SEEDS = [20020130, 20020131, 20020132]
N_NULL = 200
P_HIT = 1e-4
N_CHUNKS = 8
FOLDS = [f'target_D{i}' for i in range(1, 5)]

def log(s):
    print(time.strftime('%Y-%m-%d %H:%M:%S'), s, flush=True)

def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(b)
    return h.hexdigest()

def save_json(path, obj):
    tmp = Path(str(path) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)

def bh(p):
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * len(p) / np.arange(1, len(p) + 1))[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out

def bed_records(path):
    a = pd.read_csv(path, sep='\t', header=None, names=['chrom', 'start', 'end', 'name'])
    a['peak_index'] = a.name.str.rsplit('_', n=1).str[-1].astype(int)
    a['length'] = a.end - a.start
    return a

def motif_catalog(path):
    records = []
    for line in path.read_text().splitlines():
        if line.startswith('MOTIF '):
            a = line.split()
            records.append((a[1], ' '.join(a[2:]) or a[1]))
    assert len(records) == 1019 and len(set(x[0] for x in records)) == 1019
    return records

def prepare():
    for d in ['inputs', 'scan', 'results', 'diagnostics', 'figures']:
        (WORK / d).mkdir(parents=True, exist_ok=True)
    motif = WORK / 'inputs/JASPAR2026_CORE_vertebrates.meme'
    assert motif.exists(), 'Upload existing merged JASPAR file; do not redownload.'
    catalog = motif_catalog(motif)
    union = {}
    audit = []
    for fold in FOLDS:
        p = ARTIFACTS / fold
        b = sparse.load_npz(p / 'B_D.npz').tocsr()
        b.eliminate_zeros()
        names = np.load(p / 'harmonized/canonical_peak_names.npy', allow_pickle=False).astype(str)
        ep = np.load(p / 'edge_peak.npy', allow_pickle=False)
        sel = np.unique(b.nonzero()[0])
        cand = np.unique(ep)
        assert set(sel).issubset(set(cand))
        assert b.shape[0] == len(names)
        for kind, ids in [('candidate', cand), ('selected', sel)]:
            rows = []
            for idx in ids:
                m = re.fullmatch(r'(chr[^:]+):(\d+)-(\d+)', names[idx])
                assert m, (fold, names[idx])
                chrom, start, end = m.group(1), int(m.group(2)), int(m.group(3))
                assert 0 <= start < end
                coord = (chrom, start, end)
                if coord not in union:
                    union[coord] = len(union)
                rows.append([chrom, start, end, f'{fold}_{kind}_{idx}'])
            pd.DataFrame(rows).to_csv(WORK / f'inputs/{fold}_{kind}.bed', sep='\t', header=False, index=False)
        audit.append(dict(fold=fold, candidate_peaks=len(cand), selected_peaks=len(sel), background_peaks=len(cand) - len(sel), selected_effective_edges=b.nnz, b_sha256=sha(p/'B_D.npz'), peak_names_sha256=sha(p/'harmonized/canonical_peak_names.npy')))
    pd.DataFrame(audit).to_csv(WORK / 'diagnostics/frozen_input_audit.csv', index=False)
    union_rows = [[*coord, f'union_{uid}'] for coord, uid in union.items()]
    pd.DataFrame(union_rows).to_csv(WORK / 'inputs/candidate_union.bed', sep='\t', header=False, index=False)
    pd.DataFrame(union_rows, columns=['chrom', 'start', 'end', 'name']).to_csv(WORK / 'inputs/union_coordinates.csv', index=False)
    log(f'Frozen inputs prepared: {len(union)} distinct exact-coordinate intervals; no interval merging')
    fa = WORK / 'inputs/hg38.fa'
    src = SHARED / 'result_contain/pbmc_dmode_regulatory_biology_20261026/inputs/hg38.fa.gz'
    if not fa.exists():
        log('Validating/decompressing existing hg38 gzip')
        tmp = fa.with_suffix('.fa.tmp')
        with gzip.open(src, 'rb') as a, open(tmp, 'wb') as out:
            shutil.copyfileobj(a, out, 8 * 1024 * 1024)
        tmp.replace(fa)
    fasta = WORK / 'inputs/candidate_union.fa'
    if not fasta.exists():
        log('Extracting hg38 sequences with BED 0-based half-open coordinates')
        tmp = Path(str(fasta) + '.tmp')
        with open(WORK / 'diagnostics/bedtools.stderr.log', 'w') as err:
            subprocess.run([str(BEDTOOLS), 'getfasta', '-fi', str(fa), '-bed', str(WORK/'inputs/candidate_union.bed'), '-nameOnly', '-fo', str(tmp)], stderr=err, check=True)
        tmp.replace(fasta)
    # Validate every input interval, chromosome bound, sequence length and name.
    fai = {}
    for line in Path(str(fa) + '.fai').read_text().splitlines():
        a = line.split('\t'); fai[a[0]] = int(a[1])
    for c, s, e in union:
        assert c in fai and e <= fai[c], (c, s, e)
    lengths = {}; seqname = None
    with open(fasta) as f:
        for line in f:
            if line.startswith('>'):
                seqname = line[1:].strip().split()[0]
                assert seqname not in lengths
                lengths[seqname] = 0
            else:
                assert seqname is not None
                lengths[seqname] += len(line.strip())
    assert len(lengths) == len(union)
    for coord, uid in union.items():
        assert lengths[f'union_{uid}'] == coord[2] - coord[1]
    txt = motif.read_text()
    header, rest = txt.split('MOTIF ', 1)
    blocks = ['MOTIF ' + x for x in rest.split('\nMOTIF ') if x.strip()]
    assert len(blocks) == len(catalog)
    for i in range(N_CHUNKS):
        (WORK/f'inputs/chunk_{i:02d}.meme').write_text(header + '\n'.join(blocks[i::N_CHUNKS]) + '\n')
    save_json(WORK/'diagnostics/preparation.json', dict(union_intervals=len(union), validated_sequences=len(lengths), hg38_gzip=str(src), hg38_gzip_sha256=sha(src), motif_sha256=sha(motif), fimo_version=subprocess.check_output([str(FIMO),'--version'],text=True).strip(), seeds=SEEDS, null_draws_per_seed=N_NULL, motif_hit_p=P_HIT, union_exact_dedup_only=True))
    log('Preparation and sequence coverage audit complete')

def scan():
    fasta = WORK/'inputs/candidate_union.fa'
    fa_sha = sha(fasta)
    def one(i):
        m = WORK/f'inputs/chunk_{i:02d}.meme'
        out = WORK/f'scan/chunk_{i:02d}.tsv'
        done = WORK/f'scan/chunk_{i:02d}.done.json'
        fingerprint = dict(fasta_sha256=fa_sha, motif_sha256=sha(m), threshold=P_HIT, fimo_version='5.5.9')
        if done.exists() and out.exists() and json.loads(done.read_text()) == fingerprint:
            return str(out)
        tmp = Path(str(out)+'.tmp')
        log(f'FIMO chunk {i} starting')
        with open(tmp,'w') as f, open(WORK/f'scan/chunk_{i:02d}.stderr.log','w') as err:
            subprocess.run([str(FIMO),'--text','--no-qvalue','--skip-matched-sequence','--thresh',str(P_HIT),str(m),str(fasta)], stdout=f,stderr=err,check=True)
        tmp.replace(out)
        save_json(done,fingerprint)
        log(f'FIMO chunk {i} complete')
        return str(out)
    with concurrent.futures.ThreadPoolExecutor(max_workers=N_CHUNKS) as pool:
        list(pool.map(one,range(N_CHUNKS)))
    log('All 8 FIMO chunks complete')

def matched_samples(cand, selected, rng):
    # Preserve the original sampler exactly; only statistics count repeats correctly.
    bg = cand[~cand.peak_index.isin(set(selected.peak_index))].copy()
    cb = pd.qcut(cand.length,q=min(5,max(1,cand.length.nunique())),labels=False,duplicates='drop').fillna(0).astype(int)
    bg['len_bin'] = cb.reindex(bg.index).fillna(0).astype(int)
    sel = selected.copy()
    sel['len_bin'] = sel.peak_index.map(dict(zip(cand.peak_index,cb))).astype(int)
    assert len(bg)>0
    pools={(c,b):g.peak_index.to_numpy() for (c,b),g in bg.groupby(['chrom','len_bin'])}
    by_chr={c:g.peak_index.to_numpy() for c,g in bg.groupby('chrom')}
    all_bg=bg.peak_index.to_numpy()
    attrs=bg.set_index('peak_index')
    samples=[]; diagnostics=[]
    for draw in range(N_NULL):
        chosen=[];used=set(); levels=[0,0,0];missing=0;matched_chr=0;matched_bin=0
        for (chrom,b),g in sel.groupby(['chrom','len_bin']):
            n=len(g);level=0
            pool=np.asarray([x for x in pools.get((chrom,b),[]) if x not in used])
            if len(pool)<n:
                level=1;pool=np.asarray([x for x in by_chr.get(chrom,[]) if x not in used])
            if len(pool)<n:
                level=2;pool=np.asarray([x for x in all_bg if x not in used])
            if len(pool)==0:
                missing+=n;continue
            add=rng.choice(pool,size=n,replace=len(pool)<n)
            chosen.extend(add.tolist());used.update(add.tolist());levels[level]+=n
            aa=attrs.loc[add]
            matched_chr+=int((aa.chrom==chrom).sum())
            matched_bin+=int(((aa.chrom==chrom)&(aa.len_bin==b)).sum())
        assert chosen and not (set(chosen)&set(sel.peak_index))
        samples.append(chosen)
        diagnostics.append(dict(draw=draw,requested_peaks=len(sel),sampled_peaks=len(chosen),unique_peaks=len(set(chosen)),duplicate_fraction=1-len(set(chosen))/len(chosen),missing_peaks=missing,exact_pool_draws=levels[0],chrom_relaxed_draws=levels[1],global_relaxed_draws=levels[2],actual_chrom_match_fraction=matched_chr/len(chosen),actual_chrom_lengthbin_match_fraction=matched_bin/len(chosen),mean_peak_length=float(attrs.loc[chosen,'length'].mean()),selected_mean_peak_length=float(sel.length.mean())))
    return samples,bg,pd.DataFrame(diagnostics)

def duplicate_regression_check():
    sample=[2,2,3]
    positive={2}
    old=len(set(sample)&positive)/len(sample)
    corrected=sum(x in positive for x in sample)/len(sample)
    assert old==1/3 and corrected==2/3
    return dict(sample=sample,positive_peaks=[2],old_rate=old,corrected_rate=corrected,passed=True)

def statistics():
    catalog=motif_catalog(WORK/'inputs/JASPAR2026_CORE_vertebrates.meme')
    mids=[x[0] for x in catalog];mnames=dict(catalog);mi={x:i for i,x in enumerate(mids)}
    union=pd.read_csv(WORK/'inputs/union_coordinates.csv')
    hits=np.zeros((len(union),len(mids)),dtype=bool)
    scan_audit=[]
    for chunk in range(N_CHUNKS):
        expected={x[0] for x in catalog[chunk::N_CHUNKS]}
        log(f'Reading FIMO hits chunk {chunk}')
        seen=set();count=0
        with open(WORK/f'scan/chunk_{chunk:02d}.tsv') as f:
            for line in f:
                if line.startswith('#') or line.startswith('motif_id') or not line.strip():continue
                a=line.rstrip('\n').split('\t')
                if a[0] not in expected:raise ValueError(a[:3])
                uid=int(a[2].split('::',1)[0].rsplit('_',1)[-1])
                assert 0<=uid<len(union) and float(a[7])<=P_HIT*(1+1e-6)
                hits[uid,mi[a[0]]]=True;seen.add(a[0]);count+=1
        scan_audit.append(dict(chunk=chunk,motifs_expected=len(expected),motifs_with_hits=len(seen),hit_rows=count,tsv_sha256=sha(WORK/f'scan/chunk_{chunk:02d}.tsv')))
    pd.DataFrame(scan_audit).to_csv(WORK/'diagnostics/scan_audit.csv',index=False)
    sparse.save_npz(WORK/'results/motif_peak_presence_union.npz',sparse.csr_matrix(hits))
    save_json(WORK/'diagnostics/duplicate_regression_check.json',duplicate_regression_check())
    coord_uid={(r.chrom,int(r.start),int(r.end)):i for i,r in union.iterrows()}
    all_rows=[];all_diag=[]
    for fold in FOLDS:
        cand=bed_records(WORK/f'inputs/{fold}_candidate.bed')
        sel=bed_records(WORK/f'inputs/{fold}_selected.bed')
        assert cand.peak_index.is_unique and sel.peak_index.is_unique
        coords=[(r.chrom,int(r.start),int(r.end)) for r in cand.itertuples()]
        fold_hits=hits[[coord_uid[c] for c in coords]]
        row_for_peak=dict(zip(cand.peak_index,range(len(cand))))
        srows=[row_for_peak[i] for i in sel.peak_index]
        observed=fold_hits[srows].mean(axis=0);selected_counts=fold_hits[srows].sum(axis=0)
        for seed in SEEDS:
            log(f'Null statistics {fold} seed {seed}')
            samples,bg,diag=matched_samples(cand,sel,np.random.RandomState(seed))
            diag.insert(0,'seed',seed);diag.insert(0,'fold',fold);all_diag.append(diag)
            # Multiplicity matrix: a Peak drawn k times contributes k hits and k trials.
            weights=np.zeros((N_NULL,len(cand)),dtype=np.int32)
            for j,sample in enumerate(samples):
                idx=[row_for_peak[i] for i in sample]
                weights[j]=np.bincount(idx,minlength=len(cand))
            den=weights.sum(axis=1)
            assert np.array_equal(den,diag.sampled_peaks.to_numpy())
            assert np.all(den>0)
            null=(weights.astype(float)@fold_hits.astype(float))/den[:,None]
            # Direct independent check: repeated indexing must match weighted counting.
            for j in [0,99,199]:
                idx=[row_for_peak[i] for i in samples[j]]
                assert np.allclose(null[j],fold_hits[idx].mean(axis=0),rtol=0,atol=1e-14)
            oldweights=(weights>0).astype(float)
            oldnull=(oldweights@fold_hits.astype(float))/den[:,None]
            nm=null.mean(axis=0);sd=null.std(axis=0,ddof=1)
            p=(1+(null>=observed[None,:]).sum(axis=0))/(N_NULL+1)
            q=bh(p)
            bidx=[row_for_peak[i] for i in bg.peak_index]
            bc=fold_hits[bidx].sum(axis=0)
            for k,m in enumerate(mids):
                fp=fisher_exact([[int(selected_counts[k]),len(sel)-int(selected_counts[k])],[int(bc[k]),len(bg)-int(bc[k])]],alternative='greater').pvalue
                all_rows.append(dict(fold=fold,seed=seed,motif_id=m,tf_name=mnames[m],selected_peaks=len(sel),background_peaks=len(bg),selected_hit_peaks=int(selected_counts[k]),background_hit_peaks=int(bc[k]),selected_hit_rate=float(observed[k]),background_hit_rate=float(bc[k]/len(bg)),matched_null_mean=float(nm[k]),matched_null_sd=float(sd[k]),fold_change=float(observed[k]/nm[k]) if nm[k]>0 else np.nan,enrichment_p=float(p[k]),fdr_bh=float(q[k]),fisher_p=float(fp),legacy_bug_null_mean=float(oldnull[:,k].mean()),legacy_bug_enrichment_p=float((1+(oldnull[:,k]>=observed[k]).sum())/(N_NULL+1))))
    df=pd.DataFrame(all_rows)
    df.to_csv(WORK/'results/motif_enrichment_by_seed.csv',index=False)
    pd.concat(all_diag,ignore_index=True).to_csv(WORK/'diagnostics/null_sampling_diagnostics.csv',index=False)
    summary=df.groupby(['fold','seed'],as_index=False).agg(motifs_tested=('motif_id','nunique'),significant_fdr05=('fdr_bh',lambda x:int((x<.05).sum())),min_fdr=('fdr_bh','min'),median_fold_change=('fold_change','median'))
    summary.to_csv(WORK/'results/motif_per_fold_per_seed_summary.csv',index=False)
    fs=summary.groupby('fold',as_index=False).agg(motifs_tested=('motifs_tested','first'),significant_fdr05_mean=('significant_fdr05','mean'),significant_fdr05_sd=('significant_fdr05',lambda x:x.std(ddof=1)),min_fdr_mean=('min_fdr','mean'))
    fs.to_csv(WORK/'results/motif_per_fold_summary.csv',index=False)
    aggregate=df.groupby(['fold','motif_id','tf_name'],as_index=False).agg(selected_peaks=('selected_peaks','first'),background_peaks=('background_peaks','first'),selected_hit_peaks=('selected_hit_peaks','first'),background_hit_peaks=('background_hit_peaks','first'),selected_hit_rate=('selected_hit_rate','first'),background_hit_rate=('background_hit_rate','first'),matched_null_mean=('matched_null_mean','mean'),matched_null_mean_seed_sd=('matched_null_mean',lambda x:x.std(ddof=1)),fold_change=('fold_change','mean'),fold_change_seed_sd=('fold_change',lambda x:x.std(ddof=1)),enrichment_p_min=('enrichment_p','min'),enrichment_p_max=('enrichment_p','max'),fdr_bh_max=('fdr_bh','max'),significant_seed_count=('fdr_bh',lambda x:int((x<.05).sum())))
    aggregate['robust_significant']=aggregate.significant_seed_count==len(SEEDS)
    aggregate.sort_values(['fold','fdr_bh_max','fold_change'],ascending=[True,True,False]).to_csv(WORK/'results/motif_enrichment_fimo.csv',index=False)
    sig=aggregate[aggregate.robust_significant]
    sig.to_csv(WORK/'results/motif_enrichment_fimo_fdr05.csv',index=False)
    recurrent=sig.groupby(['motif_id','tf_name'],as_index=False).agg(folds=('fold','nunique'),mean_fold_change=('fold_change','mean'),max_fdr=('fdr_bh_max','max')).sort_values(['folds','mean_fold_change'],ascending=[False,False])
    recurrent.to_csv(WORK/'results/motif_recurrent_summary.csv',index=False)
    log('RESULTS '+fs.to_json(orient='records'))
    make_figure(aggregate)
    save_json(WORK/'completion.json',dict(status='completed',analysis='duplicate-aware correction preserving original sampler',seeds=SEEDS,null_draws_per_seed=N_NULL,rows_by_seed=len(df),fold_motif_rows=len(aggregate),robust_significant_per_fold={f:int((sig.fold==f).sum()) for f in FOLDS},regression_check_passed=True,source_code_sha256=sha(__file__)))

def make_figure(df):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    # Rank by repeatability across seeds and folds, then effect; no threshold tuning.
    ranked=df.groupby(['motif_id','tf_name']).agg(folds=('robust_significant','sum'),effect=('fold_change','mean')).sort_values(['folds','effect'],ascending=False).head(20)
    ids=[x[0] for x in ranked.index];names=[x[1] for x in ranked.index]
    fig,ax=plt.subplots(figsize=(8,9))
    xs=[];ys=[];cs=[];ss=[]
    for y,m in enumerate(ids):
        for x,f in enumerate(FOLDS):
            r=df[(df.motif_id==m)&(df.fold==f)].iloc[0]
            xs.append(x);ys.append(y);cs.append(np.log2(r.fold_change) if r.fold_change>0 else 0);ss.append(90 if r.robust_significant else 22)
    c=ax.scatter(xs,ys,c=cs,s=ss,cmap='coolwarm',edgecolors='black',linewidths=.3)
    ax.set_xticks(range(4),['D1','D2','D3','D4']);ax.set_yticks(range(len(ids)),[f'{n} ({m})' for n,m in zip(names,ids)]);ax.invert_yaxis()
    ax.set_title('Corrected motif matched null; large dots: FDR < 0.05 in all 3 seeds')
    fig.colorbar(c,ax=ax,label='log2(mean fold change)');fig.tight_layout()
    for ext in ['png','svg']:fig.savefig(WORK/f'figures/motif_enrichment_dotplot.{ext}',dpi=220)
    plt.close(fig)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['prepare','scan','statistics','all'],default='all');args=parser.parse_args()
    if args.stage in ['prepare','all']:prepare()
    if args.stage in ['scan','all']:scan()
    if args.stage in ['statistics','all']:statistics()

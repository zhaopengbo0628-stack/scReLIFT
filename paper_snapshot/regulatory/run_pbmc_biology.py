"""Read-only PBMC regulatory biology analyses on frozen Task99 artifacts.

Outputs are written to a user-owned directory. Shared inputs are never changed.
"""
import json, gzip, hashlib, re, csv, os
from pathlib import Path
from collections import defaultdict
import numpy as np
import pandas as pd
from scipy import sparse, stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path("/eaas/default/groups/public_cluster/home/share/zpb")
BASE = ROOT / "result_contain/99_pbmc_lodo_dmode_20261024/full_b2/validated_source_artifacts"
PREP = ROOT / "data/staging/PBMC/legacy_dmode_data_20261004/derived/common"
LEGACY = ROOT / "data/cache/d_mode/PBMC/legacy_full_b2_corrected"
REF = ROOT / "data/reference/external_regulatory/ENCODE_SCREEN_V4/hg38"
BIO = ROOT / "result_contain/pbmc_dmode_regulatory_biology_20261026"
OUT = Path("/home/u2021082/zpb/main/core/regulatory/pbmc_biology_20261027")
FOLDS = ["target_D1", "target_D2", "target_D3", "target_D4"]
DS = {f: f.replace("target_", "") for f in FOLDS}
SEED = 20020130

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for b in iter(lambda: fh.read(8 << 20), b""):
            h.update(b)
    return h.hexdigest()

def parse_peak(x):
    m = re.match(r"^(chr[^:]+):(\d+)-(\d+)$", str(x))
    if not m:
        raise ValueError("bad peak name: " + str(x))
    return m.group(1), int(m.group(2)), int(m.group(3))

def load_bed(p):
    d = pd.read_csv(p, sep="\t", header=None, usecols=[0, 1, 2], names=["chr", "start", "end"], comment="#")
    out = {}
    for c, g in d.groupby("chr"):
        g = g.sort_values("start")
        starts = g.start.to_numpy(np.int64)
        ends = g.end.to_numpy(np.int64)
        out[c] = (starts, ends, np.maximum.accumulate(ends))
    return out

def overlap_mask(peaks, ref):
    hit = np.zeros(len(peaks), dtype=bool)
    for chrom, (starts, ends, prefix_max_end) in ref.items():
        idx = np.flatnonzero(peaks.chr.to_numpy() == chrom)
        if len(idx) == 0:
            continue
        qstart = peaks.start.to_numpy(np.int64)[idx]
        qend = peaks.end.to_numpy(np.int64)[idx]
        j = np.searchsorted(starts, qend, side="left") - 1
        ok = j >= 0
        ok[ok] &= prefix_max_end[j[ok]] > qstart[ok]
        hit[idx] = ok
    return hit

def bh(p):
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if not ok.any(): return out
    order = np.argsort(p[ok]); vals = p[ok][order]; n = len(vals)
    q = np.minimum.accumulate((vals * n / np.arange(1, n + 1))[::-1])[::-1]
    tmp = np.empty(n); tmp[order] = np.minimum(q, 1.0); out[ok] = tmp
    return out

def selected_edges(d):
    b = np.load(d / "B_D.npz")
    mat = sparse.csr_matrix((b["data"], b["indices"], b["indptr"]), shape=tuple(b["shape"]))
    coo = mat.tocoo()
    return {(int(r), int(c)): float(v) for r, c, v in zip(coo.row, coo.col, coo.data) if v != 0}

def load_fold(fold):
    d = BASE / fold
    m = json.loads((d / "manifest.json").read_text())
    genes = np.load(d / "gene_ids.npy", allow_pickle=True).astype(str)
    names = np.load(d / "harmonized/canonical_peak_names.npy", allow_pickle=True).astype(str)
    cg = np.load(d / "candidate_graph.npz")
    ep = np.load(d / "edge_peak.npy")
    eg = np.load(d / "edge_gene.npy")
    sel = selected_edges(d)
    rows = []
    for p, g in zip(ep, eg):
        chrom, start, end = parse_peak(names[int(p)])
        rows.append((fold, int(p), int(g), genes[int(g)], chrom, start, end, int(end - start), int((int(p), int(g)) in sel), sel.get((int(p), int(g)), 0.0)))
    ed = pd.DataFrame(rows, columns=["fold", "peak_index", "gene_index", "gene_id", "chr", "start", "end", "length", "selected", "weight"])
    ed["sign"] = np.sign(ed.weight).astype(int)
    # Candidate graph distance is aligned to candidate_index[gene, rank]; map by (gene, peak).
    dist = {}
    cidx, cmask, cdist = cg["candidate_index"], cg["candidate_mask"], cg["distance_bp"]
    for g in range(cidx.shape[0]):
        for r in range(cidx.shape[1]):
            if bool(cmask[g, r]): dist[(int(cidx[g, r]), g)] = float(cdist[g, r])
    ed["distance_bp"] = [dist.get((int(p), int(g)), np.nan) for p, g in zip(ed.peak_index, ed.gene_index)]
    return {"fold": fold, "manifest": m, "genes": genes, "names": names, "edges": ed, "candidates": cg}

def main():
    for sub in ["audit", "stability", "enhancer", "pathway", "marker", "motif", "gene_links"]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)
    folds = {f: load_fold(f) for f in FOLDS}
    print("STAGE_LOAD_FOLDS", flush=True)

    # 1. Benchmark alignment: prepared RNA/ATAC, IDs, peaks, barcodes, labels.
    audit = []
    label_root = ROOT / "main/result/s_mode/intra/62_smode_full_b2_pbmc_scoring_timing_5seed_20261014/audits"
    for f, x in folds.items():
        ds = DS[f]; prep = PREP / ds
        genes = pd.read_csv(prep / "genes.tsv", sep="\t")
        bars = pd.read_csv(prep / "barcodes.tsv", header=None, names=["barcode"])
        bed = pd.read_csv(prep / "peaks.bed", sep="\t", header=None, usecols=[0, 1, 2], names=["chr", "start", "end"])
        canon = pd.DataFrame([parse_peak(v) for v in x["names"]], columns=["chr", "start", "end"])
        # Any interval overlap with prepared target peaks is an alignment check, not a replacement.
        bychr = {}
        for c, g in bed.groupby("chr"):
            g = g.sort_values("start"); st = g.start.to_numpy(); en = g.end.to_numpy(); bychr[c] = (st, en, np.maximum.accumulate(en))
        hits = overlap_mask(canon, bychr)
        lpath = label_root / (ds + "_label_harmonized.csv")
        exact = normalized = 0
        if lpath.exists():
            labs = pd.read_csv(lpath)
            exact = len(set(bars.barcode) & set(labs.barcode))
            normalized = len(set(bars.barcode) & set(labs.barcode.str.replace(r"^[^_]+_", "", regex=True)))
        audit.append(dict(fold=f, dataset=ds, rna_counts=str(prep / "rna_counts.npz"), atac_counts=str(prep / "atac_counts.npz"), rna_cells=len(bars), rna_genes=genes.shape[0], atac_peaks=bed.shape[0], task99_genes=len(x["genes"]), task99_canonical_peaks=len(x["names"]), canonical_peak_interval_overlap_fraction=float(hits.mean()), labels_file=str(lpath) if lpath.exists() else "", label_exact_barcode_overlap=exact, label_prefix_stripped_overlap=normalized, sources=",".join(x["manifest"]["sources"])))
    pd.DataFrame(audit).to_csv(OUT / "audit/benchmark_alignment.csv", index=False)
    print("STAGE_AUDIT", flush=True)

    # 2. Interval-overlap candidate-space stability. Connected components are built only for analysis.
    allrows = pd.concat([x["edges"] for x in folds.values()], ignore_index=True)
    common_genes = set.intersection(*[set(x["genes"]) for x in folds.values()])
    q = allrows[allrows.gene_id.isin(common_genes)].reset_index(drop=True)
    parent = np.arange(len(q), dtype=np.int64)
    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]; a = parent[a]
        return a
    def union(a, b):
        a, b = find(a), find(b)
        if a != b: parent[b] = a
    for _, idx in q.groupby(["gene_id", "chr"]).groups.items():
        ii = np.asarray(list(idx), int); ii = ii[np.argsort(q.loc[ii, "start"].to_numpy())]
        rep = None; frontier = -1
        for i in ii:
            st = int(q.loc[i, "start"]); en = int(q.loc[i, "end"])
            if rep is None or st >= frontier:
                rep, frontier = int(i), en
            else:
                union(int(i), rep)
                frontier = max(frontier, en)
    roots = [find(i) for i in range(len(q))]
    comp_map = {r: i for i, r in enumerate(sorted(set(roots)))}
    q["component"] = [comp_map[r] for r in roots]
    q.to_csv(OUT / "stability/common_candidate_edges.csv", index=False)
    summ = []
    for f in FOLDS:
        z = q[q.fold == f]
        summ.append(dict(fold=f, common_genes=len(common_genes), candidate_edges=len(z), candidate_components=z.component.nunique(), selected_edges=int(z.selected.sum()), selected_components=z.loc[z.selected.astype(bool), "component"].nunique()))
    pd.DataFrame(summ).to_csv(OUT / "stability/common_candidate_fold_summary.csv", index=False)
    pair = []
    exact_pair = []
    exact_sets = {}
    for f, x in folds.items():
        z = x["edges"]
        exact_sets[f] = set(zip(z.loc[z.selected.astype(bool), "gene_id"], z.loc[z.selected.astype(bool), "chr"], z.loc[z.selected.astype(bool), "start"], z.loc[z.selected.astype(bool), "end"]))
    for i, a in enumerate(FOLDS):
        for b in FOLDS[i + 1:]:
            A, B = q[q.fold == a], q[q.fold == b]
            Ac, Bc = set(A.loc[A.selected.astype(bool), "component"]), set(B.loc[B.selected.astype(bool), "component"])
            U, I = Ac | Bc, Ac & Bc
            wa = A[A.selected.astype(bool)].groupby("component").weight.mean(); wb = B[B.selected.astype(bool)].groupby("component").weight.mean(); common = sorted(I)
            sa = A[A.selected.astype(bool)].groupby("component").sign.mean(); sb = B[B.selected.astype(bool)].groupby("component").sign.mean()
            pair.append(dict(fold_a=a, fold_b=b, common_candidate_components=len(set(A.component)&set(B.component)), selected_a=len(Ac), selected_b=len(Bc), matched_selected=len(I), interval_jaccard=len(I)/len(U) if U else np.nan, sign_consistency=float(np.mean(np.sign(sa.loc[common]) == np.sign(sb.loc[common]))) if common else np.nan, weight_spearman=float(stats.spearmanr(wa.loc[common], wb.loc[common])[0]) if len(common)>2 else np.nan))
            exact_inter = exact_sets[a] & exact_sets[b]; exact_union = exact_sets[a] | exact_sets[b]
            exact_pair.append(dict(fold_a=a, fold_b=b, selected_edges_a=len(exact_sets[a]), selected_edges_b=len(exact_sets[b]), exact_matched_edges=len(exact_inter), exact_jaccard=len(exact_inter)/len(exact_union) if exact_union else np.nan))
    pd.DataFrame(pair).to_csv(OUT / "stability/interval_candidate_pairwise.csv", index=False)
    pd.DataFrame(exact_pair).to_csv(OUT / "stability/exact_coordinate_pairwise.csv", index=False)
    print("STAGE_STABILITY", flush=True)

    # 3. SCREEN V4 overlap with matched peak null. No edge-level null is used.
    refs = {k: load_bed(REF / fn) for k, fn in [("PLS", "GRCh38-cCREs.PLS.bed"), ("pELS", "GRCh38-cCREs.pELS.bed"), ("dELS", "GRCh38-cCREs.dELS.bed")]}
    print("STAGE_LOAD_ENCODE", flush=True)
    enhancer_rows = []; rng = np.random.RandomState(SEED)
    for f, x in folds.items():
        z = x["edges"]; cand = z.drop_duplicates(["peak_index"])[["peak_index", "chr", "start", "end", "length"]].copy()
        selp = z.loc[z.selected.astype(bool)].drop_duplicates(["peak_index"])["peak_index"].to_numpy(int)
        # Peak-level source availability from frozen harmonization masks.
        avail = np.full(len(x["names"]), np.nan)
        for s in x["manifest"]["sources"]:
            p = LEGACY / f / "source_artifacts/harmonized/PBMC-MO-{int(s[1:]):03d}/canonical_peak_available.npy"
            if not p.exists(): continue
            a = np.load(p).astype(float); avail[:len(a)] = np.nan_to_num(avail[:len(a)], nan=0.0) + a
        if np.isfinite(avail).any():
            nsrc = len(x["manifest"]["sources"]); avail = avail / nsrc
        else: avail = np.zeros(len(x["names"]), dtype=float)
        cand["avail"] = [avail[int(p)] if int(p) < len(avail) else 0.0 for p in cand.peak_index]
        dmin = z.groupby("peak_index").distance_bp.min(); cand["distance_bp"] = [float(dmin.get(int(p), np.nan)) for p in cand.peak_index]
        for col in ["length", "distance_bp", "avail"]:
            vals = cand[col].replace([np.inf, -np.inf], np.nan).fillna(0).to_numpy(float)
            cand[col + "_bin"] = pd.qcut(vals, q=min(5, max(1, len(np.unique(vals)))), labels=False, duplicates="drop") if len(np.unique(vals)) > 1 else 0
        sel = cand[cand.peak_index.isin(selp)].reset_index(drop=True)
        cand["stratum"] = list(zip(cand.chr, cand.length_bin, cand.distance_bp_bin, cand.avail_bin))
        sel["stratum"] = list(zip(sel.chr, sel.length_bin, sel.distance_bp_bin, sel.avail_bin))
        excluded = set(int(p) for p in selp)
        eligible = cand[~cand.peak_index.isin(excluded)]
        pool_exact = {k: g.peak_index.to_numpy(int) for k, g in eligible.groupby("stratum")}
        pool_chr_len = {k: g.peak_index.to_numpy(int) for k, g in eligible.groupby(["chr", "length_bin"])}
        pool_chr = {k: g.peak_index.to_numpy(int) for k, g in eligible.groupby("chr")}
        pool_all = eligible.peak_index.to_numpy(int)
        pool_all_with_selected = cand.peak_index.to_numpy(int)
        group_counts = sel.stratum.value_counts().to_dict()
        match_relax = defaultdict(int)
        def matched_sample():
            out = []
            for key, count in group_counts.items():
                chrom, lenbin, _, _ = key
                pools = [pool_exact.get(key, np.array([], int)),
                         pool_chr_len.get((chrom, lenbin), np.array([], int)),
                         pool_chr.get(chrom, np.array([], int)), pool_all]
                chosen = []
                remaining = int(count)
                for level, pool in enumerate(pools):
                    if remaining <= 0:
                        break
                    if chosen:
                        pool = pool[~np.isin(pool, np.asarray(chosen, dtype=int))]
                    if len(pool) == 0:
                        continue
                    # Every background pool excludes selected peaks. Sampling with
                    # replacement is only used to fill a sparse matched stratum.
                    take = min(remaining, len(pool))
                    add = rng.choice(pool, size=take, replace=False)
                    chosen.extend(add.tolist())
                    remaining -= take
                    match_relax[level] += take
                    if remaining and level == len(pools) - 1:
                        add = rng.choice(pool, size=remaining, replace=True)
                        chosen.extend(add.tolist())
                        match_relax[4] += remaining
                        remaining = 0
                if remaining:
                    # A fold with no nonselected candidate peaks cannot support a
                    # matched null; preserve the failure instead of leaking signal.
                    match_relax[5] += remaining
                out.extend(chosen)
            return np.asarray(out, int)
        for label, ref in refs.items():
            peakdf = cand.copy(); hit = overlap_mask(peakdf, ref); hmap = dict(zip(peakdf.peak_index, hit))
            obs = float(np.mean([hmap.get(int(p), False) for p in selp])) if len(selp) else np.nan
            null = []; null_n = []
            for _ in range(200):
                pp = matched_sample(); null_n.append(len(pp)); null.append(float(np.mean([hmap.get(int(p), False) for p in pp])) if len(pp) else np.nan)
            null = np.asarray(null, float)
            enhancer_rows.append(dict(fold=f, cCRE=label, selected_peaks=len(selp), candidate_peaks=len(cand), observed_overlap=obs, matched_null_mean=float(np.nanmean(null)), matched_null_sd=float(np.nanstd(null, ddof=1)), empirical_p=float((1 + np.sum(null >= obs)) / (len(null) + 1)), fold_change=float(obs / np.nanmean(null)) if np.nanmean(null) else np.nan, mean_null_match_count=float(np.mean(null_n)), fallback_relaxations=dict(match_relax)))
    enh = pd.DataFrame(enhancer_rows)
    enh["fdr_bh"] = bh(enh.empirical_p.to_numpy())
    enh.to_csv(OUT / "enhancer/enhancer_overlap_matched_peak_null.csv", index=False)

    # 4. GO/Reactome enrichment with fold-specific Task99 gene universe.
    def read_gmt(p):
        out = []
        with open(p, errors="replace") as fh:
            for line in fh:
                a = line.rstrip("\n").split("\t")
                if len(a) >= 3: out.append((a[0], set(a[2:])))
        return out
    gmt_paths = {"GO_BP": ROOT / "result_contain/AD_Dmode_4Cohort_20261024/biology_predicted_ad_control_20261025/source/gene_sets/GO_Biological_Process_2025.gmt", "Reactome": ROOT / "result_contain/AD_Dmode_4Cohort_20261024/biology_predicted_ad_control_20261025/source/gene_sets/Reactome_2022.gmt"}
    for kind, gp in gmt_paths.items():
        rows = []
        sets = read_gmt(gp) if gp.exists() else []
        for f, x in folds.items():
            prep_genes = pd.read_csv(PREP / DS[f] / "genes.tsv", sep="\t")
            names = dict(zip(prep_genes.gene_id.astype(str), prep_genes.gene_name.astype(str)))
            z = x["edges"]; selected_ids = set(z.loc[z.selected.astype(bool), "gene_id"]); universe_ids = set(x["genes"])
            sym = {g: names.get(g, "") for g in x["genes"]}
            selected = {sym[g] for g in selected_ids if sym[g]}; universe = {sym[g] for g in universe_ids if sym[g]}
            for term, members in sets:
                M = len(members & universe); n = len(selected & universe); k = len(members & selected); N = len(universe)
                if M == 0 or n == 0 or k == 0: continue
                p = float(stats.hypergeom.sf(k - 1, N, M, n))
                rows.append(dict(fold=f, term=term, overlap=k, selected_genes=n, universe_genes=N, term_genes=M, p_value=p, fold_enrichment=(k / n) / (M / N)))
        cols = ["fold", "term", "overlap", "selected_genes", "universe_genes", "term_genes", "p_value", "fold_enrichment"]
        df = pd.DataFrame(rows, columns=cols)
        df["fdr_bh"] = np.nan
        for _, g in df.groupby("fold"):
            df.loc[g.index, "fdr_bh"] = bh(g.p_value.to_numpy())
        df.sort_values(["fold", "fdr_bh", "p_value"]).to_csv(OUT / f"pathway/{kind}.csv", index=False)
    for f, x in folds.items():
        z = x["edges"]; ids = sorted(set(z.loc[z.selected.astype(bool), "gene_id"]))
        (OUT / "pathway" / f"{f}_selected_gene_ids.txt").write_text("\n".join(ids) + "\n")

    # 5. Marker expression using prepared RNA matrices and harmonized labels.
    markers = {"T": ["CD3D", "CD3E", "TRBC1", "IL7R", "LTB"], "B": ["CD79A", "MS4A1", "CD37", "HLA-DRA"], "NK": ["NKG7", "GNLY", "KLRD1", "FCER1G"], "Mono": ["LYZ", "S100A8", "S100A9", "CTSD"], "DC": ["FCER1A", "CST3", "GZMB"], "Platelet": ["PPBP", "PF4", "GNG11"]}
    marker_rows = []; label_status = []
    for f, x in folds.items():
        ds = DS[f]; prep = PREP / ds; rna = sparse.load_npz(prep / "rna_counts.npz").tocsr()
        genes = pd.read_csv(prep / "genes.tsv", sep="\t"); gnames = genes.gene_name.astype(str).to_numpy(); bars = pd.read_csv(prep / "barcodes.tsv", header=None, names=["barcode"]).barcode.astype(str).to_numpy()
        lpath = label_root / (ds + "_label_harmonized.csv"); labels = pd.read_csv(lpath) if lpath.exists() else pd.DataFrame(columns=["barcode", "cell_type"])
        lm = dict(zip(labels.barcode.astype(str), labels.cell_type.astype(str))); idx = np.array([i for i, b in enumerate(bars) if b in lm], int)
        if len(idx) < len(bars) * 0.8:
            idx = np.array([], int)
        if len(idx) < len(bars) * 0.8:
            stripped = {re.sub(r"^[^_]+_", "", k): v for k, v in lm.items()}; lm = stripped; idx = np.array([i for i, b in enumerate(bars) if b in lm], int)
        if len(idx) < len(bars) * 0.8: idx = np.array([], int)
        label_status.append(dict(fold=f, cells=len(bars), labeled_cells=len(idx), label_file=str(lpath), labels_found=bool(len(idx))))
        if len(idx) == 0: continue
        lib = np.asarray(rna.sum(axis=1)).ravel(); scale = 1e4 / np.maximum(lib, 1)
        gene_index = {g: i for i, g in enumerate(gnames)}
        selected_ids = set(x["edges"].loc[x["edges"].selected.astype(bool), "gene_id"])
        # Use Task99 gene names for selected flag where the prepared symbol matches.
        gene_map = dict(zip(genes.gene_id.astype(str), genes.gene_name.astype(str)))
        selected_symbols = {gene_map.get(gid, "") for gid in selected_ids if gene_map.get(gid, "")}
        for group, glist in markers.items():
            cells_by_label = defaultdict(list)
            for i in idx: cells_by_label[lm[bars[i]]].append(i)
            for symbol in glist:
                j = gene_index.get(symbol)
                if j is None: continue
                vals = rna[:, j].toarray().ravel() * scale
                vals = np.log1p(vals)
                for label, ci in cells_by_label.items():
                    raw = rna[:, j].toarray().ravel()
                    marker_rows.append(dict(fold=f, marker_group=group, gene_symbol=symbol, label=label, n_cells=len(ci), mean_log1p=float(vals[ci].mean()), pct_expressing=float(np.mean(raw[ci] > 0)), selected_by_B=symbol in selected_symbols))
    pd.DataFrame(label_status).to_csv(OUT / "marker/label_alignment.csv", index=False); pd.DataFrame(marker_rows).to_csv(OUT / "marker/marker_expression.csv", index=False)
    if marker_rows:
        md = pd.DataFrame(marker_rows)
        top = md.groupby(["fold", "gene_symbol"], as_index=False).mean(numeric_only=True).head(40); top.to_csv(OUT / "marker/marker_summary.csv", index=False)
        for f in FOLDS:
            plot = md[md.fold == f]
            if plot.empty: continue
            means = plot.pivot_table(index="label", columns="gene_symbol", values="mean_log1p", aggfunc="mean").fillna(0)
            perc = plot.pivot_table(index="label", columns="gene_symbol", values="pct_expressing", aggfunc="mean").fillna(0)
            fig, ax = plt.subplots(figsize=(max(9, 0.42*len(means.columns)), max(5, 0.25*len(means.index))))
            xs, ys, cs, ss = [], [], [], []
            for yi, label in enumerate(means.index):
                for xi, gene in enumerate(means.columns):
                    xs.append(xi); ys.append(yi); cs.append(means.loc[label, gene]); ss.append(18 + 110*perc.loc[label, gene])
            sc = ax.scatter(xs, ys, c=cs, s=ss, cmap="viridis", edgecolors="none")
            ax.set_xticks(range(len(means.columns))); ax.set_xticklabels(means.columns, rotation=90, fontsize=7)
            ax.set_yticks(range(len(means.index))); ax.set_yticklabels(means.index, fontsize=7); ax.invert_yaxis()
            ax.set_xlabel("PBMC marker genes"); ax.set_ylabel("cell type")
            fig.colorbar(sc, ax=ax, label="mean log1p normalized expression")
            fig.tight_layout(); fig.savefig(OUT / f"marker/{f}_marker_dotplot.png", dpi=180); plt.close(fig)

    # Save a compact run manifest.
    run = {"task": "PBMC four-fold Task99 regulatory biology", "seed": SEED, "inputs": {"base": str(BASE), "prepared": str(PREP), "legacy_cache": str(LEGACY)}, "outputs": str(OUT), "analyses": ["interval_candidate_stability", "ENCODE_matched_peak_null", "GO_BP", "Reactome", "marker_expression"], "motif": "FIMO is a separate tool step", "gene_links": "blocked by incomplete Human-Gene-Links.zip"}
    (OUT / "run_manifest.json").write_text(json.dumps(run, indent=2))
    print(json.dumps({"out": str(OUT), "common_genes": len(common_genes), "alignment": audit, "labels": label_status}, indent=2))

if __name__ == "__main__": main()

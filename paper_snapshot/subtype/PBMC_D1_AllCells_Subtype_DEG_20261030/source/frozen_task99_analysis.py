#!/usr/bin/env python3
"""Systematic, post-hoc Task99 PBMC candidate subtype analysis."""
from __future__ import annotations

import argparse
import bisect
import json
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import sparse, stats
from scipy.stats import hypergeom
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.neighbors import NearestNeighbors

SEEDS = [20020130, 20020131, 20020132]
FOLDS = ["D1", "D2", "D3", "D4"]
COLORS = ["#287C8E", "#D28E5B", "#7A9D68", "#9A7895", "#6B8795", "#C7A76C"]
MARKERS = ["CCR7", "IL7R", "TCF7", "LEF1", "SELL", "CD3D", "CD3E", "CD8A", "CD8B",
           "GZMK", "GZMB", "GZMH", "NKG7", "PRF1", "MKI67", "TOP2A", "MS4A1",
           "CD79A", "CD74", "GNLY", "FCGR3A", "LYZ", "S100A8", "S100A9", "FCER1A"]
CC_GENES = ["MKI67", "TOP2A", "STMN1", "TYMS", "HMGB2", "TUBB", "PCNA", "MCM5", "MCM6",
            "MCM7", "CCNB1", "CCNB2", "CDC20", "CENPF", "UBE2C"]
CURRENT_ROOT: Path


def read_sparse(path: Path) -> sparse.csr_matrix:
    z = np.load(path)
    return sparse.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))


def bh(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, float)
    order = np.argsort(p)
    q = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(len(p), float)
    out[order] = np.clip(q, 0, 1)
    return out


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "_", value).strip("_")


def kmeans(x: np.ndarray, k: int, seed: int, n_init: int = 8) -> np.ndarray:
    n = len(x); best = None; best_inertia = math.inf
    x2 = np.einsum("ij,ij->i", x, x)
    for restart in range(n_init):
        rng = np.random.RandomState(seed + restart * 104729)
        centers = np.empty((k, x.shape[1]), np.float64)
        centers[0] = x[rng.randint(n)]
        closest = np.einsum("ij,ij->i", x - centers[0], x - centers[0])
        for c in range(1, k):
            total = closest.sum()
            pick = rng.randint(n) if not np.isfinite(total) or total <= 0 else rng.choice(n, p=closest / total)
            centers[c] = x[pick]
            closest = np.minimum(closest, np.einsum("ij,ij->i", x - centers[c], x - centers[c]))
        previous = None
        for _ in range(100):
            d2 = x2[:, None] - 2 * (x @ centers.T) + np.einsum("ij,ij->i", centers, centers)[None, :]
            labels = np.argmin(d2, axis=1)
            if previous is not None and np.array_equal(previous, labels): break
            previous = labels.copy()
            for c in range(k):
                members = x[labels == c]
                if len(members): centers[c] = members.mean(axis=0)
                else: centers[c] = x[np.argmax(np.min(d2, axis=1))]
        d2 = x2[:, None] - 2 * (x @ centers.T) + np.einsum("ij,ij->i", centers, centers)[None, :]
        inertia = float(np.maximum(d2[np.arange(n), labels], 0).sum())
        if inertia < best_inertia: best, best_inertia = labels, inertia
    return best


def k_extra(nlabels: int) -> int:
    return min(max(2 * nlabels, 30), 50)


def find_splits(clusters: np.ndarray, labels: np.ndarray, fold: str, seed: int, k: int) -> list[dict]:
    rows = []
    for label in sorted(set(labels)):
        choices = []
        for cl in np.unique(clusters):
            cm = clusters == cl; n = int(cm.sum())
            if n < 30: continue
            vals, counts = np.unique(labels[cm], return_counts=True)
            ix = int(np.argmax(counts)); majority, purity = str(vals[ix]), float(counts[ix] / n)
            if majority == label and purity >= .60:
                choices.append((int(cl), n, purity))
        if len(choices) >= 2:
            choices.sort(key=lambda z: z[1], reverse=True)
            a, b = choices[:2]
            rows.append({"fold": fold, "seed": seed, "k_extra": k, "reference_cell_type": label,
                         "cluster_a": a[0], "cluster_b": b[0], "n_a": a[1], "n_b": b[1],
                         "purity_a": a[2], "purity_b": b[2]})
    return rows


def candidate_partition(cand: dict, clusters: np.ndarray, labels: np.ndarray, barcodes: np.ndarray) -> pd.Series:
    ma = (clusters == int(cand["cluster_a"])) & (labels == cand["reference_cell_type"])
    mb = (clusters == int(cand["cluster_b"])) & (labels == cand["reference_cell_type"])
    ix = np.flatnonzero(ma | mb)
    return pd.Series(np.where(clusters[ix] == int(cand["cluster_a"]), "A", "B"), index=barcodes[ix])


def rank_genes(x: sparse.csr_matrix, a: np.ndarray, b: np.ndarray, symbols: np.ndarray,
               gene_ids: np.ndarray, candidate_id: str, fold: str, seed: int, contrast: str,
               min_cells: int = 10, input_mode: str = "counts") -> pd.DataFrame:
    na, nb = int(a.sum()), int(b.sum())
    if na < min_cells or nb < min_cells: return pd.DataFrame()
    xa, xb = x[a].tocsr(), x[b].tocsr()
    la = np.maximum(np.asarray(xa.sum(1)).ravel(), 1); lb = np.maximum(np.asarray(xb.sum(1)).ravel(), 1)
    parts=[]
    for lo in range(0, x.shape[1], 512):
        hi=min(x.shape[1],lo+512)
        ca, cb = xa[:,lo:hi], xb[:,lo:hi]
        if input_mode == "log1p_cp10k":
            ma,mb=ca.toarray(),cb.toarray()
            mean_a=np.expm1(ma).mean(0);mean_b=np.expm1(mb).mean(0)
        else:
            ma = np.log1p(ca.toarray() * (1e4/la)[:,None]); mb=np.log1p(cb.toarray()*(1e4/lb)[:,None])
            mean_a=np.asarray(ca.multiply((1e4/la)[:,None]).mean(0)).ravel()
            mean_b=np.asarray(cb.multiply((1e4/lb)[:,None]).mean(0)).ravel()
        test=stats.mannwhitneyu(ma,mb,axis=0,alternative="two-sided",method="asymptotic")
        p=np.maximum(np.asarray(test.pvalue),np.finfo(float).tiny)
        delta=ma.mean(0)-mb.mean(0); score=np.sign(delta)*stats.norm.isf(p/2)
        parts.append(pd.DataFrame({"candidate_id":candidate_id,"fold":fold,"seed":seed,"contrast":contrast,
            "gene_symbol":symbols[lo:hi],"gene_id":gene_ids[lo:hi],"log2FC_meanCP10K":np.log2((mean_a+1e-4)/(mean_b+1e-4)),
            "wilcoxon_score_signed":score,"p_value":p,"mean_log1pCP10K_A":ma.mean(0),"mean_log1pCP10K_B":mb.mean(0),
            "pct_detected_A":np.asarray((ca>0).mean(0)).ravel(),"pct_detected_B":np.asarray((cb>0).mean(0)).ravel(),"n_A":na,"n_B":nb}))
    out=pd.concat(parts,ignore_index=True)
    out["FDR_BH_within_comparison"]=bh(out.p_value.to_numpy())
    out["strict_DEG"]=(out.FDR_BH_within_comparison<.05)&(out.log2FC_meanCP10K.abs()>.25)
    return out


def rank_peaks(x: sparse.csr_matrix,a: np.ndarray,b: np.ndarray,peak_names: np.ndarray,
               candidate_id: str,fold: str,seed: int) -> pd.DataFrame:
    if min(int(a.sum()),int(b.sum()))<10: return pd.DataFrame()
    xa,xb=x[a].tocsr(),x[b].tocsr(); parts=[]
    for lo in range(0,x.shape[1],512):
        hi=min(x.shape[1],lo+512); ca,cb=xa[:,lo:hi].toarray(),xb[:,lo:hi].toarray()
        t=stats.mannwhitneyu(ca,cb,axis=0,alternative="two-sided",method="asymptotic")
        p=np.maximum(np.asarray(t.pvalue),np.finfo(float).tiny); d=(ca>0).mean(0)-(cb>0).mean(0)
        score=np.sign(ca.mean(0)-cb.mean(0))*stats.norm.isf(p/2)
        parts.append(pd.DataFrame({"candidate_id":candidate_id,"fold":fold,"seed":seed,"peak_index":np.arange(lo,hi),"peak":peak_names[lo:hi],
            "mean_fragments_A":ca.mean(0),"mean_fragments_B":cb.mean(0),"pct_accessible_A":(ca>0).mean(0),"pct_accessible_B":(cb>0).mean(0),
            "delta_accessibility":d,"wilcoxon_score_signed":score,"p_value":p}))
    out=pd.concat(parts,ignore_index=True);out["FDR_BH_within_comparison"]=bh(out.p_value.to_numpy())
    out["strict_DA"]=(out.FDR_BH_within_comparison<.05)&(out.delta_accessibility.abs()>=.05)
    return out


def peak_interval_map(canonical: np.ndarray, peak_table: pd.DataFrame) -> dict[int,tuple[int,int]]:
    bychr={}
    for chrom,g in peak_table.groupby(0,sort=False):
        bychr[str(chrom)]=(g.index.to_numpy(),g[1].to_numpy(int),g[2].to_numpy(int))
    mapping={}
    for i,name in enumerate(canonical.astype(str)):
        try: chrom,coords=name.split(":",1); st,en=map(int,coords.split("-",1))
        except ValueError: continue
        if chrom not in bychr:continue
        ids,starts,ends=bychr[chrom];j=max(0,bisect.bisect_left(starts,en)-1);best=None
        while j<len(starts) and starts[j]<en:
            overlap=max(0,min(en,ends[j])-max(st,starts[j]))
            if overlap and (best is None or overlap>best[1]):best=(int(ids[j]),overlap)
            j+=1
        if best:mapping[i]=best
    return mapping


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--root",type=Path,required=True)
    ap.add_argument("--data",type=Path,required=True)
    ap.add_argument("--regulatory",type=Path,required=True)
    ap.add_argument("--gmt",type=Path,required=True)
    ap.add_argument("--out",type=Path,required=True)
    ap.add_argument("--gsea-permutations",type=int,default=2000)
    args=ap.parse_args(); global CURRENT_ROOT; CURRENT_ROOT=args.root
    out=args.out; tables=out/"tables"; figures=out/"figures"
    tables.mkdir(parents=True,exist_ok=True);figures.mkdir(parents=True,exist_ok=True)
    raw_splits=pd.read_csv(args.root/"candidate_subtype_splits.csv")
    old_summary=pd.read_csv(args.root/"candidate_subtype_summary.csv")
    old_markers=pd.read_csv(args.root/"candidate_subtype_markers.csv")
    composition=pd.read_csv(args.root/"cluster_composition.csv")
    ranked_markers=pd.read_csv(args.root/"cluster_markers_ranked.csv")
    manifest=json.loads((args.root/"analysis_manifest.json").read_text())
    candidates={}; partitions={}; fold_data={}; reconstruction=[]; detected_splits=[]; fold_data_status={}
    for fold in FOLDS:
        labeldf=pd.read_csv(args.root/f"{fold}_label_harmonized.csv")
        label_map=dict(zip(labeldf.barcode.astype(str),labeldf.cell_type.astype(str)))
        regrows=raw_splits[raw_splits.fold==fold].copy()
        target_candidates={}
        for _,r in regrows.iterrows():
            key=(fold,str(r.reference_cell_type))
            target_candidates.setdefault(key,[]).append(r.to_dict())
        # D4 had no candidate meeting the saved screen. Repeat that same screen at its recorded K range.
        fold_dir=args.root/"task99_embeddings"/f"target_{fold}"
        for seed in SEEDS:
            embdir=fold_dir/f"seed_{seed}"/"embeddings"
            order=pd.read_csv(embdir/"target_cell_order.tsv",sep="\t",header=0,dtype=str).iloc[:,0].astype(str).to_numpy()
            xemb=np.load(embdir/"target_embedding_standardized.npy").astype(np.float64)
            labels=np.array([label_map[b] for b in order],object)
            if len(order)!=len(xemb) or len(set(order))!=len(order):raise RuntimeError(f"Embedding/barcode mismatch {fold}/{seed}")
            k=k_extra(len(set(labels))); kmseed=900000+int(fold[1:])*10000+seed%1000
            cl=kmeans(xemb,k,kmseed,8)
            discovered=find_splits(cl,labels,fold,seed,k)
            detected_splits.extend(discovered)
            for split in discovered:
                key=(fold,str(split["reference_cell_type"]))
                rows=target_candidates.setdefault(key,[])
                if not any(int(r["seed"])==seed for r in rows):
                    split["exploratory_rescan"]=True;rows.append(split)
            for key, rows in list(target_candidates.items()):
                occurrences=[r for r in rows if int(r["seed"])==seed]
                for r in occurrences:
                    ca,cb=int(r["cluster_a"]),int(r["cluster_b"])
                    na,nb=int((cl==ca).sum()),int((cl==cb).sum())
                    match=na==int(r["n_a"]) and nb==int(r["n_b"])
                    reconstruction.append({"fold":fold,"candidate_id":f"{fold}_{slug(key[1])}_C1","seed":seed,
                        "reference_subclass":key[1],"cluster_a":ca,"cluster_b":cb,"recorded_n_a":int(r["n_a"]),
                        "recorded_n_b":int(r["n_b"]),"recomputed_n_a":na,"recomputed_n_b":nb,"exact_reconstruction":match,
                        "exploratory_rescan":bool(r.get("exploratory_rescan",False))})
                    if match:
                        cand={**r,"fold":fold,"candidate_id":f"{fold}_{slug(key[1])}_C1","seed":seed,
                              "reference_cell_type":key[1],"k_extra":k,"exploratory_rescan":bool(r.get("exploratory_rescan",False)),
                              "global_comp_a":pd.Series(labels[cl==ca]).value_counts().to_dict(),
                              "global_comp_b":pd.Series(labels[cl==cb]).value_counts().to_dict()}
                        p=candidate_partition(cand,cl,labels,order)
                        cand["candidate_cells_label_filtered"]=len(p);cand["partition"]=p
                        candidates.setdefault(cand["candidate_id"],[]).append(cand)
                        partitions[(cand["candidate_id"],seed)]=p
            fold_data[fold] = {"labels":labels,"order":order,"embedding":xemb,"clusters":cl,"k":k}
        # Label files may include all cells; target-test order controls target data selection.
        data_dir=args.data/fold
        try:
            raw_barcodes=pd.read_csv(data_dir/"barcodes.tsv",header=None,dtype=str)[0].astype(str).to_numpy()
            rna=read_sparse(data_dir/"rna_counts.npz")
            gdf=pd.read_csv(data_dir/"genes.tsv",sep="\t",dtype=str); gene_ids=gdf.iloc[:,0].astype(str).to_numpy();symbols=gdf.iloc[:,1].fillna(gdf.iloc[:,0]).astype(str).to_numpy()
            atac=None;ptab=pd.DataFrame();peak_names=np.array([],dtype=str)
            if (data_dir/"atac_counts.npz").exists() and (data_dir/"peaks.bed").exists():
                atac=read_sparse(data_dir/"atac_counts.npz")
                ptab=pd.read_csv(data_dir/"peaks.bed",sep="\t",header=None,dtype=str); peak_names=(ptab[0]+":"+ptab[1]+"-"+ptab[2]).to_numpy()
            order_set=set(fold_data[fold]["order"]);raw_lookup={b:i for i,b in enumerate(raw_barcodes)}
            if len(order_set)!=len(fold_data[fold]["order"]):raise RuntimeError("duplicate target embedding barcode")
            if not order_set<=set(raw_lookup):raise RuntimeError(f"Target barcode absent from paired {fold} data ({len(order_set & set(raw_lookup))}/{len(order_set)} exact matches)")
            target_rows=np.array([raw_lookup[b] for b in fold_data[fold]["order"]],int)
            xt=rna[target_rows,:].tocsr();at=atac[target_rows,:].tocsr() if atac is not None else None
            if xt.shape[0]!=len(fold_data[fold]["order"]) or (atac is not None and atac[target_rows].shape[0]!=len(fold_data[fold]["order"])):raise RuntimeError("target dimensions mismatch")
            matrix_mode=(data_dir/"matrix_mode.txt").read_text().strip() if (data_dir/"matrix_mode.txt").exists() else "counts"
            if matrix_mode not in {"counts","log1p_cp10k"}:raise RuntimeError(f"unsupported matrix mode {matrix_mode}")
            at=atac[target_rows].tocsr() if atac is not None else None
            fold_data[fold].update({"rna":xt,"atac":at,"gene_ids":gene_ids,"symbols":symbols,"peaks":peak_names,"peak_table":ptab,"raw_barcodes":raw_barcodes})
            fold_data[fold]["matrix_mode"]=matrix_mode
            fold_data_status[fold]={"status":"PASS" if atac is not None else "RNA_ONLY","paired_barcode_matches":len(target_rows),"target_barcodes":len(order_set),"matrix_mode":matrix_mode,"gene_features":len(gene_ids)}
            print(f"{fold}: target={len(target_rows)} RNA barcode match=PASS paired_ATAC={'AVAILABLE' if atac is not None else 'UNAVAILABLE'} candidates={sum(1 for c in candidates if c.startswith(fold+'_'))}",flush=True)
        except Exception as e:
            fold_data_status[fold]={"status":"BLOCKED","reason":str(e)}
            print(f"{fold}: paired RNA/ATAC unavailable; retaining inventory only: {e}",flush=True)
    # Per candidate seed inventory and exact-barcode stability.
    inv=[]; stability=[]; candidate_members=[]; all_de=[]; all_da=[]; qc_rows=[]; da_summaries=[]; edge_summaries=[]
    for cid, rows in candidates.items():
        fold=rows[0]["fold"]; label=rows[0]["reference_cell_type"]; fdat=fold_data[fold]
        rows=sorted(rows,key=lambda r:r["seed"]); seedset=[r["seed"] for r in rows]
        markers=old_markers[(old_markers.fold==fold)&(old_markers.reference_cell_type==label)]
        marker_counts=markers.groupby("seed").size().to_dict() if len(markers) else {}
        for r in rows:
            p=r["partition"]
            candidate_members.extend([{"candidate_id":cid,"fold":fold,"seed":r["seed"],"barcode":bc,"candidate_group":g} for bc,g in p.items()])
        for i,ra in enumerate(rows):
            for rb in rows[i+1:]:
                pa,pb=ra["partition"],rb["partition"]; common=sorted(set(pa.index)&set(pb.index)); union=set(pa.index)|set(pb.index)
                if common:
                    ari=adjusted_rand_score(pa.loc[common],pb.loc[common]);nmi=normalized_mutual_info_score(pa.loc[common],pb.loc[common])
                else:ari=np.nan;nmi=np.nan
                memjac=len(set(pa.index)&set(pb.index))/max(1,len(union))
                same=max(len(set(pa[pa=="A"].index)&set(pb[pb=="A"].index))+len(set(pa[pa=="B"].index)&set(pb[pb=="B"].index)),
                         len(set(pa[pa=="A"].index)&set(pb[pb=="B"].index))+len(set(pa[pa=="B"].index)&set(pb[pb=="A"].index)))
                aligned=same/max(1,len(union))
                stability.append({"candidate_id":cid,"fold":fold,"reference_subclass":label,"seed_a":ra["seed"],"seed_b":rb["seed"],
                    "n_shared_barcodes":len(common),"member_jaccard":memjac,"partition_ARI_common":ari,"partition_NMI_common":nmi,
                    "partition_group_overlap_aligned_jaccard":aligned})
        stabrows=[q for q in stability if q["candidate_id"]==cid]
        if "rna" not in fdat:
            source_rows=raw_splits[(raw_splits.fold==fold)&(raw_splits.reference_cell_type==label)]
            rec_seeds=sorted(set(source_rows.seed.astype(int).tolist()) | {int(r["seed"]) for r in rows if r.get("exploratory_rescan",False)})
            inv.append({"candidate_id":cid,"fold":fold,"reference_major_cell_type":"not separately available in harmonized label metadata",
                "reference_subclass":label,"recorded_or_rescanned_seeds":";".join(map(str,rec_seeds)),"reproducible_seeds":";".join(map(str,seedset)),
                "candidate_A_n_by_seed":";".join(f"{r['seed']}:{int((r['partition']=='A').sum())}" for r in rows),
                "candidate_B_n_by_seed":";".join(f"{r['seed']}:{int((r['partition']=='B').sum())}" for r in rows),
                "mean_member_jaccard":float(np.nanmean([s['member_jaccard'] for s in stabrows])) if stabrows else np.nan,
                "mean_partition_ARI":float(np.nanmean([s['partition_ARI_common'] for s in stabrows])) if stabrows else np.nan,
                "mean_partition_NMI":float(np.nanmean([s['partition_NMI_common'] for s in stabrows])) if stabrows else np.nan,
                "old_marker_rows":int(len(markers)),"RNA_only_result":"UNAVAILABLE strict Task99-matched baseline not located",
                "global_cluster_composition_by_seed":json.dumps({str(r['seed']):{"A":r['global_comp_a'],"B":r['global_comp_b']} for r in rows},sort_keys=True),
                "paired_ATAC":"UNAVAILABLE paired target counts not staged" if fdat.get("atac") is None else "AVAILABLE",
                "DEG_status":f"PARTIAL {len(fdat['gene_ids'])}-feature {fdat['matrix_mode']} RNA cache" if fdat.get("matrix_mode")=="log1p_cp10k" else "UNAVAILABLE full-gene paired count cache missing",
                "GSEA_status":"EXPLORATORY restricted feature set" if fdat.get("matrix_mode")=="log1p_cp10k" else "UNAVAILABLE no gene-level ranking available",
                "confounder_risk":"sample/batch/donor/doublet/TSS metadata absent from Task99 target package",
                "exploratory_D4_rescan":bool(any(r.get('exploratory_rescan',False) for r in rows))})
            continue
        fold_labels=fdat["labels"]; order=fdat["order"]; x=fdat["rna"]; a=fdat["atac"]
        refmask=fold_labels==label
        valid_union=set().union(*(set(r["partition"].index) for r in rows))
        # Candidate A/B groups are restricted to the reference label; remaining comparison group is all other cells with this label.
        for r in rows:
            seed=r["seed"]; part=r["partition"]; amap=np.array([part.get(b)=="A" for b in order]);bmap=np.array([part.get(b)=="B" for b in order])
            rest=refmask&~(amap|bmap)
            groups={"A_vs_B":(amap,bmap),"A_vs_other_reference":(amap,rest),"B_vs_other_reference":(bmap,rest)}
            for contrast,(ma,mb) in groups.items():
                de=rank_genes(x,ma,mb,fdat["symbols"],fdat["gene_ids"],cid,fold,seed,contrast,input_mode=fdat.get("matrix_mode","counts"))
                if len(de):all_de.append(de)
            da=rank_peaks(a,amap,bmap,fdat["peaks"],cid,fold,seed) if a is not None else pd.DataFrame()
            if len(da):
                sig=da[da.strict_DA].copy();top=da.sort_values(["FDR_BH_within_comparison","p_value"]).head(100)
                all_da.append(da)
                da_summaries.append({"candidate_id":cid,"fold":fold,"seed":seed,"n_A":int(amap.sum()),"n_B":int(bmap.sum()),
                                     "tested_peaks":len(da),"strict_DA_peaks":int(da.strict_DA.sum()),
                                     "top_peak_nominal_p_lt_0.01":int((da.p_value<.01).sum())})
            lib=np.asarray(x.sum(1)).ravel();det=np.asarray((x>0).sum(1)).ravel();mt=np.array([s.upper().startswith("MT-") for s in fdat["symbols"]])
            mtpct=100*np.asarray(x[:,mt].sum(1)).ravel()/np.maximum(lib,1) if mt.any() else np.full(len(lib),np.nan)
            ix={g:i for i,g in enumerate(fdat["symbols"])};cc=[ix[z] for z in CC_GENES if z in ix]
            ccscore=np.log1p(x[:,cc].toarray()*(1e4/np.maximum(lib,1))[:,None]).mean(1) if cc else np.full(len(lib),np.nan)
            for group,mask in [("A",amap),("B",bmap),("other_reference",rest)]:
                metrics=[("RNA_UMI",lib),("detected_genes",det),("mitochondrial_pct",mtpct),("cell_cycle_descriptive_score",ccscore)]
                if fdat.get("matrix_mode")=="log1p_cp10k":metrics=[("log1p_subset_sum_not_UMI",lib),("detected_features_449",det),("mitochondrial_pct_subset",mtpct),("cell_cycle_descriptive_score_subset",ccscore)]
                for metric,vals in metrics:
                    q=vals[mask];qc_rows.append({"candidate_id":cid,"fold":fold,"seed":seed,"group":group,"metric":metric,"n":len(q),
                        "mean":float(np.nanmean(q)) if len(q) else np.nan,"sd":float(np.nanstd(q,ddof=1)) if len(q)>1 else np.nan,
                        "median":float(np.nanmedian(q)) if len(q) else np.nan,"q25":float(np.nanquantile(q,.25)) if len(q) else np.nan,
                        "q75":float(np.nanquantile(q,.75)) if len(q) else np.nan})
            candidate_membership=amap|bmap
            for group,mask in [("A",amap),("B",bmap)]:
                comp=pd.Series(fold_labels[mask]).value_counts()
                for typ,n in comp.items(): pass
            # Source edge support and target peak overlap is analyzed after all candidate DA are pooled below.
        seed_sizes=[]
        for r in rows:
            p=r["partition"];seed_sizes.append((int((p=="A").sum()),int((p=="B").sum())))
        comp_a={};comp_b={}
        # Aggregate reference composition over the seed-specific exact candidate cells.
        for r in rows:
            p=r["partition"]
            for group in ["A","B"]:
                for bc in p[p==group].index:
                    comp_a[(group,str(fold_data[fold]["labels"][np.where(fold_data[fold]["order"]==bc)[0][0]]))]=comp_a.get((group,str(label)),0)+1
        source_rows=raw_splits[(raw_splits.fold==fold)&(raw_splits.reference_cell_type==label)]
        rec_seeds=sorted(set(source_rows.seed.astype(int).tolist()) | {int(r["seed"]) for r in rows if r.get("exploratory_rescan",False)})
        inv.append({"candidate_id":cid,"fold":fold,"reference_major_cell_type":"not separately available in harmonized label metadata",
            "reference_subclass":label,"recorded_or_rescanned_seeds":";".join(map(str,rec_seeds)),"reproducible_seeds":";".join(map(str,seedset)),
            "candidate_A_n_by_seed":";".join(f"{r['seed']}:{int((r['partition']=='A').sum())}" for r in rows),
            "candidate_B_n_by_seed":";".join(f"{r['seed']}:{int((r['partition']=='B').sum())}" for r in rows),
            "mean_member_jaccard":float(np.nanmean([s['member_jaccard'] for s in stabrows])) if stabrows else np.nan,
            "mean_partition_ARI":float(np.nanmean([s['partition_ARI_common'] for s in stabrows])) if stabrows else np.nan,
            "mean_partition_NMI":float(np.nanmean([s['partition_NMI_common'] for s in stabrows])) if stabrows else np.nan,
            "old_marker_rows":int(len(markers)),"RNA_only_result":"UNAVAILABLE strict Task99-matched baseline not located",
            "global_cluster_composition_by_seed":json.dumps({str(r["seed"]):{"A":r["global_comp_a"],"B":r["global_comp_b"]} for r in rows},sort_keys=True),
            "paired_ATAC":"UNAVAILABLE" if fold_data_status.get(fold,{}).get("status") in {"BLOCKED","RNA_ONLY"} else "AVAILABLE",
            "DEG_status":("BLOCKED no matched RNA matrix" if fold_data_status.get(fold,{}).get("status")=="BLOCKED" else
                          f"PARTIAL {len(fdat['gene_ids'])}-feature normalized RNA cache" if fold_data_status.get(fold,{}).get("matrix_mode")=="log1p_cp10k" else "COMPLETED full-gene target counts"),
            "GSEA_status":"RUN after DE","confounder_risk":"sample/batch/donor/doublet/TSS metadata absent from Task99 target package",
            "exploratory_D4_rescan":bool(any(r.get('exploratory_rescan',False) for r in rows))})
    for fold in FOLDS:
        if not any(r["fold"]==fold for r in inv):
            inv.append({"candidate_id":f"{fold}_NO_SPLIT_AT_K{k_extra(len(set(fold_data[fold]['labels'])))}","fold":fold,
                "reference_major_cell_type":"NA","reference_subclass":"No reference subclass met the saved >=2 clusters, >=30 cells, >=0.60 purity screen",
                "recorded_or_rescanned_seeds":";".join(map(str,SEEDS)),"reproducible_seeds":";".join(map(str,SEEDS)),
                "candidate_A_n_by_seed":"","candidate_B_n_by_seed":"","mean_member_jaccard":np.nan,"mean_partition_ARI":np.nan,
                "mean_partition_NMI":np.nan,"old_marker_rows":0,"RNA_only_result":"UNAVAILABLE strict Task99-matched baseline not located",
                "global_cluster_composition_by_seed":"{}","paired_ATAC":"UNAVAILABLE" if fold_data_status.get(fold,{}).get("status")=="BLOCKED" else "AVAILABLE","DEG_status":"NOT_APPLICABLE no candidate passed screen",
                "GSEA_status":"NOT_APPLICABLE","confounder_risk":"sample/batch/donor/doublet/TSS metadata absent from paired cache",
                "exploratory_D4_rescan":fold=="D4"})
    # Correct all tested gene-comparison rows as one prespecified family in addition to within-comparison BH.
    if all_de:
        deg=pd.concat(all_de,ignore_index=True)
        deg["FDR_BH_global_all_candidate_tests"]=bh(deg.p_value.to_numpy())
        deg["strict_DEG_global"]=(deg.FDR_BH_global_all_candidate_tests<.05)&(deg.log2FC_meanCP10K.abs()>.25)
        deg.to_csv(tables/"all_candidate_full_gene_deg.csv.gz",index=False,compression="gzip")
        summary=deg.groupby(["candidate_id","fold","seed","contrast"],dropna=False).agg(tested_genes=("gene_id","size"),
            within_FDR_DEG=("strict_DEG","sum"),global_FDR_DEG=("strict_DEG_global","sum"),nominal_p_lt_0_01=("p_value",lambda z:int((z<.01).sum())),
            median_abs_log2FC=("log2FC_meanCP10K",lambda z:float(np.nanmedian(np.abs(z))))).reset_index()
        summary.to_csv(tables/"all_candidate_deg_summary.csv",index=False)
        # Nominal/trend markers and seed-direction consistency, explicitly exploratory.
        mark=deg[deg.gene_symbol.isin(MARKERS)].copy();mark.to_csv(tables/"canonical_marker_checks_all_candidates.csv",index=False)
        trend=deg[(deg.p_value<.01)&(deg.log2FC_meanCP10K.abs()>.5)].copy()
        trend.to_csv(tables/"exploratory_nominal_markers.csv.gz",index=False,compression="gzip")
    else:
        pd.DataFrame().to_csv(tables/"all_candidate_full_gene_deg.csv.gz",index=False,compression="gzip")
        summary=pd.DataFrame(columns=["candidate_id","fold","seed","contrast","tested_genes","within_FDR_DEG","global_FDR_DEG","nominal_p_lt_0_01","median_abs_log2FC"])
        summary.to_csv(tables/"all_candidate_deg_summary.csv",index=False)
    # Candidate-level ATAC testing and source-edge join. Only preserve top 100 peaks per contrast plus strict DA calls.
    if all_da:
        da=pd.concat(all_da,ignore_index=True).drop_duplicates(["candidate_id","seed","peak_index"])
        da["FDR_BH_global_all_candidate_tests"]=bh(da.p_value.to_numpy())
        da["strict_DA_global"]=(da.FDR_BH_global_all_candidate_tests<.05)&(da.delta_accessibility.abs()>=.05)
        da.to_csv(tables/"all_candidate_full_peak_differential_accessibility.csv.gz",index=False,compression="gzip")
        dasum=pd.DataFrame(da_summaries)
        if len(dasum):
            global_count=da[da.strict_DA_global].groupby(["candidate_id","seed"]).size().to_dict()
            dasum["strict_DA_global_count"]=[int(global_count.get((r.candidate_id,r.seed),0)) for r in dasum.itertuples()]
        dasum.to_csv(tables/"atac_evidence_summary.csv",index=False)
    else: pd.DataFrame(columns=["candidate_id","fold","seed","n_A","n_B","tested_peaks","strict_DA_peaks","top_peak_nominal_p_lt_0.01"]).to_csv(tables/"atac_evidence_summary.csv",index=False)
    # Source-only regulatory template linked after frozen inference; interval overlap is explicit.
    edge_rows=[]
    # Preserve recorded candidates even when their saved cluster IDs cannot be reconstructed.
    for (fold,label),source_rows in raw_splits.groupby(["fold","reference_cell_type"]):
        cid=f"{fold}_{slug(str(label))}_C1"
        if any(x["candidate_id"]==cid for x in inv):continue
        rec=sorted(set(source_rows.seed.astype(int).tolist()))
        inv.append({"candidate_id":cid,"fold":fold,"reference_major_cell_type":"not separately available in harmonized label metadata",
            "reference_subclass":label,"recorded_or_rescanned_seeds":";".join(map(str,rec)),"reproducible_seeds":"",
            "candidate_A_n_by_seed":"","candidate_B_n_by_seed":"","mean_member_jaccard":np.nan,"mean_partition_ARI":np.nan,
            "mean_partition_NMI":np.nan,"old_marker_rows":int(((old_markers.fold==fold)&(old_markers.reference_cell_type==label)).sum()),
            "global_cluster_composition_by_seed":"{}","RNA_only_result":"UNAVAILABLE strict Task99-matched baseline not located",
            "paired_ATAC":"UNAVAILABLE" if fold_data_status.get(fold,{}).get("status") in {"BLOCKED","RNA_ONLY"} else "AVAILABLE",
            "DEG_status":"BLOCKED paired barcode validation failed" if fold_data_status.get(fold,{}).get("status")=="BLOCKED" else "NOT_RUN cluster reconstruction failed","GSEA_status":"NOT_RUN",
            "confounder_risk":"sample/batch/donor/doublet/TSS metadata absent from paired cache","exploratory_D4_rescan":False})
    for fold in FOLDS:
        ep=args.regulatory/f"target_{fold}";fdat=fold_data[fold]
        if fdat.get("atac") is None: continue
        if not (ep/"gene_ids.npy").exists():continue
        canonical=np.load(ep/"candidate_graph.npz",allow_pickle=True)["peak_names"].astype(str)
        pmap=peak_interval_map(canonical,fdat["peak_table"])
        rg=np.load(ep/"gene_ids.npy",allow_pickle=True).astype(str);egi=np.load(ep/"edge_gene.npy").astype(int);epi=np.load(ep/"edge_peak.npy").astype(int)
        sup=np.load(ep/"edge_support.npy").astype(float);sign=np.load(ep/"sign_D.npy").astype(float)
        candidates_here=[c for c in candidates if c.startswith(fold+"_")]
        for cid in candidates_here:
            fold_rows=[r for r in candidates[cid]]
            for r in fold_rows:
                seed=r["seed"];p=r["partition"];order=fdat["order"]
                amap=np.array([p.get(b)=="A" for b in order]);bmap=np.array([p.get(b)=="B" for b in order])
                da=rank_peaks(fdat["atac"],amap,bmap,fdat["peaks"],cid,fold,seed)
                if not len(da):continue
                da_ix=da.set_index("peak_index");de_file=tables/"all_candidate_full_gene_deg.csv.gz"
                # Defer detailed edge join until pooled DEG data exists below.
                for gi,pi,su,sg in zip(egi,epi,sup,sign):
                    if gi>=len(rg) or pi not in pmap:continue
                    target_pi,ov=pmap[pi]
                    if target_pi not in da_ix.index:continue
                    q=da_ix.loc[target_pi]
                    edge_rows.append({"candidate_id":cid,"fold":fold,"seed":seed,"gene_id":rg[gi],"source_peak":canonical[pi],
                        "target_peak":fdat["peaks"][target_pi],"target_peak_index":target_pi,"interval_overlap_bp":ov,
                        "target_delta_accessibility":q.delta_accessibility,"target_ATAC_FDR":q.FDR_BH_within_comparison,
                        "target_DA_strict":bool(q.strict_DA),"source_edge_support":su,"source_sign_D":sg})
    edges=pd.DataFrame(edge_rows)
    if not edges.empty and all_de:
        deall=pd.concat(all_de,ignore_index=True)
        ab=deall[deall.contrast=="A_vs_B"].copy()
        ab=ab.drop_duplicates(["candidate_id","seed","gene_id"])
        edges=edges.merge(ab[["candidate_id","seed","gene_id","gene_symbol","log2FC_meanCP10K","p_value","FDR_BH_within_comparison","strict_DEG"]],
                          on=["candidate_id","seed","gene_id"],how="left")
        edges["RNA_ATAC_direction_concordant"]=np.sign(edges.log2FC_meanCP10K)==np.sign(edges.target_delta_accessibility)
        edges["triple_evidence"]=edges.strict_DEG.fillna(False)&edges.target_DA_strict
    edges.to_csv(tables/"all_candidate_source_edges.csv.gz",index=False,compression="gzip")
    for cid,g in edges.groupby("candidate_id") if len(edges) else []:
        edge_summaries.append({"candidate_id":cid,"source_link_rows":len(g),"strict_target_DA_links":int(g.target_DA_strict.sum()),
                               "RNA_DEG_ATAC_DA_source_edge_triples":int(g.triple_evidence.sum()),
                               "direction_concordant_links":int(g.RNA_ATAC_direction_concordant.fillna(False).sum())})
    pd.DataFrame(edge_summaries,columns=["candidate_id","source_link_rows","strict_target_DA_links","RNA_DEG_ATAC_DA_source_edge_triples","direction_concordant_links"]).to_csv(tables/"source_edge_summary.csv",index=False)
    pd.DataFrame(qc_rows).to_csv(tables/"candidate_qc_descriptives.csv",index=False)
    pd.DataFrame(stability).to_csv(tables/"candidate_seed_stability.csv",index=False)
    pd.DataFrame(reconstruction).to_csv(tables/"candidate_reconstruction_audit.csv",index=False)
    pd.DataFrame(candidate_members).to_csv(tables/"candidate_membership_barcodes.csv.gz",index=False,compression="gzip")
    pd.DataFrame(inv).to_csv(out/"candidate_inventory.csv",index=False)
    # Full B2 UMAPs are descriptive only; no RNA-only representation is fabricated.
    try:
        import umap
        for fold in FOLDS:
            if "embedding" not in fold_data.get(fold,{}): continue
            fd=fold_data[fold];emb=fd["embedding"];co=umap.UMAP(n_neighbors=20,min_dist=.25,metric="euclidean",random_state=20020130).fit_transform(emb)
            refs=fd["labels"]
            fig,axs=plt.subplots(1,2,figsize=(10,4.2))
            for label in sorted(set(refs)):
                m=refs==label;axs[0].scatter(co[m,0],co[m,1],s=5,alpha=.7,label=label,rasterized=True)
            axs[0].set_title(f"{fold} Full B2 embedding: reference labels")
            candmask=np.zeros(len(refs),bool);grp=np.full(len(refs),"other",object)
            for cid,rr in candidates.items():
                if not cid.startswith(fold+"_"):continue
                r=next((z for z in rr if z["seed"]==20020130),None)
                if r is None:continue
                pp=r["partition"]
                for j,b in enumerate(fd["order"]):
                    if b in pp.index:grp[j]=f"{cid}:{pp.loc[b]}";candmask[j]=True
            axs[1].scatter(co[~candmask,0],co[~candmask,1],s=4,c="#C7CFCC",alpha=.35,rasterized=True)
            for ci,g in enumerate(sorted(set(grp[candmask]))):
                m=grp==g;axs[1].scatter(co[m,0],co[m,1],s=9,color=COLORS[ci%len(COLORS)],label=g,rasterized=True)
            axs[1].set_title(f"{fold} post-hoc candidate membership; seed 20020130")
            for ax in axs:ax.set_xticks([]);ax.set_yticks([]);ax.legend(frameon=False,fontsize=6,loc="best")
            fig.tight_layout();fig.savefig(figures/f"{fold}_FullB2_candidate_UMAP.png",dpi=600);fig.savefig(figures/f"{fold}_FullB2_candidate_UMAP.pdf");plt.close(fig)
    except Exception as e:
        (out/"umap_status.json").write_text(json.dumps({"status":"UNAVAILABLE","error":str(e)},indent=2))
    # candidate-level plots and summary.
    if len(summary):
        fig,ax=plt.subplots(figsize=(11,4)); grouped=summary.groupby(["candidate_id","contrast"]).agg(n_sig=("within_FDR_DEG","sum"),n_global=("global_FDR_DEG","sum")).reset_index()
        labelsx=[f"{r.candidate_id}\n{r.contrast.replace('_',' ')}" for r in grouped.itertuples()];xpos=np.arange(len(grouped))
        ax.bar(xpos,grouped.n_sig,color="#287C8E",label="Within-contrast FDR");ax.scatter(xpos,grouped.n_global,color="#D28E5B",label="Global family FDR",zorder=3)
        ax.set_xticks(xpos,labelsx,rotation=75,ha="right",fontsize=6);ax.set_ylabel("Number of DEG");ax.legend(frameon=False);fig.tight_layout()
        fig.savefig(figures/"all_candidates_DEG_counts.png",dpi=600);fig.savefig(figures/"all_candidates_DEG_counts.pdf");plt.close(fig)
    if stability:
        st=pd.DataFrame(stability);fig,ax=plt.subplots(figsize=(7,4));ax.bar(st.candidate_id+"\n"+st.seed_a.astype(str).str[-2:]+"-"+st.seed_b.astype(str).str[-2:],st.partition_ARI_common,color="#287C8E")
        ax.axhline(0,color="#555",lw=.7);ax.set_ylabel("Partition ARI on shared candidate barcodes");ax.tick_params(axis="x",labelrotation=75,labelsize=7);fig.tight_layout()
        fig.savefig(figures/"candidate_seed_stability.png",dpi=600);fig.savefig(figures/"candidate_seed_stability.pdf");plt.close(fig)
    # Minimal transparent status outputs; detailed GSEA is generated only if gseapy imports.
    gsea_status={"status":"UNAVAILABLE","reason":"GSEApy unavailable; no preranked result fabricated","permutations":args.gsea_permutations}
    try:
        import gseapy as gp
        gsea_status={"status":"AVAILABLE","implementation":"gseapy.prerank","permutations":args.gsea_permutations}
        # Average signed ranks over model seeds; seeds are not biological replicates.
        if all_de:
            deall=pd.concat(all_de,ignore_index=True);gsea_rows=[]
            deall=deall[deall.contrast=="A_vs_B"]
            consensus=deall.groupby(["candidate_id","fold","gene_id","gene_symbol"],as_index=False).agg(
                wilcoxon_score_signed=("wilcoxon_score_signed","mean"),seeds_contributing=("seed","nunique"))
            for (cid,fold),d in consensus.groupby(["candidate_id","fold"]):
                for db,name in [("GO_BP","GO_Biological_Process_2025.gmt"),("Reactome","Reactome_2022.gmt")]:
                    path=args.gmt/name
                    if not path.exists():continue
                    pr=d[["gene_symbol","wilcoxon_score_signed"]].dropna().drop_duplicates("gene_symbol").sort_values("wilcoxon_score_signed",ascending=False)
                    res=gp.prerank(rnk=pr,gene_sets=str(path),permutation_num=args.gsea_permutations,min_size=10,max_size=500,
                                   outdir=None,seed=20020130,verbose=False)
                    table=res.res2d.copy();table["candidate_id"]=cid;table["fold"]=fold;table["seed_aggregation"]="mean_signed_Wilcoxon_across_available_model_seeds";table["contrast"]="A_vs_B_consensus";table["database"]=db;table["n_seeds_contributing"]=int(d.seeds_contributing.max());gsea_rows.append(table)
            if gsea_rows:
                gsea=pd.concat(gsea_rows,ignore_index=True)
                pcol="NOM p-val" if "NOM p-val" in gsea else "NOM p-val"
                if pcol in gsea:gsea["FDR_BH_global_all_candidate_gsea_tests"]=bh(pd.to_numeric(gsea[pcol],errors="coerce").fillna(1).to_numpy())
                gsea.to_csv(tables/"all_candidate_gsea_summary.csv",index=False)
            else:gsea_status["status"]="NO_RESULTS"
    except Exception as e:gsea_status={"status":"UNAVAILABLE","reason":str(e),"permutations":args.gsea_permutations}
    (out/"gsea_status.json").write_text(json.dumps(gsea_status,indent=2))
    if not (tables/"all_candidate_gsea_summary.csv").exists():pd.DataFrame(columns=["candidate_id","fold","seed","database","Term","NES","NOM p-val","FDR q-val","Lead_genes"]).to_csv(tables/"all_candidate_gsea_summary.csv",index=False)
    if True:
        evid=[]
        degsum=summary.groupby("candidate_id") if len(summary) else None
        das=pd.DataFrame(da_summaries)
        es=pd.DataFrame(edge_summaries)
        for r in inv:
            cid=r["candidate_id"];nwithin=int(degsum.get_group(cid).within_FDR_DEG.sum()) if degsum is not None and cid in degsum.groups else 0
            nd=int(das.loc[das.candidate_id==cid,"strict_DA_peaks"].sum()) if len(das) else 0
            tri=int(es.loc[es.candidate_id==cid,"RNA_DEG_ATAC_DA_source_edge_triples"].sum()) if len(es) else 0
            evid.append({"candidate_id":cid,"fold":r["fold"],"reference_subclass":r["reference_subclass"],
                "stability":"PASS" if (r["mean_partition_ARI"] is not None and pd.notna(r["mean_partition_ARI"]) and r["mean_partition_ARI"]>=.5) else ("PARTIAL" if r["recorded_or_rescanned_seeds"].count(";")>=1 else "FAIL"),
                "RNA_only":"UNAVAILABLE","full_gene_DEG":"PARTIAL" if nwithin else "FAIL","GSEA":"UNAVAILABLE" if gsea_status["status"]=="UNAVAILABLE" else "PARTIAL",
                "ATAC":"UNAVAILABLE" if fold_data_status.get(r["fold"],{}).get("status") in {"BLOCKED","RNA_ONLY"} else ("PASS" if nd else "NO_STRICT_DA"),"source_supported_triple":"PASS" if tri else "UNAVAILABLE",
                "confounders":"PARTIAL","biology":"PARTIAL","cell_count":"PASS" if min([int(v.split(":")[1]) for v in (r["candidate_A_n_by_seed"]+";"+r["candidate_B_n_by_seed"]).split(";") if ":" in v] or [0])>=20 else "PARTIAL",
                "within_contrast_DEG_count":nwithin,"strict_DA_peak_count":nd,"triple_link_count":tri})
        pd.DataFrame(evid).to_csv(out/"candidate_evidence_matrix.csv",index=False)
    readme={"input_candidate_rows":len(raw_splits),"candidate_inventory_rows":len(inv),"reconstruction_audit_rows":len(reconstruction),
        "valid_candidate_seed_units":sum(len(x) for x in candidates.values()),"folds":FOLDS,"seeds":SEEDS,
        "fold_data_status":fold_data_status,"paired_atac_available_folds":[f for f in FOLDS if fold_data_status.get(f,{}).get("status")=="PASS"],"rna_only_baseline":"unavailable strict Task99 match not located",
        "gsea":gsea_status,"candidate_claim":"all post-hoc label-assisted candidate evidence is exploratory; no independent cohort validation"}
    (out/"analysis_summary.json").write_text(json.dumps(readme,indent=2))
    summary.to_csv(out/"all_candidate_deg_summary.csv",index=False)
    pd.read_csv(tables/"all_candidate_gsea_summary.csv").to_csv(out/"all_candidate_gsea_summary.csv",index=False)
    pd.read_csv(tables/"atac_evidence_summary.csv").to_csv(out/"atac_evidence_summary.csv",index=False)
    print(json.dumps(readme,indent=2),flush=True)


if __name__=="__main__":main()

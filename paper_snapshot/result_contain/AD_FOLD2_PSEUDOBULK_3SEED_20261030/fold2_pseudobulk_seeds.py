"""Separate exploratory raw-count pseudobulk analysis; frozen rev04 is read-only."""
import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['OMP_NUM_THREADS'] = '4'
os.environ['OPENBLAS_NUM_THREADS'] = '4'
import sys
import json
import hashlib
import warnings
from pathlib import Path
from datetime import datetime, timezone
import importlib.metadata
import numpy as np
import pandas as pd
from scipy import sparse
from scipy.stats import ttest_ind, hypergeom
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/u202522802045/share/zpb')
BENCH = ROOT / 'result_contain/AD_FOURFOLD_P2_LINEAR_GSE288063_20261028_rev04'
OUT = ROOT / 'result_contain/AD_FOLD2_PSEUDOBULK_3SEED_20261030'
sys.path.insert(0, str(BENCH / 'source'))
import ad_runtime as a
import ad_rna as r
SEEDS = [20020130, 20020131, 20020132]
TYPES = list(a.TYPES)
ANALYSES = ['ALL_CELLS'] + TYPES
SETS = ROOT / 'result_contain/AD_Dmode_4Cohort_20261024/source_gene_sets'
LIBRARIES = {'GO_BP': SETS / 'GO_Biological_Process_2025.gmt', 'Reactome': SETS / 'Reactome_2022.gmt'}

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()

def js(path, obj):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n')

def bh(p):
    p = np.nan_to_num(np.asarray(p, float), nan=1., posinf=1., neginf=1.)
    ix = np.argsort(p, kind='stable')
    out = np.empty_like(p)
    if len(p):
        out[ix] = np.minimum(1., np.minimum.accumulate((p[ix] * len(p) / np.arange(1, len(p)+1))[::-1])[::-1])
    return out

def md(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] + ['| ' + ' | '.join(str(x).replace('|','/') for x in row) + ' |' for row in rows])

def aggregate(meta, identities, units):
    typ = a.author_types('GSE268609', meta)
    ui = identities.unit_id.map({u:i for i,u in enumerate(units)}).to_numpy(int)
    ti = np.array([TYPES.index(t) if t in TYPES else -1 for t in typ])
    arr = np.zeros((len(ANALYSES), len(units), len(r.GENES)), np.int64)
    coverage = np.zeros((len(ANALYSES), len(units)), np.int64)
    read_total = np.zeros(len(r.GENES), np.int64)
    n_read = 0
    for first, x in r.reader('GSE268609').blocks():
        assert first == n_read
        assert x.shape[1] == len(r.GENES) and np.issubdtype(x.dtype, np.integer)
        assert (x.data >= 0).all()
        x = x.astype(np.int64)
        read_total += np.asarray(x.sum(0)).ravel()
        u = ui[first:first+x.shape[0]]
        t = ti[first:first+x.shape[0]]
        for j, _ in enumerate(ANALYSES):
            mask = np.ones(len(u), bool) if j == 0 else t == j-1
            if not mask.any():
                continue
            uu = u[mask]
            s = sparse.csr_matrix((np.ones(len(uu), np.int64), (np.arange(len(uu)), uu)), shape=(len(uu), len(units)))
            arr[j] += (s.T @ x[mask]).toarray()
            coverage[j] += np.bincount(uu, minlength=len(units))
        n_read += x.shape[0]
        if n_read % 5120 == 0:
            print('COUNTS_READ', n_read, '/', len(meta), flush=True)
    assert n_read == len(meta) == 51246
    assert np.array_equal(arr[0].sum(0), read_total)
    assert np.array_equal(coverage[0], np.bincount(ui, minlength=len(units)))
    assert np.all(arr[1:].sum(0) <= arr[0])
    pd.DataFrame({'unit_id': np.tile(units, len(ANALYSES)), 'cell_type': np.repeat(ANALYSES, len(units)),
                  'n_cells': coverage.ravel(), 'library_size': arr.sum(2).ravel(),
                  'eligible_20cells': (coverage >= 20).ravel()}).to_csv(OUT/'DONOR_CELLTYPE_COVERAGE.csv', index=False)
    np.savez_compressed(OUT/'PSEUDOBULK_RAW_COUNTS.npz', counts=arr, n_cells=coverage,
                        cell_types=np.array(ANALYSES), units=np.array(units), genes=r.GENES.astype(str))
    js(OUT/'AGGREGATION_CHECK.json', dict(status='PASS', cells_read=n_read, gene_count=len(r.GENES),
         unit_count=len(units), integer_counts=True, gene_wise_count_conservation=True,
         cell_count_conservation=True, five_types_subset_of_all=True,
         counts_total=int(read_total.sum()), unit_type='technical_sample_group'))
    return arr, coverage

def load_sets():
    libraries = {}
    for lib, path in LIBRARIES.items():
        entries = [line.split('\t') for line in path.read_text().splitlines()]
        libraries[lib] = [(z[0], set(z[2:])) for z in entries if len(z) >= 3]
    return libraries

DEG_COLS = ['gene_id','symbol','cell_type','statistic','log2FC','pvalue','FDR','n_AD','n_Control']
ORA_COLS = ['direction','library','term','overlap','term_size','query_size','background_size','pvalue','FDR','genes']

def compare(scope, labels, min_units, arr, coverage, units, libraries):
    dest = OUT / scope
    dest.mkdir(parents=True, exist_ok=True)
    labels = np.asarray(labels, int)
    assignment = pd.DataFrame({'unit_id': units, 'label': labels, 'unit_type': 'technical_sample_group'})
    assignment.to_csv(dest/'GROUP_ASSIGNMENTS.csv', index=False)
    summaries, all_sig, deg_sig = [], [], []
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for j, (celltype, ax) in enumerate(zip(ANALYSES, axes.ravel())):
        d = dest / celltype
        d.mkdir(exist_ok=True)
        ok = (coverage[j] >= 20) & (arr[j].sum(1) > 0)
        x, lab = arr[j, ok], labels[ok]
        positive, negative = lab == 1, lab == 0
        n1, n0 = int(positive.sum()), int(negative.sum())
        membership = assignment.copy()
        membership['n_cells'] = coverage[j]
        membership['eligible'] = ok
        membership['reason'] = np.where(ok, 'INCLUDED', 'FEWER_THAN_20_CELLS_OR_ZERO_LIBRARY')
        membership.to_csv(d/'UNIT_INCLUSION.csv', index=False)
        base = dict(scope=scope, fold='fold2', target='GSE268609', method='Full_B2_frozen_prediction' if scope.startswith('seed_') else 'real_AD_HA_labels',
                    seed=int(scope.split('_')[1]) if scope.startswith('seed_') else 'SHARED_ACROSS_SEEDS', cell_type=celltype,
                    n_AD=n1, n_Control=n0, unit_type='technical_sample_group', significant_genes=None,
                    GO_AD_up=None, GO_Control_up=None, Reactome_AD_up=None, Reactome_Control_up=None)
        if min(n1,n0) < min_units:
            row = dict(**base, status='NOT_EVALUABLE', reason=f'at least {min_units} eligible technical groups required per label')
            pd.DataFrame(columns=DEG_COLS).to_csv(d/'DEG_significant.csv', index=False)
            pd.DataFrame(columns=ORA_COLS).to_csv(d/'ENRICHMENT_significant.csv', index=False)
            ax.text(.5,.5,f'{celltype}\nNOT EVALUABLE\n{n1} vs {n0} eligible units', ha='center',va='center',transform=ax.transAxes)
            ax.set_axis_off()
            summaries.append(row)
            js(d/'COMPLETE.json', row)
            continue
        cpm = x / x.sum(1, keepdims=True) * 1e6
        log = np.log1p(cpm)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', RuntimeWarning)
            test = ttest_ind(log[positive], log[negative], axis=0, equal_var=False, nan_policy='omit')
        p = np.nan_to_num(test.pvalue, nan=1., posinf=1., neginf=1.)
        fc = np.log2((cpm[positive].mean(0)+1)/(cpm[negative].mean(0)+1))
        tab = pd.DataFrame(dict(gene_id=r.GENES, symbol=r.FEATURES.gene_name.fillna('').astype(str),
                               cell_type=celltype, statistic=test.statistic, log2FC=fc, pvalue=p,
                               FDR=bh(p), n_AD=n1, n_Control=n0))
        tab = tab.sort_values(['FDR','pvalue','gene_id'])
        sig = tab[(tab.FDR < .05) & (tab.log2FC.abs() >= .25)].copy()
        tab.to_csv(d/'DEG_all_tested.csv.gz', index=False)
        tab.head(200).to_csv(d/'DEG_top200_EXPLORATORY.csv', index=False)
        sig.to_csv(d/'DEG_significant.csv', index=False)
        deg_sig.append(sig.assign(scope=scope))
        rows, enrich_status = [], []
        background = set(tab.symbol) - {'','nan'}
        for direction, mask in [('AD_up',sig.log2FC>0),('Control_up',sig.log2FC<0)]:
            query = set(sig.loc[mask].head(200).symbol) - {'','nan'}
            for lib, entries in libraries.items():
                library_genes = set().union(*(members for _, members in entries))
                bg = background & library_genes
                q = query & bg
                stat = dict(direction=direction, library=lib, background_size=len(bg), query_size=len(q),
                            status='READY' if q else 'NO_ELIGIBLE_DEG_EMPTY_QUERY')
                enrich_status.append(stat)
                if not q:
                    base[('GO_' if lib=='GO_BP' else 'Reactome_') + direction] = 0
                    continue
                out = []
                for term, members0 in entries:
                    members = members0 & bg
                    if not 10 <= len(members) <= 2000:
                        continue
                    overlap = sorted(q & members)
                    out.append(dict(direction=direction, library=lib, term=term, overlap=len(overlap),
                                    term_size=len(members), query_size=len(q), background_size=len(bg),
                                    pvalue=float(hypergeom.sf(len(overlap)-1,len(bg),len(members),len(q))), genes=';'.join(overlap)))
                out = pd.DataFrame(out)
                out['FDR'] = bh(out.pvalue)
                out = out.sort_values(['FDR','pvalue','term'])
                out.to_csv(d/(direction+'_'+lib+'_all.csv.gz'), index=False)
                sig_terms = out[(out.FDR<.05)&(out.overlap>=3)]
                rows.append(sig_terms)
                base[('GO_' if lib=='GO_BP' else 'Reactome_') + direction] = len(sig_terms)
        enriched = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=ORA_COLS)
        enriched.to_csv(d/'ENRICHMENT_significant.csv', index=False)
        all_sig.append(enriched.assign(scope=scope, cell_type=celltype))
        base['significant_genes'] = len(sig)
        row = dict(**base, status='COMPLETE_EXPLORATORY', reason='Welch fallback; biological independence unresolved')
        summaries.append(row)
        js(d/'ENRICHMENT_STATUS.json', enrich_status)
        js(d/'COMPLETE.json', row)
        colors = np.where((tab.FDR<.05)&(tab.log2FC.abs()>=.25), '#b34f60', '#9a9fa5')
        ax.scatter(tab.log2FC, -np.log10(np.maximum(tab.FDR,1e-300)), s=2, c=colors, alpha=.6)
        ax.axhline(-np.log10(.05), lw=.6, ls='--', c='black')
        ax.axvline(-.25,lw=.5,ls=':',c='black'); ax.axvline(.25,lw=.5,ls=':',c='black')
        ax.set(title=f'{celltype} ({n1} vs {n0})', xlabel='log2FC: positive / negative', ylabel='-log10 BH FDR')
        print('ANALYSIS',scope,celltype,'DEG',len(sig),'GO',row['GO_AD_up'],row['GO_Control_up'], flush=True)
    fig.suptitle('GSE268609 exploratory technical-group pseudobulk | '+scope, fontsize=12)
    fig.tight_layout()
    fig.savefig(dest/'PSEUDOBULK_VOLCANO.png', dpi=200)
    fig.savefig(dest/'PSEUDOBULK_VOLCANO.pdf')
    plt.close(fig)
    summary = pd.DataFrame(summaries)
    summary.to_csv(dest/'SUMMARY.csv', index=False)
    (pd.concat(all_sig,ignore_index=True) if all_sig else pd.DataFrame(columns=ORA_COLS)).to_csv(dest/'ALL_SIGNIFICANT_ENRICHMENT.csv',index=False)
    (pd.concat(deg_sig,ignore_index=True) if deg_sig else pd.DataFrame(columns=DEG_COLS)).to_csv(dest/'ALL_SIGNIFICANT_DEG.csv',index=False)
    return summary

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    if (OUT/'RUN_COMPLETE.json').exists():
        raise RuntimeError('Completed analysis exists; refusing overwrite')
    protocol_path = BENCH/'frozen_protocol.json'
    original_protocol_path = OUT/'ORIGINAL_REV04_PROTOCOL_SNAPSHOT.json'
    assert sha(original_protocol_path) == a.EXPECTED_PROTOCOL
    original = json.loads(original_protocol_path.read_text())
    prot = json.loads(protocol_path.read_text())
    changed = sorted(k for k in set(original)|set(prot) if original.get(k)!=prot.get(k))
    assert changed in [[], ['supersedes']], changed
    if changed:
        assert prot['supersedes'] == original['supersedes'].replace('GSE272082','retired_legacy_cohort')
    js(OUT/'PROTOCOL_COMPATIBILITY_AUDIT.json', dict(status='PASS', changed_keys=changed,
       original_sha256=sha(original_protocol_path), current_sha256=sha(protocol_path),
       scientific_fields_identical=True, cause='earlier dataset-retirement prose-only edit; legacy runtime hash stale',
       canonical_files_modified=False))
    assert prot['folds']['fold2']['target']=='GSE268609'
    assert list(prot['seeds'])==SEEDS
    meta = a.cells('GSE268609')
    ids = a.identities('GSE268609', meta)
    roster = a.roster('GSE268609').sort_values('unit_id').reset_index(drop=True)
    units = roster.unit_id.tolist()
    assert len(units)==13 and int(roster.binary_label.sum())==8
    assert meta.count_file.nunique()==1 and not meta.count_row.duplicated().any()
    roster.to_csv(OUT/'UNIT_ROSTER.csv',index=False)
    r.FEATURES.to_csv(OUT/'GENE_DICTIONARY.csv',index=False)
    protected = [BENCH/'frozen_protocol.json', BENCH/'metrics_per_fold_seed.csv', BENCH/'predictions_all.csv']
    predictions = {}
    for seed in SEEDS:
        p = BENCH/'evaluation/fold2'/f'seed_{seed}'/'Full_B2/predictions_blind.csv'
        completion = json.loads((p.parent/'COMPLETE.json').read_text())
        assert sha(p)==completion['blind_predictions_sha256']
        pred = pd.read_csv(p,dtype={'unit_id':str})
        assert len(pred)==13 and pred.unit_id.is_unique and set(pred.unit_id)==set(units)
        assert np.array_equal(pred.prediction.to_numpy(), (pred.probability_AD.to_numpy()>=.5).astype(int))
        predictions[seed] = pred.set_index('unit_id').loc[units,'prediction'].to_numpy(int)
        protected += [p, BENCH/'models/fold2'/f'seed_{seed}'/'student/best.pt']
    before = {str(p):sha(p) for p in protected}
    config = dict(fold='fold2', target='GSE268609', seeds=SEEDS, method='Full_B2',
                  raw_counts='sum integer counts per sample_group_id x author broad type; ALL_CELLS is additional context',
                  biological_unit='13 technical sample groups, not 13 verified independent patients',
                  cell_types=TYPES, min_cells_per_unit_type=20, min_units_per_prediction_label=2,
                  min_units_per_true_label=3, tested_genes='all 36601 genes; undefined/constant tests p=1; BH within comparison',
                  normalization='library-size CPM then natural log1p', test='two-sided unequal-variance Welch fallback',
                  effect='log2 ratio of group mean CPM plus pseudocount 1', DEG='FDR<0.05 and abs(log2FC)>=0.25',
                  ORA='hypergeometric; tested symbol background intersected with each GMT; term size10..2000; BH within each type/direction/library',
                  ORA_query='at most200 significant genes per direction by FDR,pvalue,gene_id; term FDR<.05 overlap>=3',
                  true_label_reference='computed once; linked by all three seeds, not independent seed results',
                  no_gpu=True, no_retraining=True, exploratory=True, timestamp_utc=datetime.now(timezone.utc).isoformat())
    js(OUT/'FROZEN_ANALYSIS_CONFIG.json',config)
    js(OUT/'PROTECTED_INPUT_HASHES_BEFORE.json',before)
    source_paths = [Path(__file__),original_protocol_path,BENCH/'source/ad_runtime.py',BENCH/'source/ad_rna.py',Path(r.LEGACY),
                    BENCH/'preprocessing/cohorts/GSE268609/cells.csv.gz',BENCH/'preprocessing/cohorts/GSE268609/evaluation_units.csv',
                    Path(meta.count_file.iloc[0])] + list(LIBRARIES.values())
    js(OUT/'SOURCE_MANIFEST.json',{str(p):dict(bytes=p.stat().st_size,sha256=sha(p)) for p in source_paths})
    js(OUT/'SOFTWARE_VERSIONS.json',dict(python=sys.version,executable=sys.executable,
           packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','scipy','h5py','matplotlib']}))
    print('CONFIG_FROZEN', OUT, flush=True)
    arrays,coverage = aggregate(meta,ids,units)
    libraries=load_sets()
    summaries = [compare('true_label_reference',roster.binary_label.to_numpy(int),3,arrays,coverage,units,libraries)]
    for seed in SEEDS:
        scope=f'seed_{seed}'
        summaries.append(compare(scope,predictions[seed],2,arrays,coverage,units,libraries))
        js(OUT/scope/'TRUE_LABEL_REFERENCE.json',dict(reference='../true_label_reference', shared_across_seeds=True,
             note='true-label pseudobulk does not depend on encoder seed; do not count as three replications'))
    summary=pd.concat(summaries,ignore_index=True)
    summary.to_csv(OUT/'ALL_ANALYSES_SUMMARY.csv',index=False)
    terms=[]
    for p in sorted(OUT.glob('*/ALL_SIGNIFICANT_ENRICHMENT.csv')):
        frame=pd.read_csv(p)
        if len(frame):terms.append(frame)
    allt=pd.concat(terms,ignore_index=True) if terms else pd.DataFrame(columns=ORA_COLS+['scope','cell_type'])
    allt.to_csv(OUT/'ALL_SIGNIFICANT_GO_REACTOME.csv',index=False)
    after={str(p):sha(p) for p in protected}
    assert before==after
    js(OUT/'PROTECTED_INPUT_HASHES_AFTER.json',after)
    display=summary.fillna('NA')
    rows=display[['scope','cell_type','n_AD','n_Control','significant_genes','GO_AD_up','GO_Control_up','Reactome_AD_up','Reactome_Control_up','status']].values.tolist()
    report=['# Fold2 / GSE268609：三个 Seed 的原始 counts pseudobulk 与 GO', '',
            '日期：2026-10-30。本报告为新增探索性分析；正式 rev04 checkpoint、Probe、概率与结果保持不变。', '',
            '**统计单位是13个技术组（真实标签8 AD、5 HA），不是已验证的13名独立患者。**', '',
            '按技术组×五类 P2 分别汇总原始整数 RNA counts，另附 ALL_CELLS 汇总。每类型每组至少20个细胞。', '',
            '每个 Seed 使用相应冻结 Full B2 的预测AD/Control分组，每组至少2个合格技术组；不按细胞数重复标签。真实AD/HA参考每组至少3个合格技术组，只计算一次，供三个Seed共同引用。', '',
            '与 Fold1 一样采用 library-size CPM → log1p → Welch → BH 的探索性替代方法；并非 edgeR/DESeq2。原始 counts 的和、基因顺序及细胞计数均已核验。', '',
            'DEG：FDR<0.05且|log2FC|≥0.25。ORA沿用既有GO BP2025和Reactome2022，背景为实际检验且可映射至各库的基因，按方向最多200个显著基因，术语大小10–2000，FDR<0.05且overlap≥3。缺少合格DEG记为空查询；缺少独立技术组记NOT_EVALUABLE，不冒称零差异。', '',
            md(['分组/Seed','细胞类型','AD/正类','HA/负类','DEG','GO AD-up','GO Control-up','Reactome AD-up','Reactome Control-up','状态'],rows), '',
            '## 如何解释三个 Seed', '',
            '原始pseudobulk counts、作者细胞类型与真实AD/HA标签跨Seed相同；只有预测分组随冻结模型Seed变化。真实标签参考不构成三次独立验证。', '',
            '预测分组源于RNA，随后在RNA上做差异，结果存在选择依赖；技术组之间生物学独立性尚未解决，不能作独立患者或因果机制推断。FDR按各比较校正，未对所有Seed/类型/库做额外全局校正。', '',
            '## 文件', '',
            '- ALL_ANALYSES_SUMMARY.csv：24项比较（真实标签6项 + 3 Seed×6项）。',
            '- ALL_SIGNIFICANT_GO_REACTOME.csv：全部显著通路、方向、基因及所属Seed/类型。',
            '- seed_*/：该Seed的分组名单、逐类型DEG和GO/Reactome、火山图及真实标签参考指针。',
            '- true_label_reference/：真实标签的共同参考分析。',
            '- DONOR_CELLTYPE_COVERAGE.csv：逐技术组逐类型覆盖；FROZEN_ANALYSIS_CONFIG.json：预先冻结设置。',
            '- PSEUDOBULK_RAW_COUNTS.npz及完整检验表只留服务器；本地只同步必要结果、图和核验文件。',
            '- PROTECTED_INPUT_HASHES_BEFORE/AFTER.json：原有正式文件未变的校验。', '']
    if len(allt):
        report += ['## 显著通路（每个比较每库按FDR前5）','']
        for (scope,ct,lib),g in allt.groupby(['scope','cell_type','library']):
            g=g.sort_values(['FDR','pvalue']).head(5)
            report += [f'### {scope} / {ct} / {lib}','',md(['方向','术语','FDR'],g[['direction','term','FDR']].values.tolist()),'']
    else:
        report += ['本次没有满足既定阈值的显著GO/Reactome；不以Top排名冒充显著结果。','']
    (OUT/'FOLD2_PSEUDOBULK_REPORT_ZH.md').write_text('\n'.join(report))
    js(OUT/'RUN_COMPLETE.json',dict(status='COMPLETE',fold='fold2',seeds=SEEDS,analyses=len(summary),
         complete=int(summary.status.eq('COMPLETE_EXPLORATORY').sum()),not_evaluable=int(summary.status.eq('NOT_EVALUABLE').sum()),
         significant_enrichment_rows=len(allt),protected_hashes_unchanged=True,raw_count_sum_verified=True,
         no_gpu=True,no_retraining=True,technical_groups_not_independent_donors=True))
    exports=[]
    for p in sorted(OUT.rglob('*')):
        if not p.is_file() or p.suffix=='.npz' or p.name.endswith('.csv.gz') or p.name in ['GENE_DICTIONARY.csv']:
            continue
        exports.append(dict(path=str(p.relative_to(OUT)),bytes=p.stat().st_size,sha256=sha(p)))
    js(OUT/'EXPORT_MANIFEST.json',exports)
    print('FOLD2_PSEUDOBULK_COMPLETE',OUT,flush=True)

if __name__=='__main__':
    main()

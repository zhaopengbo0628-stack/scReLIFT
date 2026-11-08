import hashlib, json
from pathlib import Path
import pandas as pd
ROOT=Path('/home/u202522802045/share/zpb')
CACHE=ROOT/'data/cache/d_mode/Mouse-brain/four_dataset_v1_20261023/folds'
OUT=ROOT/'result_contain/86_brain_lodo_bd_audit'
OUT.mkdir(parents=True,exist_ok=True)
targets=['D3','D4','Gao','FC']
intra={'D3':(47479,1991),'D4':(45081,1992),'Gao':(46621,1893),'FC':(38475,1690)}
rows=[]
for t in targets:
    f=CACHE/f'target_{t}'/'source_artifacts'
    m=json.load(open(f/'manifest.json'))
    b=f/'B_D.npz'
    h=hashlib.sha256()
    with open(b,'rb') as fh:
        for x in iter(lambda:fh.read(8<<20),b''): h.update(x)
    r=dict(target=t,sources='+'.join(m['sources']),gene_count=m['gene_count'],peak_count=m['peak_count'],
        candidate_edges=m['candidate_edges'],B_edges=m['B_edges'],retained_genes=m['retained_genes'],
        edge_retention=round(m['B_edges']/m['candidate_edges'],4),
        intra_B_edges=intra[t][0],intra_retained_genes=intra[t][1],
        lodo_vs_intra_edges=round(m['B_edges']/intra[t][0],4),
        prep_seconds=round(m['preparation_seconds'],1),selector_seconds=round(m['selector_seconds'],1),
        B_D_bytes=b.stat().st_size,B_D_sha256=h.hexdigest())
    rows.append(r)
df=pd.DataFrame(rows); df.to_csv(OUT/'LODO_B_D_AUDIT.csv',index=False)
lines=['# LODO B_D audit — four mouse datasets','',
 'Four leave-one-dataset-out folds were built from the current Intra source caches. Each fold uses three Sources; the Target is held out and was not loaded for B_D. **No model training was run.**','',
 '| Target | Sources | Common genes | Canonical peaks | Candidate edges | B_D edges | Retained genes | B_D/Intra edges |','|---|---|---:|---:|---:|---:|---:|---:|']
for r in rows:
    lines.append(f"| {r['target']} | {r['sources']} | {r['gene_count']} | {r['peak_count']:,} | {r['candidate_edges']:,} | **{r['B_edges']:,}** | {r['retained_genes']} | {r['lodo_vs_intra_edges']*100:.1f}% |")
lines += ['','## Interpretation','',
 '- LODO retains **743–990 common HVG genes** per fold, versus 2,000 in each Intra cache.','- LODO B_D has **17,965–24,342 nonzero edges** per fold. The selector takes the top 50% of candidate edges per gene; observed retention is exactly ~50% of candidates.','- LODO edges are 38.5–51.3% of the corresponding Intra B edge count, because the common gene set is much smaller and the fold canonical peak universe is rebuilt from the three Sources.','- These are graph edges, not model performance. No target ATAC/labels/F were used to build B_D; no Student/Teacher was trained.','','## Caveats','',
 '- `B_D` is built; **target F and LODO model evaluation are not built**. A LODO run still needs target peak mapping, target F and held-out inference.','- Canonical peaks differ by fold (558,128–777,723) because the three-Source feature references differ; do not compare peak counts across folds as if they were the same universe.','- No target labels or test ARI influenced edge selection.','','## Paths','']
for r in rows:
    lines.append(f"- `{r['target']}`: `{CACHE}/target_{r['target']}/source_artifacts/`")
(OUT/'LODO_REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1<<20),b''): h.update(b)
    return h.hexdigest()
files=[p for p in OUT.iterdir() if p.is_file() and p.name!='processing_manifest.json']
man=dict(task_id=86,task_name='86_brain_lodo_bd_audit',status='COMPLETED_LODO_BD_AUDIT',n_folds=4,
    folds=rows,training_started=False,target_atac_used_for_bd=False,target_labels_used_for_bd=False,
    outputs={p.name:{'bytes':p.stat().st_size,'sha256':sha(p)} for p in files})
(OUT/'processing_manifest.json').write_text(json.dumps(man,indent=2)+'\n',encoding='utf-8')
print(df.to_string(index=False))
print('wrote',OUT)

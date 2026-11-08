"""Summarize only completed frozen evaluations; never select by Target score."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def write_csv(path, rows):
    if not rows: return
    with path.open('w', newline='', encoding='utf-8') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def summarize(result_root, cache_root):
    out=Path(result_root);destination=out/'metrics';destination.mkdir(parents=True,exist_ok=True)
    rows=[];efficiency=[];graphs=[]
    for path in sorted((out/'models').glob('D*/seed_*/evaluation/*/manifest.json')):
        record=json.loads(path.read_text())
        if record['status']!='COMPLETE': continue
        rows.append(dict(source=record['source'],target=record['target'],seed=record['seed'],
                         **{k:record['metrics'][k] for k in ['ari','nmi','ami','cluster_acc','accuracy',
                                                           'balanced_accuracy','macro_f1','weighted_f1','knn_purity_at_15']},
                         gene_hit_fraction=record['gene_hit_fraction'],manifest=str(path)))
        training=json.loads((path.parents[2]/'training_manifest.json').read_text())
        resources=[v['resources'] for v in training['stages'].values()]
        efficiency.append(dict(source=record['source'],target=record['target'],seed=record['seed'],
            training_seconds=sum(v['training_seconds'] for v in resources),
            trainable_parameters=sum(v['parameters'] for v in resources),
            peak_allocated_bytes=max(v['peak_allocated_bytes'] for v in resources),
            peak_reserved_bytes=max(v['peak_reserved_bytes'] for v in resources),
            inference_seconds=record['inference_seconds'],cells_per_second=record['cells_per_second'],
            gpu=training['environment']['gpu']))
    summaries=[]
    for source in ['D1','D2','D3','D4']:
        cache=Path(cache_root)/source/'manifest.json'
        if cache.exists():
            m=json.loads(cache.read_text())
            graphs.append(dict(source=source,hvg_count=m['hvg_count'],candidate_edges=m['candidate_edges'],
                retained_edges=m['effective_edges'],supported_genes=m['supported_genes'],
                nondegenerate_F_genes=m['field']['nondegenerate_genes'],selector_seconds=m['selector_seconds'],
                field_seconds=m['field_seconds'],preprocessing_seconds=m['preprocessing_seconds'],identity=m['identity']))
        for target in ['D1','D2','D3','D4']:
            selected=[r for r in rows if r['source']==source and r['target']==target]
            seeds=[r['seed'] for r in selected]
            if len(set(seeds))!=len(seeds): raise ValueError('Duplicate evaluation seed')
            row=dict(source=source,target=target,completed=len(selected),expected=5,run_tier='pre_final')
            for metric in ['ari','nmi']:
                values=[r[metric] for r in selected]
                row[metric+'_mean']=float(np.mean(values)) if values else None
                row[metric+'_sample_sd']=float(np.std(values,ddof=1)) if len(values)>1 else None
            summaries.append(row)
    write_csv(destination/'per_seed.csv',rows)
    write_csv(destination/'summary_mean_std.csv',summaries)
    write_csv(destination/'efficiency.csv',efficiency)
    write_csv(destination/'regulatory_graphs.csv',graphs)
    completed_models=list((out/'models').glob('D*/seed_*/manifest.json'))
    status=dict(status='COMPLETE' if len(rows)==80 and len(completed_models)==20 else 'PARTIAL',
                completed_evaluations=len(rows),expected_evaluations=80,completed_models=len(completed_models),
                expected_models=20,run_tier='pre_final',sample_sd_ddof=1,
                interpretation='zero-shot frozen Student clustering; known K and evaluation-only scaling; not supervised label transfer')
    (destination/'summary.json').write_text(json.dumps(dict(**status,results=summaries),indent=2))
    lines=['# S-mode Full B2 — pre_final',f'Models {len(completed_models)}/20; evaluations {len(rows)}/80.',
           '| Source → Target | Seeds | ARI mean ± sample SD | NMI mean ± sample SD |',
           '|---|---:|---|---|']
    for row in summaries:
        def display(metric):
            mean,sd=row[metric+'_mean'],row[metric+'_sample_sd']
            return 'NA' if mean is None else f'{mean:.6f} ± '+('NA' if sd is None else f'{sd:.6f}')
        lines.append(f'| {row["source"]} → {row["target"]} | {row["completed"]}/5 | {display("ari")} | {display("nmi")} |')
    (destination/'summary.md').write_text('\n'.join(lines)+'\n')
    return status


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    args=parser.parse_args();config=json.loads(args.config.read_text())
    print(json.dumps(summarize(config['result_root'],config['cache_root'])))

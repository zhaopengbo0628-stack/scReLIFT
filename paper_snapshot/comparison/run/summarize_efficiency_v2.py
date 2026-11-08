"""Combined scores and efficiency with complete per-metric coverage checks."""
import argparse
import csv
from pathlib import Path

import numpy as np

from pbmc_intra_common import read, write, sha
from efficiency_v2 import PROTOCOL

NAMES=dict(multigrate='Multigrate',cobolt='Cobolt',palette='Palette-S',eclare='ECLARE-S',scmomer='scMomer-S',b2='Full B2-S')
SCORES={'ARI':'ari','NMI':'nmi','ACC':'cluster_acc','Macro-F1':'macro_f1',
        'Balanced ACC':'balanced_accuracy','Purity@15':'knn_purity_at_15','Silhouette':'silhouette'}
KEYS=['D1','D2','D3','D4']


def table(path, rows):
    write(path.with_suffix('.json'),rows)
    with path.with_suffix('.csv').open('w',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    columns=list(rows[0])
    text='5-seed pre_final; inter. Scores/inference: 12 pairs per seed; training/end-to-end: 4 Sources per seed. NA includes incomplete coverage.\n\n'
    text+='| '+' | '.join(columns)+' |\n|'+'---|'*len(columns)+'\n'
    for row in rows:
        text+='| '+' | '.join('NA' if row[c] is None else f'{row[c]:.6f}' if isinstance(row[c],float) else str(row[c]) for c in columns)+' |\n'
    path.with_suffix('.md').write_text(text)


def average(values, expected):
    return float(np.mean(values)) if len(values)==expected and all(isinstance(v,(int,float)) for v in values) else None


def summarize(roots, output):
    columns=read(PROTOCOL)['display_columns']
    means=[]; deviations=[]; per_seed=[]; coverage=[]
    for root in roots:
        request=read(root/'efficiency_request.json')
        assert request['protocol_sha256']==sha(PROTOCOL)
        method=request['method']
        workflows=[read(p) for p in sorted((root/'workflows').glob('D*/seed_*/efficiency.json'))]
        units=[read(p) for p in sorted((root/'evaluations').glob('D*/seed_*/D*/manifest.json'))]
        for record in workflows+units: assert record['protocol_sha256']==sha(PROTOCOL)
        seed_rows=[]
        for seed in request['seeds']:
            selected=[u for u in units if u['seed']==seed]
            fits=[w for w in workflows if w['seed']==seed]
            pairs={(u['source'],u['target']) for u in selected}
            assert len(pairs)==len(selected),'Duplicate evaluation units'
            assert len({w['source'] for w in fits})==len(fits),'Duplicate Source fits'
            full_pairs=pairs=={(s,t) for s in KEYS for t in KEYS if s!=t}
            full_sources={w['source'] for w in fits}==set(KEYS)
            row=dict.fromkeys(columns)
            row.update(Method=NAMES[method],**{'Seeds / tier':'1 / pre_final'})
            if full_pairs:
                for column,key in SCORES.items(): row[column]=average([u['metrics'][key] for u in selected],12)
                row['Inference time (s)']=average([u['inference_seconds'] for u in selected],12)
                row['Cells/sec']=average([u['display']['Cells/sec'] for u in selected],12)
                row['Inference peak VRAM (GiB)']=average([u['inference_peak_allocated_bytes']/2**30 if u['inference_peak_allocated_bytes'] is not None else None for u in selected],12)
                row['Inference resident params']=average([u['inference_parameters'] for u in selected],12)
                batches={u['batch_size'] for u in selected}
                row['Batch size']=next(iter(batches)) if len(batches)==1 else 'mixed'
                row['GPU']=', '.join(sorted({u['display']['GPU'] for u in selected}))
            if full_sources:
                row['Train / fit time (s)']=average([w['source_record']['training']['seconds'] for w in fits],4)
                row['End-to-end time (s)']=average([w['end_to_end_seconds'] for w in fits],4)
                row['Trainable params']=average([w['source_record']['trainable_parameters'] for w in fits],4)
                row['Train peak VRAM (GiB)']=average([w['source_record']['training']['peak_allocated_bytes']/2**30 if w['source_record']['training']['peak_allocated_bytes'] is not None else None for w in fits],4)
            seed_rows.append(row)
            per_seed.append(dict(Seed=seed,**row))
            coverage.append(dict(Method=NAMES[method],Seed=seed,Sources=len(fits),Directions=len(selected),
                complete_sources=full_sources,complete_directions=full_pairs))
        mean=dict.fromkeys(columns); sd=dict.fromkeys(columns)
        mean.update(Method=NAMES[method],**{'Seeds / tier':'5 / pre_final'})
        sd.update(Method=NAMES[method],**{'Seeds / tier':'5 / pre_final'})
        for column in columns[1:]:
            values=[r[column] for r in seed_rows]
            if column in ['GPU','Batch size']:
                if all(v is not None for v in values): mean[column]=sd[column]=', '.join(sorted({str(v) for v in values}))
            elif column!='Seeds / tier':
                mean[column]=average(values,5)
                if mean[column] is not None: sd[column]=float(np.std(values,ddof=1))
        means.append(mean);deviations.append(sd)
    table(output/'combined_mean',means)
    table(output/'combined_sample_sd',deviations)
    table(output/'combined_per_seed',per_seed)
    write(output/'coverage.json',coverage)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--roots',type=Path,nargs='+',required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    summarize(args.roots,args.output)

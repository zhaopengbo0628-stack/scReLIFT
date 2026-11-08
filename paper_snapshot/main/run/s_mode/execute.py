"""Serial pre-final S-mode execution; frozen model code is not modified."""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from scipy import sparse

ROOT = Path('/home/u202522802045/share/zpb')
sys.path[:0] = [str(ROOT), str(ROOT/'main/run'), str(ROOT/'main/run/d_mode'), str(ROOT/'main/core/regulatory/common/legacy_support')]
from data.code.s_mode.prepare import legacy, splits


def check_cache(key, cache_root):
    cache = legacy.cache_dir(cache_root, key)
    m = legacy.checked(cache/'manifest.json')
    expected, cells = splits(key)
    for split, index in expected.items():
        if not np.array_equal(np.load(cache/f'{split}_index.npy'), index):
            raise ValueError('Frozen split differs: '+key+'/'+split)
    if (cache/'barcodes.txt').read_text().splitlines() != cells:
        raise ValueError('Frozen barcode order differs: '+key)
    state = np.load(cache/'rna_state.npz')
    if not np.array_equal(state['fit_cell_index'], expected['train']):
        raise ValueError('RNA fit cells differ')
    rna = np.load(cache/'rna_train.npy')
    np.testing.assert_allclose(state['mean'],rna.mean(0),rtol=1e-5,atol=1e-6)
    np.testing.assert_allclose(state['std'],rna.std(0),rtol=1e-5,atol=1e-6)
    genes = np.load(cache/'gene_ids.npy')
    if len(genes)!=2000 or len(set(genes))!=2000:
        raise ValueError('Gene IDs or dimensions differ')
    matrix = sparse.load_npz(cache/'B_S.npz')
    l1 = np.asarray(abs(matrix).sum(0)).ravel()
    np.testing.assert_allclose(l1[l1>0],1,rtol=1e-5,atol=1e-6)
    for split in ('train','val'):
        x=np.load(cache/f'rna_{split}.npy'); f=np.load(cache/f'F_{split}.npy')
        if x.shape!=f.shape or x.shape!=(len(expected[split]),2000):
            raise ValueError('RNA/F dimensions differ')
        if not np.isfinite(x).all() or not np.isfinite(f).all():
            raise ValueError('RNA/F NaN/Inf')
    if m['target_used'] or m['labels_used'] or m['test_atac_loaded']:
        raise ValueError('Source-only boundary violated')
    return dict(dataset=key,status='VERIFIED',manifest_sha256=legacy.sha(cache/'manifest.json'),
                split_counts=m['split_counts'],unsupported_genes=int((l1==0).sum()),
                diagnostic_policy='WARNING_ONLY; unsupported genes retained')


def execute(config_path):
    config=json.loads(config_path.read_text())
    if config['seeds']!=list(range(20020130,20020135)):
        raise ValueError('Expected fixed five pre-final seeds')
    out=Path(config['result_root'])
    def unchanged():
        for path,digest in config['execution_source_hashes'].items():
            if legacy.sha(path)!=digest: raise ValueError('Execution source changed: '+path)
    unchanged()
    print('UNIT_TESTS',flush=True)
    subprocess.run([sys.executable,'-m','unittest','discover','-s',str(ROOT/'main/run/s_mode/tests'),
                    '-p','test_smode_full_b2.py','-v'],check=True)
    for key in ('D1','D2','D3','D4'):
        unchanged()
        print('PREPARE',key,flush=True)
        subprocess.run([sys.executable,str(ROOT/'main/run/s_mode/train.py'),'--config',str(config_path),
                        '--stage','prepare','--source',key],check=True)
        audit=check_cache(key,config['cache_root'])
        legacy.write(out/'audits'/f'{key}_cache.json',audit)
        print('CACHE_VERIFIED',json.dumps(audit),flush=True)
        subprocess.run([sys.executable,str(ROOT/'main/run/s_mode/train.py'),'--config',str(config_path),
                        '--stage','smoke','--source',key],check=True)
        for seed in config['seeds']:
            unchanged()
            print('TRAIN',key,seed,flush=True)
            subprocess.run([sys.executable,str(ROOT/'main/run/s_mode/train.py'),'--config',str(config_path),
                            '--stage','run','--source',key,'--seed',str(seed)],check=True)
            subprocess.run([sys.executable,str(ROOT/'main/run/s_mode/summarize.py'),
                            '--config',str(config_path)],check=True)
    summary=json.loads((out/'metrics/summary.json').read_text())
    if summary['status']!='COMPLETE': raise ValueError('Expected 20 models and 80 evaluations')
    print('EXPERIMENT_COMPLETE',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--config',type=Path,required=True)
    execute(parser.parse_args().config)

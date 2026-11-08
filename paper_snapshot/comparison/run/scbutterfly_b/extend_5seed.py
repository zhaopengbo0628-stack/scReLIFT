"""Extend the frozen Task49 protocol without rerunning its completed units."""
import argparse
import importlib.util
from pathlib import Path

import numpy as np
import torch

from model_dev.run.brra.common import ROOT
from model_dev.run.brra.intra import wait_for_idle
from model_dev.run.brra.train import registry_status


def launch(out):
    previous = ROOT / 'comparison/result/49_smode_scbutterfly_corrected_pbmc_3seed_20261012'
    path = previous / 'source/s_mode_corrected_pbmc_v2.py'
    spec = importlib.util.spec_from_file_location('scb_task49_frozen', path)
    module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
    np.Inf, np.NaN = np.inf, np.nan
    torch.set_num_threads(2)
    module.SEEDS = list(range(20020130, 20020135))
    module.sc.PREP = out; module.sc.GOAL1 = out
    config = module.read(out / 'config.json')
    for name, digest in config['source_hashes'].items():
        if module.sha(name) != digest: raise ValueError('Frozen implementation changed')
    lock = wait_for_idle(out)
    try:
        registry_status(out, 'RUNNING_MISSING_SEEDS')
        for source in module.KEYS:
            for seed in module.SEEDS:
                unit = out / 'models' / source / f'seed_{seed}'
                if seed <= 20020132:
                    manifest = module.read(unit / 'manifest.json')
                    if module.sha(unit / 'best_all.pt') != manifest['best_checkpoint_sha256']:
                        raise ValueError('Reused checkpoint changed')
                    for target in module.KEYS:
                        row = module.read(unit / 'evaluations' / f'{target}.json')
                        if module.sha(unit / f'{target}_predicted_atac.npy') != row['prediction_sha256']:
                            raise ValueError('Reused prediction changed')
                    continue
                module.unit_run(out, source, seed, False)
        module.summarize(out)
        status = module.read(out / 'metrics/status.json')
        if status['trained_models'] != 20 or status['evaluations'] != 80:
            raise ValueError('Missing units')
        status.update(status='COMPLETE', expected_models=20, expected_evaluations=80)
        module.write(out / 'metrics/status.json', status)
        for p in (out / 'metrics').rglob('summary.md'):
            p.write_text(p.read_text().replace('3-seed pre_final', '5-seed pre_final'))
        registry_status(out, 'COMPLETED_PRE_FINAL')
    except Exception as exc:
        module.write(out / 'failure.json', dict(error=repr(exc)))
        registry_status(out, 'FAILED'); raise
    finally: lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--out', type=Path, required=True)
    launch(parser.parse_args().out)

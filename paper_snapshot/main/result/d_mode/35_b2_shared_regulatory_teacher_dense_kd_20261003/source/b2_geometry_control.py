"""Preparation only: Teacher row-L2 geometry, preserving Task31 landmarks."""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


def corrected_target(graph, original, parent, train_mean=None, train_std=None):
    _, coordinates = parent._spectral_coordinates(graph)
    coordinates = coordinates / np.maximum(np.linalg.norm(coordinates, axis=1, keepdims=True), 1e-8)
    landmarks = np.asarray(original['landmarks'], dtype=np.int64)
    distances = parent._landmark_distances(coordinates, landmarks)
    mean = float(distances.mean()) if train_mean is None else float(train_mean)
    std = max(float(distances.std()) if train_std is None else float(train_std), 1e-6)
    return dict(landmarks=landmarks, values=(distances-mean)/std, mean=np.asarray(mean), std=np.asarray(std))


def prepare(out, root):
    out.mkdir(parents=True, exist_ok=True)
    runner = Path(__file__)
    parent_source = root/'dmode_dev/experiments/v5d/run_task31_dmode_spectral_landmark_global.py'
    sys.path[:0] = [str(parent_source.parent), str(root/'dmode_dev')]
    import run_task31_dmode_spectral_landmark_global as parent

    rng = np.random.default_rng(20020130)
    x = rng.normal(size=(80, 12))
    graph = np.exp(-((x[:, None]-x[None, :])**2).sum(2)/12)
    np.fill_diagonal(graph, 0)
    original, _, _ = parent._build_global_target(graph)
    corrected = corrected_target(graph, dict(landmarks=original.landmarks), parent)
    _, coordinates = parent._spectral_coordinates(graph)
    coordinates /= np.maximum(np.linalg.norm(coordinates, axis=1, keepdims=True), 1e-8)
    distances = parent._landmark_distances(coordinates, original.landmarks)
    expected = (distances-float(corrected['mean']))/float(corrected['std'])
    np.testing.assert_array_equal(corrected['landmarks'], original.landmarks)
    np.testing.assert_allclose(corrected['values'], expected, atol=1e-6)
    fixed = {}
    for i in range(1, 5):
        unit = root/f'result_contain/strict_lodo_two_step_20261002/parent/target_D{i}/seed_20020130'
        config = json.loads((unit/'config.json').read_text())
        training = json.loads((unit/'training.json').read_text())
        fixed[f'target_D{i}'] = dict(config=config, local_initial_scale=training['local_initial_scale'],
                                    global_initial_scale=training['global_initial_scale'])
    manifest = dict(status='PREPARED_NOT_RUN', C_CONTROL_READY=True,
        sole_change='row-L2 normalize Teacher spectral coordinates before landmark distances; Student row-L2 unchanged',
        unchanged=['B_D','F','SNF graph','pair-bank','q','landmark identities','RNA backbone','Student capacity',
                   'local loss','optimizer','schedule','checkpoint rule','numeric objective initial scales'],
        normalization='distance mean/std fit on corrected SOURCE train distances; SOURCE val reuses train values',
        original_landmarks_required=True, original_numeric_loss_scales_required=True,
        geometry_content_caveat='row normalization changes Teacher distance ordering, not a neutral affine calibration',
        static_synthetic_identity_test='PASS', target_used=False, training_performed=False,
        parent_sha256=hashlib.sha256(parent_source.read_bytes()).hexdigest(),
        control_sha256=hashlib.sha256(runner.read_bytes()).hexdigest(), fixed_parent_settings=fixed,
        launch_authorized=False, reason_not_run='B2 priority; preparation does not launch another scientific Task')
    (out/'CONTROL_MANIFEST.json').write_text(json.dumps(manifest, indent=2)+'\n')
    print('C_CONTROL_PREPARED', flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--root', type=Path, default=Path('/home/u202522802045/share/zpb'))
    a = p.parse_args()
    prepare(a.out, a.root)

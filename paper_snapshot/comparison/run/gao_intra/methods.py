"""Gao input/timing adapters; model classes and objectives come from frozen runners."""
import copy
import importlib.util
import random
import sys
import time

import numpy as np
from scipy import sparse

from common import ROOT, CACHE, SOURCE, module, read, write, sha, stamp, memory, inference

sys.path[:0] = [str(ROOT), str(ROOT/'comparison/run'), str(ROOT/'data/run/s_mode')]


def data_for(out, unit, method, smoke):
    tick = time.perf_counter()
    inputs = out/'inputs'
    data = {}
    for split in ['train', 'val', 'test']:
        stop = dict(train=128, val=64, test=64)[split] if smoke else None
        data['rna_log_'+split] = sparse.csr_matrix(np.load(CACHE/f'rna_{split}.npy')[:stop])
        idx = np.load(CACHE/f'{split}_index.npy')[:stop]
        data['rna_'+split] = idx
    if method in ['rna_pca', 'rna_encoder']:
        return data, None, time.perf_counter()-tick
    counts = sparse.load_npz(inputs/'rna_hvg_counts.npz').tocsr()
    for split in ['train', 'val', 'test']:
        data['rna_'+split] = counts[data['rna_'+split]].astype(np.float32)
    del counts
    if method == 'b2':
        return data, None, time.perf_counter()-tick
    ntrain = 24635
    raw = sparse.load_npz(inputs/'atac_train_val_counts.npz').tocsr().astype(bool).astype(np.float32)
    keep = np.asarray(raw[:ntrain].sum(0)).ravel() >= np.ceil(.005*ntrain)
    columns = np.flatnonzero(keep)
    peaks = np.load(CACHE/'peak_names.npy')[columns]
    if method == 'scbutterfly':
        chromosome = np.asarray([p.split('_')[0] for p in peaks])
        order = np.argsort(chromosome, kind='stable')
        columns, peaks, chromosome = columns[order], peaks[order], chromosome[order]
        boundaries = np.r_[0, np.flatnonzero(chromosome[1:] != chromosome[:-1])+1, len(peaks)]
        write(unit/'chromosomes.json', np.diff(boundaries).tolist())
    raw = raw[:, columns].tocsr()
    idf = np.log1p(ntrain/np.maximum(np.asarray(raw[:ntrain].sum(0)).ravel(), 1)).astype(np.float32)
    tfidf = raw.multiply(1/np.maximum(np.asarray(raw.sum(1)).ravel(), 1)[:, None]).multiply(idf).tocsr()
    scale = max(float(tfidf[:ntrain].max()), 1e-8)
    tfidf.data = np.clip(tfidf.data/scale, 0, 1)
    for split, rows in [('train', slice(0, ntrain)), ('val', slice(ntrain, None))]:
        stop = dict(train=128, val=64)[split] if smoke else None
        data['atac_'+split] = raw[rows][:stop].tocsr()
        data['atac_tfidf_'+split] = tfidf[rows][:stop].tocsr()
    cache = unit/'derived'
    cache.mkdir()
    np.save(cache/'peak_ids.npy', peaks)
    np.save(cache/'gene_ids.npy', np.load(CACHE/'gene_ids.npy'))
    np.save(cache/'train_index.npy', np.load(CACHE/'train_index.npy'))
    (cache/'barcodes.txt').symlink_to(inputs/'barcodes.txt')
    np.savez_compressed(cache/'atac_state.npz', peak_columns=columns, train_idf=idf, train_max=scale)
    write(cache/'manifest.json', dict(fit_cells='frozen train only', prevalence=.005,
        peaks=len(peaks), test_atac_loaded=False, input_manifest_sha256=sha(out/'DATASET_AND_SPLIT_MANIFEST.json')))
    return data, cache, time.perf_counter()-tick


def b2_run(out, unit, seed, smoke, config):
    import torch
    # Load the same preparation namespace first as Task80; avoids obsolete duplicate imports.
    import prepare_adult_cortex_intra as prep
    from main.run.s_mode import train as trainer
    from main.core.regulatory.s_mode.single_source import build_bs, build_field
    b = trainer.b
    tick = stamp('cuda')
    rna = {s: np.load(CACHE/f'rna_{s}.npy') for s in ['train', 'val', 'test']}
    if smoke:
        cache = CACHE
        manifest = dict(hashes={str(CACHE/name):sha(CACHE/name) for name in
            ['rna_train.npy','rna_val.npy','F_train.npy','F_val.npy','rna_state.npz','gene_ids.npy']})
        preprocessing = stamp('cuda')-tick
        return b2_fit_predict(trainer, b, rna, cache, manifest, unit, seed, smoke, preprocessing)
    atac = sparse.load_npz(out/'inputs/atac_train_val_counts.npz').tocsr().astype(bool).astype(np.float32)
    from v5d.regulatory_graph import read_tss
    graph = prep.common.graph_fast(np.load(CACHE/'gene_ids.npy'), np.load(CACHE/'peak_names.npy'), read_tss(prep.GTF))
    graph['peak_names'] = np.load(CACHE/'peak_names.npy')
    selector_tick = time.perf_counter()
    matrix, stability = build_bs(rna['train'], atac[:24635], graph)
    selector_seconds = time.perf_counter()-selector_tick
    field_tick = time.perf_counter()
    field, state, report = build_field(atac, matrix, np.arange(24635))
    field_seconds = time.perf_counter()-field_tick
    historical = sparse.load_npz(CACHE/'B_S.npz')
    difference = matrix-historical
    assert not difference.nnz or np.max(np.abs(difference.data)) < 1e-7, 'Fresh B_S differs from frozen Task80'
    np.testing.assert_allclose(field[:24635], np.load(CACHE/'F_train.npy'), rtol=1e-5, atol=1e-5)
    cache = unit/'derived/b2'
    cache.mkdir(parents=True)
    for s in ['train', 'val', 'test']:
        (cache/f'rna_{s}.npy').symlink_to(CACHE/f'rna_{s}.npy')
    for name in ['rna_state.npz', 'gene_ids.npy']:
        (cache/name).symlink_to(CACHE/name)
    np.save(cache/'F_train.npy', field[:24635])
    np.save(cache/'F_val.npy', field[24635:])
    sparse.save_npz(cache/'B_S.npz', matrix)
    np.savez_compressed(cache/'field_state.npz', **state)
    write(cache/'regulatory_graph.json', dict(candidate_edges=int(graph['candidate_mask'].sum()),
        retained_edges=matrix.nnz, retained_genes=int(np.count_nonzero(np.asarray(abs(matrix).sum(0)).ravel())),
        selector_seconds=selector_seconds, field_seconds=field_seconds, report=report))
    manifest = dict(hashes={str(p):sha(p) for p in cache.iterdir() if p.is_file()})
    write(cache/'manifest.json', manifest)
    preprocessing = stamp('cuda')-tick
    del atac, field, graph, stability, matrix, historical, difference
    return b2_fit_predict(trainer, b, rna, cache, manifest, unit, seed, smoke, preprocessing)


def b2_fit_predict(trainer, b, rna, cache, manifest, unit, seed, smoke, preprocessing):
    import torch
    fit_start = stamp('cuda')
    trainer.train('GaoVIS', seed, unit/'fit', cache, manifest, smoke=smoke, evaluation_targets=())
    fitted = unit/'fit'/('_smoke' if smoke else 'models')/'GaoVIS'/f'seed_{seed}'
    record = read(fitted/'training_manifest.json')
    fit_seconds = stamp('cuda')-fit_start
    peak = dict(allocated_GiB=record['efficiency_v2']['training']['peak_allocated_bytes']/2**30,
                reserved_GiB=record['efficiency_v2']['training']['peak_reserved_bytes']/2**30)
    checkpoint = fitted/'checkpoints/student/best.pt'
    student = b.Encoder(2000)
    student.load_state_dict(torch.load(checkpoint, map_location='cpu', weights_only=False)['model'])
    student.to('cuda').eval().requires_grad_(False)
    state = np.load(CACHE/'rna_state.npz')
    x = ((rna['test']-state['mean'])/(state['std']+1e-6)).astype(np.float32)
    if smoke:
        x = x[:64]
    def predict(values):
        with torch.inference_mode():
            return np.concatenate([student(torch.from_numpy(values[i:i+256]).cuda()).cpu().numpy() for i in range(0, len(values), 256)])
    z, inf = inference(predict, x, 'cuda')
    (unit/'checkpoint').symlink_to(fitted/'checkpoints', target_is_directory=True)
    stages = record['efficiency_v2']['stages']
    return z, dict(preprocessing_seconds=preprocessing, fitting_seconds=fit_seconds,
        teacher_pretraining_seconds=stages['teacher']['seconds'], student_adaptation_seconds=stages['student']['seconds'],
        validation_checkpoint_seconds=None, validation_scope='included in original stage timers',
        trainable_parameters=record['efficiency_v2']['trainable_parameters'],
        inference_resident_parameters=sum(p.numel() for p in student.parameters()), train_peak=peak,
        training_history={s:read(fitted/f'checkpoints/{s}/history.json') for s in ['teacher','student']},
        **inf)


def vae_run(method, data, config, unit, seed, smoke):
    import s_mode_intra as original
    original_predict = original.predict
    # Training keeps its PBMC batch. Only frozen inference is standardized to 256.
    first_call = True
    def predict(model, kind, rna, peaks, device, batch):
        nonlocal first_call
        if first_call:
            rna = data['rna_test'][:256]
            first_call = False
        return original_predict(model, kind, rna, peaks, device, 256)
    original.predict = predict
    try:
        result = original.torch_run(method, data, config, unit, seed, smoke, 'cuda')
    finally:
        original.predict = original_predict
    (unit/'checkpoint').mkdir()
    (unit/'checkpoint/best.pt').symlink_to(unit/'best.pt')
    v = result['efficiency_v2']
    return np.load(unit/'rna_test_embedding.npy'), dict(fitting_seconds=v['source_fitting']['seconds'],
        optimizer_validation_checkpoint_seconds=v['training']['seconds'],
        teacher_pretraining_seconds=None, student_adaptation_seconds=None, validation_checkpoint_seconds=None,
        validation_scope='included in fitting; model-native single stage',
        trainable_parameters=result['trainable_parameters'], inference_resident_parameters=result['inference_parameters'],
        train_peak=dict(allocated_GiB=v['training']['peak_allocated_bytes']/2**30, reserved_GiB=v['training']['peak_reserved_bytes']/2**30),
        inference_peak=dict(allocated_GiB=v['inference']['peak_allocated_bytes']/2**30, reserved_GiB=v['inference']['peak_reserved_bytes']/2**30),
        inference_seconds=result['inference_seconds'], cells_per_second=result['cells_per_second'], inference_batch=256,
        training_history=read(unit/'history.json'))


def eclare_run(data, config, unit, seed, smoke, cache):
    import torch
    import neural_s_mode_intra as original
    model_adapter = original.adapter('eclare')
    history, stages = [], []
    train = original.training(data, config, seed, smoke, 'cuda', history, stages)
    fit_started = stamp('cuda')
    model, result = model_adapter.fit(data, config, unit, seed, smoke, 'cuda', cache, train)
    fitting_seconds = stamp('cuda')-fit_started
    peak = memory('cuda')
    model.zero_grad(set_to_none=True)
    model.eval().requires_grad_(False)
    checkpoint = torch.load(unit/'best.pt', map_location='cpu', weights_only=False)
    reloaded = model_adapter.make_model(checkpoint['peaks'], 'cuda')
    reloaded.load_state_dict(checkpoint['state_dict'])
    reloaded.eval().requires_grad_(False)
    x = data['rna_log_test']
    np.testing.assert_allclose(model_adapter.predict(model, x[:16], checkpoint, 'cuda', 256),
        model_adapter.predict(reloaded, x[:16], checkpoint, 'cuda', 256), rtol=1e-5, atol=1e-6)
    del model
    z, inf = inference(lambda rows: model_adapter.predict(reloaded, rows, checkpoint, 'cuda', 256), x, 'cuda')
    (unit/'checkpoint').mkdir()
    for path in unit.glob('*.pt'):
        (unit/'checkpoint'/path.name).symlink_to(path)
    return z, dict(fitting_seconds=fitting_seconds,
        optimizer_validation_checkpoint_seconds=sum(s['seconds'] for s in stages),
        teacher_pretraining_seconds=sum(s['seconds'] for s in stages[:-1]), student_adaptation_seconds=stages[-1]['seconds'],
        validation_checkpoint_seconds=None, validation_scope='included in original stage timers',
        trainable_parameters=result['trainable_parameters'], inference_resident_parameters=sum(p.numel() for p in reloaded.parameters()),
        train_peak=peak, training_history=history, stage_records=stages, **inf)


def scbutterfly_run(data, config, unit, seed, smoke):
    import torch
    import anndata as ad
    np.Inf, np.NaN = np.inf, np.nan
    official = ROOT/'comparison/methods/scbutterfly_b/official'
    spec = importlib.util.spec_from_file_location('scButterfly', official/'__init__.py', submodule_search_locations=[str(official)])
    package = importlib.util.module_from_spec(spec)
    sys.modules['scButterfly'] = package
    spec.loader.exec_module(package)
    from scButterfly.butterfly import Butterfly
    import scButterfly.train_model as training
    modalities = ['RNA_encoder', 'ATAC_encoder', 'RNA_decoder', 'ATAC_decoder', 'R_translator', 'A_translator', 'translator', 'discriminator_A', 'discriminator_R']
    fit_started = stamp('cuda')
    obj = Butterfly()
    nr = data['rna_log_train'].shape[0]
    r = sparse.vstack([data['rna_log_train'], data['rna_log_val']]).tocsr()
    a = sparse.vstack([data['atac_tfidf_train'], data['atac_tfidf_val']]).tocsr()
    obj.load_data(ad.AnnData(r), ad.AnnData(a), list(range(nr)), [], list(range(nr, r.shape[0])))
    obj.RNA_data_p, obj.ATAC_data_p, obj.is_processed = obj.RNA_data, obj.ATAC_data, True
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    obj.construct_model(read(unit/'chromosomes.json'), R_encoder_dim_list=[256,128], A_encoder_dim_list=[32,128],
        R_decoder_dim_list=[128,256], A_decoder_dim_list=[128,32], logging_path=str(unit))
    model = obj.model
    original = training.EarlyStopping
    history, phases, optimized = [], {}, {}
    tick = stamp('cuda')
    phase_start = [tick]
    class AuditedStopping(original):
        def __call__(self, loss, current, path):
            assert np.isfinite(loss)
            stage = next(s for s in ['R2R', 'A2A', 'all'] if getattr(current, 'early_stopping_'+s) is self)
            if history and history[-1]['stage'] != stage:
                phases[history[-1]['stage']] = stamp('cuda')-phase_start[0]
                phase_start[0] = stamp('cuda')
            previous = self.best_score
            super().__call__(loss, current, path)
            optimizers = {k:v for k,v in vars(current).items() if k.startswith('optimizer_')}
            for opt in optimizers.values():
                for group in opt.param_groups:
                    optimized.update({id(p):p.numel() for p in group['params']})
            state = dict(modules={k:getattr(current,k).state_dict() for k in modalities},
                optimizer_state_dict={k:v.state_dict() for k,v in optimizers.items()}, scheduler_state_dict=None,
                early_stopping=dict(best_score=self.best_score, counter=self.counter), stage=stage,
                rng_state=dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all()))
            assert all(torch.isfinite(p).all() for k in modalities for p in getattr(current,k).parameters())
            if smoke:
                grads = [p.grad for k in modalities for p in getattr(current,k).parameters() if p.grad is not None]
                assert grads and all(torch.isfinite(g).all() for g in grads)
            if previous is None or self.best_score != previous or self.counter == 0:
                torch.save(state, unit/f'best_{stage}.pt')
            history.append(dict(stage=stage, epoch=sum(x['stage']==stage for x in history)+1, source_validation_loss=float(loss)))
            write(unit/'training_history.json', history)
            print('SCBUTTERFLY', history[-1], flush=True)
    training.EarlyStopping = AuditedStopping
    try:
        obj.train_model(R2R_pretrain_epoch=1 if smoke else 100, A2A_pretrain_epoch=1 if smoke else 100,
            translator_epoch=1 if smoke else 200, patience=50, batch_size=64, loss_weight=[1,2,1],
            output_path=str(unit), logging_path=str(unit), seed=seed)
    finally:
        training.EarlyStopping = original
    phases['all'] = stamp('cuda')-phase_start[0]
    fit_seconds = stamp('cuda')-fit_started
    peak = memory('cuda')
    best = torch.load(unit/'best_all.pt', map_location='cpu', weights_only=False)
    for k in modalities:
        getattr(model,k).load_state_dict(best['modules'][k])
    model.set_eval()
    for k in list(vars(model)):
        if k.startswith('optimizer_'):
            delattr(model,k)
    resident = {id(p):p.numel() for k in modalities for p in getattr(model,k).parameters()}
    x = data['rna_log_test'].toarray()
    def predict(values):
        outputs = []
        with torch.inference_mode():
            for i in range(0,len(values),256):
                batch = torch.from_numpy(values[i:i+256]).cuda()
                _, latent, _, _ = model.translator.test_model(model.RNA_encoder(batch), 'RNA')
                y = model.ATAC_decoder(latent)
                y[y < 1e-4] = 0
                outputs.append(y.cpu().numpy())
        return np.concatenate(outputs)
    z, inf = inference(predict, x, 'cuda')
    assert np.isfinite(z).all()
    (unit/'checkpoint').mkdir()
    for p in unit.glob('best_*.pt'):
        (unit/'checkpoint'/p.name).symlink_to(p)
    return z, dict(fitting_seconds=fit_seconds, teacher_pretraining_seconds=None,
        student_adaptation_seconds=None, callback_intervals=phases,
        phase_boundary_note='Monolithic official train_model timer; callback intervals are not exact stage timers',
        validation_checkpoint_seconds=None, trainable_parameters=sum(optimized.values()),
        inference_resident_parameters=sum(resident.values()), train_peak=peak, training_history=history,
        representation='RNA-predicted ATAC, threshold <1e-4 to zero; inherited PBMC route', **inf)


def run(method, data, config, unit, seed, smoke, cache):
    if method in ['multigrate', 'cobolt']:
        return vae_run(method, data, config, unit, seed, smoke)
    if method == 'eclare':
        return eclare_run(data, config, unit, seed, smoke, cache)
    if method == 'scbutterfly':
        return scbutterfly_run(data, config, unit, seed, smoke)
    if method == 'palette':
        from s_mode_intra import palette_run
        result = palette_run(data, dict(config, batch_size=256), unit, seed, cache)
        (unit/'checkpoint').mkdir()
        for name in ['source_palette.rds', 'frozen_map.npz']:
            (unit/'checkpoint'/name).symlink_to(unit/name)
        weights = np.load(unit/'frozen_map.npz')['weights']
        def predict(rows):
            return np.concatenate([rows[i:i+256]@weights for i in range(0, rows.shape[0], 256)])
        z, inf = inference(predict, data['rna_log_test'], 'cpu')
        np.testing.assert_allclose(z, np.load(unit/'rna_test_embedding.npy'), rtol=1e-5, atol=1e-6)
        return z, dict(fitting_seconds=result['training_seconds'],
            teacher_pretraining_seconds=None, student_adaptation_seconds=None, validation_checkpoint_seconds=None,
            trainable_parameters=None, inference_resident_parameters=None, train_peak=memory('cpu'), inference_peak=memory('cpu'),
            training_history=[], inference_device='CPU', adaptation='Official unsupervised Palette plus Source-train minimum-norm RNA map', **inf)
    if method == 'rna_pca':
        from sklearn.decomposition import PCA
        x = {s:data['rna_log_'+s].toarray() for s in ['train', 'val', 'test']}
        tick = time.perf_counter()
        mean, std = x['train'].mean(0), x['train'].std(0)
        model = PCA(n_components=64, svd_solver='randomized', random_state=seed).fit((x['train']-mean)/(std+1e-6))
        (unit/'checkpoint').mkdir()
        np.savez_compressed(unit/'checkpoint/pca.npz', mean=mean, std=std, components=model.components_, pca_mean=model.mean_)
        state = np.load(unit/'checkpoint/pca.npz')
        predict = lambda a: ((a-state['mean'])/(state['std']+1e-6)-state['pca_mean'])@state['components'].T
        np.testing.assert_allclose(predict(x['test'][:16]), model.transform((x['test'][:16]-mean)/(std+1e-6)), rtol=1e-4, atol=1e-4)
        fit_seconds = time.perf_counter()-tick
        z, inf = inference(lambda rows: np.concatenate([predict(rows[i:i+256]) for i in range(0, len(rows), 256)]), x['test'], 'cpu')
        return z, dict(fitting_seconds=fit_seconds, teacher_pretraining_seconds=None, student_adaptation_seconds=None,
            validation_checkpoint_seconds=None, trainable_parameters=None, inference_resident_parameters=None,
            train_peak=memory('cpu'), training_history=[], **inf)
    raise RuntimeError('BLOCKED_METHOD_ADAPTATION: '+method)

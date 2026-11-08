"""Donor metadata, fixed covariate-only split, pooling and nested probe fitting."""
from itertools import combinations, product

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEEDS = list(range(20020130, 20020135))
PCA_DIMS = (2, 4, 6)
REGULARIZATION = (0.01, 0.1, 1.0, 10.0, 100.0)


def normalize_disease(values):
    mapping = {'ad': 1, 'alzheimer': 1, "alzheimer's disease": 1,
               'alzheimer disease': 1, 'ctrl': 0, 'control': 0, 'unaffected': 0}
    return values.astype(str).str.strip().str.lower().map(mapping)


def identify_metadata(frame, barcodes):
    """Resolve semantic aliases, validate values, and align exact barcode identities."""
    aliases = {
        'donor': ('donor_id', 'donor', 'subject_id', 'subject', 'individual', 'id'),
        'disease': ('disease_status', 'status', 'diagnosis', 'disease', 'condition'),
        'cell_type': ('author_predicted_id', 'predicted.id', 'cell_type', 'celltype', 'annotation'),
        'age': ('age', 'age_at_death'), 'sex': ('sex', 'gender'),
        'region': ('structure', 'region', 'brain_region'), 'repository': ('repository',),
    }
    lookup = {str(c).lower(): c for c in frame.columns}
    fields = {key: next((lookup[a] for a in names if a in lookup), None)
              for key, names in aliases.items()}
    barcode_fields = [c for c in frame if frame[c].astype(str).isin(set(barcodes)).all()
                      and frame[c].nunique() == len(barcodes) and len(frame) == len(barcodes)]
    if len(barcode_fields) != 1 or any(fields[k] is None for k in ('donor', 'disease', 'cell_type')):
        raise ValueError(f'Missing/ambiguous metadata fields: {fields}, barcodes={barcode_fields}')
    fields['barcode'] = barcode_fields[0]
    if len(set(barcodes)) != len(barcodes):
        raise ValueError('Duplicate matrix barcodes')
    aligned = frame.set_index(fields['barcode']).loc[list(barcodes)]
    cells = pd.DataFrame({'barcode': barcodes})
    for key, col in fields.items():
        if key != 'barcode':
            cells[key] = aligned[col].to_numpy() if col is not None else np.nan
    cells['disease'] = normalize_disease(cells['disease'])
    if cells[['donor', 'disease', 'cell_type']].isna().any().any():
        raise ValueError('Unresolved required donor/disease/type values')
    cells['donor'] = cells['donor'].astype(str)
    cells['age'] = pd.to_numeric(cells['age'], errors='raise')
    for name in ('donor', 'cell_type'):
        if cells[name].astype(str).str.strip().eq('').any():
            raise ValueError(f'Empty {name}')
    for name in ('disease', 'age', 'sex', 'region', 'repository'):
        if (cells.groupby('donor')[name].nunique(dropna=False) > 1).any():
            raise ValueError(f'Contradictory donor metadata: {name}')
    summary = cells.groupby('donor', sort=True).agg(
        disease=('disease', 'first'), age=('age', 'first'), sex=('sex', 'first'),
        region=('region', 'first'), repository=('repository', 'first'), cells=('barcode', 'size'))
    counts = pd.crosstab(cells['donor'], cells['cell_type']).add_prefix('count_')
    summary = summary.join(counts).reset_index()
    if summary.groupby('disease').size().to_dict() != {0: 8, 1: 7}:
        raise ValueError('Expected 8 Control and 7 AD donors')
    return cells, summary, fields


def balanced_split(donors):
    """Exhaustive 2+2 holdout, equal covariate terms; no representation/outcome score."""
    donors = donors.sort_values('donor').reset_index(drop=True)
    age_scale = float(donors.age.std(ddof=0))
    cell_scale = float(np.log1p(donors.cells).std(ddof=0))
    terms = []
    for ad, ctrl in product(combinations(donors.index[donors.disease == 1], 2),
                            combinations(donors.index[donors.disease == 0], 2)):
        test = list(ad + ctrl)
        components = dict(age=0.0, sex=0.0, cells=0.0)
        for state in (0, 1):
            group = donors[donors.disease == state]
            a, b = group.loc[group.index.isin(test)], group.loc[~group.index.isin(test)]
            if age_scale > 0:
                components['age'] += ((a.age.mean()-b.age.mean())/age_scale)**2
            if cell_scale > 0:
                components['cells'] += ((np.log1p(a.cells).mean()-np.log1p(b.cells).mean())/cell_scale)**2
            for sex in sorted(donors.sex.dropna().unique()):
                components['sex'] += ((a.sex == sex).mean()-(b.sex == sex).mean())**2
        names = tuple(sorted(donors.loc[test, 'donor']))
        terms.append((sum(components.values()), names, components))
    score, names, parts = min(terms, key=lambda x: (x[0], x[1]))
    result = donors.copy()
    result['split'] = np.where(result.donor.isin(names), 'test', 'train')
    return result, dict(objective=score, components=parts, candidates=len(terms),
                        rule='min sum squared within-diagnosis train/test age, sex, log-cell-count differences; equal terms; donor-ID tie break')


def cell_indices(cells, donors, seed=SEEDS[0]):
    """Within the eleven outer-train donors, reserve 10% cells for label-free loss."""
    rng = np.random.default_rng(seed)
    train, val = [], []
    for donor in sorted(donors.loc[donors.split == 'train', 'donor']):
        ix = rng.permutation(np.flatnonzero(cells.donor.eq(donor)))
        n = max(1, int(np.floor(len(ix)*0.1)))
        if n == len(ix):
            raise ValueError('Insufficient cells for source validation')
        val.extend(ix[:n]); train.extend(ix[n:])
    test = np.flatnonzero(cells.donor.isin(donors.loc[donors.split == 'test', 'donor']))
    result = {k: np.sort(v).astype(np.int64) for k, v in dict(train=train, val=val, test=test).items()}
    if not np.array_equal(np.sort(np.concatenate(list(result.values()))), np.arange(len(cells))):
        raise ValueError('Incomplete/overlapping cell split')
    if set(cells.iloc[result['train']].donor) & set(cells.iloc[result['test']].donor):
        raise ValueError('Test donor leakage')
    return result


def pool(z, cells, donors, types):
    if len(z) != len(cells) or z.ndim != 2 or not np.isfinite(z).all():
        raise ValueError('Invalid cell embedding/alignment')
    if not set(cells.cell_type).issubset(types):
        raise ValueError('Cell type absent from frozen training taxonomy')
    global_rows, typed_rows, proportions, records = [], [], [], []
    for donor in donors.donor:
        mask = cells.donor.eq(donor).to_numpy()
        if not mask.any():
            raise ValueError('Donor without cells')
        global_rows.append(z[mask].mean(0))
        row, fractions = [], []
        for name in types:
            ix = mask & cells.cell_type.eq(name).to_numpy()
            row.extend(z[ix].mean(0) if ix.any() else np.full(z.shape[1], np.nan))
            fractions.append(ix.sum()/mask.sum())
            records.append(dict(donor=donor, cell_type=name, cells=int(ix.sum()),
                                missing=not bool(ix.any()), low_count=bool(ix.sum() < 20)))
        typed_rows.append(row); proportions.append(fractions)
    p2 = np.asarray(typed_rows)
    return dict(P1=np.asarray(global_rows), P2=p2, P3=np.column_stack([p2, proportions])), records


def pipeline(dim, regularization):
    return make_pipeline(SimpleImputer(strategy='mean'), StandardScaler(),
                         PCA(n_components=dim, svd_solver='full'),
                         LogisticRegression(C=regularization, penalty='l2', solver='lbfgs',
                                            max_iter=5000, random_state=SEEDS[0]))


def fit_probe(x_train, y_train, x_test):
    """Each LODO fold refits imputation, scaler, PCA and L2 classifier from scratch."""
    y_train = np.asarray(y_train)
    if len(x_train) != 11 or np.bincount(y_train, minlength=2).tolist() != [6, 5]:
        raise ValueError('Expected 6 Control + 5 AD training donors')
    if np.isinf(x_train).any() or np.isinf(x_test).any():
        raise ValueError('Infinite donor feature')
    rows, best = [], None
    for dim, regularization in product(PCA_DIMS, REGULARIZATION):
        probabilities = np.empty(len(y_train))
        for fit, held in LeaveOneOut().split(x_train):
            if dim > min(len(fit)-1, x_train.shape[1]) or np.isnan(x_train[fit]).all(0).any():
                raise ValueError('Unidentifiable PCA or all-missing inner-train feature')
            model = pipeline(dim, regularization)
            model.fit(x_train[fit], y_train[fit])
            probabilities[held] = model.predict_proba(x_train[held])[:, 1]
        score = roc_auc_score(y_train, probabilities)
        rows.append(dict(pca_dim=dim, C=regularization, lodo_auroc=float(score),
                         lodo_balanced_accuracy=balanced_accuracy_score(y_train, probabilities >= 0.5),
                         probabilities=probabilities.tolist()))
        # Fixed order breaks ties: smaller PCA dimension, then stronger L2 regularization.
        if best is None or score > best['lodo_auroc']:
            best = rows[-1]
    final = pipeline(best['pca_dim'], best['C']).fit(x_train, y_train)
    probability = final.predict_proba(x_test)[:, 1]
    if not np.isfinite(probability).all():
        raise ValueError('Nonfinite donor predictions')
    return probability, best, rows, final


def metrics(y, probability):
    prediction = np.asarray(probability) >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, prediction, labels=[0, 1]).ravel()
    return dict(AUROC=float(roc_auc_score(y, probability)),
                Balanced_Accuracy=float(balanced_accuracy_score(y, prediction)),
                Macro_F1=float(f1_score(y, prediction, average='macro', zero_division=0)),
                Sensitivity=float(tp/(tp+fn)), Specificity=float(tn/(tn+fp)),
                TN=int(tn), FP=int(fp), FN=int(fn), TP=int(tp))

"""BMMC Site-level comparison cache/evaluation adapter."""
import csv, hashlib, json
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.metrics import (accuracy_score, adjusted_mutual_info_score, adjusted_rand_score,
    balanced_accuracy_score, f1_score, normalized_mutual_info_score, silhouette_score)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

ROOT=Path(__file__).resolve().parents[2]
CACHE_ROOT=ROOT/'data/cache/s_mode/BMMC'
SEEDS=[0,1,2]
EVALUATOR=dict(init='k-means++',n_init=100,random_state=20020130,max_iter=1000,tol=1e-4,algorithm='lloyd')


def read(path): return json.loads(Path(path).read_text())
def write(path,value):
    path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()


def cache_for(key,version='bmmc_comparison_intra_v1'):
    return CACHE_ROOT/version/key

def benchmark_for(key):
    return ROOT/'benchmark/BMMC'/key

def prepare(key):
    cache=cache_for(key)
    manifest=read(cache/'manifest.json')
    if manifest.get('test_atac_loaded') or manifest.get('labels_used_for_fitting'):
        raise ValueError('Invalid BMMC cache boundary')
    return cache

def load(cache,smoke=False):
    data={}
    for split in ('train','val','test'):
        limit={'train':128,'val':64,'test':64}[split] if smoke else None
        for modality in ('rna','rna_log'):
            data[f'{modality}_{split}']=sparse.load_npz(cache/f'{modality}_{split}.npz')[:limit].tocsr()
        if split!='test':
            for modality in ('atac','atac_tfidf'):
                data[f'{modality}_{split}']=sparse.load_npz(cache/f'{modality}_{split}.npz')[:limit].tocsr()
    return data, read(cache/'manifest.json')

def evaluate(unit,cache,key):
    prediction=unit/'rna_test_embedding.npy'; frozen=sha(prediction)
    label_path=benchmark_for(key)/'label.csv'
    with label_path.open(newline='') as fh:
        raw=np.asarray([row['cell_type'] for row in csv.DictReader(fh)])
    test=np.load(cache/'test_index.npy')
    _,labels=np.unique(raw[test],return_inverse=True)
    z=np.load(prediction)
    assert len(z)==len(labels) and np.isfinite(z).all()
    z=StandardScaler().fit_transform(z); k=len(np.unique(labels))
    clusters=KMeans(n_clusters=k,**EVALUATOR).fit_predict(z)
    table=np.zeros((k,k),np.int64); np.add.at(table,(labels,clusters),1)
    a,b=linear_sum_assignment(-table); mapping=dict(zip(b,a))
    matched=np.asarray([mapping[c] for c in clusters])
    nn=NearestNeighbors(n_neighbors=min(16,len(labels))).fit(z).kneighbors(z,return_distance=False)
    nn=np.asarray([row[row!=i][:15] for i,row in enumerate(nn)])
    acc=float(accuracy_score(labels,matched))
    metrics=dict(ari=float(adjusted_rand_score(labels,clusters)),
        nmi=float(normalized_mutual_info_score(labels,clusters)),
        ami=float(adjusted_mutual_info_score(labels,clusters)),cluster_acc=acc,accuracy=acc,
        balanced_accuracy=float(balanced_accuracy_score(labels,matched)),
        macro_f1=float(f1_score(labels,matched,average='macro',zero_division=0)),
        weighted_f1=float(f1_score(labels,matched,average='weighted',zero_division=0)),
        knn_purity_at_15=float(np.mean(labels[nn]==labels[:,None])),
        silhouette=float(silhouette_score(z,labels)) if 1<k<len(labels) else None,
        geometry_pearson=None,geometry_spearman=None,
        annotation_scope='Posthoc test Hungarian cluster scores; classification metrics are NOT label transfer',
        evaluator=EVALUATOR,labels_loaded_after_prediction_frozen=True,test_atac_used=False,
        prediction_sha256=frozen,label_file_sha256=sha(label_path))
    write(unit/'metrics.json',metrics)
    with (unit/'metrics.csv').open('w',newline='') as fh:
        w=csv.DictWriter(fh,fieldnames=list(metrics));w.writeheader();w.writerow(metrics)
    names=['ari','nmi','ami','cluster_acc','accuracy','balanced_accuracy','macro_f1','weighted_f1','knn_purity_at_15','silhouette','geometry_spearman','geometry_pearson']
    (unit/'metrics.md').write_text('| '+' | '.join(names)+' |\n|'+'---|'*len(names)+'\n| '+' | '.join('NA' if metrics[n] is None else f'{metrics[n]:.6f}' for n in names)+' |\n')

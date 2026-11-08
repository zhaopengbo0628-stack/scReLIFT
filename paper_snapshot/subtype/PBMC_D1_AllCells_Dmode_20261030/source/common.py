import ast
import hashlib
import json
import sys
from pathlib import Path
from bisect import bisect_left, bisect_right
import numpy as np
from scipy import sparse

RUN = Path(__file__).resolve().parents[1]
SRC = RUN/'source'
FROZEN = SRC/'frozen'
SHARED = Path('/home/u202522802045/share/zpb')
sys.path.insert(0, str(FROZEN))
from v5d.regulatory_graph import parse_peak, normalize_chromosome
SEEDS = [20020130, 20020131, 20020132]

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''): h.update(b)
    return h.hexdigest()

def write(path, value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_name(path.name+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    tmp.replace(path)

def frozen_function(file,name):
    tree=ast.parse(Path(file).read_text(encoding='utf-8'))
    node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
    ns=dict(globals())
    exec(compile(ast.Module(body=[node],type_ignores=[]),str(file),'exec'),ns)
    return ns[name]

normalize_log1p=frozen_function(FROZEN/'v5s/data.py','normalize_log1p')
hvg_train_indices=frozen_function(FROZEN/'prepare_pbmc10k_model_v1.py','hvg_train_indices')
graph_fast=frozen_function(FROZEN/'prepare_updated_b2_sources.py','graph_fast')

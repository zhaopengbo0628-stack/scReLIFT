"""rev02 source balancing adapter; frozen legacy edge formulas otherwise unchanged."""
import ast
import hashlib
import importlib
import sys
from pathlib import Path

LEGACY_SHA256 = 'a03b65ac9dc2c4697b814148b4d097e2b49af88bc3b24cb7e8ed301fb952eb11'


def load_selector(code_root):
    code_root = Path(code_root).resolve()
    path = code_root/'v5d/dmode_reliable_selector.py'
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != LEGACY_SHA256:
        raise RuntimeError('Selector differs from audited source; review before adapting')
    sys.path.insert(0,str(code_root))
    legacy = importlib.import_module('v5d.dmode_reliable_selector')
    if Path(legacy.__file__).resolve() != path:
        raise RuntimeError('Another v5d namespace is already loaded')
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='build_source_only_b')
    changes = 0
    for statement in function.body:
        if isinstance(statement,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='raw_dataset_weight' for t in statement.targets):
            if ast.unparse(statement.value) != 'n_train * q_dataset':
                raise RuntimeError('Unexpected source weighting expression')
            statement.value = ast.parse('q_dataset.copy()',mode='eval').body
            changes += 1
    if changes != 1:
        raise RuntimeError('Expected exactly one source weighting assignment')
    namespace = dict(vars(legacy))
    exec(compile(ast.fix_missing_locations(ast.Module(body=[function],type_ignores=[])),str(path),'exec'),namespace)
    adapted = namespace['build_source_only_b']

    def select(datasets,graph,genes,config=None,seed=20020130):
        if len(datasets)!=3 or len({d.name for d in datasets})!=3:
            raise ValueError('Provide exactly three distinct SOURCE training partitions')
        if config is None:
            config = legacy.DModeReliabilityConfig()
        result = adapted(datasets,graph,genes,config,seed)
        result['manifest'].update(method='rev02 source-balanced legacy Full B2 reliability screening',
            dataset_weight='omega_d = Q_d / sum(Q_k); no n_train multiplier',
            legacy_selector_sha256=digest,adapter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            changed_formula='source weighting only; all edge reliability/selection/L1 formulas retained',
            ccrs_used=False,target_used=False,labels_used=False)
        return result

    return select

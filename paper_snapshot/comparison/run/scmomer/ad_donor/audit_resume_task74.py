"""Audit a storage-only helper migration before resuming the existing Task74."""
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path('/home/u202522802045/share/zpb')
OUT = ROOT / 'result_contain/74_scmomer_ad_donor_benchmark_20261015'


def sha(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


config_path = OUT / 'config.json'
config = json.loads(config_path.read_text())
helper = ROOT / 'comparison/run/pbmc_intra_common.py'
old = OUT / 'source/pbmc_intra_common.py'
assert sha(old) == 'c81f72d992daffe4c03e0e9c14bbe8677c4236631f4e96aed534d19354a057e3'
for name, expected in config['source_hashes'].items():
    if Path(name) != helper:
        assert sha(Path(name)) == expected, name
import sys
sys.path.insert(0, str(ROOT / 'comparison/run'))
import pbmc_intra_common as common
import ast
def functions(path):
    return {n.name: ast.dump(n, include_attributes=False) for n in ast.parse(path.read_text()).body
            if isinstance(n, ast.FunctionDef)}
before, after = functions(old), functions(helper)
assert {name for name in before if before[name] != after.get(name)} == {'sha', 'prepare', 'evaluate'}
before_sha = before['sha']
new_tree = ast.parse(helper.read_text())
node = next(n for n in new_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sha')
assert ast.unparse(node.body[0]) == 'path = relocated(path)'
node.body = node.body[1:]
for n in ast.walk(node):
    if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == 'Path':
        raise ValueError('Unexpected additional Path conversion')
old_tree = ast.parse(old.read_text())
old_node = next(n for n in old_tree.body if isinstance(n, ast.FunctionDef) and n.name == 'sha')
class RemovePath(ast.NodeTransformer):
    def visit_Call(self, n):
        n = self.generic_visit(n)
        if isinstance(n.func, ast.Name) and n.func.id == 'Path':
            return n.args[0]
        return n
old_node = RemovePath().visit(old_node)
assert ast.dump(node, include_attributes=False) == ast.dump(old_node, include_attributes=False)
reference = ROOT / 'result_contain/69_smode_ad_donor_prediction_5seed_20261014'
ref_config = json.loads((reference / 'source/config.json').read_text())
paths = [Path(name) for name in config['source_hashes']]
paths += [Path(ref_config['benchmark']) / 'GSE214979_filtered_feature_bc_matrix.h5',
          ROOT / 'scMomer_official/panglao_10000.h5ad', ROOT / 'scMomer_official/pretrained_scmomer.pth']
checks = []
for path in paths:
    direct, relocated = sha(path), common.sha(path)
    assert direct == relocated, path
    checks.append(dict(path=str(path), resolved=str(common.relocated(path)), sha256=direct))
backup = OUT / 'config_before_resume_20261022.json'
assert not backup.exists(), 'This audit has already been applied.'
shutil.copy2(config_path, backup)
shutil.copy2(helper, OUT / 'source/pbmc_intra_common_storage_20261022.py')
config['source_hashes'][str(helper)] = sha(helper)
config['resume_20261022'] = dict(reason='storage path migration; original config and helper snapshot preserved',
    model_and_training_and_evaluator_files_unchanged=True,
    helper_used_by_task74='sha and write only; sha verified equivalent on task inputs',
    helper_prepare_and_evaluate_used_by_task74=False)
config_path.write_text(json.dumps(config, indent=2) + '\n')
audit = dict(status='PASS_STORAGE_ONLY_RESUME', timestamp=datetime.now(timezone.utc).isoformat(),
             input_checks=checks, original_config=str(backup), no_model_fit=True)
(OUT / 'resume_audit_20261022.json').write_text(json.dumps(audit, indent=2) + '\n')
print(json.dumps(audit), flush=True)

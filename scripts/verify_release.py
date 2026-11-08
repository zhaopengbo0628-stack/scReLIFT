"""Verify original source hashes and parse public Python/JSON without imports."""
from pathlib import Path
import ast
import hashlib
import json

ROOT = Path(__file__).resolve().parents[1]
manifest = json.loads((ROOT / 'provenance/SNAPSHOT_MANIFEST.json').read_text(encoding='utf-8'))
errors = []
for row in manifest['files']:
    path = ROOT / row['release_path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['source_sha256']:
        errors.append('snapshot mismatch: ' + row['release_path'])
figure_manifest = ROOT / 'provenance/FIGURE_SNAPSHOT_MANIFEST.json'
figure_rows = json.loads(figure_manifest.read_text(encoding='utf-8')) if figure_manifest.exists() else []
for row in figure_rows:
    path = ROOT / row['release_path']
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
        errors.append('figure snapshot mismatch: ' + row['release_path'])
parsed = 0
for folder in ['src', 'tests', 'scripts']:
    for path in (ROOT / folder).rglob('*.py'):
        try:
            ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
            parsed += 1
        except SyntaxError as exc:
            errors.append(str(exc))
for path in (ROOT / 'configs').glob('*.json'):
    json.loads(path.read_text(encoding='utf-8-sig'))
result = dict(status='PASS' if not errors else 'FAIL', snapshot_files=len(manifest['files']),
              figure_snapshot_files=len(figure_rows), public_python_files_parsed=parsed, errors=errors)
print(json.dumps(result, ensure_ascii=False))
if errors:
    raise SystemExit(1)

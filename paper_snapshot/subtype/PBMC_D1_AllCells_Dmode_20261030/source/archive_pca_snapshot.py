"""Preserve exact PCA script before Source integrity assertions were added."""
import hashlib,json
from pathlib import Path
run=Path(__file__).resolve().parents[1]
p=run/'source/run_model.py'
block="""    for filename,digest in prep['hashes'].items():
        if filename.startswith('artifacts/D1_'): continue
        assert sha(RUN/filename)==digest, f'Frozen Source artifact changed: {filename}'
"""
t=p.read_text(encoding='utf-8');assert block in t
q=run/'source/frozen/run_model_pca_snapshot.py'
q.write_text(t.replace(block,''),encoding='utf-8',newline='\n')
digest=hashlib.sha256(q.read_bytes()).hexdigest()
manifest=run/'PCA_sourcefit/manifest.json'
if manifest.exists():
    expected=json.loads(manifest.read_text())['input_hashes'][str(p)]
    assert digest==expected,(digest,expected)
    (run/'audit/PCA_CODE_PROVENANCE.json').write_text(json.dumps(dict(status='PASS',original_sha256=expected,
        archived_code=str(q),note='PCA ran before adding three Source-file integrity assertion lines. Numerical operations unchanged.'),indent=2)+'\n')
print('PCA_SNAPSHOT',digest)

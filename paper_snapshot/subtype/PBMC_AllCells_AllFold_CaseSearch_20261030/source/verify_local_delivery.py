"""Verify the remote receipt, extract within this result folder and verify every file."""
import hashlib
import json
import shutil
import sys
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def main():
    root = Path(__file__).resolve().parents[1]
    archive = Path(sys.argv[1]).resolve()
    receipt = json.loads((root / 'DELIVERY_RECEIPT.json').read_text(encoding='utf-8'))
    assert archive.stat().st_size == receipt['bytes']
    assert sha(archive) == receipt['sha256'], 'Archive checksum mismatch'
    print('ARCHIVE_SHA256_PASS', flush=True)
    seen = set()
    with tarfile.open(archive, 'r:gz') as package:
        for member in package:
            rel = PurePosixPath(member.name)
            assert not rel.is_absolute() and '..' not in rel.parts
            assert not any(':' in part or '\\' in part for part in rel.parts)
            target = (root / Path(*rel.parts)).resolve()
            assert target.is_relative_to(root)
            assert member.isfile(), f'Nonregular archive entry: {member.name}'
            assert member.name not in seen, f'Duplicate archive entry: {member.name}'
            seen.add(member.name)
            target.parent.mkdir(parents=True, exist_ok=True)
            with package.extractfile(member) as source, target.open('wb') as output:
                shutil.copyfileobj(source, output, 8 << 20)
    manifest = json.loads((root / 'DELIVERY_MANIFEST.json').read_text(encoding='utf-8'))
    assert len(manifest['files']) == receipt['files']
    assert seen == {record['path'] for record in manifest['files']} | {'DELIVERY_MANIFEST.json'}
    total = 0
    for index, record in enumerate(manifest['files'], 1):
        path = root / record['path']
        assert path.stat().st_size == record['size'], record['path']
        assert sha(path) == record['sha256'], record['path']
        total += record['size']
        if index % 500 == 0:
            print('VERIFIED_FILES', index, flush=True)
    required = ['中文总报告.md', 'final_report.md', 'candidate_inventory.csv',
                'candidate_evidence_matrix.csv', 'candidate_ranking.md',
                'all_candidate_deg_summary.csv', 'all_candidate_gsea_summary.csv',
                'atac_evidence_summary.csv', 'audit/FINAL_NUMERICAL_VERIFICATION.json',
                'audit/FIGURE_REVIEW.md']
    for name in required:
        assert (root / name).stat().st_size > 0, name
    result = dict(status='PASS', utc=datetime.now(timezone.utc).isoformat(),
                  archive_sha256=receipt['sha256'], files_verified=len(manifest['files']),
                  bytes_verified=total, extraction_paths_validated=True,
                  all_file_sha256_match=True, required_outputs_present=True)
    (root / 'audit/LOCAL_DELIVERY_VERIFIED.json').write_text(
        json.dumps(result, indent=2), encoding='utf-8')
    print('LOCAL_DELIVERY_PASS', json.dumps(result), flush=True)


if __name__ == '__main__':
    main()

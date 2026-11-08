"""Archive a verified immutable snapshot of primary and expressed-background results."""
import tarfile,time
from analysis_common import *


def main():
    while not (RUN/'audit/CHINESE_OVERVIEW_COMPLETE.json').exists():
        time.sleep(15)
    while not (RUN/'audit/ANCHOR_FIGURES_COMPLETE.json').exists():
        time.sleep(15)
    pieces = ['source', 'input', 'tables', 'partitions', 'deg', 'figures', 'fgsea', 'fgsea_expressed', 'audit', 'logs',
              'candidate_inventory.csv', 'all_candidate_deg_summary.csv', 'ANALYSIS_PROTOCOL.md', 'env-spec.json', '中文结果总览.md']
    files = [p for name in pieces for p in ([RUN/name] if (RUN/name).is_file() else (RUN/name).rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts]
    records = [dict(file=str(p.relative_to(RUN)), size=p.stat().st_size, SHA256=sha(p)) for p in files]
    pd.DataFrame(records).to_csv(RUN/'FILE_MANIFEST_SHA256.csv', index=False)
    archive = RUN.parent/(RUN.name+'_verified_delivery.tar.gz')
    assert not archive.exists(), 'Do not overwrite a prior delivery archive'
    with tarfile.open(archive, 'w:gz', compresslevel=4) as tf:
        for p in files:
            tf.add(p, arcname=str(p.relative_to(RUN)), recursive=False)
        tf.add(RUN/'FILE_MANIFEST_SHA256.csv', arcname='FILE_MANIFEST_SHA256.csv')
    receipt = dict(status='COMPLETE', archive=str(archive), bytes=archive.stat().st_size,
                   SHA256=sha(archive), content_files=len(files)+1,
                   includes='All primary and expressed-universe sensitivity outputs, full DEG, plots, source, databases and audit',
                   excludes='Private binary R/Python dependencies and unfinished original GSEApy run; install scripts retained')
    write(RUN/'audit/DELIVERY_ARCHIVE.json', receipt)
    print(json.dumps(receipt, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    main()

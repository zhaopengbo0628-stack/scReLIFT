"""Resume only the two identity-verified SEA-AD H5AD objects; never overwrite raw data."""
import argparse
import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

import h5py
import requests

ROOT = Path(__file__).resolve().parent
CHUNK = 4 * 1024 * 1024
LOCAL = threading.local()


def session():
    if not hasattr(LOCAL, 'session'):
        LOCAL.session = requests.Session()
        LOCAL.session.trust_env = False
    return LOCAL.session


def save(path, value):
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    os.replace(temporary, path)


def download(name):
    audit = json.loads((ROOT/(name+'.inspection.json')).read_text())
    size, etag, url = audit['size_bytes'], audit['etag'], audit['url']
    target, partial = ROOT/name, ROOT/(name+'.part')
    state_path = ROOT/(name+'.download.json')
    if target.exists():
        raise FileExistsError('Existing raw H5AD preserved; consult prior download audit')
    head = session().head(url, timeout=(10, 30))
    head.raise_for_status()
    if int(head.headers['Content-Length']) != size or head.headers['ETag'] != etag:
        raise ValueError('Object changed since pairing verification')
    state = json.loads(state_path.read_text()) if state_path.exists() else dict(
        key=audit['key'], url=url, size_bytes=size, etag=etag, chunk_bytes=CHUNK,
        completed_chunks=[], status='DOWNLOADING')
    if (state['etag'], state['size_bytes'], state['chunk_bytes']) != (etag, size, CHUNK):
        raise ValueError('Resume state mismatch')
    if state['completed_chunks'] and not partial.exists():
        raise ValueError('Resume state exists but partial file is missing')
    if not partial.exists():
        partial.touch(exist_ok=False)
    completed = set(state['completed_chunks'])

    def block(number):
        lo, hi = number*CHUNK, min((number+1)*CHUNK, size)-1
        for attempt in range(2):
            try:
                with session().get(url, headers={'Range': f'bytes={lo}-{hi}', 'If-Match': etag},
                                   stream=True, timeout=(10, 60)) as response:
                    response.raise_for_status()
                    if response.status_code != 206 or response.headers.get('Content-Range') != f'bytes {lo}-{hi}/{size}':
                        raise ValueError('Server failed exact bounded range verification')
                    written = 0
                    with partial.open('r+b') as handle:
                        handle.seek(lo)
                        for data in response.iter_content(256*1024):
                            handle.write(data)
                            written += len(data)
                        handle.flush()
                        os.fsync(handle.fileno())
                    if written != hi-lo+1:
                        raise ValueError('Incomplete range transfer')
                return number
            except requests.RequestException:
                if attempt:
                    raise

    save(state_path, state)
    try:
        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = [pool.submit(block, n) for n in range((size+CHUNK-1)//CHUNK) if n not in completed]
            for future in as_completed(futures):
                try:
                    completed.add(future.result())
                except Exception:
                    for pending in futures:
                        pending.cancel()
                    raise
                state.update(completed_chunks=sorted(completed),
                    downloaded_verified_bytes=sum(min(CHUNK, size-n*CHUNK) for n in completed),
                    updated_utc=datetime.now(timezone.utc).isoformat())
                save(state_path, state)
                if len(completed) % 20 == 0:
                    print(name, state['downloaded_verified_bytes'], '/', size, flush=True)
        if partial.stat().st_size != size:
            raise ValueError('Final file size mismatch')
        with h5py.File(partial, 'r') as handle:
            if len(handle['obs'][audit['obs_index']]) != audit['rows']:
                raise ValueError('Downloaded HDF5 cell count differs from metadata audit')
        digest = hashlib.sha256()
        with partial.open('rb') as handle:
            for data in iter(lambda: handle.read(16*1024*1024), b''):
                digest.update(data)
        partial.rename(target)
        state.update(status='COMPLETE', sha256=digest.hexdigest(), hdf5_verified=True)
        save(state_path, state)
        print('COMPLETE', name, digest.hexdigest(), flush=True)
    except Exception as error:
        state.update(status='FAILED_RESUMABLE', error=repr(error))
        save(state_path, state)
        raise


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', type=Path, required=True)
    ROOT = parser.parse_args().raw
    report = json.loads((ROOT/'PAIRING_REPORT.json').read_text())
    if report['status'] != 'COMPATIBLE_PAIRED_SUBSET' or not report['permitted_full_download']:
        raise ValueError('Pairing not verified; full download prohibited')
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download, report[key]) for key in ('rna', 'atac')]
        for future in as_completed(futures):
            future.result()

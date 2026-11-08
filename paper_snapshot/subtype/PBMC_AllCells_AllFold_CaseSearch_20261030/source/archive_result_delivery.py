"""Make a self-contained analysis/plot result bundle without copying raw cohorts."""
import json,tarfile,hashlib
from pathlib import Path
from datetime import datetime,timezone

ROOT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()

def include(p):
    rel=p.relative_to(ROOT);parts=rel.parts
    if len(parts)==1 and p.name in ['DELIVERY_MANIFEST.json','DELIVERY_RECEIPT.json']:return False
    if '__pycache__' in parts or p.suffix in ['.lock','.tmp']:return False
    if parts[0]=='audit' and len(parts)>1 and parts[1]=='pre_nonfinite_pathway_BH_repair':return False
    if parts[0] in ['deg_allfold_global','atac_posthoc','case_studies','figures','source','audit','logs','.aris']:return True
    if parts[0]=='analyses':
        if len(parts)>2 and parts[2] in ['R_library','vendor']:return False
        # Keep complete aggregated pathway tables/ranks and per-comparison results in the delivered bundle.
        return True
    if parts[0]=='folds':
        if 'artifacts' in parts:return False
        # Checkpoints, embeddings, frozen training sources and run metadata are reproducibility artifacts.
        return True
    if parts[0]=='input':return p.name=='datasets.json'
    return len(parts)==1 and p.suffix in ['.csv','.md','.json']

def main():
    entries=[]
    for p in sorted(ROOT.rglob('*')):
        if p.is_file() and include(p):
            entries.append(dict(path=p.relative_to(ROOT).as_posix(),size=p.stat().st_size,sha256=sha(p)))
    manifest=ROOT/'DELIVERY_MANIFEST.json'
    manifest.write_text(json.dumps(dict(files=entries,created=datetime.now(timezone.utc).isoformat(),
        raw_counts_included=False,input_registry='input/datasets.json',
        D1_checkpoints='Already delivered separately in PBMC_D1_AllCells_Dmode_20261030; verified hashes in audit/CHECKPOINT_VERIFICATION.csv',
        excluded_initial_BH_outputs='Preserved on server under audit/pre_nonfinite_pathway_BH_repair; repaired audited tables delivered'),indent=2),encoding='utf-8')
    archive=ROOT.parent/(ROOT.name+'_analysis_delivery.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=2,dereference=True) as tf:
        for rec in entries:tf.add(ROOT/rec['path'],arcname=rec['path'],recursive=False)
        tf.add(manifest,arcname=manifest.name,recursive=False)
    digest=sha(archive)
    (ROOT/'DELIVERY_RECEIPT.json').write_text(json.dumps(dict(archive=str(archive),sha256=digest,
        bytes=archive.stat().st_size,files=len(entries),analysis_bundle=True),indent=2))
    print('DELIVERY_ARCHIVE_COMPLETE',digest,archive.stat().st_size,flush=True)

if __name__=='__main__':main()

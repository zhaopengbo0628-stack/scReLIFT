from pathlib import Path
import json

base=Path('/home/u2021082/share/zpb')
for key in ['D1','D2','D3','D4']:
    root=base/'data/staging/PBMC/legacy_dmode_data_20261004/derived/common'/key
    p=root/'barcodes.tsv'
    print('STAGING',key,'barcode_head',p.read_text().splitlines()[:3])
    print('FILES',[(x.name,x.stat().st_size) for x in root.iterdir() if x.is_file()])
for key in ['PBMC-MO-001','PBMC-MO-002','PBMC-MO-003','PBMC-MO-004']:
    root=base/'data/raw/PBMC'/key
    print('RAW',key)
    print('FILES',[(x.name,x.stat().st_size) for x in root.iterdir() if x.is_file()])
    p=root/'DATASET_IDENTITY.json'
    print(p.read_text() if p.exists() else 'NO_IDENTITY')

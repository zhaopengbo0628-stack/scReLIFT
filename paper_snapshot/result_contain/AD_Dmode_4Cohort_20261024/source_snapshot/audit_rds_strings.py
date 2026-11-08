"""Inspect every serialized RDS slot for identity strings without loading assay arrays."""
import gzip
import hashlib
import json
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
from rdata.parser._parser import Parser, RObject, RObjectInfo, RObjectType
from rdata.parser._xdr import ParserXDR

OUT = Path(__file__).resolve().parent.parent / 'data_audit'
RDS = Path('E:/my_research/RAWdata/AD/2/GSE268609_seurat_object.rds')


class StringAudit(ParserXDR):
    def __init__(self, stream):
        Parser.__init__(self, expand_altrep=False)
        self.file = stream
        self.strings = Counter()
        self.numeric_bytes = 0
        self.objects = 0

    def _parse_array_values(self, dtype, length):
        dtype = np.dtype(dtype)
        if length < 0:
            raise ValueError('Unsupported long vector')
        if length <= 256:
            return super()._parse_array_values(dtype, length)
        remaining = length * dtype.itemsize
        self.numeric_bytes += remaining
        while remaining:
            data = self.file.read(min(8 << 20, remaining))
            if not data:
                raise EOFError('Truncated numeric array')
            remaining -= len(data)
        return np.empty(0, dtype=dtype)

    def parse_string(self, length):
        value = self.file.read(length)
        if len(value) != length:
            raise EOFError('Truncated string')
        text = value.decode('utf-8', errors='replace')
        if (0 < len(text) <= 2048 and not re.fullmatch(r'[ACGTN]{12,}(?:[-_]\d+)*', text)
                and not re.fullmatch(r'(?:ENS[GTP]\d+(?:\.\d+)?|chr\w+[:-]\d+[-:]\d+)', text)):
            self.strings[text] += 1
        return value

    def parse_R_object(self, reference_list=None, bytecode_rep_list=None, info_int=None):
        if reference_list is None:
            reference_list = []
        kind = int.from_bytes(self.file.peek(4)[:4], 'big') & 255 if info_int is None else None
        if kind in {248, 249}:
            # R serialize.c InStringVec: package/namespace names, then AddReadRef.
            self.file.read(4)
            if self.parse_int() != 0:
                raise ValueError('Named persistent namespace strings unsupported')
            for _ in range(self.parse_int()):
                self.parse_R_object(reference_list, bytecode_rep_list)
            obj = RObject(RObjectInfo(RObjectType.EMPTYENV, False, False, False, 0, 0), None, None)
            reference_list.append(obj)
        else:
            obj = super().parse_R_object(reference_list, bytecode_rep_list, info_int)
        self.objects += 1
        # Keep symbol references and list tags; release completed feature/cell vectors.
        if obj.info.type in {RObjectType.STR, RObjectType.VEC, RObjectType.EXPR}:
            obj.value = []
        return obj


if __name__ == '__main__':
    started = time.time()
    status = dict(input=str(RDS), size_bytes=RDS.stat().st_size, status='RUNNING',
                  scope='all serialized slots; numeric arrays deliberately skipped',
                  counts_validated=False, donor_identity_resolved=False)
    status_path = OUT / 'GSE268609_rds_all_slots_status.json'
    if (OUT / 'GSE268609_rds_strings.json').exists():
        raise FileExistsError('Preserve completed string audit')
    status_path.write_text(json.dumps(status, indent=2))
    parser = None
    try:
        with gzip.open(RDS, 'rb') as stream:
            if stream.read(2) != b'X\n':
                raise ValueError('Expected gzip XDR RDS')
            parser = StringAudit(stream)
            parser.parse_all()
        status.update(status='COMPLETE', serialized_objects=parser.objects,
                      skipped_numeric_bytes=parser.numeric_bytes,
                      unique_nonfeature_strings=len(parser.strings))
    except Exception as error:
        status.update(status='FAILED_PARTIAL', error=repr(error))
    if parser is not None:
        strings_path = OUT / 'GSE268609_rds_strings.json'
        strings_path.write_text(json.dumps(dict(parser.strings), ensure_ascii=False, indent=2), encoding='utf-8')
        status['strings_sha256'] = hashlib.sha256(strings_path.read_bytes()).hexdigest()
    status['wall_seconds'] = time.time() - started
    status_path.write_text(json.dumps(status, indent=2))
    print(json.dumps(status), flush=True)

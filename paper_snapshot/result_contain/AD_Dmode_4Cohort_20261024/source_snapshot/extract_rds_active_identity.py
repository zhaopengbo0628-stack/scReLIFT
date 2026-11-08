"""Recover the previously unaudited active.ident factor and its explicit cell names."""
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rdata
from rdata.parser._parser import Parser, RObjectType
from rdata.parser._xdr import ParserXDR

OUT = Path(__file__).resolve().parent.parent


class FoundIdentity(Exception):
    pass


class IdentityParser(ParserXDR):
    def __init__(self, stream):
        Parser.__init__(self)
        self.file = stream
        self.capture_next = self.capture = False
        self.identity = None

    def _parse_array_values(self, dtype, length):
        dtype = np.dtype(dtype)
        if length < 0:
            raise ValueError('Long vector unsupported')
        if length <= 256 or self.capture:
            return super()._parse_array_values(dtype,length)
        remaining = length*dtype.itemsize
        while remaining:
            data = self.file.read(min(8<<20, remaining))
            if not data:
                raise EOFError('Truncated numeric array')
            remaining -= len(data)
        return np.empty(0,dtype=dtype)

    def parse_R_object(self, *args, **kwargs):
        capture = self.capture_next
        self.capture_next = False
        if capture:
            self.capture = True
        obj = super().parse_R_object(*args, **kwargs)
        if capture:
            attributes = rdata.conversion.convert(obj.attributes)
            levels = np.asarray(attributes['levels']).astype(str)
            codes = np.asarray(obj.value, dtype=np.int64)
            barcodes = np.asarray(attributes['names']).astype(str)
            if len(codes) != len(barcodes) or np.any(codes<1) or np.any(codes>len(levels)):
                raise ValueError('Factor codes or named cell axis invalid')
            self.identity = pd.DataFrame(dict(barcode=barcodes,active_ident_code=codes,
                                               author_active_ident=levels[codes-1]))
            raise FoundIdentity
        symbol = obj.referenced_object if obj.info.type == RObjectType.REF else obj
        if symbol is not None and symbol.info.type == RObjectType.SYM and symbol.value.value == b'active.ident':
            self.capture_next = True
        if not self.capture and obj.info.type in {RObjectType.STR,RObjectType.VEC,RObjectType.EXPR}:
            obj.value = []
        return obj


if __name__ == '__main__':
    path = OUT/'data_audit/GSE268609_active_identity_by_cell.csv.gz'
    if path.exists():
        raise FileExistsError('Preserve recovered identities')
    with gzip.open('E:/my_research/RAWdata/AD/2/GSE268609_seurat_object.rds','rb') as stream:
        assert stream.read(2) == b'X\n'
        parser = IdentityParser(stream)
        try:
            parser.parse_all()
        except FoundIdentity:
            pass
    assert parser.identity is not None
    frame = parser.identity
    assert len(frame) == 153530 and frame.barcode.is_unique
    original = pd.read_csv('E:/my_research/data/audit/data_organization_20261021/AD_GSE268609/author_subset/metadata.csv',
                           usecols=['barcode','orig.ident','SampleNumber','Group'])
    merged = frame.merge(original,on='barcode',validate='one_to_one')
    assert len(merged) == len(frame)
    merged.to_csv(path,index=False)
    cross = merged.groupby(['orig.ident','SampleNumber','Group','author_active_ident'],dropna=False).size().rename('cells').reset_index()
    cross.to_csv(OUT/'data_audit/GSE268609_active_ident_crosswalk.csv',index=False)
    print(cross.to_string(index=False))
    print(json.dumps(dict(status='ACTIVE_IDENT_RECOVERED_NOT_YET_BIOBANK_VERIFIED',cells=len(merged),
                          active_ident_values=merged.author_active_ident.nunique())))

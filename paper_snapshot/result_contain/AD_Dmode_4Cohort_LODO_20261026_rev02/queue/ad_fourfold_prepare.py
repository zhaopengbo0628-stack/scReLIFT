from __future__ import annotations
import ast, gc, hashlib, json, os, sys, time
from pathlib import Path
import h5py, numpy as np, pandas as pd
from scipy import sparse
from anndata.io import read_elem
ROOT=Path('/home/u202522802045/share/zpb')
OUT=ROOT/'result_contain/AD_Dmode_4Cohort_LODO_20261026_rev02'
QUEUE=OUT/'queue'
SEEDS=[20020130,20020131,20020132]
COHORTS=['GSE214979','GSE268609','SEAAD_MTG','GSE272082']
FOLDS={
 'fold1':('GSE214979',['GSE268609','SEAAD_MTG','GSE272082']),
 'fold2':('GSE268609',['GSE214979','SEAAD_MTG','GSE272082']),
 'fold3':('SEAAD_MTG',['GSE214979','GSE268609','GSE272082']),
 'fold4':('GSE272082',['GSE214979','GSE268609','SEAAD_MTG']),
}
ALLOWED2={1,2,3,4,7,8,26,27,28,30,31,34,36}
sys.path[:0]=[str(ROOT/'main/core/regulatory/common/legacy_support'),str(ROOT/'result_contain/AD_Dmode_4Cohort_20261024/queue')]
from v5d.regulatory_graph import read_tss,build_candidate_graph,parse_peak
from v5d.dmode_teacher import fit_dmode_field
from source_balanced_selector import load_selector
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''): h.update(b)
 return h.hexdigest()
def write(p,x):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);q=p.with_suffix(p.suffix+'.tmp');q.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n');q.replace(p)
def state(s,**kw):
 print(s,kw,flush=True);write(QUEUE/'cpu_status.json',dict(status=s,time=time.time(),**kw))
class TenX:
 def __init__(self,path,rows,cols):
  self.path=Path(path);self.rows=np.asarray(rows,dtype=np.int64);self.cols=np.asarray(cols,dtype=np.int64)
 def blocks(self):
  with h5py.File(self.path,'r') as f:
   x=f['matrix'];ptr=x['indptr'][:];shape=tuple(x['shape'][:]);assert ptr[-1]==len(x['data'])
   for first in range(0,len(self.rows),512):
    rr=self.rows[first:first+512];lo,hi=int(ptr[rr[0]]),int(ptr[rr[-1]+1]);ip=ptr[rr[0]:rr[-1]+2]-lo
    m=sparse.csr_matrix((x['data'][lo:hi],x['indices'][lo:hi],ip),shape=(rr[-1]-rr[0]+1,shape[0]))
    yield first,m[rr-rr[0]][:,self.cols].tocsr()
class H5AD:
 def __init__(self,path,rows,cols):
  self.path=Path(path);self.rows=np.asarray(rows,dtype=np.int64);self.cols=np.asarray(cols,dtype=np.int64)
 def blocks(self):
  with h5py.File(self.path,'r') as f:
   x=f['X'];ptr=x['indptr'][:];ncol=int(x.attrs['shape'][1])
   for first in range(0,len(self.rows),512):
    rr=self.rows[first:first+512];lo,hi=int(ptr[rr[0]]),int(ptr[rr[-1]+1]);ip=ptr[rr[0]:rr[-1]+2]-lo
    m=sparse.csr_matrix((x['data'][lo:hi],x['indices'][lo:hi],ip),shape=(rr[-1]-rr[0]+1,ncol))
    yield first,m[rr-rr[0]][:,self.cols].tocsr()
class Concat:
 def __init__(self,parts,mappings=None,width=None):
  self.parts=parts;self.mappings=mappings;self.width=width;self.rows=np.arange(sum(len(p.rows) for p in parts));self.path=Path('concat:'+','.join(str(x.path) for x in parts))
 def blocks(self):
  off=0
  for j,p in enumerate(self.parts):
   for first,m in p.blocks():
    if self.mappings is not None:
     m=sparse.csr_matrix((m.data,self.mappings[j][m.indices],m.indptr),shape=(m.shape[0],self.width));m.sort_indices()
    yield off+first,m
   off += len(p.rows)

def read_obs(path):
 with h5py.File(path,'r') as f: obs=read_elem(f['obs']);var=read_elem(f['var'])
 return obs,var
def tenx_features(path):
 with h5py.File(path,'r') as f:
  ids=f['matrix/features/id'][:].astype(str);typ=f['matrix/features/feature_type'][:].astype(str)
 return ids[typ=='Gene Expression'],ids[typ!='Gene Expression']
def norm_stats(reader,genes,weights,dest):
 if dest.exists(): return pd.read_csv(dest)
 sums=np.zeros(len(genes));sq=sums.copy();det=sums.copy()
 for first,m in reader.blocks():
  total=np.asarray(m.sum(1)).ravel(); z=sparse.diags(10000/np.maximum(total,1)).dot(m).tocsr();z.data=np.log1p(z.data)
  assert np.isfinite(m.data).all() and (m.data>=0).all() and np.equal(m.data,np.floor(m.data)).all()
  w=weights[first:first+len(total)];sums += np.asarray(z.T@w).ravel();sq += np.asarray(z.power(2).T@w).ravel();det += np.asarray((m>0).T@(w>0).astype(np.int64)).ravel()
  if first%10240==0: print('HVG_BLOCK',dest.stem,first,flush=True)
 var=np.maximum(sq-sums*sums,0);disp=var/np.maximum(sums,1e-6);order=np.argsort(disp,kind='stable');rank=np.empty(len(genes),int);rank[order]=np.arange(len(genes),0,-1)
 t=pd.DataFrame({'gene_id':genes,'mean':sums,'variance':var,'dispersion':disp,'hvg_rank':rank,'top2000':rank<=2000,'train_detected_cells':det});tmp=dest.with_suffix('.tmp');t.to_csv(tmp,index=False);tmp.replace(dest);return t
def split(meta,donor,label,name):
 p=QUEUE/(name+'_donor_split.csv')
 if p.exists():
  d=pd.read_csv(p,dtype=str);mp=dict(zip(d.donor_id,d.split));assert set(mp)==set(meta[donor].astype(str));tr=np.flatnonzero(~meta[donor].astype(str).isin([k for k,v in mp.items() if v=='val']));va=np.flatnonzero(meta[donor].astype(str).isin([k for k,v in mp.items() if v=='val']));return tr,va
 tab=meta[[donor,label]].copy();tab[donor]=tab[donor].astype(str);g=tab.groupby(donor)[label].first();rng=np.random.default_rng(20020130);val=[]
 for _,ids in g.groupby(g):
  a=np.asarray(ids.index);val.extend(rng.permutation(a)[:max(1,min(len(a)-1,round(.2*len(a))))])
 out=pd.DataFrame({'donor_id':g.index,'split':np.where(g.index.isin(val),'val','train')});out.to_csv(p,index=False)
 mask=meta[donor].astype(str).isin(val).to_numpy();return np.flatnonzero(~mask),np.flatnonzero(mask)
def source_defs():
 p1=ROOT/'data/benchmark/AD/GSE214979/author_qc_v1_20261012_bundle/post_qc';c1=pd.read_csv(p1/'cells.csv',dtype={'donor_id':str});genes=pd.read_csv(p1/'rna_features.tsv',sep='\t').gene_id.astype(str).to_numpy();assert len(genes)==36601
 r1=TenX(p1/'GSE214979_filtered_feature_bc_matrix.h5',np.arange(len(c1)),np.load(p1/'rna_feature_index.npy'));a1=TenX(r1.path,r1.rows,np.load(p1/'atac_feature_index.npy'));pk1=np.load(ROOT/'result_contain/AD_Dmode_4Cohort_20261024/regulatory_edges/GSE214979/peak_names.npy',allow_pickle=True).astype(str);tr1,va1=split(c1,'donor_id','disease_status','GSE214979')
 p2=ROOT/'data/staging/AD/GSE268609/verified_csr_20261024';rp=p2/'rna_counts.h5ad';ap=p2/'atac_counts.h5ad';s2all,v2=read_obs(rp);s2all=s2all.reset_index().rename(columns={'_index':'barcode'});mask=s2all['orig.ident'].astype(int).isin(ALLOWED2).to_numpy();s2=s2all.loc[mask].reset_index(drop=True);rows=np.flatnonzero(mask);g2=np.asarray(v2['id']).astype(str);lookup={g:i for i,g in enumerate(g2)};r2=H5AD(rp,rows,np.asarray([lookup[g] for g in genes]));_,v2a=read_obs(ap);pk2=np.asarray(v2a['id']).astype(str);a2=H5AD(ap,rows,np.arange(len(pk2)));s2['sample_group_id']=s2['orig.ident'].astype(str);s2['supervision_status']='sample_group_level_only';tr2=np.arange(len(s2));va2=np.array([],dtype=np.int64)
 p3=ROOT/'result_contain/AD_Dmode_4Cohort_20261024/data_audit/seaad_source3_eligible_counts';s3=pd.read_csv(p3/'cells.csv.gz',dtype={'Donor ID':str});orig=ROOT/'result_contain/AD_Dmode_Goal_20261024/rev02_three_source_seaad_holdout/preprocessing/seaad_paired_counts';g3=pd.read_csv(orig/'rna_features.tsv',sep='\t').gene_ids.astype(str).to_numpy();lookup3={g:i for i,g in enumerate(g3)};r3=H5AD(p3/'rna_counts.h5',np.arange(len(s3)),np.asarray([lookup3[g] for g in genes]));pk3=pd.read_csv(orig/'atac_features.tsv',sep='\t').iloc[:,0].astype(str).to_numpy();a3=H5AD(p3/'atac_counts.h5',np.arange(len(s3)),np.arange(len(pk3)));tr3,va3=split(s3,'Donor ID','Overall AD neuropathological Change','SEAAD_MTG')
 with h5py.File(ap) as f: assert np.array_equal(read_elem(f['obs']).index.astype(str),s2all.iloc[:,0].astype(str))
 assert len(s2['sample_group_id'].unique())==13 and set(s2.Group.astype(str))=={'AD','HA'}
 assert len(s3)==113019 and s3['Donor ID'].nunique()==24 and not s3['Overall AD neuropathological Change'].eq('Low').any()
 pm=pd.read_csv(ROOT/'data/raw/AD/GSE272082/original_20261024/manifests/sample_manifest.csv');pm=pm[(pm.modality=='GEX')&(pm.file_status=='SUCCESS')].copy();parts_r=[];parts_a=[];met=[];peaks4=[]
 for _,row in pm.iterrows():
  path=Path(row.absolute_path);gg,pp=tenx_features(path);assert np.array_equal(gg,genes)
  with h5py.File(path,'r') as f: n=int(f['matrix/shape'][1]);bc=f['matrix/barcodes'][:].astype(str)
  qr=TenX(path,np.arange(n),np.arange(len(gg)));detected=np.concatenate([m.getnnz(axis=1) for _,m in qr.blocks()]);keep=np.flatnonzero((detected>200)&(detected<10000))
  met.extend([{'barcode':f'{row.library_id}:{bc[x]}','count_row':int(x),'count_file':str(path),'library_id':row.library_id,'donor_id':str(row.donor_id),'disease_status':row['disease state'],'tissue':row.tissue,'qc_status':'RNA_only_rebuilt_QC_not_author_final'} for x in keep])
  parts_r.append(TenX(path,keep,np.arange(len(gg))));parts_a.append(TenX(path,keep,np.arange(len(gg),len(gg)+len(pp))));peaks4.append(pp)
 pk4=np.unique(np.concatenate(peaks4));maps4=[np.searchsorted(pk4,p) for p in peaks4]
 s4=pd.DataFrame(met);r4=Concat(parts_r);a4=Concat(parts_a,maps4,len(pk4));tr4,va4=split(s4,'donor_id','disease_status','GSE272082')
 return genes,{'GSE214979':(r1,a1,c1,pk1,tr1,va1,'donor_id'),'GSE268609':(r2,a2,s2,pk2,tr2,va2,'sample_group_id'),'SEAAD_MTG':(r3,a3,s3,pk3,tr3,va3,'Donor ID'),'GSE272082':(r4,a4,s4,pk4,tr4,va4,'donor_id')}

def map_peaks(source,canon):
 path=ROOT/'result_contain/AD_Dmode_4Cohort_20261024/source_snapshot/harmonize_frozen.py';tree=ast.parse(path.read_text());node=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='_overlap_mapping');ns=dict(np=np,parse_peak=parse_peak)
 exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 return ns['_overlap_mapping'](source,canon)[0]
def build_dense(reader,n,selected):
 z=np.lib.format.open_memmap(n,mode='w+',dtype=np.float32,shape=(reader.rows.size,len(selected)))
 for first,m in reader.blocks():
  total=np.asarray(m.sum(1)).ravel();m=sparse.diags(10000/np.maximum(total,1)).dot(m).tocsr();m.data=np.log1p(m.data);z[first:first+len(total)]=m[:,selected].toarray()
 z.flush();return z
def build_atac(reader,mapping,n,dirp):
 chunks=[]
 for _,m in reader.blocks():
  keep=m.data>0;rr=np.repeat(np.arange(m.shape[0]),np.diff(m.indptr))[keep];cc=m.indices[keep];ok=mapping[cc]>=0
  chunks.append(sparse.coo_matrix((np.ones(ok.sum(),np.float32),(rr[ok],mapping[cc[ok]])),shape=(m.shape[0],n)).tocsr())
 b=sparse.vstack(chunks).tocsr();b.data[:]=1;sparse.save_npz(dirp/'atac_binary.npz',b);return b
def main():
 OUT.mkdir(parents=True,exist_ok=True);QUEUE.mkdir(exist_ok=True);(OUT/'hvg').mkdir(exist_ok=True);genes,defs=source_defs()
 pd.DataFrame([{'cohort':n,'cells':len(v[2]),'groups_or_donors':int(v[2][v[6]].astype(str).nunique()),'source2_supervision':'sample_group_level_only' if n=='GSE268609' else 'verified_or_pathology'} for n,v in defs.items()]).to_csv(OUT/'source_manifest.csv',index=False)
 stats={}
 for n,(rna,_,meta,peaks,tr,va,donor) in defs.items():
  dest=OUT/'cohorts'/n;dest.mkdir(parents=True,exist_ok=True)
  if n!='GSE272082':meta['count_row']=rna.rows;meta['count_file']=str(rna.path)
  meta.to_csv(dest/'cells.csv.gz',index=False);np.save(dest/'peak_names.npy',peaks);np.savez(dest/'metadata.npz',train_index=tr,val_index=va)
  state('HVG_STATS',cohort=n,cells=len(meta));w=np.zeros(len(meta));ids=meta.iloc[tr][donor].astype(str);cnt=ids.value_counts();w[tr]=1/(ids.map(cnt).to_numpy()*len(cnt)) if len(ids) else 1/len(tr);stats[n]=norm_stats(rna,genes,w,OUT/'hvg'/f'{n}.csv')
 for fold,(target,sources) in FOLDS.items():
  fd=OUT/'folds'/fold;fd.mkdir(parents=True,exist_ok=True);state('FOLD_START',fold=fold,target=target,sources=sources)
  if (fd/'COMPLETE.json').exists():
   for p,h in json.loads((fd/'COMPLETE.json').read_text())['hashes'].items():assert sha(p)==h,p
   continue
  top=[set(stats[n].loc[stats[n].top2000,'gene_id']) for n in sources];inter=sorted(set.intersection(*top));rank=np.mean([stats[n].set_index('gene_id').loc[genes].hvg_rank.to_numpy()/len(genes) for n in sources],axis=0)
  method='SOURCE_top2000_intersection' if len(inter)>=256 else 'SOURCE_equal_mean_rank_top2000';selected=np.flatnonzero(np.isin(genes,inter)) if len(inter)>=256 else np.lexsort((genes,rank))[:2000]
  pd.DataFrame({'gene_id':inter}).to_csv(fd/'source_hvg_intersection.csv',index=False);np.save(fd/'gene_ids.npy',genes[selected]);write(fd/'hvg_audit.json',dict(fold=fold,target=target,sources=sources,intersection=len(inter),selected=len(selected),method=method,target_used=False))
  peaks=[]
  for n in sources:peaks.extend(defs[n][3].tolist())
  ints=sorted(parse_peak(p) for p in peaks);merged=[]
  for c,s,e in ints:
   if not merged or merged[-1][0]!=c or s>=merged[-1][2]:merged.append([c,s,e])
   else:merged[-1][2]=max(merged[-1][2],e)
  canon=np.asarray([f'chr{c}:{s}-{e}' for c,s,e in merged],str);np.save(fd/'canonical_peak_names.npy',canon)
  tss=read_tss(ROOT/'data/reference/genome_annotation/Homo_sapiens/GRCh38/gencode_32/gencode.v32.primary_assembly.annotation.gtf.gz')
  missing=[g for g in genes[selected] if g not in tss];assert not missing
  datasets=[];definitions=[]
  for n in sources:
   rna,atac,meta,native,tr,va,donor=defs[n];dest=fd/n;dest.mkdir(exist_ok=True)
   state('RNA_ATAC_CACHE',fold=fold,source=n)
   rr=build_dense(rna,dest/'rna.npy',selected);mapping=map_peaks(native,canon);np.save(dest/'peak_mapping.npy',mapping);b=build_atac(atac,mapping,len(canon),dest)
   assert np.all(b.getnnz(axis=1)>0),'Zero ATAC rows require audit'
   np.savez(dest/'metadata.npz',train_index=tr,val_index=va,test_index=np.array([],dtype=np.int64))
   np.save(dest/'canonical_peak_available.npy',np.bincount(mapping[mapping>=0],minlength=len(canon))>0)
   definitions.append((n,rr,b,meta,tr,va,donor));datasets.append(type('D',(),dict(name=n,rna=rr,atac=b,train=tr,val=va,canonical_peak_available=np.load(dest/'canonical_peak_available.npy')))())
  graph,_=build_candidate_graph(genes[selected],canon,tss,250000,50);graph['peak_names']=canon;np.savez_compressed(fd/'candidate_graph.npz',**graph)
  state('STABLE_REGULATORY_SELECTION',fold=fold,candidate_edges=int(graph['candidate_mask'].sum()))
  selector=load_selector(ROOT/'main/core/regulatory/common/legacy_support');sel=selector(datasets,graph,genes[selected],seed=20020130);assert sel['B_D'].nnz>0;sparse.save_npz(fd/'B_D.npz',sel['B_D']);edges=sel['B_D'].tocoo()
  pd.DataFrame({'gene_id':genes[selected][edges.col],'peak':canon[edges.row],'weight':edges.data}).to_csv(fd/'source_regulatory_edges.csv',index=False);write(fd/'selector_manifest.json',sel['manifest'])
  for n,rr,b,meta,tr,va,donor in definitions:
   field,par,man=fit_dmode_field(b,sel['B_D'],tr);np.save(fd/n/'field.npy',field);np.savez_compressed(fd/n/'field_state.npz',**par);write(fd/n/'field_manifest.json',man)
  files=[p for p in fd.rglob('*') if p.is_file() and p.name!='COMPLETE.json'];files += [Path(__file__),QUEUE/'protocol.json']
  write(fd/'COMPLETE.json',dict(status='COMPLETE',fold=fold,target=target,sources=sources,genes=len(selected),canonical_peaks=len(canon),edges=int(sel['B_D'].nnz),source2_labels_used=False,target_used=False,hashes={str(p):sha(p) for p in files}))
  del datasets,definitions,rr,b,field,sel,graph;gc.collect()
 write(OUT/'CPU_COMPLETE.json',dict(status='COMPLETE',folds=list(FOLDS),seeds=SEEDS,source2_groups_ad=[1,2,3,4,26,28,30,34],source2_groups_ha=[7,8,27,31,36],source2_supervision='sample_group_level_only',target_used=False))
 state('COMPLETED',manifest=str(OUT/'CPU_COMPLETE.json'))
if __name__=='__main__':
 try:main()
 except Exception as exc:
  state('FAILED',error=repr(exc));raise



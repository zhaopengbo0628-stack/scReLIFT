"""Finalize immutable local-download bundle after all training/plot workers finish."""
import os
import subprocess
import tarfile
import time
from common import *

def main():
    import pandas as pd
    while True:
        queue=pd.read_csv(RUN/'JOB_QUEUE.csv')
        if queue.Status.eq('COMPLETED').all(): break
        if queue.Status.isin(['COMPLETED','FAILED','BLOCKED']).all():
            raise RuntimeError('Queue contains failed/blocked task; refusing to declare completion')
        time.sleep(30)
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=''
    subprocess.run(['/home/u202522802045/envs/graph_attention/bin/python',str(SRC/'verify_results.py')],env=env,check=True)
    while not (RUN/'audit/POSTPROCESS_STATUS.json').exists(): time.sleep(30)
    while not (RUN/'audit/QC_RNA_PLOT_STATUS.json').exists(): time.sleep(30)
    post=json.loads((RUN/'audit/POSTPROCESS_STATUS.json').read_text());assert post['status']=='COMPLETE'
    expected=[('RNA_protocol_no_PCA',20020130),('RNA_QC_fullgene_log1p_no_PCA',20020130)]
    expected += [(method,seed) for method in ['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2'] for seed in SEEDS]
    for method,seed in expected:
        p=RUN/'figures'/method/f'seed_{seed}'
        manifest=json.loads((p/'manifest.json').read_text())
        assert manifest['status']=='COMPLETE' and manifest['cells']==9347
        assert sha(p/'allcells_umap.png')==manifest['plot_sha256']
        assert sha(p/'plot_data.csv')==manifest['data_sha256']
    # Keep final plotting code/data and the full prepared caches, not only a screenshot.
    code=[p for p in SRC.rglob('*') if p.is_file() and '__pycache__' not in p.parts]
    write(RUN/'audit/CODE_INPUT_MANIFEST.json',dict(status='COMPLETE',
          code_hashes={str(p.relative_to(RUN)):sha(p) for p in code},
          gtf='/home/u202522802045/glue_156478/graph_files/gencode.v32.primary_assembly.annotation.gtf.gz',
          gtf_sha256='0c36c37abb0c88637b04a8e401c93f0199a65bbdd0702c9e13bdfbadc053fe1e',
          prepared_artifact_hashes='artifacts/PREPARATION_COMPLETE.json',
          raw_input_evidence='DATA_READINESS.csv and audit/*_data_audit.json'))
    report=RUN/'final_report.md'
    text=report.read_text(encoding='utf-8')
    text += '\n\n图件已全部完成：Source-fit PCA、Target-fit PCA、Full B2 各三个 seeds；另有全基因 QC 后 RNA（无 HVG/PCA，cosine 邻域）与模型协议 RNA（355 基因、Source Scaler、无 PCA，Euclidean 邻域）。\n'
    text += 'RNA 两种邻域度量不同，分别用于展示输入状态；不能仅凭 UMAP 外观声称公平的定量方法优势。每图都含全部细胞、盲 Leiden 及全部 622 个 CD8 TEM 高亮，输出 600 dpi PNG、矢量 PDF 与绘图 CSV。\n'
    text += '完整性核验通过：9 组方法/seed 评估，3 个 Full B2 的 Teacher/Student 均为最终 100 轮，所有 Target barcode、embedding 和 checkpoint SHA256 均匹配。\n'
    report.write_text(text,encoding='utf-8')
    write(RUN/'audit/FINAL_COMPLETION.json',dict(status='COMPLETE',gpu_training_tasks=3,
          failed_tasks=0,pending_tasks=0,plots=len(expected),cells=9347,cd8_tem=622,
          checkpoint_reused=False,old_results_overwritten=False))
    archive=RUN.parent/(RUN.name+'_download.tar.gz')
    with tarfile.open(archive,'w:gz',compresslevel=3) as tf:
        tf.add(RUN,arcname=RUN.name,filter=lambda ti:None if '__pycache__' in Path(ti.name).parts else ti)
    digest=sha(archive)
    archive.with_suffix(archive.suffix+'.sha256').write_text(digest+'  '+archive.name+'\n')
    print('FINAL_BUNDLE_COMPLETE',archive,archive.stat().st_size,digest,flush=True)

if __name__=='__main__': main()

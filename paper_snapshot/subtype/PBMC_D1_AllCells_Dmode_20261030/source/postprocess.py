"""CPU-only posthoc views. All cells retained; no model or candidate selection."""
import itertools
import time
import traceback
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score,normalized_mutual_info_score
import anndata as ad
import scanpy as sc
from common import *

FIG=RUN/'figures';FIG.mkdir(exist_ok=True)
TABLE=RUN/'tables';TABLE.mkdir(exist_ok=True)
PALETTE=['#AEC7E8','#C5B0D5','#9467BD','#766bd3','#7200DA','#17BECF','#C49C94','#8C564B',
         '#FFBB78','#FF7F0E','#2CA02C','#E376C2','#009eff','#BCBD22','#D62728','#8DD3C7','#B3B3B3']
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,
                     'axes.spines.top':False,'axes.spines.right':False})

def render(method,seed,x,metric='euclidean'):
    out=FIG/method/f'seed_{seed}';out.mkdir(parents=True,exist_ok=True)
    if (out/'manifest.json').exists(): return
    cells=(RUN/'artifacts/D1_barcodes.tsv').read_text().splitlines()
    labs=pd.read_csv(RUN/'audit/D1_allcell_labels.csv',index_col=0).loc[cells,'cell_type'].astype(str).to_numpy()
    levels=sorted(set(labs));colors=dict(zip(levels,PALETTE))
    # Fit graph and UMAP before attaching any annotation to AnnData.
    obj=ad.AnnData(x.astype(np.float32))
    sc.pp.neighbors(obj,n_neighbors=20,use_rep='X',metric=metric,random_state=seed)
    sc.tl.umap(obj,min_dist=.25,random_state=seed)
    sc.tl.leiden(obj,resolution=1.0,random_state=seed,key_added='blind_leiden',flavor='igraph',n_iterations=2,directed=False)
    z=obj.obsm['X_umap'];cl=obj.obs['blind_leiden'].astype(str).to_numpy()
    data=pd.DataFrame(dict(barcode=cells,UMAP1=z[:,0],UMAP2=z[:,1],reference_label=labs,blind_leiden=cl,is_CD8_TEM=labs=='CD8 TEM'))
    data.to_csv(out/'plot_data.csv',index=False)
    np.save(out/'umap.npy',z)
    pd.crosstab(data.reference_label,data.blind_leiden).to_csv(out/'leiden_reference_composition.csv')
    fig,axs=plt.subplots(1,3,figsize=(14,4.2),layout='constrained')
    order=np.random.default_rng(20020130).permutation(len(labs))
    axs[0].scatter(z[order,0],z[order,1],s=2,c=[colors[v] for v in labs[order]],linewidths=0,alpha=.85)
    axs[0].set_title('Reference labels | 9,347 cells')
    cluster_levels=sorted(set(cl),key=int)
    cluster_colors={v:PALETTE[int(v)%len(PALETTE)] for v in cluster_levels}
    axs[1].scatter(z[order,0],z[order,1],s=2,c=[cluster_colors[v] for v in cl[order]],linewidths=0,alpha=.85)
    for k in cluster_levels:
        pos=np.median(z[cl==k],axis=0)
        axs[1].text(*pos,k,ha='center',va='center',fontsize=8,bbox=dict(facecolor='white',alpha=.7,edgecolor='none',pad=.5))
    axs[1].set_title(f'Blind Leiden | resolution 1.0, {len(cluster_levels)} clusters')
    tem=labs=='CD8 TEM'
    axs[2].scatter(z[~tem,0],z[~tem,1],s=1.5,c='#D9D9D9',linewidths=0,alpha=.45)
    axs[2].scatter(z[tem,0],z[tem,1],s=4,c='#766bd3',linewidths=0,alpha=.95)
    axs[2].set_title(f'All CD8 TEM | n = {tem.sum()}')
    for ax in axs:
        ax.set_xlabel('UMAP1');ax.set_ylabel('UMAP2');ax.set_xticks([]);ax.set_yticks([])
    handles=[Line2D([0],[0],marker='o',linestyle='none',color=colors[v],markersize=4,label=v) for v in levels]
    fig.legend(handles=handles,loc='outside lower center',ncol=6,frameon=False,fontsize=7)
    fig.suptitle(f'{method} | seed {seed}',fontsize=11)
    fig.savefig(out/'allcells_umap.png',dpi=600,bbox_inches='tight')
    fig.savefig(out/'allcells_umap.pdf',bbox_inches='tight')
    plt.close(fig)
    write(out/'manifest.json',dict(status='COMPLETE',method=method,seed=seed,cells=len(labs),
          cd8_tem=int(tem.sum()),input_dimensions=x.shape[1],metric=metric,
          graph_neighbors=20,min_dist=.25,leiden_resolution=1.0,
          annotation_used_for_graph_or_umap=False,cell_subsampling=False,
          palette_source='User supplied scMAGCA Fig6.R second colors block; two additional muted colors for 17 PBMC classes',
          plot_sha256=sha(out/'allcells_umap.png'),data_sha256=sha(out/'plot_data.csv'),
          interpretation='UMAP and Leiden are posthoc exploratory views; cluster separation is not independent biological validation'))
    print('PLOT_COMPLETE',method,seed,flush=True)

def summarize():
    metrics=[]
    for method in ['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2']:
        for seed in SEEDS:
            p=RUN/method/f'seed_{seed}/metrics.json'
            if p.exists(): metrics.append(json.loads(p.read_text()))
    cols=['method','seed','cells','subclasses','ari','nmi','ami','cluster_acc','macro_f1','balanced_accuracy','purity_at_15']
    df=pd.DataFrame(metrics)[cols];df.to_csv(RUN/'RESULT_SUMMARY.csv',index=False)
    means=df.groupby('method')[cols[4:]].agg(['mean','std'])
    means.to_csv(TABLE/'metric_mean_sample_sd.csv')
    stabilities=[]
    for method in ['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2']:
        paths={s:FIG/method/f'seed_{s}/plot_data.csv' for s in SEEDS}
        for a,b in itertools.combinations(SEEDS,2):
            if not paths[a].exists() or not paths[b].exists(): continue
            aa=pd.read_csv(paths[a]).set_index('barcode');bb=pd.read_csv(paths[b]).set_index('barcode')
            assert set(aa.index)==set(bb.index);bb=bb.loc[aa.index]
            stabilities.append(dict(method=method,seed_a=a,seed_b=b,cells=len(aa),
                  partition_ARI=adjusted_rand_score(aa.blind_leiden,bb.blind_leiden),
                  partition_NMI=normalized_mutual_info_score(aa.blind_leiden,bb.blind_leiden),
                  scope='all D1 blind Leiden; not CD8 TEM A/B membership stability'))
    pd.DataFrame(stabilities).to_csv(TABLE/'blind_partition_seed_stability.csv',index=False)
    done=len(df[df.method=='FullB2'])==3
    lines=['# PBMC D1 全细胞 D-mode 中文报告','',
          '状态：'+('三 seed Full B2 与 PCA 全部完成。' if done else '实验正在运行；下表仅列已完成结果。'),'',
          'D2/D3/D4 的全部 16,835 个合格细胞训练；D1 全部 9,347 个合格细胞仅用于冻结后的 RNA 推断和事后评估。',
          'D1 含 17 个参考类别，CD8 TEM 为 622 个；旧 Task99 test-only 为 1,403 个细胞，其中 CD8 TEM 93 个，本次单独存储。',
          'Source 各自重新选 Top2000 HVG，交集为 355 基因；B_D 和 F 均由全部 Source 细胞重建。',
          'Teacher/Student 各固定 100 轮，选最终 checkpoint；无独立 Source 留出验证集，日志 val 只是训练内监测。', '',
          '| 方法 | 完成 seeds | ARI 均值±样本SD | NMI 均值±样本SD | Macro-F1 均值±样本SD | Purity@15 均值±样本SD |',
          '| --- | ---: | --- | --- | --- | --- |']
    for method,g in df.groupby('method',sort=False):
        fmt=lambda k:f"{g[k].mean():.4f} ± {g[k].std(ddof=1):.4f}" if len(g)>1 else f"{g[k].mean():.4f}（单 seed）"
        lines.append(f"| {method} | {len(g)} | {fmt('ari')} | {fmt('nmi')} | {fmt('macro_f1')} | {fmt('purity_at_15')} |")
    lines += ['', 'PCA 使用完整 SVD、固定已知 K KMeans，因此三行 seed 的 PCA 数值相同；这不代表三次独立 PCA 模型训练。',
              'Source-fit PCA 是本次 D-mode 的主要基线。Target-fit PCA 使用 D1 分布拟合，属于额外描述性对照，两者均完整保留，不能按哪个更有利来挑选。',
              '所有已知 K 指标依照旧评估器，对 Target embedding 事后标准化、按 17 个参考类聚类并进行 Hungarian 匹配，不是监督分类准确率，也不能直接证明发现了新亚型。',
              'UMAP 使用全体细胞，不选择最有利 seed；CD8 TEM 图显示全部 622 个已注释细胞。盲 Leiden 不使用参考标签构图。',
              '本次没有补训 RNA-only Encoder，其历史 checkpoint 训练池不同，不能混入当前全 Source 公平对照。',
              '真实基因分组、Marker、混杂排除及独立生物学验证尚需另行检查，不能把 seed 重复或同批 RNA/ATAC 视为独立生物学重复。',
              '旧 test-only checkpoint 不适用于新增训练池与 355 基因面板，本次从头重拟合；原有结果未覆盖。', '',
              '完整配置见 EXPERIMENT_PROTOCOL.md；实际进度见 JOB_QUEUE.csv / RUN_STATUS.md；每个任务保留 checkpoint、日志、embedding 和完整 barcode。']
    if done:
        fb=df[df.method=='FullB2'];pc=df[df.method=='PCA_sourcefit'];delta=float(fb.ari.mean()-pc.ari.mean())
        lines += ['',f'Full B2 相对 Source-fit PCA 的全局 ARI 均值差为 {delta:+.4f}。'+
                  ('此指标上有数值提升，尚不能据此单独证明 CD8 TEM 新亚型的生物学真实性。' if delta>0 else '此指标上未显示优势，应如实保留并结合亚群与其他指标分析。')]
    (RUN/'final_report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')

def main():
    write(FIG/'palette.json',dict(colors=PALETTE,source='scMAGCA Fig6.R'))
    paths=[(method,s,RUN/method/f'seed_{s}') for method in ['PCA_sourcefit','PCA_targetfit_fullnorm','FullB2'] for s in SEEDS]
    failures={}
    while paths:
        progress=False
        for method,seed,p in list(paths):
            complete=(RUN/method/'manifest.json').exists() if method.startswith('PCA') else (p/'manifest.json').exists()
            if not complete: continue
            try:
                render(method,seed,np.load(p/'target_embedding_standardized.npy'))
                paths.remove((method,seed,p));progress=True;summarize()
            except Exception:
                failures[f'{method}_{seed}']=traceback.format_exc()
                write(RUN/'audit/PLOT_FAILURES.json',failures)
                paths.remove((method,seed,p))
        summarize()
        if paths:
            queue=pd.read_csv(RUN/'JOB_QUEUE.csv')
            if queue.Status.isin(['COMPLETED','FAILED','BLOCKED']).all():
                missing=[f'{m}_{s}' for m,s,p in paths]
                write(RUN/'audit/PLOT_MISSING_INPUTS.json',dict(missing=missing));break
            if not progress: time.sleep(60)
    # Protocol RNA coordinates are computed from the actual model input without PCA.
    try:
        stats=np.load(RUN/'artifacts/source_scaler.npz')
        x=np.load(RUN/'artifacts/D1_rna_all.npy')
        render('RNA_protocol_no_PCA',20020130,(x-stats['mean'])/(stats['std']+1e-6))
    except Exception:
        write(RUN/'audit/RNA_PLOT_FAILURE.json',dict(error=traceback.format_exc()))
    summarize()
    write(RUN/'audit/POSTPROCESS_STATUS.json',dict(status='COMPLETE' if not failures else 'PARTIAL',failures=failures))

if __name__=='__main__': main()

"""Show the same anchor members in all three seeds, including seeds without an independent split."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from analysis_common import *


def main():
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    inv = pd.read_csv(RUN/'candidate_inventory.csv')
    membership = pd.read_csv(RUN/'tables/candidate_membership.csv.gz')
    ora = pd.read_csv(RUN/'fgsea/all_candidate_ora_summary.csv.gz')
    q = 'FDR_BH_global_all_candidate_contrast_database_direction_tests'
    cells, labels = context()
    out = RUN/'fgsea/figures/anchor_all_three_seeds'
    out.mkdir(parents=True, exist_ok=True)
    oradir = RUN/'fgsea/figures/ora'
    oradir.mkdir(parents=True, exist_ok=True)
    colors = {'A': '#766bd3', 'B': '#FFBB78'}
    for cid in ['D1_ALL_FullB2_CD8_TEM_C1', 'D1_ALL_FullB2_NK_C1', 'D1_ALL_PCA_sourcefit_CD4_Naive_C1']:
        r = inv[inv.candidate_id.eq(cid)].iloc[0]
        mem = membership[membership.candidate_id.eq(cid) & membership.seed.eq(r.anchor_seed)]
        groups = mem.set_index('barcode').group.reindex(cells)
        fig, axes = plt.subplots(3, 3, figsize=(11, 9), layout='constrained')
        for i, seed in enumerate(SEEDS):
            for j, method in enumerate(METHODS):
                d = pd.read_csv(BASE/'figures'/method/f'seed_{seed}/plot_data.csv').set_index('barcode').loc[cells]
                z = d[['UMAP1', 'UMAP2']].to_numpy()
                ax = axes[i, j]
                ax.scatter(z[:, 0], z[:, 1], s=.7, c='#D6D6D6', alpha=.35, linewidths=0)
                same = labels == r.reference_subclass
                ax.scatter(z[same, 0], z[same, 1], s=1.4, c='#AAAAAA', alpha=.55, linewidths=0)
                for group, color in colors.items():
                    mask = groups.eq(group).fillna(False).to_numpy()
                    ax.scatter(z[mask, 0], z[mask, 1], s=3, c=color, alpha=.9, linewidths=0)
                ax.set_title(f'{method} | seed {seed}', fontsize=8)
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_xlabel('UMAP1'); ax.set_ylabel('UMAP2')
        fig.suptitle(f'{r.method}: {r.reference_subclass} | anchor {r.anchor_seed}, A={sum(groups=="A")}, B={sum(groups=="B")}\n'
                     'Identical anchor barcodes in every panel; does not assert a split exists in every seed', fontsize=10)
        fig.legend(handles=[Line2D([], [], marker='o', linestyle='', color=colors[g], label='Anchor '+g) for g in ['A', 'B']],
                   loc='outside lower center', ncol=2, frameon=False)
        fig.savefig(out/(cid+'.png'), dpi=600, bbox_inches='tight')
        fig.savefig(out/(cid+'.pdf'), bbox_inches='tight'); plt.close(fig)
        mem.to_csv(out/(cid+'_fixed_membership.csv'), index=False)
        d = ora[ora.candidate_id.eq(cid) & ora.contrast.eq('A_vs_B')]
        top = pd.concat([g.sort_values([q, 'nominal_p']).head(5) for _, g in d.groupby('database')])
        top.to_csv(oradir/(cid+'_plot_data.csv'), index=False)
        fig, ax = plt.subplots(figsize=(10, 5), layout='constrained')
        ax.scatter(-np.log10(top[q].clip(lower=1e-12)), range(len(top)), s=25+top.overlap*10,
                   c=[colors['A' if x=='A_high' else 'B'] for x in top.direction],
                   edgecolor=np.where(top[q]<.05, '#555555', '#DDDDDD'), linewidth=.8)
        ax.set_yticks(range(len(top)), [' / '.join([x.direction, x.Term]) for x in top.itertuples()], fontsize=6)
        ax.axvline(-np.log10(.05), ls='--', color='#999999', lw=.7)
        ax.set_xlabel('-log10 global BH FDR (ORA; dashed: FDR=0.05)')
        ax.set_title(f'{r.method}: {r.reference_subclass}\nDEG-ORA top terms; filled color: A-high / B-high; full tables retained', fontsize=9)
        fig.savefig(oradir/(cid+'.png'), dpi=600, bbox_inches='tight')
        fig.savefig(oradir/(cid+'.pdf'), bbox_inches='tight'); plt.close(fig)
    write(RUN/'audit/ANCHOR_FIGURES_COMPLETE.json', dict(status='COMPLETE', cases=3, all_three_seeds_shown=True,
          fixed_anchor_barcodes=True, independent_split_not_asserted_in_failed_seeds=True, dpi=600, vector_pdf=True))
    print('ANCHOR_FIGURES_COMPLETE', flush=True)


if __name__ == '__main__':
    main()

# Reproduction scope and design record

## Preserved scientific computations

The complete frozen Teacher–Student script is retained byte-for-byte. `provenance/CORE_EXTRACTION.json` records its SHA256 and the verbatim computational definitions extracted into the installable package. An AST equality check compares every extracted function/class against the original, including architecture, losses, optimizer/resume behavior, masking and evaluator. Regulatory support implementations are copied with relative-import adaptation only where required.

The portable interface is a deep module around the existing computations: callers supply explicit SOURCE inputs and frozen splits; preprocessing, checkpoint provenance and RNA-only mapping remain inside the workflow. No alternative model implementation, tuning scheme or new feature-selection formula is introduced. The portable D-mode entry point implements the three-source PBMC-style protocol, not every historical experiment variant.

The original SOURCE-only rules remain: fit preprocessing and checkpoint selection on SOURCE partitions, freeze a Student for RNA-only TARGET inference, save/hash the embedding before scoring, and preserve known-K/evaluation scaling as explicit evaluator metadata. New runs use exactly seeds 20020130, 20020131 and 20020132; summaries use sample SD with ddof=1. Historical sources/configs are not edited to disguise old seed protocols.

## Portable adaptations

New CLI orchestration removes task registry mutations, machine-specific GPU lock assumptions and fixed server paths from the public workflow. Paths are supplied by configuration. Training artifacts retain input/config hashes and resume checks. The exported inference bundle uses a weights-only tensor dictionary and NumPy preprocessing state; CPU inference requires no raw training assets. These orchestration changes have synthetic integration coverage; equivalence to all full-dataset paper scores has not been established.

CUDA remains required by the unchanged train-stage timing/RNG implementation. Training defaults are 100 epochs, patience 15, batch size 256, teacher LR 1e-3, student LR 1e-4, weight decay 1e-4, gradient clipping 1 and RNA masking 0.15. SOURCE batches are [86,85,85] for the inherited three-source protocol or [256] for S-mode. The smoke option changes epochs/patience to 2 and is excluded from formal summaries.

## Full paper reproduction

The dataset-specific original PBMC/BMMC/AD, ablation and downstream code is included in `paper_snapshot/`. These original scripts still require their exact datasets, references, historical namespaces and manifests, and often retain Linux absolute paths. Their data-path migration is not claimed complete. The portable CLI does not run donor-level AD, motif enrichment or subtype discovery as an automatic generic pipeline.

Formal AD uses its rev04 adapter, source-balanced regulatory weighting and donor/sample-group P2 probe. Follow PAPER_PIPELINES.md rather than substituting the portable PBMC D-mode selector. Exploratory raw-count pseudobulk is distinct from the formal donor-mean CP10K/log1p analysis. Subtype candidate states remain exploratory.

Source snapshots were taken from the local verified mirror, then checked against currently accessible zpb paths. Files no longer present at their mirrored relative paths remain attributed to the local transfer, not falsely certified as current remote files. Newly fetched sources have explicit live origin records. See the manifests for exact counts and checksums.

## Publication metadata still required

MIT was selected by the user on 2026-11-06. Manuscript authors/title are copied from the available manuscript snapshot. Public repository: https://github.com/zhaopengbo0628-stack/scReLIFT (publication prepared on 2026-11-08). Add the manuscript DOI/URL and frozen data/checkpoint archive locations when assigned. Code publication does not establish a full reproduction of manuscript scores.

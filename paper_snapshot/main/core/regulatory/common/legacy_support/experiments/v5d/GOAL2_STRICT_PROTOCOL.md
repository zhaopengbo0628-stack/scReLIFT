# Goal2 Strict D-Mode Protocol

This is a pre-final, user-declared experiment, not a final 15-seed conclusion.
The immutable result directory date is inherited from the October 2 launch.

## Scope

Goal1 supplies four SOURCE-only gene/cache protocols and frozen RNA backbones.
Strictness applies to model fitting and frozen target RNA inference on the
inherited cohort. Historical paired/QC cohort selection and label-stratified
splits remain a limitation. Posthoc evaluation uses target StandardScaler and
known-K KMeans identically for all methods. Target labels never select weights.

## Independent D Branches

Task33 changes only residual parameterization to the requested smooth
vector-norm bound. Fixed rho=0.02867579460144043 was selected from SOURCE
validation relative hard budgets, without target metrics or a sweep. It
redistributes per-cell absolute budgets and intrinsically changes the initial
residual Jacobian. The original source artifacts, objectives, normalization,
optimizer, schedule and source topology checkpoint criterion are retained.

Seed30 failed the predeclared joint source/target promotion gate. No additional
capacity seeds are run. Coordinate tanh is removed, but radial vector-budget
saturation persists; this intervention does not disprove every capacity limit.

The deterministic SOURCE-only square-loss GW gate compares intra-modal RNA
latent and regulatory F distances, never naive cross-modal cosine distances.
All 12 source-validation units fail the fixed stable identity/structure gate.
No OT Student, OT scientific Task, variant sweep or combined model is created.
The historical weak Sinkhorn numeric artifact was not located; no numeric
comparison is fabricated.

## scButterfly-B

Upstream version 0.0.9, commit eb31e04bb8c4abdf85c4cbecd044fcd359105caa.
All 14 upstream Python files are hash-verified and unmodified.
Four folds use seeds 20020130, 20020131 and 20020132 sequentially on one A30.
One technical smoke precedes the 12 formal units.

SOURCE RNA uses the Goal1 SOURCE-defined 443/449/523/593 gene universe.
SOURCE ATAC reuses the proven SOURCE-only overlapping half-open interval merged
union; adjacent intervals are not merged and target peaks are excluded.
Pooled SOURCE-train prevalence filtering uses ceil(0.005*N). TF-IDF document
frequencies and maximum are SOURCE-train fitted. SOURCE-validation values are
clipped to [0,1] for BCE, without validation-fitted scaling.

Architecture, widths, optimizer rates, warmups, losses, default batch64,
100/100/200 epochs and patience50 are unchanged. No T/C augmentation or labels
are used. Seed is set after Butterfly's constructor reset and before model
construction. Fresh default dimension/loss lists avoid upstream mutable-list
accumulation. Independent best SOURCE-validation checkpoints prevent upstream
end-of-training overwrite; latest checkpoints include optimizers and RNG state.
An automated exact mid-epoch resumption path is not claimed.

Target inference opens only the strict RNA cache, then uses the frozen RNA
encoder, translator mean path and ATAC decoder. Primary predicted ATAC applies
the upstream <1e-4 zero threshold. The paired test routine is never called.
No output representation is selected by target ARI. Labels are loaded only
after predictions and hashes are frozen.

## Artifacts and Stop Rule

Per-unit best/latest checkpoints, configuration, provenance/data manifest,
environment, histories, frozen predictions and metrics are retained. Annotation
classification-style columns describe posthoc Hungarian-mapped clustering,
not supervised source label transfer. ATAC-ground-truth topology is unavailable.
Reported standard deviation is over seed-specific four-fold means; one-seed
standard deviation is NA.

Finalize only after all 12 scButterfly units, metrics, checkpoint and prediction
hashes pass; mirror complete archives and verify local hashes. Update state and
ledger with actual results, then STOP. No automatic capacity+OT combination or
further scientific experiment is authorized by this goal.

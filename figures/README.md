# Frozen figure sources

`source_snapshot/` contains copies of the explicitly approved Figure 2.1 PBMC-D scoring and Figure 2.3 PBMC-imtra efficiency scripts, their small plotting inputs and reference PNGs. It also contains the registered Figure 3A vertical reproducibility plot and Figure 3B cCRE plot, with their inputs and version record. Original approved files remain unchanged. SHA256 values are in `provenance/FIGURE_SNAPSHOT_MANIFEST.json`.

These are preserved figure versions, not newly rendered results. Historical values, filenames and metadata are retained. Some plotting helpers reference the original research workspace; they have not all been made standalone. Typical dependencies are NumPy, pandas, matplotlib and, for Figure 2.1, plottable. Existing representative plots do not replace a complete source-data archive for the paper.

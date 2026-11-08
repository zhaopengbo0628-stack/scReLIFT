# Data and asset availability

Raw human single-cell data, processed matrices, model checkpoints and third-party motif/reference databases are excluded from this source release. Obtain each asset under its original access terms and retain the original cohort/barcode identity and frozen split arrays. No download URL or checksum is fabricated by this repository.

| Study | Dataset identity | Protocol notes |
| --- | --- | --- |
| PBMC D1 | 10x granulocyte-sorted 10k Multiome, ARC 1.0.0; PBMC-MO-001 | Audited paired cohort 9347 cells |
| PBMC D2 | GEO GSE196990, GSE196988 + GSE196987, HDa; PBMC-MO-002 | Audited paired cohort 6317 cells; preserve the corrected identity |
| PBMC D3 | 10x unsorted 10k Multiome, ARC 2.0.0; PBMC-MO-003 | Audited paired cohort 8105 cells |
| PBMC D4 | 10x unsorted 3k Multiome, ARC 2.0.0; PBMC-MO-004 | Audited paired cohort 2413 cells |
| BMMC | GEO GSE194122 | Four Sites; 69249 paired cells; 10 donor IDs, with cross-Site donor overlap |
| AD fold 1 | GEO GSE214979 | Author-QC paired cohort; preserve frozen donor handling |
| AD fold 2 | GEO GSE268609 | 13 technical sample groups; independence of biological donors unverified |
| AD fold 3 | SEA-AD MTG | Author metadata, paired-count mapping and frozen eligible cohort |
| AD fold 4 | Joint GSE288063_GSE278576 | Fixed 10 donors, 11 libraries, 54999 author-retained cells; ad22 technical libraries merge to one donor |

Reference resources include GENCODE v32 GRCh38 gene annotation and, for exploratory regulatory analyses, the exact ENCODE SCREEN, motif and pathway versions identified by the original scripts/manifests. FIMO/MEME is an external executable; enrichment may require R/fgsea and pathway database assets in addition to the Python analysis extras.

Before a public paper release, publish or link the precise data retrieval manifest, original frozen splits, stable gene/barcode dictionaries, preprocessing checksums and trained inference bundles through an appropriate archive. Existing local/remote research assets are not granted a new public redistribution license by MIT code licensing. The portable data contract and original preparation scripts are available now; asset hosting locations remain to be supplied.

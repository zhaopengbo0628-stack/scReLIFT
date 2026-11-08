# Observed validation: 2026-11-06

Executed on zpb using the existing Python 3.10 environment with PyTorch 2.4.1+cu121, NumPy 2.2.6, SciPy 1.15.3 and scikit-learn 1.7.2. See SYNTHETIC_CUDA_TEST.log for the captured output. The test used a uniquely named temporary directory and synthetic matrices, not paper datasets.

- Four checks passed: extracted core AST equality; Teacher/Student forward/backward; SOURCE statistics unaffected by TARGET-population changes; strict three-seed/equal-fold/sample-SD aggregation.
- Both S-mode and D-mode passed preparation, two-epoch Teacher and Student CUDA training, unchanged-checkpoint resume, RNA-only CPU inference and frozen-embedding clustering scoring.
- Modified cache evidence was rejected.
- A wheel was successfully built and installed into an isolated temporary target with existing dependencies; `python -m screlift --help` listed all six CLI commands. The existing server environment was not modified by that installation.

This validates the exercised integration path and selected invariants. It does not establish that all manuscript scores have been reproduced, all historical scripts are portable, or every optional downstream environment is installed. Full-dataset training was not launched.

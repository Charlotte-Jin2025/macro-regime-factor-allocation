# Do Macro-Financial Signals Improve U.S. Equity Factor Allocation?

**Exploratory macro-financial factor allocation research with United Kingdom and Germany cross-market checks.**

## Overview
This repository showcases the research code, methodology, automated checks, and selected figures from a three-market macro-financial factor allocation project. The U.S. is the primary market; UK and Germany provide cross-market comparisons.

## Methods
- Macroeconomic regimes and next-period factor forecasting
- Expanding-window models, Ridge regression, and portfolio allocation
- Transaction costs, turnover, statistical inference, and robustness checks
- Cross-market comparisons and computational validation

## Interpretation
The results are exploratory. Improvements against simple benchmarks do not automatically establish incremental macroeconomic timing. International results should not be interpreted as universal replication of U.S. findings.

## Explore
- `Run_Study.ipynb` — notebook entry point
- `run_study.py` — pipeline runner
- `src/` — research implementation
- `tests/` — numerical and scientific checks
- `figures/` — selected outputs from the original completed study
- `specification.json` and `cross_market_specification.json` — model settings
- `userguide.md` — detailed usage notes for the original complete package

## Reproducibility and data limitation
**This is a public-facing code-and-figures edition, not a complete standalone reproduction archive.** Raw source datasets, archived input hashes, numerical result tables, and the full report have intentionally been excluded. The original pipeline verifies exact input hashes and expects those datasets and generated outputs. It cannot be run successfully from this ZIP alone. See `data/README.md` for details. The original private archive is retained separately.

## Figure gallery
The `figures/` folder contains eight selected figures generated from the complete original project.

## Attribution
The project acknowledges Dr. Jie He for feedback and guidance. External data and literature remain subject to their respective provider terms and citations.

## Disclaimer
Historical exploratory research, not investment advice.

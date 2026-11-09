# OSSE_CrossModel_Analysis

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24092597.svg)](https://doi.org/10.5281/zenodo.24092597)

Cross-model analysis and figure scripts for the AI-weather-model truth-insertion OSSE study,
released with the manuscript *An AI-Based OSSE Framework to Evaluate Learned Physical Consistency: A Multi-Model Study Using FourCastNet, Aurora, Prithvi WxC, and GraphCast* (Diggikar, Halem, and Raman) (submitted to AGU *JGR: Machine Learning and
Computation*). They compare the four models whose experiment code is released separately:

| Repository | Model |
|---|---|
| `Aurora_OSSE` | Microsoft Aurora |
| `FourCastNetv2_OSSE` | NVIDIA FourCastNet v2-small |
| `GraphCast_OSSE` | Google DeepMind GraphCast (0.25°, 37 levels) |
| `PrithviWxC_OSSE` | NASA-IBM Prithvi WxC |

These scripts read the per-model forecast files and RMSE tables those repositories produce.
They do not run any model.

## Layout

```
analysis/
├── figures/             vertical propagation heatmaps and efficiency (analyze_vertical_propagation.py),
│                        all-level 3-day RMSE, cross-variable RMSE bars, spillover-distance decay,
│                        spatial sensitivity, perturbed-run RMSE
├── regime_synthesis/    cloud-regime (tcc, cloud2), terrain (terrain, terrain2), latitude-band/slice,
│                        T+UV synergy (tuv_plots, tuv_final), normalized-RMSE comparison, bias/lat replots;
│                        each subfolder carries its PHYSICAL_ANALYSIS.txt interpretation notes
└── perturbation_cop/    cross-model COP comparison and paper figures (+ Prithvi COP analysis)
preprocessing/           ERA5 total cloud cover and orography download, cloud-regime classification
docs/                    PROVENANCE.tsv, PATHS.md, results map, original file paths, replotting
                         instructions, COP method and results notes
environment/             frozen pip environment
```

## Important: which model runs each script reads
Several scripts in `regime_synthesis/` and `perturbation_cop/` were written for an **earlier
configuration** of the study:
* GraphCast-Small (1°, 13 levels, 2022-01-01 IC)
* Prithvi WxC runs from a 2022-01-01 IC

The input paths are listed in `docs/original_file_paths.md` and `docs/results_map.md`. The
matched-IC (2023-02-01) GraphCast and Prithvi runs used in the manuscript come from
`GraphCast_OSSE` and `PrithviWxC_OSSE`. Their own analysis scripts live in those repositories.
Check each script's input paths before reusing it with a different set of runs.

## Usage
Python 3.10, `pip install -r environment/requirements-frozen.txt`. Scripts contain the original
absolute cluster paths to model outputs; see `docs/PATHS.md`. `docs/replotting_instructions.md`
explains how to regenerate figures from the cached CSV results.

## License
MIT (`LICENSE`).

## Citation
Diggikar, Y., Halem, M., & Raman, A. (2026). *OSSE_CrossModel_Analysis* (Version 1.0.0) [Software]. Zenodo. https://doi.org/10.5281/zenodo.24092598
(All versions: https://doi.org/10.5281/zenodo.24092597.) See also `CITATION.cff`.

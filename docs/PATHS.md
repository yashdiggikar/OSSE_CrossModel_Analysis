# Hard-coded paths

The scripts in this repository are released **exactly as they were run** on the UMBC HPC
cluster (chip / `pi_halem` allocation), so they contain absolute paths to the original
locations of data, model weights and outputs. To re-run them elsewhere, replace the
prefixes below with your own locations (e.g. with `sed` or by editing the constants at the
top of each script). No other code changes are needed.

Root prefix used everywhere: `/umbc/rs/pi_halem/users/yashd1` (some legacy Aurora drivers use `/xace/d1/yashd1`).

| Original path prefix | Occurrences | Example file |
|---|---|---|
| `/umbc/rs/pi_halem/users/yashd1/common` | 60 | `analysis/regime_synthesis/cloud2/replot_fig_cloud2.py` |
| `/umbc/rs/pi_halem/users/yashd1` | 24 | `analysis/figures/analyze_vertical_propagation.py` |
| `/umbc/rs/pi_halem/users/yashd1/venvs` | 21 | `analysis/figures/run_analyze_vertical_propagation.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/FourCastNet` | 7 | `analysis/regime_synthesis/nrmse_plots/plot_nrmse_improvements.py` |
| `/umbc/rs/pi_halem/users/yashd1/PERTURBATION_EXPERIMENTS` | 6 | `analysis/perturbation_cop/compute_cross_model_comparison.py` |
| `/umbc/rs/pi_halem/users/yashd1/prithviOSSE` | 6 | `analysis/regime_synthesis/tuv_plots/compute_tuv_rmse.py` |
| `/umbc/rs/pi_halem/users/yashd1/logs` | 5 | `analysis/figures/run_perturbed_rmse.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/ALL_ERA5` | 5 | `preprocessing/download_era5_total_cloud_cover.py` |
| `/umbc/rs/pi_halem/users/yashd1/Aurora_Pred_Files` | 4 | `analysis/regime_synthesis/nrmse_plots/plot_nrmse_improvements.py` |
| `/umbc/rs/pi_halem/users/yashd1/gc_pred_files` | 4 | `analysis/regime_synthesis/nrmse_plots/plot_nrmse_improvements.py` |
| `/umbc/rs/pi_halem/users/yashd1/logs_analysis` | 3 | `analysis/figures/run_analyze_vertical_propagation.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/prithvi_new` | 3 | `analysis/regime_synthesis/nrmse_plots/plot_nrmse_improvements.py` |
| `/umbc/rs/pi_halem/users/yashd1/Graphcast-full` | 2 | `analysis/regime_synthesis/tuv_plots/compute_graphcast_tuv_cache.py` |
| `/umbc/rs/pi_halem/users/yashd1/Multimodel-study` | 2 | `analysis/regime_synthesis/tuv_final/plot_q850_tuv_timeseries.py` |
| `/umbc/rs/pi_halem/users/yashd1/random` | 1 | `analysis/figures/run_vertical_propagation.slurm` |
| `/umbc/rs/pi_halem/users/yashd1/fcn_inference` | 1 | `analysis/regime_synthesis/tuv_plots/compute_tuv_rmse.py` |
| `/umbc/rs/pi_halem/users/yashd1/AURORA_DATA` | 1 | `analysis/regime_synthesis/tuv_plots/compute_tuv_rmse.py` |

# TCC & Terrain Stratified RMSE Analysis
## OSSE Insertion Experiments — 4 Models

---

## 1. Overview

These analyses quantify the **cross-variable impact** of two OSSE insertion experiments
(Temperature insertion and UV-wind insertion) on downstream atmospheric variables,
stratified by two environmental partitions:

- **Cloud regime** (ERA5 total cloud cover, TCC) — `common/tcc/` and `common/cloud2/`
- **Terrain class** (MERRA2 surface type + elevation) — `common/terrain/` and `common/terrain2/`

Four models are compared: **Aurora**, **FourCastNet (FCN)**, **Prithvi-WxC**, **GraphCast (GC)**.

---

## 2. Experiments

### 2A. Temperature Insertion → effect on U500, V500, Z500, Q700

| Script | `common/tcc/fig5_75_76_temp_ins_cloud_regime.py` |
|--------|--------------------------------------------------|
| Terrain variant | `common/terrain/fig_terrain_temp_ins.py` |
| Thesis section | 5.12.1 |
| Figures | 5.75 (bar), 5.76 (time series) |

T500 truth was inserted into each model's forecast. We measure how much that injection
propagates to improve U500, V500, Z500, Q700.

### 2B. UV-Wind Insertion → effect on T500, Z500, Q850

| Script | `common/cloud2/fig_cloud2_uv_ins.py` |
|--------|--------------------------------------|
| Terrain variant | `common/terrain2/fig_terrain2_uv_ins.py` |

U500+V500 truth was inserted. We measure cross-variable improvement in T500, Z500, Q850.

---

## 3. Models & Forecast Setup

| Model | IC date | Timestep | Total steps | Insertion step | Notes |
|-------|---------|----------|-------------|----------------|-------|
| Aurora | 2023-02-01 12:00 UTC | 12-hourly | 120 | step 60 (Day 30) | 2023 ERA5 truth |
| FourCastNet | 2023-02-01 00:00 UTC | 12-hourly | 120 | step 60 (Day 30) | 2023 ERA5 truth |
| Prithvi-WxC | 2022-01-01 12:00 UTC | 6-hourly | 240 | step 120 (Day 30) | MERRA2 truth |
| GraphCast | 2022-01-01 06:00 UTC | 6-hourly | 240 | step 120 (Day 30) | ERA5 2022 bilinear regrid to 1° |

### Phase definitions (relative to injection step)

| Phase | Aurora/FCN steps | Prithvi/GC steps | Meaning |
|-------|-----------------|------------------|---------|
| Injection | step 60 / step 120 | step 120 | Baseline RMSE at day of insertion |
| Adjustment | steps 60–80 / 120–160 | steps 120–160 | Signal propagating, Days 30–40 |
| Mature | steps 80–120 / 160–240 | steps 160–240 | Settled response, Days 40–60 |

### Improvement formula

```
Improvement (%) = 100 × (RMSE_ctrl[injection_step] − mean(RMSE_insertion[phase])) / RMSE_ctrl[injection_step]
```

Positive = insertion helped; negative = insertion hurt.

---

## 4. Variables Analysed

### T-insertion cross-variables (Sect 2A)

| Variable | Level | Aurora/Prithvi/GC | FCN |
|----------|-------|-------------------|-----|
| U500 | 500 hPa | specific wind m/s | channel 15 |
| V500 | 500 hPa | specific wind m/s | channel 28 |
| Z500 | 500 hPa | geopotential m²/s² | channel 41 |
| Q700 | 700 hPa | specific humidity kg/kg | RH% channel 69 |

### UV-insertion cross-variables (Sect 2B)

| Variable | Level | Aurora/Prithvi/GC | FCN |
|----------|-------|-------------------|-----|
| T500 | 500 hPa | temperature K | channel 54 |
| Z500 | 500 hPa | geopotential m²/s² | channel 41 |
| Q850 | 850 hPa | specific humidity kg/kg | RH% channel 70 |

### Model level indices

| Level | Aurora (MERRA2 lev, descending) | FCN channel | Prithvi (ascending) | GraphCast (ascending, 13 levels) |
|-------|---------------------------------|-------------|---------------------|----------------------------------|
| 500 hPa | idx 6 | — | idx 7 | idx 7 |
| 700 hPa | idx 5 | — | idx 8 | idx 9 |
| 850 hPa | idx 3 | — | idx 10 | idx 10 |

---

## 5. Cloud Regime Classification (TCC)

ERA5 Total Cloud Cover (fraction 0–1) from `ALL_ERA5/tcc/` GRIB files.

| Regime | Short | TCC range | Color |
|--------|-------|-----------|-------|
| Clear | CLR | TCC < 0.10 | `#4dac26` green |
| Cirrus | CIR | 0.10 ≤ TCC < 0.40 | `#2b83ba` blue |
| Mixed | MIX | 0.40 ≤ TCC < 0.70 | `#fdae61` orange |
| Opaque | OPQ | TCC ≥ 0.70 | `#d7191c` red |

### TCC truth files used

- Aurora/FCN: `ALL_ERA5/tcc/era5_tcc_2023_*.grib` (Feb–Apr 2023)
- Prithvi: `ALL_ERA5/tcc/era5_tcc_2022_*.grib` (Jan–Mar 2022), regridded nearest-neighbour to 360×576
- GraphCast: same 2022 GRIB files, regridded nearest-neighbour to 181×360

---

## 6. Terrain Classification

Source: **MERRA2 surface file** `prithvi/data/merra2/MERRA2_sfc_20220103.nc`
- Dims: `lat: 361` (−90 to 90, 0.5°), `lon: 576` (−180 to 179.4°)
- `PHIS` (no time dim, shape 361×576) → elevation = PHIS/9.80665 m
- `FRLAND` (no time dim) → land fraction
- `FROCEAN` (no time dim) → ocean fraction
- Row 0 (lat = −90) skipped → working array shape (360, 576)

| Class | ID | Condition | Color |
|-------|----|-----------|----- |
| Land Low | 0 | frland>0.5, elev<300 m, \|lat\|≤60° | `#66bd63` |
| Land Mod | 1 | frland>0.5, 300≤elev<1500 m, \|lat\|≤60° | `#1a9850` |
| Land High | 2 | frland>0.5, elev≥1500 m, \|lat\|≤60° | `#006837` |
| Ocean | 3 | frocean>0.5, \|lat\|≤60° | `#4393c3` |
| Polar | 4 | \|lat\|>60° (any surface) | `#9ecae1` |

### Terrain regridding

- Aurora (720×1440): nearest-neighbour from MERRA2 360×576
- FCN (721×1440): nearest-neighbour from MERRA2 360×576
- Prithvi (360×576): MERRA2 used directly
- GraphCast (181×360): nearest-neighbour from MERRA2 360×576 via `regrid_merra_to_gc()`

---

## 7. ERA5 Truth Files

### Aurora & FCN (2023)
```
AURORA_DATA/rmse_data/era5_pressure_2023_{02,03,04}.grib
```
Variables loaded with `cfgrib`, `filter_by_keys={'typeOfLevel':'isobaricInhPa','level':500/700/850}`.
Grid: 721×1440 (0.25°).

### Prithvi (2022, MERRA2)
```
prithvi/data/merra2/MERRA2_pres_202201{03..}.nc4   (pressure-level files)
```
Grid: 361×576 (0.5°). Row 0 skipped → 360×576. Level order descending.

### GraphCast (2022, ERA5 bilinear → 1°)
```
era5_vs_cams/era5_2022/era5_pressure_2022_{01,02,03}.grib
```
ERA5 0.25° (721×1440) bilinearly regridded to GC 1° grid (181×360) using
`scipy.interpolate.RegularGridInterpolator`. ERA5 lats flipped to ascending before interpolation.

---

## 8. Model Prediction Files

### Control (no insertion)

| Model | File |
|-------|------|
| Aurora | `auroraoutput_1year_1stfeb23.nc` |
| FCN | `FourCastNet/forecast_normal_60days_12hr.nc` |
| Prithvi | `prithviOSSE/run2/output/osse_forecast_Jan2022.nc` |
| GraphCast | `gc_pred_files/gc_sm_240steps_STRICT.nc` |

### Temperature insertion

| Model | File |
|-------|------|
| Aurora | `Aurora_Pred_Files/auroraoutput_temp_inserted.nc` |
| FCN | `FourCastNet/Prediction_Files_Fcn/t_ins/t_ins/forecast_truth_injected_60days_12hr_FINAL.nc` |
| Prithvi | `prithviOSSE/temp_ins_run/output/temp_osse_forecast_Jan2022.nc` |
| GraphCast | `gc_pred_files/gc_sm_240steps_temp_inj.nc` |

### UV-wind insertion

| Model | File |
|-------|------|
| Aurora | `Aurora_Pred_Files/auroraoutput_uv_inserted.nc` |
| FCN | `FourCastNet/Prediction_Files_Fcn/wind/wind_ins/forecast_uv_injected_60days_12hr_FINAL.nc` |
| Prithvi | `prithviOSSE/uv_ins_run/output/uv_osse_forecast_Jan2022.nc` |
| GraphCast | `gc_pred_files/gc_sm_240steps_uv_inj.nc` |

### GraphCast file specs
```
dims: time=240, batch=1, lat=181, lon=360, level=13
lat: −90 to 90 (1° ascending)
lon: 0 to 359 (1° ascending)
levels: [50,100,150,200,250,300,400,500,600,700,850,925,1000] hPa
IC: 2022-01-01 06:00 UTC → valid times 2022-01-01 12:00 to 2022-03-02 06:00
NOTE: decode_timedelta=False required (dtype attr conflict in 'time' variable)
```

---

## 9. Output Directory Structure

```
common/
├── tcc/                          ← T-insertion × cloud regime
│   ├── aurora/
│   │   ├── fig5_75_t_ins_rmse_improvement_by_regime.png   (bar chart)
│   │   ├── fig5_76_t_ins_rmse_timeseries_by_regime.png    (time series)
│   │   ├── t_ins_cloud_regime_summary.csv
│   │   └── intermediates/{var}_t_ins_regime_rmse.csv
│   ├── fcn/           (same structure)
│   ├── prithvi/       (same structure)
│   ├── graphcast/     (same structure)
│   └── table_t_ins_cloud_regime_summary.csv   (all models combined)
│
├── terrain/                      ← T-insertion × terrain
│   ├── aurora/
│   │   ├── fig_terrain_t_ins_rmse_improvement_bar.png
│   │   ├── fig_terrain_t_ins_rmse_timeseries.png
│   │   ├── t_ins_terrain_summary.csv
│   │   └── intermediates/{var}_t_ins_terrain_rmse.csv
│   ├── fcn/ prithvi/ graphcast/  (same structure)
│   └── table_t_ins_terrain_summary.csv
│
├── cloud2/                       ← UV-insertion × cloud regime
│   ├── aurora/
│   │   ├── fig_cloud2_uv_ins_rmse_improvement_bar.png
│   │   ├── fig_cloud2_uv_ins_rmse_timeseries.png
│   │   ├── uv_ins_cloud_regime_summary.csv
│   │   └── intermediates/{var}_uv_ins_regime_rmse.csv
│   ├── fcn/ prithvi/ graphcast/  (same structure)
│   └── table_uv_ins_cloud_regime_summary.csv
│
└── terrain2/                     ← UV-insertion × terrain
    ├── aurora/
    │   ├── fig_terrain2_uv_ins_rmse_improvement_bar.png
    │   ├── fig_terrain2_uv_ins_rmse_timeseries.png
    │   ├── uv_ins_terrain_summary.csv
    │   └── intermediates/{var}_uv_ins_terrain_rmse.csv
    ├── fcn/ prithvi/ graphcast/  (same structure)
    └── table_uv_ins_terrain_summary.csv
```

---

## 10. Plot Descriptions

### Bar chart (improvement %)

- Layout: rows = variables (4 for T-ins, 3 for UV-ins), columns = 4 models
- Each subplot: grouped bar per regime/terrain class, 2 bars = Adjustment phase & Mature phase
- Y-axis: RMSE improvement % (positive = better, negative = degradation)
- Bars grouped by regime (4 groups × 2 bars) or terrain (5 groups × 2 bars)
- dpi=300

### Time series

- Layout: rows = variables, columns = 4 models
- Each subplot: RMSE vs forecast lead time (days)
- Lines: control (dashed), insertion (solid), per regime/terrain class (colour-coded)
- Vertical grey dashed line at injection point
- Shaded regions for Adjustment and Mature phases
- dpi=300

---

## 11. SLURM Jobs

| Script | SLURM file | Partition | Memory | Walltime |
|--------|-----------|-----------|--------|---------|
| tcc | `common/tcc/run_fig5_75_76.slurm` | gpu / chip-gpu | 256G | 6h |
| terrain | `common/terrain/run_terrain_fig.slurm` | gpu / chip-gpu | 256G | 6h |
| cloud2 | `common/cloud2/run_cloud2.slurm` | gpu / chip-gpu | 256G | 6h |
| terrain2 | `common/terrain2/run_terrain2.slurm` | gpu / chip-gpu | 256G | 6h |

Python env: `/umbc/rs/pi_halem/users/yashd1/venvs/aurora310/` (Python 3.10)

### Job history (as of 2026-06-13)

| Job ID | Script | Result | Note |
|--------|--------|--------|------|
| 187962/187963 | tcc (3-model) | ✅ Done | Before GraphCast added |
| 187966 | terrain (3-model) | ✅ Done | Before GraphCast added |
| 187969/187975 | cloud2 | ✅ Done | All 4 models complete |
| 187973 | tcc (4-model) | ❌ Failed | Wrong Aurora T-ins path |
| 187974 | terrain (4-model) | ❌ Failed | Wrong MERRA2 SFC path |
| 187976 | terrain2 | ❌ Failed | Wrong MERRA2 SFC path |
| 187977 | tcc (4-model) | ❌ Failed | Wrong FCN T-ins path |
| 187978 | terrain (4-model) | ❌ Failed | PHIS indexing wrong (no time dim) |
| 187979 | terrain2 | ❌ Failed | PHIS indexing wrong (no time dim) |
| 187981 | tcc (4-model) | ❌ Failed | Wrong Prithvi ctrl/ins paths |
| 187982 | terrain (4-model) | ❌ Failed | Wrong Prithvi ctrl/ins paths |
| 187983 | terrain2 | ✅ Done | All 4 models complete |
| 188311 | tcc (4-model) | 🔄 Running | Prithvi paths fixed + 361-lat crop |
| 188312 | terrain (4-model) | 🔄 Running | Prithvi paths fixed + 361-lat crop |

---

## 12. Key Technical Notes

### HDF5 file locking
All scripts set `os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"` and use `engine="h5netcdf"`
for xarray on UMBC cluster (parallel filesystem conflict).

### GraphCast time decode bug
`xr.open_dataset(gc_file, engine="h5netcdf", decode_timedelta=False)` — the GC NetCDF
has a `dtype` attribute conflict in the `time` variable that crashes xarray's default decoder.

### ERA5 RH computation (for FCN truth comparison)
```python
e_s = 6.112 * exp(17.67 * (T−273.15) / (T−29.65))
q_s = 0.622 * e_s / (P_hPa − e_s)
RH  = clip(q / q_s, 0, 1.5) * 100
```

### MERRA2 SFC latitude handling
`PHIS`, `FRLAND`, `FROCEAN` have dims `(lat=361, lon=576)` — no time dimension.
Row 0 (lat=−90°) must be skipped: `values[1:, :]` to get working (360, 576) array.

### Prithvi control vs T-insertion files

| Role | File | Lats |
|------|------|------|
| Control (no injection) | `prithvi/output/forecast_60day_Jan2022.nc` | 361 (−90 to 90, includes south-pole row) |
| T-insertion | `prithviOSSE/run2/output/osse_forecast_Jan2022.nc` | 360 (−89.5 to 90, south-pole row cropped) |

`load_pri()` crops row 0 when `arr.shape[1]==361` to align both files to the 360-lat MERRA2 working grid.

For UV-insertion (cloud2/terrain2), the control role is played by the T-insertion file (the Prithvi UV-only control was never run), so both Prithvi files there are 360-lat OSSE outputs with no crop needed.

### GraphCast ERA5 bilinear regrid
ERA5 0.25° lats are descending (90→−90); must flip to ascending before passing to
`RegularGridInterpolator`. Interpolated to GC 1° grid (181×360, lat −90→90, lon 0→359).

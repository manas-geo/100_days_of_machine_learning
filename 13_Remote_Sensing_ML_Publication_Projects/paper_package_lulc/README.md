# Code and data for: Multi-season Sentinel-1/2 land-cover mapping in the Damodar valley

This repository reproduces all results of the article *<title>* (<journal>, <year>, doi:<...>).

## Quick start
```bash
conda env create -f environment.yml
conda activate lulc-paper
python run_all.py --config config.yaml       # ~1 minute; writes outputs/
python tests/test_core.py                    # or: pytest -q
```

## What `run_all.py` produces
| File | Content | Paper item |
|---|---|---|
| `outputs/lulc_map.tif` | land-cover map, uint8, EPSG:32645, 20 m | Fig. 6 |
| `outputs/table_accuracy_area.csv` | UA, PA with 95% CI, bias-corrected areas | Table 4 |
| `outputs/error_matrix_area_proportions.csv` | error matrix in area proportions | Table 4b |
| `outputs/results.json` | all headline numbers + provenance (versions, input hashes) | text |

Headline results of the archived run: spatial-CV macro F1 = 0.911; map overall accuracy = 0.948 +/- 0.013 (95% CI).

## Data
Input rasters (`data/study_area/*.tif`) and field points (`field_points.gpkg`) are archived at Zenodo (doi:10.5281/zenodo.<record-id>) under CC-BY-4.0.
Sentinel-1/2 data: Copernicus Sentinel data 2024, accessed via Google Earth Engine (collections listed in `docs/data_availability.md`).

## Structure
`config.yaml` (all parameters) - `run_all.py` (entry point) - `src/` (modules) - `tests/` - `docs/` (model card, availability statements)

## Licence
Code: MIT. Data: CC-BY-4.0. Please cite using `CITATION.cff`.

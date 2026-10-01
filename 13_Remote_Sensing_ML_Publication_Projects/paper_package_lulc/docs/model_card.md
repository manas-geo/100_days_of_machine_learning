# Model card: LULC random forest, Damodar valley 2024

**Model**: random forest (300 trees, max_features=sqrt), scikit-learn 1.9.1.
**Inputs**: 81 features from seasonal Sentinel-2 composites, Sentinel-1 VV/VH, texture, terrain (20 m, EPSG:32645).
**Classes**: water, forest, cropland, built_up, mining_bare, shrub_grass.
**Intended use**: land-cover mapping and area estimation in the study area for 2024. **Not intended** for other regions or years without
re-validation, nor for parcel-level decisions (pixel accuracy varies, see the confidence layer).

## Training data
598 field points, stratified random sampling on a prior map, GPS (~8 m), ~3-7% label noise (estimated in QC).

## Evaluation
- Spatial 5-fold block CV (2000 m blocks): OA 0.910, macro F1 0.911.
- Independent stratified random validation sample (Olofsson et al. 2014): **OA 0.948 +/- 0.013** (95% CI).

| class | adjusted area (ha) | 95% CI |
|---|---|---|
| water | 94.4 | +/- 5.1 |
| forest | 3,434.5 | +/- 111.3 |
| cropland | 1,548.8 | +/- 61.9 |
| built_up | 273.7 | +/- 16.6 |
| mining_bare | 285.5 | +/- 15.4 |
| shrub_grass | 4,363.1 | +/- 126.2 |

## Limitations
Spectrally similar classes (forest / shrub, cropland / shrub) are confused at boundaries; thin linear features are under-mapped at 20 m.
Predictions outside the area of applicability (computed as in Day 91) are extrapolations; check the AOA before applying the model elsewhere.

## Provenance
Inputs SHA-256: feature_stack_2024.tif: 594b289596fa4b3d, field_points.gpkg: 24242af04b1ea449, s1_2024_vv_vh_db.tif: ff22005605b7967a, s2_2024_cloudmask.tif: e60a1acddb9e9849, s2_2024_stack.tif: 28c3103a2af9de34, terrain.tif: 98cba3badd80e6a6; config hash 1466521d2bc8bb38.

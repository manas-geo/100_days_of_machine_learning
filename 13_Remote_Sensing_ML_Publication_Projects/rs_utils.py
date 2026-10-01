"""
rs_utils.py - helper functions shared by the remote-sensing notebooks of Module 13 (Days 86-100).

Contents
--------
1. Synthetic but realistic study areas (terrain, land cover, Sentinel-2 / Sentinel-1 time series)
   so that every notebook runs offline. Replace them with real data (Day 87) for your own study.
2. Raster / vector I/O helpers (rasterio, geopandas).
3. Spectral indices.
4. Sampling designs (stratified random, systematic) and pixel extraction at points.
5. Spatial cross-validation helpers (blocks, buffered leave-one-out).
6. Accuracy assessment and area estimation following Olofsson et al. (2014).
7. Area of applicability (dissimilarity index, Meyer & Pebesma 2021).
8. Publication-quality map helpers (categorical colours, scale bar, north arrow).

All coordinates use UTM zone 45N (EPSG:32645), the zone of the Damodar valley / Dhanbad region (Jharkhand, India),
which the synthetic landscape loosely imitates. Nothing here is real data.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter, distance_transform_edt, sobel, uniform_filter

CRS = "EPSG:32645"
ORIGIN = (430000.0, 2640000.0)                      # upper-left corner (easting, northing) in metres

# ---------------------------------------------------------------------------------------------
# 1. Synthetic study areas
# ---------------------------------------------------------------------------------------------
LC_CLASSES = ["water", "forest", "cropland", "built_up", "mining_bare", "shrub_grass"]
LC_COLORS = ["#2b83ba", "#1a9641", "#f4d03f", "#d7191c", "#5d4037", "#a6d96a"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

# Mean surface reflectance (B2..B12) at the peak of each class's season.
_PEAK = {"water": [0.07, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.02, 0.01, 0.01],
         "forest": [0.03, 0.05, 0.03, 0.08, 0.22, 0.28, 0.32, 0.33, 0.16, 0.08],
         "cropland": [0.04, 0.08, 0.05, 0.12, 0.27, 0.33, 0.37, 0.38, 0.21, 0.11],
         "built_up": [0.11, 0.12, 0.13, 0.15, 0.17, 0.18, 0.19, 0.20, 0.23, 0.21],
         "mining_bare": [0.08, 0.09, 0.10, 0.11, 0.12, 0.12, 0.13, 0.13, 0.17, 0.16],
         "shrub_grass": [0.05, 0.08, 0.07, 0.12, 0.20, 0.23, 0.26, 0.27, 0.24, 0.15]}
_BARE_SOIL = np.array([0.08, 0.11, 0.14, 0.16, 0.19, 0.20, 0.21, 0.22, 0.29, 0.25])
_DATES = pd.to_datetime(["2024-01-15", "2024-03-15", "2024-05-15", "2024-07-15", "2024-09-15", "2024-11-15"])
# greenness (0 = bare soil, 1 = peak) of each class at each date: kharif rice peaks in September,
# a rabi crop (wheat / mustard) in January-February on irrigated fields; forest is deciduous (dry-season dip).
_GREEN = {"water": [0, 0, 0, 0, 0, 0], "forest": [0.85, 0.7, 0.5, 0.95, 1.0, 0.95], "cropland": [0.7, 0.35, 0.05, 0.45, 1.0, 0.3],
          "built_up": [0, 0, 0, 0, 0, 0], "mining_bare": [0, 0, 0, 0, 0, 0], "shrub_grass": [0.5, 0.3, 0.15, 0.8, 0.9, 0.6]}


def _field(shape, sigma, rng):
    a = gaussian_filter(rng.normal(size=shape), sigma)
    return (a - a.mean()) / a.std()


def make_terrain(n=500, pixel=20.0, seed=0):
    """DEM (m), slope (deg), aspect (deg), a river-valley mask and distance rasters."""
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:n, 0:n]
    valley_x = n * 0.45 + 0.12 * n * np.sin(yy / (0.18 * n)) + 0.04 * n * _field((n, n), 30, rng)
    dist_valley = np.abs(xx - valley_x) * pixel
    hills = 120 * np.clip(_field((n, n), n / 12, rng) + 0.4, 0, None) ** 1.6
    dem = 180 + 0.035 * dist_valley + hills + 6 * _field((n, n), 3, rng)
    dzdy, dzdx = sobel(dem, axis=0) / (8 * pixel), sobel(dem, axis=1) / (8 * pixel)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360
    river = dist_valley < 2.5 * pixel
    return {"dem": dem.astype(np.float32), "slope": slope.astype(np.float32), "aspect": aspect.astype(np.float32),
            "river": river, "dist_river_m": (distance_transform_edt(~river) * pixel).astype(np.float32)}


def make_landcover(terrain, pixel=20.0, seed=0, year_offset=0):
    """Land-cover map driven by terrain, a town, roads and mining. year_offset (years) grows the town and mines."""
    rng = np.random.default_rng(seed)
    n = terrain["dem"].shape[0]; yy, xx = np.mgrid[0:n, 0:n]
    # roads: two winding roads through the town (one roughly N-S, one towards the south-west)
    town = (int(n * 0.35), int(n * 0.62))
    t = np.linspace(-1, 1, 6 * n)
    curves = [(town[0] + t * n, town[1] + 0.06 * n * np.sin(2.5 * t + 0.6) - 0.08 * n * t),
              (town[0] + 0.6 * n * (t + 1) / 2, town[1] - 0.55 * n * (t + 1) / 2 + 0.04 * n * np.sin(6 * t))]
    road = np.zeros((n, n), bool)
    for rr, cc in curves:
        ok = (rr >= 0) & (rr < n) & (cc >= 0) & (cc < n)
        road[rr[ok].astype(int), cc[ok].astype(int)] = True
    dist_road = distance_transform_edt(~road) * pixel
    dist_town = np.hypot(yy - town[0], xx - town[1]) * pixel
    lc = np.full((n, n), 5, np.int8)                                                     # shrub/grass default
    flat = terrain["slope"] < 4
    crop_score = (flat * 1.0) + 0.8 * np.exp(-terrain["dist_river_m"] / 1500) + 0.3 * _field((n, n), 8, rng)
    lc[crop_score > 1.05] = 2
    forest_score = terrain["slope"] / 10 + (terrain["dem"] - np.percentile(terrain["dem"], 60)) / 80 + 0.4 * _field((n, n), 10, rng)
    lc[forest_score > 0.9] = 1
    urban_radius = 1100 + 120 * year_offset
    urban = dist_town + 900 * _field((n, n), 12, rng).clip(-1, 1) < urban_radius
    # villages: compact clusters along the roads that grow with time (not continuous ribbons)
    road_px = np.argwhere(road & (dist_town > 1500))
    villages = road_px[rng.choice(len(road_px), 9, replace=False)]
    vshape = _field((n, n), 3, rng)
    for (vr, vc), r0 in zip(villages, rng.uniform(3, 6, 9)):
        rad = max(r0 + 0.5 * year_offset, 1.0)
        urban |= np.hypot(yy - vr, xx - vc) + 1.5 * vshape < rad
    lc[urban] = 3
    mines = np.zeros((n, n), bool)
    for cy, cx, r0 in [(0.75, 0.25, 0.07), (0.8, 0.7, 0.05), (0.2, 0.2, 0.04)]:
        rr = (r0 * n + 0.012 * n * year_offset) * (1 + 0.3 * _field((n, n), 6, rng))
        mines |= np.hypot(yy - cy * n, xx - cx * n) < rr
    lc[mines] = 4
    ponds = _field((n, n), 3, rng) > 2.6
    lc[terrain["river"] | (ponds & flat)] = 0
    return lc, dist_road.astype(np.float32), dist_town.astype(np.float32)


def simulate_s2(lc, terrain, seed=0, dates=_DATES, cloud_dates=(3,), noise=0.012, regional=0.25, haze=0.02, mixing=1.0):
    """
    Sentinel-2-like surface reflectance time series: array (T, 10 bands, n, n) + cloud masks (T, n, n).
    Realism knobs (set to 0 for an "easy" textbook scene):
      regional : amplitude of a smooth (~2 km) greenness / soil-brightness anomaly (rainfall, soils)  -> non-stationarity
      haze     : mean amplitude of a smooth, date-specific additive haze (strongest in blue)          -> atmospheric residuals
      mixing   : sub-pixel forest/shrub canopy mixing and field-level sowing offsets                  -> class overlap
    """
    rng = np.random.default_rng(seed); n = lc.shape[0]
    hetero = 1 + 0.08 * _field((n, n), 4, rng)                                             # within-class variability
    shade = np.clip(np.cos(np.radians(terrain["slope"])) * (1 + 0.25 * np.cos(np.radians(terrain["aspect"] - 135)) * np.sin(np.radians(terrain["slope"]))), 0.6, 1.2)
    green_anom = regional * _field((n, n), n / 10, rng)                                   # regional phenology anomaly
    soil = 1 + 0.6 * regional * _field((n, n), n / 10, rng)                                # regional soil brightness
    fz = _field((n, n), 3, rng)                                                            # canopy-cover field
    ffrac = (np.where(lc == 1, np.clip(0.72 + 0.22 * fz, 0.25, 1), np.clip(0.18 + 0.2 * fz, 0, 0.65)) if mixing
             else (lc == 1).astype(float))                                                 # forest fraction in forest / shrub pixels
    sow = mixing * 0.35 * _field((n, n), 2.5, rng)                                         # field-level sowing offset
    hz_w = np.array([1, .85, .7, .6, .5, .45, .4, .4, .15, .1])                            # haze decreases with wavelength
    stack = np.zeros((len(dates), len(S2_BANDS), n, n), np.float32); clouds = np.zeros((len(dates), n, n), bool)
    for t in range(len(dates)):
        img = np.zeros((len(S2_BANDS), n, n), np.float32)
        for k, c in enumerate(LC_CLASSES):
            m = lc == k
            if not m.any(): continue
            soil_m = _BARE_SOIL[:, None] * soil[m][None]
            def veg(cls, extra=0.0):
                g = np.clip(_GREEN[cls][t] + green_anom[m] + extra, 0, 1)
                return g[None] * np.array(_PEAK[cls])[:, None] + (1 - g[None]) * soil_m
            if c in ("water", "built_up"):
                spec = np.array(_PEAK[c])[:, None] * np.ones(m.sum())[None]
            elif c == "mining_bare":
                spec = np.array(_PEAK[c])[:, None] * soil[m][None]
            elif c in ("forest", "shrub_grass"):
                f = ffrac[m][None]
                spec = f * veg("forest") + (1 - f) * veg("shrub_grass")
            else:                                                                          # cropland: early / late sown fields
                spec = veg("cropland", sow[m] * (1 if t in (0, 4) else -0.5))
            img[:, m] = spec * hetero[m][None]
        hz = haze * (1 + _field((n, n), n / 6, rng))
        img = (gaussian_filter(img * shade[None], sigma=(0, 0.7, 0.7))                     # sensor PSF -> mixed edge pixels
               + hz[None] * hz_w[:, None, None] + rng.normal(0, noise, img.shape))
        if t in cloud_dates or rng.random() < 0.15:
            cl = _field((n, n), 15, rng) > (0.3 if t in cloud_dates else 1.2)
            img[:, cl] = 0.35 + 0.05 * rng.random((len(S2_BANDS), cl.sum())); clouds[t] = cl
        stack[t] = np.clip(img, 0.0005, 1)
    return stack, clouds


def simulate_s1(lc, terrain, seed=0, n_dates=6, looks=4):
    """Sentinel-1-like VV/VH backscatter (dB) with multiplicative gamma speckle: (T, 2, n, n)."""
    rng = np.random.default_rng(seed); n = lc.shape[0]
    base_vv = np.array([-21, -8, -11, -4, -12, -10.0]); base_vh = np.array([-28, -14, -17, -11, -19, -16.0])
    out = np.zeros((n_dates, 2, n, n), np.float32)
    for t in range(n_dates):
        vv = base_vv[lc] + 1.5 * _field((n, n), 3, rng) + 0.04 * (terrain["slope"] - 5)
        vh = base_vh[lc] + 1.5 * _field((n, n), 3, rng)
        if t in (3, 4): vv = np.where(lc == 2, vv - 4, vv); vh = np.where(lc == 2, vh - 3, vh)   # flooded paddy in the monsoon
        for b, v in enumerate([vv, vh]):
            lin = 10 ** (v / 10) * rng.gamma(looks, 1 / looks, (n, n))                      # speckle
            out[t, b] = 10 * np.log10(lin)
    return out


def make_study_area(n=500, pixel=20.0, seed=0):
    """Everything needed for the land-cover project in one dict."""
    terrain = make_terrain(n, pixel, seed)
    lc, dist_road, dist_town = make_landcover(terrain, pixel, seed)
    s2, clouds = simulate_s2(lc, terrain, seed)
    s1 = simulate_s1(lc, terrain, seed)
    return {"lc": lc, "s2": s2, "clouds": clouds, "s1": s1, "dist_road_m": dist_road, "dist_town_m": dist_town,
            "transform": transform_from_origin(ORIGIN, pixel), "crs": CRS, "dates": _DATES, "pixel": pixel, **terrain}


# ---------------------------------------------------------------------------------------------
# 2. Raster / vector I/O
# ---------------------------------------------------------------------------------------------
def transform_from_origin(origin=ORIGIN, pixel=20.0):
    from rasterio.transform import from_origin
    return from_origin(origin[0], origin[1], pixel, pixel)


def write_geotiff(path, array, transform, crs=CRS, nodata=None, band_names=None, dtype=None, compress="deflate"):
    """Write a 2-D (rows, cols) or 3-D (bands, rows, cols) array as a tiled, compressed GeoTIFF."""
    import rasterio
    arr = array[None] if array.ndim == 2 else array
    dtype = dtype or arr.dtype
    profile = dict(driver="GTiff", height=arr.shape[1], width=arr.shape[2], count=arr.shape[0], dtype=dtype, crs=crs,
                   transform=transform, nodata=nodata, compress=compress, tiled=True, blockxsize=256, blockysize=256)
    with rasterio.open(path, "w", **profile) as dst:
        dst.write(arr.astype(dtype))
        if band_names:
            for i, name in enumerate(band_names, 1):
                dst.set_band_description(i, name)
    return path


def read_geotiff(path):
    import rasterio
    with rasterio.open(path) as src:
        return src.read(), src.profile, list(src.descriptions)


def pixel_centres(transform, rows, cols):
    """Map coordinates (x, y) of pixel centres."""
    x = transform.c + (np.asarray(cols) + 0.5) * transform.a
    y = transform.f + (np.asarray(rows) + 0.5) * transform.e
    return x, y


def points_to_rowcol(transform, x, y):
    cols = ((np.asarray(x) - transform.c) / transform.a).astype(int)
    rows = ((np.asarray(y) - transform.f) / transform.e).astype(int)
    return rows, cols


# ---------------------------------------------------------------------------------------------
# 3. Spectral indices (Sentinel-2 band names)
# ---------------------------------------------------------------------------------------------
def indices(b):
    """b: dict-like of band arrays (B2..B12). Returns a dict of common indices."""
    eps = 1e-6
    return {"NDVI": (b["B8"] - b["B4"]) / (b["B8"] + b["B4"] + eps),
            "NDWI": (b["B3"] - b["B8"]) / (b["B3"] + b["B8"] + eps),                   # McFeeters
            "MNDWI": (b["B3"] - b["B11"]) / (b["B3"] + b["B11"] + eps),
            "NDBI": (b["B11"] - b["B8"]) / (b["B11"] + b["B8"] + eps),
            "NDRE": (b["B8A"] - b["B5"]) / (b["B8A"] + b["B5"] + eps),
            "BSI": ((b["B11"] + b["B4"]) - (b["B8"] + b["B2"])) / ((b["B11"] + b["B4"]) + (b["B8"] + b["B2"]) + eps),
            "EVI": 2.5 * (b["B8"] - b["B4"]) / (b["B8"] + 6 * b["B4"] - 7.5 * b["B2"] + 1)}


# ---------------------------------------------------------------------------------------------
# 4. Sampling designs
# ---------------------------------------------------------------------------------------------
def stratified_random_sample(strata, n_per_stratum, seed=0, exclude=None):
    """Return (rows, cols, stratum) with a fixed number of random pixels per stratum (dict or int)."""
    rng = np.random.default_rng(seed); rows, cols, lab = [], [], []
    valid = np.ones(strata.shape, bool) if exclude is None else ~exclude
    for k in np.unique(strata[valid]):
        nk = n_per_stratum[k] if isinstance(n_per_stratum, dict) else n_per_stratum
        idx = np.flatnonzero((strata == k) & valid)
        pick = rng.choice(idx, min(nk, len(idx)), replace=False)
        r, c = np.unravel_index(pick, strata.shape); rows += list(r); cols += list(c); lab += [k] * len(pick)
    return np.array(rows), np.array(cols), np.array(lab)


def systematic_sample(shape, spacing, offset=None, seed=0):
    rng = np.random.default_rng(seed); off = offset if offset is not None else rng.integers(0, spacing, 2)
    rr, cc = np.mgrid[off[0]:shape[0]:spacing, off[1]:shape[1]:spacing]
    return rr.ravel(), cc.ravel()


# ---------------------------------------------------------------------------------------------
# 5. Spatial cross-validation helpers
# ---------------------------------------------------------------------------------------------
def spatial_blocks(x, y, block_size):
    """Integer block id for each point on a square grid of block_size (map units)."""
    bx = np.floor((np.asarray(x) - np.min(x)) / block_size).astype(int)
    by = np.floor((np.asarray(y) - np.min(y)) / block_size).astype(int)
    return bx * 10000 + by


def buffered_loo_splits(x, y, buffer, max_tests=None, seed=0):
    """Leave-one-out with a buffer: training excludes all points within `buffer` of the test point."""
    xy = np.c_[x, y]; n = len(xy); idx = np.arange(n)
    if max_tests is not None:
        idx = np.random.default_rng(seed).choice(n, min(max_tests, n), replace=False)
    for i in idx:
        d = np.hypot(xy[:, 0] - xy[i, 0], xy[:, 1] - xy[i, 1])
        yield np.flatnonzero(d > buffer), np.array([i])


def morans_i(values, x, y, k=8):
    """Global Moran's I with a k-nearest-neighbour row-standardised weight matrix."""
    from sklearn.neighbors import NearestNeighbors
    xy = np.c_[x, y]; z = np.asarray(values, float) - np.mean(values)
    nbrs = NearestNeighbors(n_neighbors=k + 1).fit(xy).kneighbors(xy, return_distance=False)[:, 1:]
    num = np.sum(z[:, None] * z[nbrs]) / k                                     # sum_ij w_ij z_i z_j with w_ij = 1/k
    return num / np.sum(z ** 2)                                                 # row-standardised weights: S0 = n cancels


# ---------------------------------------------------------------------------------------------
# 6. Accuracy assessment and area estimation (Olofsson et al., 2014)
# ---------------------------------------------------------------------------------------------
def olofsson_accuracy(map_class, ref_class, map_pixel_counts, pixel_area_ha, class_names=None):
    """
    Stratified estimators of overall / user's / producer's accuracy and of class areas with 95% CIs.
    map_class, ref_class : 1-D arrays for the validation sample (strata = map classes)
    map_pixel_counts     : dict {class: number of pixels mapped as that class}
    Returns (summary DataFrame, error matrix of estimated area proportions).
    """
    with np.errstate(invalid="ignore", divide="ignore"):                               # empty strata / classes give NaN, not warnings
        return _olofsson(map_class, ref_class, map_pixel_counts, pixel_area_ha, class_names)


def _olofsson(map_class, ref_class, map_pixel_counts, pixel_area_ha, class_names=None):
    classes = sorted(map_pixel_counts)
    names = class_names or [str(c) for c in classes]
    A_tot = sum(map_pixel_counts.values()); W = np.array([map_pixel_counts[c] / A_tot for c in classes])
    n = pd.crosstab(pd.Categorical(map_class, classes), pd.Categorical(ref_class, classes), dropna=False).values.astype(float)
    n_i = n.sum(1)
    p = W[:, None] * n / np.maximum(n_i[:, None], 1)                                   # estimated area proportions p_ij
    OA = np.trace(p); UA = np.diag(p) / p.sum(1); PA = np.diag(p) / p.sum(0); area_prop = p.sum(0)
    # standard errors (Olofsson et al. 2014, eqs 5, 6, 7, 10)
    se_OA = np.sqrt(np.sum(W ** 2 * UA * (1 - UA) / np.maximum(n_i - 1, 1)))
    se_UA = np.sqrt(UA * (1 - UA) / np.maximum(n_i - 1, 1))
    se_area = np.sqrt([np.sum((W ** 2 * (n[:, j] / np.maximum(n_i, 1)) * (1 - n[:, j] / np.maximum(n_i, 1))) / np.maximum(n_i - 1, 1)) for j in range(len(classes))])
    N_j = A_tot * area_prop
    se_PA = []
    for j in range(len(classes)):
        term1 = (A_tot * W[j]) ** 2 * (1 - PA[j]) ** 2 * UA[j] * (1 - UA[j]) / max(n_i[j] - 1, 1)
        term2 = PA[j] ** 2 * sum((A_tot * W[i]) ** 2 * n[i, j] / max(n_i[i], 1) * (1 - n[i, j] / max(n_i[i], 1)) / max(n_i[i] - 1, 1) for i in range(len(classes)) if i != j)
        se_PA.append(np.sqrt((term1 + term2) / max(N_j[j], 1e-9) ** 2))
    se_PA = np.array(se_PA)
    mapped_ha = np.array([map_pixel_counts[c] for c in classes]) * pixel_area_ha
    summary = pd.DataFrame({"class": names, "mapped_area_ha": mapped_ha, "sample_n": n_i.astype(int),
                            "UA": UA, "UA_95CI": 1.96 * se_UA, "PA": PA, "PA_95CI": 1.96 * se_PA,
                            "adjusted_area_ha": area_prop * A_tot * pixel_area_ha, "area_95CI_ha": 1.96 * se_area * A_tot * pixel_area_ha})
    summary.attrs["OA"] = OA; summary.attrs["OA_95CI"] = 1.96 * se_OA
    return summary, pd.DataFrame(p, index=[f"map {c}" for c in names], columns=[f"ref {c}" for c in names])


# ---------------------------------------------------------------------------------------------
# 7. Area of applicability (Meyer & Pebesma, 2021)
# ---------------------------------------------------------------------------------------------
def dissimilarity_index(X_train, X_new, weights=None, cv_folds=None):
    """
    DI = distance to the nearest training sample in the (importance-weighted, standardised) predictor space,
    divided by the mean pairwise distance between training samples. The AOA threshold is the upper whisker
    (75th percentile + 1.5 IQR) of the training DI computed in cross-validation (nearest sample in OTHER folds).
    Returns (DI_new, threshold).
    """
    from sklearn.neighbors import NearestNeighbors
    mu, sd = X_train.mean(0), X_train.std(0) + 1e-12
    w = np.ones(X_train.shape[1]) if weights is None else np.asarray(weights) / np.max(weights)
    T = (X_train - mu) / sd * w; Nw = (X_new - mu) / sd * w
    rng = np.random.default_rng(0); sub = T[rng.choice(len(T), min(2000, len(T)), replace=False)]
    dbar = np.mean(np.sqrt(((sub[:, None] - sub[None]) ** 2).sum(-1))[np.triu_indices(len(sub), 1)])
    di_new = NearestNeighbors(n_neighbors=1).fit(T).kneighbors(Nw)[0][:, 0] / dbar
    if cv_folds is None:
        di_tr = NearestNeighbors(n_neighbors=2).fit(T).kneighbors(T)[0][:, 1] / dbar
    else:
        di_tr = np.zeros(len(T))
        for f in np.unique(cv_folds):
            m = cv_folds == f
            di_tr[m] = NearestNeighbors(n_neighbors=1).fit(T[~m]).kneighbors(T[m])[0][:, 0] / dbar
    q1, q3 = np.percentile(di_tr, [25, 75])
    return di_new, q3 + 1.5 * (q3 - q1)


# ---------------------------------------------------------------------------------------------
# 8. Publication-quality map helpers
# ---------------------------------------------------------------------------------------------
def pub_style():
    import matplotlib.pyplot as plt
    plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 300, "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                         "legend.fontsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8, "font.family": "DejaVu Sans"})


def categorical_cmap(colors=LC_COLORS):
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(colors); norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5), len(colors))
    return cmap, norm


def add_scalebar(ax, length_m, extent, label=None, loc=(0.05, 0.05), color="k"):
    """Draw a simple scale bar on an axis whose data coordinates are metres (use imshow(..., extent=extent))."""
    x0 = extent[0] + loc[0] * (extent[1] - extent[0]); y0 = extent[2] + loc[1] * (extent[3] - extent[2])
    ax.plot([x0, x0 + length_m], [y0, y0], color=color, lw=3, solid_capstyle="butt")
    ax.text(x0 + length_m / 2, y0 + 0.015 * (extent[3] - extent[2]), label or f"{length_m / 1000:g} km", ha="center", va="bottom", color=color, fontsize=8)


def add_north_arrow(ax, loc=(0.93, 0.88), size=0.07, color="k"):
    ax.annotate("N", xy=(loc[0], loc[1]), xytext=(loc[0], loc[1] - size), xycoords="axes fraction", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color, arrowprops=dict(facecolor=color, width=4, headwidth=10))


def map_extent(transform, shape):
    """matplotlib extent (left, right, bottom, top) in map coordinates."""
    left, top = transform.c, transform.f
    return [left, left + shape[1] * transform.a, top + shape[0] * transform.e, top]


# ---------------------------------------------------------------------------------------------
# 9. Crop-type time series (parcel level) for Project 2 (Days 92-93)
# ---------------------------------------------------------------------------------------------
CROP_CLASSES = ["rice", "maize", "pigeon_pea", "wheat", "mustard"]
CROP_BANDS = ["B2", "B3", "B4", "B5", "B8", "B11"]
# day 0 = 1 June. sos / eos = inflection points of a double-logistic greenness curve (days), r = steepness (days),
# amp = peak fractional cover, kharif = sowing depends on monsoon onset.
_CROP_PHENO = {"rice": dict(sos=55, eos=150, r1=8, r2=9, amp=0.85, kharif=True, flood=(25, 65)),
               "maize": dict(sos=30, eos=112, r1=7, r2=7, amp=0.85, kharif=True),
               "pigeon_pea": dict(sos=50, eos=245, r1=18, r2=10, amp=0.65, kharif=True),
               "wheat": dict(sos=180, eos=290, r1=9, r2=8, amp=0.85, kharif=False),
               "mustard": dict(sos=150, eos=245, r1=8, r2=9, amp=0.7, kharif=False, flower=(195, 225))}
_CANOPY = {"rice": [0.03, 0.06, 0.03, 0.10, 0.40, 0.18], "maize": [0.03, 0.07, 0.04, 0.12, 0.43, 0.21],
           "pigeon_pea": [0.04, 0.07, 0.05, 0.12, 0.36, 0.22], "wheat": [0.03, 0.07, 0.04, 0.11, 0.45, 0.20],
           "mustard": [0.04, 0.08, 0.05, 0.13, 0.42, 0.22]}
_SOIL6 = np.array([0.09, 0.12, 0.15, 0.17, 0.22, 0.30]); _WATER6 = np.array([0.06, 0.07, 0.05, 0.04, 0.03, 0.01])
_YELLOW = np.array([0.0, 0.08, 0.06, 0.04, 0.0, 0.0])                     # mustard flowers raise green and red


def make_crop_dataset(n_per_region=900, regions=("A", "B", "C"), years=(2023, 2024), s2_step=5, s1_step=12, seed=0):
    """
    Parcel-level (field-mean) Sentinel-2 and Sentinel-1 time series from 1 June to 31 May for several regions and years.
    Region B sows kharif crops ~14 days later than A, region C ~6 days earlier and with more pigeon pea; 2023 had a late
    monsoon onset (+18 days for kharif crops). Monsoon-season optical data are mostly cloudy (NaN); a few undetected
    thin clouds remain. Returns (parcels DataFrame, s2 (N, T2, 6) with NaN, s2_days, s1 (N, T1, 2) dB, s1_days).
    """
    rng = np.random.default_rng(seed)
    s2_days = np.arange(0, 365, s2_step); s1_days = np.arange(3, 365, s1_step)
    region_shift = {"A": 0, "B": 14, "C": -6}; year_shift = {2023: 18, 2024: 0}
    mix = {"A": [0.42, 0.16, 0.08, 0.2, 0.14], "B": [0.45, 0.12, 0.08, 0.2, 0.15], "C": [0.35, 0.15, 0.18, 0.17, 0.15]}
    cloud_p = np.select([s2_days < 30, s2_days < 120, s2_days < 150], [0.55, 0.8, 0.35], 0.1)   # Jun / Jul-Sep / Oct / dry
    rows, S2, S1 = [], [], []
    for yi, year in enumerate(years):
        for ri, reg in enumerate(regions):
            crops = rng.choice(len(CROP_CLASSES), n_per_region, p=mix.get(reg, mix["A"]))
            px = 430000 + 12000 * ri + rng.uniform(0, 10000, n_per_region); py = 2630000 + rng.uniform(0, 10000, n_per_region)
            local = 4 * np.sin(px / 2500) + 4 * np.cos(py / 3000)                        # smooth local sowing pattern (days)
            day_clouds = rng.random((len(s2_days), n_per_region)) < cloud_p[:, None]
            rain = np.zeros(len(s1_days)); rain[rng.choice(len(s1_days), 8, replace=False)] = rng.uniform(1.5, 4, 8)   # wet-soil dates
            for i in range(n_per_region):
                c = CROP_CLASSES[crops[i]]; p = dict(_CROP_PHENO[c])
                variant = "standard"
                if c == "rice" and rng.random() < 0.25:                                     # direct-seeded rice: no flooding, earlier
                    p.pop("flood"); p["sos"] -= 15; variant = "direct_seeded"
                if c == "pigeon_pea" and rng.random() < 0.35: variant = "intercropped_maize"
                if c == "wheat" and rng.random() < 0.2: p["sos"] += 20; p["eos"] += 12; variant = "late_sown"
                shift = rng.normal(0, 11) + local[i] + ((region_shift[reg] if reg in region_shift else 0) + year_shift.get(year, 0)) * p["kharif"]
                if not p["kharif"]: shift += rng.normal(0, 4) + (4 if reg == "B" else 0)
                amp = np.clip(p["amp"] * rng.normal(1, 0.14), 0.3, 0.98)
                def fc(t, p=p, amp=amp, shift=shift):
                    out = amp * (1 / (1 + np.exp(-(t - p["sos"] - shift) / p["r1"])) - 1 / (1 + np.exp(-(t - p["eos"] - shift) / p["r2"])))
                    if variant == "intercropped_maize":
                        m = _CROP_PHENO["maize"]
                        out = 0.55 * out + 0.45 * amp * (1 / (1 + np.exp(-(t - m["sos"] - shift) / m["r1"])) - 1 / (1 + np.exp(-(t - m["eos"] - shift) / m["r2"])))
                    return out
                f2 = np.clip(fc(s2_days), 0, 1)[:, None]
                bg = np.tile(_SOIL6 * rng.normal(1, 0.08), (len(s2_days), 1))
                if "flood" in p:
                    wet = (s2_days > p["flood"][0] + shift) & (s2_days < p["flood"][1] + shift)
                    bg[wet] = _WATER6 * 1.3 + 0.35 * _SOIL6
                spec = f2 * np.array(_CANOPY[c]) * rng.normal(1, 0.05) + (1 - f2) * bg
                if "flower" in p:
                    fl = np.exp(-0.5 * ((s2_days - np.mean(p["flower"]) - shift) / 8) ** 2)[:, None]
                    spec = spec + fl * _YELLOW * rng.uniform(0.6, 1.2)
                spec = spec + rng.normal(0, 0.012, spec.shape)
                cl = day_clouds[:, i].copy()
                thin = (~cl) & (rng.random(len(s2_days)) < 0.06)                              # undetected thin cloud
                spec[thin] += 0.08 * np.array([1, 0.9, 0.8, 0.7, 0.6, 0.3])
                spec[cl] = np.nan
                S2.append(np.clip(spec, 0.001, 1).astype(np.float32))
                f1 = np.clip(fc(s1_days), 0, 1)
                vh = -24 + 11 * f1 ** 0.8; vv = -13 + 4 * f1 - (3 * f1 if c == "wheat" else 0)
                if "flood" in p:
                    wet1 = (s1_days > p["flood"][0] + shift) & (s1_days < p["flood"][1] + shift)
                    vh = np.where(wet1, -27 + 4 * f1, vh); vv = np.where(wet1, -19 + 6 * f1, vv)
                rough = rng.normal(0, [1.5, 1.2])                                                # row direction / roughness offset
                wet_soil = rain * (1 - f1)                                                       # rain raises backscatter of bare soil
                vv = vv + rough[0] + wet_soil; vh = vh + rough[1] + 0.6 * wet_soil
                n_pix = 50
                noise = lambda: 10 * np.log10(rng.gamma(4 * n_pix, 1 / (4 * n_pix), len(s1_days)))      # speckle after field averaging
                S1.append(np.stack([vv + noise() + rng.normal(0, 1.0, len(s1_days)), vh + noise() + rng.normal(0, 1.0, len(s1_days))], 1).astype(np.float32))
                rows.append({"parcel_id": len(rows), "year": year, "region": reg, "x": px[i], "y": py[i],
                             "area_ha": float(np.round(rng.lognormal(-0.6, 0.6), 3)), "crop_id": int(crops[i]), "crop": c, "variant": variant})
    parcels = pd.DataFrame(rows)
    noisy = rng.random(len(parcels)) < 0.03                                                     # 3% label errors in the survey
    parcels["crop_id_true"] = parcels["crop_id"]
    parcels.loc[noisy, "crop_id"] = rng.integers(0, len(CROP_CLASSES), noisy.sum())
    parcels["crop"] = [CROP_CLASSES[k] for k in parcels["crop_id"]]
    return parcels, np.stack(S2), s2_days, np.stack(S1), s1_days


# ---------------------------------------------------------------------------------------------
# 10. SAR flood scenes for Project 3 (Day 94)
# ---------------------------------------------------------------------------------------------
def height_above_nearest_drainage(dem, river):
    """HAND (m): elevation minus the elevation of the nearest river pixel (Euclidean nearest, a simple approximation)."""
    _, (ri, ci) = distance_transform_edt(~river, return_indices=True)
    return np.clip(dem - dem[ri, ci], 0, None)


def make_flood_scene(n=256, pixel=20.0, seed=0, looks=4.4):
    """
    Pre- and post-event Sentinel-1-like VV/VH (dB) with a flood driven by HAND.
    Includes the classic difficulties: speckle, wind-roughened flood water (brighter), flooded vegetation and
    built-up areas (double bounce -> brighter), and radar shadow on steep slopes facing away from the sensor.
    Returns dict: pre (2, n, n), post (2, n, n), flood (bool, new water incl. flooded vegetation), permanent_water,
    lc, hand, slope, condition (int map: 0 none, 1 open flood, 2 wind-roughened, 3 flooded vegetation, 4 flooded urban, 5 shadow).
    """
    rng = np.random.default_rng(seed + 1000)
    ter = make_terrain(n, pixel, seed); lc, _, _ = make_landcover(ter, pixel, seed)
    hand = height_above_nearest_drainage(ter["dem"], ter["river"])
    permanent = lc == 0
    noisy_hand = hand + 2.5 * _field((n, n), 8, rng)
    level = np.percentile(noisy_hand[~permanent], rng.uniform(8, 25))                       # flood stage: 8-25% of the land floods
    flood = (noisy_hand < level) & ~permanent
    base_vv = np.array([-21, -8, -11, -5, -12, -10.0]); base_vh = np.array([-28, -14, -17, -11, -19, -16.0])
    shadow = (ter["slope"] > 14) & (np.cos(np.radians(ter["aspect"] - 80)) > 0.3)          # slopes facing away (descending pass)
    cond = np.zeros((n, n), np.int8)

    def speckled(db):
        return 10 * np.log10(10 ** (db / 10) * rng.gamma(looks, 1 / looks, db.shape))

    def scene(flooded):
        vv = base_vv[lc] + 1.0 * _field((n, n), 3, rng) + 0.05 * (ter["slope"] - 5)
        vh = base_vh[lc] + 1.0 * _field((n, n), 3, rng)
        vv[shadow] = -19 + rng.normal(0, 1, shadow.sum()); vh[shadow] = -26 + rng.normal(0, 1, shadow.sum())
        if flooded:
            wind = _field((n, n), 12, rng) > 1.0
            open_w = flood & np.isin(lc, [2, 4, 5])
            vv = np.where(open_w, -20.5 + rng.normal(0, 0.8, (n, n)), vv); vh = np.where(open_w, -27.5 + rng.normal(0, 0.8, (n, n)), vh)
            vv = np.where(open_w & wind, -13.5, vv); vh = np.where(open_w & wind, -21, vh)
            fveg = flood & (lc == 1); furb = flood & (lc == 3)
            vv = np.where(fveg, vv + 3.0, vv); vh = np.where(fveg, vh + 0.5, vh)
            vv = np.where(furb, vv + 2.0, vv); vh = np.where(furb, vh + 1.0, vh)
            cond[open_w] = 1; cond[open_w & wind] = 2; cond[fveg] = 3; cond[furb] = 4
        return np.stack([speckled(vv), speckled(vh)]).astype(np.float32)

    pre = scene(False); post = scene(True)
    cond[shadow & ~flood] = 5
    return {"pre": pre, "post": post, "flood": flood, "permanent_water": permanent, "lc": lc, "hand": hand.astype(np.float32),
            "slope": ter["slope"], "condition": cond}


# ---------------------------------------------------------------------------------------------
# 11. Terrain hydrology and a landslide-prone area for Project 4 (Day 95)
# ---------------------------------------------------------------------------------------------
_D8 = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]


def fill_sinks(dem, eps=1e-3):
    """Priority-flood depression filling (Barnes et al. 2014) so that every cell drains to the edge."""
    import heapq
    n, m = dem.shape; filled = dem.astype(np.float64).copy(); done = np.zeros((n, m), bool); heap = []
    for r in range(n):
        for c in (0, m - 1):
            heapq.heappush(heap, (filled[r, c], r, c)); done[r, c] = True
    for c in range(1, m - 1):
        for r in (0, n - 1):
            heapq.heappush(heap, (filled[r, c], r, c)); done[r, c] = True
    while heap:
        z, r, c = heapq.heappop(heap)
        for dr, dc in _D8:
            rr, cc = r + dr, c + dc
            if 0 <= rr < n and 0 <= cc < m and not done[rr, cc]:
                done[rr, cc] = True
                if filled[rr, cc] <= z: filled[rr, cc] = z + eps
                heapq.heappush(heap, (filled[rr, cc], rr, cc))
    return filled


def d8_flow_accumulation(dem_filled):
    """Number of upslope cells draining through each cell (D8, steepest descent)."""
    n, m = dem_filled.shape; pad = np.pad(dem_filled, 1, constant_values=np.inf)
    drops = np.stack([(dem_filled - pad[1 + dr:1 + dr + n, 1 + dc:1 + dc + m]) / np.hypot(dr, dc) for dr, dc in _D8])
    k = drops.argmax(0); has = drops.max(0) > 0
    rr, cc = np.mgrid[0:n, 0:m]
    recv = np.where(has, (rr + np.array([d[0] for d in _D8])[k]) * m + cc + np.array([d[1] for d in _D8])[k], -1).ravel()
    acc = np.ones(n * m)
    for i in np.argsort(-dem_filled.ravel(), kind="stable"):
        if recv[i] >= 0: acc[recv[i]] += acc[i]
    return acc.reshape(n, m)


def make_landslide_area(n=400, pixel=20.0, seed=3, relief=2.0):
    """
    A hilly study area with landslide conditioning factors and two landslide inventories (two storms).
    Returns dict: factors (dict of 2-D arrays), lithology_names, lc, true_prob (simulation only),
    inventory DataFrame (row, col, x, y, period), transform.
    """
    rng = np.random.default_rng(seed)
    ter = make_terrain(n, pixel, seed)
    dem = 150 + relief * (ter["dem"] - ter["dem"].min())
    dzdy, dzdx = sobel(dem, axis=0) / (8 * pixel), sobel(dem, axis=1) / (8 * pixel)
    slope = np.degrees(np.arctan(np.hypot(dzdx, dzdy))); aspect = (np.degrees(np.arctan2(-dzdx, dzdy)) + 360) % 360
    filled = fill_sinks(dem); acc = d8_flow_accumulation(filled)
    twi = np.log(acc * pixel / np.tan(np.radians(np.clip(slope, 0.5, None))))
    from scipy.ndimage import laplace
    curvature = -laplace(gaussian_filter(dem, 1.5)) / pixel ** 2 * 100                      # >0 convex, <0 concave (1/100 m)
    streams = acc > 400
    dist_stream = distance_transform_edt(~streams) * pixel
    ter2 = dict(ter, dem=dem.astype(np.float32), slope=slope.astype(np.float32), aspect=aspect.astype(np.float32),
                river=streams & (acc > 4000), dist_river_m=dist_stream.astype(np.float32))
    lc, dist_road, _ = make_landcover(ter2, pixel, seed)
    lith_names = ["granite_gneiss", "shale", "sandstone", "alluvium"]
    lf = _field((n, n), n / 8, rng); lith = np.digitize(lf, np.quantile(lf, [0.35, 0.6, 0.85]))
    lith[(slope < 3) & (dist_stream < 300)] = 3
    rain = 1350 + 120 * (np.mgrid[0:n, 0:n][1] / n - 0.5) * 2 + 60 * _field((n, n), n / 5, rng)
    z = (-8.4 + 3.2 * np.exp(-0.5 * ((slope - 30) / 9) ** 2) + 1.3 * np.exp(-dist_road / 80)
         + np.array([0.0, 1.1, 0.5, -0.6])[lith] + 0.8 * np.tanh(-curvature / 2) + 0.18 * (twi - twi.mean())
         + 0.7 * (rain - 1350) / 120 + np.array([0, -0.9, 0.1, 0.0, 0.5, 0.3])[lc])
    true_prob = 1 / (1 + np.exp(-z))
    inv = []
    for period, scale in [(1, 1.0), (2, 0.45)]:
        storm = np.exp(0.8 * _field((n, n), n / 6, rng))                                     # rainfall footprint of the event
        hit = rng.random((n, n)) < np.clip(true_prob * storm * scale, 0, 1)
        r, c = np.nonzero(hit & (lc != 0))
        inv.append(pd.DataFrame({"row": r, "col": c, "period": period}))
    inv = pd.concat(inv, ignore_index=True).drop_duplicates(["row", "col"])
    T = transform_from_origin((ORIGIN[0] + 15000, ORIGIN[1]), pixel)
    inv["x"], inv["y"] = pixel_centres(T, inv["row"].values, inv["col"].values)
    factors = {"elevation_m": dem, "slope_deg": slope, "aspect_deg": aspect, "curvature": curvature, "TWI": twi,
               "dist_stream_m": dist_stream, "dist_road_m": dist_road, "rainfall_mm": rain, "lithology": lith, "landcover": lc}
    return {"factors": {k: np.asarray(v, np.float32) for k, v in factors.items()}, "lithology_names": lith_names,
            "true_prob": true_prob.astype(np.float32), "inventory": inv, "transform": T}


# ---------------------------------------------------------------------------------------------
# 12. Above-ground biomass for Project 5 (Day 96)
# ---------------------------------------------------------------------------------------------
def make_biomass_area(n=300, pixel=20.0, seed=5, n_plots=250):
    """
    Wooded landscape with true above-ground biomass (AGB, Mg/ha), predictor rasters with realistic saturation
    (optical indices saturate early, C-band SAR around 100-150 Mg/ha), a canopy-height-model product, and field plots
    (simple random sample within the wooded area; allometric/measurement error ~12%).
    Returns dict: predictors (dict of 2-D arrays), agb_true, wooded (mask), plots DataFrame (row, col, x, y, agb), transform.
    """
    rng = np.random.default_rng(seed)
    ter = make_terrain(n, pixel, seed); lc, _, _ = make_landcover(ter, pixel, seed)
    wooded = np.isin(lc, [1, 5])
    h_forest = np.clip(20 + 6 * _field((n, n), 15, rng) + 2.5 * _field((n, n), 2, rng), 6, 36)
    h_shrub = np.clip(3 + 1.2 * _field((n, n), 6, rng), 0.5, 7)
    H = np.where(lc == 1, h_forest, np.where(lc == 5, h_shrub, np.where(lc == 2, 0.8, 0.0)))
    CC = np.where(lc == 1, np.clip(0.75 + 0.15 * _field((n, n), 5, rng), 0.3, 0.98),
                  np.where(lc == 5, np.clip(0.3 + 0.12 * _field((n, n), 4, rng), 0.05, 0.6), np.where(lc == 2, 0.4, 0.02)))
    agb = 2.0 * CC * H ** 1.35 * np.exp(rng.normal(0, 0.1, (n, n)))
    agb[~np.isin(lc, [1, 2, 5])] = 0.0
    nz = lambda s: rng.normal(0, s, (n, n))
    pred = {"NDVI": 0.15 + 0.7 * (1 - np.exp(-3.5 * CC)) + nz(0.03),
            "NDRE": 0.05 + 0.45 * (1 - np.exp(-3.0 * CC)) + nz(0.025),
            "B11": 0.28 - 0.12 * (1 - np.exp(-agb / 60)) + nz(0.012),
            "VH_dB": -22 + 8 * (1 - np.exp(-agb / 70)) + 0.08 * (ter["slope"] - 5) + nz(0.9),
            "VV_dB": -14 + 5 * (1 - np.exp(-agb / 90)) + 0.1 * (ter["slope"] - 5) + nz(0.9),
            "NIR_texture": 0.006 + 0.0004 * np.clip(H, 0, 35) + np.abs(nz(0.008)),
            "elevation_m": ter["dem"], "slope_deg": ter["slope"]}
    pred["canopy_height_product_m"] = np.clip(gaussian_filter(H, 0.8) + nz(2.5), 0, None)     # e.g. a global canopy-height map
    idx = rng.choice(np.flatnonzero(wooded.ravel()), n_plots, replace=False)
    r, c = np.divmod(idx, n)
    rr = np.clip(r + rng.integers(-1, 2, n_plots) * (rng.random(n_plots) < 0.25), 0, n - 1)       # GPS error: 25% of plots off by a pixel
    cc = np.clip(c + rng.integers(-1, 2, n_plots) * (rng.random(n_plots) < 0.25), 0, n - 1)
    plot_agb = agb[rr, cc] * np.exp(rng.normal(0, 0.12, n_plots))                                # allometric + measurement error
    T = transform_from_origin((ORIGIN[0] + 30000, ORIGIN[1]), pixel)
    x, y = pixel_centres(T, r, c)
    plots = pd.DataFrame({"plot_id": np.arange(n_plots), "row": r, "col": c, "x": x, "y": y, "agb_Mg_ha": plot_agb.round(1)})
    return {"predictors": {k: np.asarray(v, np.float32) for k, v in pred.items()}, "agb_true": agb.astype(np.float32),
            "wooded": wooded, "lc": lc, "plots": plots, "transform": T}


# ---------------------------------------------------------------------------------------------
# 13. Multispectral scene patches for Project 6 (Day 97)
# ---------------------------------------------------------------------------------------------
SCENE_CLASSES = ["forest", "cropland", "shrub_grass", "river", "pond_lake", "residential", "industrial", "highway", "mining"]
_SP = {"veg": [0.04, 0.07, 0.04, 0.35], "forest": [0.03, 0.05, 0.03, 0.30], "shrub": [0.06, 0.09, 0.09, 0.25],
       "soil": [0.10, 0.13, 0.16, 0.22], "water": [0.06, 0.07, 0.04, 0.02], "roof": [0.18, 0.19, 0.20, 0.24],
       "asphalt": [0.09, 0.10, 0.11, 0.13], "coal": [0.05, 0.05, 0.06, 0.07], "spoil": [0.15, 0.16, 0.18, 0.20]}


def _scene_patch(cls, size, rng, dry=False):
    S = {k: np.array(v, np.float32) for k, v in _SP.items()}
    if dry:                                                             # dry season: less NIR, more red in vegetation
        for k in ("veg", "forest", "shrub"): S[k] = S[k] * np.array([1.1, 1.05, 1.5, 0.7], np.float32)
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    f = lambda s: gaussian_filter(rng.normal(size=(size, size)), s)
    norm01 = lambda a: (a - a.min()) / (np.ptp(a) + 1e-9)
    bg_mix = norm01(f(4))
    background = (bg_mix[..., None] * S["shrub"] + (1 - bg_mix[..., None]) * S[rng.choice(["veg", "soil", "shrub"])])
    img = background.copy()
    ang = rng.uniform(0, np.pi)
    u = (xx - size / 2) * np.cos(ang) + (yy - size / 2) * np.sin(ang); v = -(xx - size / 2) * np.sin(ang) + (yy - size / 2) * np.cos(ang)
    if cls == "forest":
        crowns = norm01(f(1.3)); img = crowns[..., None] * S["forest"] * 1.15 + (1 - crowns[..., None]) * S["forest"] * 0.6
    elif cls == "shrub_grass":
        img = S["shrub"] * (0.9 + 0.2 * norm01(f(2)))[..., None]
        dots = f(0.7) > 1.2 * f(0.7).std(); img[dots] = S["forest"]
    elif cls == "cropland":
        w = rng.uniform(5, 10); strip = np.floor(u / w).astype(int)
        g = rng.random(strip.max() - strip.min() + 1)[strip - strip.min()]
        img = g[..., None] * S["veg"] + (1 - g[..., None]) * S["soil"]
        img[(np.abs((u / w) % 1) < 0.08)] = S["shrub"]                    # field bunds
    elif cls == "river":
        center = 6 * np.sin(u / rng.uniform(6, 12) + rng.uniform(0, 6)); width = rng.uniform(2, 4.5)
        img[np.abs(v - center) < width] = S["water"]
        img[(np.abs(v - center) >= width) & (np.abs(v - center) < width + 2)] = S["soil"] * 1.1   # sand bars
    elif cls == "pond_lake":
        for _ in range(rng.integers(1, 3)):
            cy, cx, r = rng.uniform(10, size - 10, 2).tolist() + [rng.uniform(5, 12)]
            blob = np.hypot(yy - cy, xx - cx) + 3 * f(2) < r; img[blob] = S["water"]
    elif cls == "residential":
        pitch = rng.uniform(5, 7); m = ((np.abs(u) % pitch) < pitch - 2.5) & ((np.abs(v) % pitch) < pitch - 2.5)
        img[m & (rng.random((size, size)) < 0.95)] = S["roof"] * rng.uniform(0.85, 1.15)
        img[~m] = S["asphalt"]; trees = f(0.8) > 1.6 * f(0.8).std(); img[trees & ~m] = S["veg"]
    elif cls == "industrial":
        img[:] = S["asphalt"]
        for _ in range(rng.integers(2, 5)):
            h, w_ = rng.integers(8, 20, 2); r0, c0 = rng.integers(0, size - 8, 2)
            img[r0:r0 + h, c0:c0 + w_] = S["roof"] * rng.uniform(0.9, 1.3)
    elif cls == "highway":
        width = rng.uniform(2, 3.5); img[np.abs(v) < width] = S["asphalt"] * 1.2; img[np.abs(np.abs(v) - width - 1) < 0.6] = S["soil"]
    elif cls == "mining":
        cy, cx = rng.uniform(size * 0.3, size * 0.7, 2); d = np.hypot(yy - cy, xx - cx) + 2 * f(3)
        pit = d < rng.uniform(14, 24); terr = (np.floor(d / 3) % 2 == 0)
        img[pit & terr] = S["coal"]; img[pit & ~terr] = S["spoil"]
    img = img * rng.uniform(0.8, 1.2) + np.array([1, 0.7, 0.4, 0.1], np.float32) * rng.uniform(0, 0.04 if not dry else 0.07)
    img = gaussian_filter(img, (0.6, 0.6, 0)) + rng.normal(0, 0.01, img.shape)
    return np.clip(img, 0.001, 1).astype(np.float32).transpose(2, 0, 1)


def make_scene_patches(n_per_class, size=48, seed=0, dry=False, classes=None):
    """Multispectral (B, G, R, NIR) scene patches (N, 4, size, size) and labels for the scene classes."""
    rng = np.random.default_rng(seed); classes = classes or SCENE_CLASSES
    X = np.stack([_scene_patch(c, size, rng, dry) for c in classes for _ in range(n_per_class)])
    y = np.repeat(np.arange(len(classes)), n_per_class)
    p = rng.permutation(len(y))
    return X[p], y[p]

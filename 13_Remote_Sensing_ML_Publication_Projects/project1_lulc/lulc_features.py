"""Feature engineering for Project 1 (LULC). Written by Day 89 and imported by Days 90-91."""
import numpy as np
import rasterio
from scipy.ndimage import uniform_filter

SEASONS = ["winter", "premonsoon", "monsoon", "postmonsoon"]
INDICES = ["NDVI", "MNDWI", "NDBI", "NDRE", "BSI"]
S2_BANDS = ["B2", "B3", "B4", "B5", "B6", "B7", "B8", "B8A", "B11", "B12"]

def local_std(a, size):
    m = uniform_filter(a, size)
    return np.sqrt(np.clip(uniform_filter(a * a, size) - m * m, 0, None))

def local_semivariance(a, size):
    """Mean half squared difference between horizontally / vertically adjacent pixels in a window.
    For continuous data this is what the GLCM 'contrast' measures for a lag of one pixel."""
    dx = np.zeros_like(a); dx[:, :-1] = (a[:, 1:] - a[:, :-1]) ** 2
    dy = np.zeros_like(a); dy[:-1] = (a[1:] - a[:-1]) ** 2
    return 0.5 * uniform_filter(0.5 * (dx + dy), size)

def build_feature_stack(raw_dir):
    """Return (features (F, H, W) float32, names, groups, gap_mask (H, W), rasterio profile)."""
    with rasterio.open(f"{raw_dir}/feature_stack_2024.tif") as src:
        base = src.read().astype(np.float32); names = list(src.descriptions); profile = src.profile
    gap = np.isnan(base).any(0)                                    # pixels with no clear observation in some season
    med = np.nanmedian(base.reshape(len(base), -1), axis=1)
    for i in range(len(base)):                                     # gap filling: band median (documented in the paper)
        base[i][np.isnan(base[i])] = med[i]
    get = lambda n: base[names.index(n)]
    feats, fnames, groups = [], [], []

    def add(a, n, g):
        feats.append(np.asarray(a, np.float32)); fnames.append(n); groups.append(g)

    for s in SEASONS:
        for bnd in S2_BANDS:
            add(get(f"{s}_{bnd}"), f"{s}_{bnd}", "S2_monsoon" if s == "monsoon" else "S2_multiseason")
        for ix in INDICES:
            add(get(f"{s}_{ix}"), f"{s}_{ix}", "S2_index")
    ndvi = np.stack([get(f"{s}_NDVI") for s in SEASONS])
    add(ndvi.max(0), "NDVI_max", "temporal"); add(ndvi.min(0), "NDVI_min", "temporal")
    add(ndvi.max(0) - ndvi.min(0), "NDVI_amplitude", "temporal"); add(ndvi.std(0), "NDVI_sd", "temporal")
    add(np.argmax(ndvi, 0), "NDVI_peak_season", "temporal")
    add(np.max([get(f"{s}_MNDWI") for s in SEASONS], 0), "MNDWI_max", "temporal")
    for s in ["monsoon", "winter"]:
        add(get(f"{s}_VV"), f"{s}_VV", "S1"); add(get(f"{s}_VH"), f"{s}_VH", "S1")
        add(get(f"{s}_VH") - get(f"{s}_VV"), f"{s}_VHminusVV", "S1")
    nir, nd = get("monsoon_B8"), get("monsoon_NDVI")
    for size in (3, 7):
        add(local_std(nir, size), f"tex_B8_sd{size}", "texture"); add(local_semivariance(nir, size), f"tex_B8_semivar{size}", "texture")
    add(local_std(nd, 7), "tex_NDVI_sd7", "texture")
    slope, aspect = np.radians(get("slope_deg")), np.radians(get("aspect_deg"))
    add(get("elevation_m"), "elevation_m", "terrain"); add(get("slope_deg"), "slope_deg", "terrain")
    add(np.cos(aspect) * np.sin(slope), "northness", "terrain"); add(np.sin(aspect) * np.sin(slope), "eastness", "terrain")
    return np.stack(feats), fnames, groups, gap, profile

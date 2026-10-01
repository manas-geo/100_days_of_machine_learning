"""Project 1 pipeline: training table -> model -> spatial CV -> wall-to-wall map -> stratified accuracy assessment."""
import hashlib, json, platform, time
from pathlib import Path
import numpy as np
import pandas as pd
import geopandas as gpd
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold, cross_val_predict
from sklearn.metrics import f1_score, accuracy_score
import rs_utils as ru
import lulc_features as lf

def sha256(path, n=16):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:n]

def training_table(raw_dir, F, fnames, transform, block_size):
    pts = gpd.read_file(Path(raw_dir) / "field_points.gpkg")
    r, c = ru.points_to_rowcol(transform, pts.geometry.x.values, pts.geometry.y.values)
    keep = ~pd.DataFrame({"r": r, "c": c}).duplicated().values
    pts, r, c = pts[keep].reset_index(drop=True), r[keep], c[keep]
    X = F[:, r, c].T
    blocks = ru.spatial_blocks(pts.geometry.x.values, pts.geometry.y.values, block_size)
    return X, pts["class_id"].values, blocks, (r, c)

def make_model(cfg):
    m = cfg["model"]
    return RandomForestClassifier(n_estimators=m["n_estimators"], max_features=m["max_features"], min_samples_leaf=m["min_samples_leaf"],
                                  n_jobs=-1, random_state=cfg["seed"])

def spatial_cv(cfg, X, y, blocks):
    pred = cross_val_predict(make_model(cfg), X, y, cv=GroupKFold(cfg["cv"]["n_folds"]), groups=blocks)
    return {"cv_overall_accuracy": float(accuracy_score(y, pred)), "cv_macro_f1": float(f1_score(y, pred, average="macro"))}

def predict_map(model, F):
    K, H, W = F.shape
    out = np.zeros((H, W), np.uint8)
    for r0 in range(0, H, 100):
        out[r0:r0 + 100] = model.predict(F[:, r0:r0 + 100].reshape(K, -1).T).reshape(-1, W)
    return out

def accuracy_assessment(cfg, lc_map, reference_raster):
    K = len(cfg["classes"]); H, W = lc_map.shape; v = cfg["validation"]
    counts = {k: int((lc_map == k).sum()) for k in range(K)}
    Wi = np.array([counts[k] for k in range(K)]) / (H * W); U = np.full(K, v["user_accuracy_guess"])
    n_total = int(np.ceil((np.sum(Wi * np.sqrt(U * (1 - U))) / v["target_se_oa"]) ** 2))
    rare = Wi < 0.05; alloc = np.where(rare, v["n_rare_class"], 0)
    alloc[~rare] = np.round((n_total - alloc.sum()) * Wi[~rare] / Wi[~rare].sum())
    r, c, strata = ru.stratified_random_sample(lc_map, {k: int(a) for k, a in enumerate(alloc)}, seed=cfg["seed"] + 1)
    ref = reference_raster[r, c]                       # in a real study: interpreted reference labels (CSV) instead
    pix_ha = cfg["pixel_size_m"] ** 2 / 1e4
    return ru.olofsson_accuracy(strata, ref, counts, pix_ha, cfg["classes"])

def provenance(cfg, raw_dir, t0):
    import rasterio, geopandas, scipy
    return {"timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "runtime_s": round(time.time() - t0, 1),
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": {"numpy": np.__version__, "pandas": pd.__version__, "scikit-learn": sklearn.__version__, "scipy": scipy.__version__,
                         "rasterio": rasterio.__version__, "geopandas": geopandas.__version__},
            "inputs_sha256": {p.name: sha256(p) for p in sorted(Path(raw_dir).glob("*")) if p.suffix in (".tif", ".gpkg") and not p.name.startswith("_")},
            "config_sha256": hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()[:16]}

"""Reproduce every result of the paper:  python run_all.py --config config.yaml"""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import yaml
sys.path.insert(0, str(Path(__file__).parent / "src"))
import rasterio
import rs_utils as ru
import lulc_features as lf
import pipeline as pl

def main(config_path):
    t0 = time.time()
    root = Path(config_path).parent
    cfg = yaml.safe_load(Path(config_path).read_text())
    np.random.seed(cfg["seed"])
    raw = (root / cfg["raw_data_dir"]).resolve(); out = root / cfg["output_dir"]; out.mkdir(exist_ok=True)
    print("[1/5] features"); F, fnames, groups, gap, profile = lf.build_feature_stack(raw)
    print("[2/5] training table"); X, y, blocks, _ = pl.training_table(raw, F, fnames, profile["transform"], cfg["cv"]["block_size_m"])
    print("[3/5] spatial CV"); cv = pl.spatial_cv(cfg, X, y, blocks)
    print("[4/5] final model and map"); model = pl.make_model(cfg).fit(X, y); lc_map = pl.predict_map(model, F)
    ru.write_geotiff(out / "lulc_map.tif", lc_map, profile["transform"], nodata=255, dtype="uint8", band_names=["land_cover"])
    print("[5/5] accuracy assessment")
    with rasterio.open(raw / "_truth_landcover_SIMULATION_ONLY.tif") as src: reference = src.read(1)
    summary, pmat = pl.accuracy_assessment(cfg, lc_map, reference)
    summary.to_csv(out / "table_accuracy_area.csv", index=False); pmat.to_csv(out / "error_matrix_area_proportions.csv")
    results = {"n_training": int(len(y)), "n_features": len(fnames), **cv,
               "OA": float(summary.attrs["OA"]), "OA_95CI": float(summary.attrs["OA_95CI"]),
               "areas_ha": {r["class"]: [round(float(r["adjusted_area_ha"]), 2), round(float(r["area_95CI_ha"]), 2)] for _, r in summary.iterrows()},
               "map_sha256": pl.hashlib.sha256(lc_map.tobytes()).hexdigest()[:16],
               "provenance": pl.provenance(cfg, raw, t0)}
    (out / "results.json").write_text(json.dumps(results, indent=2))
    print(f"done in {time.time() - t0:.0f} s -> {out / 'results.json'}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--config", default=str(Path(__file__).parent / "config.yaml"))
    main(ap.parse_args().config)

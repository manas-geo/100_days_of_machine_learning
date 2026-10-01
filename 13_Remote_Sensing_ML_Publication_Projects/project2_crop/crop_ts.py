"""Time-series preprocessing and features for Project 2 (crop types). Written by Day 92, used by Day 93."""
import numpy as np

S2_BANDS = ["B2", "B3", "B4", "B5", "B8", "B11"]

def whittaker(Y, lam=20.0, d=2, chunk=500):
    """Weighted Whittaker smoother along axis 1 of Y (N, T); NaN = missing (weight 0). Returns (N, T)."""
    N, T = Y.shape
    D = np.diff(np.eye(T), d, axis=0); P = lam * D.T @ D
    w_all = (~np.isnan(Y)).astype(np.float64); Y0 = np.nan_to_num(Y.astype(np.float64))
    out = np.empty((N, T))
    for s in range(0, N, chunk):
        w = w_all[s:s + chunk]
        A = P[None] + w[:, :, None] * np.eye(T)[None] + 1e-9 * np.eye(T)[None]
        out[s:s + chunk] = np.linalg.solve(A, (w * Y0[s:s + chunk])[..., None])[..., 0]
    return out

def screen_outliers(s2, thresh=0.03, lam=20.0, n_iter=2):
    """Flag undetected thin clouds: B2 well above a smooth fit of B2. Returns a copy of s2 with outliers set to NaN."""
    s2 = s2.copy()
    for _ in range(n_iter):
        b2 = s2[..., 0]; fit = whittaker(b2, lam)
        bad = (b2 - fit) > thresh
        s2[bad] = np.nan
    return s2

def ndvi(s2):
    return (s2[..., 4] - s2[..., 2]) / (s2[..., 4] + s2[..., 2])

def smooth_all(s2, lam=20.0):
    """Whittaker-smooth every band and NDVI. Returns (N, T, 7): 6 bands + NDVI."""
    bands = [whittaker(s2[..., b], lam) for b in range(s2.shape[-1])]
    return np.stack(bands + [whittaker(ndvi(s2), lam)], axis=-1).astype(np.float32)

def harmonic_fit(y, days, n_harm=3, period=365.0):
    """Least-squares harmonic coefficients per row of y (N, T) using valid observations. Returns (N, 1 + 2 n_harm)."""
    t = 2 * np.pi * np.asarray(days) / period
    X = np.column_stack([np.ones_like(t)] + [f(k * t) for k in range(1, n_harm + 1) for f in (np.cos, np.sin)])
    out = np.zeros((y.shape[0], X.shape[1]))
    for i, row in enumerate(y):
        ok = ~np.isnan(row)
        if ok.sum() >= X.shape[1] + 2:
            out[i] = np.linalg.lstsq(X[ok], row[ok], rcond=None)[0]
    return out

def season_metrics(v, days, window, prefix, frac=0.5):
    """Phenological metrics of a smoothed index v (N, T) within a day window: threshold method at frac of the amplitude."""
    m = (days >= window[0]) & (days <= window[1]); v = v[:, m]; d = days[m]; step = np.median(np.diff(d))
    base = np.minimum(v[:, :3].mean(1), v[:, -3:].mean(1)); peak = v.max(1); amp = peak - base
    above = v >= (base + frac * amp)[:, None]
    sos = d[above.argmax(1)]; eos = d[len(d) - 1 - above[:, ::-1].argmax(1)]
    return {f"{prefix}_SOS": sos, f"{prefix}_POS": d[v.argmax(1)], f"{prefix}_EOS": eos, f"{prefix}_LOS": eos - sos,
            f"{prefix}_peak": peak, f"{prefix}_amplitude": amp, f"{prefix}_integral": np.clip(v - base[:, None], 0, None).sum(1) * step,
            f"{prefix}_greenup_rate": np.max(np.diff(v, axis=1), 1) / step, f"{prefix}_senescence_rate": -np.min(np.diff(v, axis=1), 1) / step}

def sar_metrics(s1, days):
    vv, vh = s1[..., 0], s1[..., 1]
    w = lambda a, b: (days >= a) & (days <= b)
    return {"VH_min_JulAug": vh[:, w(30, 92)].min(1), "VH_mean_Sep": vh[:, w(92, 122)].mean(1), "VH_max_kharif": vh[:, w(0, 180)].max(1),
            "VH_mean_rabi": vh[:, w(180, 300)].mean(1), "VV_mean_rabi": vv[:, w(180, 300)].mean(1),
            "VHminusVV_Sep": (vh - vv)[:, w(92, 122)].mean(1), "VH_range": vh.max(1) - vh.min(1)}

def to_grid(arr, days, grid):
    """Linear interpolation of a smoothed (N, T, C) array onto a regular day grid."""
    return np.stack([np.stack([np.interp(grid, days, arr[i, :, c]) for c in range(arr.shape[2])], -1) for i in range(arr.shape[0])])

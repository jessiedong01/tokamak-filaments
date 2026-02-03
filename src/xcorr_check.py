"""
Independent radial-velocity estimate that uses no detection or tracking: for each shot,
cross-correlate the row-averaged fluctuation profile of consecutive frames inside the
near-edge window and read the sub-pixel lag. Writes results/xcorr.parquet with one row
per shot: the median lag per frame (px), its quartiles, and the fraction of frames with
a usable correlation peak.

Usage:
  python src/xcorr_check.py [--workers 4] [--frames 4000]
"""
import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filaments import fluctuation, edge_column  # noqa: E402
from calibrate import limb_peak  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
WIN = (-10, 40)      # columns relative to the limb peak, same window as the track selection
MAX_LAG = 6          # px per frame


def one(path: Path, n_frames: int) -> dict:
    shot = int(path.stem)
    try:
        z = np.load(path)
        fr, t = z["frames"], z["time"].astype(float)
        fps = float(1 / np.median(np.diff(t)))
        i0 = max(0, len(fr) // 2 - n_frames // 2); sub = fr[i0:i0 + n_frames]
        edge, s = edge_column(sub)
        prof = ndi.gaussian_filter1d(ndi.uniform_filter1d(sub.astype(np.float32).mean(axis=1), size=200, axis=0, mode="nearest"), 2.0, axis=1)
        peak = limb_peak(prof, s, edge)
        xp = int(round(np.nanmedian(peak)))
        lo, hi = (xp + s * WIN[0], xp + s * WIN[1]) if s == 1 else (xp - WIN[1], xp - WIN[0])
        lo, hi = max(0, min(lo, hi)), min(sub.shape[2], max(lo, hi))
        d = fluctuation(sub)[:, :, lo:hi].mean(axis=1)          # (n, w) row-averaged fluctuation in the window
        d = d - d.mean(axis=1, keepdims=True)
        lags, ok = [], 0
        for k in range(len(d) - 1):
            a, b = d[k], d[k + 1]
            if a.std() < 1e-3 or b.std() < 1e-3:
                lags.append(np.nan); continue
            c = np.array([np.dot(a[max(0, -L):len(a) - max(0, L)], b[max(0, L):len(b) - max(0, -L)]) /
                          max(1, len(a) - abs(L)) for L in range(-MAX_LAG, MAX_LAG + 1)])
            j = int(np.argmax(c))
            if j == 0 or j == len(c) - 1:
                lags.append(np.nan); continue
            den = c[j - 1] - 2 * c[j] + c[j + 1]
            lag = (j - MAX_LAG) + (0.5 * (c[j - 1] - c[j + 1]) / den if abs(den) > 1e-9 else 0.0)
            lags.append(s * lag); ok += 1
        lags = np.array(lags)
        return {"shot": shot, "fps": fps, "n_pairs": len(lags), "frac_ok": ok / max(1, len(lags)),
                "lag_med": float(np.nanmedian(lags)), "lag_q25": float(np.nanpercentile(lags, 25)), "lag_q75": float(np.nanpercentile(lags, 75)),
                "lag_mean": float(np.nanmean(lags)), "win_lo": lo, "win_hi": hi, "sol_sign": s}
    except Exception as e:
        return {"shot": shot, "err": f"{type(e).__name__}: {str(e)[:80]}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); ap.add_argument("--frames", type=int, default=4000)
    args = ap.parse_args()
    paths = sorted(q for q in (ROOT / "data" / "shots").glob("*.npz") if q.stem.isdigit())
    rows = []
    with ProcessPoolExecutor(args.workers) as ex:
        for r in ex.map(one, paths, [args.frames] * len(paths)):
            rows.append(r); print(f"[{time.strftime('%H:%M:%S')}] {r}", flush=True)
    pd.DataFrame(rows).to_parquet(ROOT / "results" / "xcorr.parquet")


if __name__ == "__main__":
    main()

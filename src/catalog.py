"""
Build the shot list. Every rbb recording in 48x256 mode is a 100 kHz burst embedded in a
slow (1-2 kHz) record of the whole discharge, so the selection has to look at the time
base, not just the frame count.

Writes
  data/strip_timebase.parquet   one row per strip recording: fast-burst start/end, frame counts
  data/candidates.parquet       recordings whose burst overlaps the plasma flat top, with shot metadata

Criteria (stated in the paper): flat top defined and at least 50 ms long, peak plasma
current at least 400 kA, and at least 2,000 fast frames (20 ms) inside the flat top
extended by 20 ms on each side.

Usage:
  python src/catalog.py [--workers 8]
"""
import argparse
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

from fetch import BASE, PAD

ROOT = Path(__file__).resolve().parent.parent
FAST_DT = 2e-5          # frames closer than this are part of the 100 kHz burst
MIN_FAST_FRAMES = 2000
MIN_FLAT_TOP_S = 0.05
MIN_IP_KA = 400.0


def timebase(shot: int) -> dict:
    try:
        g = zarr.open_group(zarr.storage.FsspecStore.from_url(f"{BASE}/{shot}.zarr", storage_options={"asynchronous": True}), mode="r")
        t = np.asarray(g["rbb"]["time"][:], dtype=float)
        dt = np.diff(t); fast = dt < FAST_DT
        if fast.sum() < 10:
            return {"shot": shot, "n_total": len(t), "n_fast": int(fast.sum())}
        idx = np.flatnonzero(fast)
        return {"shot": shot, "n_total": len(t), "n_fast": int(fast.sum()), "fast_t0": float(t[idx[0]]), "fast_t1": float(t[idx[-1] + 1]),
                "fast_dt_us": float(np.median(dt[fast]) * 1e6), "slow_dt_us": float(np.median(dt[~fast]) * 1e6) if (~fast).any() else np.nan,
                "n_bursts": int(np.sum(np.diff(idx) > 1) + 1)}
    except Exception as e:
        return {"shot": shot, "err": f"{type(e).__name__}: {str(e)[:60]}"}


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=8); args = ap.parse_args()
    L = pd.read_parquet(ROOT / "data" / "camera_listing.parquet")
    strip = L[(L.src == "rbb") & (L.h == 48) & (L.w == 256)].drop_duplicates("shot").copy()
    strip["mode"] = strip.h.astype(str) + "x" + strip.w.astype(str)
    t0 = time.time()
    with ThreadPoolExecutor(args.workers) as ex:
        tb = pd.DataFrame(list(ex.map(timebase, strip.shot.tolist())))
    print(f"time bases of {len(tb)} recordings read in {time.time() - t0:.0f} s; {tb.n_fast.ge(10).sum()} have a fast burst")
    meta = pd.read_parquet(ROOT / "data" / "shots_meta.parquet").rename(columns={"shot_id": "shot"})
    df = strip.merge(tb, on="shot", how="left").merge(meta, on="shot", how="left")
    w0, w1 = df.plasma_flat_top_start_time - PAD, df.plasma_flat_top_end_time + PAD
    overlap = (np.minimum(df.fast_t1, w1) - np.maximum(df.fast_t0, w0)).clip(lower=0)
    df["n_fast_in_window"] = (overlap / (df.fast_dt_us * 1e-6)).fillna(0).astype(int)
    df["flat_top_s"] = df.plasma_flat_top_end_time - df.plasma_flat_top_start_time
    df["ok_flat_top"] = df.flat_top_s >= MIN_FLAT_TOP_S
    df["ok_current"] = df.plasma_max_current.abs() >= MIN_IP_KA
    df["ok_burst"] = df.n_fast_in_window >= MIN_FAST_FRAMES
    df.drop(columns=[c for c in meta.columns if c != "shot"]).to_parquet(ROOT / "data" / "strip_timebase.parquet")
    cand = df[df.ok_flat_top & df.ok_current & df.ok_burst].copy()
    cand["flat_top_ms"] = cand.flat_top_s * 1e3
    cand.to_parquet(ROOT / "data" / "candidates.parquet")
    print(f"strip recordings {len(df)}; flat top ok {df.ok_flat_top.sum()}; current ok {df.ok_current.sum()}; burst in window {df.ok_burst.sum()}; candidates {len(cand)}")
    print(cand.campaign.value_counts().to_dict())
    print("burst length (ms) median:", round(float((df.fast_t1 - df.fast_t0).median() * 1e3)), " fast dt (us):", df.fast_dt_us.round(1).value_counts().head(3).to_dict(), " slow dt (us):", df.slow_dt_us.round(0).value_counts().head(3).to_dict())


if __name__ == "__main__":
    main()

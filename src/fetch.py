"""
Download fast-camera strips and the plasma signals needed per shot from FAIR-MAST.

For each candidate shot this saves data/shots/<shot>.npz with
  frames   uint8 (n, h, w)   camera frames in the analysis window
  time     float32 (n,)      frame times (s)
  attrs    json              camera attributes (view, ROI, exposure, ...)
  efit     json              EFIT: time, outboard LCFS radius at |Z|<5 cm, Ip, q95, R_axis, a
  esm      json              ne_bar(t), n_greenwald(t)
  dalpha   float32           D-alpha trace (xim/da_to10) and its time base
  aye      json              edge Thomson: time, r, te, ne (if present)

The analysis window is the plasma flat top, padded by 20 ms, clipped to the camera record.

Usage:
  python src/fetch.py --candidates data/candidates.parquet --workers 6
"""
import argparse
import io
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import zarr

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "shots"
BASE = "https://s3.echo.stfc.ac.uk/mast/level1/shots"
PAD = 0.02  # s


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def arr(g, path):
    try:
        return np.asarray(g[path][:])
    except Exception:
        return None


def efit_summary(g):
    """Scalar EFIT series plus the outboard LCFS radius at the magnetic axis height.

    rbdry/zbdry in this archive are scalar per time (the outboard boundary point);
    lcfs_r/lcfs_z hold the full contour when present. Both are kept.
    """
    efm = g["efm"]
    t = arr(efm, "time")
    out = {"time": t.astype(float).tolist()}
    for k in ("rbdry", "zbdry", "plasma_current", "plasma_current_c", "plasma_current_x", "q_95",
              "magnetic_axis_r", "magnetic_axis_z", "minor_radius", "geom_axis_rc", "elongation",
              "bvac_rmag", "bphi_rmag", "bvac_val", "lcfs_length"):
        v = arr(efm, k)
        if v is not None and v.ndim == 1 and len(v) == len(t):
            out[k] = v.astype(float).tolist()
    if "lcfs_r" in efm and "lcfs_z" in efm:
        lr_all = np.asarray(efm["lcfs_r"][:], dtype=float); lz_all = np.asarray(efm["lcfs_z"][:], dtype=float)
        za = np.asarray(out.get("magnetic_axis_z", np.zeros(len(t))), dtype=float)
        r_out = []
        for i in range(len(t)):
            lr, lz = lr_all[i], lz_all[i]
            good = np.isfinite(lr) & (lr > 0.1) & np.isfinite(lz)
            if good.sum() < 10 or np.nanmax(lr[good]) - np.nanmin(lr[good]) < 0.3:
                r_out.append(np.nan); continue
            m = good & (np.abs(lz - za[i]) < 0.05)
            r_out.append(float(np.nanmax(lr[m])) if m.any() else np.nan)
        out["r_out_mid"] = r_out
    for k in ("cnvrgd_times",):
        v = arr(efm, k)
        if v is not None:
            out[k] = v.astype(float).tolist()
    return out


def fetch(row):
    shot, src = int(row.shot), row.src
    out = OUT / f"{shot}.npz"
    if out.exists():
        return f"{shot}: exists"
    try:
        st = zarr.storage.FsspecStore.from_url(f"{BASE}/{shot}.zarr", storage_options={"asynchronous": True})
        g = zarr.open_group(st, mode="r")
        cam = g[src]; attrs = dict(cam.attrs); t = np.asarray(cam["time"][:], dtype=np.float32)
        t0 = max(float(t[0]), float(row.plasma_flat_top_start_time) - PAD)
        t1 = min(float(t[-1]), float(row.plasma_flat_top_end_time) + PAD)
        i0, i1 = int(np.searchsorted(t, t0)), int(np.searchsorted(t, t1))
        # keep only the 100 kHz burst: the recording also holds slow (1-2 kHz) frames of the whole discharge
        dt = np.diff(t); fast = np.r_[dt[:1] < 2e-5, dt < 2e-5]
        idx = np.flatnonzero(fast[i0:i1]) + i0
        if len(idx):
            i0, i1 = int(idx[0]), int(idx[-1]) + 1
        if i1 - i0 < 500:
            return f"{shot}: only {i1 - i0} frames in window, skipped"
        frames = np.asarray(cam["data"][i0:i1])
        payload = {"frames": frames, "time": t[i0:i1], "attrs": json.dumps(attrs, default=str)}
        if "efm" in g:
            payload["efit"] = json.dumps(efit_summary(g))
        if "esm" in g:
            esm = g["esm"]
            payload["esm"] = json.dumps({k: arr(esm, k).astype(float).tolist() for k in ("time", "ne_bar", "n_greenwald", "p_loss", "w_dot") if arr(esm, k) is not None})
        if "ane" in g:  # CO2 interferometer line density (the early campaigns have no esm summary)
            ane = g["ane"]
            payload["ane"] = json.dumps({k: arr(ane, k).astype(float).tolist() for k in ane.array_keys()
                                         if arr(ane, k) is not None and arr(ane, k).ndim == 1 and arr(ane, k).size < 200000})
        if "xim" in g and "da_to10" in g["xim"]:
            xt = arr(g["xim"], "time"); da = arr(g["xim"], "da_to10")
            sel = (xt >= t0 - 0.05) & (xt <= t1 + 0.05)
            payload["dalpha_t"] = xt[sel].astype(np.float32); payload["dalpha"] = da[sel].astype(np.float32)
        for name, rkey in (("aye", "r"), ("ayc", "radius")):   # edge and core Thomson scattering
            if name in g:
                ts = g[name]
                payload[name] = json.dumps({k: arr(ts, k).astype(float).tolist() for k in ("time", rkey, "te", "ne", "te_error", "ne_error")
                                            if arr(ts, k) is not None})
        OUT.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.rename(out)
        return f"{shot}: {len(frames)} frames ({t0*1e3:.0f}-{t1*1e3:.0f} ms), {out.stat().st_size/1e6:.0f} MB"
    except Exception as e:
        return f"{shot}: ERROR {type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default=str(ROOT / "data" / "candidates.parquet"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--src", default="rbb")
    ap.add_argument("--mode", default="48x256")
    args = ap.parse_args()
    c = pd.read_parquet(args.candidates)
    c = c[(c.src == args.src) & (c["mode"] == args.mode)].drop_duplicates("shot").sort_values("shot")
    log(f"{len(c)} shots to fetch ({args.src} {args.mode})")
    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(fetch, c.itertuples()):
            log(msg)
    log("done")


if __name__ == "__main__":
    main()

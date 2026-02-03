"""
Re-pull the small per-shot signals (EFIT scalars, interferometer, Thomson, D-alpha)
for shots whose npz predates a fetcher change, without re-downloading the frames.

Usage:
  python src/refresh.py [--workers 6]
"""
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import zarr

from fetch import BASE, OUT, arr, efit_summary

NEEDED = ("ayc", "ane")   # keys whose absence marks an old file


def refresh(path: Path) -> str:
    shot = int(path.stem)
    try:
        z = np.load(path)
        if all(k in z.files for k in NEEDED) and "plasma_current_c" in json.loads(str(z["efit"])):
            return f"{shot}: current"
        payload = {k: z[k] for k in z.files}
        st = zarr.storage.FsspecStore.from_url(f"{BASE}/{shot}.zarr", storage_options={"asynchronous": True})
        g = zarr.open_group(st, mode="r")
        if "efm" in g:
            payload["efit"] = json.dumps(efit_summary(g))
        if "ane" in g:
            ane = g["ane"]
            payload["ane"] = json.dumps({k: arr(ane, k).astype(float).tolist() for k in ane.array_keys()
                                         if arr(ane, k) is not None and arr(ane, k).ndim == 1 and arr(ane, k).size < 200000})
        for name, rkey in (("aye", "r"), ("ayc", "radius")):
            if name in g:
                ts = g[name]
                payload[name] = json.dumps({k: arr(ts, k).astype(float).tolist() for k in ("time", rkey, "te", "ne", "te_error", "ne_error")
                                            if arr(ts, k) is not None})
        if "esm" in g:
            esm = g["esm"]
            payload["esm"] = json.dumps({k: arr(esm, k).astype(float).tolist() for k in ("time", "ne_bar", "n_greenwald", "p_loss", "w_dot") if arr(esm, k) is not None})
        tmp = path.with_suffix(".tmp.npz")
        np.savez_compressed(tmp, **payload)
        tmp.rename(path)
        return f"{shot}: refreshed ({', '.join(k for k in ('ayc', 'aye', 'ane', 'esm') if k in payload)})"
    except Exception as e:
        return f"{shot}: ERROR {type(e).__name__}: {str(e)[:120]}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=6); args = ap.parse_args()
    paths = sorted(q for q in OUT.glob("*.npz") if q.stem.isdigit())
    with ThreadPoolExecutor(args.workers) as ex:
        for msg in ex.map(refresh, paths):
            print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


if __name__ == "__main__":
    main()

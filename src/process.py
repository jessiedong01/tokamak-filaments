"""
Run detection, tracking, and calibration sampling on every downloaded shot.

Per shot this writes results/tracks/<shot>.parquet (one row per track) and
results/shotinfo/<shot>.json (edge column, SOL direction, noise level, ELM times,
frame counts). Calibration is done at aggregation time (aggregate.py). Shots already processed are skipped.

ELMs: D-alpha spikes above median + 6 MAD within the camera window; tracks within
+/- ELM_PAD of a spike are flagged so they can be excluded from the inter-ELM/L-mode set.

Usage:
  python src/process.py [--workers 4] [--chunk 3000]
"""
import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from filaments import detect, edge_column, fluctuation, link_tracks, lucas_kanade, noise_sigma, summarize_tracks

ROOT = Path(__file__).resolve().parent.parent
SHOTS = ROOT / "data" / "shots"
TRACKS = ROOT / "results" / "tracks"
INFO = ROOT / "results" / "shotinfo"
ELM_PAD = 0.001   # s
OVERLAP = 60      # frames of overlap between chunks so tracks are not cut


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def elm_times(z) -> np.ndarray:
    if "dalpha" not in z.files:
        return np.array([])
    da, tt = z["dalpha"].astype(float), z["dalpha_t"].astype(float)
    if len(da) < 100:
        return np.array([])
    base = ndi.median_filter(da, size=2001)             # ~2 ms at 1 MHz
    resid = da - base
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid))) + 1e-9
    spikes = resid > 6 * mad
    lab, n = ndi.label(spikes)
    if n == 0:
        return np.array([])
    centers = ndi.center_of_mass(resid, lab, range(1, n + 1))
    return np.array([tt[int(c[0])] for c in centers])


def process(path: Path, chunk: int) -> str:
    shot = int(path.stem)
    out = TRACKS / f"{shot}.parquet"
    if out.exists():
        return f"{shot}: exists"
    try:
        z = np.load(path)
        frames, t = z["frames"], z["time"].astype(float)
        edge_all, sol_sign = edge_column(frames[: min(len(frames), 6000)])
        elms = elm_times(z)
        rows, sigmas = [], []
        n = len(frames)
        start = 0
        while start < n:
            stop = min(n, start + chunk)
            sub = frames[start:stop]
            d = fluctuation(sub)
            edge, _ = edge_column(sub)
            sigmas.append(noise_sigma(d))
            dets = detect(d, edge, sol_sign)
            lk = lucas_kanade(d, dets)
            tracks = link_tracks(dets)
            keep = [tr for tr in tracks if tr[0].frame < stop - start - OVERLAP or stop == n]
            # per-track LK flow: mean over its detections
            det_index = {id(dd): k for k, dd in enumerate(dets)}
            keep = [tr for tr in keep if len(tr) >= 5]
            for tr, tsum in zip(keep, summarize_tracks(keep, sol_sign, min_len=5)):
                flows = [lk[det_index[id(dd)]][0] for dd in tr if np.isfinite(lk[det_index[id(dd)]][0])]
                f0 = start + tsum.start_frame
                t0 = t[f0]
                rows.append({
                    "shot": shot, "frame0": f0, "t0": t0, "n": tsum.n, "x0": tsum.x0, "y0": tsum.y0,
                    "vx_px": tsum.vx_px, "vx_err": tsum.vx_err, "lk_vx_px": sol_sign * float(np.mean(flows)) if flows else np.nan,
                    "fwhm_px": tsum.fwhm_x, "amp": tsum.amp, "sol_dist_px": tsum.sol_dist, "angle": tsum.angle,
                    "elong": tsum.elong, "edge_px": float(edge[tsum.start_frame]),
                    "near_elm": bool(len(elms) and np.min(np.abs(elms - t0)) < ELM_PAD),
                })
            start = stop - OVERLAP if stop < n else n
        df = pd.DataFrame(rows)
        TRACKS.mkdir(parents=True, exist_ok=True); INFO.mkdir(parents=True, exist_ok=True)
        df.to_parquet(out)
        info = {"shot": shot, "n_frames": int(n), "t_start": float(t[0]), "t_end": float(t[-1]),
                "fps": float(1 / np.median(np.diff(t))), "sol_sign": int(sol_sign),
                "edge_px_median": float(np.median(edge_all)), "noise_sigma": float(np.median(sigmas)),
                "n_tracks": int(len(df)), "n_elms": int(len(elms)), "elm_times": elms.tolist(),
                "calib_samples": [], "attrs": json.loads(str(z["attrs"]))}
        (INFO / f"{shot}.json").write_text(json.dumps(info))
        return f"{shot}: {n} frames, {len(df)} tracks, {len(elms)} ELMs, sol_sign {sol_sign:+d}"
    except Exception as e:
        return f"{shot}: ERROR {type(e).__name__}: {str(e)[:150]}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--chunk", type=int, default=3000)
    ap.add_argument("--loop", action="store_true", help="keep polling for new shots until a STOP file appears")
    args = ap.parse_args()
    while True:
        paths = sorted(q for q in SHOTS.glob("*.npz") if q.stem.isdigit())
        todo = [p for p in paths if not (TRACKS / f"{p.stem}.parquet").exists()]
        if todo:
            log(f"{len(todo)} shots to process")
            with ProcessPoolExecutor(args.workers) as ex:
                for msg in ex.map(process, todo, [args.chunk] * len(todo)):
                    log(msg)
        if not args.loop or (ROOT / "STOP").exists():
            break
        time.sleep(60)
    log("done")


if __name__ == "__main__":
    main()

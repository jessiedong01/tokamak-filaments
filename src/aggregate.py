"""
Combine per-shot tracks into the analysis tables.

Steps
  1. camera configurations: shots grouped by (campaign, view, ROI top/left); one pixel
     scale per configuration from a pooled Theil-Sen fit of edge column on EFIT R_LCFS,
     with a per-shot intercept (plasma position differs shot to shot, scale does not)
  2. shot table: Ip, B_t, q95, R0, R_LCFS, line-averaged density, Greenwald fraction,
     NBI power, edge T_e (Thomson at the LCFS when available), ELM count, scale
  3. track table in physical units: v_r (m/s) = scale * vx_px * fps, delta_r (m) = scale * fwhm_px,
     with selection flags (near edge, not near an ELM, good fit)
  4. per-shot filament statistics (medians and counts) for the inter-ELM near-edge set

Per-shot quantities that need the frames (calibration samples, ELM times, frame times,
EFIT/density/Thomson medians) are computed once, in parallel, into results/cache/<shot>.npz
and rebuilt only when the shot's npz is newer than its cache entry.

Outputs: results/configs.parquet, results/shots.parquet, results/tracks_all.parquet
Usage:   python src/aggregate.py [--workers 4]
"""
import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import theilslopes

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
CACHE = RES / "cache"
CACHE_VERSION = 3           # bump when build_cache changes; stale entries are rebuilt
NEAR = (-10.0, 40.0)        # px window around the edge used for the physics set
MIN_R_RANGE_CM = 2.0        # a shot contributes to the scale fit only if its LCFS moved this much
MIN_CAL_POINTS = 8


def load_info():
    rows = []
    for p in sorted((RES / "shotinfo").glob("*.json")):
        d = json.loads(p.read_text()); a = d["attrs"]
        rows.append({"shot": d["shot"], "fps": d["fps"], "sol_sign": d["sol_sign"], "edge_px": d["edge_px_median"],
                     "noise_sigma": d["noise_sigma"], "n_tracks": d["n_tracks"], "n_elms": d["n_elms"],
                     "n_frames": d["n_frames"], "t_start": d["t_start"], "t_end": d["t_end"],
                     "view": a.get("view", ""), "left": a.get("left"), "top": a.get("top"), "exposure": a.get("exposure"),
                     "date_time": a.get("date_time"),
                     "calib": np.array(d["calib_samples"]).reshape(-1, 3) if d["calib_samples"] else np.zeros((0, 3))})
    return pd.DataFrame(rows)



def build_cache(args) -> str:
    """Everything aggregate needs from one shot's npz, computed once."""
    shot, sol_sign, t_flat, t_start, t_end = args
    src = ROOT / "data" / "shots" / f"{shot}.npz"; out = CACHE / f"{shot}.npz"
    if out.exists() and out.stat().st_mtime >= src.stat().st_mtime:
        try:
            if json.loads(str(np.load(out)["scalars"])).get("cache_version") == CACHE_VERSION:
                return f"{shot}: cached"
        except Exception:
            pass
    from calibrate import edge_samples
    z = np.load(src)
    calib = edge_samples(z, sol_sign, t_flat)
    t = z["time"].astype(float)
    elms_da, elms_cam = dalpha_elms(z), camera_elms(z)
    efit = json.loads(str(z["efit"])); te = np.asarray(efit["time"], float); w = (te >= t_start) & (te <= t_end)
    def med(k):
        v = efit.get(k); return float(np.nanmedian(np.asarray(v, float)[w])) if v is not None and w.any() else np.nan
    sc = {"r_lcfs": med("r_out_mid"), "q95": med("q_95"), "r_mag": med("magnetic_axis_r"), "r_geo": med("geom_axis_rc"),
          "a_minor": med("minor_radius"), "bvac_rmag": med("bvac_rmag"),
          "ip_efit": med("plasma_current_c") if "plasma_current_c" in efit else med("plasma_current"),
          "ne_bar": np.nan, "n_gw": np.nan, "ne_line_int": np.nan, "cache_version": CACHE_VERSION}
    if "esm" in z.files:
        e = json.loads(str(z["esm"]))
        if "time" in e and "ne_bar" in e:
            tt = np.asarray(e["time"], float); ww = (tt >= t_start) & (tt <= t_end)
            if ww.any():
                sc["ne_bar"] = float(np.nanmedian(np.asarray(e["ne_bar"], float)[ww]))
                if "n_greenwald" in e:
                    sc["n_gw"] = float(np.nanmedian(np.asarray(e["n_greenwald"], float)[ww]))
    if "ane" in z.files:
        a = json.loads(str(z["ane"]))
        if "time" in a and "density" in a:
            tt = np.asarray(a["time"], float); ww = (tt >= t_start) & (tt <= t_end)
            if ww.any():
                sc["ne_line_int"] = float(np.nanmedian(np.asarray(a["density"], float)[ww]))   # m^-2
    sc["te_lcfs"], sc["te_source"], sc["te_n"] = edge_te(z, t_start, t_end, sc["r_lcfs"]) if np.isfinite(sc["r_lcfs"]) else (np.nan, None, 0)
    CACHE.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(tmp, calib=calib, time=t, elms_da=elms_da, elms_cam=elms_cam, scalars=json.dumps(sc))
    tmp.rename(out)
    return f"{shot}: built"


def load_cache(shot: int):
    z = np.load(CACHE / f"{shot}.npz")
    return {"calib": z["calib"], "time": z["time"], "elms_da": z["elms_da"], "elms_cam": z["elms_cam"], "scalars": json.loads(str(z["scalars"]))}


def _ts_boot(R: np.ndarray, X: np.ndarray, groups: list, n_boot: int = 2000, seed: int = 0):
    """Theil-Sen slope of X on R (both already centred per group) with a bootstrap over shots within groups."""
    slope = theilslopes(X, R)[0]
    rng = np.random.default_rng(seed); boots = []
    for _ in range(n_boot):
        RR, XX = [], []
        for rv, xv in groups:
            idx = rng.integers(0, len(rv), len(rv)); rr, xx = rv[idx], xv[idx]
            if rr.max() - rr.min() < 0.005:
                continue
            RR.append(rr - rr.mean()); XX.append(xx - xx.mean())
        if RR:
            boots.append(theilslopes(np.concatenate(XX), np.concatenate(RR))[0])
    lo, hi = np.percentile(boots, [2.5, 97.5]) if boots else (np.nan, np.nan)
    return float(slope), float(lo), float(hi)


def _scale_row(slope, lo, hi, sign):
    return {"slope_px_per_m": slope, "slope_lo": lo, "slope_hi": hi, "sign_ok": bool(np.sign(slope) == sign),
            "mm_per_px": 1000 / abs(slope) if slope else np.nan,
            "mm_per_px_lo": 1000 / max(abs(lo), abs(hi)) if np.isfinite(lo) else np.nan,
            "mm_per_px_hi": 1000 / max(min(abs(lo), abs(hi)), 1e-9) if np.isfinite(lo) else np.nan}


CFG_MIN_SHOTS = 8          # a camera configuration gets its own scale only with this many shots,
CFG_MIN_R_RANGE_M = 0.03   # this much spread in R_LCFS,
CFG_MIN_RHO = 0.5          # and at least this |Spearman| between limb column and R_LCFS


def fit_configs(info: pd.DataFrame, meta: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """Pixel scale from a between-shot regression of the limb-brightening peak on the EFIT
    outboard LCFS radius.

    Each shot contributes one point: the median peak column over its flat top, converted to
    sensor coordinates (column + ROI left), and the median R_LCFS over the same times. A camera
    configuration is (campaign, view string, ROI left offset): a change in any of these can mean
    a change of lens, pointing, or sensor region. Configurations with enough shots, enough spread
    in R_LCFS, and a clear correlation get their own Theil-Sen slope; the others use the campaign
    slope, fitted on all configurations of the campaign after centring within each (per-config
    intercepts). 95% intervals come from 2,000 bootstrap resamples of shots.
    """
    from scipy.stats import spearmanr
    df = info.merge(meta[["shot", "campaign", "plasma_flat_top_start_time", "plasma_flat_top_end_time"]], on="shot", how="left")
    df["config"] = df.campaign.astype(str) + "|" + df.view.astype(str).str.strip() + "|" + df.left.astype(str)
    offsets, pts = {}, []
    for _, r in df.iterrows():
        c = load_cache(int(r.shot))["calib"]
        offsets[int(r.shot)] = float(np.nanmedian(c[:, 3] - c[:, 2])) if len(c) else np.nan
        if len(c) >= 3 and np.isfinite(c[:, 3]).all():
            r_range = float(c[:, 1].max() - c[:, 1].min())
            within = float(theilslopes(c[:, 3], c[:, 1])[0]) if len(c) >= 6 and r_range >= 0.02 else np.nan
            pts.append({"shot": int(r.shot), "campaign": r.campaign, "config": r.config, "view": str(r.view).strip(), "left": int(r.left), "sol_sign": int(r.sol_sign),
                        "R": float(np.median(c[:, 1])), "x_peak_sensor": float(np.median(c[:, 3])) + float(r.left), "n_efit": len(c),
                        "r_range_m": r_range, "within_slope_px_per_m": within})
    pts = pd.DataFrame(pts)
    out, use_own = [], set()
    for camp, g in pts.groupby("campaign"):
        sign = int(g.sol_sign.mode().iloc[0])
        # campaign pool: centre within configurations
        groups = [(gg.R.values, gg.x_peak_sensor.values) for _, gg in g.groupby("config") if len(gg) >= 2 and gg.R.max() - gg.R.min() >= 0.01]
        if groups:
            R = np.concatenate([rv - rv.mean() for rv, _ in groups]); X = np.concatenate([xv - xv.mean() for _, xv in groups])
            slope, lo, hi = _ts_boot(R, X, groups)
            out.append({"config": camp, "campaign": camp, "kind": "campaign", "view": "all", "left": -1, "n_shots": len(g), "n_shots_cal": sum(len(rv) for rv, _ in groups),
                        "n_views": len(groups), "r_range_m": float(g.R.max() - g.R.min()), "spearman": float(spearmanr(R, X).correlation), "used": True,
                        **_scale_row(slope, lo, hi, sign)})
        else:
            out.append({"config": camp, "campaign": camp, "kind": "campaign", "view": "all", "left": -1, "n_shots": len(g), "n_shots_cal": 0, "n_views": 0, "used": True, "mm_per_px": np.nan})
        for cfg, gg in g.groupby("config"):
            row = {"config": cfg, "campaign": camp, "kind": "config", "view": gg.view.iloc[0], "left": int(gg.left.iloc[0]), "n_shots": len(gg), "n_shots_cal": len(gg),
                   "n_views": 1, "r_range_m": float(gg.R.max() - gg.R.min()), "used": False}
            if len(gg) >= CFG_MIN_SHOTS and row["r_range_m"] >= CFG_MIN_R_RANGE_M:
                R, X = gg.R.values - gg.R.mean(), gg.x_peak_sensor.values - gg.x_peak_sensor.mean()
                slope, lo, hi = _ts_boot(R, X, [(gg.R.values, gg.x_peak_sensor.values)])
                row.update(_scale_row(slope, lo, hi, sign)); row["spearman"] = float(spearmanr(R, X).correlation)
                row["used"] = bool(row["sign_ok"] and abs(row["spearman"]) >= CFG_MIN_RHO and np.isfinite(lo) and np.sign(lo) == np.sign(hi))
                if row["used"]:
                    use_own.add(cfg)
            out.append(row)
    cfgs = pd.DataFrame(out)
    cfgs["edge_def"] = "peak"
    pts["own_scale"] = pts.config.isin(use_own)
    pts.to_parquet(RES / "calib_points.parquet")
    cmap = df[["shot", "campaign", "config"]].copy()
    cmap["config"] = np.where(cmap.config.isin(use_own), cmap.config, cmap.campaign)
    cmap = cmap[["shot", "config"]]
    return cfgs, cmap, offsets


def edge_te(z, t0, t1, r_lcfs):
    """Electron temperature (eV) at the LCFS from Thomson scattering, median over the window.
    Uses the edge system (aye) if present, else the core system (ayc). Interpolates each profile
    in R at R_LCFS of that time; profiles with fewer than 3 valid points within 10 cm are skipped."""
    for name, rkey in (("aye", "r"), ("ayc", "radius")):
        if name not in z.files:
            continue
        d = json.loads(str(z[name])); t = np.asarray(d["time"], float); te = np.asarray(d["te"], float); r = np.asarray(d[rkey], float)
        if te.ndim != 2:
            continue
        vals = []
        for i in np.flatnonzero((t >= t0) & (t <= t1)):
            ri = r[i] if r.ndim == 2 else r
            prof = te[i]; ok = np.isfinite(prof) & (prof > 1) & np.isfinite(ri)
            near = ok & (np.abs(ri - r_lcfs) < 0.10)
            if near.sum() < 3:
                continue
            order = np.argsort(ri[near]); vals.append(float(np.interp(r_lcfs, ri[near][order], prof[near][order])))
        if vals:
            return float(np.median(vals)), name, len(vals)
    return np.nan, None, 0


def shot_table(info: pd.DataFrame, meta: pd.DataFrame, cfg_map: pd.DataFrame, cfgs: pd.DataFrame, offsets: dict) -> pd.DataFrame:
    rows = []
    for _, r in info.iterrows():
        c = load_cache(int(r.shot))
        row = {"shot": int(r.shot), **c["scalars"]}
        row["peak_minus_half_px"] = offsets.get(int(r.shot), np.nan)
        row["n_elms_da"] = int(len(c["elms_da"])); row["n_elms_cam"] = int(len(c["elms_cam"]))
        rows.append(row)
    st = pd.DataFrame(rows).merge(info.drop(columns=["calib"]), on="shot").merge(cfg_map, on="shot").merge(
        cfgs[["config", "mm_per_px", "mm_per_px_lo", "mm_per_px_hi", "sign_ok", "n_shots_cal", "edge_def", "spearman"]], on="config", how="left").merge(
        meta[["shot", "campaign", "heating", "scenario", "plasma_shape", "plasma_max_current", "thomson_line_avg_density_co2",
              "thomson_greenwald_density_limit", "nbi_power_max_current", "generic_toroidal_max_current",
              "plasma_flat_top_start_time", "plasma_flat_top_end_time"]], on="shot", how="left")
    st["n_elms"] = st.n_elms_da          # D-alpha count with the time-based window, not the process.py count
    return st


def _spikes(x: np.ndarray, t: np.ndarray, window_s: float, k: float = 6.0, floor: float = 0.0) -> np.ndarray:
    """Times of spikes in x(t): residual after a running median of `window_s` seconds exceeds
    max(k robust standard deviations, floor). Each connected spike contributes its centre of mass."""
    from scipy import ndimage as ndi
    ok = np.isfinite(x)
    if ok.sum() < 100 or ok.mean() < 0.5:
        return np.array([])
    x, t = x[ok], t[ok]
    size = int(round(window_s / np.median(np.diff(t)))) | 1
    base = ndi.median_filter(x, size=max(size, 3))
    resid = x - base
    mad = 1.4826 * np.median(np.abs(resid - np.median(resid)))
    lab, n = ndi.label(resid > max(k * mad, floor))
    if n == 0:
        return np.array([])
    return np.array([t[int(c[0])] for c in ndi.center_of_mass(resid, lab, range(1, n + 1))])


def _quantum(x: np.ndarray) -> float:
    """Digitizer step: the smallest positive difference between distinct sample values."""
    u = np.unique(x[np.isfinite(x)])
    d = np.diff(u)
    return float(d[d > 0].min()) if (d > 0).any() else 0.0


ELM_WINDOW_S = 0.005   # running-median window for the spike detectors
ELM_PAD_S = 0.001      # tracks within this of a spike are flagged


def dalpha_elms(z) -> np.ndarray:
    """ELM times from the midplane D-alpha monitor (sampling differs by campaign, so the
    running-median window is set in seconds)."""
    if "dalpha" not in z.files:
        return np.array([])
    da = z["dalpha"].astype(float)
    # the monitor is coarsely digitized (tens of distinct values in some campaigns), so the MAD can be
    # zero; the threshold is floored at six digitizer steps and at half the inter-ELM level
    floor = max(6 * _quantum(da), 0.5 * float(np.nanmedian(da)))
    return _spikes(da, z["dalpha_t"].astype(float), ELM_WINDOW_S, floor=floor)


def camera_elms(z) -> np.ndarray:
    """ELM proxy from the camera itself: spikes of the frame-mean brightness."""
    fr, t = z["frames"], z["time"].astype(float)
    return _spikes(fr.reshape(len(fr), -1).mean(axis=1).astype(float), t, ELM_WINDOW_S)


def track_table(st: pd.DataFrame) -> pd.DataFrame:
    parts = []
    for _, r in st.iterrows():
        p = RES / "tracks" / f"{int(r.shot)}.parquet"
        if not p.exists():
            continue
        t = pd.read_parquet(p)
        if not len(t):
            continue
        c = load_cache(int(r.shot)); de, ce = c["elms_da"], c["elms_cam"]
        t["near_da_elm"] = np.array([bool(len(de)) and np.min(np.abs(de - t0)) < ELM_PAD_S for t0 in t.t0], bool) if len(t) else np.zeros(0, bool)
        t["near_cam_elm"] = np.array([bool(len(ce)) and np.min(np.abs(ce - t0)) < ELM_PAD_S for t0 in t.t0], bool) if len(t) else np.zeros(0, bool)
        t["near_elm"] = t.near_da_elm | t.near_cam_elm      # replaces the flag stored by process.py
        scale = r.mm_per_px / 1000.0
        if r.edge_def == "peak" and np.isfinite(r.peak_minus_half_px):   # tracks store distance from the half-level edge
            # sol_dist = sol_sign*(x - x_half); relative to the peak: sol_sign*(x - x_peak) = sol_dist - sol_sign*(x_peak - x_half)
            t["sol_dist_px"] = t.sol_dist_px - r.sol_sign * r.peak_minus_half_px
        t["flat_top"] = (t.t0 >= r.plasma_flat_top_start_time) & (t.t0 <= r.plasma_flat_top_end_time)
        dts = np.diff(c["time"])                      # recordings may hold slow frames outside the 100 kHz burst
        t["fast"] = np.abs(dts[np.clip(t.frame0.values, 0, len(dts) - 1)] - 1.0 / r.fps) < 2e-6
        t["v_r"] = t.vx_px * r.fps * scale
        t["v_r_lk"] = t.lk_vx_px * r.fps * scale
        t["delta_r"] = t.fwhm_px * scale
        t["sol_dist_m"] = t.sol_dist_px * scale
        t["near_edge"] = (t.sol_dist_px > NEAR[0]) & (t.sol_dist_px < NEAR[1])
        t["good_fit"] = (t.vx_err < 0.5) & (t.n >= 5) & np.isfinite(t.fwhm_px)
        t["select"] = t.near_edge & t.good_fit & ~t.near_elm & t.flat_top & t.fast
        parts.append(t)
    return pd.concat(parts, ignore_index=True)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--workers", type=int, default=4); args = ap.parse_args()
    meta = pd.read_parquet(ROOT / "data" / "candidates.parquet").drop_duplicates("shot")
    info = load_info()
    dropped = info[~info.shot.isin(meta.shot)]
    if len(dropped):
        print(f"{len(dropped)} processed shots are not in the current catalogue and are skipped: {dropped.shot.tolist()}")
    info = info[info.shot.isin(meta.shot)].reset_index(drop=True)
    m = info.merge(meta[["shot", "plasma_flat_top_start_time", "plasma_flat_top_end_time"]], on="shot")
    jobs = [(int(r.shot), int(r.sol_sign), (float(r.plasma_flat_top_start_time), float(r.plasma_flat_top_end_time)), float(r.t_start), float(r.t_end)) for _, r in m.iterrows()]
    with ProcessPoolExecutor(args.workers) as ex:
        msgs = list(ex.map(build_cache, jobs))
    print(f"cache: {sum(m.endswith('built') for m in msgs)} built, {sum(m.endswith('cached') for m in msgs)} reused")
    cfgs, cfg_map, offsets = fit_configs(info, meta)
    st = shot_table(info, meta, cfg_map, cfgs, offsets)
    tr = track_table(st)
    sel = tr[tr.select]
    agg = sel.groupby("shot").agg(n_sel=("v_r", "size"), v_r_med=("v_r", "median"), v_r_q25=("v_r", lambda x: x.quantile(.25)),
                                  v_r_q75=("v_r", lambda x: x.quantile(.75)), delta_med=("delta_r", "median"),
                                  outward_frac=("v_r", lambda x: float((x > 0).mean())), amp_med=("amp", "median"))
    st = st.merge(agg, on="shot", how="left")
    cfgs.to_parquet(RES / "configs.parquet"); st.to_parquet(RES / "shots.parquet"); tr.to_parquet(RES / "tracks_all.parquet")
    pd.set_option("display.width", 250)
    print(cfgs[["config", "n_shots", "n_shots_cal", "n_views", "spearman", "mm_per_px", "mm_per_px_lo", "mm_per_px_hi", "sign_ok"]].round(2).to_string(index=False))
    print(f"\n{len(st)} shots, {len(tr)} tracks, {len(sel)} selected (near edge, inter-ELM, good fit)")
    print(st[["shot", "campaign", "exposure", "mm_per_px", "n_elms", "n_sel", "v_r_med", "delta_med", "outward_frac", "te_lcfs", "te_source"]].to_string(index=False))


if __name__ == "__main__":
    main()

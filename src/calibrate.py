"""
Pixel scale at the plasma edge from the EFIT boundary.

The plasma edge column in the strip follows the outboard LCFS radius from EFIT as the
plasma moves during a shot. A linear fit of edge column against R_LCFS gives the scale
(mm per pixel) along the strip at the tangency plane, independent of any CAD model.
The fit is pooled over all shots that share a camera configuration (campaign, view, ROI).

Usage (per shot, returns the paired samples):
  from calibrate import edge_vs_lcfs
"""
import json

import numpy as np
from scipy import ndimage as ndi
from scipy.stats import theilslopes


def smoothed_edge(frames: np.ndarray, time: np.ndarray, sol_sign: int, window_s: float = 0.002) -> np.ndarray:
    """Edge column per frame (half-level crossing) from the row-averaged profile smoothed over `window_s` seconds."""
    from filaments import _half_level_edge
    prof = frames.astype(np.float32).mean(axis=1)
    n = max(3, int(round(window_s / np.median(np.diff(time)))))
    prof = ndi.uniform_filter1d(prof, size=n, axis=0, mode="nearest")
    prof = ndi.gaussian_filter1d(prof, sigma=2.0, axis=1)
    return _half_level_edge(prof, sol_sign)


def limb_peak(prof_smoothed: np.ndarray, sol_sign: int, x_half: np.ndarray, search: int = 40) -> np.ndarray:
    """Column of the limb-brightening peak: the maximum of the smoothed profile within
    `search` px inboard of the half-level crossing. Sub-pixel by parabolic interpolation."""
    n, w = prof_smoothed.shape
    out = np.full(n, np.nan)
    for i in range(n):
        x0 = int(round(x_half[i]))
        if sol_sign == -1:
            lo, hi = max(0, x0), min(w, x0 + search)
        else:
            lo, hi = max(0, x0 - search), min(w, x0 + 1)
        if hi - lo < 3:
            continue
        seg = prof_smoothed[i, lo:hi]; k = int(np.argmax(seg)); j = lo + k
        if 1 <= j <= w - 2:
            g0, g1, g2 = prof_smoothed[i, j - 1], prof_smoothed[i, j], prof_smoothed[i, j + 1]
            den = g0 - 2 * g1 + g2
            j = j + (0.5 * (g0 - g2) / den if abs(den) > 1e-6 else 0.0)
        out[i] = j
    return out


def edge_samples(z, sol_sign: int, t_flat: tuple, window_s: float = 0.002, r_tol: float = 0.08):
    """At each EFIT time inside the plasma flat top: (t, R_lcfs, x_half, x_peak).
    Samples whose R_lcfs is more than r_tol from the flat-top median are dropped
    (unconverged or terminating equilibria)."""
    from filaments import _half_level_edge
    frames, t = z["frames"], z["time"].astype(float)
    efit = json.loads(str(z["efit"]))
    te = np.asarray(efit["time"], float); r = np.asarray(efit.get("r_out_mid", efit.get("rbdry")), float)
    prof = frames.astype(np.float32).mean(axis=1)
    n = max(3, int(round(window_s / np.median(np.diff(t)))))
    prof = ndi.uniform_filter1d(prof, size=n, axis=0, mode="nearest")
    prof = ndi.gaussian_filter1d(prof, sigma=2.0, axis=1)
    x_half = _half_level_edge(prof, sol_sign); x_peak = limb_peak(prof, sol_sign, x_half)
    inwin = (te >= max(t[0] + 0.001, t_flat[0])) & (te <= min(t[-1] - 0.001, t_flat[1])) & np.isfinite(r)
    if inwin.sum() < 3:
        return np.zeros((0, 4))
    rmed = np.nanmedian(r[inwin]); ok = inwin & (np.abs(r - rmed) < r_tol) & (r > 1.15)
    out = []
    for ti, ri in zip(te[ok], r[ok]):
        k = int(np.searchsorted(t, ti)); out.append((ti, ri, float(x_half[k]), float(x_peak[k])))
    return np.array(out)


def pooled_slope(samples_by_shot: list, col: int, sol_sign: int) -> dict:
    """Pooled Theil-Sen slope of edge column (col 2 = half level, 3 = limb peak) on R_lcfs,
    after centring each shot on its own means (per-shot intercepts)."""
    from scipy.stats import spearmanr
    R, X, used = [], [], 0
    for c in samples_by_shot:
        if len(c) < 6 or (c[:, 1].max() - c[:, 1].min()) * 100 < 1.5 or not np.isfinite(c[:, col]).all():
            continue
        R.append(c[:, 1] - c[:, 1].mean()); X.append(c[:, col] - c[:, col].mean()); used += 1
    if used == 0:
        return {"n_shots_cal": 0, "mm_per_px": np.nan}
    R, X = np.concatenate(R), np.concatenate(X)
    slope, _, lo, hi = theilslopes(X, R); rho = spearmanr(R, X).correlation
    return {"n_shots_cal": used, "n_points": int(len(R)), "slope_px_per_m": float(slope), "slope_lo": float(lo), "slope_hi": float(hi),
            "sign_ok": bool(np.sign(slope) == sol_sign), "spearman": float(rho),
            "mm_per_px": 1000 / abs(slope) if slope else np.nan,
            "mm_per_px_lo": 1000 / max(abs(lo), abs(hi)), "mm_per_px_hi": 1000 / max(min(abs(lo), abs(hi)), 1e-9),
            "r_range_cm_median": float(np.median([(c[:, 1].max() - c[:, 1].min()) * 100 for c in samples_by_shot if len(c)]))}

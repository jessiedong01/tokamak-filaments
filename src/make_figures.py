"""
Paper figures from results/. All data-driven; nothing typed by hand.

  fig_overview.pdf     strip frames, background-subtracted frames with detections, x-t with tracks
  fig_calibration.pdf  limb-peak column vs EFIT R_LCFS per camera view, with the campaign slope
  fig_distributions.pdf per-shot distributions of v_r and delta_r (all selected tracks, by campaign)
  fig_vd.pdf           v_r vs delta_r, each shot normalised by its medians, with regime slopes
  fig_params.pdf       per-shot median v_r and delta_r vs plasma current and Greenwald fraction
  fig_exposure.pdf     v-delta slope by exposure time; width vs exposure
  fig_regimes.pdf      normalised v/v* vs a/a* with the two-region model scalings
  fig_elm.pdf          D-alpha trace with flagged ELMs and camera brightness proxy for one shot

Usage: python src/make_figures.py
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

sys.path.insert(0, str(Path(__file__).resolve().parent))
from filaments import detect, edge_column, fluctuation, link_tracks

ROOT = Path(__file__).resolve().parent.parent
RES, FIGS = ROOT / "results", ROOT / "paper" / "figs"
C1, C2, C3, C4, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#8a8985"
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Palatino", "TeX Gyre Pagella", "DejaVu Serif"],
    "font.size": 8.5, "axes.titlesize": 9, "axes.labelsize": 8.5, "legend.fontsize": 7.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "axes.linewidth": 0.6, "axes.edgecolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "pdf.fonttype": 42, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})
W = 5.5
CAMP_COLOR = {"M6": C1, "M7": C2, "M8": C3, "M9": C4}


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf"); fig.savefig(FIGS / f"{name}.png", dpi=170); plt.close(fig); print("saved", name)


def fig_overview(shot: int):
    z = np.load(ROOT / "data" / "shots" / f"{shot}.npz"); fr, t = z["frames"], z["time"]
    i0 = len(fr) // 2; sub = fr[i0:i0 + 1500]
    d = fluctuation(sub); edge, sgn = edge_column(sub); dets = detect(d, edge, sgn); tracks = link_tracks(dets)
    fig = plt.figure(figsize=(W, 3.9)); gs = fig.add_gridspec(4, 2, height_ratios=[1, 1, 1, 2.6], hspace=0.35, wspace=0.08)
    for k in range(3):
        j = 700 + 3 * k
        ax = fig.add_subplot(gs[k, 0]); ax.imshow(sub[j], cmap="gray", vmin=0, vmax=np.percentile(sub, 99.7)); ax.set_xticks([]); ax.set_yticks([])
        if k == 0: ax.set_title("Raw frames (10 µs apart)", fontsize=8)
        ax.set_ylabel(f"+{30*k} µs", fontsize=7)
        ax2 = fig.add_subplot(gs[k, 1]); ax2.imshow(d[j], cmap="Reds", vmin=0, vmax=np.percentile(d, 99.5)); ax2.axvline(edge[j], color=INK, lw=0.6)
        for det in dets:
            if det.frame == j: ax2.plot(det.x, det.y, "+", color=C1, ms=6, mew=0.8)
        ax2.set_xticks([]); ax2.set_yticks([])
        if k == 0: ax2.set_title("Background-subtracted, detections (+), edge (line)", fontsize=8)
    ax = fig.add_subplot(gs[3, :]); a, b = 600, 1000
    ax.imshow(d[a:b, 24, :].T, aspect="auto", cmap="Reds", vmin=0, vmax=np.percentile(d, 99.5), extent=[0, (b - a) * 10, 256, 0])
    for tr in tracks:
        if len(tr) >= 5 and a <= tr[0].frame < b: ax.plot([(q.frame - a) * 10 for q in tr], [q.x for q in tr], "-", color=C1, lw=0.9)
    ax.axhline(np.median(edge), color=INK, lw=0.6)
    ax.set_ylim(np.median(edge) + 70, max(0, np.median(edge) - 60)); ax.set_xlabel("time (µs)"); ax.set_ylabel("column (px)")
    ax.set_title("Central row against time; tracked filaments in blue (outward is toward the top)", fontsize=8)
    save(fig, "fig_overview")


def fig_calibration():
    pts = pd.read_parquet(RES / "calib_points.parquet"); cf = pd.read_parquet(RES / "configs.parquet").set_index("config")
    camps = sorted(pts.campaign.unique())
    fig, axes = plt.subplots(2, 2, figsize=(W, 4.4), squeeze=False); axes = np.array([axes.ravel()])
    for ax, camp in zip(axes[0], camps):
        g = pts[pts.campaign == camp]; camp_slope = cf.loc[camp, "slope_px_per_m"] if camp in cf.index else np.nan
        for k, (cfg, gg) in enumerate(g.groupby("config")):
            if len(gg) < 2: continue
            col = [C1, C2, C3, C4, GRAY, INK2, "#b15bd6"][k % 7]
            own = cfg in cf.index and bool(cf.loc[cfg, "used"]) and cf.loc[cfg, "kind"] == "config"
            slope = cf.loc[cfg, "slope_px_per_m"] if own else camp_slope
            short = gg.view.iloc[0].replace("HM10, image normal, ", "").replace("HM10, image normal", "image normal").replace("photron HM10", "HM10").replace("Dalpha filter 10 nm", "D-alpha").replace("Dalpha filter", "D-alpha").strip()
            label = f"{short[:20]} @{int(gg.left.iloc[0])}" + (" (own)" if own else "") if len(gg) >= 3 else None
            ax.scatter(gg.R * 100, gg.x_peak_sensor, s=11, color=col, edgecolors="white", linewidths=0.4, label=label)
            if np.isfinite(slope) and gg.R.max() - gg.R.min() >= 0.01:
                rr = np.array([gg.R.min(), gg.R.max()]); ax.plot(rr * 100, gg.x_peak_sensor.mean() + slope * (rr - gg.R.mean()), color=col, lw=0.8, ls="-" if own else "--")
        ax.set_title(f"{camp}: pooled {cf.loc[camp, 'mm_per_px']:.1f} mm/px" if camp in cf.index else camp)
        ax.set_xlabel("EFIT outboard LCFS radius (cm)"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=5.5, loc="best")
    axes[0][0].set_ylabel("limb-peak column (sensor px)"); axes[0][2].set_ylabel("limb-peak column (sensor px)")
    fig.tight_layout(); save(fig, "fig_calibration")


def fig_distributions(ss, sel):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1))
    for camp, g in sel.merge(ss[["shot", "campaign"]], on="shot").groupby("campaign"):
        axes[0].hist(g.v_r / 1e3, bins=np.linspace(0, 3, 61), histtype="step", lw=1.3, color=CAMP_COLOR.get(camp, GRAY), label=f"{camp} ({g.shot.nunique()} shots)", density=True)
        axes[1].hist(g.delta_r * 100, bins=np.linspace(0, 12, 49), histtype="step", lw=1.3, color=CAMP_COLOR.get(camp, GRAY), density=True)
    axes[0].set_xlabel("radial velocity $v_r$ (km/s)"); axes[0].set_ylabel("density"); axes[0].legend(frameon=False)
    axes[1].set_xlabel("radial width $\\delta_r$, FWHM (cm)")
    for ax in axes: ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig_distributions")


def fig_vd(ss, sel, slope):
    m = sel.merge(ss[["shot", "v_med", "d_med"]], on="shot")
    x, y = (m.delta_r / m.d_med).values, (m.v_r / m.v_med).values
    fig, ax = plt.subplots(figsize=(W * 0.52, 2.6))
    ax.hexbin(np.log10(x), np.log10(y), gridsize=45, cmap="Blues", mincnt=1, linewidths=0.1)
    xx = np.linspace(-0.7, 0.6, 10)
    ax.plot(xx, 0.5 * xx, color=C3, lw=1.2, ls="--", label="inertial, $v \\propto \\delta^{1/2}$")
    ax.plot(xx, -2 * xx, color=C2, lw=1.2, ls="--", label="sheath-connected, $v \\propto \\delta^{-2}$")
    ax.plot(xx, slope * xx, color=INK, lw=1.4, label=f"fit, $v \\propto \\delta^{{{slope:+.2f}}}$")
    ax.set_xlim(-0.7, 0.6); ax.set_ylim(-1.2, 1.0)
    ax.set_xlabel("$\\log_{10}(\\delta_r / \\mathrm{median}_{shot})$"); ax.set_ylabel("$\\log_{10}(v_r / \\mathrm{median}_{shot})$")
    ax.legend(frameon=False, loc="lower right"); ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
    save(fig, "fig_vd")


def fig_params(ss):
    L = ss[~ss.h_mode_like]; H = ss[ss.h_mode_like]
    fig, axes = plt.subplots(2, 2, figsize=(W, 3.9), sharex="col")
    for col, (xc, xl) in enumerate((("ip_ka", "plasma current (kA)"), ("f_gw", "Greenwald fraction"))):
        for row, (yc, yl, f) in enumerate((("v_med", "median $v_r$ (m/s)", 1), ("d_med", "median $\\delta_r$ (cm)", 100))):
            ax = axes[row, col]
            for d, mk, lab, c in ((L, "o", "L-mode / no ELMs", C1), (H, "s", "ELMy (inter-ELM)", C2)):
                ax.errorbar(d[xc], d[yc] * f, yerr=[(d[yc] - d[yc.replace("med", "q25")]) * f, (d[yc.replace("med", "q75")] - d[yc]) * f],
                            fmt=mk, ms=3.2, color=c, ecolor=c, elinewidth=0.4, alpha=0.8, label=lab, capsize=0)
            ax.grid(color=GRID, lw=0.5); ax.set_axisbelow(True)
            if col == 0: ax.set_ylabel(yl)
            if row == 1: ax.set_xlabel(xl)
    axes[0, 0].legend(frameon=False, loc="upper left")
    fig.tight_layout(); save(fig, "fig_params")


def fig_exposure(ss):
    fig, axes = plt.subplots(1, 2, figsize=(W, 2.1))
    g = ss.groupby("exposure")
    xs = [e for e in sorted(ss.exposure.dropna().unique()) if (ss.exposure == e).sum() >= 3]   # groups with too few shots are left out
    axes[0].boxplot([ss.vd_slope[ss.exposure == e].dropna() for e in xs], positions=range(len(xs)), widths=0.5, showfliers=False)
    axes[0].axhline(0.5, color=C3, ls="--", lw=1); axes[0].axhline(0, color=INK2, lw=0.6)
    axes[0].set_xticks(range(len(xs)), [f"{e:.0f} µs\n(n={int((ss.exposure == e).sum())})" for e in xs]); axes[0].set_ylabel("per-shot $v$–$\\delta$ slope")
    axes[1].boxplot([100 * ss.d_med[ss.exposure == e] for e in xs], positions=range(len(xs)), widths=0.5, showfliers=False)
    axes[1].set_xticks(range(len(xs)), [f"{e:.0f} µs" for e in xs]); axes[1].set_ylabel("median $\\delta_r$ (cm)")
    for ax in axes: ax.grid(axis="y", color=GRID, lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); save(fig, "fig_exposure")


def fig_regimes(ss):
    ok = np.isfinite(ss.a_hat) & np.isfinite(ss.v_hat)
    fig, ax = plt.subplots(figsize=(W * 0.52, 2.6))
    for meas, mk, lab in ((True, "o", "$T_e$ measured"), (False, "^", "$T_e$ assumed")):
        g = ss[ok & (ss.te_measured == meas)]
        ax.scatter(g.a_hat, g.v_hat, s=14, marker=mk, color=C1 if meas else GRAY, edgecolors="white", linewidths=0.4, label=f"{lab} ({len(g)})")
    aa = np.logspace(-0.6, 0.4, 20)
    ax.plot(aa, 0.05 * aa ** 0.5, color=C3, ls="--", lw=1, label="$\\hat v \\propto \\hat a^{1/2}$"); ax.plot(aa, 0.05 * aa ** -2, color=C2, ls="--", lw=1, label="$\\hat v \\propto \\hat a^{-2}$")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlabel("$\\hat a = (\\delta_r/2)/a^*$"); ax.set_ylabel("$\\hat v = v_r / v^*$")
    ax.grid(color=GRID, lw=0.5, which="both"); ax.set_axisbelow(True); ax.legend(frameon=False, fontsize=6.5)
    save(fig, "fig_regimes")


def fig_elm(shot: int):
    z = np.load(ROOT / "data" / "shots" / f"{shot}.npz"); t = z["time"].astype(float)
    sys.path.insert(0, str(ROOT / "src")); from aggregate import dalpha_elms, camera_elms, ELM_PAD_S
    e, ce = dalpha_elms(z), camera_elms(z)
    fig, axes = plt.subplots(2, 1, figsize=(W, 2.6), sharex=True)
    if "dalpha" in z.files:
        da, tt = z["dalpha"].astype(float), z["dalpha_t"].astype(float); s = (tt >= t[0]) & (tt <= t[-1])
        axes[0].plot(tt[s] * 1e3, da[s], lw=0.4, color=INK2); axes[0].plot(e * 1e3, np.interp(e, tt, da), "|", color=C2, ms=9)
    axes[0].set_ylabel("$D_\\alpha$ (a.u.)"); axes[0].set_title(f"shot {shot}: {len(e)} $D_\\alpha$ spikes, {len(ce)} camera spikes", fontsize=8)
    m = z["frames"].reshape(len(t), -1).mean(axis=1); axes[1].plot(t * 1e3, m, lw=0.4, color=INK2); axes[1].plot(ce * 1e3, np.interp(ce, t, m), "|", color=C2, ms=9)
    for ax in axes:
        for x in np.concatenate([e, ce]):
            ax.axvspan((x - ELM_PAD_S) * 1e3, (x + ELM_PAD_S) * 1e3, color=C2, alpha=0.12, lw=0)
    axes[1].set_ylabel("frame mean (counts)"); axes[1].set_xlabel("time (ms)"); axes[1].set_xlim(t[0] * 1e3, t[-1] * 1e3)
    fig.tight_layout(); save(fig, "fig_elm")


def main():
    ss = pd.read_parquet(RES / "shot_stats.parquet"); tr = pd.read_parquet(RES / "tracks_all.parquet")
    sel = tr[tr.select & np.isfinite(tr.v_r) & np.isfinite(tr.delta_r) & (tr.v_r > 0) & tr.shot.isin(ss.shot)]
    import re
    nums = dict(re.findall(r"\\newcommand\{\\(\w+)\}\{([^}]*)\}", (ROOT / "paper" / "numbers.tex").read_text()))
    slope = float(nums["vdSlopePooled"])
    best = ss.sort_values("n", ascending=False).iloc[0]
    fig_overview(15235 if (ss.shot == 15235).any() else int(best.shot)); fig_calibration(); fig_distributions(ss, sel); fig_vd(ss, sel, slope); fig_params(ss); fig_exposure(ss); fig_regimes(ss)
    elmshot = ss[ss.h_mode_like & (ss.n_elms.between(20, 80))].sort_values("n", ascending=False)   # a clean ELMy example
    fig_elm(27305 if (ss.shot == 27305).any() else (int(elmshot.shot.iloc[0]) if len(elmshot) else int(best.shot)))


if __name__ == "__main__":
    main()

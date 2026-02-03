"""
Physics analysis: per-shot filament statistics, scaling fits, and paper numbers.

Inputs: results/shots.parquet, results/tracks_all.parquet, results/configs.parquet
Outputs: results/shot_stats.parquet, paper/numbers.tex, paper/tables/*.tex

Analyses
  1. per-shot distributions of radial velocity v_r and radial width delta_r (selected tracks)
  2. velocity-size relation: Theil-Sen slope of log v_r on log delta_r within each shot (scale-free),
     and pooled across shots after normalising by each shot's medians
  3. dependence of per-shot median v_r and delta_r on plasma current, Greenwald fraction, NBI power,
     each as a Spearman correlation and a log-log slope with bootstrap CIs over shots
  4. two-region model normalisation (a/a*, v/v*) for shots with an edge T_e measurement
  5. exposure-time check: delta_r versus camera exposure at fixed current

Usage:  python src/analysis.py
"""
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, theilslopes

from physics import b_t_at, reference_scales

ROOT = Path(__file__).resolve().parent.parent
RES = ROOT / "results"
PAPER = ROOT / "paper"
MIN_TRACKS = 50          # per shot, to enter the shot-level statistics
TE_DEFAULT = 30.0        # eV at the LCFS when no Thomson measurement exists (stated in the paper)

macros: dict[str, str] = {}


def macro(name, value):
    assert re.fullmatch(r"[A-Za-z]+", name), name
    macros[name] = str(value)


def boot_ci(x, stat=np.median, n=2000, seed=0):
    rng = np.random.default_rng(seed); x = np.asarray(x); x = x[np.isfinite(x)]
    if len(x) < 3:
        return np.nan, np.nan
    b = [stat(x[rng.integers(0, len(x), len(x))]) for _ in range(n)]
    return float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


N_MAX_TS = 20000         # Theil-Sen is O(n^2); pooled fits use a fixed random subsample above this size


def loglog_slope(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y) & (x > 0) & (y > 0)
    if ok.sum() < 8:
        return np.nan, np.nan, np.nan, np.nan
    x, y = x[ok], y[ok]
    if len(x) > N_MAX_TS:
        idx = np.random.default_rng(0).choice(len(x), N_MAX_TS, replace=False); x, y = x[idx], y[idx]
    sl, ic, lo, hi = theilslopes(np.log10(y), np.log10(x))
    return float(sl), float(lo), float(hi), float(spearmanr(x, y).correlation)


def main():
    st = pd.read_parquet(RES / "shots.parquet"); tr = pd.read_parquet(RES / "tracks_all.parquet"); cf = pd.read_parquet(RES / "configs.parquet")
    slow = st[~(np.abs(st.fps - 1e5) < 5e3)]
    macro("nShotsSlowFps", len(slow)); macro("slowFpsKhz", f"{slow.fps.median() / 1e3:.0f}" if len(slow) else "--")
    st = st[np.abs(st.fps - 1e5) < 5e3]            # statistics use the 100 kHz recordings only
    tr = tr[tr.shot.isin(st.shot)]
    sel = tr[tr.select & np.isfinite(tr.v_r) & np.isfinite(tr.delta_r) & (tr.v_r > 0)]
    # effective interferometer chord length: line average = line integral / l_eff, fitted on shots with both
    both = st[np.isfinite(st.ne_bar) & np.isfinite(st.ne_line_int) & (st.ne_bar > 5e18)]
    l_eff = float((both.ne_line_int / both.ne_bar).median()) if len(both) >= 5 else np.nan
    macro("lEffM", f"{l_eff:.2f}"); macro("nLEff", len(both))
    if len(both) >= 5:
        q = (both.ne_line_int / both.ne_bar); macro("lEffQLo", f"{q.quantile(.25):.2f}"); macro("lEffQHi", f"{q.quantile(.75):.2f}")
    # ── per-shot statistics ──
    rows = []
    for shot, g in sel.groupby("shot"):
        if len(g) < MIN_TRACKS:
            continue
        s = st[st.shot == shot].iloc[0]
        sl, lo, hi, rho = loglog_slope(g.delta_r.values, g.v_r.values)
        rows.append({"shot": shot, "n": len(g), "v_med": g.v_r.median(), "v_q25": g.v_r.quantile(.25), "v_q75": g.v_r.quantile(.75),
                     "d_med": g.delta_r.median(), "d_q25": g.delta_r.quantile(.25), "d_q75": g.delta_r.quantile(.75),
                     "vd_slope": sl, "vd_lo": lo, "vd_hi": hi, "vd_rho": rho,
                     "lk_agreement": float(np.corrcoef(g.v_r, g.v_r_lk)[0, 1]) if g.v_r_lk.notna().sum() > 10 else np.nan,
                     "v_lk_med": g.v_r_lk.median(),
                     "outward_all": float((tr[(tr.shot == shot) & tr.near_edge & tr.good_fit & tr.flat_top & tr.fast & ~tr.near_elm].v_r > 0).mean()),
                     "ip_ka": abs(s.ip_efit) / 1e3 if np.isfinite(s.ip_efit) else abs(s.plasma_max_current),
                     "ne_bar": s.ne_bar if np.isfinite(s.get("ne_bar", np.nan)) else (s.thomson_line_avg_density_co2 if np.isfinite(s.thomson_line_avg_density_co2) else s.ne_line_int / l_eff),
                     "ne_source": "esm" if np.isfinite(s.get("ne_bar", np.nan)) else ("meta" if np.isfinite(s.thomson_line_avg_density_co2) else ("ane" if np.isfinite(s.ne_line_int) else "none")),
                     "f_gw": np.nan, "nbi_mw": s.nbi_power_max_current if np.isfinite(s.nbi_power_max_current) else 0.0,
                     "q95": s.q95, "r_lcfs": s.r_lcfs, "r_mag": s.r_mag, "bvac_rmag": s.bvac_rmag, "a_minor": s.a_minor,
                     "te_lcfs": s.te_lcfs, "te_source": s.te_source, "exposure": s.exposure, "campaign": s.campaign,
                     "mm_per_px": s.mm_per_px, "n_elms": s.n_elms, "heating": s.heating, "fps": s.fps, "view": s.view})
    ss = pd.DataFrame(rows)
    # Greenwald fraction: n_GW = Ip[MA] / (pi a^2) [1e20 m^-3]
    ss.loc[~(ss.ne_bar > 5e18), "ne_bar"] = np.nan          # missing or garbage density summaries
    ss["f_gw"] = ss.ne_bar / (ss.ip_ka / 1e3 / (np.pi * ss.a_minor ** 2) * 1e20)
    ss["h_mode_like"] = ss.n_elms > 5
    # ── two-region model scales ──
    sc = []
    for _, r in ss.iterrows():
        te = r.te_lcfs if np.isfinite(r.te_lcfs) and r.te_lcfs > 5 else TE_DEFAULT
        bt = b_t_at(r.bvac_rmag, r.r_mag, r.r_lcfs) if np.isfinite(r.bvac_rmag) and np.isfinite(r.r_lcfs) else np.nan
        ne_sep = 0.5 * r.ne_bar if np.isfinite(r.ne_bar) else np.nan     # separatrix density ~ half the line average (stated assumption)
        if np.isfinite(bt) and np.isfinite(r.q95) and np.isfinite(ne_sep):
            q = reference_scales(te, ne_sep, bt, r.q95, r.r_mag, r.r_lcfs)
            sc.append({"shot": r.shot, "a_star": q["a_star"], "v_star": q["v_star"], "Lambda": q["Lambda"], "rho_s": q["rho_s"], "L_par": q["L_par"], "te_used": te, "te_measured": np.isfinite(r.te_lcfs) and r.te_lcfs > 5})
        else:
            sc.append({"shot": r.shot, "a_star": np.nan, "v_star": np.nan, "Lambda": np.nan, "rho_s": np.nan, "L_par": np.nan, "te_used": te, "te_measured": False})
    ss = ss.merge(pd.DataFrame(sc), on="shot")
    ss["a_hat"] = (ss.d_med / 2) / ss.a_star        # half width as the blob radius
    ss["v_hat"] = ss.v_med / ss.v_star
    ss.to_parquet(RES / "shot_stats.parquet")

    # ── headline numbers ──
    L = ss[~ss.h_mode_like]  # inter-ELM / L-mode set
    macro("nShotsStats", len(ss)); macro("nShotsL", len(L)); macro("nTracksSel", f"{len(sel):,}".replace(",", "{,}"))
    macro("vMedAll", f"{ss.v_med.median():.0f}"); lo, hi = boot_ci(ss.v_med); macro("vMedAllLo", f"{lo:.0f}"); macro("vMedAllHi", f"{hi:.0f}")
    macro("dMedAll", f"{100*ss.d_med.median():.1f}"); lo, hi = boot_ci(ss.d_med); macro("dMedAllLo", f"{100*lo:.1f}"); macro("dMedAllHi", f"{100*hi:.1f}")
    macro("outwardMed", f"{100*ss.outward_all.median():.0f}")
    macro("vdSlopeMed", f"{ss.vd_slope.median():+.2f}"); lo, hi = boot_ci(ss.vd_slope); macro("vdSlopeMedLo", f"{lo:+.2f}"); macro("vdSlopeMedHi", f"{hi:+.2f}")
    macro("vdRhoMed", f"{ss.vd_rho.median():+.2f}")
    macro("vdSlopePosShare", f"{100*(ss.vd_slope > 0).mean():.0f}")
    # pooled, scale-free: normalise each shot by its medians
    pooled = sel.merge(ss[["shot", "v_med", "d_med"]], on="shot")
    sl, lo, hi, rho = loglog_slope((pooled.delta_r / pooled.d_med).values, (pooled.v_r / pooled.v_med).values)
    macro("vdSlopePooled", f"{sl:+.2f}"); macro("vdSlopePooledLo", f"{lo:+.2f}"); macro("vdSlopePooledHi", f"{hi:+.2f}"); macro("vdRhoPooled", f"{rho:+.2f}")
    # dependence on plasma parameters (per-shot medians, L-mode set)
    for xcol, key in (("ip_ka", "Ip"), ("f_gw", "Fgw"), ("nbi_mw", "Nbi"), ("exposure", "Exp")):
        for ycol, ykey in (("v_med", "V"), ("d_med", "D")):
            x, y = L[xcol].values.astype(float), L[ycol].values.astype(float)
            if xcol == "nbi_mw":
                x = x + 0.05
            sl, lo, hi, rho = loglog_slope(x, y)
            macro(f"sl{ykey}{key}", f"{sl:+.2f}"); macro(f"sl{ykey}{key}Lo", f"{lo:+.2f}"); macro(f"sl{ykey}{key}Hi", f"{hi:+.2f}"); macro(f"rho{ykey}{key}", f"{rho:+.2f}")
    macro("ipMin", f"{L.ip_ka.min():.0f}"); macro("ipMax", f"{L.ip_ka.max():.0f}")
    # exposure-time check: motion blur adds ~v*t_exp to the measured width and can induce a positive v-delta slope
    for exp_us, g in ss.groupby("exposure"):
        key = {2.0: "Two", 4.0: "Four", 5.0: "Five", 7.0: "Seven", 10.0: "Ten"}.get(float(exp_us))
        if key is None or len(g) < 3:
            continue
        macro(f"nExp{key}", len(g)); macro(f"vdSlopeExp{key}", f"{g.vd_slope.median():+.2f}")
        lo, hi = boot_ci(g.vd_slope); macro(f"vdSlopeExp{key}Lo", f"{lo:+.2f}"); macro(f"vdSlopeExp{key}Hi", f"{hi:+.2f}")
        macro(f"dMedExp{key}", f"{100*g.d_med.median():.1f}"); macro(f"vMedExp{key}", f"{g.v_med.median():.0f}")
    # blur-corrected width: subtract v * t_exp in quadrature from the FWHM (Gaussian smear approximation)
    blur = sel.merge(ss[["shot", "exposure"]], on="shot")
    blur["delta_corr"] = np.sqrt(np.maximum(blur.delta_r ** 2 - (blur.v_r * blur.exposure * 1e-6) ** 2, 0))
    sl2, lo2, hi2, rho2 = loglog_slope((blur.delta_corr / blur.groupby("shot").delta_corr.transform("median")).values,
                                       (blur.v_r / blur.groupby("shot").v_r.transform("median")).values)
    macro("vdSlopePooledBlur", f"{sl2:+.2f}"); macro("vdSlopePooledBlurLo", f"{lo2:+.2f}"); macro("vdSlopePooledBlurHi", f"{hi2:+.2f}")
    macro("blurMedCm", f"{100*np.median(blur.v_r * blur.exposure * 1e-6):.2f}")
    macro("fgwMin", f"{L.f_gw.min():.2f}"); macro("fgwMax", f"{L.f_gw.max():.2f}")
    macro("nTeMeasured", int(ss.te_measured.sum())); macro("nTeAssumed", int((~ss.te_measured.astype(bool)).sum())); macro("teDefault", f"{TE_DEFAULT:.0f}")
    ok = np.isfinite(ss.a_hat) & np.isfinite(ss.v_hat)
    macro("aHatMed", f"{ss.a_hat[ok].median():.2f}"); macro("aHatMin", f"{ss.a_hat[ok].min():.2f}"); macro("aHatMax", f"{ss.a_hat[ok].max():.2f}")
    macro("vHatMed", f"{ss.v_hat[ok].median():.2f}"); macro("LambdaMed", f"{ss.Lambda[ok].median():.1f}")
    macro("aStarMed", f"{100*ss.a_star[ok].median():.1f}"); macro("vStarMed", f"{ss.v_star[ok].median():.0f}")
    macro("lkAgreeMed", f"{ss.lk_agreement.median():+.2f}")
    okk = np.isfinite(ss.v_lk_med) & np.isfinite(ss.v_med)
    macro("lkShotRho", f"{spearmanr(ss.v_med[okk], ss.v_lk_med[okk]).correlation:+.2f}"); macro("lkRatioMed", f"{(ss.v_lk_med[okk] / ss.v_med[okk]).median():.2f}")
    # calibration
    words = {"5": "Five", "6": "Six", "7": "Seven", "8": "Eight", "9": "Nine"}
    for _, c in cf[cf.kind == "campaign"].iterrows():
        k = "Camp" + "".join(words.get(ch, ch) for ch in str(c.config) if ch.isdigit())
        macro(f"scale{k}", f"{c.mm_per_px:.1f}"); macro(f"scale{k}Lo", f"{c.mm_per_px_lo:.1f}"); macro(f"scale{k}Hi", f"{c.mm_per_px_hi:.1f}")
        macro(f"scale{k}N", int(c.n_shots_cal) if np.isfinite(c.n_shots_cal) else 0); macro(f"scale{k}Rho", f"{c.spearman:+.2f}" if np.isfinite(c.spearman) else "--")
        own = cf[(cf.campaign == c.campaign) & (cf.kind == "config") & cf.used].sort_values("n_shots", ascending=False)
        macro(f"nOwn{k}", len(own))
        if len(own):
            b = own.iloc[0]
            macro(f"scaleCfg{k}", f"{b.mm_per_px:.1f}"); macro(f"scaleCfg{k}Lo", f"{b.mm_per_px_lo:.1f}"); macro(f"scaleCfg{k}Hi", f"{b.mm_per_px_hi:.1f}")
            macro(f"scaleCfg{k}N", int(b.n_shots)); macro(f"scaleCfg{k}Rho", f"{b.spearman:+.2f}"); macro(f"scaleCfg{k}View", str(b.view).replace("_", " "))
    macro("nConfigs", int((cf.kind == "config").sum())); macro("nConfigsOwn", int(((cf.kind == "config") & cf.used).sum()))
    macro("nShotsOwnScale", int(st.config.str.contains(r"\|").sum()))
    macro("scaleOwnMin", f"{cf[(cf.kind == 'config') & cf.used].mm_per_px.min():.1f}"); macro("scaleOwnMax", f"{cf[(cf.kind == 'config') & cf.used].mm_per_px.max():.1f}")
    own_mm = cf[(cf.kind == "config") & cf.used].mm_per_px; macro("scaleOwnRatio", f"{own_mm.max() / own_mm.min():.1f}")
    own_rho = cf[(cf.kind == "config") & cf.used].spearman
    macro("scaleOwnRhoWeak", f"${own_rho.max():+.2f}$"); macro("scaleOwnRhoStrong", f"${own_rho.min():+.2f}$")
    # ── campaign and view tables for the paper ──
    (PAPER / "tables").mkdir(exist_ok=True)
    rows = []
    for camp, g in st.groupby("campaign"):
        yrs = pd.to_datetime(g.date_time, errors="coerce").dt.year.dropna().astype(int)
        yr = "--" if yrs.empty else (f"{yrs.min()}" if yrs.min() == yrs.max() else f"{yrs.min()}--{yrs.max()}")
        exps = ", ".join(f"{e:g}" for e in sorted(g.exposure.dropna().unique()))
        gs = ss[ss.campaign == camp]; c = cf[(cf.config == camp) & (cf.kind == "campaign")]
        scale = (f"{c.mm_per_px.iloc[0]:.1f} ({c.mm_per_px_lo.iloc[0]:.1f}--{c.mm_per_px_hi.iloc[0]:.1f})"
                 if len(c) and np.isfinite(c.mm_per_px.iloc[0]) else "--")
        own = cf[(cf.campaign == camp) & (cf.kind == "config") & cf.used]
        ownscale = f"{len(own)} ({own.mm_per_px.min():.1f}--{own.mm_per_px.max():.1f})" if len(own) > 1 else (f"1 ({own.mm_per_px.iloc[0]:.1f})" if len(own) else "0")
        stats = f"{gs.v_med.median():.0f} & {100 * gs.d_med.median():.1f}" if len(gs) else "-- & --"
        rows.append(f"{camp} & {yr} & {len(g)} & {len(gs)} & {exps} & {scale} & {ownscale} & {stats} \\\\")
    head = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrllllr}\n\\toprule\nCampaign & Years & Shots & In stats & Exp. ($\\mu$s) & "
            "Pooled scale (mm/px) & Own scales & $\\tilde v_r$ (m/s) & $\\tilde\\delta_r$ (cm) \\\\\n\\midrule\n")
    (PAPER / "tables" / "campaigns.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    rows = []
    for _, c in cf[cf.kind == "config"].sort_values(["campaign", "n_shots"], ascending=[True, False]).iterrows():
        g = st[(st.campaign == c.campaign) & (st.view.astype(str).str.strip() == str(c.view)) & (st.left == c.left)]
        exps = ", ".join(f"{e:g}" for e in sorted(g.exposure.dropna().unique()))
        sc = (f"{c.mm_per_px:.1f} ({c.mm_per_px_lo:.1f}--{c.mm_per_px_hi:.1f})" if np.isfinite(c.get("mm_per_px", np.nan)) and np.isfinite(c.get("mm_per_px_lo", np.nan)) else "--")
        rho = f"{c.spearman:+.2f}" if np.isfinite(c.get("spearman", np.nan)) else "--"
        view = " ".join(str(c.view).replace("_", " ").split())
        for a, b in (("Dalpha filter 10 nm", "D$\\alpha$ 10 nm"), ("Dalpha filter", "D$\\alpha$ filter"), ("photron ", ""), ("image normal", "normal"), ("Midplane probe", "probe")):
            view = view.replace(a, b)
        rows.append(f"{c.campaign} & {view} & {int(c.left)} & {int(c.n_shots)} & {100 * c.r_range_m:.0f} & {rho} & {sc} & "
                    f"{'own' if c.used else 'pooled'} \\\\")
    head = ("\\setlength{\\tabcolsep}{4pt}\\begin{tabular}{llrrrlll}\n\\toprule\nCamp. & View string & ROI left & Shots & "
            "$\\Delta R_{\\rm LCFS}$ (cm) & $\\rho$ & Scale (mm/px) & Used \\\\\n\\midrule\n")
    (PAPER / "tables" / "configs.tex").write_text(head + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")
    macro("nShotsProcessed", len(st)); macro("nTracksAll", f"{len(tr):,}".replace(",", "{,}"))
    macro("nShotsElmy", int(ss.h_mode_like.sum())); macro("nCampaigns", int(st.campaign.nunique()))
    macro("nShotsCalTotal", int(cf[cf.kind == "campaign"].n_shots_cal.fillna(0).sum()))
    # ── data-set bookkeeping for the Data section ──
    lst = pd.read_parquet(ROOT / "data" / "camera_listing.parquet")
    strip = lst[(lst.src == "rbb") & (lst.h == 48) & (lst.w == 256)].drop_duplicates("shot")
    macro("nArchiveShots", f"{len(pd.read_parquet(ROOT / 'data' / 'shots_meta.parquet')):,}".replace(",", "{,}")); macro("nStripShots", len(strip))
    tb = pd.read_parquet(ROOT / "data" / "strip_timebase.parquet")
    macro("nStripBurst", int((tb.n_fast >= 10).sum())); macro("burstMedianMs", f"{((tb.fast_t1 - tb.fast_t0) * 1e3).median():.0f}")
    macro("burstMinMs", f"{((tb.fast_t1 - tb.fast_t0) * 1e3).min():.0f}"); macro("burstMaxMs", f"{((tb.fast_t1 - tb.fast_t0) * 1e3).max():.0f}")
    macro("nStripTenUs", int((np.abs(tb.fast_dt_us - 10) < 1).sum()))
    macro("nOkFlatTop", int(tb.ok_flat_top.sum())); macro("nOkCurrent", int(tb.ok_current.sum())); macro("nOkBurst", int(tb.ok_burst.sum()))
    cand = pd.read_parquet(ROOT / "data" / "candidates.parquet"); cand = cand[(cand.src == "rbb") & (cand["mode"] == "48x256")].drop_duplicates("shot")
    macro("nShotsCandidates", len(cand)); macro("ipCandMin", "400")
    macro("nShotsFetched", int(sum(1 for q in (ROOT / "data" / "shots").glob("*.npz") if q.stem.isdigit() and int(q.stem) in set(cand.shot))))
    macro("nFramesProcessedM", f"{st.n_frames.sum() / 1e6:.1f}"); macro("windowMedianMs", f"{1e3 * (st.t_end - st.t_start).median():.0f}")
    # ── pixel-unit medians per camera configuration (scale-independent view of the campaign differences) ──
    cfgsel = sel.merge(st[["shot", "config"]], on="shot")
    own_cfgs = cf[(cf.kind == "config") & cf.used & (cf.n_shots >= 10)].config
    pc = cfgsel[cfgsel.config.isin(own_cfgs)].groupby("config").agg(vx=("vx_px", "median"), fw=("fwhm_px", "median"), n=("shot", "nunique"))
    macro("nCfgPx", len(pc)); macro("vxPxCfgMin", f"{pc.vx.min():.2f}"); macro("vxPxCfgMax", f"{pc.vx.max():.2f}")
    macro("fwhmPxCfgMin", f"{pc.fw.min():.0f}"); macro("fwhmPxCfgMax", f"{pc.fw.max():.0f}")
    # parameter dependences inside the single best-calibrated configuration (one lens, one scale)
    best_cfg = cf[(cf.kind == "config") & cf.used].sort_values("spearman").iloc[0].config    # most negative rho = tightest
    Lb = L[L.shot.isin(st[st.config == best_cfg].shot)]
    macro("bestCfgView", str(cf.set_index("config").loc[best_cfg, "view"]).replace("_", " ")); macro("nShotsBestCfgL", len(Lb))
    macro("bestCfgScale", f"{cf.set_index('config').loc[best_cfg, 'mm_per_px']:.1f}")
    for xcol, key in (("ip_ka", "Ip"), ("f_gw", "Fgw")):
        for ycol, ykey in (("v_med", "V"), ("d_med", "D")):
            sl, lo, hi, rho = loglog_slope(Lb[xcol].values.astype(float), Lb[ycol].values.astype(float))
            macro(f"slB{ykey}{key}", f"{sl:+.2f}"); macro(f"slB{ykey}{key}Lo", f"{lo:+.2f}"); macro(f"slB{ykey}{key}Hi", f"{hi:+.2f}")
    macro("ipBestMin", f"{Lb.ip_ka.min():.0f}"); macro("ipBestMax", f"{Lb.ip_ka.max():.0f}")
    # ── independent checks: cross-correlation velocity and within-shot calibration ──
    xp = RES / "xcorr.parquet"
    if xp.exists():
        xc = pd.read_parquet(xp).drop(columns=["fps", "sol_sign"], errors="ignore").merge(ss[["shot", "v_med", "mm_per_px", "fps", "campaign"]], on="shot")
        xc = xc[np.isfinite(xc.lag_med) & (xc.frac_ok > 0.5)]
        xc["v_xcorr"] = xc.lag_med * xc.fps * xc.mm_per_px / 1000.0
        macro("nXcorr", len(xc)); macro("xcorrVMed", f"{xc.v_xcorr.median():.0f}")
        macro("xcorrRatioMed", f"{(xc.v_xcorr / xc.v_med).median():.2f}")
        lo, hi = boot_ci((xc.v_xcorr / xc.v_med).values); macro("xcorrRatioLo", f"{lo:.2f}"); macro("xcorrRatioHi", f"{hi:.2f}")
        macro("xcorrShotRho", f"{spearmanr(xc.v_med, xc.v_xcorr).correlation:+.2f}")
        xc.to_parquet(RES / "xcorr_merged.parquet")
    cp = RES / "calib_points.parquet"
    if cp.exists():
        pts = pd.read_parquet(cp); w = pts.within_slope_px_per_m.abs()
        w = w[np.isfinite(w) & (w > 0)]
        if len(w):
            mm = 1000 / w
            macro("nWithinShot", len(w)); macro("withinScaleMed", f"{mm.median():.1f}"); macro("withinScaleQLo", f"{mm.quantile(.25):.1f}"); macro("withinScaleQHi", f"{mm.quantile(.75):.1f}")
            macro("withinScaleMin", f"{mm.min():.1f}"); macro("withinScaleMax", f"{mm.max():.1f}")
            macro("withinScaleMaxLog", f"{np.floor(np.log10(mm.max())):.0f}")
    (PAPER / "numbers.tex").write_text("% Generated by src/analysis.py. Do not edit by hand.\n" + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    pd.set_option("display.width", 250)
    print(ss[["shot", "campaign", "n", "v_med", "d_med", "vd_slope", "vd_rho", "outward_all", "ip_ka", "f_gw", "nbi_mw", "te_lcfs", "a_hat", "v_hat", "Lambda", "h_mode_like"]].round(3).to_string(index=False))
    print(f"\nwrote {len(macros)} macros")


if __name__ == "__main__":
    main()

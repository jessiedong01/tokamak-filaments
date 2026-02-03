"""
Plasma parameters and the two-region blob-model normalization.

Reference quantities (Myra, Russell and D'Ippolito 2006; D'Ippolito, Myra and Zweben 2011;
Paruta et al. 2018), with order-unity amplitude factors set to one:

  rho_s  = c_s / Omega_i,             c_s = sqrt(T_e / m_i)      (deuterium)
  a*     = rho_s (L_par^2 / (rho_s R))^(1/5)
  v*     = c_s (2 a* / R)^(1/2)
  Theta  = (a / a*)^(5/2)
  Lambda = nu_ei L_par / (rho_s Omega_e)

Regime scalings for the normalized velocity v/v* against a/a*:
  sheath-connected (Lambda < 1, Theta > 1):   v/v* ~ (a/a*)^(-2)
  inertial / resistive ballooning:            v/v* ~ (a/a*)^(+1/2)

Connection length: L_par = pi q95 R is a standard estimate for the midplane-to-target
length in a diverted plasma; the paper states the approximation.
"""
import numpy as np

E = 1.602176634e-19
M_P = 1.67262192e-27
M_E = 9.1093837e-31
M_I = 2.0 * M_P          # deuterium


def c_s(te_ev: float) -> float:
    return np.sqrt(te_ev * E / M_I)


def rho_s(te_ev: float, b_t: float) -> float:
    return c_s(te_ev) / (E * b_t / M_I)


def coulomb_log(ne: float, te_ev: float) -> float:
    """NRL formulary, electron-ion, T_e > 10 eV."""
    return 24.0 - np.log(np.sqrt(ne * 1e-6) / te_ev)


def nu_ei(ne: float, te_ev: float) -> float:
    """Electron-ion collision frequency (s^-1), NRL formulary (ne in m^-3, Te in eV)."""
    return 2.91e-6 * (ne * 1e-6) * coulomb_log(ne, te_ev) * te_ev ** (-1.5)


def l_par(q95: float, r0: float) -> float:
    return np.pi * q95 * r0


def reference_scales(te_ev: float, ne: float, b_t: float, q95: float, r0: float, r_edge: float) -> dict:
    """a*, v*, Lambda and the inputs, for a blob at major radius r_edge."""
    cs = c_s(te_ev); rs = rho_s(te_ev, b_t); L = l_par(q95, r0)
    a_star = rs * (L ** 2 / (rs * r_edge)) ** 0.2
    v_star = cs * np.sqrt(2 * a_star / r_edge)
    omega_e = E * b_t / M_E
    lam = nu_ei(ne, te_ev) * L / (rs * omega_e)
    return {"c_s": cs, "rho_s": rs, "L_par": L, "a_star": a_star, "v_star": v_star, "Lambda": lam,
            "te_ev": te_ev, "ne": ne, "b_t": b_t, "q95": q95, "r0": r0, "r_edge": r_edge}


def theta(a: np.ndarray, a_star: float) -> np.ndarray:
    return (a / a_star) ** 2.5


def b_t_at(b_vac_at_rmag: float, r_mag: float, r: float) -> float:
    """Vacuum toroidal field scales as 1/R."""
    return abs(b_vac_at_rmag) * r_mag / r

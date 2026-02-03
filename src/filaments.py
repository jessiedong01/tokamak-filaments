"""
Filament detection and tracking in 100 kHz midplane camera strips.

Pipeline per shot (follows Kirk et al. 2016 and Farley et al. 2019 where they overlap):
  1. background  = running minimum over +/-10 frames (200 us), subtracted
  2. denoise     = 3x3 median, then Gaussian sigma 1 px
  3. edge        = plasma edge column per frame from the smoothed background gradient
  4. detect      = pixels above k * sigma_noise, connected components, min area;
                   per component: centroid, radial FWHM along the central rows,
                   orientation by PCA of the pixel coordinates, amplitude
  5. track       = greedy nearest-neighbour linking frame to frame with a gate,
                   radial velocity by least squares on x(t) for tracks >= 5 frames
  6. flow        = Lucas-Kanade optical flow at each detection as an independent velocity

Radial direction: the SOL is the dark end of the strip, found automatically per shot
(some campaigns mounted the camera upside down). Outward velocity is sol_sign * dx/dt
in pixels per frame; the calibration converts to m/s.
"""
from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.measure import label, regionprops

BG_HALF = 10        # frames each side for the running minimum
K_SIGMA = 3.0       # detection threshold in noise sigma
MIN_AREA = 12       # pixels
GATE_PX = 12        # max centroid jump per frame for linking (12 px = 6 cm at 5 mm/px, 6 km/s)
MIN_TRACK = 5       # frames


def running_min_background(frames: np.ndarray, half: int = BG_HALF) -> np.ndarray:
    """Per-pixel minimum over a window of 2*half+1 frames, as float32."""
    f = frames.astype(np.float32)
    return ndi.minimum_filter1d(f, size=2 * half + 1, axis=0, mode="nearest")


def fluctuation(frames: np.ndarray) -> np.ndarray:
    """Background-subtracted, denoised frames (n, h, w) float32."""
    bg = running_min_background(frames)
    d = frames.astype(np.float32) - bg
    d = ndi.median_filter(d, size=(1, 3, 3))
    return ndi.gaussian_filter(d, sigma=(0, 1.0, 1.0))


def noise_sigma(d: np.ndarray) -> float:
    """Robust noise estimate from the median absolute deviation of all fluctuation pixels."""
    med = np.median(d)
    return float(1.4826 * np.median(np.abs(d - med)) + 1e-6)


def _half_level_edge(prof: np.ndarray, sol_sign: int, dark_cols: int = 10, interior: tuple = (0.4, 0.75)) -> np.ndarray:
    """Column where the smoothed row-averaged profile first rises above half way between the
    dark level (outermost `dark_cols` columns on the SOL side) and the interior level (median
    over the interior fraction of the strip), scanning from the SOL side. Sub-pixel by
    linear interpolation. Robust to fixed bright structures and to dim edges."""
    n, w = prof.shape
    if sol_sign == -1:          # SOL at small x: scan left to right
        p = prof
    else:                       # SOL at large x: flip so the scan is always left to right
        p = prof[:, ::-1]
    dark = p[:, :dark_cols].mean(axis=1, keepdims=True)
    lo, hi = int(interior[0] * w), int(interior[1] * w)
    level = dark + 0.5 * (np.median(p[:, lo:hi], axis=1, keepdims=True) - dark)
    above = p > level
    first = np.argmax(above, axis=1).astype(float)
    rows = np.arange(n); i = np.clip(first.astype(int), 1, w - 1)
    p0, p1 = p[rows, i - 1], p[rows, i]
    frac = np.where(p1 > p0, (level[:, 0] - p0) / np.maximum(p1 - p0, 1e-6), 0.0)
    x = i - 1 + np.clip(frac, 0, 1)
    return x if sol_sign == -1 else (w - 1) - x


def edge_column(frames: np.ndarray, smooth_frames: int = 200) -> tuple[np.ndarray, int]:
    """Plasma-edge column per frame and the SOL direction.

    The row-averaged profile is smoothed over `smooth_frames` so filaments do not move
    the edge. The SOL is the darker end of the strip; sol_sign is +1 if the SOL lies at
    larger x and -1 if it lies at smaller x. Outward motion is sol_sign * dx/dt.
    """
    prof = frames.astype(np.float32).mean(axis=1)                     # (n, w)
    prof = ndi.uniform_filter1d(prof, size=smooth_frames, axis=0, mode="nearest")
    prof = ndi.gaussian_filter1d(prof, sigma=2.0, axis=1)
    w = prof.shape[1]
    left, right = prof[:, : w // 4].mean(), prof[:, -w // 4:].mean()
    sol_sign = 1 if right < left else -1
    return _half_level_edge(prof, sol_sign), sol_sign


@dataclass
class Detection:
    frame: int
    x: float          # centroid column
    y: float          # centroid row
    area: int
    amp: float        # peak fluctuation
    fwhm_x: float     # radial full width at half maximum (px), from the row-profile
    angle: float      # orientation of the major axis, radians from the x axis
    elong: float      # major / minor axis length
    sol_dist: float   # signed distance outside the plasma edge (px); positive = in the SOL


def radial_fwhm(patch: np.ndarray, cy: float, half_rows: int = 2) -> float:
    """FWHM in x of the profile along a horizontal chord through the component centroid
    (mean over 2*half_rows+1 rows). Measuring on a chord avoids widening by the streak's tilt."""
    r0 = int(round(cy)); r0 = min(max(r0, 0), patch.shape[0] - 1)
    rows = slice(max(0, r0 - half_rows), min(patch.shape[0], r0 + half_rows + 1))
    p = patch[rows].mean(axis=0)
    if p.max() <= 0:
        return float("nan")
    half = 0.5 * p.max()
    above = np.flatnonzero(p >= half)
    return float(above[-1] - above[0] + 1)


def detect_frame(d: np.ndarray, sigma: float, x_edge: float, sol_sign: int = -1, k: float = K_SIGMA,
                 min_area: int = MIN_AREA, frame_index: int = 0) -> list[Detection]:
    mask = d > k * sigma
    lab = label(mask, connectivity=2)
    out = []
    for r in regionprops(lab, intensity_image=d):
        if r.area < min_area:
            continue
        cy, cx = r.centroid_weighted
        minr, minc, maxr, maxc = r.bbox
        patch = np.where(lab[minr:maxr, minc:maxc] == r.label, d[minr:maxr, minc:maxc], 0.0)
        elong = r.axis_major_length / max(r.axis_minor_length, 1e-3)
        out.append(Detection(frame_index, float(cx), float(cy), int(r.area), float(r.intensity_max),
                             radial_fwhm(patch, cy - minr), float(r.orientation), float(elong), float(sol_sign * (cx - x_edge))))
    return out


def detect(d: np.ndarray, edge: np.ndarray, sol_sign: int, **kw) -> list[Detection]:
    sigma = noise_sigma(d)
    dets = []
    for i in range(d.shape[0]):
        dets.extend(detect_frame(d[i], sigma, float(edge[i]), sol_sign, frame_index=i, **kw))
    return dets


def link_tracks(dets: list[Detection], gate: float = GATE_PX) -> list[list[Detection]]:
    """Frame-to-frame assignment by centroid distance (Hungarian within the gate)."""
    by_frame: dict[int, list[Detection]] = {}
    for det in dets:
        by_frame.setdefault(det.frame, []).append(det)
    frames = sorted(by_frame)
    tracks: list[list[Detection]] = []
    active: dict[int, list[Detection]] = {}   # index into tracks -> track (last frame == previous)
    for fi, f in enumerate(frames):
        cur = by_frame[f]
        prev_ids = [tid for tid, tr in active.items() if tr[-1].frame == f - 1]
        assigned = set()
        if prev_ids and cur:
            P = np.array([[tracks[tid][-1].x, tracks[tid][-1].y] for tid in prev_ids])
            C = np.array([[c.x, c.y] for c in cur])
            cost = np.linalg.norm(P[:, None, :] - C[None, :, :], axis=-1)
            cost[cost > gate] = 1e6
            rows, cols = linear_sum_assignment(cost)
            for r, c in zip(rows, cols):
                if cost[r, c] < gate:
                    tracks[prev_ids[r]].append(cur[c]); assigned.add(c)
        for c, det in enumerate(cur):
            if c not in assigned:
                tracks.append([det]); active[len(tracks) - 1] = tracks[-1]
        active = {tid: tr for tid, tr in active.items() if tr[-1].frame >= f - 1}
    return tracks


@dataclass
class Track:
    start_frame: int
    n: int
    x0: float
    y0: float
    vx_px: float      # px per frame, positive = outward (least squares slope of sol_sign * x)
    vx_err: float
    fwhm_x: float     # median radial FWHM (px)
    amp: float        # peak amplitude over the track
    sol_dist: float   # median distance outside the edge (px)
    angle: float
    elong: float


def summarize_tracks(tracks: list[list[Detection]], sol_sign: int, min_len: int = MIN_TRACK) -> list[Track]:
    out = []
    for tr in tracks:
        if len(tr) < min_len:
            continue
        t = np.array([d.frame for d in tr], float); x = sol_sign * np.array([d.x for d in tr])
        A = np.vstack([t - t[0], np.ones_like(t)]).T
        coef, res, *_ = np.linalg.lstsq(A, x, rcond=None)
        resid = x - A @ coef
        err = float(np.sqrt(np.sum(resid ** 2) / max(len(t) - 2, 1) / np.sum((t - t.mean()) ** 2)))
        out.append(Track(tr[0].frame, len(tr), float(x[0]), float(tr[0].y), float(coef[0]), err,
                         float(np.nanmedian([d.fwhm_x for d in tr])), float(max(d.amp for d in tr)),
                         float(np.median([d.sol_dist for d in tr])), float(np.median([d.angle for d in tr])),
                         float(np.median([d.elong for d in tr]))))
    return out


def lucas_kanade(d: np.ndarray, dets: list[Detection], win: int = 7) -> np.ndarray:
    """Lucas-Kanade flow (u, v) in px/frame at each detection, from frames f and f+1."""
    out = np.full((len(dets), 2), np.nan, np.float32)
    n, h, w = d.shape
    Ix = ndi.sobel(d, axis=2) / 8.0; Iy = ndi.sobel(d, axis=1) / 8.0
    half = win // 2
    for k, det in enumerate(dets):
        f = det.frame
        if f + 1 >= n:
            continue
        y, x = int(round(det.y)), int(round(det.x))
        y0, y1, x0, x1 = max(0, y - half), min(h, y + half + 1), max(0, x - half), min(w, x + half + 1)
        ix = Ix[f, y0:y1, x0:x1].ravel(); iy = Iy[f, y0:y1, x0:x1].ravel()
        it = (d[f + 1, y0:y1, x0:x1] - d[f, y0:y1, x0:x1]).ravel()
        A = np.vstack([ix, iy]).T
        AtA = A.T @ A
        if np.linalg.cond(AtA) > 1e4:
            continue
        out[k] = -np.linalg.solve(AtA, A.T @ it)
    return out

#!/usr/bin/env python3
"""
eq_params.py — B_T (on-axis), q95, and (optionally) LCFS n_e for a discharge,
from a BOUT++ grid (.grd.nc), a G-EQDSK g-file, or a FIESTA/CREATE .equ
equilibrium.

  B_T(axis)  toroidal field at the magnetic axis  [T]
  q95        safety factor at psi_n = 0.95
  LCFS n_e   separatrix density at the OMP         [1e19 m^-3]  (needs a dump)

How each is obtained
--------------------
* grid  : B_T = stored `Bt_axis`; q95 from `ShiftAngle` (= 2*pi*q on closed
          surfaces) interpolated to psi_n=0.95 via `psixy`,`psi_axis`,`psi_bdry`.
* eqdsk : B_T = |BCENTR|*RCENTR/RMAXIS; q95 from the `QPSI` profile at psi_n=0.95
          (falls back to a flux-surface integral if QPSI is absent/all-zero, as
          pyEquilibrium-written g-files can be).
* .equ  : B_T = btf*rtf/R_axis; q95 = (F/2pi) * closed-contour integral of
          dl/(R|grad psi|) on the psi_n=0.95 surface, with F = btf*rtf (vacuum
          toroidal field — the .equ stores no F(psi) profile, so this is a
          vacuum-F approximation, good to a few % at MAST-U field).
* LCFS n_e : small targeted read of `n` at the OMP separatrix of a STORM dump
             (× n_0 from the [filaments] block). Optional; omit `dump` to skip.

Edit SOURCES and run.  (q95 needs matplotlib for the .equ contour integral.)
"""

import os
import re
import numpy as np
import netCDF4

# ============================================================================
# CONFIG — one dict per discharge. `path` is a grid/eqdsk/equ; `dump` optional.
# ============================================================================
_R = "debug/equ_eqdsk"
SOURCES = [
    dict(label="46860 (.equ)",  path=f"{_R}/sxd_46860_equ_450ms.symm.x4.crop.equ"),
    dict(label="45460 (eqdsk)", path=f"{_R}/exp45460_new.eqdsk"),
    # dict(label="45456 (grid)", path=".../mastu_sxd_45456/bout_v3.grd.nc"),
    # add dump=".../<n>/boutdata.nc" to any entry to also get LCFS n_e
]
# ============================================================================

e = 1.602176634e-19
u = 1.66053906660e-27
_STORM_SECTIONS = ("filaments", "storm3d", "storm2d", "storm")


# ---------------------------------------------------------------------------
# BOUT++ grid
# ---------------------------------------------------------------------------
def from_grid(path):
    g = netCDF4.Dataset(path)
    s = lambda k: float(np.asarray(g[k][:]).flat[0])
    Bt_axis = abs(s("Bt_axis"))
    psia, psib = s("psi_axis"), s("psi_bdry")
    ixs = int(s("ixseps1"))
    j12, j22 = int(s("jyseps1_2")), int(s("jyseps2_2"))
    y = (j12 + j22) // 2
    SA = np.asarray(g["ShiftAngle"][:])
    psixy = np.asarray(g["psixy"][:])
    psn = (psixy[:, y] - psia) / (psib - psia)
    q = np.abs(SA) / (2 * np.pi)
    core = np.arange(ixs)
    good = np.isfinite(q[core])
    q95 = float(np.interp(0.95, psn[core][good], q[core][good])) if good.sum() > 2 else np.nan
    g.close()
    return dict(Bt_axis=Bt_axis, q95=q95, note="grid ShiftAngle")


# ---------------------------------------------------------------------------
# G-EQDSK
# ---------------------------------------------------------------------------
def from_eqdsk(path):
    txt = open(path).read()
    ints = re.findall(r"\d+", txt.splitlines()[0])
    NW, NH = int(ints[-2]), int(ints[-1])
    F = np.array([float(x) for x in re.findall(r"[-+]?\d*\.\d+[EeDd][-+]?\d+", txt)])
    RCENTR, RMAXIS, SIMAG, SIBRY, BCENTR = F[2], F[5], F[7], F[8], F[9]
    Bt_axis = abs(BCENTR) * RCENTR / RMAXIS
    off = 20 + 4 * NW + NW * NH
    QPSI = np.abs(F[off:off + NW])
    if QPSI.size == NW and np.any(QPSI > 1e-6):
        q95 = float(np.interp(0.95, np.linspace(0, 1, NW), QPSI))
        note = "eqdsk QPSI"
    else:                                       # QPSI missing/zero → integrate PSIRZ
        psirz = F[20 + 4 * NW:20 + 4 * NW + NW * NH].reshape(NH, NW)
        RDIM, ZDIM, RLEFT, ZMID = F[0], F[1], F[3], F[4]
        R = RLEFT + np.linspace(0, RDIM, NW)
        Z = ZMID + np.linspace(-ZDIM / 2, ZDIM / 2, NH)
        Fpol = abs(BCENTR) * RCENTR                       # vacuum F = R*Bt
        Ra, Za, psia = _magnetic_axis(R, Z, psirz, SIBRY)
        q95 = _q_from_psi(R, Z, psirz, psia, SIBRY, Fpol, 0.95, (Ra, Za))
        note = "eqdsk PSIRZ integral (QPSI absent)"
    return dict(Bt_axis=Bt_axis, q95=q95, note=note)


# ---------------------------------------------------------------------------
# FIESTA / CREATE .equ
# ---------------------------------------------------------------------------
def from_equ(path):
    txt = open(path).read()
    def scal(name):
        m = re.search(rf"\b{name}\s*=\s*([-+]?[\d.]+(?:[EeDd][-+]?\d+)?)", txt)
        return float(m.group(1).replace("D", "E")) if m else None
    jm, km = int(scal("jm")), int(scal("km"))
    psib, btf, rtf = scal("psib"), scal("btf"), scal("rtf")
    # all floats AFTER the rtf assignment are the arrays: r[jm], z[km], psi[jm*km]
    tail = txt[re.search(r"\brtf\s*=.*", txt).end():]
    nums = np.array([float(x.replace("D", "E"))
                     for x in re.findall(r"[-+]?\d*\.\d+(?:[EeDd][-+]?\d+)?", tail)])
    if nums.size < jm + km + jm * km:
        raise ValueError(f".equ parse: found {nums.size} floats, need {jm+km+jm*km} "
                         f"(jm={jm}, km={km}). File layout differs from the assumed "
                         "r[jm], z[km], psi[jm*km] order — inspect the file.")
    R = nums[:jm]
    Z = nums[jm:jm + km]
    psi = nums[jm + km:jm + km + jm * km].reshape(km, jm) + psib   # stored as psi-psib
    R_axis, Z_axis, psi_axis = _magnetic_axis(R, Z, psi, psib)
    Fpol = abs(btf) * abs(rtf)
    Bt_axis = Fpol / R_axis
    q95 = _q_from_psi(R, Z, psi, psi_axis, psib, Fpol, 0.95, (R_axis, Z_axis))
    return dict(Bt_axis=Bt_axis, q95=q95,
                note=f"equ contour (vacuum F; R_axis={R_axis:.3f} m, "
                     f"psi_axis={psi_axis:.4g}, psi_bnd={psib:.4g})")


def _magnetic_axis(R, Z, psi, psib):
    """O-point on the midplane: the psi extremum along Z~=0 furthest from psib.
    Robust against divertor-coil / X-point psi extrema, which sit off-midplane."""
    k0 = int(np.argmin(np.abs(Z)))
    j = int(np.argmax(np.abs(psi[k0] - psib)))
    return float(R[j]), float(Z[k0]), float(psi[k0, j])


def _q_from_psi(R, Z, psi, psi_axis, psi_bnd, Fpol, psin, axis_RZ):
    """q at normalised flux `psin` from psi(Z,R): q = (F/2pi) oint dl/(R|grad psi|),
    integrated along the closed psi=const contour that ENCLOSES the magnetic axis
    `axis_RZ=(R_axis,Z_axis)` (so a same-level coil/leg contour is never picked)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.path import Path
    from scipy.interpolate import RegularGridInterpolator
    level = psi_axis + psin * (psi_bnd - psi_axis)
    dpsi_dZ, dpsi_dR = np.gradient(psi, Z, R)
    gmag = np.hypot(dpsi_dR, dpsi_dZ)
    gi = RegularGridInterpolator((Z, R), gmag, bounds_error=False, fill_value=np.nan)
    fig = plt.figure(); cs = plt.contour(R, Z, psi, levels=[level]); plt.close(fig)
    best = None
    for seg in cs.allsegs[0]:
        if len(seg) < 20:
            continue
        closed = np.hypot(*(seg[0] - seg[-1])) < 0.05 * (R.max() - R.min())
        if not closed or not Path(seg).contains_point(axis_RZ):
            continue                         # keep only the loop around the axis
        length = np.sum(np.hypot(np.diff(seg[:, 0]), np.diff(seg[:, 1])))
        if best is None or length > best[1]:
            best = (seg, length)
    if best is None:
        return np.nan
    seg = best[0]
    Rc, Zc = seg[:, 0], seg[:, 1]
    dl = np.hypot(np.diff(Rc), np.diff(Zc))
    Rm = 0.5 * (Rc[1:] + Rc[:-1]); Zm = 0.5 * (Zc[1:] + Zc[:-1])
    gm = gi(np.column_stack([Zm, Rm]))
    good = np.isfinite(gm) & (gm > 0)
    integ = np.sum(dl[good] / (Rm[good] * gm[good]))
    return float(abs(Fpol) * integ / (2 * np.pi))


# ---------------------------------------------------------------------------
# LCFS n_e from a STORM dump (optional)
# ---------------------------------------------------------------------------
def lcfs_ne(dump):
    folder = os.path.dirname(dump)
    n0 = None
    for d in (folder, os.path.dirname(folder)):
        for nm in ("BOUT.settings", "BOUT.inp"):
            p = os.path.join(d, nm)
            if os.path.exists(p):
                sec = None
                for line in open(p, errors="replace"):
                    line = line.split("#")[0].strip()
                    if line.startswith("["):
                        sec = line.strip("[]").strip().lower()
                    elif sec in _STORM_SECTIONS and line.lower().startswith("n_0"):
                        try: n0 = float(line.split("=")[1]) * 1e6
                        except Exception: pass
        if n0:
            break
    if not n0:
        return np.nan
    d = netCDF4.Dataset(dump)
    g = lambda k: int(d[k][:].item())
    ny_full = len(d.dimensions["y"]); ny = g("ny"); MYG = g("MYG")
    gg = MYG if ny_full > ny else 0
    up = 3 * gg if 0 < g("ny_inner") < ny else gg
    y = int(0.5 * ((g("jyseps1_2") + up) + (g("jyseps2_2") + up)))
    ix = g("ixseps1")
    ne = float(np.asarray(d["n"][:, ix, y, :]).mean()) * n0
    d.close()
    return ne / 1e19


def kind(path):
    p = path.lower()
    if p.endswith((".grd.nc", ".nc")):
        return "grid"
    if p.endswith(".equ"):
        return "equ"
    return "eqdsk"


def main():
    os.chdir("/home/eric/Downloads/BOUT_code/work_dir/github_tools")
    print(f"{'label':22s}{'B_T(axis)':>11s}{'q95':>8s}{'LCFS n_e':>11s}   note")
    print(f"{'':22s}{'[T]':>11s}{'':>8s}{'[1e19]':>11s}")
    print("-" * 80)
    disp = {"grid": from_grid, "eqdsk": from_eqdsk, "equ": from_equ}
    for src in SOURCES:
        path = src["path"]
        if not os.path.exists(path):
            print(f"{src['label']:22s}  [skip] not found: {path}")
            continue
        try:
            r = disp[kind(path)](path)
        except Exception as ex:                                # noqa: BLE001
            print(f"{src['label']:22s}  [error] {ex}")
            continue
        ne = lcfs_ne(src["dump"]) if src.get("dump") else np.nan
        ne_s = f"{ne:11.3f}" if np.isfinite(ne) else f"{'-':>11s}"
        print(f"{src['label']:22s}{r['Bt_axis']:11.3f}{r['q95']:8.2f}{ne_s}   {r['note']}")


if __name__ == "__main__":
    main()

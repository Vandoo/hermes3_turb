"""
particle_fill_time.py
---------------------
"Is this run anywhere near a particle steady state, and how long would it take?"

Integrates the Hermes-3 particle and energy inventories and the volumetric
source/sink diagnostics over one or more squashed folders, then reports the
timescales that decide whether a fuelling change can possibly show up in the
simulated window:

  tau_fill(plasma) = N_e / (dN_e/dt)     how fast the plasma inventory is moving now
  tau_fill(neutral)= N_d / (dN_d/dt)     the neutral gas fills first, and faster
  tau_p            = N_e / Gamma_loss    particle replacement time (sheath + pump)
  tau_E            = W / P_in            energy confinement - T re-equilibrates on this

dN/dt is a least-squares slope through every output time in the requested
folders, NOT a two-point difference, so turbulent fluctuations average out.

All integrals are full-torus and exclude guard cells (BOUT's x-guards always,
y-guards when the dump retains them) - guard Jacobians are garbage under
extrapolate_y=false and will fake enormous sources.

Volume element: dV = J*dx*dy*dz * rho_s0^3, scaled by 2*pi/(nz*dz) to go from
the simulated wedge to the full torus.  Sanity check: the printed Sd_src must
equal the puff rate in the grid-file name.

Usage
-----
    python particle_fill_time.py <casedir> [dir_1] [dir_2]
    python particle_fill_time.py <casedir>/136/boutdata.nc

    casedir     directory holding the numbered squash folders (e.g. .../data)
    dir_1 dir_2 inclusive folder range; omit to use every numeric folder found

Edit TARGET_NE_SEP / US_PER_DAY below to get the "how long would I have to wait"
projection at the end.
"""

from __future__ import annotations

import os
import re
import sys
import glob

import numpy as np
import netCDF4 as nc

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
TARGET_NE_SEP = 5.0e18     # [m^-3] separatrix density you are aiming at (None to skip)
US_PER_DAY    = 1.0        # observed wall-clock rate, microseconds of plasma time per day
DUMP_NAMES    = ("boutdata_guards.nc", "boutdata.nc")   # preference order

# Volumetric diagnostics: (name, sign convention already in the file)
PARTICLE_SOURCES = [
    "Sd_src",              # external gas puff (from the grid file)
    "Sd_target_recycle",   # neutrals returned at the divertor plates
    "Sd_wall_recycle",     # neutrals returned at the radial walls
    "Sd_pump",             # negative: neutrals absorbed by the pump
    "Se_src",              # core electron particle source
]

# --------------------------------------------------------------------------- #
# Geometry / masks
# --------------------------------------------------------------------------- #
def cell_volume(ds: nc.Dataset) -> tuple[np.ndarray, float]:
    """Return (dV[x,y,1] in m^3 per z-cell, torus_factor)."""
    J   = np.array(ds["J"][:])
    dx  = np.array(ds["dx"][:])
    dy  = np.array(ds["dy"][:])
    dz  = np.array(ds["dz"][:])
    rho = float(np.asarray(ds["rho_s0"][:]).ravel()[0])
    nz  = ds.dimensions["z"].size

    dV = (J * dx * dy * dz)[:, :, None] * rho**3
    # simulated toroidal extent -> full torus
    torus = 2.0 * np.pi / (nz * float(np.mean(dz)))
    return dV, torus


def interior_mask(ds: nc.Dataset, mxg: int = 2) -> np.ndarray:
    """Boolean [x,y] mask excluding x-guards and (if present) y-boundary guards."""
    nx = ds.dimensions["x"].size
    ny_loaded = ds.dimensions["y"].size
    ny_meta = int(np.asarray(ds["ny"][:]).ravel()[0])

    mask = np.zeros((nx, ny_loaded), bool)
    mask[mxg:nx - mxg, :] = True

    if ny_loaded == ny_meta:
        return mask                                   # no y-guards retained

    # y-guards present: how many rows per target boundary?
    j21 = int(np.asarray(ds["jyseps2_1"][:]).ravel()[0])
    j12 = int(np.asarray(ds["jyseps1_2"][:]).ravel()[0])
    n_targets = 2 if j21 == j12 else 4                # single vs double null
    myg = (ny_loaded - ny_meta) // n_targets

    if n_targets == 2:
        guard_rows = list(range(myg)) + list(range(ny_loaded - myg, ny_loaded))
    else:
        ny_inner = int(np.asarray(ds["ny_inner"][:]).ravel()[0])
        guard_rows = (list(range(myg))
                      + list(range(ny_inner + myg, ny_inner + 3 * myg))
                      + list(range(ny_loaded - myg, ny_loaded)))
    mask[:, guard_rows] = False
    return mask


def dump_weight(ds: nc.Dataset) -> np.ndarray:
    """Interior volume weight [x, y, 1] in m^3, full torus, for THIS dump."""
    dV, torus = cell_volume(ds)
    return interior_mask(ds)[:, :, None] * dV * torus


# --------------------------------------------------------------------------- #
# Integration helpers
# --------------------------------------------------------------------------- #
def volume_integral(ds, name, weight, tidx=None):
    """Integrate a saved field over the interior, full torus, in SI.

    Returns a scalar if tidx is given, else an array over time.
    """
    if name not in ds.variables:
        return None
    var = ds[name]
    conv = float(getattr(var, "conversion", 1.0))
    if tidx is None:
        data = np.array(var[:])
        return np.array([float((data[i] * weight).sum()) * conv
                         for i in range(data.shape[0])])
    return float((np.array(var[tidx]) * weight).sum()) * conv


def folder_list(casedir, d1=None, d2=None):
    out = []
    for p in sorted(glob.glob(os.path.join(casedir, "*")),
                    key=lambda s: (not os.path.basename(s).isdigit(),
                                   int(os.path.basename(s)) if os.path.basename(s).isdigit() else 0)):
        b = os.path.basename(p)
        if not b.isdigit() or not os.path.isdir(p):
            continue
        n = int(b)
        if d1 is not None and n < d1:
            continue
        if d2 is not None and n > d2:
            continue
        for name in DUMP_NAMES:
            f = os.path.join(p, name)
            if os.path.exists(f):
                out.append((n, f))
                break
    return out


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 1

    arg = argv[1]
    if arg.endswith(".nc"):
        dumps = [(0, arg)]
        casedir = os.path.dirname(arg)
    else:
        casedir = arg.rstrip("/")
        d1 = int(argv[2]) if len(argv) > 2 else None
        d2 = int(argv[3]) if len(argv) > 3 else None
        dumps = folder_list(casedir, d1, d2)

    if not dumps:
        print(f"no squashed dumps found under {casedir}")
        return 1

    print(f"=== {len(dumps)} dump(s): folders "
          f"{dumps[0][0]} .. {dumps[-1][0]} ===")

    # ---- geometry and normalisations from the first dump -------------------
    ds0 = nc.Dataset(dumps[0][1])
    dV, torus = cell_volume(ds0)
    mask = interior_mask(ds0)
    Omega = float(np.asarray(ds0["Omega_ci"][:]).ravel()[0])
    Nnorm = float(np.asarray(ds0["Nnorm"][:]).ravel()[0])
    Tnorm = float(np.asarray(ds0["Tnorm"][:]).ravel()[0])
    ny_loaded = ds0.dimensions["y"].size
    ny_meta = int(np.asarray(ds0["ny"][:]).ravel()[0])
    nz = ds0.dimensions["z"].size
    ds0.close()

    # weight has shape [x, y, 1] and broadcasts over z; summing it therefore
    # counts each (x,y) once, so the interior volume needs an explicit * nz.
    weight = mask[:, :, None] * dV * torus            # m^3 per cell
    volume = float(weight.sum()) * nz

    print(f"Omega_ci = {Omega:.6e} s^-1   ->  1 time unit = {1e9/Omega:.3f} ns")
    print(f"Nnorm    = {Nnorm:.3e} m^-3   Tnorm = {Tnorm:g} eV")
    print(f"y-guards in dump: {'yes' if ny_loaded > ny_meta else 'no'}"
          f"  (loaded ny={ny_loaded}, metadata ny={ny_meta})")
    print(f"interior volume (full torus) = {volume:.4g} m^3\n")

    # ---- time series of the inventories ------------------------------------
    t_all, Ne_all, Nd_all, We_all, Wi_all = [], [], [], [], []
    for n, path in dumps:
        ds = nc.Dataset(path)
        # Recompute the weight for EVERY dump: a run can mix guard-retaining
        # (boutdata_guards.nc) and guardless (boutdata.nc) folders, and reusing
        # one folder's mask silently mis-integrates the others.
        w = dump_weight(ds)
        tname = "t" if "t" in ds.variables else "t_array"
        t_all.append(np.array(ds[tname][:]))
        Ne_all.append(volume_integral(ds, "Ne", w))
        nd = volume_integral(ds, "Nd", w)
        Nd_all.append(nd if nd is not None else np.zeros_like(t_all[-1]))
        # stored energy: 3/2 * P, P conversion is Pa
        pe = volume_integral(ds, "Pe", w)
        pi = volume_integral(ds, "Pd+", w)
        We_all.append(1.5 * pe if pe is not None else np.zeros_like(t_all[-1]))
        Wi_all.append(1.5 * pi if pi is not None else np.zeros_like(t_all[-1]))
        ds.close()

    t = np.concatenate(t_all) / Omega                 # seconds
    order = np.argsort(t)
    t = t[order]
    Ne = np.concatenate(Ne_all)[order]
    Nd = np.concatenate(Nd_all)[order]
    We = np.concatenate(We_all)[order]
    Wi = np.concatenate(Wi_all)[order]

    span = t[-1] - t[0]
    print(f"time window: {t[0]*1e6:.2f} -> {t[-1]*1e6:.2f} us "
          f"({span*1e6:.2f} us, {len(t)} outputs)\n")

    # ---- sources at the last time -----------------------------------------
    ds = nc.Dataset(dumps[-1][1])
    weight_last = dump_weight(ds)
    print("--- volumetric particle sources (last output, full torus, interior) ---")
    totals = {}
    for name in PARTICLE_SOURCES + ["Sd+_iz", "Sd+_rec"]:
        val = volume_integral(ds, name, weight_last, tidx=-1)
        if val is None:
            print(f"  {name:<20} (not saved)")
            continue
        totals[name] = val
        print(f"  {name:<20} {val:+.4e} /s")
    ds.close()

    puff = totals.get("Sd_src", 0.0)
    iz = totals.get("Sd+_iz", 0.0)
    rec = totals.get("Sd+_rec", 0.0)
    pump = totals.get("Sd_pump", 0.0)
    core_src = totals.get("Se_src", 0.0)
    trecyc = totals.get("Sd_target_recycle", 0.0)
    wrecyc = totals.get("Sd_wall_recycle", 0.0)

    if "Sd_src" in totals:
        print("\n  (Sd_src must match the puff rate in the grid-file name - "
              "if it does not, the volume element is wrong)")

    # ---- rates -------------------------------------------------------------
    def slope(y):
        return float(np.polyfit(t, y, 1)[0]) if len(t) > 2 else np.nan

    dNe, dNd = slope(Ne), slope(Nd)
    W = We[-1] + Wi[-1]

    print("\n--- inventories and rates ---")
    has_neutrals = Nd[-1] > 0
    print(f"  N_e            {Ne[-1]:.4e} particles")
    print(f"  N_d            {Nd[-1]:.4e} particles"
          if has_neutrals else "  N_d            (no neutral species in dump)")
    print(f"  W = 3/2*int(Pe+Pi)  {W:.4e} J")
    print(f"  dN_e/dt        {dNe:+.4e} /s   (least-squares over the window)")
    if has_neutrals:
        print(f"  dN_d/dt        {dNd:+.4e} /s")

    print("\n--- timescales ---")
    if dNe > 0:
        print(f"  tau_fill(plasma)   N_e/(dN_e/dt) = {Ne[-1]/dNe*1e3:8.3f} ms")
    if dNd > 0:
        print(f"  tau_fill(neutral)  N_d/(dN_d/dt) = {Nd[-1]/dNd*1e3:8.3f} ms")

    # particle replacement: ions lost at the sheath.  The target recycling
    # diagnostic is multiplier * Gamma_target, so back out Gamma_target if we
    # can find the multiplier; otherwise report the recycled flux itself.
    gamma_t = trecyc            # lower bound on the sheath ion flux
    if gamma_t > 0:
        print(f"  tau_p              N_e/Gamma_target = {Ne[-1]/gamma_t*1e3:8.3f} ms")

    # net volumetric plasma source, for reference
    plasma_src = iz + rec + core_src
    if plasma_src > 0:
        print(f"  N_e/(ionisation+core src)          = {Ne[-1]/plasma_src*1e3:8.3f} ms")

    return _report_energy_and_projection(W, t, Ne, dNe, puff, pump, gamma_t,
                                         core_src, wrecyc, dumps)


def _report_energy_and_projection(W, t, Ne, dNe, puff, pump, gamma_t,
                                  core_src, wrecyc, dumps):
    # ---- energy: read the input power from the dump if it is saved ---------
    ds = nc.Dataset(dumps[-1][1])
    w = dump_weight(ds)
    P_in = 0.0
    for name in ("Pe_src", "Pd+_src", "Pi_src"):
        v = volume_integral(ds, name, w, tidx=-1)
        if v is not None:
            P_in += 1.5 * v              # Pa/s -> W
    ds.close()

    if P_in:
        tE = W / P_in
        tE_str = f"{tE*1e6:.1f} us" if tE < 1e-3 else f"{tE*1e3:.3f} ms"
        print(f"  tau_E              W/P_in = {tE_str:>12s} "
              f"(P_in = {P_in/1e3:.1f} kW)")
        print("    -> T re-equilibrates this fast; n does not.")
    else:
        print("  tau_E              (P*_src not saved - set diagnose=true to get it)")

    # ---- projection --------------------------------------------------------
    if TARGET_NE_SEP is None or dNe <= 0:
        return 0
    print("\n--- projection ---")
    print("  Assuming n_e,sep tracks the total inventory (uniform profile shape),")
    print("  and that dN_e/dt stays at its present value (it will not - the fill")
    print("  slows as the sheath sink grows - so this is a LOWER bound on the wait).")
    frac_rate = dNe / Ne[-1]                       # 1/s
    for factor in (1.25, 1.5, 2.0):
        t_need = (factor - 1.0) / frac_rate
        print(f"    x{factor:<4} inventory: {t_need*1e3:7.3f} ms "
              f"= {t_need*1e6/US_PER_DAY:9.0f} days at {US_PER_DAY} us/day")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

#!/usr/bin/env python
"""Compute a radial energy-flux time series and its autocorrelation.

Each retained snapshot is integrated over a selected flux surface using
radial_flux_profile and the ExB energy coefficient from fluxes.py. Data is
loaded in BOUT-normalised units, y-guards are removed, and restart overlaps
are deduplicated by time. Frames are processed individually to limit memory.

Outputs include F(t), its mean and standard deviation, the autocorrelation,
the integral correlation time truncated at the first zero crossing, and
N_eff = span/(2*tau_int). The estimated mean error uses sigma/sqrt(N_eff).
The required spans for N_eff = 10, 20, and 50 are also reported.

This calculates total ExB flux per frame. The ensemble mean/turbulent split
uses averages over both toroidal angle and time and is provided separately
by v3_radial_conservation.py.

Edit CONFIG and run from the case's data directory.
"""

import sys
import gc

import numpy as np
import xarray as xr
import matplotlib
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from xhermes import open_hermesdataset


# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
gridfile     = "../bout_sxd_v4_nx244_new.grd.nc"
dir_range    = range(397, 399)   # 397+398: 20 frames at 10.4 us over ~190 us
keep_yguards = True
include_y_derivs = True
x_surface    = None              # None -> ixseps1; or an int radial index
theta_band   = "closed"          # "closed" (x <= ixseps1) or "full"
save_png     = "fout_timeseries.png"
# ─────────────────────────────────────────────────────────────────────────────


def _theta_indices(ds, pdim, band):
    """closed_idx / interior_idx exactly as v3_radial_conservation.py builds them."""
    m = ds.metadata
    MYG = int(m.get("MYG", 2))
    nth = ds.sizes[pdim]
    ny = int(m.get("ny", nth))
    nyi = int(m.get("ny_inner", 0) or 0)
    # same defaults as v3_radial_conservation.py
    j11 = int(m.get("jyseps1_1", -1)); j21 = int(m.get("jyseps2_1", nth // 4))
    j12 = int(m.get("jyseps1_2", nth // 2)); j22 = int(m.get("jyseps2_2", nth - 1))
    has_g = nth > ny
    g = MYG if has_g else 0
    band_A = list(range(j11 + 1 + g, j21 + 1 + g))
    band_B = list(range(j12 + 1 + 3 * g, j22 + 1 + 3 * g))
    closed = [j for j in band_A + band_B if 0 <= j < nth]
    guard = set()
    if g:
        for a, b in ((0, g), (nyi + g, nyi + 3 * g), (ny + 3 * g, ny + 4 * g)):
            guard.update(range(a, min(b, nth)))
    interior = [j for j in range(nth) if j not in guard]
    return closed if band == "closed" else interior


def _drop_yguards(ds, pdim):
    """Mirror of the v3_* guard-drop block (xBOUT honours the GRID's flag)."""
    import netCDF4 as _ncg
    with _ncg.Dataset(gridfile) as g0:
        ny_g = int(g0["ny"][:])
    MYG = int(ds.metadata.get("MYG", 2))
    nyi = int(ds.metadata.get("ny_inner", 0) or 0)
    npol = ds.sizes[pdim]
    if nyi > 0 and npol == ny_g + 4 * MYG:
        rows = (list(range(MYG))
                + list(range(nyi + MYG, nyi + 3 * MYG))
                + list(range(ny_g + 3 * MYG, ny_g + 4 * MYG)))
    elif npol == ny_g + 2 * MYG:
        rows = list(range(MYG)) + list(range(ny_g + MYG, ny_g + 2 * MYG))
    else:
        rows = []
    if rows:
        keep = [i for i in range(npol) if i not in set(rows)]
        print(f"  dropping y-guard rows {rows}: {npol} -> {len(keep)} poloidal cells")
        ds = ds.isel({pdim: keep})
    return ds


def main():
    # ── load all folders, concatenate, de-duplicate restarts by time VALUE ───
    datasets = []
    for dirnum in dir_range:
        print(f"loading {data_root}{dirnum}")
        datasets.append(open_hermesdataset(
            f"{data_root}/{dirnum}/boutdata.nc",
            geometry="toroidal",
            inputfilepath=f"{data_root}/{dirnum}/BOUT.inp",
            gridfilepath=gridfile,
            keep_xboundaries=True,
            keep_yboundaries=keep_yguards,
            unnormalise=False,
            info=False,
        ))
    ds = xr.concat(datasets, dim="t")
    ds = h3t.dedup_restart_times(ds)

    pdim = h3t.detect_poloidal_dim(ds)
    ds = _drop_yguards(ds, pdim)

    m = ds.metadata
    Nnorm = float(m["Nnorm"]); Tnorm = float(m["Tnorm"])
    Cs0 = float(m["Cs0"]); rho_s0 = float(m["rho_s0"])
    Omega_ci = float(m["Omega_ci"])
    Pnorm = Nnorm * Tnorm * 1.602e-19
    F_q_to_SI = Pnorm * Cs0 * rho_s0 ** 2        # surface-integrated flux -> W

    ixsep1 = int(m.get("ixseps1", ds.sizes["x"] // 2))
    ix = ixsep1 if x_surface is None else int(x_surface)
    theta_slices = _theta_indices(ds, pdim, theta_band)

    nt = ds.sizes["t"]
    t_norm = np.asarray(ds["t"].values, dtype=float)
    t_us = (t_norm - t_norm[0]) / Omega_ci * 1e6
    print(f"\n{nt} frames, span {t_us[-1]:.1f} us, "
          f"theta_band='{theta_band}' ({len(theta_slices)} rows), x = {ix} "
          f"(ixseps1 = {ixsep1})")

    # ── per-frame surface integral, ONE frame in memory at a time ────────────
    F = np.full(nt, np.nan)
    for k in range(nt):
        d1 = ds.isel(t=[k])
        q_tot = h3t.decompose_radial_heat_flux(
            d1, d1["Pe"] + d1["Pi"], d1["phi"],
            include_y_derivs=include_y_derivs)[2]
        prof = h3t.radial_flux_profile(d1, q_tot, theta_slices=theta_slices)
        F[k] = float(np.asarray(prof.values)[ix]) * F_q_to_SI
        print(f"  frame {k + 1:3d}/{nt}  t = {t_us[k]:8.2f} us   "
              f"F_out = {F[k] / 1e3:9.1f} kW", flush=True)
        del d1, q_tot, prof
        gc.collect()

    # ── statistics ───────────────────────────────────────────────────────────
    dt_us = float(np.median(np.diff(t_us))) if nt > 1 else np.nan
    mean, sd = float(np.mean(F)), float(np.std(F, ddof=1))
    print("\n" + "=" * 66)
    print(f"  mean                 {mean / 1e3:9.1f} kW")
    print(f"  sigma (per frame)    {sd / 1e3:9.1f} kW   ({100 * sd / abs(mean):.1f} % of mean)")
    print(f"  naive SEM  s/sqrt(N) {sd / np.sqrt(nt) / 1e3:9.1f} kW   "
          f"<- WRONG if the series drifts")
    print(f"  dt (median)          {dt_us:9.2f} us")

    f = F - mean
    denom = float(np.dot(f, f))
    rho = np.array([float(np.dot(f[:nt - L], f[L:])) / denom
                    for L in range(nt)]) if denom > 0 else np.zeros(nt)

    # integral time: sum rho to the FIRST ZERO CROSSING (not the 1/e point)
    cut = nt
    for L in range(1, nt):
        if rho[L] <= 0.0:
            cut = L
            break
    tau_int = dt_us * (0.5 + float(np.sum(rho[1:cut]))) if cut > 1 else dt_us * 0.5
    tau_int = max(tau_int, dt_us * 0.5)
    span = t_us[-1] if nt > 1 else dt_us
    N_eff = max(1.0, span / (2.0 * tau_int))
    print("\n  autocorrelation of F_out(t):")
    for L in range(0, min(cut + 2, nt, 9)):
        print(f"    lag {L:2d} ({L * dt_us:7.2f} us)   rho = {rho[L]:+.3f}"
              + ("   <- first zero crossing" if L == cut else ""))
    print(f"\n  tau_int (slow-mode time) {tau_int:9.2f} us")
    print(f"  N_eff = span/(2 tau)     {N_eff:9.1f}   of {nt} frames")
    print(f"  HONEST error bar         {sd / np.sqrt(N_eff) / 1e3:9.1f} kW")
    print(f"\n  => {mean / 1e3:.1f} +/- {sd / np.sqrt(N_eff) / 1e3:.1f} kW")
    if cut >= nt - 1:
        print("\n  WARNING: rho never crosses zero inside this window, so "
              "tau_int is a LOWER\n           bound and N_eff an UPPER bound "
              "— the window is too short to\n           measure its own "
              "slow-mode time. Widen dir_range.")
    print("\n  span required for a given number of independent samples:")
    for n in (10, 20, 50):
        need = 2.0 * tau_int * n
        print(f"    N_eff = {n:2d}  ->  {need:8.1f} us  "
              f"= {need * 1e-6 * Omega_ci:8.0f} normalised  "
              f"(~{need / 104.0:.1f} folders at dt_dump=1000)")
    print("=" * 66)

    # ── plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].plot(t_us, F / 1e3, "o-", ms=3, lw=1)
    ax[0].axhline(mean / 1e3, color="r", ls="--", lw=1, label=f"mean {mean/1e3:.0f} kW")
    ax[0].fill_between([t_us[0], t_us[-1]],
                       (mean - sd) / 1e3, (mean + sd) / 1e3,
                       color="r", alpha=0.12, label=r"$\pm1\sigma$ per frame")
    ax[0].set_xlabel("t [µs]"); ax[0].set_ylabel(r"$F_{out}$ [kW]")
    ax[0].set_title(f"radial heat flux at x={ix} ({theta_band})")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=0.3)

    lags = np.arange(min(nt, max(8, cut + 3)))
    ax[1].plot(lags * dt_us, rho[:len(lags)], "o-", ms=3, lw=1)
    ax[1].axhline(0, color="k", lw=0.8)
    ax[1].axvline(tau_int, color="g", ls=":", lw=1.4,
                  label=rf"$\tau_{{int}}$ = {tau_int:.1f} µs")
    ax[1].set_xlabel("lag [µs]"); ax[1].set_ylabel(r"$\rho$")
    ax[1].set_title("autocorrelation"); ax[1].legend(fontsize=8)
    ax[1].grid(alpha=0.3)
    fig.tight_layout()
    if save_png:
        fig.savefig(save_png, dpi=140)
        print(f"wrote {save_png}")
    plt.show()


if __name__ == "__main__":
    main()

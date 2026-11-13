#!/usr/bin/env python3
"""Report cross-field transport at the LCFS for each main-SOL band.

Outputs particle and energy fluxes split into mean ExB, turbulent ExB,
magnetic drift, and a prescribed-diffusivity reference. Effective D_turb
and chi_turb compare turbulent transport with that reference on the same
profile. Relative density and pressure fluctuations are reported separately.
The displayed chi_turb percentage uses CHI_BAR_LFS as a configurable reference.

Usage:
    lcfs_transport_monitor.py GRID DUMP [DUMP ...] [SPECIES]

Each DUMP is a squashed boutdata.nc. Its folder should contain the associated
BOUT.inp so the prescribed transport coefficients can be read. Run near the
data to avoid transferring full dumps over a network mount.
"""
import os
import sys

import netCDF4 as nc
import numpy as np
from xhermes import open_hermesdataset

import hermes3_turb as h3t

# from_archer_6kW/428, LFS, species e -- see the table in the docstring.
CHI_BAR_LFS = 1.793


def drop_y_guards(ds, grid, pdim):
    """Remove y-boundary guards before evaluating poloidal derivatives.

    The guard count is inferred from the dump size and the grid's interior ny,
    including the inter-segment band in double-null geometry.
    """
    with nc.Dataset(grid) as g:
        ny_grid = int(g["ny"][:])
    npol = ds.sizes[pdim]
    guards = h3t.guard_rows(ds, ny_grid=ny_grid)
    if not guards:
        return ds
    keep = [i for i in range(npol) if i not in set(guards)]
    print(f"  dropped {len(guards)} y-guard rows")
    return ds.isel({pdim: keep})


def report(dump, grid, species="e"):
    inp = os.path.join(os.path.dirname(dump), "BOUT.inp")
    ds = open_hermesdataset(dump, geometry="toroidal", gridfilepath=grid,
                            inputfilepath=inp, keep_xboundaries=True,
                            keep_yboundaries=False, unnormalise=False,
                            info=False)
    pdim = h3t.detect_poloidal_dim(ds)
    zdim = h3t.detect_toroidal_dim(ds)
    ds = drop_y_guards(ds, grid, pdim)
    gv = h3t.load_grid_vars(grid, ds=ds, pdim=pdim)

    print(f"  nx={ds.sizes['x']} ny={ds.sizes[pdim]} nz={ds.sizes[zdim]} "
          f"nt={ds.sizes.get('t', 1)}  ixseps1={int(ds.metadata['ixseps1'])}")
    if ds.sizes.get("t", 1) < 3:
        print("  NOTE: <3 frames -- a turbulent flux needs ~80 independent "
              "frames for a reliable flux estimate; this sample is indicative.")

    res = h3t.lcfs_transport(ds, species=species, grid_vars=gv)
    rows = []
    for band, r in res.items():
        y0, y1 = r["rows"]
        print(f"\n  === LCFS x={r['x_idx']}, {band}, rows {y0}:{y1} "
              f"(reference D={r['D_ref']:.3g} chi={r['chi_ref']:.3g} m^2/s) ===")
        print(f"  {'channel':22s}{'particles [/s]':>16}{'heat [kW]':>14}")
        for ch in r["heat"]:
            print(f"  {ch:22s}{r['particles'][ch]:>16.3e}"
                  f"{r['heat'][ch] / 1e3:>14.2f}")
        print(f"  -> D_turb   = {r['D_turb']:+.4f} m^2/s")
        bar = ""
        if "LFS" in band:
            bar = f"   ({100 * r['chi_turb'] / CHI_BAR_LFS:.0f}% of the bar)"
        print(f"  -> chi_turb = {r['chi_turb']:+.4f} m^2/s{bar}")
        rows.append((band, r["chi_turb"], r["heat"]["ExB turbulent"] / 1e3))

    # Fluctuation levels on the same bands, at the separatrix.
    ix1 = int(ds.metadata["ixseps1"])
    print()
    for name in ("Ne", f"P{species}"):
        if name not in ds:
            continue
        rf = h3t.relative_fluctuation(ds[name]) * 100.0
        rf = rf.mean("t") if "t" in rf.dims else rf
        for band, (y0, y1) in h3t.sol_band_regions(ds).items():
            prof = rf.isel(x=ix1).isel({pdim: slice(y0, y1)}).values
            print(f"  {name:4s} {band:18s} x=ixseps1: mean {prof.mean():5.1f}% "
                  f"max {prof.max():5.1f}% (row {y0 + int(np.argmax(prof))})")
    return rows


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    grid, dumps = sys.argv[1], sys.argv[2:]
    species = "e"
    if not dumps[-1].endswith(".nc"):
        species = dumps.pop()

    summary = []
    for dump in dumps:
        print(f"\n########## {dump}")
        try:
            for band, chi, q in report(dump, grid, species):
                summary.append((dump, band, chi, q))
        except Exception as exc:                    # keep the scan going
            print(f"  FAILED: {type(exc).__name__}: {exc}")

    if len(summary) > 2:
        print(f"\n{'dump':44s}{'band':20s}{'chi_turb':>10}{'q_turb[kW]':>12}")
        for dump, band, chi, q in summary:
            print(f"{dump[-44:]:44s}{band:20s}{chi:>10.3f}{q:>12.1f}")

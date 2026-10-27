"""
source_radiation_audit.py
-------------------------
Audit the particle / energy budget of a Hermes-3 run with neutrals and show
where things sit on the poloidal (R, Z) plane.

Three parts:

  A. SOURCE COMPARISON  - external gas puff vs target recycling vs wall
                          recycling vs pump vs net ionisation. Tells you what
                          actually fuels the plasma (usually recycling, not the
                          puff).
  B. REACTION/RADIATION AUDIT - volume-integrate every R*/E*/S* diagnostic,
                          reported BOTH including guard cells and over the
                          physical interior. The two columns expose the
                          guard-cell artifact: with `extrapolate_y=false` the
                          radial guard cells carry garbage Jacobian/rate values
                          and can fake tens of MW of radiation. ALWAYS trust the
                          'interior' column.
  C. POLOIDAL MAPS      - (R, Z) maps of chosen fields (z-averaged for 3D) via
                          the xBOUT accessor (da.bout.polygon / .pcolormesh)
                          with OMP / IMP / target / separatrix markers.

Design notes
------------
* The dump is opened ONCE (open_dataset) and the single dataset is passed to
  every part. Opening the same multi-GB file twice (e.g. netCDF4 + xbout) can
  segfault via clashing HDF5 handles - so we never do that.
* open_dataset prefers xhermes (geometry -> .bout accessor works), falls back to
  xbout, and finally to a plain xarray open (A/B still work; C is skipped with a
  message because it needs the toroidal geometry).
* Units come from each variable's stored `conversion` attribute (Pa/s, W/m^3 or
  m^-3 s^-1). Pa/s variables (P*_src) get the extra 3/2 to become power [W].
* Full-torus integral:  sum(f * J*dx*dy) * rho_s0^3 * (2*pi / nz)
  which reduces to *2*pi for 2D (nz=1) and completes the wedge for 3D.

Usage
-----
    python source_radiation_audit.py [casedir] [gridfile]

or edit the CONFIG block below.
"""

from __future__ import annotations

import os
import sys
import warnings
import numpy as np
import matplotlib.pyplot as plt

# Rxy/Zxy are non-monotonic (poloidal) - pcolormesh warns but renders fine
warnings.filterwarnings("ignore", message=".*not monotonically increasing.*")

# --------------------------------------------------------------------------- #
# CONFIG
# --------------------------------------------------------------------------- #
CONFIG = {
    "casedir":  "./99",         # dir holding the dump
    "dumpname": "boutdata.nc",
    "gridfile": "../exp45470_x132_y28_D1.5e21.nc",
    "tind":     -1,            # time index to analyse
    "ion":      "d+",          # main ion species
    "neutral":  "d",           # neutral species
    "keep_yboundaries": False,  # dump retains y-guards (squash keep_yboundaries)
    # fields to draw on the poloidal plane. Real dump variables, plus the
    # derived token "plasma_source" = S<ion>_iz + S<ion>_rec (net ionisation
    # minus recombination) - shows WHERE the plasma is fuelled (separatrix vs
    # target).
    "map_fields": ["Ne", "Te", "Td+", "Nd", "plasma_source"],
    "map_how":   "polygon",    # "polygon" (true cell geometry) or "pcolormesh"
    # Colour range per panel. Default is robust percentile clipping of the
    # interior (excludes guard outliers). Override a field with an explicit
    # (vmin, vmax) - units are the PHYSICAL units shown on the colorbar, e.g.
    #   "color_limits": {"plasma_source": (0, 5e22), "Te": (0, 60)}
    # Set a field's log scale with map_log, e.g.  "map_log": ["plasma_source"].
    "color_limits":    {"plasma_source": (0, 1.3e22)},
    "clip_percentile": (2, 98),   # (low, high) percentiles for auto range
    "map_log":         [],        # fields to draw on a log colour scale
    "show_plots": True,
}

QE = 1.602176634e-19
MP = 1.67262192369e-27


# --------------------------------------------------------------------------- #
# Loading (open ONCE, reuse everywhere)
# --------------------------------------------------------------------------- #
def open_dataset(cfg):
    """Open the dump exactly once. Returns (ds, has_geometry).

    Prefers xhermes (gives the toroidal geometry needed by .bout.polygon),
    falls back to xbout, then to a plain xarray open (A/B still work, C skips).
    Opened normalised (unnormalise=False) so the stored `conversion` attributes
    are intact and used uniformly for the integrals.
    """
    path = os.path.join(os.path.abspath(cfg["casedir"]), cfg["dumpname"])
    grid = os.path.abspath(cfg["gridfile"])
    keep_y = bool(cfg["keep_yboundaries"])

    try:
        import xhermes
        ds = xhermes.open_hermesdataset(
            path, geometry="toroidal", gridfilepath=grid,
            keep_xboundaries=True, keep_yboundaries=keep_y,
            unnormalise=False, info=False,
        )
        return ds, True
    except Exception as exc:                                   # noqa: BLE001
        print(f"[load] xhermes open failed ({type(exc).__name__}: {exc})")

    try:
        import xbout
        ds = xbout.open_boutdataset(
            path, gridfilepath=grid, geometry="toroidal",
            keep_xboundaries=True, keep_yboundaries=keep_y, info=False,
        )
        return ds, True
    except Exception as exc:                                   # noqa: BLE001
        print(f"[load] xbout geometry open failed ({type(exc).__name__}: {exc})")

    import xarray as xr
    print("[load] falling back to plain xarray (no geometry -> maps skipped)")
    return xr.open_dataset(path), False


def meta(ds, key, default=None):
    """Read a scalar from ds.metadata, ds[var], or ds.attrs (any open style)."""
    md = getattr(ds, "metadata", {}) or {}
    if key in md:
        return float(np.asarray(md[key]).ravel()[0])
    if key in ds.variables:
        return float(np.asarray(ds[key].values).ravel()[0])
    if key in ds.attrs:
        return float(np.asarray(ds.attrs[key]).ravel()[0])
    return default


def dims_of(ds):
    pol = "theta" if "theta" in ds.dims else "y"
    tor = "zeta" if "zeta" in ds.dims else ("z" if "z" in ds.dims else None)
    return pol, tor


# --------------------------------------------------------------------------- #
# Geometry / normalisation / guard masks (no R,Z needed - that's only for maps)
# --------------------------------------------------------------------------- #
def geometry(ds, cfg):
    geo = {}
    pol, tor = dims_of(ds)
    geo["pol"], geo["tor"] = pol, tor

    Nnorm = meta(ds, "Nnorm"); Tnorm = meta(ds, "Tnorm"); Bnorm = meta(ds, "Bnorm")
    Omega_ci = QE * Bnorm / MP
    rho_s0 = np.sqrt(QE * Tnorm / MP) / Omega_ci
    geo.update(Nnorm=Nnorm, Tnorm=Tnorm, Bnorm=Bnorm, rho_s0=rho_s0)

    J  = ds["J"].transpose("x", pol).values
    dx = ds["dx"].transpose("x", pol).values
    dy = ds["dy"].transpose("x", pol).values
    nx, ny = J.shape
    nz = ds.sizes[tor] if tor else 1
    geo.update(J=J, nx=nx, ny=ny, nz=nz)

    MXG = int(meta(ds, "MXG", 2)); MYG = int(meta(ds, "MYG", 2))
    ny_inner = int(meta(ds, "ny_inner", 0) or 0)
    j1_1 = int(meta(ds, "jyseps1_1", -1)); j2_1 = int(meta(ds, "jyseps2_1", 0))
    j1_2 = int(meta(ds, "jyseps1_2", 0)); j2_2 = int(meta(ds, "jyseps2_2", ny - 1))
    ixseps1 = int(meta(ds, "ixseps1", nx // 2))
    double_null = (ny_inner > 0) and (j2_1 != j1_2)

    # y-boundary guards are present only if the loaded poloidal size exceeds the
    # no-boundary global ny (BOUT convention: metadata 'ny' excludes y-guards,
    # while 'nx' includes x-guards). With keep_yboundaries=False there are none,
    # so we must NOT carve target/X-point cells out of the domain.
    ny_nogc = int(meta(ds, "ny", ny))
    gy = MYG if ny > ny_nogc else 0
    geo.update(MXG=MXG, MYG=MYG, ny_inner=ny_inner, gy=gy,
               ixseps1=ixseps1, double_null=double_null)

    # ── Midplane rows from the TOPOLOGY, never from argmax(R) ────────────────
    # This used to be `yidx[argmax(Rsep[yidx])]`, i.e. the poloidal row of
    # largest major radius on the separatrix. That is wrong on MAST-U:
    # `valid_y` excludes only the guard rows, so the search spans the divertor
    # legs and the X-point regions, where a Super-X reaches a LARGER R than the
    # outboard midplane does. Measured on the SXD nx244 grid it returned
    # theta=46 at R=1.4293, Z=+1.9713 m — essentially the top of the machine —
    # against the true OMP at theta=81, R=1.3821, Z=+0.0599 m.
    # Same region offsets as target.get_midplane_y_indices (A: +g, B: +3g);
    # dropping the +3*g is the recurring bug in this codebase (see the `ddy`
    # mis-registration and the v3_radial_flux lambda-arc offset).
    y_omp = (j1_2 + 3 * gy + j2_2 + 3 * gy) // 2
    y_imp = (j1_1 + gy + j2_1 + gy) // 2 if double_null else None
    geo.update(y_omp=y_omp, y_imp=y_imp)

    # full-torus physical volume weight per (x, y)
    geo["dVxy"] = J * dx * dy * rho_s0**3 * (2 * np.pi / nz)

    valid_x = np.zeros(nx, bool); valid_x[MXG:nx - MXG] = True
    valid_y = np.ones(ny, bool)
    if gy:                                   # guards present -> exclude them
        valid_y[:] = False
        if double_null:
            valid_y[gy:ny_inner + gy] = True             # lower region
            valid_y[ny_inner + 3 * gy:ny - gy] = True     # upper region
        else:
            valid_y[gy:ny - gy] = True
    geo["valid_x"], geo["valid_y"] = valid_x, valid_y
    geo["mask_xy"] = np.outer(valid_x, valid_y)
    return geo


def integrate(ds, geo, name, interior=True, tind=-1):
    """Volume integral of a diagnostic in SI (power [W] or rate [s^-1])."""
    if name not in ds.variables:
        return None
    var = ds[name]
    conv = float(var.attrs.get("conversion", 1.0))
    units = str(var.attrs.get("units", ""))
    da = var.isel(t=tind) if "t" in var.dims else var
    order = ["x", geo["pol"]] + ([geo["tor"]] if geo["tor"] in da.dims else [])
    a = da.transpose(*order).values
    if a.ndim == 2:
        a = a[:, :, None]
    w = geo["dVxy"][:, :, None] * conv
    m = geo["mask_xy"][:, :, None] if interior else 1.0
    total = float((a * w * m).sum())
    if "Pa" in units:           # Pa/s -> W needs 3/2 (energy = 3/2 pressure)
        total *= 1.5
    return total


# --------------------------------------------------------------------------- #
# PART A : source comparison
# --------------------------------------------------------------------------- #
def part_a_sources(ds, geo, cfg):
    ion, neut = cfg["ion"], cfg["neutral"]
    print("\n" + "=" * 64)
    print("A. PARTICLE SOURCE COMPARISON  (interior, full torus) [s^-1]")
    print("=" * 64)
    rows = [
        (f"external gas puff   S{neut}_src",            f"S{neut}_src"),
        (f"target recycling    S{neut}_target_recycle", f"S{neut}_target_recycle"),
        (f"wall recycling      S{neut}_wall_recycle",   f"S{neut}_wall_recycle"),
        (f"pump (sink)         S{neut}_pump",           f"S{neut}_pump"),
        (f"core e source       Se_src",                 "Se_src"),
        (f"ionisation source   S{ion}_iz",              f"S{ion}_iz"),
        (f"recombination sink  S{ion}_rec",             f"S{ion}_rec"),
    ]
    vals = {}
    for label, var in rows:
        v = integrate(ds, geo, var, interior=True, tind=cfg["tind"])
        vals[var] = v
        if v is not None:
            print(f"  {label:38s} {v:+.3e}")
    puff = vals.get(f"S{neut}_src") or 0.0
    recyc = vals.get(f"S{neut}_target_recycle") or 0.0
    if (puff + recyc) > 0:
        print("-" * 64)
        print(f"  puff / (puff + target-recycle)   = {100*puff/(puff+recyc):5.1f} %")
        print(f"  target-recycle amplification     = {recyc/max(puff,1e-30):5.1f} x puff")
    return vals


# --------------------------------------------------------------------------- #
# PART B : reaction / radiation audit (guard artifact exposed)
# --------------------------------------------------------------------------- #
def part_b_reactions(ds, geo, cfg):
    ion, neut = cfg["ion"], cfg["neutral"]
    print("\n" + "=" * 78)
    print("B. ENERGY / REACTION AUDIT   [kW power]")
    print("   with-guards vs interior  ->  big mismatch = guard-cell artifact")
    print("=" * 78)
    energy_vars = [
        "Pe_src", f"P{ion}_src",
        f"R{ion}_ex", f"R{ion}_rec",
        f"E{ion}_iz", f"E{ion}_rec", f"E{neut}{ion}_cx",
        f"E{ion}e_coll", f"Ee{ion}_coll",
        f"E{neut}_target_refl", f"E{neut}_wall_refl", f"E{neut}_pump",
    ]
    print(f"  {'variable':22s} {'with-guards':>14s} {'interior':>14s}   units")
    print("  " + "-" * 70)
    for v in energy_vars:
        if v not in ds.variables:
            continue
        units = str(ds[v].attrs.get("units", ""))
        wg = integrate(ds, geo, v, interior=False, tind=cfg["tind"])
        it = integrate(ds, geo, v, interior=True, tind=cfg["tind"])
        flag = ""
        if wg and abs(wg) > 50e3 and abs(it) / (abs(wg) + 1e-30) < 0.5:
            flag = "  <== guard artifact"
        print(f"  {v:22s} {wg/1e3:14.1f} {it/1e3:14.1f}   {units}{flag}")
    print("  (kW; positive = source to plasma, negative = sink)")


# --------------------------------------------------------------------------- #
# PART C : poloidal (R, Z) maps via the xBOUT .bout accessor
# --------------------------------------------------------------------------- #
def build_map_field(ds, fld, cfg, geo):
    """Return (DataArray scaled to physical units, label) for a map field.

    Supports real dump variables and the derived token "plasma_source"
    = S<ion>_iz + S<ion>_rec  (S<ion>_rec is the signed recombination sink,
    so this is net ionisation minus recombination). Returns (None, None) if
    unavailable.
    """
    import xarray as xr
    ion, tor = cfg["ion"], geo["tor"]
    with xr.set_options(keep_attrs=True):
        if fld == "plasma_source":
            iz, rec = f"S{ion}_iz", f"S{ion}_rec"
            if iz not in ds or rec not in ds:
                return None, None
            da = ds[iz] + ds[rec]
            scale = float(ds[iz].attrs.get("conversion", 1.0))
            label = f"net plasma source\n(S{ion}_iz + S{ion}_rec) [m$^{{-3}}$ s$^{{-1}}$]"
        elif fld in ds:
            da = ds[fld]
            if fld.startswith("N"):
                scale, unit = geo["Nnorm"], "m$^{-3}$"
            elif fld.startswith("T"):
                scale, unit = geo["Tnorm"], "eV"
            elif fld.startswith(("R", "E")):
                scale = float(ds[fld].attrs.get("conversion", 1.0)); unit = "W m$^{-3}$"
            else:
                scale, unit = 1.0, "norm"
            label = f"{fld} [{unit}]"
        else:
            return None, None
        if "t" in da.dims:
            da = da.isel(t=cfg["tind"])
        if tor and da.sizes.get(tor, 1) > 1:
            da = da.mean(tor)                       # z-average for 3D
        elif tor:
            da = da.isel({tor: 0})
        da = da * scale
    return da, label


def part_c_maps(ds, geo, cfg, has_geometry, how="polygon"):
    if not has_geometry:
        print("\n[C] dataset has no toroidal geometry (.bout accessor) - "
              "maps skipped. Run where xhermes/xbout can load the geometry.")
        return
    fields = [f for f in cfg["map_fields"]
              if f == "plasma_source" or f in ds]
    if not fields:
        print("\n[C] none of map_fields present, skipping plots")
        return

    pol, tor = geo["pol"], geo["tor"]
    Rname = "R" if "R" in ds.coords or "R" in ds else "Rxy"
    Zname = "Z" if "Z" in ds.coords or "Z" in ds else "Zxy"
    Rxy = ds[Rname].transpose("x", pol).values
    Zxy = ds[Zname].transpose("x", pol).values

    ix1 = geo["ixseps1"]
    yidx = np.where(geo["valid_y"])[0]
    y_omp = geo["y_omp"]                 # from the topology, see build_geometry
    y_imp = geo["y_imp"]
    # Cheap guard against a future topology surprise: the midplane must sit near
    # Z of the magnetic axis, i.e. near the middle of the separatrix Z range.
    _zs = Zxy[ix1, yidx]
    _zmid = 0.5 * (_zs.max() + _zs.min())
    _zhalf = 0.5 * (_zs.max() - _zs.min())
    for _nm, _y in (("OMP", y_omp), ("IMP", y_imp)):
        if _y is None:
            continue
        _off = abs(Zxy[ix1, _y] - _zmid) / max(_zhalf, 1e-12)
        print(f"  {_nm} marker: theta={_y}  R={Rxy[ix1, _y]:.4f}  "
              f"Z={Zxy[ix1, _y]:+.4f} m  ({_off:.3f} of the separatrix "
              "half-height off the midplane)")
        if _off > 0.5:
            print(f"  WARNING: that {_nm} row is nowhere near the midplane — "
                  "check jyseps / y_boundary_guards for this grid.")
    regions, start = [], yidx[0]                       # split at branch cut
    for a, b in zip(yidx[:-1], yidx[1:]):
        if b != a + 1:
            regions.append((start, a + 1)); start = b
    regions.append((start, yidx[-1] + 1))

    Nnorm, Tnorm = geo["Nnorm"], geo["Tnorm"]
    ncol = min(3, len(fields)); nrow = int(np.ceil(len(fields) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(5 * ncol, 5.4 * nrow),
                             squeeze=False)
    import matplotlib.colors as mcolors
    for ax, fld in zip(axes.flat, fields):
        da, label = build_map_field(ds, fld, cfg, geo)
        if da is None:
            ax.axis("off"); continue

        # colour range: explicit override, else robust percentile of interior
        kw = {"cmap": "inferno"}
        kw["separatrix"] = False
        clim = cfg.get("color_limits", {}).get(fld)
        if clim is not None:
            vmin, vmax = clim
        else:
            arr = da.transpose("x", pol).values
            sel = arr[geo["mask_xy"]] if geo["mask_xy"].shape == arr.shape else arr
            sel = sel[np.isfinite(sel)]
            plo, phi = cfg.get("clip_percentile", (2, 98))
            vmin, vmax = np.nanpercentile(sel, plo), np.nanpercentile(sel, phi)
        if fld in cfg.get("map_log", []) and vmax > 0:
            kw["norm"] = mcolors.LogNorm(vmin=max(vmin, vmax * 1e-4), vmax=vmax)
        else:
            kw["vmin"], kw["vmax"] = vmin, vmax

        getattr(da.bout, how)(ax=ax, **kw)              # <-- xBOUT plotting
        ax.set_title(label, fontsize=9); ax.set_aspect("equal")
    if axes.flat[0].get_legend_handles_labels()[0]:   # only if markers enabled
        axes.flat[0].legend(loc="upper right", fontsize=8)
    fig.suptitle("Poloidal maps")
    fig.tight_layout()


# --------------------------------------------------------------------------- #
# PART E : where does the pump remove particles?
# --------------------------------------------------------------------------- #
def _xy_field(ds, names, pol):
    """First of *names* present in ds, as a (x, pol) numpy array, else None."""
    for n in names:
        if n in ds:
            return ds[n].transpose("x", pol).values
    return None


def part_e_pump(ds, geo, cfg):
    neut = cfg["neutral"]
    pumpvar = f"S{neut}_pump"
    if pumpvar not in ds:
        return
    pol, tor = geo["pol"], geo["tor"]

    ispump = _xy_field(ds, ["is_pump"], pol)
    Rxy = _xy_field(ds, ["R", "Rxy"], pol)
    Zxy = _xy_field(ds, ["Z", "Zxy"], pol)
    if ispump is None or Rxy is None or Zxy is None:   # fall back to grid file
        import xarray as xr
        g = xr.open_dataset(os.path.abspath(cfg["gridfile"]))
        if ispump is None and "is_pump" in g: ispump = g["is_pump"].values
        if Rxy is None: Rxy = g["Rxy"].values
        if Zxy is None: Zxy = g["Zxy"].values
        g.close()
    if ispump is None:
        print(f"\n[E] no 'is_pump' field found - pump-location skipped")
        return

    conv = float(ds[pumpvar].attrs.get("conversion", 1.0))
    da = ds[pumpvar].isel(t=cfg["tind"]) if "t" in ds[pumpvar].dims else ds[pumpvar]
    order = ["x", pol] + ([tor] if tor in da.dims else [])
    sp = da.transpose(*order).values
    if sp.ndim == 2:
        sp = sp[:, :, None]
    pump_xy = (sp * geo["dVxy"][:, :, None] * conv).sum(axis=2)   # /s per (x,y)

    m = ispump > 0.5
    tot = pump_xy[m].sum()
    print("\n" + "=" * 64)
    print(f"E. PUMP REMOVAL BY LOCATION   total {pumpvar} = {tot:.3e} /s")
    print("=" * 64)
    if abs(tot) < 1e-30:
        print("  pump inactive (is_pump empty or neutral_pump off)"); return
    div = pump_xy[m & (np.abs(Zxy) > 1.0)].sum()
    print(f"  divertor (|Z|>1.0): {div:.2e}/s ({100*div/tot:4.0f}%)   "
          f"main chamber: {tot-div:.2e}/s ({100*(tot-div)/tot:4.0f}%)")
    xs, ys = np.where(m)
    print(f"  {'x':>3}{'y':>3}{'R':>7}{'Z':>7}{'rate[/s]':>12}")
    for x, y in sorted(zip(xs, ys), key=lambda t: -abs(pump_xy[t]))[:8]:
        print(f"  {x:3d}{y:3d}{Rxy[x,y]:7.2f}{Zxy[x,y]:+7.2f}{pump_xy[x,y]:12.3e}")


# --------------------------------------------------------------------------- #
def main():
    cfg = dict(CONFIG)
    if len(sys.argv) > 1:
        cfg["casedir"] = sys.argv[1]
    if len(sys.argv) > 2:
        cfg["gridfile"] = sys.argv[2]

    ds, has_geometry = open_dataset(cfg)                 # <-- single open
    geo = geometry(ds, cfg)
    print(f"case: {os.path.abspath(cfg['casedir'])}")
    print(f"grid: nx={geo['nx']} ny={geo['ny']} nz={geo['nz']}  "
          f"{'double-null' if geo['double_null'] else 'single-null'}  "
          f"ixseps1={geo['ixseps1']}  geometry={'yes' if has_geometry else 'no'}")

    pe = integrate(ds, geo, "Pe_src", interior=True, tind=cfg["tind"])
    if pe is not None:
        print(f"calibration: interior Pe_src = {pe/1e3:.1f} kW "
              f"(should match [Pe] heating in BOUT.inp)")

    part_a_sources(ds, geo, cfg)
    part_b_reactions(ds, geo, cfg)
    part_e_pump(ds, geo, cfg)
    sys.stdout.flush()                                   # audit out before plots
    if cfg["show_plots"]:
        part_c_maps(ds, geo, cfg, has_geometry, how=cfg["map_how"])
        out = os.environ.get("AUDIT_SAVEFIG")
        if out and plt.get_fignums():
            plt.savefig(out, dpi=110, bbox_inches="tight")
            print(f"\nsaved figure -> {out}")
        elif plt.get_fignums():
            plt.show()


if __name__ == "__main__":
    main()

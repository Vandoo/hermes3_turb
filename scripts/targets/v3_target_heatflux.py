#!/usr/bin/env python3
"""
v3_target_heatflux.py — divertor target heat flux, λ_q (Eich + double-exponential)
and total deposited power for a Hermes-3 3D turbulence run.

Same physics as scripts/target_heatflux.py, but follows the v3_* convention of
loading MULTIPLE output folders and concatenating them in time, so the fits use
a properly time-averaged saturated window (important — a single near-crash frame
gives anomalously narrow λ_q).

Caching
-------
The slow part is loading the guard-retaining dumps and evaluating
target_heat_flux_density at every target — fits and plotting are cheap by
comparison. The per-target (s_mm, q_MWm2) arrays + total power + input
source power are pickled to ./cache/v3_target_heatflux_*.pkl. On rerun, the
script reloads them and applies whatever you change in CONFIG below
(fit_s_range_mm, fit_coordinate, fit_s0, …) in seconds. Set use_cache = False
to force recomputation.

Loading notes
-------------
- The target heat flux is evaluated on the plate (cell boundary between the last
  plasma cell and the first y-guard), so the dump MUST retain the
  (Hermes-3-extrapolated) y-boundary guard cells (squash keep_yboundaries=True).
- Loaded WITHOUT geometry="toroidal"/gridfile: a guardless Hypnotoad grid
  conflicts with a guard-retaining dump (dimension-'y' size clash). All metric
  and Bp/B come from the dump (Bp/B via the metric identity).
- unnormalise=True (SI), so Pe[Pa], Te[eV], metric in SI — what target.py wants.

λ_q is fit in OMP/IMP-mapped coordinates by default (Eich convention). Edit the
CONFIG block.
"""

import os
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xhermes import open_hermesdataset

import hermes3_turb as h3t
from hermes3_turb.load import _no_t, get_parallel_velocity
from hermes3_turb.target import (
    get_target_y_indices,
    get_target_boundaries,
    get_midplane_y_indices,
    target_heat_flux_density,
    target_heat_flux_from_diagnostics,
    target_sheath_power_from_diagnostics,
    target_physical_distance,
    field_line_incidence,
    fit_eich,
    fit_double_exp,
    total_target_power,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
gridfile     = None                # path to the grid .nc, read with netCDF4 ONLY: to
                                   # recover a P<sp>_src the dump does not save (see
                                   # the POWER BUDGET block) and, for fit_coordinate
                                   # "omp", to map flux surfaces to the TRUE midplane
                                   # by ψ (embedded geqdsk). Never handed to xBOUT.
dir_range    = range(140, 141)     # output folders to load & concat in time
ion_species  = "i"
AA_ion       = 2.0                 # ion mass number (deuterium)
Z_ion        = 1.0                 # ion charge number (hydrogenic = 1)
phi_wall     = 0.0                 # [V] wall potential (Hermes-3 default 0)
prefer_diagnostics = True          # try target_heat_flux_from_diagnostics first.
                                   # NOTE: the names it looks for are not what
                                   # Hermes-3 writes, so in practice this always
                                   # falls through to the γ·P·V formula. The real
                                   # diagnostic (E<sp>_sheath) is read separately
                                   # by target_sheath_power_from_diagnostics.
use_phi_gamma = True               # sheath γ: True  = φ-dependent, eqs. 12a/12b
                                   #   (γ_e = 2 + max(φ−φ_wall,0)/T_e, γ_i = 2.5 +
                                   #   ½m_i v_∥²/eT_i) — matches `sheath_boundary`.
                                   # False = FIXED γ_e = γ_i = 3.5 — matches
                                   #   `sheath_boundary_simple` defaults, so use
                                   #   False to validate against E<sp>_sheath, and
                                   #   for noisy/crash frames where φ is unreliable.
                                   # Flows to target_heat_flux_density(phi=...) in
                                   # _q_for(); it is in the cache key, so flipping
                                   # it cannot give you a stale cached answer.
bohm_cap_v   = False               # DEPRECATED (no-op). |V_∥|≤c_s clips real
                                   # supersonic flow that sheath_boundary_simple
                                   # allows. Keep False.
fit_coordinate = "omp"             # "omp" (Eich convention) | "target" | "wetted"
fit_s_range_mm = (-5, 30)          # fit window [mm] in the fit coordinate
targets_to_plot = None             # None = all targets in the topology
include_double_exp = True
target_band  = True                # shade ±1σ on the target profiles, σ over
                                   # TIME of the per-snapshot q (same convention
                                   # as the OMP band in v3_radial_flux.py)
target_band_n = 1.0                # band half-width in units of σ
target_mean = "per_snapshot"       # "per_snapshot" = plot ⟨q(P,V)⟩_t, the mean of
                                   #   the per-frame heat fluxes — correct, and the
                                   #   centre of the band.
                                   # "avg_fields"   = q(⟨P⟩,⟨V⟩), the old behaviour.
                                   #   q is NONLINEAR in the fields (γ·P·V, and γ_e
                                   #   depends on φ/T_e), so the two differ; the run
                                   #   prints by how much.
fit_s0       = False               # strike-point shift in the fits [mm]:
                                   #   False (default) = peak at separatrix (s0 = 0).
                                   #   True            = s0 added as a free fit
                                   #                     parameter — ONLY use after
                                   #                     visually confirming the
                                   #                     q-peak sits inside the SOL.
                                   #                     Otherwise the fit can find
                                   #                     a spurious small shift that
                                   #                     artificially narrows λ_q.
                                   #   float           = fix s0 at that value [mm]
                                   #                     (best when you know the
                                   #                     strike-point offset).
use_cache    = True
# ──────────────────────────────────────────────────────────────────────────────
dir_range = h3t.existing_dirs(data_root, dir_range, ("boutdata_guards.nc", "boutdata_yguards.nc",
                                                     "boutdata.nc", "BOUT.dmp.*.nc"))


def _compute():
    """Load + concatenate dumps, evaluate target heat flux per target, return
    a flat dict for pickling."""
    datasets, shapes = [], []
    for i, dirnum in enumerate(dir_range):
        folder = os.path.join(data_root, str(dirnum))
        datapath = next((c for c in (os.path.join(folder, "boutdata_guards.nc"),
                                     os.path.join(folder, "boutdata_yguards.nc"),
                                     os.path.join(folder, "boutdata.nc"))
                         if os.path.exists(c)), os.path.join(folder, "BOUT.dmp.*.nc"))
        settings = next((s for s in (os.path.join(data_root, "BOUT.settings"),
                                     os.path.join(folder, "BOUT.settings"),
                                     os.path.join(folder, "BOUT.inp"))
                         if os.path.exists(s)), None)
        print(f"loading {datapath}")
        dsi = open_hermesdataset(datapath, inputfilepath=settings,
                                 keep_yboundaries=True, unnormalise=True)
        shapes.append((datapath, dsi.sizes[h3t.detect_poloidal_dim(dsi)]))
        # De-duplicate the restart overlap by TIME VALUE (running max), not by
        # dropping each folder's first frame: a folder squashed to a single
        # output time would otherwise contribute nothing at all.
        if datasets and "t" in dsi.dims:
            t_prev = max(float(d["t"].max()) for d in datasets)
            keep = np.nonzero(dsi["t"].values > t_prev)[0]
            if len(keep) == 0:
                print(f"  no frames past t={t_prev:g} — folder skipped")
                continue
            if len(keep) < dsi.sizes["t"]:
                print(f"  dropping {dsi.sizes['t'] - len(keep)} overlapping frame(s)")
            dsi = dsi.isel(t=keep)
        datasets.append(dsi)
    # A folder squashed WITH y-boundary guards and one squashed without have
    # different poloidal sizes; xr.concat's default join="outer" pads the short
    # one with NaN rather than failing, so refuse the mix explicitly.
    if len({n for _, n in shapes}) > 1:
        raise SystemExit(
            "folders disagree on poloidal size — some dumps were squashed with "
            "y-boundary guard cells and some without:\n  "
            + "\n  ".join(f"{n:4d} rows   {p}" for p, n in shapes)
            + "\n\nRe-squash them all the same way (keep_yboundaries=True), or "
              "narrow dir_range to the folders that already match.")
    ds = xr.concat(datasets, dim="t") if len(datasets) > 1 else datasets[0]

    pdim    = h3t.detect_poloidal_dim(ds)
    zdim    = h3t.detect_toroidal_dim(ds)
    m       = ds.metadata
    ixseps1 = int(m.get("ixseps1", ds.sizes["x"] // 2))

    _bnd, has_guards = get_target_boundaries(ds)
    if has_guards:
        print(f"y-boundary guard cells present (ny={ds.sizes[pdim]}) — "
              "profiles interpolated to the target plate.")
    else:
        print("WARNING: no y-boundary guard cells — target values fall back to "
              "the last domain cell. Re-squash with keep_yboundaries=True.")

    print(f"Time-averaging over all {ds.sizes['t']} loaded frames")
    ds_avg = ds.mean("t")
    # xr.concat gives a t dimension to any COORDINATE whose values differ
    # between folders — dx/dy/dz do, if only by float noise — and
    # Dataset.reduce drops coordinates that carry the reduced dim (data
    # variables are reduced and kept). Without this, ds.mean("t") silently
    # loses dx and the target radial coordinate dies with KeyError: 'dx'.
    _lost = {c: ds[c] for c in ds.coords
             if c != "t" and c not in ds_avg.variables}
    if _lost:
        ds_avg = ds_avg.assign_coords(
            {c: (v.mean("t") if "t" in v.dims else v) for c, v in _lost.items()})
        print(f"  restored coords dropped by mean('t'): {sorted(_lost)}")

    Pe  = ds_avg["Pe"]
    Pi  = ds_avg[f"P{ion_species}"]
    Ve  = get_parallel_velocity(ds_avg, "e")
    Vi  = get_parallel_velocity(ds_avg, ion_species)
    Ve  = Ve.where(np.isfinite(Ve), 0.0)
    Vi  = Vi.where(np.isfinite(Vi), 0.0)
    phi = ds_avg["phi"] if use_phi_gamma else None

    all_targets = get_target_y_indices(ds_avg)
    print(f"Targets found: {list(all_targets.keys())}")

    def _midplane_s(tgt):
        """(s [m] at the TRUE midplane for each x of this target, side) by ψ —
        h3t.omp_mapped_distance — or None without a gridfile. The midplane
        ROW is off the midplane on coarse ny (SXD ny32: Z=+0.475 m), and its
        x-columns gave PFR cells the core ψ; this uses the leg's own ψ."""
        if not gridfile:
            return None
        s = h3t.omp_mapped_distance(gridfile, tgt)
        if s.size != ds_avg.sizes["x"]:
            raise SystemExit(f"gridfile nx={s.size} but dump nx="
                             f"{ds_avg.sizes['x']} — wrong grid for this run")
        side = "inner" if (tgt.startswith("inner")
                           and "imp" in get_midplane_y_indices(ds_avg)) else "outer"
        return s, side

    def _radial_coord(tgt, y_idx):
        if fit_coordinate == "omp":
            mp = _midplane_s(tgt)
            if mp is not None:
                s, side = mp
                return s * 1e3, ("IMP-mapped (ψ)" if side == "inner" else "OMP-mapped (ψ)")
            print("  WARNING: no gridfile — mapping by the midplane ROW, which is "
                  "off the midplane on coarse-ny grids (ny32: s ~1.25x too wide)")
            mids   = get_midplane_y_indices(ds_avg)
            is_imp = tgt.startswith("inner") and "imp" in mids
            y_ref  = mids["imp"] if is_imp else mids["omp"]
            s = target_physical_distance(ds_avg, y_ref, ixseps1=ixseps1)
            return s * 1e3, ("IMP-mapped" if is_imp else "OMP-mapped")
        if fit_coordinate == "wetted":
            s = target_physical_distance(ds_avg, y_idx, ixseps1=ixseps1)
            sin_a = float(field_line_incidence(ds_avg, y_idx).mean())
            return s / max(sin_a, 1e-3) * 1e3, "wetted (along plate)"
        s = target_physical_distance(ds_avg, y_idx, ixseps1=ixseps1)
        return s * 1e3, "target (⊥ flux surf)"

    # Need the (target, OMP-mapped) coordinate pair for the flux-expansion
    # diagnostic, even when fit_coordinate is something else.
    _bnd, _ = get_target_boundaries(ds_avg)
    _mids = get_midplane_y_indices(ds_avg)

    def _q_for(dsx, tgt, announce=False):
        """Target heat-flux density [W/m²] for one dataset — averaged or a
        single frame. Shared so the mean and the per-frame band cannot drift
        apart through a different code path."""
        q = target_heat_flux_from_diagnostics(dsx, target=tgt) \
            if prefer_diagnostics else None
        if q is not None:
            if announce:
                print("  Using Hermes-3 sheath diagnostic variables")
            return q
        if announce:
            print("  Computing from (γ P v_∥)(Bp/B)")
        Pe_ = dsx["Pe"]
        Pi_ = dsx[f"P{ion_species}"]
        Ve_ = get_parallel_velocity(dsx, "e")
        Vi_ = get_parallel_velocity(dsx, ion_species)
        Ve_ = Ve_.where(np.isfinite(Ve_), 0.0)
        Vi_ = Vi_.where(np.isfinite(Vi_), 0.0)
        _, q, _ = target_heat_flux_density(
            dsx, Pe_, Pi_, Ve_, Vi_,
            phi=(dsx["phi"] if use_phi_gamma else None),
            target=tgt, phi_wall=phi_wall,
            AA_ion=AA_ion, Z_ion=Z_ion, bohm_cap_v=bohm_cap_v)
        return q

    per_target = {}
    for tgt, y_idx in all_targets.items():
        print(f"\n── Target: {tgt} ──")
        # Prefer Hermes-3's own sheath diagnostics when available — they're
        # built directly from sheath_boundary.cxx so they capture exactly what
        # the simulation deposited, including any γ_e/γ_i corrections we don't
        # reproduce here. Fall back to the γ·P·V formula if absent.
        q_avg = _q_for(ds_avg, tgt, announce=True)
        s_mm, coord_label = _radial_coord(tgt, y_idx)
        q_avg_MWm2 = q_avg.values * 1e-6

        # Per-snapshot q, for the ±1σ band. It cannot be recovered from the
        # time-averaged fields: q is NONLINEAR in them (γ·P·V, and γ_e depends
        # on φ/T_e), so q(⟨P⟩,⟨V⟩) ≠ ⟨q(P,V)⟩. Computing each frame gives both
        # the spread and the correct mean — the same reason balance.py warns
        # never to `ds.mean("t")` before forming products.
        _nt = ds.sizes.get("t", 1)
        if _nt > 1:
            q_t_MWm2 = np.stack([
                _q_for(ds.isel(t=k), tgt).values * 1e-6 for k in range(_nt)])
        else:
            q_t_MWm2 = q_avg_MWm2[None, :]
        q_snap_MWm2 = q_t_MWm2.mean(0)

        _sc = np.nanmax(np.abs(q_avg_MWm2)) or 1.0
        _d = 100 * np.nanmax(np.abs(q_snap_MWm2 - q_avg_MWm2)) / _sc
        print(f"  mean of per-frame q vs q of mean fields: max {_d:.1f} % of peak"
              f"   ({_nt} frames)")
        q_MWm2 = q_snap_MWm2 if target_mean == "per_snapshot" else q_avg_MWm2
        # Preserve radial coordinates and integrate the profile used by the plot.
        q_selected = q_avg.copy(data=q_MWm2 * 1e6)
        P_total_kW = float(total_target_power(ds_avg, q_selected, y_idx)) / 1e3
        print(f"  total power = {P_total_kW:.2f} kW   [{coord_label}]")

        # Hermes-3's OWN sheath power, when the run used sheath_boundary_simple
        # with diagnose=true. This is the number that validates γ·P·V — set
        # use_phi_gamma = False to match that component's fixed γ=3.5/3.5,
        # otherwise the ratio below is a γ comparison, not a formula check.
        _dg = target_sheath_power_from_diagnostics(
            ds_avg, _bnd[tgt][0], ion_species=ion_species)
        P_diag_kW = float("nan")
        if _dg is not None:
            P_diag_kW = _dg["power_W"] / 1e3
            _rat = P_total_kW / P_diag_kW if P_diag_kW else float("nan")
            print(f"  Hermes-3 sheath diagnostic = {P_diag_kW:.2f} kW"
                  f"   → formula/diagnostic = {_rat:.4f}   ({'+'.join(_dg['vars'])})")
            if _dg["missing"]:
                print(f"    !! {_dg['missing']} absent → diagnostic UNDER-counts;"
                      f" set [{ion_species}] diagnose = true")
            if use_phi_gamma:
                print("    !! use_phi_gamma=True but sheath_boundary_simple uses"
                      " FIXED γ=3.5/3.5 — set it False for an exact comparison")
        # Flux-expansion factor at the LCFS-side SOL band: ratio of the
        # target-plate radial distance to the OMP-mapped distance for the same
        # flux surface. Eich λ_q at the target should equal f_x × Eich λ_q at
        # the OMP — this number lets you cross-check fits done in different
        # coordinate frames (e.g. target_poloidal_check.py uses target-frame s).
        _y_ref = (_mids["imp"] if (tgt.startswith("inner") and "imp" in _mids)
                  else _mids.get("omp"))
        i_dom_tgt = _bnd[tgt][0]
        # target_physical_distance returns a 1-D numpy array indexed by radial x.
        _mp = _midplane_s(tgt)
        _s_omp = (_mp[0] if _mp is not None else
                  target_physical_distance(ds_avg, _y_ref, ixseps1=ixseps1))
        _s_tgt = target_physical_distance(ds_avg, i_dom_tgt, ixseps1=ixseps1)
        _xev   = min(int(ixseps1) + 5, ds_avg.sizes["x"] - 1)
        f_x    = float(_s_tgt[_xev]) / max(float(_s_omp[_xev]), 1e-12)
        print(f"  flux expansion f_x ≈ {f_x:.2f}   "
              f"(λ_q[target] ≈ f_x · λ_q[OMP])")
        per_target[tgt] = dict(
            s_mm=np.asarray(s_mm),
            q_MWm2=np.asarray(q_MWm2),
            q_t_MWm2=np.asarray(q_t_MWm2),      # (nt, nx) — the band
            q_avg_fields_MWm2=np.asarray(q_avg_MWm2),
            nonlin_pct=float(_d),               # ⟨q(P,V)⟩ vs q(⟨P⟩,⟨V⟩)
            coord_label=coord_label,
            P_total_kW=P_total_kW,
            P_diag_kW=P_diag_kW,
            y_idx=int(y_idx),
            f_x=f_x,
        )

    # ── Input source power: 3/2 ∫ P_src dV per species, full torus ───────────
    _mxg = int(m.get("MXG", 2))
    elem_vol = _no_t(ds_avg["J"]) * _no_t(ds_avg["dx"]) * _no_t(ds_avg["dy"])
    P_in, P_in_origin = {}, {}
    for sp in ("e", ion_species):
        src = f"P{sp}_src"
        if src in ds_avg:
            S = ds_avg[src].isel({zdim: 0}) if zdim in ds_avg[src].dims else ds_avg[src]
            P_in[sp] = float((1.5 * S * elem_vol * 2 * np.pi)
                             .isel(x=slice(_mxg, -_mxg or None)).sum())
            P_in_origin[sp] = "dump"

    # A species' P<sp>_src is APPLIED from the grid file by evolve_pressure
    # (`mesh->get(source, "P"+name+"_src")`, evolve_pressure.cxx:117) whether or
    # not `[<sp>] diagnose = true` writes it to the dump. A dump-only total
    # therefore HALVES the input power on a deck that saves only the electron
    # source — so recover the rest from the grid with netCDF4.
    missing_src = [f"P{s}_src" for s in ("e", ion_species) if f"P{s}_src" not in ds_avg]
    src_xcheck = None
    if gridfile:
        import netCDF4 as _ncg
        try:
            with _ncg.Dataset(gridfile) as _g:
                _Jg = np.asarray(_g["J"][:], dtype=float)
                _wg = (_Jg * np.asarray(_g["dx"][:], dtype=float)
                       * np.asarray(_g["dy"][:], dtype=float))
                _wg = _wg[_mxg:_Jg.shape[0] - _mxg]
                for sp in ("e", ion_species):
                    src = f"P{sp}_src"
                    if src not in _g.variables:
                        continue
                    _S = np.asarray(_g[src][:], dtype=float)[_mxg:_Jg.shape[0] - _mxg]
                    _P = 1.5 * float((_S * _wg).sum()) * 2 * np.pi
                    if sp in P_in:
                        # Both paths available — a free validation that the grid
                        # source really is the SI Pa/s the dump path assumes.
                        src_xcheck = (sp, P_in[sp], _P)
                    else:
                        P_in[sp] = _P
                        P_in_origin[sp] = "grid"
        except (OSError, KeyError, IndexError) as e:
            print(f"  could not read sources from {gridfile}: {e}")
    still_missing = [f"P{s}_src" for s in ("e", ion_species) if s not in P_in]

    # ── Stored thermal energy and its rate of change ─────────────────────────
    # W = 3/2 ∫ (Pe + Pi) dV, full torus. Target load can only be compared with
    # the input power once dW/dt is accounted for: a run that is still heating
    # shows up here rather than as an apparent formula error.
    def _domain_rows():
        npol = ds.sizes[pdim]
        if not has_guards:
            return list(range(npol))
        _myg = int(m.get("MYG", 2))
        _nyi = int(m.get("ny_inner", 0) or 0)
        if _nyi > 0:                                        # double null
            _ny = npol - 4 * _myg
            g = (list(range(_myg))
                 + list(range(_nyi + _myg, _nyi + 3 * _myg))
                 + list(range(_ny + 3 * _myg, _ny + 4 * _myg)))
        else:                                               # single null
            _ny = npol - 2 * _myg
            g = list(range(_myg)) + list(range(_ny + _myg, _ny + 2 * _myg))
        return [i for i in range(npol) if i not in set(g)]

    W_t, t_s, dWdt = np.zeros(0), np.zeros(0), None
    try:
        _vol = (_no_t(ds["J"]) * _no_t(ds["dx"])
                * _no_t(ds["dy"]) * _no_t(ds["dz"]))        # [m³] per cell
        _zper = float(m.get("zperiod", 1) or 1)
        _Psum = ds["Pe"] + ds[f"P{ion_species}"]
        _sel = {"x": slice(_mxg, ds.sizes["x"] - _mxg), pdim: _domain_rows()}
        _W = (1.5 * _Psum * _vol).isel(_sel)
        # Σ_z dz = 2π/zperiod, so ×zperiod gives the full torus.
        _W = _W.sum(dim=[d for d in ("x", pdim, zdim) if d in _W.dims]) * _zper
        W_t = np.atleast_1d(np.asarray(_W.values, dtype=float))
        t_s = np.atleast_1d(np.asarray(ds["t"].values, dtype=float))
        if W_t.size > 1 and np.ptp(t_s) > 0:
            dWdt = float(np.linalg.lstsq(
                np.vstack([t_s, np.ones_like(t_s)]).T, W_t, rcond=None)[0][0])
    except (KeyError, ValueError) as e:
        print(f"  stored-energy term unavailable: {e}")

    return dict(
        **_flatten_targets(per_target),
        P_in=P_in,
        P_in_origin=P_in_origin,
        missing_src=missing_src,
        still_missing=still_missing,
        src_xcheck=list(src_xcheck) if src_xcheck else None,
        W_t=W_t,
        t_s=t_s,
        dWdt=dWdt if dWdt is not None else float("nan"),
        ion_species=ion_species,
        fit_coordinate=fit_coordinate,
    )



# ── Cache (de)serialisation for the per-target dict ──────────────────────────
# `save_cache` writes a portable .npz and keeps allow_pickle off, so it can
# only hold arrays plus JSON-able scalars. `per_target` is a dict of dicts of
# ARRAYS — neither — so numpy stored it as an object array and every cached
# rerun died with "Object arrays cannot be loaded when allow_pickle=False".
# (Pre-existing: this script could write a cache but never read one back.)
# Split it into one array per (target, field) plus a JSON-able scalar table.
def _flatten_targets(per_target):
    flat = {"target_names": list(per_target), "target_scalars": {}}
    for tgt, d in per_target.items():
        for k, v in d.items():
            if isinstance(v, np.ndarray):
                flat[f"tgt_{tgt}__{k}"] = v
            else:
                flat["target_scalars"].setdefault(tgt, {})[k] = v
    return flat


def _unflatten_targets(C):
    names = C.get("target_names", [])
    names = [n.decode() if isinstance(n, bytes) else str(n)
             for n in np.atleast_1d(names).tolist()]
    scal = C.get("target_scalars", {}) or {}
    out = {}
    for tgt in names:
        d = dict(scal.get(tgt, {}))
        pref = f"tgt_{tgt}__"
        for k in C:
            if k.startswith(pref):
                d[k[len(pref):]] = C[k]
        out[tgt] = d
    return out


# ── Cache load or compute ────────────────────────────────────────────────────
_cache_pkl = h3t.cache_path("v3_target_heatflux", dir_range,
                            ion_species, fit_coordinate,
                            "phig" if use_phi_gamma else "noPhi",
                            "bcap" if bohm_cap_v else "nocap",
                            "diag" if prefer_diagnostics else "nodiag",
                            f"Z{Z_ion:g}",
                            "psimid" if gridfile else "rowmid",
                            "fxv8")
                                       # fxv8: power integrates the selected mean profile
                                       # fxv2: f_x column; fxv3: per-frame q_t for the ±1σ band,
                                       # fxv4: POWER BUDGET (grid P_src + storage term),
                                       # fxv5: Hermes-3 sheath power diagnostic,
                                       # fxv6: total_target_power drops x-guards,
                                       #        and the plotted mean is now
                                       #        ⟨q(P,V)⟩ not q(⟨P⟩,⟨V⟩)
                                       # fxv7: "omp" coordinate = TRUE midplane by ψ
                                       #        (needs gridfile; else the old row)
if use_cache and _cache_pkl.exists():
    print(f"Cache hit: {_cache_pkl}")
    _C = h3t.load_cache(_cache_pkl)
else:
    _C = _compute()
    h3t.save_cache(_cache_pkl, **_C)
    print(f"Saved cache: {_cache_pkl}")
globals().update(_C)
per_target = _unflatten_targets(_C)


# ── Apply fits + plot from cached per-target profiles ────────────────────────
targets = (list(per_target.keys()) if targets_to_plot is None
           else [t for t in targets_to_plot if t in per_target])

n_t_plot = len(targets)
fig_hf,   axes_hf   = plt.subplots(1, n_t_plot, figsize=(5.5 * n_t_plot, 4.5), squeeze=False)
fig_eich, axes_eich = plt.subplots(1, n_t_plot, figsize=(5.5 * n_t_plot, 4.5), squeeze=False)
axes_hf, axes_eich = axes_hf[0], axes_eich[0]
results = {}

for ax_hf, ax_eich, tgt in zip(axes_hf, axes_eich, targets):
    pt = per_target[tgt]
    s_mm    = pt["s_mm"]
    q_MWm2  = pt["q_MWm2"]
    coord_label = pt["coord_label"]
    P_total = pt["P_total_kW"]
    print(f"\n── Target: {tgt} ── total power = {P_total:.2f} kW")

    # ±1σ over time of the per-frame q. ddof=1: with a handful of frames the
    # population σ understates the spread. Returns None when there is only one
    # frame, so a single-folder run simply draws no band.
    q_t = np.asarray(pt.get("q_t_MWm2", np.empty((0, 0))), dtype=float)
    q_sd = (q_t.std(0, ddof=1)
            if (target_band and q_t.ndim == 2 and q_t.shape[0] > 1) else None)

    fit_mask = (s_mm >= fit_s_range_mm[0]) & (s_mm <= fit_s_range_mm[1])
    s_fit, q_fit_data = s_mm[fit_mask], q_MWm2[fit_mask]

    res = {"P_total_kW": P_total}
    try:
        ep, _, s_fine, q_eich = fit_eich(s_fit, q_fit_data, fit_s0=fit_s0)
        res["eich"] = ep
        print(f"  Eich: λ_q={ep['lambda_q']:.2f} mm, S={ep['S']:.2f} mm, "
              f"s0={ep['s0']:.2f} mm, λ_int={ep['lambda_int']:.2f} mm")
    except Exception as exc:
        ep = None
        print(f"  Eich fit failed: {exc}")
    if include_double_exp:
        try:
            dp, _, s_fine_de, q_de = fit_double_exp(
                s_fit, q_fit_data,
                s_min=fit_s_range_mm[0], fit_s0=fit_s0)
            res["dexp"] = dp
            print(f"  double-exp: λ_near={dp['lambda_near']:.2f}, "
                  f"λ_far={dp['lambda_far']:.2f} mm, s0={dp['s0']:.2f} mm, "
                  f"far={dp['power_frac_far']*100:.0f}%")
        except Exception as exc:
            dp = None
            print(f"  double-exp fit failed: {exc}")
    else:
        dp = None
    if "nonlin_pct" in pt:
        print(f"  ⟨q(P,V)⟩ vs q(⟨P⟩,⟨V⟩): max {pt['nonlin_pct']:.1f} % of peak"
              f"   (plotting '{target_mean}')")
    if q_sd is not None:
        _pk = np.nanmax(np.abs(q_MWm2)) or 1.0
        _in = q_sd[fit_mask]
        print(f"  time scatter of q over the fit window: "
              f"median {100*np.median(_in)/_pk:.1f} %, "
              f"max {100*np.nanmax(_in)/_pk:.1f} % of peak q")
    results[tgt] = res

    # profile
    if q_sd is not None:
        ax_hf.fill_between(s_mm, q_MWm2 - target_band_n * q_sd,
                           q_MWm2 + target_band_n * q_sd,
                           color="b", alpha=0.20, lw=0, zorder=0,
                           label=rf"$\pm{target_band_n:g}\sigma_t$ "
                                 rf"({q_t.shape[0]} frames)")
    ax_hf.plot(s_mm, q_MWm2, "b.-", ms=4)
    ax_hf.axvline(0, color="k", ls="--", lw=0.8)
    if ep is not None and abs(ep["s0"]) > 0.05:
        ax_hf.axvline(ep["s0"], color="r", ls=":", lw=0.8,
                      label=f"s0={ep['s0']:.1f} mm")
    if ax_hf.get_legend_handles_labels()[0]:
        ax_hf.legend(fontsize=8)
    ax_hf.set_xlabel(f"s from separatrix [mm] — {coord_label}")
    ax_hf.set_ylabel(r"$q_\perp$ [MW/m²]")
    ax_hf.set_title(f"{tgt.replace('_', ' ').title()}\n$P$ = {P_total:.1f} kW")
    ax_hf.set_xlim(fit_s_range_mm[0] - 2, fit_s_range_mm[1])

    # fits
    if q_sd is not None:
        ax_eich.fill_between(s_mm, q_MWm2 - target_band_n * q_sd,
                             q_MWm2 + target_band_n * q_sd,
                             color="b", alpha=0.18, lw=0, zorder=0)
    ax_eich.plot(s_mm[fit_mask], q_fit_data, "b.", ms=5, label="sim (fit range)")
    ax_eich.plot(s_mm[~fit_mask], q_MWm2[~fit_mask], "b.", ms=3, alpha=0.3)
    if ep is not None:
        ax_eich.plot(s_fine, q_eich, "r-", lw=2,
                     label=rf"Eich $\lambda_q$={ep['lambda_q']:.1f} mm"
                            + (f", $s_0$={ep['s0']:.1f}" if abs(ep['s0']) > 0.05 else ""))
    if dp is not None:
        ax_eich.plot(s_fine_de, q_de, "g--", lw=2,
                     label=(rf"2-exp $\lambda_n$={dp['lambda_near']:.1f},"
                            rf"$\lambda_f$={dp['lambda_far']:.1f} mm"
                            + (f", $s_0$={dp['s0']:.1f}" if abs(dp['s0']) > 0.05 else "")))
    ax_eich.axvline(0, color="k", ls="--", lw=0.8)
    ax_eich.set_xlabel(f"s from separatrix [mm] — {coord_label}")
    ax_eich.set_ylabel(r"$q_\perp$ [MW/m²]")
    ax_eich.set_title(tgt.replace("_", " ").title())
    ax_eich.legend(fontsize=8)
    ax_eich.set_xlim(fit_s_range_mm[0] - 2, fit_s_range_mm[1])


# ── Summary + power balance ───────────────────────────────────────────────────
print("\n" + "=" * 92)
print(f"Widths in '{fit_coordinate}' coordinate "
      f"(fit_s0={fit_s0!r})")
print(f"{'Target':<14}{'P [kW]':>9}{'λ_q [mm]':>10}{'S [mm]':>9}"
      f"{'s0 [mm]':>10}{'λ_near':>9}{'λ_far':>9}{'far%':>7}{'f_x':>7}")
print("-" * 99)
P_tgt_tot = 0.0
for tgt, res in results.items():
    e, d = res.get("eich"), res.get("dexp")
    P_tgt_tot += res["P_total_kW"]
    lq = f"{e['lambda_q']:.2f}" if e else "N/A"
    S_ = f"{e['S']:.2f}" if e else "N/A"
    s0 = f"{e['s0']:.2f}" if e else "N/A"
    ln = f"{d['lambda_near']:.2f}" if d else "N/A"
    lf = f"{d['lambda_far']:.2f}" if d else "N/A"
    ff = f"{d['power_frac_far'] * 100:.0f}" if d else "N/A"
    fx = f"{per_target[tgt]['f_x']:.2f}" if "f_x" in per_target.get(tgt, {}) else "N/A"
    print(f"{tgt:<14}{res['P_total_kW']:>9.1f}{lq:>10}{S_:>9}{s0:>10}"
          f"{ln:>9}{lf:>9}{ff:>7}{fx:>7}")
print("-" * 99)
print(f"  λ_q here is in '{fit_coordinate}' coordinate. To compare with the "
      f"target-frame fit (e.g. target_poloidal_check.py): "
      f"λ_q[target] = f_x · λ_q[OMP].")

# ── POWER BUDGET ────────────────────────────────────────────────────────────
# Target load vs input is NOT a closed balance for this model, so the block
# prints the terms it can measure and labels the rest honestly. See
# Dudson et al. 2026 NF 66 036015 §5 + appendix B.
print("")
print("POWER BUDGET")
_P_tot_W = sum(P_in.values()) if P_in else 0.0
for sp in sorted(P_in):
    print(f"  {('input  P' + sp + '_src (' + P_in_origin.get(sp, '?') + ')'):<26}"
          f"{P_in[sp]/1e3:>+10.1f} kW")
if still_missing:
    print(f"  !! {still_missing} found in NEITHER dump nor grid — input is "
          f"incomplete. Set `gridfile` or re-run with [<sp>] diagnose = true.")
elif missing_src:
    print(f"  ({missing_src} not in the dump — recovered from the grid file; "
          f"evolve_pressure applies it either way.)")
if src_xcheck:
    _sp, _dmp, _grd = src_xcheck
    print(f"  cross-check P{_sp}_src: dump {_dmp/1e3:.1f} kW vs grid "
          f"{_grd/1e3:.1f} kW  ({100*abs(_dmp-_grd)/max(abs(_grd),1e-9):.1f} % apart)")
print(f"  {'input TOTAL':<26}{_P_tot_W/1e3:>+10.1f} kW")
print(f"  {'target load (all targets)':<26}{-P_tgt_tot:>+10.1f} kW"
      + (f"   {100*P_tgt_tot*1e3/_P_tot_W:5.1f} % of input" if _P_tot_W else ""))
if np.isfinite(dWdt):
    print(f"  {'storage 3/2 d∫(Pe+Pi)/dt':<26}{-dWdt/1e3:>+10.1f} kW"
          + (f"   {100*abs(dWdt)/_P_tot_W:5.1f} %" if _P_tot_W else "")
          + f"   (W = {W_t[-1]:.3g} J, {100*(W_t[-1]-W_t[0])/W_t[0]:+.2f} % over "
            f"{1e6*np.ptp(t_s):.3f} µs, {len(W_t)} frames)")
    if _P_tot_W:
        _tauE_us = 1e6 * W_t[-1] / _P_tot_W
        print(f"  {'':<26}{'':>10}      τ_E = W/P_in = {_tauE_us:.1f} µs")
        if len(W_t) < 3:
            print(f"  {'':<26}{'':>10}      !! only {len(W_t)} frames: dW/dt is a "
                  f"secant, not a rate. Load more folders.")
        elif np.ptp(t_s) * 1e6 < 0.2 * _tauE_us:
            print(f"  {'':<26}{'':>10}      !! window is {1e6*np.ptp(t_s)/_tauE_us:.2f} τ_E "
                  f"— too short to resolve a trend in W.")
    _resid = _P_tot_W - P_tgt_tot * 1e3 - dWdt
else:
    print(f"  {'storage 3/2 d∫(Pe+Pi)/dt':<26}{'n/a':>10}      "
          f"(needs ≥2 frames at distinct times)")
    _resid = _P_tot_W - P_tgt_tot * 1e3
print("  " + "-" * 46)
print(f"  {'residual, unmeasured here':<26}{-_resid/1e3:>+10.1f} kW"
      + (f"   {100*_resid/_P_tot_W:5.1f} % of input" if _P_tot_W else ""))
print("  The residual is radial-boundary transport PLUS the model's own")
print("  non-thermal channels: thermal energy is transferred to ion parallel")
print("  and perpendicular kinetic energy and not returned (the polarisation-")
print("  current compression term, Dudson 2026 app. A.2/B). On a RELAXED TCV")
print("  case that paper finds 61 % of input at the targets, 8 % to the radial")
print("  boundaries and 31 % in those channels — so a target fraction near")
print("  60 % is the expected ceiling, not 100 %. Use v3_plasma_balance.py for")
print("  the core/SOL/PFR radial split before suspecting the sheath formula.")
print("=" * 92)

fig_hf.suptitle("Target heat flux profiles (time-averaged)", y=1.02)
fig_hf.tight_layout()
fig_eich.suptitle(f"Eich (red) + double-exp (green) — {fit_coordinate} coordinate"
                  + (f"  [fit_s0={fit_s0}]" if fit_s0 else ""),
                  y=1.02)
fig_eich.tight_layout()
plt.show()

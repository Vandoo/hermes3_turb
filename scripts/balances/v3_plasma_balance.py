#!/usr/bin/env python3
"""
Particle and energy conservation check for a Hermes-3 3D turbulence simulation.

Validates that the surface-integrated radial flux through bounding flux surfaces
matches the volume-integrated particle/energy sources, following the approach
in the STORM scripts (v2_fconserv_plo.py).

Conservation check (steady state):
  Particle: F_x(outer) - F_x(inner) ≈ ∫ S_n dV  (+ target losses)
  Energy:   Q_x(outer) - Q_x(inner) ≈ ∫ S_E dV  (+ target losses)

Also plots the local poloidal conservation check:
  d(F_pol)/dθ  vs  F_x   (should agree if particle balance holds locally)

Caching
-------
The heavy load + flux computation is cached to ./cache/v3_plasma_balance_*.pkl.
On rerun with the same dir_range / species / include_y_derivs the global
balance dict + the local divergence arrays are reloaded from the pickle in
seconds. Set use_cache = False to force recomputation.

Usage
-----
    Edit the CONFIG block below, then run:
        python v3_plasma_balance.py
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from xhermes import open_hermesdataset
from hermes3_turb.load import get_parallel_velocity

# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
gridfile     = "../exp45470_x132_y28.nc"
dir_range    = range(140, 141)    # end is exclusive: 146 only
keep_yguards = False              # load y boundary guards (presence auto-detected below)
species      = "e"                # species for the balance (e, i, d+, ...)
include_y_derivs = True           # full curvilinear ExB (matches Hermes-3); False = xstorm convention
use_cache    = True               # reload pre-computed arrays from ./cache/*.pkl on rerun
# ─────────────────────────────────────────────────────────────────────────────
dir_range = h3t.existing_dirs(data_root, dir_range, "boutdata.nc")


def _compute():
    """Load datasets, compute every flux channel needed for the global balance
    and the local divergence plot. Returns a flat dict for pickling."""
    n_name = f"N{species}"
    P_name = f"P{species}"
    src_n_name = f"S{species}_src"
    src_P_name = f"P{species}_src"

    # ── Load ──────────────────────────────────────────────────────────────────
    datasets = []
    for i, dirnum in enumerate(dir_range):
        print(f"loading {data_root}{dirnum}")
        ds_i = open_hermesdataset(
            f"{data_root}/{dirnum}/boutdata.nc",
            geometry="toroidal",
            inputfilepath=f"{data_root}/{dirnum}/BOUT.inp",
            gridfilepath=gridfile,
            keep_xboundaries=True,
            keep_yboundaries=keep_yguards,
            unnormalise=False,
            info=False,
        )
        datasets.append(ds_i)
    # NB do NOT drop each folder's first frame: that assumes BOUT++'s default
    # dump_at_restart=true, and with dump_at_restart=false the first frame is a
    # genuinely NEW time, so the positional rule deletes real data (and a folder
    # squashed to a single output time contributes nothing at all). Restart
    # overlap is removed after concatenation by time VALUE -- see
    # load.dedup_restart_times, which handles both settings and mixed ones.
    ds = xr.concat(datasets, dim="t")
    ds = h3t.dedup_restart_times(ds)

    zdim = h3t.detect_toroidal_dim(ds)
    pdim = h3t.detect_poloidal_dim(ds)

    # ── Drop y-boundary guard rows if the load kept them ──────────────────────
    # xBOUT honours the GRID's `y_boundary_guards`, so a grid built with
    # y_boundary_guards=2 returns a guard-retaining dataset even when this
    # script asks for keep_yboundaries=False. That is not merely untidy:
    # `.bout.ddy()` returns only the domain rows but RE-LABELS them onto the
    # FIRST n_domain theta values rather than the domain rows' own labels.
    # xarray then inner-joins on those labels inside `ExB_radial_velocity`, so
    # each domain cell's ∂φ/∂θ is paired with the ∂φ/∂ζ and metric of a
    # *different* cell — offset by the guards, and wrong everywhere past the
    # first guard pair, not just at the ends.
    # Visible symptom: Fig 1 dies in pcolormesh with
    #   Dimensions of C (244, 112) should be one smaller than X(120)
    # Invisible symptom: a mis-registered ExB flux, plus `Gamma_mag_*` left on
    # the full 120 axis (load_grid_vars pads to the dataset) and `j1/j2`
    # computed for the guarded layout — so the θ-band of every 1-D profile is
    # taken from the wrong rows.
    # Measured on from_archer_6kW/data/396 (grid ny=112, MYG=2 → 120 rows):
    # ddy returns theta == full[:112] exactly, confirming the re-labelling.
    # Guard layout is this codebase's usual one: region A at +MYG, B at +3·MYG.
    import netCDF4 as _nc_g
    with _nc_g.Dataset(gridfile) as _g0:
        _ny_g = int(_g0["ny"][:])
    _MYG0 = int(ds.metadata.get("MYG", 2))
    _nyi0 = int(ds.metadata.get("ny_inner", 0) or 0)
    _npol = ds.sizes[pdim]
    if _nyi0 > 0 and _npol == _ny_g + 4 * _MYG0:            # double null
        _guard_rows = (list(range(_MYG0))
                       + list(range(_nyi0 + _MYG0, _nyi0 + 3 * _MYG0))
                       + list(range(_ny_g + 3 * _MYG0, _ny_g + 4 * _MYG0)))
    elif _npol == _ny_g + 2 * _MYG0:                        # single null
        _guard_rows = (list(range(_MYG0))
                       + list(range(_ny_g + _MYG0, _ny_g + 2 * _MYG0)))
    else:
        _guard_rows = []
    if _guard_rows:
        _keep_rows = [i for i in range(_npol) if i not in set(_guard_rows)]
        print(f"  dataset kept y-guards despite keep_yguards={keep_yguards} "
              f"(grid y_boundary_guards>0): dropping rows {_guard_rows}, "
              f"{_npol} -> {len(_keep_rows)} poloidal cells")
        ds = ds.isel({pdim: _keep_rows})
    m    = ds.metadata
    print(f"Loaded {ds.sizes['t']} time steps (all assumed saturated)")

    phi   = ds["phi"]
    n     = ds[n_name]
    P     = ds[P_name]
    v_par = get_parallel_velocity(ds, species)

    # ── Topology indices ──────────────────────────────────────────────────────
    import netCDF4 as _nc
    with _nc.Dataset(gridfile) as _grd:
        _ny_grid = int(_grd["ny"][:])
    _has_yguards = ds.sizes[pdim] > _ny_grid
    MYG = int(m.get("MYG", 2)) if _has_yguards else 0
    print(f"ny_grid={_ny_grid}, ny_data={ds.sizes[pdim]}, "
          f"y-guards in data: {_has_yguards}, MYG={MYG}")

    ixsep1 = m.get("ixseps1", ds.sizes["x"] // 2)
    j1_1   = m.get("jyseps1_1", MYG) + MYG + 1
    j2_1   = m.get("jyseps2_1", ds.sizes[pdim] // 4) + MYG + 1
    j1_2   = m.get("jyseps1_2", ds.sizes[pdim] // 2) + MYG * 3 + 1
    j2_2   = m.get("jyseps2_2", 3 * ds.sizes[pdim] // 4) + MYG * 3 + 1

    inner_sol = slice(j1_1, j2_1)
    outer_sol = slice(j1_2, j2_2)
    x_inner = ixsep1
    x_outer = ds.sizes["x"] - 3

    print(f"ixseps1 = {ixsep1},  inner SOL: [{j1_1}:{j2_1}],  outer SOL: [{j1_2}:{j2_2}]")

    # ── Flux channels (per species) ───────────────────────────────────────────
    def tmean(da):
        return da.mean("t") if "t" in getattr(da, "dims", ()) else da

    is_electron = (species == "e")
    ion_species = "i" if "Ni" in ds.data_vars else "d+"
    Ne = ds["Ne"]; Ni = ds[f"N{ion_species}"]
    Pe = ds["Pe"]; Pi = ds[f"P{ion_species}"]
    # Te/Ti are in the dump only if the input asks for them; a folder that
    # did not leaves whole NaN frames after the concat (nx32 dirs 23-28:
    # folder 23 has no Ti), so fill those frames from P/N.
    Te_eV = Pe / Ne.clip(min=1e-10)
    Ti_eV = Pi / Ni.clip(min=1e-10)
    if "Te" in ds:
        Te_eV = ds["Te"].fillna(Te_eV)
    if f"T{ion_species}" in ds:
        Ti_eV = ds[f"T{ion_species}"].fillna(Ti_eV)

    grid_vars = h3t.load_grid_vars(gridfile, ds=ds, pdim=pdim)
    bxcvx, bxcvy, Bxy = grid_vars["bxcvx"], grid_vars["bxcvy"], grid_vars["Bxy"]

    print("Computing radial fluxes (ExB + mag-drift + anomalous) ...")
    _, _, Gamma_ExB = h3t.decompose_radial_particle_flux(ds, n, phi, include_y_derivs=include_y_derivs)
    _, _, q_ExB     = h3t.decompose_radial_heat_flux(ds, P, phi, include_y_derivs=include_y_derivs)

    Gmag_e, Gmag_i    = h3t.radial_magnetic_drift_particle_flux(ds, Ne, Ni, bxcvx, Bxy, Te_eV, Ti_eV)
    qmag_e, qmag_i, _ = h3t.radial_magnetic_drift_heat_flux(ds, Pe, Pi, bxcvx, Bxy, Te_eV=Te_eV, Ti_eV=Ti_eV)
    Gamma_mag = tmean(Gmag_e if is_electron else Gmag_i)
    q_mag     = tmean(qmag_e if is_electron else qmag_i)

    def _opt(section, key, default=0.0):
        try:
            return float(ds.options[section][key])
        except (KeyError, TypeError, AttributeError):
            return default
    chi_s = _opt(P_name, "anomalous_chi") or _opt(species, "anomalous_chi")
    D_s   = _opt(n_name, "anomalous_D")   or _opt(species, "anomalous_D")
    _dn   = float(m.get("rho_s0", 1.0))**2 * float(m.get("Omega_ci", 1.0))
    chi_n = chi_s / _dn if _dn > 0 else chi_s
    D_n   = D_s   / _dn if _dn > 0 else D_s
    T_s   = Te_eV if is_electron else Ti_eV
    Gamma_anom = tmean(h3t.radial_anomalous_particle_flux(ds, n, D_n)) if D_s > 0 else 0.0
    q_anom     = tmean(h3t.radial_anomalous_heat_flux(ds, n, T_s, chi_n, D=D_n)) if (chi_s > 0 or D_s > 0) else 0.0
    print(f"  anomalous: chi={chi_s} m^2/s, D={D_s} m^2/s")

    try:
        _flutter = str(ds.options["electromagnetic"]["magnetic_flutter"]).lower() == "true"
    except (KeyError, TypeError, AttributeError):
        _flutter = False
    print(f"magnetic_flutter = {_flutter}")
    if _flutter and "Apar" in ds:
        Ve = h3t.get_parallel_velocity(ds, "e"); Vi = h3t.get_parallel_velocity(ds, ion_species)
        Gf, qf = h3t.radial_flutter_fluxes(ds, Ne, Pe, Ve, Pi, Vi, ds["Apar"])
        Gamma_flutter, q_flutter = tmean(Gf), tmean(qf)
    else:
        Gamma_flutter, q_flutter = 0.0, 0.0

    Gamma_r = Gamma_ExB + Gamma_mag + Gamma_anom
    q_r     = q_ExB     + q_mag     + q_anom
    Gamma_r_global = Gamma_r + Gamma_flutter
    q_r_global     = q_r     + q_flutter

    print("Computing poloidal fluxes (parallel + ExB + mag-drift) ...")
    F_pol = tmean(h3t.total_poloidal_particle_flux(ds, n, v_par, phi))
    Q_pol = tmean(h3t.total_poloidal_heat_flux(ds, P, v_par, phi))
    Gmag_pol_e, Gmag_pol_i = h3t.poloidal_magnetic_drift_particle_flux(ds, Ne, Ni, bxcvx, bxcvy, Bxy, Te_eV, Ti_eV)
    qmag_pol_e, qmag_pol_i = h3t.poloidal_magnetic_drift_heat_flux(ds, Pe, Pi, bxcvx, bxcvy, Bxy, Te_eV, Ti_eV)
    F_pol = F_pol + tmean(Gmag_pol_e if is_electron else Gmag_pol_i)
    Q_pol = Q_pol + tmean(qmag_pol_e if is_electron else qmag_pol_i)

    # ── Global balance ───────────────────────────────────────────────────────
    print("\n===== Particle balance =====")
    pbal = h3t.particle_balance(
        ds, n, v_par, phi,
        source_density=ds[src_n_name] if src_n_name in ds else None,
        x_inner=x_inner, x_outer=x_outer,
        theta_inner_sol=inner_sol,
        theta_outer_sol=outer_sol,
        radial_flux=Gamma_r_global,
    )
    for k, v in pbal.items():
        print(f"  {k:30s} = {v:.3e}")

    print("\n===== Energy balance =====")
    ebal = h3t.energy_balance(
        ds, P, v_par, phi,
        source_density=ds[src_P_name] if src_P_name in ds else None,
        x_inner=x_inner, x_outer=x_outer,
        theta_inner_sol=inner_sol,
        theta_outer_sol=outer_sol,
        radial_flux=q_r_global,
    )
    for k, v in ebal.items():
        print(f"  {k:30s} = {v:.3e}")

    # ── Radial boundaries: core / SOL / PFR, as sdtools splits them ──────────
    # sdtools' calculate_heat_balance sums THREE radial boundaries, not one:
    # the inner edge is the CORE wall where flux surfaces are closed and the
    # PFR wall inside the divertor legs, and the outer edge is the SOL wall.
    # Reported as flux LEAVING the domain, so these add to the target load.
    # (Cell-centred at the first/last domain cell, matching radial_flux_profile.)
    _mxg  = int(m.get("MXG", 2))
    _nyt  = ds.sizes[pdim]
    core_rows = list(range(j1_1, j2_1)) + list(range(j1_2, j2_2))
    all_rows  = list(range(MYG, _nyt - MYG)) if MYG else list(range(_nyt))
    pfr_rows  = [i for i in all_rows if i not in set(core_rows)]

    # This script loads unnormalise=False (raw metric), so convert at the end.
    _qe = 1.602176634e-19
    _Nn = float(m.get("Nnorm", 1.0)); _Tn = float(m.get("Tnorm", 1.0))
    _Cs = float(m.get("Cs0", 1.0));   _rs = float(m.get("rho_s0", 1.0))
    f_part = _Nn * _Cs * _rs**2                  # normalised → [s⁻¹]
    f_pow  = _Nn * _Tn * _qe * _Cs * _rs**2      # normalised → [W]

    # The DIAMAGNETIC WORK term φ·P·∇×(b/B) (Dudson 2026 eq. B.10) is a boundary
    # channel of the thermal ↔ perpendicular-flow exchange, so it belongs in
    # this table alongside the thermal fluxes rather than inside q_r. It is a
    # work term (potential × current), NOT an advected energy density, which is
    # why it is not bounded by the tiny ⊥KE reservoir measured below.
    _Wd = h3t.diamagnetic_work_radial_flux(ds, phi, Pe + Pi, bxcvx, Bxy)
    W_dia = tmean(h3t.toroidal_mean(_Wd, dim=zdim) if zdim in _Wd.dims else _Wd)

    # Validate it against the identity it rests on (eqs. B.9→B.10). Needs the
    # FULL curvature vector, so bxcvz and the shifted-metric twist too.
    dia_check = None
    try:
        curlb = h3t.load_curvature_vector(gridfile, ds=ds, pdim=pdim)
        # Consistency of the two load paths: diamagnetic_work_radial_flux uses
        # load_grid_vars' bxcvx/Bxy, the identity uses load_curvature_vector.
        _cx_alt = 2.0 * bxcvx / Bxy
        _d = float(np.abs(curlb["x"] - _cx_alt).max()
                   / max(float(np.abs(_cx_alt).max()), 1e-300))
        dia_check = h3t.diamagnetic_work_identity(
            ds, phi, Pe + Pi, curlb,
            branch_rows=(j1_1, j2_1, j1_2, j2_2,
                         int(m.get("ny_inner", 0) or 0) + MYG),
            mxg=_mxg)
        dia_check["curlbx_path_mismatch"] = _d
    except Exception as e:
        print(f"  diamagnetic identity check unavailable: {type(e).__name__}: {e}")

    rad_bnd = {}
    for _nm, _rows, _xi, _sgn in (
            ("core", core_rows, _mxg, -1.0),               # outward here ENTERS
            ("pfr",  pfr_rows,  _mxg, -1.0),
            ("sol",  all_rows,  ds.sizes["x"] - 1 - _mxg, +1.0)):   # LEAVES
        if not _rows:
            continue
        _F = h3t.radial_flux_profile(ds, Gamma_r_global, _rows)
        _Q = h3t.radial_flux_profile(ds, q_r_global,     _rows)
        _D = h3t.radial_flux_profile(ds, W_dia,          _rows)
        rad_bnd[_nm] = dict(
            n_theta=len(_rows), x=int(_xi),
            part_per_s=_sgn * float(_F.isel(x=_xi)) * f_part,
            power_W=_sgn * float(_Q.isel(x=_xi)) * f_pow,
            power_dia_W=_sgn * float(_D.isel(x=_xi)) * f_pow)

    # ── Is ⊥ kinetic energy a radial channel? ────────────────────────────────
    # Dudson 2026 app. B transfers 27.5 kW of 120 to ion ⊥ kinetic energy and
    # never returns it, noting the boundary diagnostic for it "has not yet been
    # implemented". It cannot be an ADVECTED energy density: ⊥KE and thermal
    # energy are carried by the same v_ExB_r, so their flux ratio through any
    # surface is just their density ratio. Measuring that ratio rules the
    # benign explanation in or out. If it is ≪1 the channel has to be the φ·J
    # work term (eqs. B.8/B.10), which involves the polarisation current
    # (∂/∂t of ∇⊥φ) and so is not recoverable from a dump.
    try:
        AA_i = float(ds.options[ion_species]["AA"])
    except (KeyError, TypeError, AttributeError, ValueError):
        AA_i = 2.0
    kperp = None
    try:
        v_r_n = h3t.ExB_radial_velocity(ds, phi, include_y_derivs=include_y_derivs)
        v_b_n = -np.sqrt(ds["g11"]) * phi.bout.ddx() / ds["Bxy"]
        # n0 (Boussinesq reference density) = Nnorm in these decks → n̂0 = 1.
        E_perp = 0.5 * AA_i * (v_r_n**2 + v_b_n**2)
        E_th = 1.5 * (Pe + Pi)
        # The RESERVOIR ratio ∫E_⊥dV / ∫E_th dV is the number the paper's budget
        # concerns, and it is the one to trust. The pointwise ratio has a long
        # tail purely because P → 0 in the far SOL and legs, where both energies
        # are negligible, so a pointwise median/percentile overstates the
        # channel's weight; it is printed only for context.
        _vol = ds["J"] * ds["dx"] * ds["dy"] * ds["dz"]
        _dims = [d for d in (E_perp.dims) if d != "t"]
        _Wp = float((E_perp * _vol).sum(dim=_dims).mean())
        _Wt = float((E_th * _vol).sum(dim=_dims).mean())
        _rat = np.asarray((E_perp / E_th.where(E_th > 0)).values, dtype=float)
        _rat = _rat[np.isfinite(_rat)]
        kperp = dict(reservoir=_Wp / _Wt if _Wt else float("nan"),
                     median=float(np.median(_rat)) if _rat.size else float("nan"),
                     p99=float(np.percentile(_rat, 99)) if _rat.size else float("nan"),
                     maxv=float(_rat.max()) if _rat.size else float("nan"),
                     AA=AA_i)
    except Exception as e:                       # diagnostic only, never fatal
        print(f"  ⊥KE ratio unavailable: {type(e).__name__}: {e}")

    # ── Local divergence: θ-dependent x_in ────────────────────────────────────
    ny  = ds.sizes[pdim]
    MXG = int(m.get("MXG", 2))
    x_out_loc = ds.sizes["x"] - 1 - MXG
    x_in_arr  = np.full(ny, MXG, dtype=int)
    x_in_arr[j1_1:j2_1] = ixsep1
    x_in_arr[j1_2:j2_2] = ixsep1

    def _column(da):
        extra = [d for d in da.dims if d not in ("x", pdim)]
        da = da.mean(extra) if extra else da
        return da.transpose("x", pdim).values

    def _radial_sum(col):
        return np.array([col[x_in_arr[j]:x_out_loc + 1, j].sum()
                         for j in range(ny)]) * 2 * np.pi

    def divergence_terms(F_pol_density, Gamma_radial):
        Gpol  = _radial_sum(_column(F_pol_density * h3t.poloidal_area_element_dx(ds)))
        dGpol = np.gradient(Gpol)
        Fx     = _column(Gamma_radial * h3t.radial_area_element_dtheta(ds)) * 2 * np.pi
        Fx_in  = Fx[x_in_arr, np.arange(ny)]
        Fx_out = Fx[x_out_loc, :]
        return dGpol, Fx_out - Fx_in

    def source_profile(src_name):
        if src_name not in ds:
            return None
        return _radial_sum(_column(ds[src_name] * ds["J"] * ds["dx"] * ds["dy"]))

    dGpol_n, dFx_n = divergence_terms(F_pol, Gamma_r)
    dGpol_q, dFx_q = divergence_terms(Q_pol, q_r)
    S_n = source_profile(src_n_name)
    S_q = source_profile(src_P_name)

    return dict(
        theta_coord=ds[pdim].values,
        ixsep1=int(ixsep1),
        j1_1=int(j1_1), j2_1=int(j2_1),
        j1_2=int(j1_2), j2_2=int(j2_2),
        pbal={k: float(v) for k, v in pbal.items()},
        ebal={k: float(v) for k, v in ebal.items()},
        dGpol_n=dGpol_n, dFx_n=dFx_n,
        dGpol_q=dGpol_q, dFx_q=dFx_q,
        S_n=S_n, S_q=S_q,
        rad_bnd=rad_bnd,
        kperp=kperp,
        dia_check=dia_check,
        species=species,
    )


# ── Cache load or compute ────────────────────────────────────────────────────
_cache_pkl = h3t.cache_path("v3_plasma_balance", dir_range, species,
                            "yd" if include_y_derivs else "noyd",
                            "b8")
                                    # b2: core/SOL/PFR radial split + ⊥KE ratio
                                    # b3: ⊥KE as a volume-integrated reservoir
                                    # b4: diamagnetic work term φP∇×(b/B)
                                    # b5: + its B.9→B.10 identity check
                                    # b8: Te/Ti NaN frames (folder without Ti in the dump) filled from P/N
if use_cache and _cache_pkl.exists():
    print(f"Cache hit: {_cache_pkl}")
    _C = h3t.load_cache(_cache_pkl)
    print("\n===== Particle balance (from cache) =====")
    for k, v in _C["pbal"].items():
        print(f"  {k:30s} = {v:.3e}")
    print("\n===== Energy balance (from cache) =====")
    for k, v in _C["ebal"].items():
        print(f"  {k:30s} = {v:.3e}")
else:
    _C = _compute()
    h3t.save_cache(_cache_pkl, **_C)
    print(f"Saved cache: {_cache_pkl}")
globals().update(_C)


def _report_dia_check(dc):
    if not dc:
        return
    print("\n===== Diamagnetic work term: validation status =====")
    print(f"  operator self-test |LHS−RHS|/rms(RHS), {dc['n_checked']} cells:"
          f"  median {dc['rel_median']:.2e}   99th {dc['rel_p99']:.2e}"
          f"   max {dc['rel_max']:.2e}")
    print("  (This is the B.9→B.10 product-rule identity. It holds for ANY")
    print("   vector field, so it does NOT validate Curlb_B — measured: dropping")
    print("   the y or z component, flipping the sign of x, or omitting the 2/B")
    print("   factor all leave this residual unchanged. It only confirms the")
    print("   differencing, Jacobian placement and periodic z wrap.)")
    print(f"  shifted-metric twist (sinty in grid): {dc['twist']}"
          + ("" if dc["twist"] else "  — absent, so Curlb_B.z carries NO twist,"
                                    " exactly as the run itself had it"))
    _mm = dc.get("curlbx_path_mismatch")
    if _mm is not None:
        print(f"  load_grid_vars vs load_curvature_vector on (Curlb_B)^x:"
              f" {_mm:.2e} relative")
    _r = dc.get("divJdia_ratio_median", float("nan"))
    if np.isfinite(_r):
        print(f"  ABSOLUTE check — Div(Jdia) / dump's DivJdia = {_r:.4f}"
              f"  (1.0 = Curlb_B scale confirmed)")
    else:
        print("  >> Curlb_B is NOT VALIDATED, so neither is the φP∇×(b/B)")
        print("     column above. The only check that constrains it is against")
        print("     Hermes-3's own DivJdia. Set [vorticity] diagnose = true and")
        print("     re-run ONE output step; this line then reports")
        print("     Div(Jdia)/DivJdia, which must be 1. Until then the formula's")
        print("     provenance in vorticity.cxx:599 is the only assurance.")


def _report_boundaries(rb, kp):
    if rb:
        print("\n===== Radial boundaries (positive = LEAVING the domain) =====")
        print(f"  {'surface':<24}{'particles [/s]':>16}{'thermal [kW]':>15}"
              f"{'φP∇×(b/B) [kW]':>17}")
        for nm, d in rb.items():
            _lbl = f"{nm} (x={d['x']}, {d['n_theta']} θ)"
            print(f"  {_lbl:<24}"
                  f"{d['part_per_s']:>16.3e}{d['power_W'] / 1e3:>15.2f}"
                  f"{d.get('power_dia_W', float('nan')) / 1e3:>17.2f}")
        print(f"  {'TOTAL':<24}"
              f"{sum(d['part_per_s'] for d in rb.values()):>16.3e}"
              f"{sum(d['power_W'] for d in rb.values()) / 1e3:>15.2f}"
              f"{sum(d.get('power_dia_W', 0.0) for d in rb.values()) / 1e3:>17.2f}")
        print("  'thermal' = ExB + magnetic drift + anomalous enthalpy/conduction;")
        print("  add it to the target load from v3_target_heatflux.py before")
        print("  comparing with 3/2 ∫P_src dV. The inner edge is the core wall")
        print("  only on closed flux surfaces, the PFR wall inside the legs.")
        print("  φP∇×(b/B) is the DIAMAGNETIC WORK term (Dudson 2026 eq. B.10) —")
        print("  a boundary channel of the thermal ↔ ⊥-flow exchange, so it is")
        print("  NOT part of the thermal column and must not be added to it")
        print("  when closing the THERMAL budget; it is one of the two routes")
        print("  into the ⊥KE reservoir. Caveat: it contains φ itself, not just")
        print("  ∇φ, so the volume/boundary SPLIT is gauge-dependent (the total")
        print("  is not). Hermes-3 fixes the gauge at the sheath via phi_wall,")
        print("  so the number is meaningful but shifts with the φ reference.")
    if kp:
        print("\n===== ⊥ kinetic vs thermal energy (ExB part, AA="
              f"{kp['AA']:g}) =====")
        print(f"  reservoir  ∫E_⊥dV / ∫(3/2(Pe+Pi))dV = {kp['reservoir']:.2e}")
        print(f"  pointwise  median {kp['median']:.2e}   99th pct {kp['p99']:.2e}"
              f"   max {kp['maxv']:.2e}   (tail is low-P cells, not weight)")
        print("  ⊥KE and thermal energy are advected by the SAME v_ExB_r, so the")
        print("  advected ⊥KE flux is this fraction of the enthalpy flux through")
        print("  any surface. With a reservoir ratio ≪1 the 27.5 kW ⊥KE channel")
        print("  in Dudson 2026 app. B cannot be boundary ADVECTION — it has to")
        print("  be a φ·J WORK term. Of those, the diamagnetic one is in the")
        print("  table above, and the parallel one ∮φj∥·dS is a TARGET channel")
        print("  the paper already quantifies (3 kW; it is the 76-vs-73 kW gap")
        print("  between its table 4 and figure 7). What is left is the")
        print("  POLARISATION current, which needs ∂/∂t of ∇⊥φ at solver")
        print("  precision and so is the one piece not recoverable from a dump.")


_report_boundaries(_C.get("rad_bnd"), _C.get("kperp"))
_report_dia_check(_C.get("dia_check"))


# ── Local conservation plot ──────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(13, 8))
fig.suptitle(f"Local conservation — species '{species}'  "
             f"(x_in: ixseps1 in main SOL, inner wall in legs; full torus)")

def mark_xpoints(ax):
    labelled = False
    for jj in (j1_1, j2_1, j1_2, j2_2):
        if 0 <= jj < len(theta_coord):
            ax.axvline(theta_coord[jj], color="gray", ls=":", lw=0.8,
                       label=None if labelled else "X-points (jyseps)")
            labelled = True

def plot_balance(row, dGpol, dFx, S, qty, unit):
    ax = axes[row, 0]
    ax.plot(theta_coord, dGpol, label=r"$\Delta_\theta F_\mathrm{pol}$")
    ax.plot(theta_coord, -dFx, "--", label=r"$-\Delta_x F_x$")
    mark_xpoints(ax)
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(f"[{unit}]")
    ax.set_title(f"{qty}: poloidal vs radial divergence")
    ax.legend()

    ax = axes[row, 1]
    ax.plot(theta_coord, dGpol + dFx, label=r"$\Delta_\theta F_\mathrm{pol}+\Delta_x F_x$")
    if S is not None:
        ax.plot(theta_coord, S, "--", label="source")
    mark_xpoints(ax)
    ax.set_xlabel(r"$\theta$"); ax.set_ylabel(f"[{unit}]")
    ax.set_title(f"{qty}: residual (≈ source)")
    ax.legend()

plot_balance(0, dGpol_n, dFx_n, S_n, "Particle", "s$^{-1}$")
plot_balance(1, dGpol_q, dFx_q, S_q, "Heat",     "W")

fig.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
Calculate and plot radial ExB particle and heat fluxes for a Hermes-3 3D
turbulence simulation.

Computes the toroidal mean/turbulent decomposition of the radial fluxes,
then plots:
  - 2D contour of mean and turbulent particle flux (x vs theta)
  - Radial profiles: flux integrated over outer SOL [jyseps1_2, jyseps2_2]
    as a function of x (flux surface index)
  - OMP density / temperature fall-off with λ_n, λ_Te, λ_Ti fits

Caching
-------
The heavy load + flux computation is cached to ./cache/v3_radial_flux_*.pkl.
On rerun with the same dir_range / include_y_derivs setting, the plot is
regenerated from the cache (seconds) instead of re-loading the dump (minutes
on slow disk / many folders). Set use_cache = False to force recomputation.
"""
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from xhermes import open_hermesdataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
#gridfile     = data_root+"../exp45470_x132_y28.nc"
gridfile     = data_root+"../new_sxd_nx132_ny36.nc"
#data_root    = '/home/qixia/remote_archer2/tmp_storage/data/'
#gridfile     = data_root+"../bout_sxd_v4_nx244_new.grd.nc"
dir_range    = range(140, 141)    # end is exclusive: 146 only
keep_yguards = False              # True if boutdata.nc has y boundary guard cells
include_y_derivs = True           # full curvilinear ExB (matches Hermes-3); False = xstorm convention
falloff_fit_range_mm = (0, 20)    # OMP SOL window [mm] for the λ_n / λ_T exp fit
omp_band     = True               # shade ±1σ on the OMP fall-off profiles, σ taken
                                  # over TIME of the ζ-averaged profile (how much
                                  # the mean profile moves between snapshots)
omp_band_n   = 1.0                # width of the band in units of σ
use_cache    = True               # reload pre-computed arrays from ./cache/*.pkl on rerun
# ─────────────────────────────────────────────────────────────────────────────
dir_range = h3t.existing_dirs(data_root, dir_range, "boutdata.nc")


def _compute():
    """Load dataset(s), compute every 2-D toroidal-/time-mean flux field and
    every 1-D profile the plots consume. Returns a flat dict (numpy arrays +
    scalars) ready for pickling."""
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

    print(f"Time-averaging fluxes over all {ds.sizes['t']} time steps")

    def tmean(da):
        return da.mean("t") if "t" in da.dims else da

    # ── SI conversion factors (data loaded normalised; convert at the end) ──
    # We load unnormalise=False because xhermes only annotates g_22 / g11 (not
    # g_33 / g_23), so a mixed-unit metric breaks `radial_flux_profile`. Then
    # the final results are converted to SI by a single scalar factor.
    m_meta = ds.metadata
    Nnorm    = float(m_meta["Nnorm"])
    Tnorm_eV = float(m_meta["Tnorm"])
    Cs0      = float(m_meta["Cs0"])
    rho_s0   = float(m_meta["rho_s0"])
    Omega_ci = float(m_meta["Omega_ci"])
    e_charge = 1.602e-19
    Pnorm    = Nnorm * Tnorm_eV * e_charge          # [Pa]
    # 2-D flux densities (×Nnorm·Cs0 for Γ [m⁻²s⁻¹] / ×Pnorm·Cs0 for q [W/m²])
    G_den_to_SI = Nnorm * Cs0
    q_den_to_SI = Pnorm * Cs0
    # 1-D radial profiles (already integrated over θ×ζ surface)
    F_n_to_SI = Nnorm * Cs0 * rho_s0 ** 2           # [s⁻¹]
    F_q_to_SI = Pnorm * Cs0 * rho_s0 ** 2           # [W]
    print(f"  SI conversion: G_den ×{G_den_to_SI:.2e}, q_den ×{q_den_to_SI:.2e}, "
          f"F_n ×{F_n_to_SI:.2e}, F_q ×{F_q_to_SI:.2e}")

    # ── Fields (time-dependent — turbulent decomposition needs δn·δv) ─────────
    phi = ds["phi"]
    Ne  = ds["Ne"]
    Pe  = ds["Pe"]

    ion_species = "i" if "Ni" in ds.data_vars else "d+"
    Ni  = ds[f"N{ion_species}"]
    Pi  = ds[f"P{ion_species}"]
    P_tot = Pe + Pi

    # Te/Ti are in the dump only if the input asks for them; a folder that
    # did not leaves whole NaN frames after the concat (nx32 dirs 23-28:
    # folder 23 has no Ti), so fill those frames from P/N.
    Te_eV = Pe / Ne.clip(min=1e-10)
    Ti_eV = Pi / Ni.clip(min=1e-10)
    if "Te" in ds:
        Te_eV = ds["Te"].fillna(Te_eV)
    if f"T{ion_species}" in ds:
        Ti_eV = ds[f"T{ion_species}"].fillna(Ti_eV)

    # ── ExB fluxes ────────────────────────────────────────────────────────────
    print("Computing radial ExB fluxes ...")
    Gamma_ExB_mean, Gamma_ExB_turb, _ = h3t.decompose_radial_particle_flux(
        ds, Ne, phi, include_y_derivs=include_y_derivs)
    q_ExB_mean, q_ExB_turb, _ = h3t.decompose_radial_heat_flux(
        ds, P_tot, phi, include_y_derivs=include_y_derivs)

    # ── Magnetic drift (curvature/grad-B) ─────────────────────────────────────
    print("Computing radial magnetic drift fluxes ...")
    grid_vars = h3t.load_grid_vars(gridfile, ds=ds, pdim=pdim)
    bxcvx, Bxy = grid_vars["bxcvx"], grid_vars["Bxy"]
    _, _, q_mag_tot = h3t.radial_magnetic_drift_heat_flux(
        ds, Pe, Pi, bxcvx, Bxy, Te_eV=Te_eV, Ti_eV=Ti_eV)
    q_mag_tot = tmean(q_mag_tot)
    Gamma_mag_e, _ = h3t.radial_magnetic_drift_particle_flux(
        ds, Ne, Ni, bxcvx, Bxy, Te_eV, Ti_eV)
    Gamma_mag_e = tmean(Gamma_mag_e)

    # ── Electromagnetic flutter ────────────────────────────────────────────────
    try:
        _has_flutter = str(ds.options["electromagnetic"]["magnetic_flutter"]).lower() == "true"
    except (KeyError, TypeError, AttributeError):
        _has_flutter = False
    print(f"magnetic_flutter = {_has_flutter}")

    if _has_flutter and "Apar" in ds:
        print("Computing electromagnetic flutter flux ...")
        Ve = h3t.get_parallel_velocity(ds, "e")
        Vi = h3t.get_parallel_velocity(ds, ion_species)
        Gamma_flutter, q_flutter = h3t.radial_flutter_fluxes(
            ds, Ne, Pe, Ve, Pi, Vi, ds["Apar"])
        Gamma_flutter, q_flutter = tmean(Gamma_flutter), tmean(q_flutter)
    else:
        Gamma_flutter = None
        q_flutter     = None

    # ── Anomalous perpendicular flux ──────────────────────────────────────────
    def _opt(section, key, default=0.0):
        try:
            return float(ds.options[section][key])
        except (KeyError, TypeError, AttributeError):
            return default

    chi_e  = _opt("Pe",              "anomalous_chi") or _opt("e",          "anomalous_chi")
    chi_i  = _opt(f"P{ion_species}", "anomalous_chi") or _opt(ion_species,  "anomalous_chi")
    D_anom = _opt("Ne",              "anomalous_D")   or _opt("e",          "anomalous_D")
    print(f"Anomalous chi_e={chi_e} m^2/s, chi_i={chi_i} m^2/s, D={D_anom} m^2/s")

    _rho_s0   = float(ds.metadata.get("rho_s0",   1.0))
    _Omega_ci = float(ds.metadata.get("Omega_ci", 1.0))
    _diff_norm = _rho_s0**2 * _Omega_ci
    chi_e_n  = chi_e  / _diff_norm if _diff_norm > 0 else chi_e
    chi_i_n  = chi_i  / _diff_norm if _diff_norm > 0 else chi_i
    D_anom_n = D_anom / _diff_norm if _diff_norm > 0 else D_anom

    if chi_e > 0 or chi_i > 0:
        q_anom_e   = h3t.radial_anomalous_heat_flux(ds, Ne, Te_eV, chi_e_n, D=D_anom_n)
        q_anom_i   = h3t.radial_anomalous_heat_flux(ds, Ni, Ti_eV, chi_i_n, D=D_anom_n)
        q_anom_tot = tmean(q_anom_e + q_anom_i)
    else:
        q_anom_tot = None
    Gamma_anom = (tmean(h3t.radial_anomalous_particle_flux(ds, Ne, D_anom_n))
                  if D_anom > 0 else None)

    # ── Topology indices ──────────────────────────────────────────────────────
    m = ds.metadata
    ixseps1 = m.get("ixseps1", ds.sizes["x"] // 2)
    ixseps2 = m.get("ixseps2", ixseps1)
    if ixseps1 > ixseps2:
        ixseps1, ixseps2 = ixseps2, ixseps1
    ixsep = ixseps1

    import netCDF4 as _nc
    with _nc.Dataset(gridfile) as _grd:
        _ny_grid = int(_grd["ny"][:])
    _has_guards = ds.sizes[pdim] > _ny_grid
    MYG = int(m.get("MYG", 0)) if _has_guards else 0

    j1 = int(m.get("jyseps1_2", ds.sizes[pdim] // 4)) + MYG * 3 + 1
    j2 = int(m.get("jyseps2_2", 3 * ds.sizes[pdim] // 4)) + MYG * 3 + 1
    sol_slice = slice(j1, j2)

    # ── 1-D radial profiles through each flux surface ────────────────────────
    Gamma_ExB_mean_prof = h3t.radial_flux_profile(ds, Gamma_ExB_mean, theta_slices=sol_slice)
    Gamma_ExB_turb_prof = h3t.radial_flux_profile(ds, Gamma_ExB_turb, theta_slices=sol_slice)
    Gamma_mag_prof      = h3t.radial_flux_profile(ds, Gamma_mag_e,    theta_slices=sol_slice)
    Gamma_tot_prof      = Gamma_ExB_mean_prof + Gamma_ExB_turb_prof + Gamma_mag_prof

    q_ExB_mean_prof = h3t.radial_flux_profile(ds, q_ExB_mean, theta_slices=sol_slice)
    q_ExB_turb_prof = h3t.radial_flux_profile(ds, q_ExB_turb, theta_slices=sol_slice)
    q_mag_prof      = h3t.radial_flux_profile(ds, q_mag_tot,  theta_slices=sol_slice)
    q_tot_prof      = q_ExB_mean_prof + q_ExB_turb_prof + q_mag_prof
    q_anom_prof = (h3t.radial_flux_profile(ds, q_anom_tot, theta_slices=sol_slice)
                   if q_anom_tot is not None else None)
    if q_anom_prof is not None:
        q_tot_prof = q_tot_prof + q_anom_prof
    q_flut_prof = (h3t.radial_flux_profile(ds, q_flutter, theta_slices=sol_slice)
                   if q_flutter is not None else None)
    if q_flut_prof is not None:
        q_tot_prof = q_tot_prof + q_flut_prof
    Gamma_anom_prof = (h3t.radial_flux_profile(ds, Gamma_anom, theta_slices=sol_slice)
                       if Gamma_anom is not None else None)
    if Gamma_anom_prof is not None:
        Gamma_tot_prof = Gamma_tot_prof + Gamma_anom_prof
    Gamma_flut_prof = (h3t.radial_flux_profile(ds, Gamma_flutter, theta_slices=sol_slice)
                       if Gamma_flutter is not None else None)
    if Gamma_flut_prof is not None:
        Gamma_tot_prof = Gamma_tot_prof + Gamma_flut_prof

    # ── OMP fall-off profiles & physical s coordinate ────────────────────────
    y_omp = h3t.get_midplane_y_indices(ds)["omp"]
    # s = distance from the separatrix at the TRUE outboard midplane, mapped
    # by ψ through the geqdsk embedded in the grid (h3t.omp_midplane_distance).
    # The profiles themselves come from row y_omp, which on coarse-ny grids is
    # well off the midplane (SXD ny32: Z=+0.475 m); the Rxy/Zxy arc along that
    # row stretched s by ~1.25x there and made an ny scan compare different
    # poloidal positions. Mapping by ψ puts every grid of one equilibrium on
    # the same coordinate. s=0 is psi_bdry (the separatrix face).
    try:
        s_omp_mm = h3t.omp_midplane_distance(gridfile) * 1e3
        print("λ fall-off radial coord: true outboard midplane, mapped by ψ (geqdsk)")
    except Exception as _exc:
        print(f"  WARNING: midplane mapping unavailable ({_exc});\n"
              f"           using the metric dx/√g11·rho_s0 at row y_omp={y_omp}, "
              f"which is the midplane only if that row is")
        s_omp_mm = h3t.target_physical_distance(ds, y_omp, ixseps1=ixsep) * _rho_s0 * 1e3

    def _omp_profile(da):
        """OMP radial profile: (time-mean, per-snapshot ζ-mean).

        The per-snapshot profile `p_t` (nt × nx — a few kB) is what the ±1σ
        band is built from: σ is taken over TIME of the ζ-averaged profile,
        i.e. how much the averaged profile moves between snapshots. That is
        the spread that says whether a short saturation window has pinned
        the mean down. (σ over ζ instead would be the instantaneous
        turbulence amplitude, which does not shrink with averaging and so
        says nothing about whether the mean is converged.)

        Caching `p_t` rather than a precomputed σ means the band definition
        can be changed at plot time without re-reading the dumps.
        """
        p = da.isel({pdim: y_omp})
        p_z = h3t.toroidal_mean(p, dim=zdim) if zdim in p.dims else p
        # ONE evaluation of the lazy graph, then the mean from the array.
        # Calling `.values` twice (once for p_t, once for tmean(p_z)) makes
        # dask redo the whole read, and it also lets the plotted mean drift
        # from the centre of the band it is drawn on.
        p_t = np.atleast_2d(p_z.values if "t" in p_z.dims
                            else p_z.values[None, :])
        return p_t.mean(0), p_t

    Ne_omp, Ne_omp_t = _omp_profile(Ne)
    Te_omp, Te_omp_t = _omp_profile(Te_eV)
    Ti_omp, Ti_omp_t = _omp_profile(Ti_eV)

    # ── 1-D coords + bookkeeping ─────────────────────────────────────────────
    theta = ds[pdim].values
    x_arr = ds["x"].values

    # ── Convert normalised → SI with the scalar factors derived above ───────
    #   • 2-D flux densities: ×Nnorm·Cs0 for Γ, ×Pnorm·Cs0 for q
    #   • 1-D radial profiles: ×Nnorm·Cs0·rho_s0² (Γ) / ×Pnorm·Cs0·rho_s0² (q)
    #   • OMP density: ×Nnorm    (m⁻³)
    #   • OMP temperature: ×Tnorm_eV  (eV)
    # Everything above this point is LAZY — the fluxes are only graphs. The
    # whole computation happens here, in the `.values` calls below, which is
    # why a run appears to stop dead after the last print above with no
    # further output for minutes. Evaluate one array at a time and say so, so
    # "it stopped" can be told apart from "it is still working", and so a
    # kill (out-of-memory, login-node CPU limit) names the array it died on.
    import time as _time
    _t_ev = _time.time()
    print("Evaluating flux arrays (the heavy step; everything above was lazy):")

    def _ev(name, da, factor=1.0):
        if da is None:
            return None
        _t0 = _time.time()
        out = np.asarray(da.values if hasattr(da, "values") else da) * factor
        print(f"    {name:22s} {str(out.shape):16s} {_time.time()-_t0:7.1f} s",
              flush=True)
        return out

    _result = dict(
        # 2-D flux DENSITIES [m⁻²s⁻¹ for Γ, W/m² for q]
        Gamma_ExB_mean=_ev("Gamma_ExB_mean", Gamma_ExB_mean, G_den_to_SI),
        Gamma_ExB_turb=_ev("Gamma_ExB_turb", Gamma_ExB_turb, G_den_to_SI),
        q_ExB_mean=_ev("q_ExB_mean", q_ExB_mean, q_den_to_SI),
        q_ExB_turb=_ev("q_ExB_turb", q_ExB_turb, q_den_to_SI),
        # 1-D radial profiles (integrated over θ-band & full torus) [s⁻¹ / W]
        Gamma_ExB_mean_prof=_ev("Gamma_ExB_mean_prof", Gamma_ExB_mean_prof, F_n_to_SI),
        Gamma_ExB_turb_prof=_ev("Gamma_ExB_turb_prof", Gamma_ExB_turb_prof, F_n_to_SI),
        Gamma_mag_prof=_ev("Gamma_mag_prof", Gamma_mag_prof, F_n_to_SI),
        Gamma_anom_prof=_ev("Gamma_anom_prof", Gamma_anom_prof, F_n_to_SI),
        Gamma_flut_prof=_ev("Gamma_flut_prof", Gamma_flut_prof, F_n_to_SI),
        Gamma_tot_prof=_ev("Gamma_tot_prof", Gamma_tot_prof, F_n_to_SI),
        q_ExB_mean_prof=_ev("q_ExB_mean_prof", q_ExB_mean_prof, F_q_to_SI),
        q_ExB_turb_prof=_ev("q_ExB_turb_prof", q_ExB_turb_prof, F_q_to_SI),
        q_mag_prof=_ev("q_mag_prof", q_mag_prof, F_q_to_SI),
        q_anom_prof=_ev("q_anom_prof", q_anom_prof, F_q_to_SI),
        q_flut_prof=_ev("q_flut_prof", q_flut_prof, F_q_to_SI),
        q_tot_prof=_ev("q_tot_prof", q_tot_prof, F_q_to_SI),
        # OMP fall-off  (Ne [m⁻³], Te/Ti [eV])
        Ne_omp=Ne_omp * Nnorm,
        Te_omp=Te_omp * Tnorm_eV,
        Ti_omp=Ti_omp * Tnorm_eV,
        # per-snapshot ζ-mean OMP profiles (nt × nx) — the ±1σ band is the
        # std over axis 0 of these; kept raw so the band can be redefined
        # without re-loading the dumps
        Ne_omp_t=Ne_omp_t * Nnorm,
        Te_omp_t=Te_omp_t * Tnorm_eV,
        Ti_omp_t=Ti_omp_t * Tnorm_eV,
        s_omp_mm=np.asarray(s_omp_mm),
        # 1-D coords
        theta=theta,
        x=x_arr,
        # scalars
        ixsep=int(ixsep),
        ion_species=ion_species,
    )
    print(f"  flux arrays evaluated in {_time.time()-_t_ev:.1f} s total")
    return _result


# ── Cache load or compute ────────────────────────────────────────────────────
_cache_pkl = h3t.cache_path("v3_radial_flux", dir_range,
                            "yd" if include_y_derivs else "noyd",
                            "si8")
                                      # bump suffix to invalidate normalised caches
                                      # si3: adds the per-snapshot OMP profiles
                                      #      (*_omp_t) the ±1σ band needs
                                      # si4: y-guard rows are now dropped before
                                      #      the fluxes are computed — an si3
                                      #      cache for a guard-retaining dump
                                      #      holds MIS-REGISTERED ExB fluxes
                                      # si7: Te/Ti NaN frames (folder without Ti in the dump) filled from P/N
                                      # si8: s_omp_mm at the TRUE midplane, mapped by ψ (ny32 row was Z=0.475 m)
if use_cache and _cache_pkl.exists():
    print(f"Cache hit: {_cache_pkl}")
    _C = h3t.load_cache(_cache_pkl)
else:
    _C = _compute()
    h3t.save_cache(_cache_pkl, **_C)
    print(f"Saved cache: {_cache_pkl}")
globals().update(_C)


# ── Fit OMP fall-off lengths (fast, redo on every plot) ──────────────────────
# Two fits per quantity:
#   • single exp:  q ∝ exp(−s/λ)         (slope of log)
#   • double exp:  q_bg + q_n·exp(−s/λ_n) + q_f·exp(−s/λ_f)
# When the far-SOL "shoulder" / flattening is present, the double-exp gives a
# λ_far ≫ λ_near; otherwise λ_near ≈ λ_far ≈ single-λ.
from hermes3_turb.target import fit_double_exp

def _fit_falloff(s_mm, prof, s_range):
    prof = np.asarray(prof, dtype=float)
    mask = ((s_mm >= s_range[0]) & (s_mm <= s_range[1])
            & np.isfinite(prof) & (prof > 0))
    if mask.sum() < 3:
        return np.nan, None
    slope, intercept = np.polyfit(s_mm[mask], np.log(prof[mask]), 1)
    lam = -1.0 / slope if slope < 0 else np.nan
    fit = np.exp(intercept + slope * s_mm)
    return lam, fit

def _fit_falloff_double(s_mm, prof, s_range):
    """Double-exponential fall-off fit. Returns (params, s_fine, q_fine)
    or (None, None, None) on failure / too few points."""
    s = np.asarray(s_mm, dtype=float)
    q = np.asarray(prof, dtype=float)
    mask = (s >= s_range[0]) & (s <= s_range[1]) & np.isfinite(q) & (q > 0)
    if mask.sum() < 6:    # 5-parameter fit needs more points
        return None, None, None
    try:
        params, _perr, s_fine, q_fit = fit_double_exp(
            s[mask], q[mask], s_min=s_range[0], fit_s0=False)
        return params, s_fine, q_fit
    except Exception as exc:
        print(f"  double-exp fit failed: {exc}")
        return None, None, None

lam_n,  fit_n  = _fit_falloff(s_omp_mm, Ne_omp, falloff_fit_range_mm)
lam_Te, fit_Te = _fit_falloff(s_omp_mm, Te_omp, falloff_fit_range_mm)
lam_Ti, fit_Ti = _fit_falloff(s_omp_mm, Ti_omp, falloff_fit_range_mm)

dp_n,  sd_n,  qd_n  = _fit_falloff_double(s_omp_mm, Ne_omp, falloff_fit_range_mm)
dp_Te, sd_Te, qd_Te = _fit_falloff_double(s_omp_mm, Te_omp, falloff_fit_range_mm)
dp_Ti, sd_Ti, qd_Ti = _fit_falloff_double(s_omp_mm, Ti_omp, falloff_fit_range_mm)

print(f"\nOMP SOL fall-off (fit window {falloff_fit_range_mm} mm):")
print(f"  single-exp:   λ_n  = {lam_n:.2f} mm,  λ_Te = {lam_Te:.2f} mm,  "
      f"λ_Ti = {lam_Ti:.2f} mm")
def _dp_str(dp, label):
    if dp is None:
        return f"  double-exp [{label}]: fit failed/too few points"
    return (f"  double-exp [{label}]: λ_near = {dp['lambda_near']:.2f} mm, "
            f"λ_far = {dp['lambda_far']:.2f} mm, "
            f"far-power frac = {dp['power_frac_far']*100:.0f}%")
print(_dp_str(dp_n,  "N_e"))
print(_dp_str(dp_Te, "T_e"))
print(_dp_str(dp_Ti, "T_i"))

# Relative band width over the fit window: the band as a picture is easy to
# over-read, so state it as a number. σ/|mean| is what decides whether the
# time-mean profile — and therefore λ — is worth quoting yet.
_wfit = (s_omp_mm >= falloff_fit_range_mm[0]) & (s_omp_mm <= falloff_fit_range_mm[1])
print(f"\nOMP time-scatter of the ζ-mean profile over the fit window "
      f"(σ_t/|mean|, {np.asarray(Ne_omp_t).shape[0]} snapshots):")
for _lbl, _pt in (("N_e", Ne_omp_t), ("T_e", Te_omp_t), ("T_i", Ti_omp_t)):
    _a = np.asarray(_pt, dtype=float)
    if _a.ndim != 2 or _a.shape[0] < 2:
        print(f"  {_lbl}: need >1 snapshot for a spread (have {_a.shape[0]})")
        continue
    _mu, _sd = _a.mean(0), _a.std(0, ddof=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        _rel = np.abs(_sd / _mu)[_wfit]
    _rel = _rel[np.isfinite(_rel)]
    if _rel.size:
        print(f"  {_lbl}: median {100*np.median(_rel):5.1f} %, "
              f"max {100*_rel.max():5.1f} %")


# ── Fig 1: 2D contour maps ────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"Radial ExB fluxes  (ions: '{ion_species}')", fontsize=13)

def cplot(ax, arr, title, cmap="seismic", sym=True):
    vmax = float(np.nanmax(np.abs(arr))) * 0.9 or 1.0
    vmin = -vmax if sym else 0.0
    cs = ax.pcolormesh(theta, x, arr, cmap=cmap, vmin=vmin, vmax=vmax)
    fig.colorbar(cs, ax=ax, fraction=0.03)
    ax.axhline(ixsep, color="w", ls="--", lw=1)
    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel("x")
    ax.set_title(title)

cplot(axes[0, 0], Gamma_ExB_mean, r"$\bar{\Gamma}_r$ (mean) [$\mathrm{m^{-2}\,s^{-1}}$]")
cplot(axes[0, 1], Gamma_ExB_turb, r"$\langle\delta N_e\,\delta v_r\rangle$ (turb) [$\mathrm{m^{-2}\,s^{-1}}$]")
cplot(axes[1, 0], q_ExB_mean,     r"$\bar{q}_r^\mathrm{ExB}$ (mean, $P_e+P_i$) [$\mathrm{W\,m^{-2}}$]")
cplot(axes[1, 1], q_ExB_turb,     r"$\langle\delta P_\mathrm{tot}\,\delta v_r\rangle\times 5/2$ (turb) [$\mathrm{W\,m^{-2}}$]")
fig.tight_layout()


# ── Fig 2: flux through flux surface vs x ────────────────────────────────────
fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
fig2.suptitle(r"Radial flux through flux surface — outer SOL ($\theta$: jyseps1\_2 to jyseps2\_2)")

ax1.plot(x, Gamma_ExB_mean_prof, label=r"ExB mean")
ax1.plot(x, Gamma_ExB_turb_prof, label=r"ExB turb")
ax1.plot(x, Gamma_mag_prof,      label=r"mag drift ($e$)")
if Gamma_anom_prof is not None:
    ax1.plot(x, Gamma_anom_prof, label=r"anomalous D")
if Gamma_flut_prof is not None:
    ax1.plot(x, Gamma_flut_prof, label=r"flutter")
ax1.plot(x, Gamma_tot_prof, "k--", lw=1.2, label="total")
ax1.axvline(ixsep, color="gray", ls="--", lw=0.8)
ax1.set_xlabel("x index")
ax1.set_ylabel(r"$2\pi\sum_\theta \Gamma_r\, dA$ (full torus) [$\mathrm{s^{-1}}$]")
ax1.set_title(r"Particle flux ($N_e$)")
ax1.legend()

ax2.plot(x, q_ExB_mean_prof, label=r"ExB mean")
ax2.plot(x, q_ExB_turb_prof, label=r"ExB turb")
ax2.plot(x, q_mag_prof,      label=r"mag drift ($e+i$)")
if q_anom_prof is not None:
    ax2.plot(x, q_anom_prof, label=r"anomalous $\chi$")
if q_flut_prof is not None:
    ax2.plot(x, q_flut_prof, label=r"flutter ($e+i$)")
ax2.plot(x, q_tot_prof, "k--", lw=1.2, label="total")
ax2.axvline(ixsep, color="gray", ls="--", lw=0.8)
ax2.set_xlabel("x index")
ax2.set_ylabel(r"$2\pi\sum_\theta q_r\, dA$ (full torus) [$\mathrm{W}$]")
ax2.set_title(r"Heat flux ($P_e + P_i$)")
ax2.legend()
fig2.tight_layout()


# ── Fig 3: SOL fall-off lengths λ_n, λ_T at the outer midplane ───────────────
fig3, (ax3a, ax3b) = plt.subplots(1, 2, figsize=(11, 4))
_nt_omp = np.asarray(Ne_omp_t).shape[0]
fig3.suptitle("OMP SOL fall-off lengths (toroidal- & time-mean)"
              + (rf",  shaded $\pm{omp_band_n:g}\sigma_t$ over {_nt_omp} snapshots"
                 if omp_band and _nt_omp > 1 else ""))
_win = (s_omp_mm >= falloff_fit_range_mm[0]) & (s_omp_mm <= falloff_fit_range_mm[1])


def _band(ax, prof_t, colour, n_sigma=None):
    """Shade mean ± n·σ_t, σ over time of the ζ-averaged profile.

    ddof=1: with only a handful of snapshots the population σ (ddof=0)
    understates the spread noticeably — at nt=11 by ~5 %.
    """
    a = np.asarray(prof_t, dtype=float)
    if not omp_band or a.ndim != 2 or a.shape[0] < 2:
        return None
    n_sigma = omp_band_n if n_sigma is None else n_sigma
    mu, sd = a.mean(0), a.std(0, ddof=1)
    ax.fill_between(s_omp_mm, mu - n_sigma * sd, mu + n_sigma * sd,
                    color=colour, alpha=0.20, lw=0, zorder=0)
    return sd


_sd_n = _band(ax3a, Ne_omp_t, "b")
ax3a.plot(s_omp_mm, Ne_omp, "b.-", ms=4, label="$N_e$ (OMP)")
if fit_n is not None:
    ax3a.plot(s_omp_mm[_win], fit_n[_win], "r-", lw=2,
              label=rf"single  $\lambda_n$ = {lam_n:.1f} mm")
if dp_n is not None:
    ax3a.plot(sd_n, qd_n, "orange", ls="--", lw=2,
              label=rf"double  $\lambda_n$={dp_n['lambda_near']:.1f}/"
                    rf"{dp_n['lambda_far']:.1f} mm "
                    rf"(far {dp_n['power_frac_far']*100:.0f}%)")
ax3a.axvline(0, color="k", ls="--", lw=0.8)
ax3a.set_xlabel("s from separatrix [mm] (OMP)")
ax3a.set_ylabel(r"$N_e$ [$\mathrm{m^{-3}}$]")
#ax3a.set_xlim(-2, falloff_fit_range_mm[1] + 5)
ax3a.set_title("Density fall-off")
ax3a.legend(fontsize=8)

_sd_Te = _band(ax3b, Te_omp_t, "b")
_sd_Ti = _band(ax3b, Ti_omp_t, "g")
ax3b.plot(s_omp_mm, Te_omp, "b.-", ms=4, label="$T_e$ (OMP)")
ax3b.plot(s_omp_mm, Ti_omp, "g.-", ms=4, label="$T_i$ (OMP)")
if fit_Te is not None:
    ax3b.plot(s_omp_mm[_win], fit_Te[_win], "r-", lw=2,
              label=rf"single $\lambda_{{Te}}$ = {lam_Te:.1f} mm")
if fit_Ti is not None:
    ax3b.plot(s_omp_mm[_win], fit_Ti[_win], "m-", lw=2,
              label=rf"single $\lambda_{{Ti}}$ = {lam_Ti:.1f} mm")
if dp_Te is not None:
    ax3b.plot(sd_Te, qd_Te, "orange", ls="--", lw=2,
              label=rf"double $\lambda_{{Te}}$={dp_Te['lambda_near']:.1f}/"
                    rf"{dp_Te['lambda_far']:.1f} mm "
                    rf"(far {dp_Te['power_frac_far']*100:.0f}%)")
if dp_Ti is not None:
    ax3b.plot(sd_Ti, qd_Ti, "brown", ls="--", lw=2,
              label=rf"double $\lambda_{{Ti}}$={dp_Ti['lambda_near']:.1f}/"
                    rf"{dp_Ti['lambda_far']:.1f} mm "
                    rf"(far {dp_Ti['power_frac_far']*100:.0f}%)")
ax3b.axvline(0, color="k", ls="--", lw=0.8)
ax3b.set_xlabel("s from separatrix [mm] (OMP)")
ax3b.set_ylabel("T [eV]")
#ax3b.set_xlim(-2, falloff_fit_range_mm[1] + 5)
ax3b.set_title("Temperature fall-off")
ax3b.legend(loc=2)
fig3.tight_layout()
plt.show()

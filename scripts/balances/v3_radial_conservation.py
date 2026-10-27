#!/usr/bin/env python3
"""
Radial conservation: source inside V(x) vs total radial flux through the
flux surface at x.  SI UNITS THROUGHOUT (unnormalise=True).

For each radial index x, plot two profiles:

  • F_in(x)  = ∫₀ˣ Σ(x', θ, ζ) dV               cumulative volume source
                                                  enclosed by the flux surface
                                                  at x  (full torus, ×2π)
  • F_out(x) = ∮  Γ_r · dA   at surface x        total radial flux through
                                                  the flux surface at x
                                                  (full torus)

In steady state with Neumann at the radial walls and **no other sinks in the
volume V(x)**, these are equal:    F_in(x)  =  F_out(x)

For x < ixseps1 this is the **clean balance** — agreement of the two curves is
the conservation closure — **but only on the between-X-point θ band**
(`theta_band = "closed"`).  With `theta_band = "full"` the integral also runs
over the divertor legs, where those same x indices are PFR and drain along B
to the targets, so the budget cannot close inside the separatrix and F_out
acquires humps that are leg transport rather than a core conservation error.  For
x > ixseps1 (open SOL + PFR field lines) target losses appear, so the
F_out curve falls below F_in, and the gap   F_in(x) − F_out(x)   is the
parallel power deposited on the plates between the LCFS and radial position
x.  The two curves separating at x = ixseps1 is the classical "LCFS power
crossing" reading.

Includes ExB + magnetic-drift + anomalous + (optional) flutter on the
radial side, matching the v3_plasma_balance flux model.  Source uses
S_e/S_i_src for particles and (3/2)·(P_e_src + P_i_src) for energy
(matching Hermes-3's pressure-equation convention).

Caching
-------
The heavy load + flux/source computation is cached to
./cache/v3_radial_conservation_*.pkl.  On rerun, the plot is regenerated
from the pickle in seconds.  Set use_cache = False to force recompute.
"""

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from xhermes import open_hermesdataset
from hermes3_turb.load import _no_t, get_parallel_velocity


# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
gridfile     = "../exp45470_x132_y28.nc"
dir_range    = range(140, 141)
keep_yguards = False
species      = "e"               # species for the particle balance (e or i / d+)
include_y_derivs = True
use_cache    = True
theta_band   = "full"            # surface integral θ-band:
                                 #   "full"       → all θ (correct ∮ for closed
                                 #                  flux surfaces; matches the
                                 #                  source-balance interpretation)
                                 #   "outer_sol"  → only the outer SOL band
                                 #                  (jyseps1_2 → jyseps2_2) —
                                 #                  same window as v3_radial_flux.
                                 #                  Useful for direct numerical
                                 #                  comparison with that script.
                                 #                  NOT a valid conservation
                                 #                  integral (only part of ∮).
# ─────────────────────────────────────────────────────────────────────────────
dir_range = h3t.existing_dirs(data_root, dir_range, "boutdata.nc")


def _compute():
    """Load data in SI, compute radial flux profile + cumulative source, return
    a flat dict suitable for pickling."""
    n_name = f"N{species}"
    P_name = f"P{species}"
    src_n_name = f"S{species}_src"
    src_P_name = f"P{species}_src"

    # ── Load (work in BOUT-normalised units; convert to SI at the end) ──────
    # We CANNOT use unnormalise=True here: xhermes only annotates conversion
    # factors for g_22, g11, J, dx, dy, dz — but NOT g_33, g_23, g_13. So under
    # unnormalise=True, g_22 is multiplied by rho_s0² (→ SI m²) while g_33/g_23
    # stay in normalised units, making radial_area_element_dtheta
    # `dy·√(g_22·g_33 − g_23²)` mix SI with normalised metric and giving
    # wrong area magnitudes (factor ~rho_s0^-1 = ~1e3, hence the 1e22 vs 1e7
    # disagreement). Loading unnormalise=False keeps the metric self-consistent
    # and the area / volume integrals correct.
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
    print(f"Loaded {ds.sizes['t']} time steps (BOUT-normalised; will be "
          "converted to SI for plotting)")

    # ── Single scalar conversion factors (normalised flux/source → SI) ──────
    # All radial-surface integrals share the same conversion to SI:
    #   F_SI = F_norm · (field conversion) · (area conversion)
    # • Particle flux density Γ_r [m⁻²s⁻¹] ↔ Γ_r_norm × (Nnorm · Cs0)
    # • Heat flux density q_r [W/m²]      ↔ q_r_norm  × (Pnorm · Cs0)
    #   where Pnorm = Nnorm·Tnorm·e is the BOUT pressure unit.
    # • Area element √(g_22·g_33 − g_23²)·dy [m²/rad] ↔ ×rho_s0² (whole sqrt)
    # • Multiply through ×2π (radial_flux_profile already does that).
    Nnorm     = float(m["Nnorm"])
    Tnorm_eV  = float(m["Tnorm"])
    Cs0       = float(m["Cs0"])
    rho_s0    = float(m["rho_s0"])
    Omega_ci  = float(m["Omega_ci"])
    e_charge  = 1.602e-19
    Pnorm     = Nnorm * Tnorm_eV * e_charge        # [Pa]
    F_n_to_SI = Nnorm * Cs0 * rho_s0 ** 2          # particle flux integral → s⁻¹
    F_q_to_SI = Pnorm * Cs0 * rho_s0 ** 2          # heat flux integral    → W
    # Cumulative source ∫S·dV: source density × volume element
    #   S_n_norm = S_SI / (Nnorm·Omega_ci);   dV_norm = dV_SI / rho_s0³
    #   ⇒  ∫S_n·dV_SI = ∫S_n_norm·dV_norm × Nnorm·Omega_ci·rho_s0³  [s⁻¹]
    # For energy source 3/2·P_src:  P_src_norm = P_src_SI / (Pnorm·Omega_ci)
    #   ⇒  ∫P_src·dV_SI = ∫P_src_norm·dV_norm × Pnorm·Omega_ci·rho_s0³  [W]
    S_n_to_SI = Nnorm * Omega_ci * rho_s0 ** 3
    S_q_to_SI = Pnorm * Omega_ci * rho_s0 ** 3
    print(f"  conversion factors: F_n×{F_n_to_SI:.3e}, F_q×{F_q_to_SI:.3e}, "
          f"S_n×{S_n_to_SI:.3e}, S_q×{S_q_to_SI:.3e}")

    # ── Fields (time-dependent — turb decomposition needs it) ────────────────
    phi = ds["phi"]
    n   = ds[n_name]
    P   = ds[P_name]
    v_par = get_parallel_velocity(ds, species)

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

    # ── Topology indices ──────────────────────────────────────────────────────
    import netCDF4 as _nc
    with _nc.Dataset(gridfile) as _grd:
        _ny_grid = int(_grd["ny"][:])
    has_yguards = ds.sizes[pdim] > _ny_grid
    MYG = int(m.get("MYG", 2)) if has_yguards else 0
    MXG = int(m.get("MXG", 2))
    ixsep1 = int(m.get("ixseps1", ds.sizes["x"] // 2))
    print(f"ixseps1 = {ixsep1}, MXG = {MXG}, MYG = {MYG}, "
          f"y-guards: {has_yguards}")

    # ── Geometry / grid vars (auto: no rescaling for SI data) ────────────────
    grid_vars = h3t.load_grid_vars(gridfile, ds=ds, pdim=pdim)
    bxcvx, Bxy = grid_vars["bxcvx"], grid_vars["Bxy"]

    # ── Radial flux channels (e+i) ───────────────────────────────────────────
    def tmean(da):
        return da.mean("t") if "t" in getattr(da, "dims", ()) else da

    print("Computing radial fluxes: ExB + mag-drift + anomalous (+flutter)...")
    # Particles: use Ne (quasi-neutral, Ne ≈ Ni)
    _, _, Gamma_ExB = h3t.decompose_radial_particle_flux(
        ds, Ne, phi, include_y_derivs=include_y_derivs)
    # Heat: total pressure (e+i).
    #
    # Keep the mean/turbulent split, don't discard it.  <v_ExB_r> over zeta is
    # carried ENTIRELY by the -(g_23/g_22)*dphi/dtheta term, because
    # <dphi/dzeta>_zeta = 0 exactly for a periodic phi.  So q_ExB_mean is a
    # direct measure of that non-orthogonality term's contribution, and on a
    # CLOSED flux surface it ought to be small: a mean ExB rotation transports
    # no net power radially.  A mean part comparable to the total is the
    # signature of a diagnostic artefact rather than turbulent transport.
    # `include_y_derivs = False` is the matching A/B (it drops the term).
    q_ExB_mean, q_ExB_turb, q_ExB = h3t.decompose_radial_heat_flux(
        ds, Pe + Pi, phi, include_y_derivs=include_y_derivs)

    # Magnetic drift (curvature + grad-B), e and i
    G_mag_e, G_mag_i = h3t.radial_magnetic_drift_particle_flux(
        ds, Ne, Ni, bxcvx, Bxy, Te_eV, Ti_eV)
    Gamma_mag = tmean(G_mag_e if is_electron else G_mag_i)
    _, _, q_mag_tot = h3t.radial_magnetic_drift_heat_flux(
        ds, Pe, Pi, bxcvx, Bxy, Te_eV=Te_eV, Ti_eV=Ti_eV)
    q_mag = tmean(q_mag_tot)

    # Anomalous (chi, D from BOUT.inp)
    def _opt(section, key, default=0.0):
        try:
            return float(ds.options[section][key])
        except (KeyError, TypeError, AttributeError):
            return default
    chi_e  = _opt("Pe",             "anomalous_chi") or _opt("e",          "anomalous_chi")
    chi_i  = _opt(f"P{ion_species}", "anomalous_chi") or _opt(ion_species,  "anomalous_chi")
    D_s    = _opt(n_name,           "anomalous_D")   or _opt(species,      "anomalous_D")
    T_s    = Te_eV if is_electron else Ti_eV
    # BOUT.inp gives χ/D in SI [m²/s]. We're in normalised mode here, so
    # rescale by 1/(rho_s0²·Omega_ci) to feed the anomalous-flux routines.
    _diff_norm = rho_s0**2 * Omega_ci
    chi_e_n = chi_e  / _diff_norm if _diff_norm > 0 else chi_e
    chi_i_n = chi_i  / _diff_norm if _diff_norm > 0 else chi_i
    D_s_n   = D_s    / _diff_norm if _diff_norm > 0 else D_s
    Gamma_anom = (tmean(h3t.radial_anomalous_particle_flux(ds, n, D_s_n))
                  if D_s > 0 else 0.0)
    q_anom_e   = (tmean(h3t.radial_anomalous_heat_flux(ds, Ne, Te_eV, chi_e_n, D=D_s_n))
                  if chi_e > 0 or D_s > 0 else 0.0)
    q_anom_i   = (tmean(h3t.radial_anomalous_heat_flux(ds, Ni, Ti_eV, chi_i_n, D=D_s_n))
                  if chi_i > 0 or D_s > 0 else 0.0)
    q_anom = q_anom_e + q_anom_i if not isinstance(q_anom_e, float) else 0.0
    print(f"  anomalous: chi_e={chi_e}, chi_i={chi_i}, D={D_s} m²/s")

    # Electromagnetic flutter (radial)
    try:
        has_flutter = str(ds.options["electromagnetic"]["magnetic_flutter"]).lower() == "true"
    except (KeyError, TypeError, AttributeError):
        has_flutter = False
    print(f"magnetic_flutter = {has_flutter}")
    if has_flutter and "Apar" in ds:
        Ve = get_parallel_velocity(ds, "e")
        Vi = get_parallel_velocity(ds, ion_species)
        Gf, qf = h3t.radial_flutter_fluxes(ds, Ne, Pe, Ve, Pi, Vi, ds["Apar"])
        Gamma_flutter = tmean(Gf)
        q_flutter     = tmean(qf)
    else:
        Gamma_flutter = 0.0
        q_flutter     = 0.0

    Gamma_r = Gamma_ExB + Gamma_mag + Gamma_anom + Gamma_flutter
    q_r     = q_ExB     + q_mag     + q_anom     + q_flutter

    # ── F_out(x): radial flux through each flux surface, full torus ─────────
    # The θ-band over which we sum the surface area element. "full" is the
    # physical ∮ for a closed flux surface (correct for conservation). The
    # "outer_sol" option restricts to the LFS jyseps1_2 → jyseps2_2 band so
    # the numbers can be directly compared to v3_radial_flux.py.
    nth = ds.sizes[pdim]
    j11 = int(m.get("jyseps1_1", -1))
    j21 = int(m.get("jyseps2_1", nth // 4))
    j12 = int(m.get("jyseps1_2", nth // 2))
    j22 = int(m.get("jyseps2_2", nth - 1))
    # Between-X-point bands: region A at +MYG, region B at +3·MYG (this
    # codebase's offset convention). Inside the separatrix these θ rows are
    # CLOSED flux surfaces; the complement is the four divertor legs, where
    # the same x < ixseps1 indices are PFR and drain along B to the targets.
    band_A = list(range(j11 + 1 + MYG,     j21 + 1 + MYG))
    band_B = list(range(j12 + 1 + 3 * MYG, j22 + 1 + 3 * MYG))
    closed_idx = [j for j in band_A + band_B if 0 <= j < nth]
    # y-GUARD rows must be excluded from every band. They carry a nonzero
    # grid source and a full volume element, so including them inflates the
    # volume integral -- measured on from_archer_6kW/408 as +5.6 kW (0.9 %),
    # on top of +32.6 kW (5.3 %) from the x-guards. `closed_idx` already
    # excludes them (its bands start at j+1+MYG), but the complement over
    # range(nth) does NOT, so `legs_idx` and the old `theta_band="full"`
    # (= slice(None)) both swept them in.
    _ny = int(m.get("ny", nth))
    _nyi = int(m.get("ny_inner", 0))
    _g = MYG if nth > _ny else 0
    guard_rows = set()
    if _g:
        for _a, _b in ((0, _g), (_nyi + _g, _nyi + 3 * _g),
                       (_ny + 3 * _g, _ny + 4 * _g)):
            guard_rows.update(range(_a, min(_b, nth)))
    interior_idx = [j for j in range(nth) if j not in guard_rows]
    legs_idx = [j for j in interior_idx if j not in set(closed_idx)]

    if theta_band == "outer_sol":
        j1 = j12 + MYG * 3 + 1
        j2 = j22 + MYG * 3 + 1
        theta_slices = slice(j1, j2)
        print(f"  θ-band: outer SOL only [{j1}:{j2}] "
              f"(direct comparison with v3_radial_flux.py)")
    elif theta_band == "closed":
        theta_slices = closed_idx
        print(f"  θ-band: between X-points only, {len(closed_idx)} of {nth} rows "
              f"(A {band_A[0]}..{band_A[-1]}, B {band_B[0]}..{band_B[-1]}) — "
              f"a genuinely CLOSED ∮ for x < ixseps1")
    elif theta_band == "legs":
        theta_slices = legs_idx
        print(f"  θ-band: divertor legs only, {len(legs_idx)} of {nth} rows "
              f"(PFR for x < ixseps1 — NOT a closed surface)")
    else:
        theta_slices = interior_idx
        print(f"  θ-band: full θ, {len(interior_idx)} of {nth} rows "
              f"({len(guard_rows)} y-guard rows excluded) — note this is NOT "
              f"a closed surface inside")
        print("    ixseps1 in a diverted geometry: the leg rows put PFR at the")
        print("    same x, and the PFR drains to the targets along B. Use")
        print("    theta_band='closed' for a budget that can actually close.")
    print("Computing F_out(x) ...")
    F_out_n = h3t.radial_flux_profile(ds, Gamma_r, theta_slices=theta_slices)
    F_out_q = h3t.radial_flux_profile(ds, q_r,     theta_slices=theta_slices)

    # Same surface integral applied to the ExB mean and turbulent parts alone,
    # so the total can be attributed.  These two sum to the ExB channel (not to
    # F_out_q, which also carries magnetic drift + anomalous + flutter).
    F_out_q_ExB_mean = h3t.radial_flux_profile(ds, q_ExB_mean,
                                               theta_slices=theta_slices)
    F_out_q_ExB_turb = h3t.radial_flux_profile(ds, q_ExB_turb,
                                               theta_slices=theta_slices)

    # Free diagnostic: how much of F_out comes from the closed band and how
    # much from the legs. A hump in the full-θ curve inside the separatrix
    # that lives in the legs column is PFR transport, not a core conservation
    # failure.
    _Fc = h3t.radial_flux_profile(ds, Gamma_r,
                                  theta_slices=closed_idx).values * F_n_to_SI
    _Fl = h3t.radial_flux_profile(ds, Gamma_r,
                                  theta_slices=legs_idx).values * F_n_to_SI
    # ...and by flux channel, on the band actually being plotted. The magnetic
    # drift is the channel that should very nearly cancel round a full poloidal
    # circuit, so a non-monotone bump living in that column is a cancellation
    # failure rather than transport.
    def _chan(G):
        if np.isscalar(G) and G == 0.0:
            return np.zeros_like(_Fc)
        return (h3t.radial_flux_profile(ds, G, theta_slices=theta_slices).values
                * F_n_to_SI)
    _Fx, _Fm, _Fa = _chan(Gamma_ExB), _chan(Gamma_mag), _chan(Gamma_anom)
    print("  F_out(x) decomposition [1e21 s⁻¹]")
    print("       x  |  closed    legs  |     ExB   magdrift   anom  |  total")
    for _x in [5, 10, 17, 22, 27, 32, 37, 50, int(m["ixseps1"]), 100]:
        if 0 <= _x < len(_Fc):
            print(f"    {_x:4d}  | {_Fc[_x]/1e21:8.2f}{_Fl[_x]/1e21:8.2f}  |"
                  f" {_Fx[_x]/1e21:8.2f} {_Fm[_x]/1e21:9.2f} {_Fa[_x]/1e21:7.2f}"
                  f"  | {(_Fc[_x]+_Fl[_x])/1e21:7.2f}")

    # ── F_in(x): cumulative volume source from x=MXG up to x ─────────────────
    # ∫_{V(x)} S dV = 2π · Σ_{x'≤x, θ} ⟨S⟩_ζ · J · dx · dy
    print("Computing F_in(x) (cumulative source) ...")
    elem_vol = _no_t(ds["J"]) * _no_t(ds["dx"]) * _no_t(ds["dy"])   # m³/rad

    # x-GUARD cells are ZEROED, not truncated: F_in(x) must keep the same x
    # axis as F_out(x) for the two curves to be compared index by index. The
    # cumsum previously ran over the FULL x range, so the two low-x guards fed
    # straight into F_in(0) and the two high-x guards into the total -- +32.6
    # kW of 619.7 (5.3 %) on from_archer_6kW/408, because the grid source is
    # nonzero there (max 1.9e-5 vs 5.7e-5 interior) and an outward-expanding
    # mesh gives those cells a large volume element. That alone made this
    # script disagree with inspect_hermes_run.py, which masks `xs >= MXG`.
    _xg = np.ones(ds.sizes["x"], dtype=float)
    if MXG > 0:
        _xg[:MXG] = 0.0
        _xg[ds.sizes["x"] - MXG:] = 0.0
    _xmask = xr.DataArray(_xg, dims=("x",))

    def _cumulative(S):
        if S is None:
            return None
        S_t = tmean(S)
        integrand = S_t * elem_vol
        if zdim in integrand.dims:
            integrand = integrand.mean(dim=zdim)
        integrand = integrand.isel({pdim: theta_slices}).sum(dim=pdim) * _xmask
        cum = integrand.cumsum(dim="x")             # Σ over x' ≤ x
        return (cum * 2.0 * np.pi).values

    # Particle source S_n (m⁻³ s⁻¹) — direct cumsum
    S_n = ds.get(src_n_name, None)
    F_in_n = _cumulative(S_n)

    # Energy source: pressure source × (3/2) to convert ∂P/∂t to ∂E/∂t.
    # Hermes-3 evolves P; internal energy is (3/2)P, so dE/dt|_src = (3/2)·dP/dt|_src.
    # For total energy balance we sum BOTH species' pressure sources.
    # A species' P<sp>_src is APPLIED from the grid file by evolve_pressure
    # (`mesh->get(source, "P"+name+"_src")`, evolve_pressure.cxx:117) whether or
    # not `[<sp>] diagnose = true` writes it to the dump — and xBOUT does NOT
    # merge grid source fields when `gridfilepath` is passed, so they are absent
    # from `ds` even though the grid carries them. A dump-only total therefore
    # HALVES the input power on a deck that saves only the electron source.
    # Recover the missing species from the grid with netCDF4.
    #
    # The grid stores P<sp>_src in SI (Pa/s) while this script works in
    # BOUT-normalised units, so divide by Pnorm·Omega_ci before integrating.
    # No coords are attached: xarray then aligns by dim NAME against elem_vol
    # rather than inner-joining on possibly-mismatched coordinate values.
    def _src_from_grid(name):
        """P<sp>_src from the grid as a normalised DataArray on ds's (x, θ)."""
        if not gridfile:
            return None
        import netCDF4 as _ncg
        try:
            with _ncg.Dataset(gridfile) as _g:
                if name not in _g.variables:
                    return None
                arr = np.asarray(_g[name][:], dtype=float)
        except (OSError, KeyError, IndexError) as exc:
            print(f"  could not read {name} from {gridfile}: {exc}")
            return None
        want = (ds.sizes["x"], ds.sizes[pdim])
        if arr.shape != want:
            print(f"  grid {name} is {arr.shape} but the dataset is {want} "
                  f"— y-guard layouts differ, not using the grid source.")
            return None
        return xr.DataArray(arr / (Pnorm * Omega_ci), dims=("x", pdim))

    Pe_src = ds.get("Pe_src", None)
    Pi_src = ds.get(f"P{ion_species}_src", None)
    src_origin = {"e": "dump" if Pe_src is not None else None,
                  ion_species: "dump" if Pi_src is not None else None}

    # Free validation of the SI→normalised conversion: when a species has BOTH
    # paths, the two integrals must agree. Done before any substitution so the
    # check is against a source the dump actually recorded.
    src_xcheck = None
    for _sp, _dmp in (("e", Pe_src), (ion_species, Pi_src)):
        if _dmp is None:
            continue
        _grd = _src_from_grid(f"P{_sp}_src")
        if _grd is not None:
            src_xcheck = (_sp,
                          float(_cumulative(1.5 * _dmp)[-1]),
                          float(_cumulative(1.5 * _grd)[-1]))
            break

    if Pe_src is None:
        Pe_src = _src_from_grid("Pe_src")
        if Pe_src is not None:
            src_origin["e"] = "grid"
    if Pi_src is None:
        Pi_src = _src_from_grid(f"P{ion_species}_src")
        if Pi_src is not None:
            src_origin[ion_species] = "grid"

    F_in_q_e = _cumulative(1.5 * Pe_src) if Pe_src is not None else None
    F_in_q_i = _cumulative(1.5 * Pi_src) if Pi_src is not None else None
    if F_in_q_e is not None and F_in_q_i is not None:
        F_in_q = F_in_q_e + F_in_q_i
    else:
        F_in_q = F_in_q_e if F_in_q_e is not None else F_in_q_i

    # ── Storage term dN/dt(x): the third term of the budget ─────────────────
    # On a closed flux surface with no parallel sink the budget is
    #       F_in(x)  =  F_out(x)  +  d/dt ∫_{V(x)} n dV
    # so a gap between the two curves is NOT automatically a flux-model error:
    # it is the rate at which the volume inside x is filling or draining. A
    # run that is still building its profile shows exactly this, and the sign
    # of the gap says which way. Measured here by least squares over all
    # loaded frames (a secant if only 2 frames — the print says how many).
    def _cumulative_inventory(F):
        """d/dt ∫₀ˣ F dV, normalised units, least squares over t."""
        if "t" not in F.dims or F.sizes["t"] < 2:
            return None
        integrand = F * elem_vol
        if zdim in integrand.dims:
            integrand = integrand.mean(dim=zdim)
        integrand = integrand.isel({pdim: theta_slices}).sum(dim=pdim) * _xmask
        cum = (integrand.cumsum(dim="x") * 2.0 * np.pi)      # (t, x)
        tt = np.asarray(ds["t"].values, dtype=float)
        A = np.vstack([tt - tt.mean(), np.ones_like(tt)]).T
        Y = cum.transpose("t", "x").values
        slope, _ = np.linalg.lstsq(A, Y, rcond=None)[0]
        return slope                                          # dN/dt per x

    dNdt_n = _cumulative_inventory(Ne)
    dNdt_q = _cumulative_inventory(1.5 * (Pe + Pi))
    print(f"  storage term from {ds.sizes.get('t', 0)} frames"
          + ("" if ds.sizes.get("t", 0) >= 3 else "  (SECANT, not a rate)"))

    # `missing_src` now means "could not be found in EITHER the dump or the
    # grid" — the only case in which F_in is genuinely incomplete.
    missing_src = [k for k in (src_n_name, "Pe_src", f"P{ion_species}_src")
                   if k not in ds]
    still_missing = ([src_n_name] if src_n_name not in ds else []) \
        + [f"P{s}_src" for s in ("e", ion_species) if src_origin.get(s) is None]
    for _sp in ("e", ion_species):
        if src_origin.get(_sp) == "grid":
            print(f"  P{_sp}_src not in the dump — recovered from {gridfile}")
    if src_xcheck is not None:
        _sp, _d, _g = src_xcheck
        print(f"  cross-check P{_sp}_src: dump {_d * S_q_to_SI / 1e3:.1f} kW vs "
              f"grid {_g * S_q_to_SI / 1e3:.1f} kW "
              f"({100 * abs(_g - _d) / abs(_d):.1f}% apart)")
    if still_missing:
        print(f"  WARNING: {still_missing} in neither dump nor grid — F_in is "
              "under-counted. Set `gridfile`, or re-run with [<sp>] diagnose = true.")

    # ── Convert normalised → SI with the single scalar factors derived above ─
    # F_out (radial flux integrated over surface):
    #     F_out_n_SI [s⁻¹] = F_out_n_norm × Nnorm·Cs0·rho_s0²
    #     F_out_q_SI [W]   = F_out_q_norm × Pnorm·Cs0·rho_s0²
    # F_in (cumulative volume source):
    #     F_in_n_SI [s⁻¹] = F_in_n_norm × Nnorm·Omega_ci·rho_s0³
    #     F_in_q_SI [W]   = F_in_q_norm × Pnorm·Omega_ci·rho_s0³
    F_out_n_SI = np.asarray(F_out_n.values) * F_n_to_SI
    F_out_q_SI = np.asarray(F_out_q.values) * F_q_to_SI
    F_out_q_ExB_mean_SI = np.asarray(F_out_q_ExB_mean.values) * F_q_to_SI
    F_out_q_ExB_turb_SI = np.asarray(F_out_q_ExB_turb.values) * F_q_to_SI
    F_in_n_SI  = None if F_in_n  is None else F_in_n  * S_n_to_SI
    F_in_q_SI  = None if F_in_q  is None else F_in_q  * S_q_to_SI
    F_in_q_e_SI = None if F_in_q_e is None else F_in_q_e * S_q_to_SI
    F_in_q_i_SI = None if F_in_q_i is None else F_in_q_i * S_q_to_SI
    dNdt_n_SI = None if dNdt_n is None else dNdt_n * S_n_to_SI
    dNdt_q_SI = None if dNdt_q is None else dNdt_q * S_q_to_SI

    # Closure report: how well does F_in = F_out + dN/dt actually hold?
    if dNdt_n_SI is not None and F_in_n_SI is not None:
        print("  budget closure  F_in = F_out + dN/dt   [1e21 s⁻¹]")
        print("       x  |    F_in   F_out   dN/dt | residual  (% of F_in)")
        for _x in [10, 17, 22, 27, 32, 37, 50, int(ixsep1)]:
            if 0 <= _x < len(F_out_n_SI):
                _r = F_in_n_SI[_x] - F_out_n_SI[_x] - dNdt_n_SI[_x]
                _p = 100 * _r / F_in_n_SI[_x] if F_in_n_SI[_x] else np.nan
                print(f"    {_x:4d}  | {F_in_n_SI[_x]/1e21:7.2f}"
                      f" {F_out_n_SI[_x]/1e21:7.2f} {dNdt_n_SI[_x]/1e21:7.2f} |"
                      f" {_r/1e21:8.2f}  ({_p:6.1f}%)")

    return dict(
        x=ds["x"].values,
        F_out_n=F_out_n_SI,
        F_out_q=F_out_q_SI,
        F_out_q_ExB_mean=F_out_q_ExB_mean_SI,
        F_out_q_ExB_turb=F_out_q_ExB_turb_SI,
        F_in_n=F_in_n_SI,
        F_in_q=F_in_q_SI,
        F_in_q_e=F_in_q_e_SI,
        F_in_q_i=F_in_q_i_SI,
        dNdt_n=dNdt_n_SI,
        dNdt_q=dNdt_q_SI,
        ixsep1=ixsep1,
        MXG=int(MXG),
        species=species,
        missing_src=missing_src,
        still_missing=still_missing,
        src_origin=src_origin,
    )


# ── Cache load or compute ────────────────────────────────────────────────────
_cache_pkl = h3t.cache_path("v3_radial_conservation_d2", dir_range, species,
                            "yd" if include_y_derivs else "noyd",
                            "si8",    # si2: broken unnormalise=True cache;
                                      # si3: P<sp>_src now recovered from the grid
                                      # si4: F_in uses the same θ-band as F_out
                                      # si5: storage term dN/dt(x) added
                                      # si7: ExB mean/turbulent split of F_out_q
                                      # si8: Te/Ti NaN frames (folder without Ti in the dump) filled from P/N
                            theta_band)
if use_cache and _cache_pkl.exists():
    print(f"Cache hit: {_cache_pkl}")
    _C = h3t.load_cache(_cache_pkl)
else:
    _C = _compute()
    h3t.save_cache(_cache_pkl, **_C)
    print(f"Saved cache: {_cache_pkl}")
globals().update(_C)
dNdt_n = _C.get("dNdt_n")
dNdt_q = _C.get("dNdt_q")

# ── ExB mean vs turbulent at the separatrix ──────────────────────────────────
# On a closed flux surface a MEAN ExB rotation carries no net radial power, and
# <dphi/dzeta>_zeta = 0 exactly, so the whole mean part comes from the
# -(g_23/g_22)*dphi/dtheta non-orthogonality term. If `mean` is comparable to
# the total, the LCFS crossing is not turbulent transport. A/B with
# `include_y_derivs = False`, which removes that term entirely.
_qm = _C.get("F_out_q_ExB_mean")
_qt = _C.get("F_out_q_ExB_turb")
if _qm is not None and _qt is not None:
    _ix = int(ixsep1)
    print("\n  ExB decomposition of the radial heat flux at x = ixseps1 "
          f"= {_ix}   [kW]   (theta_band='{theta_band}', "
          f"include_y_derivs={include_y_derivs})")
    print(f"    {'mean  <P><v_r>':>18} {_qm[_ix]/1e3:>10.1f}")
    print(f"    {'turbulent <dPdv>':>18} {_qt[_ix]/1e3:>10.1f}")
    print(f"    {'ExB total':>18} {(_qm[_ix]+_qt[_ix])/1e3:>10.1f}")
    print(f"    {'F_out_q (all ch.)':>18} {F_out_q[_ix]/1e3:>10.1f}")
    _den = abs(_qm[_ix]) + abs(_qt[_ix])
    if _den > 0:
        print(f"    mean fraction |mean|/(|mean|+|turb|) = "
              f"{abs(_qm[_ix])/_den:.3f}")
        if abs(_qm[_ix]) > abs(_qt[_ix]):
            print("    WARNING: the MEAN part dominates. On a closed surface "
                  "that is\n"
                  "             a non-orthogonality/gauge artefact, not "
                  "turbulent transport.\n"
                  "             Re-run with include_y_derivs = False to "
                  "isolate it.")


# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle(f"Radial conservation (SI) — species '{species}' "
             f"(F_in: ∫₀ˣ S dV · 2π     F_out: ∮ Γ·dA at x)")

# ── Particle balance
ax = axes[0]
if F_in_n is not None:
    ax.plot(x, F_in_n * 1e-21, "r-", lw=2,
            label=r"$F_\mathrm{in}(x) = \int_0^x S_n\,dV$  (cumulative source)")
else:
    print(f"WARNING: S{species}_src not in dump — F_in plot skipped.")
ax.plot(x, F_out_n * 1e-21, "b--", lw=2,
        label=r"$F_\mathrm{out}(x) = \oint \Gamma_r\,dA$  (flux through surface)")
if dNdt_n is not None:
    ax.plot(x, dNdt_n * 1e-21, color="0.45", ls="-.", lw=1.4,
            label=r"$d/dt\int_0^x n\,dV$  (storage)")
    ax.plot(x, (F_out_n + dNdt_n) * 1e-21, "g-", lw=1.2, alpha=0.8,
            label=r"$F_\mathrm{out} + dN/dt$  (should match $F_\mathrm{in}$)")
    ax.axhline(0.0, color="0.8", lw=0.6)
ax.axvline(ixsep1, color="gray", ls=":", lw=1.2)
ax.text(ixsep1, ax.get_ylim()[1] * 0.92, " LCFS", color="gray", fontsize=9, va="top")
ax.axvline(MXG, color="lightgray", ls=":", lw=0.8)
ax.set_xlabel("x index (radial)")
ax.set_ylabel(r"particle rate [$10^{21}$ s$^{-1}$]")
ax.set_title(f"Particle balance — species '{species}'")
ax.legend(fontsize=9, loc="best")

# ── Heat balance (e + i)
ax = axes[1]
if F_in_q is not None:
    ax.plot(x, F_in_q * 1e-6, "r-", lw=2,
            label=r"$F_\mathrm{in}(x) = (3/2)\int_0^x (P_e\!+\!P_i)_\mathrm{src}\,dV$")
if F_in_q_e is not None:
    ax.plot(x, F_in_q_e * 1e-6, "r:", lw=1, alpha=0.6, label="  e-only (input)")
if F_in_q_i is not None: 
    ax.plot(x, F_in_q_i * 1e-6, "r--", lw=1, alpha=0.6, label="  i-only (input)")
ax.plot(x, F_out_q * 1e-6, "b--", lw=2,
        label=r"$F_\mathrm{out}(x) = \oint q_r\,dA$  (flux through surface)")
if dNdt_q is not None:
    ax.plot(x, dNdt_q * 1e-6, color="0.45", ls="-.", lw=1.4,
            label=r"$\frac{3}{2}\,d/dt\int_0^x (P_e\!+\!P_i)\,dV$  (storage)")
    ax.plot(x, (F_out_q + dNdt_q) * 1e-6, "g-", lw=1.2, alpha=0.8,
            label=r"$F_\mathrm{out} + dW/dt$")
    ax.axhline(0.0, color="0.8", lw=0.6)
ax.axvline(ixsep1, color="gray", ls=":", lw=1.2)
ax.text(ixsep1, ax.get_ylim()[1] * 0.92, " LCFS", color="gray", fontsize=9, va="top")
ax.axvline(MXG, color="lightgray", ls=":", lw=0.8)
ax.set_xlabel("x index (radial)")
ax.set_ylabel("heat flux [MW]")
ax.set_title("Energy balance (e+i)")
ax.legend(fontsize=9, loc="best")

print("\nReading the plots:")
print(f"  • At x = ixseps1 = {ixsep1} (LCFS): both curves cross; the value is")
print( "    the LCFS particle / heat-flux load.")
print( "  • For x < LCFS: F_in and F_out should overlap (closed flux surfaces,")
print( "    no other sinks). Deviation = numerical error or anomalous channel")
print( "    not included.")
print( "  • For x > LCFS: F_out drops below F_in by the amount of parallel")
print( "    flux flowing to the targets from V(x).  The gap at the outer wall")
print( "    equals the total target deposition.")
if still_missing:
    print(f"  • WARNING: {still_missing} in neither dump nor grid — F_in is under-counted.")

fig.tight_layout()
plt.show()

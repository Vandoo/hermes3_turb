"""
Particle and heat flux calculations for Hermes-3 3D turbulence simulations.

All formulas are derived from the Hermes-3 physics equations (Dudson 2026) and
cross-validated against the STORM post-processing scripts (v2_fluxX.py,
v2_fluxY.py, v2_Qx_basic.py) which share the same BOUT++ metric conventions.

Coordinate system (BOUT++ field-aligned, loaded with geometry="toroidal"):
  x     : radial (across flux surfaces)
  theta : poloidal (field-following)
  zeta  : toroidal (binormal direction in the rotating frame)

Key drifts from Hermes-3 model (in SI units after xhermes unnormalisation):
  ExB drift:       v_E = (b × ∇_⊥φ) / B
  Diamagnetic:     v_p = (b × ∇p) / (qnB)   [not implemented here yet]

Turbulent transport decomposition (toroidal Reynolds decomposition):
  f = f̄ + δf   where  f̄ = <f>_ζ

  Total radial flux:     <Γ_r>_ζ = Γ̄_r + <δn δv_ExB_r>_ζ
  Mean-flow contribution: Γ̄_r  = n̄ · v̄_ExB_r
  Turbulent contribution: Γ_turb = <δn · δv_ExB_r>_ζ
"""

import numpy as np
from .decompose import toroidal_mean, turbulent_correlation, reynolds_decompose
from .load import detect_toroidal_dim, _no_t, _ensure_z_single_chunk


# ---------------------------------------------------------------------------
# ExB drift velocities
# ---------------------------------------------------------------------------

def ExB_radial_velocity(ds, phi, include_y_derivs=True):
    """
    Radial (x) component of the ExB drift velocity.

        v_ExB_x = (∂φ/∂ζ - g_{θζ}/g_{θθ} · ∂φ/∂θ) / sqrt(g^{xx})

    The y-derivative correction (g_23/g_22 term) is included by default to
    match Hermes-3, which evaluates the full curvilinear ExB term. The xstorm
    convention (include_y_derivs=False) keeps only the ∂φ/∂ζ piece — useful
    for isolating the turbulent fluctuation flux, since ⟨∂φ/∂ζ⟩_ζ = 0 by
    periodicity so the mean radial flow then collapses to zero. With y-derivs
    on, the toroidal mean ⟨v_ExB_x⟩_ζ picks up the mean equilibrium ExB shear
    flow from the y-correction — physically real in Hermes-3, but it lives on
    flux surfaces and does not by itself drive net cross-field particle
    transport (only its turbulent correlation with δn does).

    Parameters
    ----------
    ds : xarray.Dataset
    phi : xarray.DataArray
        Electrostatic potential [V].
    include_y_derivs : bool, default True
        Include the g_23/g_22·∂φ/∂θ correction. Set False to match xstorm /
        STORM convention and drop the equilibrium ExB shear contribution.

    Returns
    -------
    xarray.DataArray
        Radial ExB velocity [m/s].
    """
    phi = _ensure_z_single_chunk(phi, detect_toroidal_dim(ds))
    dzeta_phi = phi.bout.ddz()   # ∂φ/∂ζ  [V]
    g11 = _no_t(ds["g11"])
    if include_y_derivs:
        dtheta_phi = phi.bout.ddy()
        # NOTE: g_23/g_22 must be formed from a CONSISTENTLY-normalised metric.
        # xhermes only unnormalises g_22 (×rho_s0²) and g11, leaving g_23 raw, so
        # this function must be fed a raw metric — load with unnormalise=False
        # (as v3_radial_flux does, see its comment). Both raw ⇒ the dimensionless
        # g_{θζ}/g_{θθ} ratio is correct. Do NOT pass an unnormalise=True dataset.
        g_23, g_22 = _no_t(ds["g_23"]), _no_t(ds["g_22"])
        return (dzeta_phi - g_23 / g_22 * dtheta_phi) / np.sqrt(g11)
    return dzeta_phi / np.sqrt(g11)


def radial_electric_field(ds, phi):
    """
    Radial electric field  E_r = −∂φ/∂r  [V/m].

    Uses the physical radial derivative: with the flux coordinate x, the
    perpendicular radial distance element is dr = dx/√g11 (the same relation the
    target/blob radial coordinate uses), so ∂φ/∂r = √g11·∂φ/∂x and

        E_r = −√g11 · ∂φ/∂x.

    Only ``g11`` is used, which xhermes unnormalises correctly (unlike the
    g_23/g_22 pair), so there is no metric-normalisation pitfall here. E_r is the
    field that drives the poloidal ExB flow (V_ExB,θ ≈ E_r/B, see
    :func:`ExB_poloidal_velocity`); its radial gradient dE_r/dr is the ExB
    shearing rate relevant to turbulence suppression / L-H.

    Parameters
    ----------
    ds  : xarray.Dataset      (must contain g11)
    phi : xarray.DataArray    Electrostatic potential [V].

    Returns
    -------
    xarray.DataArray  E_r [V/m].
    """
    dx_phi = phi.bout.ddx()               # ∂φ/∂x  (x = flux coordinate)
    return -np.sqrt(_no_t(ds["g11"])) * dx_phi


def magnetic_drift_radial_velocity(ds, T_eV, bxcvx, Bxy):
    """
    Radial component of the magnetic (curvature/grad-B) drift velocity.

    Follows xstorm v_mag_e (accessors.py, lines 494-496):

        v_mag_s = -T_s/(q_s/e) · Curl(b/B)_radial
                = -T_s · 2·bxcvx / (Bxy · √g11)   for electrons  (q=-e → sign negative)
                = +T_s · 2·bxcvx / (Bxy · √g11)   for ions       (q=+e → sign positive)

    Pass the appropriate sign via T_eV: negative T_eV for electrons, positive for ions,
    or use radial_magnetic_drift_heat_flux which handles signs automatically.

    The full diamagnetic flux n·v_dia splits into a large divergence-free
    part and this smaller transport-relevant part; the latter is what drives
    actual radial transport (see xstorm docstring for v_mag_e).

    Parameters
    ----------
    ds : xarray.Dataset
        Must contain g11 [T²m²] (from xhermes unnormalised dataset).
    T_eV : xarray.DataArray
        Species temperature [eV] = P_s / (N_s · e).
    bxcvx : xarray.DataArray
        Contravariant x-component of B/2·Curl(b/B) from grid file.
    Bxy : xarray.DataArray
        Total magnetic field magnitude [T] from grid file.

    Returns
    -------
    xarray.DataArray
        Radial magnetic drift velocity [m/s].
    """
    return -T_eV * 2.0 * bxcvx / (Bxy * np.sqrt(_no_t(ds["g11"])))


def diamagnetic_work_radial_flux(ds, phi, P_charged, bxcvx, Bxy):
    """
    Radial flux density of the **diamagnetic work term** [W/m²].

        W   = φ · P_charged · ∇×(b/B)
        W_r = φ · P_charged · 2·bxcvx / (B · √g11)

    This is the boundary flux that the pair of volume terms in Dudson et al.
    2026 (Nucl. Fusion 66 036015) eqs. (B.9)→(B.10) collapses to:

        −φ ∇·(P ∇×(b/B))  −  P ∇×(b/B)·∇φ  =  −∇·(φ P ∇×(b/B))

    The **second** volume term is exactly what Hermes-3 applies to the species
    energy source — `vorticity.cxx:599`,
    ``subtract(species["energy_source"], Jdia_species * Grad(phi))`` with
    ``Jdia_species = P * Curlb_B`` — and the **first** is the diamagnetic drive
    of the vorticity equation (``ddt(Vort) += Div(Jdia)``). Together they are
    the channel that exchanges species *thermal* energy with *perpendicular
    flow* energy, i.e. one of the two routes into the ⊥KE reservoir that
    Dudson 2026 §5 finds is fed at 27.5 kW and never returned. ``W_r`` is the
    part of that channel which leaves through a flux surface instead of
    staying in the domain, so it is one of the "transport of perpendicular ion
    kinetic energy through the boundaries" diagnostics the paper notes "have
    not yet been implemented".

    Unlike the advected ⊥KE (which is negligible — see the reservoir ratio in
    ``v3_plasma_balance.py``), this is a **work** term: potential × current,
    not energy density × velocity. It therefore does not scale with the ⊥KE
    density and is not bounded by it.

    Conventions
    -----------
    ``Curlb_B = ∇×(b/B) = 2·bxcv/B`` is the Hermes-3 definition
    (``braginskii_ion_viscosity.cxx:108–112``: ``bxcv`` is read from the grid,
    rescaled, then ``*= 2./Bxy``). The physical radial component is
    ``(Curlb_B)^x/√g11``, which is the *same* factor
    ``2·bxcvx/(Bxy·√g11)`` already used by
    :func:`radial_magnetic_drift_particle_flux` — so the normalisation rides on
    the validated magnetic-drift path. ``bxcvx``/``Bxy`` must come from
    :func:`hermes3_turb.load.load_grid_vars` in the dataset's own
    normalisation (see [[bug-poloidal-magdrift]]).

    Integrating ``W_r`` with :func:`hermes3_turb.balance.radial_flux_profile`
    gives ``∫W^x J dy dz`` exactly, because
    ``√(g_22·g_33 − g_23²) = J·√g11`` for these metrics.

    Sign: positive = outward (increasing x). The volume term is ``−∇·W``, so
    energy **leaving** through a surface of outward normal n̂ is ``+∮W·n̂dS``.

    ⚠ **Gauge.** W contains φ itself, not only ∇φ, so the *split* between the
    volume and boundary parts of this channel depends on the zero of φ (a
    shift φ→φ+c moves ``c·∮P∇×(b/B)·dS`` between them; the total is
    unchanged). Hermes-3 fixes the gauge physically by floating φ against
    ``phi_wall`` at the sheath, so the number is meaningful — but it moves if
    ``phi_wall`` is non-zero or the core-boundary φ relaxation shifts the
    level, and it is not comparable across runs with different φ references.

    Parameters
    ----------
    ds : xarray.Dataset
    phi : xarray.DataArray
        Electrostatic potential, same normalisation as ``ds``.
    P_charged : xarray.DataArray
        Sum of pressures over **charged** species (``Pe + Pi`` for a pure
        hydrogenic plasma). Hermes-3 collects the diamagnetic current over
        every species that has a charge, so neutral pressure is excluded.
    bxcvx, Bxy : xarray.DataArray
        From :func:`load_grid_vars`.

    Returns
    -------
    xarray.DataArray
        Radial flux density of the diamagnetic work term.
    """
    sqrt_g11 = np.sqrt(_no_t(ds["g11"]))
    return phi * P_charged * 2.0 * bxcvx / (Bxy * sqrt_g11)


def radial_magnetic_drift_heat_flux(ds, Pe, Pi, bxcvx, Bxy, Te_eV=None, Ti_eV=None):
    """
    Radial heat flux from the magnetic (curvature/grad-B) drift for both species.

        q_mag_s = (5/2) · P_s · v_mag_s    [W/m²]

    Electron and ion contributions are computed separately and returned as a
    tuple so the caller can inspect each or sum them.

    Parameters
    ----------
    ds          : xarray.Dataset
    Pe, Pi      : xarray.DataArray  electron/ion pressure [Pa]
    bxcvx       : xarray.DataArray  from grid file (load_grid_vars)
    Bxy         : xarray.DataArray  total |B| [T] from grid file
    Te_eV : xarray.DataArray
        Electron temperature — ds['Te'] in normalised units, or Pe/(Ne·e) in SI.
    Ti_eV : xarray.DataArray, optional
        Ion temperature — ds['Ti'] if available. If None, computed as Pi/Ni
        (correct in normalised units; gives Ti[J] in SI which is consistent).

    Returns
    -------
    q_mag_e, q_mag_i, q_mag_total : xarray.DataArray  [W/m²]
    """
    if Te_eV is None or Ti_eV is None:
        raise ValueError(
            "Te_eV and Ti_eV must be provided. "
            "Use ds['Te']/ds['Ti'] if available, otherwise Pi/Ni for the ion species."
        )

    from .decompose import toroidal_mean
    from .load import detect_toroidal_dim
    zdim = detect_toroidal_dim(ds)

    def zmean(da):
        return toroidal_mean(da, dim=zdim) if zdim in da.dims else da

    # Geometric factor: opposite sign for ions vs electrons (opposite charge)
    #   v_mag_s = -T_s/(q_s) * Curl(b/B)_r  →  electrons (q=-e): negative, ions (q=+e): positive
    sqrt_g11 = np.sqrt(_no_t(ds["g11"]))
    geom_e = -2.0 * bxcvx / (Bxy * sqrt_g11)
    geom_i = +2.0 * bxcvx / (Bxy * sqrt_g11)

    Pe_mean = zmean(Pe)
    Pi_mean = zmean(Pi)
    Te_mean = zmean(Te_eV)
    Ti_mean = zmean(Ti_eV)

    # Total: <Pe · Te>_ζ · geom  (direct product, same pattern as ExB)
    q_mag_e_total = 2.5 * zmean(Pe * Te_eV) * geom_e
    q_mag_i_total = 2.5 * zmean(Pi * Ti_eV) * geom_i

    # Mean: <Pe>_ζ · <Te>_ζ · geom
    q_mag_e_mean  = 2.5 * Pe_mean * Te_mean * geom_e
    q_mag_i_mean  = 2.5 * Pi_mean * Ti_mean * geom_i

    # Turbulent: residual (captures <δPe · δTe> contribution)
    q_mag_e_turb  = q_mag_e_total - q_mag_e_mean
    q_mag_i_turb  = q_mag_i_total - q_mag_i_mean

    q_mag_e = q_mag_e_total
    q_mag_i = q_mag_i_total

    return q_mag_e, q_mag_i, q_mag_e + q_mag_i


def radial_magnetic_drift_particle_flux(ds, Ne, Ni, bxcvx, Bxy, Te_eV, Ti_eV):
    """
    Radial particle flux from the magnetic (curvature/grad-B) drift for both species.

    From Hermes-3 diamagnetic_drift.cxx (line 84):
        ∂N/∂t -= ∇·(N · v_D)   where  v_D = (T/q) · Curlb_B

    Radial particle flux per species:
        Γ_mag_s = N_s · v_mag_s = N_s · (T_s / q_s) · 2·bxcvx / Bxy / sqrt(g11)

    The electron and ion contributions are returned separately; they have
    opposite sign (opposite charge) and a single "total" sum is not a
    meaningful transport quantity, so it is not returned.

    Parameters
    ----------
    ds           : xarray.Dataset
    Ne, Ni       : xarray.DataArray  electron/ion density [normalised]
    bxcvx        : xarray.DataArray  from grid file (load_grid_vars)
    Bxy          : xarray.DataArray  from grid file
    Te_eV, Ti_eV : xarray.DataArray  temperatures [normalised]

    Returns
    -------
    Gamma_mag_e, Gamma_mag_i : xarray.DataArray
    """
    zdim = detect_toroidal_dim(ds)

    def zmean(da):
        return toroidal_mean(da, dim=zdim) if zdim in da.dims else da

    sqrt_g11 = np.sqrt(_no_t(ds["g11"]))
    geom_e = -2.0 * bxcvx / (Bxy * sqrt_g11)
    geom_i = +2.0 * bxcvx / (Bxy * sqrt_g11)

    Gamma_mag_e = zmean(Ne * Te_eV) * geom_e
    Gamma_mag_i = zmean(Ni * Ti_eV) * geom_i

    return Gamma_mag_e, Gamma_mag_i


def _curl_b_B_poloidal(ds, bxcvx, bxcvy, Bxy):
    """
    Poloidal projection of Curl(b/B), from xstorm Curl_b_B(direction='poloidal'):

        2/(B·J·√(g11·g_33)) · [ (−g_23·g_13 + g_33·g_12)·bxcvx
                                + (g_33·g_22 − g_23²)·bxcvy ]

    The bxcvz term has coefficient (−g_23·g_33 + g_33·g_23) = 0 and drops out, so
    only bxcvx and bxcvy are needed. The radial projection (2·bxcvx/(B·√g11)) is
    the one used by the radial_magnetic_drift_* functions, so the two are
    consistent components of the same drift.
    """
    J, g11, g_22, g_33, g_23, g_12, g_13 = (
        _no_t(ds[k]) for k in ("J", "g11", "g_22", "g_33", "g_23", "g_12", "g_13")
    )
    return (
        2.0
        / (Bxy * J * np.sqrt(g11 * g_33))
        * (
            (-g_23 * g_13 + g_33 * g_12) * bxcvx
            + (g_33 * g_22 - g_23 ** 2) * bxcvy
        )
    )


def poloidal_magnetic_drift_particle_flux(ds, Ne, Ni, bxcvx, bxcvy, Bxy, Te_eV, Ti_eV):
    """
    Poloidal particle flux from the magnetic (curvature/grad-B) drift.

    Physical poloidal magnetic-drift velocity per species:
        v_mag_pol_s = (T_s / q_s) · Curl_b_B_poloidal
    (electrons q=−e → −T_e·curl; ions q=+e → +T_i·curl), giving

        Γ_mag_pol_s = N_s · v_mag_pol_s

    This is the poloidal counterpart of radial_magnetic_drift_particle_flux,
    required so that the magnetic drift enters a ∇·Γ conservation check on both
    the radial and poloidal faces (otherwise the local residual carries the
    missing poloidal divergence).

    Parameters
    ----------
    ds           : xarray.Dataset
    Ne, Ni       : xarray.DataArray  electron/ion density
    bxcvx, bxcvy : xarray.DataArray  curvature components from load_grid_vars
    Bxy          : xarray.DataArray  total |B| [T] from the grid file
    Te_eV, Ti_eV : xarray.DataArray  temperatures

    Returns
    -------
    Gamma_mag_pol_e, Gamma_mag_pol_i : xarray.DataArray
    """
    zdim = detect_toroidal_dim(ds)

    def zmean(da):
        return toroidal_mean(da, dim=zdim) if zdim in da.dims else da

    curl_pol = _curl_b_B_poloidal(ds, bxcvx, bxcvy, Bxy)
    Gamma_mag_pol_e = -zmean(Ne * Te_eV) * curl_pol
    Gamma_mag_pol_i = +zmean(Ni * Ti_eV) * curl_pol
    return Gamma_mag_pol_e, Gamma_mag_pol_i


def poloidal_magnetic_drift_heat_flux(ds, Pe, Pi, bxcvx, bxcvy, Bxy, Te_eV, Ti_eV):
    """
    Poloidal enthalpy flux from the magnetic (curvature/grad-B) drift:

        q_mag_pol_s = (5/2) P_s · v_mag_pol_s

    Poloidal counterpart of radial_magnetic_drift_heat_flux.

    Returns
    -------
    q_mag_pol_e, q_mag_pol_i : xarray.DataArray  [W/m²]
    """
    zdim = detect_toroidal_dim(ds)

    def zmean(da):
        return toroidal_mean(da, dim=zdim) if zdim in da.dims else da

    curl_pol = _curl_b_B_poloidal(ds, bxcvx, bxcvy, Bxy)
    q_mag_pol_e = -2.5 * zmean(Pe * Te_eV) * curl_pol
    q_mag_pol_i = +2.5 * zmean(Pi * Ti_eV) * curl_pol
    return q_mag_pol_e, q_mag_pol_i


def ExB_poloidal_velocity(ds, phi):
    """
    Physical poloidal component of the ExB drift velocity.

    Two-step calculation following STORM v2_fluxY.py (lines 125-131):

    1. Contravariant theta-component of ExB:
       v^θ_ExB = (g_{xζ}/g_{θθ}) · ∂φ/∂x - (g^{xθ}/g_{θθ}) · ∂φ/∂ζ

    2. Convert to physical poloidal velocity:
       v_ExB_pol = v^θ_ExB · (g_{θθ}·g_{ζζ} - g_{θζ}²) / (J · sqrt(g^{xx}·g_{ζζ}))

    Parameters
    ----------
    ds : xarray.Dataset
    phi : xarray.DataArray
        Electrostatic potential [V].

    Returns
    -------
    xarray.DataArray
        Physical poloidal ExB velocity [m/s].
    """
    dx_phi = phi.bout.ddx()   # ∂φ/∂x  [V]
    dz_phi = phi.bout.ddz()   # ∂φ/∂ζ  [V]

    J, g11, g_22, g_33, g_23, g_12 = (
        _no_t(ds[k]) for k in ("J", "g11", "g_22", "g_33", "g_23", "g_12")
    )
    vEB_theta = dx_phi * (g_23 / g_22) - dz_phi * (g_12 / g_22)

    return (
        vEB_theta
        * (g_33 * g_22 - g_23 ** 2)
        / (J * np.sqrt(g11 * g_33))
    )


def parallel_to_poloidal(ds, v_par):
    """
    Project parallel velocity onto the physical poloidal direction.

    v_pol = v_{∥} · (g_{θθ}·g_{ζζ} - g_{θζ}²) / (J · sqrt(g^{xx}·g_{θθ}·g_{ζζ}))

    This formula (from STORM v2_fluxY.py, lines 168-173) gives the poloidal
    component of the parallel flow, used for poloidal particle/heat flux.

    Parameters
    ----------
    ds : xarray.Dataset
    v_par : xarray.DataArray
        Parallel velocity v_{∥s} [m/s].

    Returns
    -------
    xarray.DataArray
        Physical poloidal velocity [m/s].
    """
    J, g11, g_22, g_33, g_23 = (
        _no_t(ds[k]) for k in ("J", "g11", "g_22", "g_33", "g_23")
    )
    return (
        v_par
        * (g_33 * g_22 - g_23 ** 2)
        / (J * np.sqrt(g11 * g_22 * g_33))
    )


# ---------------------------------------------------------------------------
# Radial fluxes (across flux surfaces, x = const)
# ---------------------------------------------------------------------------

def radial_particle_flux(ds, n, phi, include_y_derivs=True):
    """
    Radial ExB particle flux density: Γ_r = n · v_ExB_r  [m⁻²s⁻¹].

    This is the dominant cross-field particle transport mechanism in tokamak
    turbulence. Includes all toroidal modes (mean + fluctuating).

    Parameters
    ----------
    ds : xarray.Dataset
    n  : xarray.DataArray
        Species number density [m⁻³].
    phi : xarray.DataArray
        Electrostatic potential [V].
    include_y_derivs : bool, default True
        Forwarded to :func:`ExB_radial_velocity`; see its docstring.

    Returns
    -------
    xarray.DataArray  [m⁻²s⁻¹]
    """
    return n * ExB_radial_velocity(ds, phi, include_y_derivs=include_y_derivs)


def radial_heat_flux(ds, P, phi, include_y_derivs=True):
    """
    Radial ExB enthalpy (heat) flux density: q_r = (5/2) P · v_ExB_r  [Wm⁻²].

    From the Hermes-3 pressure equation (eq. 2/8 in the physics PDF), the
    convective energy flux carried by the ExB drift is (5/2) P · v_E.
    The factor 5/2 = 3/2 (internal energy) + 1 (PdV work) = enthalpy flux
    for an ideal gas.

    Parameters
    ----------
    ds  : xarray.Dataset
    P   : xarray.DataArray
        Species pressure [Pa].
    phi : xarray.DataArray
        Electrostatic potential [V].
    include_y_derivs : bool, default True
        Forwarded to :func:`ExB_radial_velocity`; see its docstring.

    Returns
    -------
    xarray.DataArray  [Wm⁻²]
    """
    return 2.5 * P * ExB_radial_velocity(ds, phi, include_y_derivs=include_y_derivs)


def decompose_radial_particle_flux(ds, n, phi, zdim=None, include_y_derivs=True):
    """
    Decompose the radial particle flux into mean-flow and turbulent parts.

    The mean is the full ensemble average over toroidal angle ζ and, if the
    inputs carry a time dimension, over time as well. The turbulent part is the
    residual correlation of the fluctuations about that full mean:

      Γ_total = <<n · v_ExB_r>_ζ>_t
      Γ_mean  = <<n>_ζ>_t · <<v_ExB_r>_ζ>_t
      Γ_turb  = Γ_total − Γ_mean = <δn · δv_ExB_r>   (fluctuations in ζ and t)

    With no time dimension this reduces to the single-snapshot toroidal
    Reynolds decomposition. Averaging over time *inside* the decomposition is
    required: time-averaging Γ_turb computed per snapshot would instead give
    <<n>_ζ·<v>_ζ>_t for the mean term, which omits the temporal correlation of
    the zonal/mean flow that physically belongs to the turbulent flux.

    Parameters
    ----------
    ds  : xarray.Dataset
    n   : xarray.DataArray
        Species number density [m⁻³].
    phi : xarray.DataArray
        Electrostatic potential [V].
    zdim : str, optional
        Toroidal dimension name. Auto-detected if not given.
    include_y_derivs : bool, default True
        Forwarded to :func:`ExB_radial_velocity`; see its docstring.

    Returns
    -------
    Gamma_mean : xarray.DataArray  [m⁻²s⁻¹]
        Mean-flow contribution <<n>_ζ>_t · <<v_r>_ζ>_t.
    Gamma_turb : xarray.DataArray  [m⁻²s⁻¹]
        Turbulent contribution Γ_total − Γ_mean.
    Gamma_total : xarray.DataArray  [m⁻²s⁻¹]
        Total mean flux = Gamma_mean + Gamma_turb.
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    v_r = ExB_radial_velocity(ds, phi, include_y_derivs=include_y_derivs)

    def avg(da):
        da = toroidal_mean(da, dim=zdim)
        return da.mean("t") if "t" in da.dims else da

    Gamma_total = avg(n * v_r)
    Gamma_mean  = avg(n) * avg(v_r)
    Gamma_turb  = Gamma_total - Gamma_mean

    return Gamma_mean, Gamma_turb, Gamma_total


def decompose_radial_heat_flux(ds, P, phi, zdim=None, include_y_derivs=True):
    """
    Decompose the radial enthalpy flux into mean-flow and turbulent parts.

    Mean is the full ensemble average over ζ and (if present) time, the same
    convention as decompose_radial_particle_flux:

      q_total = (5/2) <<P · v_ExB_r>_ζ>_t
      q_mean  = (5/2) <<P>_ζ>_t · <<v_ExB_r>_ζ>_t
      q_turb  = q_total − q_mean

    Parameters
    ----------
    ds  : xarray.Dataset
    P   : xarray.DataArray
        Species pressure [Pa].
    phi : xarray.DataArray
        Electrostatic potential [V].
    zdim : str, optional
        Toroidal dimension name. Auto-detected if not given.
    include_y_derivs : bool, default True
        Forwarded to :func:`ExB_radial_velocity`; see its docstring.

    Returns
    -------
    q_mean  : xarray.DataArray  [Wm⁻²]
    q_turb  : xarray.DataArray  [Wm⁻²]
    q_total : xarray.DataArray  [Wm⁻²]
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    v_r = ExB_radial_velocity(ds, phi, include_y_derivs=include_y_derivs)

    def avg(da):
        da = toroidal_mean(da, dim=zdim)
        return da.mean("t") if "t" in da.dims else da

    q_total = 2.5 * avg(P * v_r)
    q_mean  = 2.5 * avg(P) * avg(v_r)
    q_turb  = q_total - q_mean

    return q_mean, q_turb, q_total


# ---------------------------------------------------------------------------
# Anomalous perpendicular transport
# ---------------------------------------------------------------------------

def radial_flutter_fluxes(ds, Ne, Pe, Ve, Pi, Vi, Apar, zdim=None):
    """
    Radial particle and heat flux from electromagnetic flutter (field-line braiding).

    When Apar is non-zero, field lines are perturbed and parallel flows project
    onto the radial direction. From Hermes-3 evolve_density/pressure.cxx:

        Γ_flutter_s = -⟨Ns · V_∥s · ∂Apar/∂ζ⟩_ζ / √g11
        q_flutter_s = -(5/2) · ⟨Ps · V_∥s · ∂Apar/∂ζ⟩_ζ / √g11

    This is the same formula as ExB but with the effective potential
    φ_eff = -V_∥ · Apar replacing φ for each species.

    Parameters
    ----------
    ds            : xarray.Dataset
    Ne, Pe, Ve    : electron density [m⁻³], pressure [Pa], parallel velocity [m/s]
    Pi, Vi        : ion pressure [Pa], parallel velocity [m/s]
    Apar          : xarray.DataArray  parallel vector potential [normalised or SI]
    zdim          : str, optional  toroidal dimension name

    Returns
    -------
    Gamma_flutter : xarray.DataArray  total electron radial particle flux [m⁻²s⁻¹]
    q_flutter     : xarray.DataArray  total (e+i) radial heat flux [Wm⁻²]
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    dApar_dz = Apar.bout.ddz()
    sqrt_g11 = np.sqrt(_no_t(ds["g11"]))

    # Radial flutter "velocity" for each species: -V_par * dApar/dz / sqrt(g11)
    # (same sign convention as ExB: phi.bout.ddz()/sqrt(g11))
    Gamma_flutter = toroidal_mean(-Ne * Ve * dApar_dz, dim=zdim) / sqrt_g11
    q_flutter_e   = toroidal_mean(-2.5 * Pe * Ve * dApar_dz, dim=zdim) / sqrt_g11
    q_flutter_i   = toroidal_mean(-2.5 * Pi * Vi * dApar_dz, dim=zdim) / sqrt_g11
    q_flutter     = q_flutter_e + q_flutter_i

    return Gamma_flutter, q_flutter


def radial_anomalous_heat_flux(ds, N, T, chi, D=0.0, zdim=None):
    """
    Radial anomalous heat flux matching Hermes-3 AnomalousDiffusion component.

    From anomalous_diffusion.cxx (lines 119, 127):
        chi term: Div_a_Grad_perp(chi * N_DC, T_DC)  →  q_chi = -chi * N * sqrt(g11) * dT/dx
        D term:   Div_a_Grad_perp((3/2)*T_DC*D, N_DC) →  q_D   = -(3/2) * T * D * sqrt(g11) * dN/dx

    Both act on toroidal-mean (DC) fields. Combined radial component:
        q_r = -chi * N̄ * sqrt(g11) * dT̄/dx  −  (3/2) * T̄ * D * sqrt(g11) * dN̄/dx

    Parameters
    ----------
    ds   : xarray.Dataset
    N    : xarray.DataArray  species density [normalised]
    T    : xarray.DataArray  species temperature [normalised]
    chi  : float  anomalous thermal diffusivity (normalised, BOUT.inp value / (rho_s0² * Omega_ci))
    D    : float  anomalous particle diffusivity (normalised), default 0
    zdim : str, optional

    Returns
    -------
    xarray.DataArray  radial anomalous heat flux (2D in x,theta)
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    def zmean(da):
        return toroidal_mean(da, dim=zdim) if zdim in da.dims else da

    N_mean = zmean(N)
    T_mean = zmean(T)
    sqrt_g11 = np.sqrt(_no_t(ds["g11"]))

    q = -chi * N_mean * sqrt_g11 * T_mean.bout.ddx()
    if D != 0.0:
        q = q - 1.5 * T_mean * D * sqrt_g11 * N_mean.bout.ddx()
    return q


def radial_anomalous_particle_flux(ds, N, D, zdim=None):
    """
    Radial anomalous particle flux matching Hermes-3 AnomalousDiffusion:
        Γ_D = -D * ∇_perp N_DC  →  Γ_r = -D * sqrt(g11) * dN̄/dx

    Parameters
    ----------
    ds   : xarray.Dataset
    N    : xarray.DataArray  density [normalised]
    D    : float  anomalous diffusivity (normalised, BOUT.inp value / (rho_s0² * Omega_ci))
    zdim : str, optional

    Returns
    -------
    xarray.DataArray  radial anomalous particle flux (2D in x,theta)
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    def zmean(da):
        return toroidal_mean(da, dim=zdim) if zdim in da.dims else da

    N_mean = zmean(N)
    return -D * np.sqrt(_no_t(ds["g11"])) * N_mean.bout.ddx()


# ---------------------------------------------------------------------------
# Poloidal fluxes (along field lines, projected onto the poloidal direction)
# ---------------------------------------------------------------------------

def poloidal_particle_flux(ds, n, v_par):
    """
    Poloidal particle flux from parallel flow: Γ_pol = n · v_{∥} · (Bp/B).

    The parallel flow carries particles along field lines. The poloidal
    component (projected onto the poloidal direction) is obtained via the
    parallel-to-poloidal projection using the BOUT++ metric tensors.

    Note: this does NOT include the ExB poloidal component, which is
    separately computed by ExB_poloidal_velocity().

    Parameters
    ----------
    ds    : xarray.Dataset
    n     : xarray.DataArray
        Species number density [m⁻³].
    v_par : xarray.DataArray
        Species parallel velocity v_{∥} [m/s].

    Returns
    -------
    xarray.DataArray  [m⁻²s⁻¹]
        Poloidal particle flux density.
    """
    return n * parallel_to_poloidal(ds, v_par)


def poloidal_heat_flux(ds, P, v_par):
    """
    Poloidal convective heat (enthalpy) flux from parallel flow:
    q_pol = (5/2) P · v_{∥,pol}  [Wm⁻²].

    This is the dominant heat transport mechanism along field lines toward
    the divertor targets.

    Parameters
    ----------
    ds    : xarray.Dataset
    P     : xarray.DataArray
        Species pressure [Pa].
    v_par : xarray.DataArray
        Species parallel velocity v_{∥} [m/s].

    Returns
    -------
    xarray.DataArray  [Wm⁻²]
    """
    return 2.5 * P * parallel_to_poloidal(ds, v_par)


def parallel_conductive_heat_flux(ds, T, kappa_par):
    """
    Parallel conductive heat flux: q_{∥,cond} = -κ_{∥} b·∇T  [Wm⁻²].

    This is the Spitzer-Härm heat conduction along field lines. The gradient
    is taken in the field-aligned y/theta direction.

    Parameters
    ----------
    ds       : xarray.Dataset
    T        : xarray.DataArray
        Species temperature [eV].
    kappa_par : float or xarray.DataArray
        Parallel thermal conductivity [W/(m eV)], consistent with T in eV.
        No additional elementary-charge factor is applied.

    Returns
    -------
    xarray.DataArray  [Wm⁻²]

    Notes
    -----
    xBOUT's ddy() returns a coordinate derivative, not a derivative per metre.
    Divide by sqrt(g_22) to obtain b·grad(T). For SI output g_22 must be in m²
    and kappa_par must have the physical units above; a raw normalised
    conductivity saved by a solver needs conversion before use here.
    """
    T = _ensure_z_single_chunk(T, detect_toroidal_dim(ds))
    dT_dpol = T.bout.ddy()   # ∂T/∂θ in BOUT++ coordinates [eV]
    return -kappa_par * dT_dpol / np.sqrt(_no_t(ds["g_22"]))


def total_poloidal_particle_flux(ds, n, v_par, phi, zdim=None):
    """
    Total toroidal-mean poloidal particle flux including both parallel
    flow and poloidal ExB drift contributions.

    Γ_pol_total = n · v_{∥,pol} + n · v_ExB_pol

    Parameters
    ----------
    ds    : xarray.Dataset
    n     : xarray.DataArray  [m⁻³]
    v_par : xarray.DataArray  [m/s]  parallel velocity
    phi   : xarray.DataArray  [V]    electrostatic potential
    zdim  : str, optional

    Returns
    -------
    xarray.DataArray  [m⁻²s⁻¹]  toroidal-mean total poloidal particle flux
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    F_par = n * parallel_to_poloidal(ds, v_par)
    F_exb = n * ExB_poloidal_velocity(ds, phi)
    return toroidal_mean(F_par + F_exb, dim=zdim)


def total_poloidal_heat_flux(ds, P, v_par, phi, zdim=None):
    """
    Total toroidal-mean poloidal heat flux including parallel flow and ExB.

    q_pol_total = (5/2) P · (v_{∥,pol} + v_ExB_pol)

    Parameters
    ----------
    ds    : xarray.Dataset
    P     : xarray.DataArray  [Pa]
    v_par : xarray.DataArray  [m/s]
    phi   : xarray.DataArray  [V]
    zdim  : str, optional

    Returns
    -------
    xarray.DataArray  [Wm⁻²]  toroidal-mean total poloidal heat flux
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    q_par = 2.5 * P * parallel_to_poloidal(ds, v_par)
    q_exb = 2.5 * P * ExB_poloidal_velocity(ds, phi)
    return toroidal_mean(q_par + q_exb, dim=zdim)

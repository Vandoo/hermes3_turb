"""
Particle and energy balance / conservation checks.

The particle continuity equation integrated over a volume V bounded by
flux surfaces x1 < x < x2 and poloidal planes θ1 < θ < θ2 reads:

  d/dt ∫∫∫ n dV = - ∮ Γ_r · dA_x + ∮ Γ_pol · dA_θ + ∫∫∫ S_n dV

In steady state (or time-averaged), the volume source must equal the net
surface flux. This is the key conservation check.

Similarly for energy:
  ∫∫∫ S_E dV = ∮ q_r · dA_x + ∮ q_pol · dA_θ + (sheath losses)

The STORM validation approach (v2_fconserv_plo.py) checks:
  d(F_pol)/dθ  ↔  F_x  (local conservation at each poloidal slice)

This module provides functions for both the global (integrated) and
local (poloidal profile) conservation checks.
"""

import numpy as np
from .geometry import (
    radial_area_element_dtheta,
    poloidal_area_element_dx,
)
from .decompose import toroidal_mean
from .load import detect_toroidal_dim, detect_poloidal_dim, _no_t


def integrate_radial_surface(ds, flux_density, x_idx, zdim=None):
    """
    Integrate a radial flux density over the flux surface at x = x_idx.

    Φ_x = ∫∫ flux_density · sqrt(g_{θθ}·g_{ζζ} - g_{θζ}²) dθ dζ

    Following STORM, the integral is performed over the toroidal-mean flux
    (fluxes should already be toroidal-mean, 2D in (x,theta)).

    Parameters
    ----------
    ds : xarray.Dataset
    flux_density : xarray.DataArray
        Radial flux density (toroidal mean, 2D or 3D). [m⁻²s⁻¹] or [Wm⁻²].
    x_idx : int or slice
        Radial index (or slice) to select.
    zdim : str, optional

    Returns
    -------
    float or xarray.DataArray
        Integrated flux through the full torus [s⁻¹] or [W].
    """
    pdim = detect_poloidal_dim(ds)
    elem = radial_area_element_dtheta(ds)  # dy·√(g_22·g_33 − g_23²) [m²/rad]
    result = (flux_density * elem).isel(x=x_idx).sum(dim=pdim)
    # Collapse any residual toroidal/time dim (the area element keeps zeta even
    # for a toroidal-mean flux); keep only a radial 'x' axis if x_idx is a slice.
    leftover = [d for d in result.dims if d != "x"]
    if leftover:
        result = result.mean(dim=leftover)
    # Σ_θ gives flux per unit toroidal angle; ×2π → full torus (as in
    # total_target_power / radial_flux_profile).
    return result * 2 * np.pi


def integrate_poloidal_surface(ds, flux_density, theta_idx, zdim=None):
    """
    Integrate a poloidal flux density over the cross-section at θ = theta_idx.

    Φ_θ = ∫∫ flux_density · sqrt(g_{xx}·g_{ζζ} - g_{xζ}²) dx dζ

    Parameters
    ----------
    ds : xarray.Dataset
    flux_density : xarray.DataArray
        Poloidal flux density (toroidal mean). [m⁻²s⁻¹] or [Wm⁻²].
    theta_idx : int or slice
        Poloidal index (or slice) to select.
    zdim : str, optional

    Returns
    -------
    float or xarray.DataArray
        Integrated poloidal flux through the full torus [s⁻¹] or [W].
    """
    pdim = detect_poloidal_dim(ds)
    elem = poloidal_area_element_dx(ds)   # dx·√(g_11·g_33 − g_13²) [m²/rad]
    result = (flux_density * elem).isel({pdim: theta_idx}).sum(dim="x")
    # Collapse any residual toroidal/time dim; keep the poloidal axis if
    # theta_idx is a slice.
    leftover = [d for d in result.dims if d != pdim]
    if leftover:
        result = result.mean(dim=leftover)
    # Σ_x gives flux per unit toroidal angle; ×2π → full torus.
    return result * 2 * np.pi


def volume_integral(ds, source_density, zdim=None):
    """
    Integrate a volumetric source density over the full torus.

    ∫∫∫ S dV = 2π · Σ_{x,θ} <S>_ζ · J · dx · dy

    The toroidal-mean source is integrated against the per-radian volume
    element J·dx·dy and scaled by 2π for the full torus — the same convention
    as the flux integrals (radial_flux_profile, total_target_power), so the
    source and net flux are directly comparable regardless of zperiod.

    Parameters
    ----------
    ds : xarray.Dataset
    source_density : xarray.DataArray
        Volumetric source [m⁻³s⁻¹] or [Wm⁻³].
    zdim : str, optional

    Returns
    -------
    float or xarray.DataArray
        Total integrated source over the full torus [s⁻¹] or [W]
        (time-averaged if a time dimension is present).
    """
    if zdim is None:
        try:
            zdim = detect_toroidal_dim(ds)
        except ValueError:
            zdim = None

    pdim = detect_poloidal_dim(ds)
    J, dx, dy = (_no_t(ds[k]) for k in ("J", "dx", "dy"))
    dV_dzeta = J * dx * dy                     # volume per unit toroidal angle [m³/rad]
    integrand = source_density * dV_dzeta
    # toroidal-mean (so ×2π gives the full torus, independent of zperiod)
    if zdim is not None and zdim in integrand.dims:
        integrand = integrand.mean(dim=zdim)
    sum_dims = [d for d in ("x", pdim) if d in integrand.dims]
    result = integrand.sum(dim=sum_dims) * 2 * np.pi
    # steady-state balance: average over time if the source carries a t axis
    if "t" in getattr(result, "dims", ()):
        result = result.mean("t")
    return result


def radial_flux_profile(ds, flux_density, theta_slices, zdim=None):
    """
    Compute the radial profile of surface-integrated flux at each x.

    For each radial index x_i, integrates flux_density over the poloidal
    range defined by theta_slices.

    Parameters
    ----------
    ds : xarray.Dataset
    flux_density : xarray.DataArray
        Toroidal-mean radial flux density (2D in x,theta).
    theta_slices : slice or list of int
        Poloidal range to integrate over (e.g., core, SOL region).
    zdim : str, optional

    Returns
    -------
    xarray.DataArray
        Total radial flux through each flux surface, full torus,
        as a function of x [s⁻¹] or [W].
    """
    pdim = detect_poloidal_dim(ds)
    elem = radial_area_element_dtheta(ds)   # dy·√(g_22·g_33 − g_23²) [m²/rad]
    # Integrate over the poloidal band (the area element already carries dy).
    result = (flux_density * elem).isel({pdim: theta_slices}).sum(dim=pdim)
    # The area element keeps the toroidal dim even when the flux density is a
    # toroidal mean, so a zeta axis survives the poloidal sum. Average out
    # everything left except the radial dim 'x' to get a single 1-D profile.
    leftover = [d for d in result.dims if d != "x"]
    if leftover:
        result = result.mean(dim=leftover)
    # Σ_θ gives flux per unit toroidal angle; ×2π → full torus (the toroidal-mean
    # flux is representative regardless of zperiod), as in total_target_power.
    return result * 2 * np.pi


def poloidal_flux_profile(ds, flux_density, x_slice=None, zdim=None):
    """
    Compute the poloidal profile of surface-integrated flux at each θ.

    Useful for identifying stagnation points and checking that
    d(F_pol)/dθ = F_x (local conservation).

    Parameters
    ----------
    ds : xarray.Dataset
    flux_density : xarray.DataArray
        Toroidal-mean poloidal flux density (2D in x,theta).
    x_slice : int, slice, or None
        If given, select only these radial indices before integrating over x.
        Typically x_slice = slice(ixseps1, -2) for SOL-only integration.
    zdim : str, optional

    Returns
    -------
    xarray.DataArray
        Total poloidal flux through the full torus as a function of θ
        [s⁻¹] or [W].
    """
    pdim = detect_poloidal_dim(ds)
    elem = poloidal_area_element_dx(ds)   # dx·√(g_11·g_33 − g_13²) [m²/rad]
    integrand = flux_density * elem
    if x_slice is not None:
        integrand = integrand.isel(x=x_slice)
    result = integrand.sum(dim="x")
    # Collapse any residual toroidal/time dim, keep the poloidal axis.
    leftover = [d for d in result.dims if d != pdim]
    if leftover:
        result = result.mean(dim=leftover)
    # Σ_x gives flux per unit toroidal angle; ×2π → full torus.
    return result * 2 * np.pi


def diamagnetic_work_identity(ds, phi, P_charged, curlb, tind=-1,
                              branch_rows=(), mxg=None):
    """
    Validate :func:`hermes3_turb.diamagnetic_work_radial_flux` against the
    identity it rests on, Dudson 2026 eqs. (B.9)→(B.10)::

        −φ ∇·(P C)  −  P C·∇φ   =   −∇·(φ P C)        with C = ∇×(b/B)

    ⚠ **The identity part validates almost nothing about the physics, and the
    DivJdia part is the whole point of this function.** This was measured, not
    assumed. ``∇·(fg) = f∇·g + g·∇f`` is a *pure product-rule identity that
    holds for ANY vector field* ``g``, so it is blind to whether ``g`` is the
    correct curvature vector. Injecting faults into ``Curlb_B`` on the SXD
    nx32 case moved the residual not at all (median 3.20e-3 when correct):
    dropping the y component 0.73×, dropping z 1.07×, flipping the sign of x
    0.98×, scaling x by 2 0.89×, omitting the ``2/B`` factor 1.00×. All of
    those would be serious errors and none is detected. Treat ``rel_*`` as a
    **self-test of the discrete operators only** — it confirms the centred
    differences, the Jacobian placement and the periodic z wrap are mutually
    consistent, which is worth knowing but was never the doubtful part.

    The **only** check here that constrains ``Curlb_B`` is
    ``divJdia_ratio_median``: ``Div(Jdia)`` computed here against Hermes-3's
    own ``DivJdia`` diagnostic, which must be 1. It requires
    ``[vorticity] diagnose = true``; a single-output re-run is enough to
    produce it. Until that is run, the diamagnetic work term is **not
    validated** — the formula's provenance in the source is the only
    assurance.

    Both sides are formed pointwise from the same centred finite differences
    and the same Jacobian, using the BOUT++ curvilinear divergence
    ``∇·A = (1/J)[∂_x(J A^x) + ∂_y(J A^y) + ∂_z(J A^z)]`` on the
    *contravariant* components.

    Rows adjacent to a branch cut have no true y-neighbour, so the y-derivative
    is meaningless there and those rows are excluded, as are the x-guards and
    the first/last radial and poloidal cells. z is periodic and wrapped.

    The check uses one snapshot (``tind``), since the identity is pointwise and
    instantaneous.

    Parameters
    ----------
    ds : xarray.Dataset
    phi, P_charged : xarray.DataArray
        Potential and the sum of pressures over charged species.
    curlb : dict
        Output of :func:`hermes3_turb.load_curvature_vector` — keys 'x','y','z'
        already carrying the ``2/B`` factor.
    tind : int
        Time index to check.
    branch_rows : iterable of int
        Poloidal rows adjacent to X-point / inter-region cuts, to exclude.
    mxg : int, optional
        Radial guard count; taken from metadata when omitted.

    Returns
    -------
    dict
        ``rel_median`` / ``rel_p99`` / ``rel_max`` — |LHS−RHS| normalised by
        the RMS of RHS over the checked region — plus ``n_checked``,
        ``divJdia_ratio_median`` (vs the dump's ``DivJdia``, or NaN) and
        ``twist``.
    """
    pdim = detect_poloidal_dim(ds)
    zdim = detect_toroidal_dim(ds)
    m = ds.metadata
    if mxg is None:
        mxg = int(m.get("MXG", 2))

    def arr(da):
        d = da.isel(t=tind) if "t" in getattr(da, "dims", ()) else da
        d = _no_t(d) if "t" in getattr(d, "dims", ()) else d
        if zdim in d.dims:
            return np.asarray(d.transpose("x", pdim, zdim).values, dtype=float)
        return np.asarray(d.transpose("x", pdim).values, dtype=float)[:, :, None]

    ph, P = arr(phi), arr(P_charged)
    J = arr(ds["J"])
    dx, dy, dz = arr(ds["dx"]), arr(ds["dy"]), arr(ds["dz"])
    cx, cy, cz = (arr(curlb["x"]), arr(curlb["y"]), arr(curlb["z"]))

    def d_dx(f):
        o = np.full_like(f, np.nan)
        o[1:-1] = (f[2:] - f[:-2]) / (2.0 * dx[1:-1])
        return o

    def d_dy(f):
        o = np.full_like(f, np.nan)
        o[:, 1:-1] = (f[:, 2:] - f[:, :-2]) / (2.0 * dy[:, 1:-1])
        return o

    def d_dz(f):
        return (np.roll(f, -1, 2) - np.roll(f, 1, 2)) / (2.0 * dz)

    def div(ax, ay, az):
        return (d_dx(J * ax) + d_dy(J * ay) + d_dz(J * az)) / J

    div_Jdia = div(P * cx, P * cy, P * cz)                 # ∇·(P C)
    C_dot_grad_phi = cx * d_dx(ph) + cy * d_dy(ph) + cz * d_dz(ph)
    lhs = -ph * div_Jdia - P * C_dot_grad_phi
    rhs = -div(ph * P * cx, ph * P * cy, ph * P * cz)

    nx, ny = lhs.shape[0], lhs.shape[1]
    keep_x = np.zeros(nx, bool); keep_x[mxg + 1:nx - mxg - 1] = True
    keep_y = np.zeros(ny, bool); keep_y[1:ny - 1] = True
    for j in branch_rows:
        for jj in (int(j) - 1, int(j), int(j) + 1):
            if 0 <= jj < ny:
                keep_y[jj] = False
    mask = keep_x[:, None, None] & keep_y[None, :, None]
    mask = mask & np.isfinite(lhs) & np.isfinite(rhs)

    scale = float(np.sqrt(np.mean(rhs[mask] ** 2))) if mask.any() else np.nan
    rel = np.abs(lhs[mask] - rhs[mask]) / (scale if scale else 1.0)
    out = dict(
        rel_median=float(np.median(rel)) if rel.size else float("nan"),
        rel_p99=float(np.percentile(rel, 99)) if rel.size else float("nan"),
        rel_max=float(rel.max()) if rel.size else float("nan"),
        n_checked=int(mask.sum()),
        twist=bool(curlb.get("twist", False)),
        divJdia_ratio_median=float("nan"),
    )

    # Absolute check on Curlb_B, when the run saved Hermes-3's own DivJdia.
    if "DivJdia" in ds:
        code = arr(ds["DivJdia"])
        good = mask & (np.abs(code) > 0.05 * np.nanmax(np.abs(code[mask])))
        if good.any():
            out["divJdia_ratio_median"] = float(
                np.median(div_Jdia[good] / code[good]))
    return out


def particle_balance(
    ds,
    n,
    v_par,
    phi,
    source_density=None,
    x_inner=None,
    x_outer=None,
    theta_inner_sol=None,
    theta_outer_sol=None,
    zdim=None,
    radial_flux=None,
):
    """
    Global particle balance check for one species.

    Computes:
      - Radial particle flux through inner flux surface (x = x_inner)
      - Radial particle flux through outer flux surface (x = x_outer)
      - Total volume particle source (if provided)

    In steady state: F_x(x_outer) - F_x(x_inner) = Volume_source + Target_flux

    Parameters
    ----------
    ds : xarray.Dataset
    n  : xarray.DataArray  [m⁻³]
    v_par : xarray.DataArray  [m/s]  parallel velocity
    phi   : xarray.DataArray  [V]
    source_density : xarray.DataArray, optional
        Net particle source [m⁻³s⁻¹] (ionisation minus recombination).
    x_inner, x_outer : int, optional
        Radial indices for the inner/outer bounding flux surfaces.
    theta_inner_sol, theta_outer_sol : slice, optional
        Poloidal ranges for the inner (HFS) and outer (LFS) SOL regions.
    zdim : str, optional
    radial_flux : xarray.DataArray, optional
        Precomputed radial particle flux density (already toroidal-/time-mean,
        2-D in x,theta). If given it is used directly instead of the ExB-only
        flux — pass the sum of ExB + magnetic-drift + anomalous (+ flutter) to
        include all channels. Defaults to the ExB flux from n, phi.

    Returns
    -------
    dict with keys:
        'Fx_inner'     : [s⁻¹] radial particle rate through x_inner surface
        'Fx_outer'     : [s⁻¹] radial particle rate through x_outer surface
        'net_Fx'       : [s⁻¹] F_x(outer) - F_x(inner)
        'volume_source': [s⁻¹] total particle source (if given)
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    if radial_flux is None:
        from .fluxes import radial_particle_flux
        Gamma_r = toroidal_mean(radial_particle_flux(ds, n, phi), dim=zdim)
    else:
        Gamma_r = radial_flux

    pdim = detect_poloidal_dim(ds)
    result = {}

    if theta_inner_sol is None:
        theta_inner_sol = slice(None)
    if theta_outer_sol is None:
        theta_outer_sol = slice(None)

    elem = radial_area_element_dtheta(ds)

    def _surface_flux(x_idx, theta_slice):
        # full-torus radial flux through x_idx over a poloidal sub-range
        r = (Gamma_r * elem).isel(x=x_idx).isel({pdim: theta_slice}).sum(pdim)
        leftover = list(r.dims)            # collapse residual zeta / time
        if leftover:
            r = r.mean(dim=leftover)
        return float(r) * 2 * np.pi

    if x_inner is not None:
        result["Fx_inner_HFS"] = _surface_flux(x_inner, theta_inner_sol)
        result["Fx_inner_LFS"] = _surface_flux(x_inner, theta_outer_sol)

    if x_outer is not None:
        result["Fx_outer_HFS"] = _surface_flux(x_outer, theta_inner_sol)
        result["Fx_outer_LFS"] = _surface_flux(x_outer, theta_outer_sol)

    if x_inner is not None and x_outer is not None:
        result["net_Fx_HFS"] = result["Fx_outer_HFS"] - result["Fx_inner_HFS"]
        result["net_Fx_LFS"] = result["Fx_outer_LFS"] - result["Fx_inner_LFS"]

    if source_density is not None:
        result["volume_source"] = float(volume_integral(ds, source_density, zdim=zdim))

    return result


def energy_balance(
    ds,
    P,
    v_par,
    phi,
    source_density=None,
    x_inner=None,
    x_outer=None,
    theta_inner_sol=None,
    theta_outer_sol=None,
    zdim=None,
    radial_flux=None,
):
    """
    Global energy balance check for one species.

    Computes the toroidal-mean radial heat flux (enthalpy flux) through
    bounding flux surfaces and compares with the volume heating source.

    Parameters
    ----------
    ds : xarray.Dataset
    P  : xarray.DataArray  [Pa]  species pressure
    v_par : xarray.DataArray  [m/s]
    phi   : xarray.DataArray  [V]
    source_density : xarray.DataArray, optional
        Volume heating source [Wm⁻³] (e.g. Pe:source from BOUT.inp).
    x_inner, x_outer : int, optional
    theta_inner_sol, theta_outer_sol : slice, optional
    zdim : str, optional
    radial_flux : xarray.DataArray, optional
        Precomputed radial heat flux density (already toroidal-/time-mean, 2-D
        in x,theta). If given it is used directly instead of the ExB-only flux —
        pass ExB + magnetic-drift + anomalous (+ flutter) to include all
        channels. Defaults to the ExB enthalpy flux from P, phi.

    Returns
    -------
    dict with keys:
        'Qx_inner_HFS', 'Qx_inner_LFS'  : [W] heat flux through x_inner
        'Qx_outer_HFS', 'Qx_outer_LFS'  : [W] heat flux through x_outer
        'net_Qx_HFS', 'net_Qx_LFS'      : [W] net radial heat transport
        'volume_source'                   : [W] total heating (if given)
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)

    if radial_flux is None:
        from .fluxes import radial_heat_flux
        q_r = toroidal_mean(radial_heat_flux(ds, P, phi), dim=zdim)
    else:
        q_r = radial_flux

    pdim = detect_poloidal_dim(ds)
    result = {}

    if theta_inner_sol is None:
        theta_inner_sol = slice(None)
    if theta_outer_sol is None:
        theta_outer_sol = slice(None)

    elem = radial_area_element_dtheta(ds)

    def _surface_flux(x_idx, theta_slice):
        # full-torus radial heat flux through x_idx over a poloidal sub-range
        r = (q_r * elem).isel(x=x_idx).isel({pdim: theta_slice}).sum(pdim)
        leftover = list(r.dims)            # collapse residual zeta / time
        if leftover:
            r = r.mean(dim=leftover)
        return float(r) * 2 * np.pi

    if x_inner is not None:
        result["Qx_inner_HFS"] = _surface_flux(x_inner, theta_inner_sol)
        result["Qx_inner_LFS"] = _surface_flux(x_inner, theta_outer_sol)

    if x_outer is not None:
        result["Qx_outer_HFS"] = _surface_flux(x_outer, theta_inner_sol)
        result["Qx_outer_LFS"] = _surface_flux(x_outer, theta_outer_sol)

    if x_inner is not None and x_outer is not None:
        result["net_Qx_HFS"] = result["Qx_outer_HFS"] - result["Qx_inner_HFS"]
        result["net_Qx_LFS"] = result["Qx_outer_LFS"] - result["Qx_inner_LFS"]

    if source_density is not None:
        result["volume_source"] = float(volume_integral(ds, source_density, zdim=zdim))

    return result

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
import xarray as xr
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


def guard_rows(ds, ny_grid=None):
    """
    Poloidal indices of the y-boundary guard rows, ``[]`` if none were kept.

    The count of guard rows per band is derived from the *dump* against
    ``metadata["ny"]``, **not** from ``MYG``. The two are equal in the usual
    case but not by construction: a grid built with ``y_boundary_guards=1``
    gives ``ny+4`` rows while ``MYG`` reports 2, so a ``MYG``-keyed test such
    as ``npol == ny + 4*MYG`` matches neither layout and strips nothing.

    Layout (this codebase's usual one, and BOUT++'s): one guard band before
    the first target, ``2g`` rows between the two upper targets of a double
    null, and one band after the last target — so a DN dump is ``ny + 4g``
    rows and a SN dump ``ny + 2g``.

    Parameters
    ----------
    ds : xarray.Dataset
    ny_grid : int, optional
        Interior poloidal size. Defaults to ``metadata["ny"]``; pass the grid
        file's own ``ny`` when that is the authority (what the `v3_*` loaders
        do, since the metadata can be rewritten by a reshaped load).

    Returns
    -------
    list of int
        Poloidal indices to exclude. Empty when the dump has no y-guards.
    """
    pdim = detect_poloidal_dim(ds)
    m = getattr(ds, "metadata", {}) or {}
    ny_arr = int(ds.sizes[pdim])
    ny_grid = int(m.get("ny", ny_arr)) if ny_grid is None else int(ny_grid)
    extra = ny_arr - ny_grid
    if extra <= 0:
        return []

    ny_inner = int(m.get("ny_inner", 0) or 0)
    j21 = int(m.get("jyseps2_1", -1))
    j12 = int(m.get("jyseps1_2", -1))
    is_dn = ny_inner > 0 and j21 != j12

    if is_dn and extra % 4 == 0:
        g = extra // 4
        return (list(range(g))
                + list(range(ny_inner + g, ny_inner + 3 * g))
                + list(range(ny_grid + 3 * g, ny_grid + 4 * g)))
    if extra % 2 == 0:
        g = extra // 2
        return list(range(g)) + list(range(ny_grid + g, ny_grid + 2 * g))
    raise ValueError(
        "cannot infer the y-guard layout: dump has %d poloidal rows against "
        "ny=%d in metadata (%d extra), which is neither ny+4g (double null) "
        "nor ny+2g (single null)" % (ny_arr, ny_grid, extra))


def interior_mask(ds, strip_yguards=True):
    """
    Boolean ``(x, θ)`` mask of the interior cells — guards excluded.

    Volume integrals of sources run over interior cells only: a guard cell
    carries a full volume element and, on some grids, a nonzero source, so
    including it inflates the integral (on `from_archer_6kW` the x-guards
    alone add +32.6 kW, +5.3 %). `dx` is separatrix-packed in these grids, so
    the outer x-guards are wide and the effect is larger than 4/nx.

    Parameters
    ----------
    ds : xarray.Dataset
    strip_yguards : bool, optional
        Also drop the y-boundary guard rows (default True). They are only
        present when the dump kept them; see :func:`guard_rows`.

    Returns
    -------
    numpy.ndarray
        Boolean array of shape ``(nx, nθ)``, True on interior cells.
    """
    pdim = detect_poloidal_dim(ds)
    m = getattr(ds, "metadata", {}) or {}
    nx_arr = int(ds.sizes["x"])
    ny_arr = int(ds.sizes[pdim])
    mxg = int(m.get("MXG", 2))

    mask = np.zeros((nx_arr, ny_arr), dtype=bool)
    mask[mxg:nx_arr - mxg, :] = True
    if strip_yguards:
        rows = guard_rows(ds)
        if rows:
            mask[:, np.asarray(rows, dtype=int)] = False
    return mask


def volume_integral(ds, source_density, zdim=None, interior_only=True):
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
    interior_only : bool, optional
        Sum over interior cells only, dropping the x- and y-boundary guards
        (default True, and the only correct choice for a source integral —
        see :func:`interior_mask`). ``False`` sums every cell, guards
        included.

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
    if interior_only and "x" in integrand.dims and pdim in integrand.dims:
        mask = interior_mask(ds)
        integrand = integrand.where(
            xr.DataArray(mask, dims=("x", pdim)), other=0.0)
    sum_dims = [d for d in ("x", pdim) if d in integrand.dims]
    result = integrand.sum(dim=sum_dims) * 2 * np.pi
    # steady-state balance: average over time if the source carries a t axis
    if "t" in getattr(result, "dims", ()):
        result = result.mean("t")
    return result


def particle_source(ds, species="e"):
    """
    The particle source ``S<sp>_src`` [m⁻³s⁻¹], or None if the dump lacks it.

    No factor: Hermes-3 evolves N, so ``S<sp>_src`` is already dN/dt.
    """
    name = f"S{species}_src"
    return ds[name] if name in ds else None


def energy_source(ds, species="e", factor=1.5):
    """
    The ENERGY source ``3/2·P<sp>_src``, or None if the dump lacks it.

    **The 3/2 is the whole point of the function.** Hermes-3 evolves the
    *pressure* P, so ``P<sp>_src`` is dP/dt, whereas the internal energy
    density is ``3/2 P``; the energy source is therefore ``3/2·P<sp>_src``.
    The other two terms of an energy budget carry their factors already — the
    storage term is ``3/2 d∫P dV/dt`` and the radial flux is the 5/2 enthalpy
    flux — so a budget built on the raw pressure source closes 1.5× high.

    Parameters
    ----------
    ds : xarray.Dataset
    species : str, optional
        Species label, e.g. ``"e"``, ``"i"``, ``"d+"``.
    factor : float, optional
        The 3/2. Exposed only so a caller that wants the raw pressure source
        can pass 1.0 to integrate the pressure source without conversion.

    Returns
    -------
    xarray.DataArray or None
        ``factor · P<sp>_src``, or None when that variable is not in the dump.
    """
    name = f"P{species}_src"
    return factor * ds[name] if name in ds else None


def parallel_energy_flow_ylow(ds, species="e", include_conduction=True,
                              include_viscous_heating=False, verbose=False):
    """Select parallel Y-face energy-flow channels [W].

    The published ef<sp>_tot_ylow includes advection, conduction, and viscous
    heating diagnostics. Numerical viscous heating can be diagnosed without
    being applied to the pressure equation, so it is excluded by default.
    Include it only for runs with numerical_viscous_heating=true.

    Conduction is also registered as an energy_source. Exclude it here when
    the volume side of a budget already includes the corresponding SP term.

    Parameters
    ----------
    ds : xarray.Dataset
    species : str, optional
        Species label, e.g. ``"e"``, ``"i"``, ``"d+"``.
    include_conduction : bool, optional
        Add ``ef<sp>_cond_ylow``. Default True (the full parallel flow). Set
        False for a budget whose volume side already has ``SP``.
    include_viscous_heating : bool, optional
        Add ``ef<sp>_visc_heat_ylow``. Default False, and only ever correct
        when the run really set ``numerical_viscous_heating = true`` — which
        the dump does not record, so the caller must know.
    verbose : bool, optional
        Print which channels were found and what ``ef<sp>_tot_ylow`` carries
        beyond them.

    Returns
    -------
    xarray.DataArray or None
        The summed Y-face flow, or None when the dump has no ``_adv_ylow``
        for this species.
    """
    adv = ds.get(f"ef{species}_adv_ylow")
    if adv is None:
        return None
    used = ["adv"]
    flow = adv
    if include_conduction and f"ef{species}_cond_ylow" in ds:
        flow = flow + ds[f"ef{species}_cond_ylow"]
        used.append("cond")
    if include_viscous_heating and f"ef{species}_visc_heat_ylow" in ds:
        flow = flow + ds[f"ef{species}_visc_heat_ylow"]
        used.append("visc_heat")
    if verbose:
        msg = "ef{}_*_ylow: using {}".format(species, " + ".join(used))
        if f"ef{species}_tot_ylow" in ds:
            extra = [n for n in ("cond", "visc_heat")
                     if f"ef{species}_{n}_ylow" in ds and n not in used]
            msg += "; ef{}_tot_ylow additionally carries {}".format(
                species, ", ".join(extra) if extra else "nothing")
        print(msg)
    return flow


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
    """Evaluate the diamagnetic-work product identity and compare Div(Jdia).

        -phi div(P C) - P C.grad(phi) = -div(phi P C), C = curl(b/B)

    The product identity tests consistency of the discrete derivatives and
    Jacobian placement; it does not independently validate C. The optional
    divJdia_ratio_median comparison uses Hermes-3's DivJdia diagnostic from
    [vorticity] diagnose=true and should approach unity for matching operators.

    The calculation uses centred differences on contravariant components.
    Branch-cut neighbours, boundary cells, and x-guards are excluded; z is
    periodic. tind selects one snapshot. See Dudson 2026, eqs. B.9-B.10.

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


# ----------------------------------------------------------------------
# Closed energy budget from the dump's own reaction/boundary channels
#
#
# WHY THIS EXISTS.  `check_source` in inspect_hermes_run.py tables every
# `R*`/`E*` channel per region, which answers "where". It deliberately does
# not SUM them, because summing requires knowing which channels leave the
# box and which merely move energy between species inside it. Adding them
# all up double-counts badly. This function encodes that classification so
# the question "is P_in actually balanced?" has a reproducible answer.
#
# THE CLASSIFICATION, and the source lines it comes from (branch
# neutral-precon-models):
#
#   LOSS  — energy leaves the simulation domain permanently
#     R<sp>_ex        line radiation AND the ionisation potential. These are
#                     one channel, not two: izn_rec_reaction.cxx:175 is
#                     commented "Electron energy loss (radiation, ionisation
#                     potential)" and :180-190 evaluates sigma_vE then
#                     subtracts `electron_heating`. So NEVER add a separate
#                     13.6 eV x S_iz term on top — it is already in here.
#     R<sp>_rec       recombination radiation.
#     E<n>_target_refl / E<n>_wall_refl
#                     neutral kinetic+thermal energy deposited on the plate
#                     and the wall (neutral_boundary.cxx:369,380 — these are
#                     the `cooling_source` accumulated at :152,214,270,329).
#                     This is where CX energy ACTUALLY leaves.
#     E<n>_pump       energy removed with pumped neutrals.
#     E<sp>_sheath    sheath heat flux (sheath_boundary_simple.cxx:751).
#                     Only present with [sheath_boundary_simple] diagnose =
#                     true; absent from most of this campaign's dumps, in
#                     which case it falls into `residual` below.
#
#   INPUT — energy enters from outside
#     P<sp>_src       the prescribed heating. PRESSURE source [Pa/s], so it
#                     takes the 3/2; R*/E* are already [W/m^3] and must not.
#     E<n>_target_recycle / E<n>_wall_recycle
#                     CAVEAT, and it is why `fast_recycle_fraction` is an
#                     argument. recycling.cxx:281-286 blends two parts:
#                       fast:    sheath_ion_heat_flow x energy_factor x f
#                       thermal: recycle_particle_flow x (1-f) x recycle_energy
#                     The fast part RETURNS energy the sheath already took
#                     (a transfer); the thermal part is a PRESCRIBED energy
#                     per returning atom (default 3 eV, this campaign 3.5 eV)
#                     created from nothing — a genuine input. With
#                     target_fast_recycle_fraction = 0 (the default, and
#                     commented out in this campaign's BOUT.inp) the channel
#                     is 100% input. Pass the real f to split it correctly.
#
#   TRANSFER — internal; cancels between species and must be EXCLUDED
#     E<a><b>_cx      THE TRAP THIS FUNCTION EXISTS FOR. Charge exchange
#                     moves energy from ions to neutrals inside the domain.
#                     It is not a loss. The loss happens when that neutral
#                     reaches a surface, which is already counted as
#                     E<n>_wall_refl / E<n>_target_refl. Counting `_cx` as a
#                     sink double-counts the entire CX pathway.
#     E<sp>_iz, E<sp>_rec    plasma<->neutral energy transfer (note the
#                     contrast with R<sp>_rec, which IS a loss: same suffix,
#                     different leading letter, opposite classification).
#     E*_coll*, E*_VgradP, E*_PdivV, E*_Dpar, E*_nvh
#                     collisional exchange and compression work.
#
# SIGN CONVENTION. Every channel is integrated signed, exactly as the dump
# stores it, so losses come out negative and the budget is a plain sum. No
# abs() anywhere — a channel whose sign is the opposite of its class is a
# finding, not something to paper over, and `warnings` reports it.
# ----------------------------------------------------------------------

#: Suffixes classified as internal transfers when carried by an `E*` variable.
_TRANSFER_SUFFIX = ("_cx", "_iz", "_rec", "_coll", "_coll_friction",
                    "_VgradP", "_PdivV", "_Dpar", "_nvh")
#: Suffixes classified as genuine losses when carried by an `E*` variable.
_LOSS_SUFFIX = ("_target_refl", "_wall_refl", "_pump", "_sheath")
#: Suffixes classified as external inputs when carried by an `E*` variable.
_INPUT_SUFFIX = ("_target_recycle", "_wall_recycle", "_src")


def classify_energy_channel(var):
    """
    Classify a dump variable as an energy input, loss or internal transfer.

    Returns one of ``"input"``, ``"loss"``, ``"transfer"`` or ``None`` (not an
    energy channel at all). See the module comment above for the physics and
    the Hermes-3 source lines behind each rule.

    The leading letter matters as much as the suffix: ``R<sp>_rec`` is
    recombination *radiation* (a loss) while ``E<sp>_rec`` is the *energy
    transfer* of the same reaction (internal). Classifying on the suffix
    alone gets one of the two wrong.

    Parameters
    ----------
    var : str
        Dump variable name, e.g. ``"Rd+_ex"``, ``"Edd+_cx"``, ``"Pe_src"``.

    Returns
    -------
    str or None
    """
    if not var:
        return None
    # Pressure sources are the prescribed heating: [Pa/s], need the 3/2.
    if var.startswith("P") and var.endswith("_src"):
        return "input"
    # Radiation: photons leave the box, always a loss.
    if var.startswith("R"):
        return "loss"
    if var.startswith("E"):
        # Order matters: "_target_recycle" must be tested before "_rec"
        # would ever be reached, and "_coll_friction" before "_coll".
        if var.endswith(_INPUT_SUFFIX):
            return "input"
        if var.endswith(_LOSS_SUFFIX):
            return "loss"
        if var.endswith(_TRANSFER_SUFFIX):
            return "transfer"
        return "transfer"      # unknown E* channel: internal until shown otherwise
    return None


def energy_sink_budget(ds, interior=None, zdim=None, fast_recycle_fraction=0.0,
                       tind=-1, species_pressures=None):
    """Sum signed energy inputs, losses, and storage from saved diagnostics.

    R*, E*, and P*_src channels are classified by classify_energy_channel:

        residual = inputs + losses - dW/dt  [kW]

    Internal transfers are reported separately and excluded from the residual.
    When E*_sheath is absent, the residual includes unresolved sheath exchange
    as well as unaccounted channels. A budget alone does not establish steady
    state; dW/dt describes storage during relaxation.

    The default mask excludes x-guards. Supply interior explicitly when the
    dataset retains y-guards or requires a different integration region.

    Parameters
    ----------
    ds : xarray.Dataset
        Loaded Hermes-3 dump, unnormalised (`R*`/`E*` in W/m^3, `P*_src` in
        Pa/s). Needs `J`, `dx`, `dy` for the volume element.
    interior : numpy.ndarray, optional
        Boolean mask of shape (nx, ny) selecting the cells to integrate.
        Defaults to stripping `MXG` guards in x and nothing in y, which is
        right for a dump loaded with `keep_yboundaries=False`. If the dump
        retains y-guards, pass a mask — the default would include them.
    zdim : str, optional
        Toroidal dimension; auto-detected.
    fast_recycle_fraction : float, optional
        `target_fast_recycle_fraction` from BOUT.inp (default 0, matching
        Hermes-3's own default). The `E*_recycle` channels are split
        `(1-f)` input / `f` transfer, per recycling.cxx:281-286. Use the value from the simulation input.
    tind : int, optional
        Frame for the channel integrals (default -1, the last). `dW/dt` uses
        the full time axis regardless.
    species_pressures : sequence of str, optional
        Pressure variables forming the stored energy W = 3/2 integral(sum P)dV.
        Defaults to every `P<sp>` present that is not a `_src`.

    Returns
    -------
    dict
        `channels`  : list of (name, class, power_kW), descending by |power|
        `inputs_kW`, `losses_kW`, `transfers_kW` : signed class totals
        `dWdt_kW`   : stored-energy rate, or None with fewer than 2 frames
        `residual_kW`, `sheath_diagnosed`, `W_J`, `frame`, `warnings`

    Notes
    -----
    `transfers_kW` is reported but NOT in the residual, by construction.

    Do NOT read a large transfer subtotal as an error. One might expect these
    to cancel across species, but Hermes-3 writes only ONE SIDE of them:
      * for a SYMMETRIC reaction such as `d + d+ -> d+ + d`,
        cx_reaction.cxx:131 (the `else` branch) registers a single
        `E<r1><r2>_cx` keyed by p1 — the equal-and-opposite partner term is
        never written to the dump;
      * the electron side of ionisation is not an `E*` channel at all, it is
        folded into `R<sp>_ex` and therefore classified a loss.
    Consequently the reported transfer subtotal need not cancel between species.
    """
    pdim = detect_poloidal_dim(ds)
    if zdim is None:
        try:
            zdim = detect_toroidal_dim(ds)
        except ValueError:
            zdim = None
    m = getattr(ds, "metadata", {}) or {}
    warnings = []

    # --- interior mask -------------------------------------------------
    nx_arr = ds.sizes["x"]
    ny_arr = ds.sizes[pdim]
    if interior is None:
        mxg = int(m.get("MXG", 2))
        interior = np.zeros((nx_arr, ny_arr), dtype=bool)
        interior[mxg:nx_arr - mxg, :] = True
        ny_grid = int(m.get("ny", ny_arr))
        if ny_arr > ny_grid:
            warnings.append(
                "dump has %d poloidal rows against ny=%d in metadata: it "
                "retains y-guards, which the default mask does NOT strip. "
                "Pass `interior` explicitly." % (ny_arr, ny_grid))
    interior = np.asarray(interior, dtype=bool)
    if interior.shape != (nx_arr, ny_arr):
        raise ValueError(
            "interior mask is %s but the dump is (x=%d, %s=%d)"
            % (interior.shape, nx_arr, pdim, ny_arr))

    J, dx, dy = (_no_t(ds[k]) for k in ("J", "dx", "dy"))
    dV_dzeta = np.asarray((J * dx * dy).transpose("x", pdim).values)
    weight = np.where(interior, dV_dzeta, 0.0) * 2 * np.pi    # full torus [m^3]

    def _integrate(var, factor):
        """Signed full-torus volume integral of `var` at `tind`, in kW."""
        da = ds[var]
        if "t" in da.dims:
            da = da.isel(t=tind)
        if zdim is not None and zdim in da.dims:
            da = da.mean(zdim)          # z-MEAN pairs with the 2pi above
        f = np.asarray(da.transpose("x", pdim).values)
        return float(np.nansum(f * weight) * factor)

    # --- channels ------------------------------------------------------
    channels = []
    totals = {"input": 0.0, "loss": 0.0, "transfer": 0.0}
    for var in ds.data_vars:
        cls = classify_energy_channel(var)
        if cls is None:
            continue
        # P*_src is [Pa/s] -> 3/2 for energy, then W->kW. R*/E* are already
        # [W/m^3] and take only the 1e-3. Mixing these up is a flat 50 %.
        factor = 1.5e-3 if var.startswith("P") else 1e-3
        val = _integrate(var, factor)
        if cls == "input" and var.endswith(("_target_recycle", "_wall_recycle")) \
                and fast_recycle_fraction:
            # Split per recycling.cxx:281-286: the fast part is a transfer.
            f = float(fast_recycle_fraction)
            channels.append((var + " [thermal]", "input", val * (1.0 - f)))
            channels.append((var + " [fast]", "transfer", val * f))
            totals["input"] += val * (1.0 - f)
            totals["transfer"] += val * f
            continue
        channels.append((var, cls, val))
        totals[cls] += val

    for name, cls, val in channels:
        if cls == "loss" and val > 0:
            warnings.append("%s is classified a loss but integrates POSITIVE "
                            "(%+.3g kW) — check the sign convention" % (name, val))
        if cls == "input" and val < 0:
            warnings.append("%s is classified an input but integrates NEGATIVE "
                            "(%+.3g kW)" % (name, val))

    # --- stored energy and its rate ------------------------------------
    if species_pressures is None:
        species_pressures = [v for v in ds.data_vars
                             if v.startswith("P") and not v.endswith("_src")
                             and "x" in ds[v].dims]
    W_J = None
    dWdt_kW = None
    if species_pressures:
        Psum = sum(ds[v] for v in species_pressures)
        if zdim is not None and zdim in Psum.dims:
            Psum = Psum.mean(zdim)
        Parr = np.asarray(Psum.transpose(*( ["t"] if "t" in Psum.dims else [] ),
                                         "x", pdim).values)
        if "t" in Psum.dims:
            W = 1.5 * np.nansum(Parr * weight[None, :, :], axis=(1, 2))
            W_J = float(W[tind])
            t = np.asarray(ds["t"].values, dtype=float)
            if W.size >= 2 and t.size == W.size and t[-1] > t[0]:
                dWdt_kW = float((W[-1] - W[0]) / (t[-1] - t[0]) * 1e-3)
            else:
                warnings.append(
                    "dW/dt needs >=2 frames with a usable time axis; this "
                    "folder has %d, so the residual below INCLUDES storage "
                    "and is not a closure test." % W.size)
        else:
            W_J = float(1.5 * np.nansum(Parr * weight))
            warnings.append("no time axis on the pressures: dW/dt unavailable")

    sheath_diagnosed = any(v.endswith("_sheath") for v in ds.data_vars)
    if not sheath_diagnosed:
        warnings.append(
            "no E*_sheath in this dump ([sheath_boundary_simple] diagnose is "
            "not set), so the sheath/target load is NOT a diagnosed channel "
            "and sits inside `residual_kW`.")

    residual = totals["input"] + totals["loss"] - (dWdt_kW or 0.0)

    channels.sort(key=lambda c: -abs(c[2]))
    return {
        "channels": channels,
        "inputs_kW": totals["input"],
        "losses_kW": totals["loss"],
        "transfers_kW": totals["transfer"],
        "dWdt_kW": dWdt_kW,
        "residual_kW": residual,
        "sheath_diagnosed": sheath_diagnosed,
        "W_J": W_J,
        "frame": int(ds.sizes.get("t", 1) - 1 if tind == -1 else tind),
        "warnings": warnings,
    }

def leg_regions(ds):
    """Return each leg as {name: (x_slice, y0, y1, target_at)}.

    y0:y1 is a half-open range of dump poloidal indices. target_at is 'start'
    or 'end'; the opposite face is the X-point boundary. Guard offsets are
    inferred from the dump size: +g for the inner segment and +3g for the outer
    segment in double-null geometry. jyseps* marks inclusive region endpoints.

    Each leg covers the radial interior, slice(MXG, nx-MXG), including the PFR.
    Upper legs are absent in single-null geometry (jyseps2_1 == jyseps1_2).
    """
    m = ds.metadata
    pdim = detect_poloidal_dim(ds)
    ny, nx = ds.sizes[pdim], ds.sizes["x"]
    mxg = int(m.get("MXG", 2))
    myg = int(m.get("MYG", 2))
    ny_grid = int(m.get("ny", ny))
    nyi = int(m.get("ny_inner", 0))
    g = myg if ny > ny_grid else 0

    j11, j21 = int(m["jyseps1_1"]), int(m["jyseps2_1"])
    j12, j22 = int(m["jyseps1_2"]), int(m["jyseps2_2"])
    xsl = slice(mxg, nx - mxg)

    legs = {
        "inner lower leg": (xsl, g, j11 + 1 + g, "start"),
        "outer lower leg": (xsl, j22 + 1 + 3 * g, ny_grid + 3 * g, "end"),
    }
    if j21 != j12:                                  # double null only
        legs["inner upper leg"] = (xsl, j21 + 1 + g, nyi + g, "end")
        legs["outer upper leg"] = (xsl, nyi + 3 * g, j12 + 1 + 3 * g, "start")
    return {k: v for k, v in legs.items() if v[2] > v[1]}


def leg_particle_budget(ds, radial_flux, poloidal_flux=None, sources=None,
                        n=None, zdim=None):
    """
    Per-leg particle budget: is a draining leg starved of SUPPLY or over-drained?

    For each divertor leg, closes

        dN/dt  =  [Φ_x(inner face) − Φ_x(outer face)]        radial delivery
                + [Φ_θ(start face) − Φ_θ(end face)]          parallel delivery
                + Σ ∫ S dV                                   in-leg sources/sinks

    The question this answers: a leg can empty either because too little is
    delivered to it or because too much is removed from it, and the in-leg
    ionisation, pump and sheath terms alone cannot tell those apart. If the
    sources and sinks balance but the leg still drains, the radial delivery is
    the fault — and on an electrostatic run that points at φ.

    Parameters
    ----------
    ds : xarray.Dataset
    radial_flux : xarray.DataArray
        Radial particle flux density [m⁻²s⁻¹], toroidal-mean, 2-D in (x, θ).
        Pass the sum of the required channels (ExB +
        magnetic-drift + anomalous, and flutter if electromagnetic) — the
        ExB term alone is not the delivery.
    poloidal_flux : xarray.DataArray, optional
        Poloidal particle flux density [m⁻²s⁻¹] (`fluxes.poloidal_particle_flux`,
        i.e. v_pol = v_∥·Bp/B). Omitted → the parallel terms are reported as
        NaN and the residual is not meaningful; a leg budget without the
        target outflow is not closed.
    sources : dict, optional
        {label: DataArray [m⁻³s⁻¹]}, e.g.
        {"S_iz": ds["Sd+_iz"], "S_rec": ds["Sd+_rec"], "S_pump": ds["Sd_pump"]}.
        Signs are taken as stored (Hermes-3 writes sinks negative), so they are
        summed, never subtracted.
    n : xarray.DataArray, optional
        Density [m⁻³]. With a time axis of ≥2 frames the storage term dN/dt is
        estimated by a finite difference over the dump's full time span, which
        is a secant over that window and not an instantaneous rate.
    zdim : str, optional

    Returns
    -------
    dict
        {leg_name: {"Fx_inner", "Fx_outer", "net_Fx", "Fpol_start",
                    "Fpol_end", "net_Fpol", "F_target", "F_xpoint",
                    <source labels>, "source_total", "dNdt", "residual",
                    "N"}}
        All rates in s⁻¹, N in particles. `residual` is
        (delivery + sources) − dN/dt; divide by the largest term to judge it.

    Notes
    -----
    A flux is evaluated at the boundary CELL CENTRE, not the cell face, which
    is the same approximation `particle_balance` makes. Expect a few per cent
    from that alone, so read the residual as "does this close to O(10 %)",
    not as a precision test.

    **Time convention.** Source and flux terms are TIME-AVERAGED over the dump
    when they carry a `t` axis, following `volume_integral` and the rest of
    this module (a steady-state balance), and `dN/dt` is the secant over the
    same window — so the two are consistently averaged over one interval.
    `inspect_hermes_run.py`'s `check_source` reaction table instead reports the
    LAST frame, because there it is a snapshot rather than a budget. The two
    therefore differ by ~1 % per-leg on a slowly drifting run; verified on
    `mw0.8_npm_m1/data/9`, where passing `.isel(t=-1)` sources here reproduces
    that table to every printed digit (`Sd+_iz` 4.101e20 / 2.126e21 / 1.549e20
    / 1.675e21 for inner-lower / outer-lower / inner-upper / outer-upper).

    **Charged species only.** `S<sp>_pump` is a NEUTRAL sink: it removes atoms,
    not ions. Passing it in a plasma budget adds a term that acts on the ion
    density only indirectly (by starving the ionisation source), and makes the
    residual meaningless. The neutral budget is a separate calculation needing
    the neutral fluxes.
    """
    if zdim is None:
        try:
            zdim = detect_toroidal_dim(ds)
        except ValueError:
            zdim = None
    pdim = detect_poloidal_dim(ds)

    elem_r = radial_area_element_dtheta(ds)
    elem_p = poloidal_area_element_dx(ds)
    J, dx, dy = (_no_t(ds[k]) for k in ("J", "dx", "dy"))
    dV_dzeta = J * dx * dy

    def _collapse(r):
        leftover = list(r.dims)
        if leftover:
            r = r.mean(dim=leftover)
        return float(r) * 2 * np.pi

    def _rad(x_idx, y0, y1):
        return _collapse((radial_flux * elem_r)
                         .isel(x=x_idx).isel({pdim: slice(y0, y1)}).sum(pdim))

    def _pol(y_idx, xsl):
        return _collapse((poloidal_flux * elem_p)
                         .isel({pdim: y_idx}).isel(x=xsl).sum("x"))

    def _vol(field, xsl, y0, y1):
        integrand = (field * dV_dzeta).isel(x=xsl).isel({pdim: slice(y0, y1)})
        if zdim is not None and zdim in integrand.dims:
            integrand = integrand.mean(dim=zdim)
        r = integrand.sum(dim=[d for d in ("x", pdim) if d in integrand.dims])
        if "t" in getattr(r, "dims", ()):
            r = r.mean("t")
        return float(r) * 2 * np.pi

    out = {}
    for name, (xsl, y0, y1, target_at) in leg_regions(ds).items():
        d = {}
        # Radial faces: the flux density is positive OUTWARD in x, so the net
        # delivery into the leg is (what enters at the inner face) minus (what
        # leaves at the outer face).
        d["Fx_inner"] = _rad(xsl.start, y0, y1)
        d["Fx_outer"] = _rad(xsl.stop - 1, y0, y1)
        d["net_Fx"] = d["Fx_inner"] - d["Fx_outer"]

        if poloidal_flux is not None:
            d["Fpol_start"] = _pol(y0, xsl)
            d["Fpol_end"] = _pol(y1 - 1, xsl)
            d["net_Fpol"] = d["Fpol_start"] - d["Fpol_end"]
            # Label the two faces physically: one is the target (sheath
            # outflow), the other opens onto the upstream SOL at the X-point.
            d["F_target"] = (d["Fpol_start"] if target_at == "start"
                             else d["Fpol_end"])
            d["F_xpoint"] = (d["Fpol_end"] if target_at == "start"
                             else d["Fpol_start"])
        else:
            for k in ("Fpol_start", "Fpol_end", "net_Fpol",
                      "F_target", "F_xpoint"):
                d[k] = float("nan")

        total_src = 0.0
        for label, field in (sources or {}).items():
            d[label] = _vol(field, xsl, y0, y1)
            total_src += d[label]
        d["source_total"] = total_src

        d["N"] = float("nan")
        d["dNdt"] = float("nan")
        if n is not None:
            d["N"] = _vol(n.isel(t=-1) if "t" in n.dims else n, xsl, y0, y1)
            if "t" in n.dims and n.sizes["t"] > 1 and "t" in ds.coords:
                tv = np.asarray(ds["t"].values, dtype=float)
                dt = float(tv[-1] - tv[0])
                if dt > 0:
                    n0 = _vol(n.isel(t=0), xsl, y0, y1)
                    d["dNdt"] = (d["N"] - n0) / dt

        d["residual"] = (d["net_Fx"] + (0.0 if np.isnan(d["net_Fpol"])
                                        else d["net_Fpol"])
                         + total_src
                         - (0.0 if np.isnan(d["dNdt"]) else d["dNdt"]))
        out[name] = d
    return out


# ---------------------------------------------------------------------------
# Main-SOL bands and the turbulent transport coefficients at the LCFS
# ---------------------------------------------------------------------------

def _is_si(ds):
    """True if the dataset was loaded with ``unnormalise=True``.

    Same test as :func:`hermes3_turb.blobs._is_unnormalised` — xhermes stamps
    ``units_type='SI'`` on every variable it converts. Duplicated rather than
    imported to keep ``balance`` free of a dependency on ``blobs``.
    """
    return ds["g_22"].attrs.get("units_type") == "SI"


def sol_band_regions(ds):
    """
    Poloidal index ranges of the two main-SOL bands, ``{name: (y0, y1)}``.

    Half-open ranges of *dump* poloidal indices spanning the high- and
    low-field-side flux tubes between the X-points, i.e. the separatrix either
    side of the core — the divertor legs are excluded (use :func:`leg_regions`
    for those).

    Index conventions, both easy to get silently wrong:

    * **``jyseps`` is the LAST row of the region below it**, so every band
      starts at ``jyseps + 1``. ``jyseps1_1`` ends the inner-lower leg,
      ``jyseps2_1`` ends the inner core, ``jyseps1_2`` ends the outer-upper
      leg and ``jyseps2_2`` ends the outer core (authority:
      ``xhermes/selectors.py:187,195``).
    * **Guard offsets differ between the segments.** A dump retaining
      y-boundary guards shifts the inner segment by ``+MYG``. The outer
      segment shifts by ``+3*MYG`` in a *double* null, because 2*MYG guard
      rows sit at the upper targets between the two segments, but by only
      ``+MYG`` in a single null, which has no inter-segment guard band. ``g``
      is derived by comparing the dump's poloidal size against
      ``metadata["ny"]``, so this works whether or not guards were kept.

    (:func:`leg_regions` applies ``+3*MYG`` to its outer segment
    unconditionally. That is correct for the double nulls it was written and
    tested against, but overshoots by ``MYG`` for a single null whose dump
    kept y-guards — a latent issue, not exercised by any current case.)
    """
    m = ds.metadata
    pdim = detect_poloidal_dim(ds)
    ny = ds.sizes[pdim]
    myg = int(m.get("MYG", 2))
    ny_grid = int(m.get("ny", ny))
    g = myg if ny > ny_grid else 0

    j11, j21 = int(m["jyseps1_1"]), int(m["jyseps2_1"])
    j12, j22 = int(m["jyseps1_2"]), int(m["jyseps2_2"])
    g_out = 3 * g if j21 != j12 else g          # double null vs single null

    bands = {
        "inner SOL (HFS)": (j11 + 1 + g, j21 + 1 + g),
        "outer SOL (LFS)": (j12 + 1 + g_out, j22 + 1 + g_out),
    }
    return {k: v for k, v in bands.items() if v[1] > v[0]}


def lcfs_transport(ds, species="e", x_idx=None, grid_vars=None,
                   chi_ref=None, D_ref=None, include_y_derivs=True,
                   phi=None):
    """Compute LCFS channel budgets and effective turbulent transport by SOL band.

        chi_turb = chi_ref * q_ExB,turb / q_anom
        D_turb   = D_ref * Gamma_ExB,turb / Gamma_anom

    The reference flux is evaluated on the same profile and geometry. The heat
    reference includes both chi*grad(T) and (3/2)*T*D*grad(n), so comparisons
    between cases require the same D_ref/chi_ref convention. If both references
    are scaled together, that common scaling cancels in the inferred transport.
    With anomalous_diffusion enabled, the input coefficients also provide the
    prescribed transport contribution for comparison with turbulent transport.

    Parameters
    ----------
    ds : xarray.Dataset
        Loaded with ``unnormalise=False`` (BOUT++ normalisation).
    species : str
        Charged species whose density/pressure drive the fluxes, e.g. ``"e"``.
    x_idx : int, optional
        Radial index of the surface. Defaults to ``metadata["ixseps1"]``.
    grid_vars : dict, optional
        From :func:`hermes3_turb.load_grid_vars`. Needed for the magnetic-drift
        channel; omit it and that channel is left out of the table.
    chi_ref, D_ref : float, optional
        Reference diffusivities [m²/s]. Default: the deck's own
        ``anomalous_chi``/``anomalous_D`` for this species if the dump carries
        its options, else ``1.0``.
    include_y_derivs : bool
        Keep the ``−(g_23/g_22)·∂φ/∂θ`` term in the ExB velocity. True is the
        Hermes-3 default.
    phi : xarray.DataArray, optional
        Defaults to ``ds["phi"]``.

    Returns
    -------
    dict
        ``{band_name: {...}}``, each holding ``particles`` [s⁻¹] and ``heat``
        [W] dicts keyed by channel, plus ``chi_turb``/``D_turb`` [m²/s],
        the ``chi_ref``/``D_ref`` used, and the band's ``(y0, y1)``.
    """
    from .fluxes import (
        decompose_radial_particle_flux,
        decompose_radial_heat_flux,
        radial_magnetic_drift_particle_flux,
        radial_magnetic_drift_heat_flux,
        radial_anomalous_particle_flux,
        radial_anomalous_heat_flux,
    )

    if _is_si(ds):
        raise ValueError(
            "lcfs_transport needs BOUT++-normalised data (unnormalise=False): "
            "it applies the Nnorm·Cs0·rho_s0² conversions itself."
        )

    m = ds.metadata
    pdim = detect_poloidal_dim(ds)
    if x_idx is None:
        x_idx = int(m["ixseps1"])
    if phi is None:
        phi = ds["phi"]

    def _opt(sec, key):
        try:
            return float(ds.options[sec][key])
        except Exception:
            return 0.0

    if chi_ref is None:
        chi_ref = _opt(species, "anomalous_chi") or _opt(f"P{species}", "anomalous_chi") or 1.0
    if D_ref is None:
        D_ref = _opt(species, "anomalous_D") or _opt(f"N{species}", "anomalous_D") or 1.0

    N, P = ds[f"N{species}"], ds[f"P{species}"]
    ion = "i" if "Ni" in ds.data_vars else "d+"
    Ne, Ni, Pe, Pi = ds["Ne"], ds[f"N{ion}"], ds["Pe"], ds[f"P{ion}"]
    Te = ds["Te"] if "Te" in ds else Pe / Ne.clip(min=1e-10)
    Ti = ds[f"T{ion}"] if f"T{ion}" in ds else Pi / Ni.clip(min=1e-10)
    is_e = species == "e"

    tm = lambda d: d.mean("t") if "t" in getattr(d, "dims", ()) else d

    G_mean, G_turb, _ = decompose_radial_particle_flux(
        ds, N, phi, include_y_derivs=include_y_derivs)
    q_mean, q_turb, _ = decompose_radial_heat_flux(
        ds, P, phi, include_y_derivs=include_y_derivs)

    # Diffusivity normalisation: [m²/s] / (rho_s0²·Omega_ci) → BOUT++ units.
    dn = float(m["rho_s0"]) ** 2 * float(m["Omega_ci"])
    G_ref = tm(radial_anomalous_particle_flux(ds, N, D_ref / dn))
    q_ref = tm(radial_anomalous_heat_flux(
        ds, N, Te if is_e else Ti, chi_ref / dn, D=D_ref / dn))

    channels_G = {"ExB mean": G_mean, "ExB turbulent": G_turb}
    channels_q = {"ExB mean": q_mean, "ExB turbulent": q_turb}
    if grid_vars is not None:
        Ge, Gi = radial_magnetic_drift_particle_flux(
            ds, Ne, Ni, grid_vars["bxcvx"], grid_vars["Bxy"], Te, Ti)
        qe_, qi_, _ = radial_magnetic_drift_heat_flux(
            ds, Pe, Pi, grid_vars["bxcvx"], grid_vars["Bxy"],
            Te_eV=Te, Ti_eV=Ti)
        channels_G["magnetic drift"] = tm(Ge if is_e else Gi)
        channels_q["magnetic drift"] = tm(qe_ if is_e else qi_)
    channels_G["reference"] = G_ref
    channels_q["reference"] = q_ref

    QE = 1.602176634e-19
    f_p = float(m["Nnorm"]) * float(m["Cs0"]) * float(m["rho_s0"]) ** 2
    f_q = f_p * float(m["Tnorm"]) * QE
    elem = radial_area_element_dtheta(ds)

    def _integrate_band(fd, y0, y1):
        r = ((fd * elem).isel(x=x_idx)
             .isel({pdim: slice(y0, y1)}).sum(pdim))
        leftover = list(r.dims)
        if leftover:
            r = r.mean(dim=leftover)
        return float(r) * 2 * np.pi

    out = {}
    for name, (y0, y1) in sol_band_regions(ds).items():
        Gp = {k: _integrate_band(v, y0, y1) * f_p for k, v in channels_G.items()}
        qp = {k: _integrate_band(v, y0, y1) * f_q for k, v in channels_q.items()}
        out[name] = {
            "particles": Gp,
            "heat": qp,
            "x_idx": x_idx,
            "rows": (y0, y1),
            "chi_ref": chi_ref,
            "D_ref": D_ref,
            "D_turb": (D_ref * Gp["ExB turbulent"] / Gp["reference"]
                       if Gp["reference"] else float("nan")),
            "chi_turb": (chi_ref * qp["ExB turbulent"] / qp["reference"]
                         if qp["reference"] else float("nan")),
        }
    return out

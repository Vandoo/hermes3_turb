"""
Target heat flux analysis for Hermes-3 3D turbulence simulations.

Computes the heat flux density at divertor targets, integrates for total power,
converts to physical coordinates, and fits the Eich (2013) profile to extract
the heat flux decay length λ_q.

Physics
-------
At the sheath entrance (divertor target), the Bohm–Chodura–Riemann boundary
condition sets the ion parallel velocity (Dudson et al. 2026, eq. 9):

    v_{∥i} = sqrt(e(5/3 T_i + Z_i T_e) / m_i)   [m/s]

The energy flux parallel to B at the sheath entrance (eq. 11):

    q_{∥,s} = γ_s · e · n_s · T_s · v_{∥,s}      [W/m²]

with sheath transmission factors (eqs. 12a–b):

    γ_e = 2 + max(φ − φ_wall, 0) / T_e    (typically 3.5–9)
    γ_i = 2.5 + 0.5 m_i v_{∥i}² / T_i    (typically 3.5–4.5)

The heat flux density incident on the target plate surface normal is:

    q_target = q_{∥,tot} · (B_p / B)       [W/m²]

where B_p/B ≈ sin(α) is the field-line incidence angle.

If Hermes-3 sheath diagnostic variables (e.g. 'sheath_energy_flux_e') are
available they are used directly; otherwise the above formula is applied at
the last-domain-cell row adjacent to the target guard cells.

Eich profile (Nuclear Fusion 53, 093031, 2013):

    q(s) = (q₀/2) · exp[(S/2λ_q)² − s/λ_q] · erfc[S/2λ_q − s/S] + q_bg

where s is the distance from the separatrix strike point at the target [mm],
λ_q is the heat flux width and S is the poloidal spreading factor.
"""

import numpy as np
import xarray as xr
from scipy.special import erfc
from scipy.optimize import curve_fit

from .geometry import poloidal_area_element_dx
from .decompose import toroidal_mean
from .load import detect_toroidal_dim, detect_poloidal_dim, _no_t


# Physical constants
_e  = 1.602e-19   # elementary charge [C]
_mp = 1.673e-27   # proton mass [kg]
_me = 9.109e-31   # electron mass [kg]


# ---------------------------------------------------------------------------
# Target index helpers
# ---------------------------------------------------------------------------

def get_target_boundaries(ds):
    """
    Return, for each divertor target, the (domain_cell, guard_cell) poloidal
    index pair straddling the target plate, plus whether y-guard cells exist.

    The divertor target plate sits at the cell *boundary* between the last
    plasma (domain) cell and the first y-boundary guard cell. To evaluate a
    plasma quantity *at the sheath* it must be interpolated across that
    boundary, which is only possible when the dump was written with the
    y-boundary guard cells kept (``keep_yboundaries=True`` / a squash that
    retains them). Hermes-3 extrapolates the plasma profiles into these guard
    cells, so ``0.5·(domain + guard)`` recovers the value on the plate.

    Note the *geometry* (metric, Jacobian) is only zero-gradient-copied into
    the y-guard cells (BOUT++ ``extrapolate_y = false`` to avoid negative
    guard-cell Jacobians), so geometric quantities should be read at the
    ``domain_cell`` index, never interpolated.

    Works for single-null (2 targets) and double-null (4 targets) topologies.

    Returns
    -------
    boundaries : dict {target_name: (domain_idx, guard_idx)}
        Always present   : 'inner_lower', 'outer_lower'
        Double-null only : 'inner_upper', 'outer_upper'
        ``domain_idx`` is the last plasma cell adjacent to the plate;
        ``guard_idx`` is the first guard cell beyond it (== domain_idx when
        no guards are present).
    has_guards : bool
        True if the dump retains y-boundary guard cells.
    """
    m       = ds.metadata
    pdim    = detect_poloidal_dim(ds)
    ny_full = ds.sizes[pdim]
    MYG     = int(m.get("MYG", 2))

    # metadata 'ny' excludes guard cells; if the loaded poloidal size exceeds
    # it, the guard cells were kept in this dump.
    ny_nogc    = int(m.get("ny", ny_full))
    has_guards = ny_full > ny_nogc
    g          = MYG if has_guards else 0

    ny_inner_raw = m.get("ny_inner", None)
    is_dn = (ny_inner_raw is not None
             and 0 < int(ny_inner_raw) < (ny_nogc if ny_nogc else ny_full))

    def _clamp_guard(domain, guard):
        # When guards are absent the "guard" falls outside the array; collapse
        # it onto the domain cell so interpolation becomes the identity.
        if guard < 0 or guard >= ny_full:
            return (domain, domain)
        return (domain, guard)

    if is_dn:
        # Double-null: two poloidal regions joined at `split`.
        split = int(ny_inner_raw) + 2 * g          # first index of outer region
        boundaries = {
            "inner_lower": _clamp_guard(g,                 g - 1),
            "inner_upper": _clamp_guard(split - g - 1,     split - g),
            "outer_upper": _clamp_guard(split + g,         split + g - 1),
            "outer_lower": _clamp_guard(ny_full - g - 1,   ny_full - g),
        }
    else:
        boundaries = {
            "inner_lower": _clamp_guard(g,               g - 1),
            "outer_lower": _clamp_guard(ny_full - g - 1, ny_full - g),
        }

    return boundaries, has_guards


def get_target_y_indices(ds):
    """
    Return a dict of named target poloidal indices (the last plasma/domain
    cell adjacent to each target plate).

    Thin wrapper over :func:`get_target_boundaries`; see it for the topology
    handling and the domain-vs-guard-cell distinction.

    Returns
    -------
    dict : {target_name: domain_y_index}
        Always present  : 'inner_lower', 'outer_lower'
        Double-null only: 'inner_upper', 'outer_upper'
    """
    boundaries, _ = get_target_boundaries(ds)
    return {name: dom for name, (dom, _gc) in boundaries.items()}


def field_line_incidence(ds, y_idx, pdim=None):
    """
    Public wrapper for the field-line incidence factor Bp/B = sin(α) at the
    poloidal slice ``y_idx`` (see :func:`_field_line_incidence`). Useful for
    building poloidal heat-flux / conservation comparisons:
    q_pol = (5/2) P V_∥ · (Bp/B) since v_pol = v_∥·(Bp/B).
    """
    if pdim is None:
        pdim = detect_poloidal_dim(ds)
    return _field_line_incidence(ds, pdim, y_idx)


def _field_line_incidence(ds, pdim, y_idx):
    """
    Field-line incidence factor Bp/B = sin(α) at a poloidal slice.

    Prefers the poloidal field directly (``Bpxy``/``Bxy``) when the grid file
    has supplied it. Otherwise derives it from the metric, which a bare squash
    (no grid file merged in) always carries:

        Bp/B = sqrt(1 − g_23² / (g_22 · g_33))

    This follows from the BOUT++ tokamak metric identity
    ``g_23 = B_t · sqrt(g_22 · g_33) / B`` and ``B² = B_p² + B_t²``. The ratio
    is dimensionless and therefore invariant under a *uniform* rescaling of the
    metric — but xhermes versions differ in WHICH covariant components they
    unnormalise (some do only ``g_22``, others also ``g_33``/``g_23``). A
    partial/inconsistent unnormalisation corrupts the ratio (e.g. Bp/B → 1).
    To be robust we revert each component to the dump's raw normalisation by
    dividing out whatever ``conversion`` factor xhermes recorded when it set
    ``units_type='SI'`` — leaving all three on one consistent footing.
    """
    if "Bpxy" in ds and "Bxy" in ds:
        Bp = _no_t(ds["Bpxy"]).isel({pdim: y_idx})
        B  = _no_t(ds["Bxy"]).isel({pdim: y_idx})
        return np.abs(Bp) / B

    rho_s0 = float(ds.metadata.get("rho_s0", 0.0))

    def _raw(name):
        g = _no_t(ds[name])
        # Undo xhermes unnormalisation if applied. It records the factor it
        # multiplied by in attrs['conversion'] and sets units_type='SI'; if a
        # version omits the conversion, fall back to the covariant-metric
        # default (rho_s0², all of g_22/g_23/g_33 share it).
        if g.attrs.get("units_type") == "SI":
            conv = float(g.attrs.get("conversion") or 0.0) or (rho_s0 ** 2)
            if conv:
                g = g / conv
        return g.isel({pdim: y_idx})

    g_22, g_23, g_33 = (_raw(n) for n in ("g_22", "g_23", "g_33"))

    ratio = g_23 ** 2 / (g_22 * g_33)
    return np.sqrt((1.0 - ratio).clip(min=0.0))


# ---------------------------------------------------------------------------
# Heat flux density at a target
# ---------------------------------------------------------------------------

def target_heat_flux_density(
    ds,
    Pe, Pi,
    Ve, Vi,
    phi=None,
    target="outer_lower",
    phi_wall=0.0,
    AA_ion=2.0,
    Z_ion=1.0,
    zdim=None,
    bohm_cap_v=True,
    sheath_ion_polytropic=1.0,
):
    """
    Compute heat flux density at a named divertor target [W/m²].

    Uses the Dudson 2026 eqs. 11–12 form:

        q_{∥,e} = γ_e · P_e · |v_{∥e}|
        q_{∥,i} = γ_i · P_i · |v_{∥i}|

    where γ_e = 2 + max(φ−φ_wall,0)/T_e and γ_i = 2.5 + m_i v_{∥i}²/(2 e T_i).

    This is the **total** wall deposition (matches an IR thermography of the
    target). It also equals what ``sheath_boundary.cxx`` deposits: line 364
    adds an *additional* sink ((γ_s − 5/2)·T_s − ½ m_s C²) n v on top of the
    bulk advection scheme's natural (5/2) P V carry through the boundary face,
    summing to γ·P·V minus a negligible electron-mass term. Hermes-3's
    ``ef*_tot_ylow`` diagnostic only captures the advection+conduction part of
    this — at the sheath face it sits below γ·P·V by the sheath sink term and
    by the ion kinetic carry (which is in the momentum equation).

    Plasma profiles (Pe, Pi, Ve, Vi, phi, Te, Ti, Ne, Ni) are interpolated onto
    the target plate using the y-boundary guard cells (see get_target_boundaries).
    The field-line incidence factor Bp/B is taken from ``Bpxy``/``Bxy`` if present,
    otherwise derived from the metric — so no separate grid file is required.

    Parameters
    ----------
    ds   : xarray.Dataset  (needs g_22, g_23, g_33, Bxy; uses Bpxy if present)
    Pe   : xarray.DataArray  electron pressure [Pa]  (full 3D, with y-guards)
    Pi   : xarray.DataArray  ion pressure      [Pa]
    Ve   : xarray.DataArray  electron parallel velocity [m/s]
    Vi   : xarray.DataArray  ion parallel velocity     [m/s]
    phi  : xarray.DataArray, optional  electrostatic potential [V]
    target : str  one of the keys from get_target_y_indices()
    phi_wall : float  wall potential [V], default 0
    AA_ion   : float  ion atomic mass number (2 for deuterium)
    Z_ion    : float  ion charge number
    zdim     : str, optional  toroidal dimension name
    bohm_cap_v : bool, default True
        DEPRECATED. Originally intended to cap |V_∥| ≤ c_s at the sheath face,
        but that's the wrong direction — Hermes-3's ``sheath_boundary_simple``
        enforces |V_∥| ≥ c_s as a FLOOR (supersonic interior flow is preserved),
        and the full ``sheath_boundary`` sets V = ±c_s exactly. A cap would
        clip legitimate supersonic flow. The kwarg is kept for API compatibility
        but is a no-op; True only emits a DeprecationWarning. Use the M_∥
        diagnostic in target_poloidal_check.py to interpret apparent M > 1
        (either physical supersonic, or Favre-averaging artefact from
        ⟨NV⟩/⟨N⟩ ≠ ⟨V⟩).
    sheath_ion_polytropic : float, default 1.0
        Kept for API compatibility; no longer used (only entered the removed
        c_s² formula for the cap).

    Returns
    -------
    q_local : xarray.DataArray (x, zeta)  heat flux density at target [W/m²]
    q_avg   : xarray.DataArray (x,)       toroidal-mean heat flux [W/m²]
    y_idx   : int  poloidal index of the target slice used
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)
    pdim = detect_poloidal_dim(ds)

    boundaries, has_guards = get_target_boundaries(ds)
    i_dom, i_gc = boundaries[target]

    # Plasma profiles are evaluated *on the target plate*, i.e. at the cell
    # boundary between the last domain cell and the first guard cell, by linear
    # interpolation. This requires the guard cells (Hermes-3's extrapolated
    # values); without them this collapses to the last domain-cell value.
    def at_sheath(da):
        return 0.5 * (da.isel({pdim: i_dom}) + da.isel({pdim: i_gc}))

    Pe_t  = at_sheath(Pe)
    Pi_t  = at_sheath(Pi)
    Ve_t  = at_sheath(Ve)
    Vi_t  = at_sheath(Vi)

    # Field-line incidence: geometry is read at the domain cell (the metric is
    # only zero-gradient-copied into y-guards), and Bp/B is derived from the
    # metric when the poloidal field is not stored in the dataset.
    Bp_B  = _field_line_incidence(ds, pdim, i_dom)

    # Electron density from pressure + temperature
    # Te_eV = Pe/(n·e), but we can write q_{∥,e} = γ_e · Pe · |v_{∥e}|
    # (since n·e·T_e = Pe when T_e is in eV = Pe already in Pa then Pe/e in eV·m^-3...)
    # To avoid needing ne separately: n·e·T_e [Pa] = Pe [Pa] and n = Pe/(e·Te_eV)
    # So γ·n·e·T·v = γ · Pe · v  directly.

    # Te / Ti at the plate — used by both the φ-based γ form and the Stangeby
    # ρ-eq subtraction. Always derive once so both branches can see them.
    if "Te" in ds:
        Te_eV = at_sheath(ds["Te"])
    elif "Ne" in ds:
        Te_eV = at_sheath(Pe) / (_e * at_sheath(ds["Ne"]))
    else:
        Te_eV = at_sheath(Pe) / (_e * 1e19)
    if "Ti" in ds:
        Ti_eV = at_sheath(ds["Ti"])
    elif "Ni" in ds:
        Ti_eV = at_sheath(Pi) / (_e * at_sheath(ds["Ni"]))
    else:
        Ti_eV = at_sheath(Pi) / (_e * 1e19)
    m_i = AA_ion * _mp

    # NOTE: No Bohm cap on Vi_t / Ve_t. Hermes-3's `sheath_boundary_simple`
    # imposes |V_∥| ≥ c_s as a FLOOR (a min/max in cxx; supersonic interior
    # flow is preserved). The full `sheath_boundary` sets V = ±c_s exactly.
    # In both cases a CAP at c_s would be wrong-direction — it would clip
    # legitimate supersonic flow. Apparent M > 1 in time-averaged ⟨NV⟩/⟨N⟩
    # is either physical (sheath_boundary_simple with supersonic interior)
    # or a Favre-averaging artefact, distinguishable only with time-resolved
    # data. The M_∥ diagnostic flags it for the user to interpret.
    if bohm_cap_v:
        import warnings
        warnings.warn(
            "bohm_cap_v=True is ignored: |V_∥| ≤ c_s clips legitimate "
            "supersonic flow that Hermes-3's sheath_boundary_simple allows. "
            "The M_∥ diagnostic flags whether the apparent M > 1 is physical "
            "or a Favre-averaging artefact.",
            DeprecationWarning, stacklevel=2,
        )
    _ = sheath_ion_polytropic  # kept for API; not used

    # Sheath transmission factors
    if phi is not None:
        phi_t = at_sheath(phi)
        # γ_e = 2 + max(φ - φ_wall, 0) / T_e(eV)  (eq. 12a)
        excess_phi = (phi_t - phi_wall).clip(min=0.0)
        gamma_e = 2.0 + excess_phi / Te_eV.clip(min=0.01)
        # γ_i = 2.5 + 0.5·m_i·v_{∥i}²/(e·T_i_eV)  (eq. 12b)
        gamma_i = 2.5 + 0.5 * m_i * Vi_t ** 2 / (_e * Ti_eV.clip(min=0.01))
    else:
        # Fixed values matching Hermes-3 `sheath_boundary_simple` defaults
        # (sheath_boundary_simple.cxx: gamma_e/gamma_i = 3.5/3.5). Pass phi
        # explicitly to use the φ-based eq. 12a/b form from `sheath_boundary`.
        gamma_e = 3.5
        gamma_i = 3.5

    # Parallel heat flux at sheath: q_{∥,s} = γ_s · P_s · |v_{∥s}|
    # This IS the total wall deposition. sheath_boundary.cxx (line 364) adds an
    # "additional" q = ((γ − 5/2)·T − ½ m C²)·n·v on top of the natural
    # advection (5/2) P V the bulk advection scheme already carries, so the net
    # is (5/2) P V + (γ − 5/2) P V = γ P V (− tiny m_e n V³ for electrons).
    # ef*_tot_ylow shows only the advection + conduction part, so it sits a
    # bit below γ P V at the sheath face by the missing electron-sheath
    # contribution and the ion kinetic carry.
    q_par_e = gamma_e * Pe_t * np.abs(Ve_t)   # [W/m²]
    q_par_i = gamma_i * Pi_t * np.abs(Vi_t)   # [W/m²]

    # Project onto target surface normal: q_t = q_{∥,tot} · (Bp/B)
    q_local = (q_par_e + q_par_i) * Bp_B      # [W/m²]
    q_avg   = toroidal_mean(q_local, dim=zdim)

    return q_local, q_avg, i_dom


def target_sheath_power_from_diagnostics(ds, y_target_idx, ion_species="i",
                                         pdim=None, zdim=None):
    """
    Total power into ONE target taken from **Hermes-3's own sheath
    diagnostic** [W] — the number that validates the γ·P·V formula.

    Only ``sheath_boundary_simple`` writes this; the default
    ``sheath_boundary`` component has no ``outputVars`` at all, so with it this
    returns None no matter what ``diagnose`` is set to.

    The variable is ``E<species>_sheath`` — *not* any of the names
    :func:`target_heat_flux_from_diagnostics` guesses at — and it is an energy
    **source density [W/m³]** in the last domain cell, not a flux density, so
    it is volume-integrated here rather than multiplied by a face area. From
    ``sheath_boundary_simple.cxx``::

        q = gamma * T_sheath * n_sheath * v_sheath   # [W/m²], the TOTAL flux
        hflux[i] += q * da / dv                      # lower y  (q < 0)
        hflux[i] -= q * da / dv                      # upper y  (q > 0)
        set(diagnostics[species]["energy_source"], hflux)

    Both signs end up negative (it is a sink), and both are placed at the
    **last domain cell**, so ``abs`` of the sum is the power through that face
    for every target.

    **Units trap.** BOUT++ writes the right ``conversion`` attribute
    (``qe·Tnorm·Nnorm·Omega_ci``) but **not** ``units_type``, and xhermes's
    ``unnormalise()`` is driven by ``units_type`` — so this variable is left
    **normalised** even under ``unnormalise=True``. The factor is taken from
    the variable's own ``conversion`` attribute when present (BOUT++'s own
    number, exactly right), else rebuilt from metadata, and skipped if
    ``units_type`` says the array is already SI.

    Parameters
    ----------
    ds : xarray.Dataset
    y_target_idx : int
        Poloidal index of the target's last **domain** cell, i.e. the first
        element of ``get_target_boundaries(ds)[target]``.
    ion_species : str
        Ion species name; ``E<ion>_sheath`` needs ``[<ion>] diagnose = true``.

    Returns
    -------
    dict or None
        ``power_W``, ``vars`` (those found), ``missing``, ``factor``.
    """
    # Check for the variables BEFORE probing the dataset's dimensions: there is
    # nothing to compute without them, and dimension detection can itself throw.
    wanted = [f"E{sp}_sheath" for sp in ("e", ion_species)]
    found = [n for n in wanted if n in ds]
    if not found:
        return None

    pdim = pdim or detect_poloidal_dim(ds)
    zdim = zdim or detect_toroidal_dim(ds)
    m = ds.metadata
    qe = 1.602176634e-19

    vol = (_no_t(ds["J"]) * _no_t(ds["dx"])
           * _no_t(ds["dy"]) * _no_t(ds["dz"])).isel({pdim: y_target_idx})
    mxg = int(m.get("MXG", 2))
    zper = float(m.get("zperiod", 1) or 1)
    xsel = dict(x=slice(mxg, ds.sizes["x"] - mxg))

    default_factor = (qe * float(m["Tnorm"]) * float(m["Nnorm"])
                      * float(m["Omega_ci"]))
    total, factor_used = 0.0, None
    for n in found:
        da = ds[n]
        if str(da.attrs.get("units_type", "")).upper() == "SI":
            f = 1.0
        else:
            f = float(da.attrs.get("conversion", default_factor))
        factor_used = f
        if "t" in da.dims:
            da = da.mean("t")
        p = (da.isel({pdim: y_target_idx}) * vol).isel(xsel)
        dims = [d for d in p.dims if d in ("x", zdim)]
        total += float(p.sum(dim=dims)) * f

    return dict(power_W=abs(total) * zper,
                vars=found,
                missing=[n for n in wanted if n not in found],
                factor=factor_used)


def target_heat_flux_from_diagnostics(ds, target="outer_lower", zdim=None):
    """
    Return target heat flux using Hermes-3 sheath diagnostic variables
    (available when `diagnose = true` is set for both species in BOUT.inp).

    Looks for variables matching 'sheath_energy_flux_*' or similar patterns.
    Returns None if no sheath diagnostics are found.

    .. warning::
       **None of the names guessed at below is what Hermes-3 actually writes.**
       ``sheath_boundary_simple`` writes ``E<species>_sheath``, and that is an
       energy source density [W/m³] rather than a flux density [W/m²], so it
       does not fit this function's contract at all. Use
       :func:`target_sheath_power_from_diagnostics`, which integrates it
       correctly and handles the units trap. This function is kept for other
       Hermes-3 versions that may export a genuine sheath *flux*; in practice
       it returns None and the caller falls through to the γ·P·V formula.

    Parameters
    ----------
    ds     : xarray.Dataset
    target : str
    zdim   : str, optional

    Returns
    -------
    q_avg : xarray.DataArray (x,) or None
    """
    if zdim is None:
        zdim = detect_toroidal_dim(ds)
    pdim = detect_poloidal_dim(ds)

    # Common Hermes-3 sheath diagnostic naming patterns
    candidates = [
        ("sheath_energy_flux_e", "sheath_energy_flux_i"),
        ("Qe_sheath", "Qi_sheath"),
        ("Pe_sheath_flux", "Pi_sheath_flux"),
    ]

    boundaries, _ = get_target_boundaries(ds)
    i_dom, i_gc   = boundaries[target]

    def at_sheath(da):
        return 0.5 * (da.isel({pdim: i_dom}) + da.isel({pdim: i_gc}))

    for flux_e_name, flux_i_name in candidates:
        if flux_e_name in ds and flux_i_name in ds:
            Bp_B    = _field_line_incidence(ds, pdim, i_dom)
            flux_e  = at_sheath(ds[flux_e_name])
            flux_i  = at_sheath(ds[flux_i_name])
            q_avg   = toroidal_mean((flux_e + flux_i) * Bp_B, dim=zdim)
            return q_avg

    return None


# ---------------------------------------------------------------------------
# Physical coordinate: distance from separatrix at target
# ---------------------------------------------------------------------------

def target_physical_distance(ds, y_target_idx, ixseps1=None):
    """
    Physical distance s from the separatrix along the target plate [m].

    Integrates the cell width dr = dx / sqrt(g^{xx}) = dx / sqrt(g11) along x
    at the target poloidal index, with the separatrix (ixseps1) as origin:

        s > 0  →  SOL side
        s < 0  →  private flux / core side

    Parameters
    ----------
    ds           : xarray.Dataset
    y_target_idx : int  poloidal index of the target
    ixseps1      : int, optional  separatrix x-index; auto-read from metadata

    Returns
    -------
    s : numpy.ndarray [m]  shape (nx,)
    """
    pdim = detect_poloidal_dim(ds)

    if ixseps1 is None:
        _ix1 = int(ds.metadata.get("ixseps1", ds.sizes["x"] // 2))
        _ix2 = int(ds.metadata.get("ixseps2", _ix1))
        ixseps1 = min(_ix1, _ix2)

    dx  = _no_t(ds["dx"]).isel({pdim: y_target_idx}).values    # [Wb = T·m²]
    g11 = _no_t(ds["g11"]).isel({pdim: y_target_idx}).values   # [T²m²]

    # Physical cell width in x: dr = dx / sqrt(g^{xx}) = dx / sqrt(g11) [m]
    dr = dx / np.sqrt(g11)

    nx = len(dr)
    s  = np.zeros(nx)

    # Integrate outward (SOL: x > ixseps1)
    for i in range(ixseps1 + 1, nx):
        s[i] = s[i - 1] + 0.5 * (dr[i - 1] + dr[i])

    # Integrate inward (core/PFR: x < ixseps1), negative convention
    for i in range(ixseps1 - 1, -1, -1):
        s[i] = s[i + 1] - 0.5 * (dr[i] + dr[i + 1])

    return s


def get_midplane_y_indices(ds):
    """
    Return midplane poloidal indices, midway between the X-points on each side.

        {'omp': outer midplane}                         (always)
        {'imp': inner midplane}   added for double-null  (region A spans the HFS)

    Guarded indices, using the same region offsets as get_target_boundaries
    (region A: +MYG ; region B: +3·MYG). For inner (HFS) targets map with
    ``imp``; for outer (LFS) targets map with ``omp``. In single-null there is a
    single SOL/midplane, so only ``omp`` is returned (use it for both targets).

    Verified geometrically 2026-09-30 on the MAST-U SXD nx244/ny112, SXD
    nx132/ny36 and CD ny96 grids: the index midpoint coincides with the
    extremum of major radius on the separatrix to within one row, and `R·Bp`
    (hence the `dx/sqrt(g11)` radial scale) agrees to 1.0000 on all three.
    Note the grid file stores Rxy/Zxy WITH y-boundary guards when the grid was
    built with `y_boundary_guards>0`, so any check of this must apply the same
    +MYG / +3·MYG region offsets — comparing guard-inclusive arrays against
    guardless jyseps arithmetic puts the "midplane" near the upper X-point and
    invents a 2x error in the radial coordinate.
    """
    m       = ds.metadata
    pdim    = detect_poloidal_dim(ds)
    ny_full = ds.sizes[pdim]
    MYG     = int(m.get("MYG", 2))
    ny_nogc = int(m.get("ny", ny_full))
    g       = MYG if ny_full > ny_nogc else 0

    j1_1 = int(m["jyseps1_1"]) + g
    j2_1 = int(m["jyseps2_1"]) + g
    j1_2 = int(m["jyseps1_2"]) + 3 * g
    j2_2 = int(m["jyseps2_2"]) + 3 * g

    mids = {"omp": (j1_2 + j2_2) // 2}
    ny_inner = m.get("ny_inner", None)
    if ny_inner is not None and int(ny_inner) > 0:
        mids["imp"] = (j1_1 + j2_1) // 2
    return mids


def omp_midplane_distance(gridfile, psi=None, side="outer"):
    """
    Distance from the separatrix [m] at the TRUE outboard midplane, by ψ.

    Every flux surface ψ is mapped to the major radius where it crosses the
    horizontal line Z = Z_axis on the low-field side, using the geqdsk that
    hypnotoad embeds in the grid (`hypnotoad_input_geqdsk_file_contents`):

        s(ψ) = R_mid(ψ) − R_mid(psi_bdry)

    Why not the Rxy/Zxy arc at the "OMP row": with few poloidal cells there is
    no row on the midplane. SXD ny32 has 4 outer-core rows, so
    `(jyseps1_2+jyseps2_2)//2` is row 21 at Z=+0.475 m, R=1.28 m (midplane
    R≈1.37 m); flux surfaces there are ~1.25x further apart, which stretched
    every ny32 λ by ~25 % and made an ny scan compare different poloidal
    positions. Mapping by ψ gives every grid of an equilibrium the same
    coordinate, whatever its ny or y-guards.

    The geqdsk ψ is evaluated with a bicubic spline; at the grid's own points
    it reproduces `psixy` to ~1e-17 (SXD nx132/nx244, CD ny28), and this is
    checked on every call. Zero is `psi_bdry` (hypnotoad's separatrix), i.e.
    the face between ixseps1-1 and ixseps1, not the first SOL cell centre.

    Parameters
    ----------
    gridfile : str   hypnotoad grid file
    psi      : array-like, optional  raw ψ [Wb/rad, grid units]. Default: the
               grid's `psixy` column at its OMP row (one value per x, guards
               included), which is what the OMP profiles are taken on.
    side     : "outer" (LFS, default) or "inner" (HFS: the inner midplane,
               for mapping inner targets). s > 0 in the SOL on both sides.

    Returns
    -------
    s : numpy.ndarray [m], same shape as psi
    """
    import re
    import netCDF4 as nc
    from scipy.interpolate import RectBivariateSpline

    with nc.Dataset(gridfile) as g:
        if "hypnotoad_input_geqdsk_file_contents" not in g.variables:
            raise KeyError(f"{gridfile}: no embedded geqdsk — cannot locate the "
                           f"true midplane")
        txt = str(g["hypnotoad_input_geqdsk_file_contents"][...])
        Rxy = np.asarray(g["Rxy"][:]); Zxy = np.asarray(g["Zxy"][:])
        psixy = np.asarray(g["psixy"][:])
        psi_bdry = float(g["psi_bdry"][...])
        if psi is None:
            jy = (int(g["jyseps1_2"][...]) + int(g["jyseps2_2"][...])) // 2
            if Rxy.shape[1] != int(g["ny"][...]):
                jy += 3 * int(g["y_boundary_guards"][...])
            psi = psixy[:, jy]

    # geqdsk: header ends in nw nh; then 20 scalars, fpol/pres/ffprim/pprime
    # (nw each), psirz (nw*nh, R fastest). Fixed-width fields can touch
    # ("0.0E+00-8.1E-02"), so split on the number pattern, not whitespace.
    lines = txt.splitlines()
    nw, nh = map(int, lines[0].split()[-2:])
    num = np.array([float(x) for x in re.findall(
        r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?", "\n".join(lines[1:]))])
    rdim, zdim, _, rleft, zmid, rmaxis, zmaxis = num[:7]
    o = 20 + 4 * nw
    psirz = num[o:o + nw * nh].reshape(nh, nw)
    R1 = rleft + rdim * np.arange(nw) / (nw - 1)
    Z1 = zmid - zdim / 2 + zdim * np.arange(nh) / (nh - 1)
    spl = RectBivariateSpline(Z1, R1, psirz)

    err = np.abs(spl.ev(Zxy, Rxy) - psixy).max()
    if err > 1e-6 * np.ptp(psixy):
        raise ValueError(f"{gridfile}: embedded geqdsk does not reproduce psixy "
                         f"(max diff {err:.2e}) — not this grid's equilibrium")

    # ψ along the outboard midplane, kept up to where it stops being monotonic
    # (a far-SOL X-point or the box edge).
    # Start at the ψ extremum (the spline's axis sits a hair off rmaxis).
    Rm = np.linspace(rmaxis, R1[-1] if side == "outer" else R1[0], 20001)
    pm = spl.ev(np.full_like(Rm, zmaxis), Rm)
    sgn = np.sign(psi_bdry - pm[0])
    pm = pm * sgn
    i0 = np.argmin(pm)
    Rm, pm = Rm[i0:], pm[i0:]
    mono = np.diff(pm) > 0
    n = np.argmin(mono) + 1 if not mono.all() else Rm.size
    Rm, pm = Rm[:n], pm[:n]
    psi = np.asarray(psi, dtype=float)
    if (psi * sgn).max() > pm[-1] or psi_bdry * sgn > pm[-1]:
        raise ValueError(f"{gridfile}: ψ beyond the monotonic part of the "
                         f"{side} midplane (R to {Rm[-1]:.3f} m)")
    R_of = lambda p: np.interp(p * sgn, pm, Rm)
    d = R_of(psi) - R_of(psi_bdry)
    return d if side == "outer" else -d


def omp_mapped_distance(gridfile, target):
    """
    Midplane-mapped distance from the separatrix [m] for each x of a target.

    The Eich convention: every flux surface reaching the plate is labelled by
    its distance from the separatrix at the upstream midplane — the outboard
    midplane for outer targets, the inboard one for the inner targets of a
    double null (a single-null inner target also maps to the outboard side,
    since there is only one SOL). ψ is taken from the target's OWN leg row of
    the grid, so private-flux cells get their own ψ (s < 0), and mapped with
    `omp_midplane_distance`.

    Replaces (2026-10-15) the old `(ds, y_target_idx, y_omp_idx)` version,
    `s_target · B_target/B_mid`: a TOTAL-field ratio, which is ≈ R_mid/R_target
    for B ≈ B_t, not the poloidal flux expansion (R·Bp ratio) it was meant to
    be; and it took B_mid on the "midplane row", which on coarse ny is well off
    the midplane (SXD ny32: Z=+0.475 m). Nothing called it.

    Parameters
    ----------
    gridfile : str  hypnotoad grid file of the run
    target   : 'inner_lower' | 'outer_lower' | 'inner_upper' | 'outer_upper'

    Returns
    -------
    s : numpy.ndarray [m], one value per x (x-guards included)
    """
    import netCDF4 as nc

    with nc.Dataset(gridfile) as g:
        psixy = np.asarray(g["psixy"][:])
        ny = int(g["ny"][...])
        nyi = int(g["ny_inner"][...]) if "ny_inner" in g.variables else 0
        ybg = int(g["y_boundary_guards"][...]) if psixy.shape[1] != ny else 0
    # Leg rows in the grid's own row space, the layout get_target_boundaries
    # uses: guards, if stored, sit at the ends (single null) or at the ends
    # and the upper targets (double null, region B offset 3·ybg).
    dn = 0 < nyi < ny
    offB = 3 * ybg if dn else ybg
    rows = {"inner_lower": ybg, "outer_lower": ny - 1 + offB,
            "inner_upper": nyi - 1 + ybg, "outer_upper": nyi + offB}
    if target not in rows:
        raise KeyError(f"unknown target {target!r}; expected one of {list(rows)}")
    if not dn and target.endswith("upper"):
        raise KeyError(f"{gridfile} is single null: no {target}")
    side = "inner" if target.startswith("inner") and dn else "outer"
    return omp_midplane_distance(gridfile, psixy[:, rows[target]], side=side)


# ---------------------------------------------------------------------------
# Eich profile and fitting
# ---------------------------------------------------------------------------

def eich_profile(s, q0, lambda_q, S, q_bg=0.0, s0=0.0):
    """
    Eich 2013 divertor heat flux profile (Nucl. Fusion 53, 093031).

        q(s) = (q₀/2) · exp[(S/2λ_q)² − (s−s₀)/λ_q]
                       · erfc[S/2λ_q − (s−s₀)/S] + q_bg

    Parameters
    ----------
    s        : array-like  distance from separatrix [mm]
    q0       : float  amplitude [MW/m²]
    lambda_q : float  heat flux decay length [mm]
    S        : float  spreading factor [mm]
    q_bg     : float  background heat flux [MW/m²]
    s0       : float  strike-point offset [mm]; the magnetic-separatrix /
               heat-flux strike-point shift used in TCV/JET Eich extensions
               for cases where the real q-peak sits well inside the SOL and
               cannot be absorbed by S alone (Eich's built-in spreading puts
               the peak at ~S²/2λ_q; anything beyond needs s0).

    Returns
    -------
    q : ndarray [MW/m²]

    Notes
    -----
    λ_q is the decay length in the SOL just outside the separatrix.
    S accounts for the diffusive broadening from X-point to target.
    The effective width at the target is approximately λ_q + 1.64·S.
    Recommend working in mm and MW/m² for numerical stability of curve_fit.
    """
    s   = np.asarray(s, dtype=float) - float(s0)
    lq  = np.abs(lambda_q)
    S   = np.abs(S)
    arg = np.clip((S / (2 * lq)) ** 2 - s / lq, -700, 700)
    return (q0 / 2.0) * np.exp(arg) * erfc(S / (2 * lq) - s / S) + q_bg


def integral_width(s_mm, q, q_bg=0.0, s_range=None):
    """Integral power decay length λ_int [mm], straight from the profile.

    This is the **primary** definition (Loarte et al. 1999; Eich 2013 §2):

        λ_int = ∫ (q(s) − q_BG) ds / q_max

    i.e. the deposited power per unit length divided by the peak heat flux —
    the width of the rectangle carrying the same power as the real profile.

    Unlike :func:`fit_eich`'s ``lambda_int`` (which is Makowski's
    ``λ_q + 1.64·S`` approximation in terms of the fitted parameters), this
    needs **no fit at all**. That matters whenever the Eich form does not
    describe the profile: it is reported to fit JET, C-Mod, COMPASS and AUG
    well but EAST and especially **MAST** poorly (Horacek 2020), and it also
    misses the spreading into the PFR (Thornton 2015). A far-SOL shoulder — a
    second, flatter decay — likewise breaks the single-exponential premise
    while leaving λ_int perfectly well defined.

    Comparing the two is therefore a fit-quality check in itself: they should
    agree to a few per cent when the Eich fit is good.

    Parameters
    ----------
    s_mm : array
        Radial coordinate [mm], the same one the fit uses (OMP-mapped or at
        the target — λ_int inherits whichever it is given).
    q : array
        Heat flux profile [MW/m²] (any consistent unit; λ_int is a ratio).
    q_bg : float or array
        Background to subtract, default 0. **Only subtract a background over
        the window it was fitted on.** ``fit_eich``'s ``q_bg`` is a local
        nuisance parameter of the fit window; extrapolating that constant
        across a full radial domain (which may include 100+ mm of PFR) removes
        a large spurious area — on one MAST-U target that turned λ_int from
        11.3 mm into 3.1 mm. Simulation data has no instrumental background at
        all, so 0 is usually right for the full profile.
    s_range : (lo, hi), optional
        Restrict the integration to this window in ``s_mm``. Use it together
        with a fitted ``q_bg`` to get a number directly comparable with
        ``λ_q + 1.64·S``; leave it None (whole profile, ``q_bg=0``) for the
        width carrying the total deposited power.

    Returns
    -------
    dict with ``lambda_int`` [mm], ``q_max``, ``power_integral`` and
    ``n_points``.

    Notes
    -----
    With ``q_bg=0`` integrate over the **whole** measured profile, PFR side
    included — power deposited at s < 0 is still deposited power, and Thornton
    2015 shows the Eich form misses that spreading.
    """
    s = np.asarray(s_mm, dtype=float)
    qq = np.asarray(q, dtype=float) - np.asarray(q_bg, dtype=float)

    good = np.isfinite(s) & np.isfinite(qq)
    if s_range is not None:
        good &= (s >= s_range[0]) & (s <= s_range[1])
    s, qq = s[good], qq[good]
    if s.size < 3:
        return dict(lambda_int=np.nan, q_max=np.nan, power_integral=np.nan,
                    n_points=int(s.size))

    order = np.argsort(s)
    s, qq = s[order], qq[order]

    integral = float(np.trapz(qq, s))
    q_max = float(np.nanmax(qq))
    return dict(lambda_int=(integral / q_max if q_max > 0 else np.nan),
                q_max=q_max, power_integral=integral, n_points=int(s.size))


def fit_eich(s_mm, q_MWm2, p0=None, bounds=None, sol_only=True, fit_s0=False):
    """
    Fit the Eich profile to target heat flux data.

    Works in mm and MW/m² to keep parameters order-unity for numerical stability.

    Parameters
    ----------
    s_mm    : array-like  distance from separatrix [mm], positive = SOL
    q_MWm2  : array-like  heat flux data [MW/m²]
    p0      : list, optional  initial guess [q0, lambda_q, S, q_bg(, s0)]
              in [MW/m², mm, mm, MW/m², mm]. Auto-estimated if None.
    bounds  : 2-tuple, optional  parameter bounds for curve_fit
    sol_only : bool  if True, include only s > −3·S_guess to suppress PFR noise
    fit_s0   : bool or float
        False (default) → s0 fixed at 0 (the magnetic separatrix is the strike
        point).
        True            → s0 is added as a free 5th parameter; use when the
        observed q-peak sits well inside the SOL (more than ~S²/2λ_q mm) and
        cannot be absorbed by Eich's built-in spreading.
        numeric value  → s0 fixed at that value.

    Returns
    -------
    params  : dict  {'q0' [MW/m²], 'lambda_q' [mm], 'S' [mm], 'q_bg' [MW/m²],
                     's0' [mm], 'lambda_int' [mm]}  where λ_int = λ_q + 1.64·S
    perr    : dict  1-sigma uncertainties on each fitted parameter
    s_fine  : ndarray  fine s array for plotting the fit [mm]
    q_fit   : ndarray  fitted profile on s_fine [MW/m²]
    """
    s_arr = np.asarray(s_mm,   dtype=float)
    q_arr = np.asarray(q_MWm2, dtype=float)

    free_s0 = (fit_s0 is True)
    fixed_s0 = 0.0 if fit_s0 in (False, True) else float(fit_s0)

    # Auto initial guess (using the location of q_peak in case it's shifted)
    if p0 is None:
        q_peak  = float(np.nanmax(q_arr))
        i_peak  = int(np.nanargmax(q_arr))
        s_peak  = float(s_arr[i_peak]) if np.isfinite(s_arr[i_peak]) else 0.0
        i_half  = int(np.argmin(np.abs(q_arr[i_peak:] - 0.5 * q_peak)))
        lq_est  = max((s_arr[i_peak + i_half] - s_peak) if i_half > 0 else 1.0, 0.5)
        S_est   = lq_est / 2.0
        q_bg_est = float(np.nanmin(q_arr))
        p0 = [q_peak, lq_est, S_est, q_bg_est]
        if free_s0:
            p0.append(s_peak)

    if bounds is None:
        q_max = float(np.nanmax(q_arr))
        lo = [0,         0.01, 0.001, -0.2 * q_max]
        hi = [10*q_max,  200., 200.,   0.5 * q_max]
        if free_s0:
            s_lo, s_hi = float(s_arr.min()), float(s_arr.max())
            lo.append(s_lo);  hi.append(s_hi)
        bounds = (lo, hi)

    mask = np.isfinite(q_arr) & np.isfinite(s_arr)
    if sol_only:
        mask &= s_arr > fixed_s0 - 3.0 * p0[2]

    if free_s0:
        def _model(s, q0, lq, S, q_bg, s0):
            return eich_profile(s, q0, lq, S, q_bg, s0)
        param_names = ["q0", "lambda_q", "S", "q_bg", "s0"]
    else:
        def _model(s, q0, lq, S, q_bg):
            return eich_profile(s, q0, lq, S, q_bg, fixed_s0)
        param_names = ["q0", "lambda_q", "S", "q_bg"]

    popt, pcov = curve_fit(_model, s_arr[mask], q_arr[mask],
                           p0=p0, bounds=bounds, maxfev=20000)
    perr_arr = np.sqrt(np.diag(pcov))

    params = dict(zip(param_names, popt))
    perr   = dict(zip(param_names, perr_arr))
    if not free_s0:
        params["s0"] = fixed_s0

    # Integral power decay length. The PRIMARY definition (Loarte 1999) is
    # λ_int = ∫(q − q_BG) dr / q_max, i.e. deposited power divided by peak heat
    # flux; `integral_width()` evaluates that directly from the data. What is
    # computed here is Makowski's closed-form approximation to it in terms of
    # the Eich fit parameters,
    #
    #     λ_int ≈ λ_q + 1.64·S            (Eich 2013 NF 53 093031, §2)
    #
    # good to <4 % for S/λ_q < 10. NOTE the coefficient multiplies S, not S/2
    # — this previously carried a spurious factor 1/2, which under-reported
    # λ_int by ~36 % at a typical S/λ_q ≈ 0.7.
    params["lambda_int"] = params["lambda_q"] + 1.64 * params["S"]
    params["lambda_int_valid"] = bool(params["S"] / max(params["lambda_q"], 1e-30) < 10)

    s_fine = np.linspace(s_arr.min(), s_arr.max(), 500)
    q_fit  = eich_profile(s_fine, params["q0"], params["lambda_q"],
                          params["S"], params["q_bg"], params["s0"])

    return params, perr, s_fine, q_fit


# ---------------------------------------------------------------------------
# Double-exponential (near + far SOL) profile and fitting
# ---------------------------------------------------------------------------

def double_exponential_profile(s, q_near, lambda_near, q_far, lambda_far,
                               q_bg=0.0, s0=0.0):
    """
    Two-exponential SOL heat-flux profile (near-SOL + far-SOL), in mm / MW/m².

        q(s) = q_bg + q_near · exp(−(s−s₀)/λ_near) + q_far · exp(−(s−s₀)/λ_far)

    Valid on the SOL side (s ≥ s₀); the PFR side (s < s₀) is clipped to the
    peak value q_near + q_far + q_bg so curve_fit doesn't see a divergent
    exponential. A common alternative to the Eich profile for targets that
    show a steep near-SOL peak plus a broader far-SOL "shoulder" (two distinct
    decay lengths λ_near < λ_far), where a single Eich λ_q does not capture
    both. ``q_near``/``q_far`` are the amplitudes of each component at the
    strike point (s = s₀).

    Parameters
    ----------
    s          : array-like  distance from the separatrix [mm]
    q_near     : float  near-SOL amplitude at s = s₀ [MW/m²]
    lambda_near: float  near-SOL decay length [mm]
    q_far      : float  far-SOL amplitude at s = s₀ [MW/m²]
    lambda_far : float  far-SOL decay length [mm]
    q_bg       : float  background heat flux [MW/m²]
    s0         : float  strike-point offset [mm]; required when the q-peak
                 sits well inside the SOL — without it the fit tries to
                 match a rising-then-falling shape with a monotonically
                 decaying model and fails.

    Returns
    -------
    q : ndarray [MW/m²]
    """
    s = np.asarray(s, dtype=float)
    ds = np.maximum(s - float(s0), 0.0)        # PFR side clipped to the peak
    return (q_bg
            + q_near * np.exp(-ds / np.abs(lambda_near))
            + q_far * np.exp(-ds / np.abs(lambda_far)))


def fit_double_exp(s_mm, q_MWm2, p0=None, bounds=None, s_min=0.0, fit_s0=False):
    """
    Fit the double-exponential profile to the SOL side (s ≥ s_min) of the
    target heat flux. Works in mm / MW/m² for numerical stability.

    Parameters
    ----------
    s_mm   : array-like  distance from separatrix [mm], positive = SOL
    q_MWm2 : array-like  heat flux [MW/m²]
    p0     : list, optional  initial guess
             [q_near, λ_near, q_far, λ_far, q_bg(, s0)]
    bounds : 2-tuple, optional  curve_fit bounds
    s_min  : float  only fit s ≥ s_min (default 0). Note: when fit_s0=True
             the strike point can sit anywhere ≥ s_min, so set s_min to be
             generous (e.g. −5 mm) if you expect a shifted peak.
    fit_s0 : bool or float
        False (default) → s0 fixed at 0 (peak at the magnetic separatrix).
        True            → s0 is added as a free 6th parameter. **Required** if
        the q-peak sits well inside the SOL — without it the fit tries to
        match a rising-then-falling shape with a monotonically decaying model
        and fails (gives unphysically small λ_near).
        numeric value  → s0 fixed at that value.

    Returns
    -------
    params : dict  {'q_near','lambda_near','q_far','lambda_far','q_bg','s0',
                    'lambda_eff','power_frac_far'}  ordered so λ_near < λ_far.
             ``lambda_eff`` is the power-weighted decay length
             (q_near·λ_near + q_far·λ_far)/(q_near+q_far); ``power_frac_far``
             is the fraction of integrated SOL power in the far component.
    perr   : dict  1-σ uncertainties on each fitted parameter
    s_fine : ndarray  fine s array for plotting [mm]
    q_fit  : ndarray  fitted profile on s_fine [MW/m²]
    """
    s_arr = np.asarray(s_mm, dtype=float)
    q_arr = np.asarray(q_MWm2, dtype=float)

    free_s0  = (fit_s0 is True)
    fixed_s0 = 0.0 if fit_s0 in (False, True) else float(fit_s0)

    mask = np.isfinite(s_arr) & np.isfinite(q_arr) & (s_arr >= s_min)
    s = s_arr[mask]
    q = q_arr[mask]

    q_peak = float(np.nanmax(q))
    q_min  = float(np.nanmin(q))
    i_peak = int(np.nanargmax(q))
    s_peak = float(s[i_peak])

    if p0 is None:
        # Seed the near/far scales from the inner/outer log-slopes of the fitted
        # side rather than a fixed 1.5/8 mm — the fixed seed lands in a bad local
        # minimum on genuinely two-scale profiles (the failure seen fitting
        # fall-off lengths), collapsing the near scale onto its bound.
        def _seed_lam(mask):
            if mask.sum() >= 3 and np.all(q[mask] > 0):
                sl = np.polyfit(s[mask], np.log(q[mask]), 1)[0]
                if sl < 0:
                    return -1.0 / sl
            return 5.0
        smid = float(np.median(s))
        ln0 = max(min(_seed_lam(s <= smid), _seed_lam(s >= smid)), 0.1)
        lf0 = max(_seed_lam(s >= smid), 1.5 * ln0)
        p0 = [0.7 * q_peak, ln0, 0.3 * q_peak, lf0, max(q_min, 0.0)]
        if free_s0:
            p0.append(s_peak)
    if bounds is None:
        lo = [0.0,         0.05, 0.0,         0.05, -0.2 * q_peak]
        hi = [10 * q_peak, 200., 10 * q_peak, 200.,  0.5 * q_peak]
        if free_s0:
            lo.append(float(s.min()))
            hi.append(float(s.max()))
        bounds = (lo, hi)

    # Clamp p0 strictly inside the bounds so curve_fit never gets an infeasible
    # start (e.g. the q_bg seed above its upper bound on a flat-ish profile —
    # the `ValueError: x0 is infeasible` that blocked the double-exp fits).
    _lo = np.asarray(bounds[0], float); _hi = np.asarray(bounds[1], float)
    _eps = 1e-9 * (_hi - _lo + 1e-30)
    p0 = np.clip(np.asarray(p0, float), _lo + _eps, _hi - _eps).tolist()

    if free_s0:
        def _model(s, qn, ln, qf, lf, qb, s0):
            return double_exponential_profile(s, qn, ln, qf, lf, qb, s0)
        names = ["q_near", "lambda_near", "q_far", "lambda_far", "q_bg", "s0"]
    else:
        def _model(s, qn, ln, qf, lf, qb):
            return double_exponential_profile(s, qn, ln, qf, lf, qb, fixed_s0)
        names = ["q_near", "lambda_near", "q_far", "lambda_far", "q_bg"]

    popt, pcov = curve_fit(_model, s, q, p0=p0, bounds=bounds, maxfev=20000)
    perr_arr = np.sqrt(np.diag(pcov))

    params = dict(zip(names, popt))
    perr   = dict(zip(names, perr_arr))
    if not free_s0:
        params["s0"] = fixed_s0

    # Ensure 'near' is the steeper component (smaller λ); swap amplitude+λ+errors.
    if abs(params["lambda_near"]) > abs(params["lambda_far"]):
        for a, b in (("q_near", "q_far"), ("lambda_near", "lambda_far")):
            params[a], params[b] = params[b], params[a]
            if a in perr and b in perr:
                perr[a], perr[b] = perr[b], perr[a]
    params["lambda_near"] = abs(params["lambda_near"])
    params["lambda_far"]  = abs(params["lambda_far"])

    int_near = params["q_near"] * params["lambda_near"]
    int_far  = params["q_far"]  * params["lambda_far"]
    amp = params["q_near"] + params["q_far"]
    tot = int_near + int_far
    params["lambda_eff"]    = (int_near + int_far) / amp if amp > 0 else float("nan")
    params["power_frac_far"] = int_far / tot if tot > 0 else float("nan")

    s_fine = np.linspace(max(s_min, float(s.min())), float(s.max()), 500)
    q_fit = double_exponential_profile(
        s_fine, params["q_near"], params["lambda_near"],
        params["q_far"], params["lambda_far"], params["q_bg"], params["s0"],
    )
    return params, perr, s_fine, q_fit


# ---------------------------------------------------------------------------
# Total target power
# ---------------------------------------------------------------------------

def total_target_power(ds, q_target_avg, y_target_idx, zperiod=None,
                       exclude_xguards=True):
    """
    Integrate toroidal-mean heat flux over the target plate: total power [W].

        P = 2π · ∫ q_target · sqrt(g_{xx}·g_{ζζ} − g_{xζ}²) dx

    where 2π covers the full torus and sqrt(g_{xx}·g_{ζζ} − g_{xζ}²)·dx is
    the area element per unit toroidal angle at the target [m²/rad].

    **The x-guard cells are excluded by default, and this matters a lot.**
    They lie outside the physical radial domain, so they carry no wetted plate
    area — but on a mesh that expands outward they dominate the area element:
    measured on `sxd_MU/nx32_ny32_45456`, the two high-x guard cells alone hold
    **21.8 %** of the outer targets' total `poloidal_area_element_dx` (and 12.3 %
    on the inner targets). Including them inflated the total power by ~19 % on
    the outer targets, which is exactly what the comparison against Hermes-3's
    own `E<sp>_sheath` diagnostic exposed — the ratio went from 1.18–1.20 to
    ~1.0 once they were dropped. `target_poloidal_check.py` already excluded
    them; this function was the inconsistent one.

    Parameters
    ----------
    ds             : xarray.Dataset
    q_target_avg   : xarray.DataArray (x,)  toroidal-mean heat flux [W/m²]
                     Repeated toroidal geometry is averaged, not summed;
                     any retained time axis is averaged after integration.
    y_target_idx   : int  poloidal target index
    zperiod        : int, optional  from BOUT.inp 'zperiod'. If the simulation
                     covers 2π/zperiod of the torus, q_avg is still representative
                     of the full torus (toroidal average). Multiply by 2π for full
                     toroidal integral regardless of zperiod.
    exclude_xguards : bool
                     Drop `MXG` cells at each radial end (default True). Set
                     False only to reproduce the pre-2026-10-05 behaviour.

    Returns
    -------
    P_total : float [W]
    """
    pdim = detect_poloidal_dim(ds)

    # Area element: dx * sqrt(g_11 * g_33 - g_13^2) [m²/rad]
    elem = poloidal_area_element_dx(ds).isel({pdim: y_target_idx})

    # Integrate over x, multiply by 2π for full torus [W/m²] * [m²/rad] * [rad] = [W]
    integrand = q_target_avg * elem     # [W/m²] * [m²] = [W] per unit ζ
    if exclude_xguards:
        mxg = int(ds.metadata.get("MXG", 2))
        if mxg > 0:
            integrand = integrand.isel(x=slice(mxg, ds.sizes["x"] - mxg))
    result = integrand.sum(dim="x")
    if result.dims:
        result = result.mean(dim=list(result.dims))
    return float(result) * 2 * np.pi

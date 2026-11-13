"""
Would a SOLPS-style flux limiter act on this run?  A posteriori estimates.

Hermes-3 has three plasma flux limiters, all off by default, all of the same
harmonic form  ``coef -> coef / (1 + |q_classical / q_limit|)``:

===========================  ==========================  ==========================
option (per species)         classical flux              limit
===========================  ==========================  ==========================
``kappa_limit_alpha``        ``kappa_par * Grad_par(T)`` ``alpha * N T sqrt(T/AA)``
(braginskii_conduction)
``eta_limit_alpha``          ``eta * Grad_par(V)``       ``alpha * P``
(braginskii_ion_viscosity,   ion:  eta = 1.28 P tau
braginskii_electron_visc.)   elec: eta = (4/3) 0.73 P tau
===========================  ==========================  ==========================

so the quantity that decides everything is the dimensionless ratio
``R1 = |q_classical| / q_limit(alpha=1)``; for a given alpha the coefficient is
multiplied by ``1 / (1 + R1/alpha)``.  Everything here works in Hermes-3
NORMALISED units straight from the dump (the ratio is dimensionless, and using
the dump's own normalised metric reproduces the code's ``Grad_par`` exactly).

``tau`` is not written to the dump, but ``kappa_par_<sp>`` is, and
``kappa_par = kappa_coefficient * P * tau / AA`` (braginskii_conduction.cxx), so
``P tau = kappa_par * AA / kappa_coefficient``.  This assumes the viscosity
uses the same collision set as conduction.  For IONS that holds in the default
``multispecies`` mode of both (all Coulomb collisions + CX, with or without
neutrals); it breaks if either ``conduction_collisions_mode`` or
``viscosity_collisions_mode`` is set to ``braginskii``.  ELECTRON viscosity uses
the species' total ``collision_frequency``, which can include more channels
than conduction's e-all Coulomb set, so the electron eta here is an upper bound.

Numpy only (no xarray/xbout) so it runs with a bare netCDF4 python on the
cluster.  Arrays are (x, y, z) with the dump's y-guard rows still present.
"""

import numpy as np

#: Default Braginskii coefficients exactly as in the Hermes-3 source, keyed by
#: species KIND ("e" = electrons, "i" = any ion species such as "i" or "d+").
KAPPA_COEFF = {"e": 3.16 / np.sqrt(2.0), "i": 3.9}   # braginskii_conduction.cxx
ETA_COEFF = {"e": (4.0 / 3.0) * 0.73, "i": 1.28}      # braginskii_*_viscosity.cxx


def species_kind(name):
    """"e" for the electron species, "i" for anything else (charged ions).

    Hermes-3 names electrons "e"; neutrals are not handled by
    braginskii_conduction/viscosity, so they should not be passed here.
    """
    return "e" if name == "e" else "i"


def shift_z(arr, zshift, dz, direction=+1):
    """Shift (x, y, z) data toroidally by ``zShift`` (FFT in z).

    ``direction=+1`` = to field-aligned.  Same convention as
    :func:`hermes3_turb.load.to_field_aligned` (the xarray version).
    """
    nz = arr.shape[-1]
    kz = 2.0 * np.pi * np.arange(nz // 2 + 1) / (nz * dz)
    phase = np.exp(direction * 1.0j * kz * zshift[..., None])
    return np.fft.irfft(np.fft.rfft(arr, axis=-1) * phase, n=nz, axis=-1)


def dump_row_to_y(npol, ny, ny_inner, myg=None, double_null=True):
    """Global interior y index for every dump row, or -1 for a y-guard row.

    The number of guard rows per target is inferred from the DUMP
    (``npol - ny``), never assumed from ``MYG``: a squash without
    ``keep_yboundaries`` has ``npol == ny`` (no guards), and a grid built with
    ``y_boundary_guards != MYG`` has a different count.  A double null has
    guards at four targets, a single null at two.  ``myg`` is ignored (kept for
    call compatibility).
    """
    nt = 4 if double_null else 2
    extra = npol - ny
    if extra < 0 or extra % nt:
        raise ValueError(f"cannot map {npol} dump rows onto ny={ny} "
                         f"({'double' if double_null else 'single'} null)")
    myg = extra // nt
    ymap = -np.ones(npol, dtype=int)
    if double_null:
        ymap[myg:myg + ny_inner] = np.arange(ny_inner)
        start = ny_inner + 3 * myg
        ymap[start:start + ny - ny_inner] = np.arange(ny_inner, ny)
    else:
        ymap[myg:myg + ny] = np.arange(ny)
    return ymap


def valid_parallel_rows(ymap, cuts):
    """Rows where a centred y-difference stays inside one topological region.

    ``cuts`` are the global y indices after which the field line is cut
    (X-point branch cuts ``jyseps*``, and ``ny_inner-1`` = upper targets).  A
    row is invalid if it or its +-1 neighbour is a guard, or the stencil spans
    a cut.  Conservative: in the SOL some X-point cuts are actually continuous,
    but a single global zShift alignment is not trusted across any of them.
    """
    npol = len(ymap)
    ok = np.zeros(npol, dtype=bool)
    cuts = set(int(c) for c in cuts)
    for r in range(1, npol - 1):
        y0, ym, yp = ymap[r], ymap[r - 1], ymap[r + 1]
        if min(y0, ym, yp) < 0:
            continue
        if ym != y0 - 1 or yp != y0 + 1:
            continue
        if (y0 - 1) in cuts or y0 in cuts:
            continue
        ok[r] = True
    return ok


def grad_par_aligned(f_aligned, dy, g22, ok_rows):
    """``Grad_par`` of a field-aligned (x, y, z) field: centred d/dy / (dy sqrt(g_22)).

    Rows not in ``ok_rows`` are NaN.
    """
    df = np.full(f_aligned.shape, np.nan)
    df[:, 1:-1, :] = (f_aligned[:, 2:, :] - f_aligned[:, :-2, :]) / 2.0
    df /= (dy * np.sqrt(g22))[..., None]
    df[:, ~ok_rows, :] = np.nan
    return df


def conduction_ratio(kappa_par, T, P, AA, gradT):
    """``R1 = |kappa_par Grad_par T| / (N T sqrt(T/AA))`` (alpha = 1)."""
    q_cl = kappa_par * gradT
    q_fl = P * np.sqrt(np.maximum(T, 1e-10) / AA)
    return np.abs(q_cl) / np.maximum(q_fl, 1e-30), q_cl


def viscosity_ratio(kappa_par, P, AA, gradV, kind, kappa_coeff=None):
    """``R1 = |eta Grad_par V| / P`` (alpha = 1), eta rebuilt from kappa_par.

    ``kind`` is "e" or "i" (see :func:`species_kind`); ``kappa_coeff`` overrides
    the default conduction coefficient if the run set ``kappa_coefficient``.
    """
    kc = KAPPA_COEFF[kind] if kappa_coeff is None else kappa_coeff
    Ptau = kappa_par * AA / kc
    eta = ETA_COEFF[kind] * Ptau
    q_cl = eta * gradV
    return np.abs(q_cl) / np.maximum(np.abs(P), 1e-30), q_cl


def limiter_summary(R1, q_cl, mask, alphas):
    """For each alpha: fraction of masked cells cut by >10 %, and the
    flux-weighted fraction of the classical flux that would survive,
    ``sum|q|/(1+R1/alpha) / sum|q|``.  Also the R1 percentiles."""
    r = R1[mask]
    q = np.abs(q_cl[mask])
    good = np.isfinite(r) & np.isfinite(q)
    r, q = r[good], q[good]
    out = {"n": r.size,
           "R1_p50": np.percentile(r, 50) if r.size else np.nan,
           "R1_p90": np.percentile(r, 90) if r.size else np.nan,
           "R1_p99": np.percentile(r, 99) if r.size else np.nan}
    for a in alphas:
        fac = 1.0 / (1.0 + r / a)
        out[a] = (np.mean(fac < 0.9) if r.size else np.nan,
                  np.sum(q * fac) / np.sum(q) if q.sum() > 0 else np.nan)
    return out

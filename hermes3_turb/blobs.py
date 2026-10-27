"""
Blob (filament) identification, tracking and radial-velocity measurement for
Hermes-3 3D turbulence simulations.

Everything here works on a **single poloidal slice** (normally the outboard
midplane) reduced to a ``(t, x, z)`` array, so a long time series stitched from
many restart folders costs ~1 MB per field per folder instead of gigabytes.

Design notes for *realistic* SOL turbulence
------------------------------------------
Blobs in a full turbulence run are not the clean circular Gaussians of a seeded
single-blob simulation. Three things follow, and this module is built around
them:

1. **The background must be removed, and then flattened.** Subtracting the mean
   alone is not enough: the SOL profile falls by orders of magnitude, so a
   single absolute threshold picks up everything near the separatrix and
   nothing further out. :func:`fluctuation_field` therefore normalises by the
   local RMS, ``f = (n − ⟨n⟩)/σ(x)``, so one threshold in units of σ applies
   across the whole radial range.

2. **Shapes are irregular, so use moments, not circle fits.** Each detected
   region is characterised by its intensity-weighted centroid (sub-cell
   accurate) and its second-moment ellipse (major/minor axis and tilt). No
   circularity is assumed anywhere.

3. **The frame-to-frame displacement is usually far below one cell.** With a
   typical ``dt_dump`` a blob moves a few hundredths of a radial cell per
   frame, which is why naive tracking returns a radial velocity near zero. Use
   :func:`resolvability_report` *before* trusting any tracked velocity, and
   give the tracker a long enough time baseline (many folders).

Because of point 3 this module deliberately provides **three independent
velocity estimators**, in increasing order of robustness:

  * :func:`track_blobs` + :func:`track_radial_velocity` — follows individual
    structures. Physically the most direct, but needs a long baseline.
  * :func:`ccf_radial_velocity` — time-delay cross-correlation between radially
    separated positions (the standard two-probe experimental technique). Needs
    no identification at all, only that the pattern propagates.
  * :func:`conditional_exb_velocity` — the ExB drift averaged over the pixels
    that are inside a blob. Needs neither identification nor a time baseline,
    and within the model it *is* the blob velocity, so it is the reference the
    other two should be compared against.

A tracked velocity that disagrees strongly with the conditional ExB velocity
means the tracking is wrong, not that the physics is surprising.
"""

import numpy as np

from .load import detect_toroidal_dim, detect_poloidal_dim

__all__ = [
    "omp_index",
    "radial_coordinate",
    "major_radius",
    "binormal_coordinate",
    "poloidal_markers",
    "exb_radial_velocity_slice",
    "open_time_series",
    "fluctuation_field",
    "Blob",
    "detect_frame",
    "detect_blobs",
    "Track",
    "track_blobs",
    "track_radial_velocity",
    "track_binormal_velocity",
    "exb_binormal_velocity_slice",
    "velocity_statistics",
    "ccf_radial_velocity",
    "conditional_exb_velocity",
    "conditional_average",
    "turbulent_flux_slice",
    "resolvability_report",
    "structure_size_report",
]


# ---------------------------------------------------------------------------
# Geometry: physical coordinates of the analysis plane
# ---------------------------------------------------------------------------

def _is_unnormalised(ds):
    """True if the dataset was loaded with ``unnormalise=True``.

    xhermes stamps ``units_type='SI'`` on every variable it converts, so this
    is a reliable flag. It matters because xhermes unnormalises ``dx``, ``J``,
    ``g11`` and ``g_22`` but **not** ``g_23``, ``g_33``, ``g_13`` or ``g_11``.
    """
    return ds["g_22"].attrs.get("units_type") == "SI"


def omp_index(ds):
    """Poloidal index of the outboard midplane.

    Delegates to :func:`hermes3_turb.target.get_midplane_y_indices` so there is
    a single guard-offset convention in the package.

    Note the guard test is ``ds.sizes[pdim] > metadata['ny']``: a squashed dump
    written *with* y-boundary guards has a larger poloidal size than the
    guardless ``ny`` from the grid. Both kinds occur in practice (an SXD case
    is ``y=44`` with guards, ``y=36`` without), and the midplane index differs
    between them — 29 vs 23 for that case — so never hard-code it.
    """
    from .target import get_midplane_y_indices
    return int(get_midplane_y_indices(ds)["omp"])


def radial_coordinate(ds, y_idx, ixseps1=None):
    """Physical radial distance [m] of each x index, zero at the separatrix.

    Integrates the true perpendicular cell width ``dr = dx/√g11`` along x at
    the given poloidal index, then offsets so ``r = 0`` at ``ixseps1``. This is
    the same construction as ``target.target_physical_distance``; ``dx`` and
    ``g11`` are the two metric quantities xhermes *does* unnormalise together,
    so the ratio is consistent in either normalisation.

    Returns the **cell-centre** positions (the cumulative sum is shifted back
    by half a cell).
    """
    if ixseps1 is None:
        ixseps1 = int(ds.metadata["ixseps1"])
    pdim = detect_poloidal_dim(ds)
    dx = np.asarray(ds["dx"].isel({pdim: y_idx}).values, dtype=float)
    g11 = np.asarray(ds["g11"].isel({pdim: y_idx}).values, dtype=float)
    dr = dx / np.sqrt(g11)
    edges = np.concatenate(([0.0], np.cumsum(dr)))
    centres = 0.5 * (edges[:-1] + edges[1:])
    return centres - centres[int(ixseps1)]


def major_radius(ds, y_idx):
    """Major radius R [m] of each x index at the given poloidal index.

    Uses ``Rxy`` when the dump carries it, otherwise the BOUT++ tokamak
    identity ``g_33 = R²``. ``g_33`` is *not* unnormalised by xhermes, so it is
    rescaled by ``rho_s0`` when the rest of the dataset is in SI.
    """
    pdim = detect_poloidal_dim(ds)
    if "Rxy" in ds:
        return np.asarray(ds["Rxy"].isel({pdim: y_idx}).values, dtype=float)
    R = np.sqrt(np.asarray(ds["g_33"].isel({pdim: y_idx}).values, dtype=float))
    if _is_unnormalised(ds):
        R = R * float(ds.metadata["rho_s0"])
    return R


def binormal_coordinate(ds, y_idx, x_idx=None, project_to_binormal=False):
    """Arc length [m] along the toroidal (z) direction, and its period.

    Returns ``(b, period)`` where ``b`` has one entry per z index.

    By default this is the **toroidal** arc length ``R·ζ`` — the natural
    in-plane coordinate of an ``(x, ζ)`` slice, and the one to use for
    tracking. Set ``project_to_binormal=True`` to multiply by ``Bp/B``, giving
    the true perpendicular (binormal) distance, which is the right choice when
    quoting a *physical* structure width perpendicular to B.

    ``x_idx`` selects the flux surface at which R is evaluated (default: the
    separatrix), since R varies across the SOL.
    """
    if x_idx is None:
        x_idx = int(ds.metadata["ixseps1"])
    zdim = detect_toroidal_dim(ds)
    nz = ds.sizes[zdim]
    dz = np.asarray(ds["dz"].values, dtype=float).flat[0]
    R = major_radius(ds, y_idx)[int(x_idx)]
    scale = R
    if project_to_binormal:
        pdim = detect_poloidal_dim(ds)
        if "Bpxy" in ds and "Bxy" in ds:
            frac = float(ds["Bpxy"].isel({pdim: y_idx})[int(x_idx)]
                         / ds["Bxy"].isel({pdim: y_idx})[int(x_idx)])
        else:
            g_22 = np.asarray(ds["g_22"].isel({pdim: y_idx}).values, dtype=float)
            if _is_unnormalised(ds):
                g_22 = g_22 / float(ds.metadata["rho_s0"]) ** 2
            g_23 = np.asarray(ds["g_23"].isel({pdim: y_idx}).values, dtype=float)
            g_33 = np.asarray(ds["g_33"].isel({pdim: y_idx}).values, dtype=float)
            frac = float(np.sqrt(1.0 - g_23[int(x_idx)] ** 2
                                 / (g_22[int(x_idx)] * g_33[int(x_idx)])))
        scale = R * frac
    b = scale * dz * np.arange(nz)
    return b, scale * dz * nz


def poloidal_markers(ds):
    """Named poloidal indices for annotating an ``(x, y)`` or ``(y, z)`` plot.

    Returns ``{name: y_index}`` covering the midplanes, the X-point /
    separatrix indices (``jyseps*``, i.e. the divertor entrances) and the
    target plates, all with the guard offset already applied so they index the
    loaded array directly.
    """
    from .target import get_midplane_y_indices, get_target_y_indices

    m = ds.metadata
    pdim = detect_poloidal_dim(ds)
    ny_full = ds.sizes[pdim]
    MYG = int(m.get("MYG", 2))
    g = MYG if ny_full > int(m.get("ny", ny_full)) else 0

    marks = {}
    marks.update(get_midplane_y_indices(ds))
    try:
        marks.update(get_target_y_indices(ds))
    except Exception:
        pass
    # Divertor entrances / X-points
    marks["xpt_inner_lower"] = int(m["jyseps1_1"]) + g
    marks["xpt_inner_upper"] = int(m["jyseps2_1"]) + g
    marks["xpt_outer_upper"] = int(m["jyseps1_2"]) + 3 * g
    marks["xpt_outer_lower"] = int(m["jyseps2_2"]) + 3 * g
    return {k: int(v) for k, v in marks.items() if 0 <= int(v) < ny_full}


def exb_radial_velocity_slice(ds, y_idx, phi=None, include_y_derivs=True,
                              z_deriv="centred"):
    """Radial ExB drift velocity [m/s] on one poloidal slice, shape ``(t,x,z)``.

    Same formula as :func:`hermes3_turb.fluxes.ExB_radial_velocity`,

        v_ExB,r = ( ∂φ/∂ζ − (g_23/g_22)·∂φ/∂θ ) / √g11

    but evaluated with plain finite differences on a three-slice stencil rather
    than through xBOUT's ``.bout.ddz()``. That removes the need for
    ``geometry="toroidal"``, so it works on a guard-retaining squashed dump
    that has no matching guardless grid file.

    The ``g_23/g_22`` ratio is formed with **both** components in normalised
    units, because xhermes unnormalises ``g_22`` but not ``g_23``.

    ``include_y_derivs=True`` (the Hermes-3 convention) keeps the
    non-orthogonality correction; ``False`` reproduces the STORM convention.
    Note the θ-derivative uses a centred difference, so it is only valid away
    from branch cuts — fine at the midplane, not at an X-point.

    ``z_deriv`` selects how ``∂φ/∂ζ`` is taken, and the two answers differ by
    a lot for a marginally resolved filament. A centred difference damps the
    gradient by ``sin(k dz)/(k dz)``: measured on a Gaussian of the SXD OMP
    binormal cell size (4.6 mm = 5.7 ρ_s), the centred value is **0.53 / 0.69
    / 0.91 / 0.98** of the exact one for a structure spanning 0.7 / 1.3 / 2.6
    / 5.3 cells. On runs where the median ``size_b`` prints as one cell, the
    diagnostic alone therefore loses 30-47 % of the radial ExB velocity.

    * ``'centred'`` (default) matches the finite-difference operator the
      *simulation itself* used, so it reports the velocity the code actually
      advected with — the right choice for a budget or a flux.
    * ``'fft'`` is exact for the stored field (ζ is exactly periodic, so the
      FFT has no boundary error) and reports the velocity the φ field
      *implies*. Use it to measure how much of a slow blob velocity is
      binormal under-resolution rather than physics; the gap between the two
      is that headroom.
    """
    pdim = detect_poloidal_dim(ds)
    zdim = detect_toroidal_dim(ds)
    if phi is None:
        phi = ds["phi"]

    dz = np.asarray(ds["dz"].values, dtype=float).flat[0]
    g11 = np.asarray(ds["g11"].isel({pdim: y_idx}).values, dtype=float)

    p0 = np.asarray(phi.isel({pdim: y_idx}).values, dtype=float)     # (t,x,z)
    if z_deriv == "fft":
        nz = p0.shape[-1]
        kz = 2.0 * np.pi * np.fft.fftfreq(nz, d=dz)
        dphi_dz = np.real(np.fft.ifft(1j * kz * np.fft.fft(p0, axis=-1),
                                      axis=-1))
    elif z_deriv == "centred":
        dphi_dz = (np.roll(p0, -1, axis=-1) - np.roll(p0, 1, axis=-1)) / (2.0 * dz)
    else:
        raise ValueError(f"unknown z_deriv={z_deriv!r} (use 'centred' or 'fft')")

    v = dphi_dz
    if include_y_derivs:
        pm = np.asarray(phi.isel({pdim: y_idx - 1}).values, dtype=float)
        pp = np.asarray(phi.isel({pdim: y_idx + 1}).values, dtype=float)
        dy = np.asarray(ds["dy"].isel({pdim: y_idx}).values, dtype=float)
        dphi_dy = (pp - pm) / (2.0 * dy[None, :, None])

        g_22 = np.asarray(ds["g_22"].isel({pdim: y_idx}).values, dtype=float)
        if _is_unnormalised(ds):
            g_22 = g_22 / float(ds.metadata["rho_s0"]) ** 2
        g_23 = np.asarray(ds["g_23"].isel({pdim: y_idx}).values, dtype=float)
        v = v - (g_23 / g_22)[None, :, None] * dphi_dy

    return v / np.sqrt(g11)[None, :, None]


# ---------------------------------------------------------------------------
# Loading: many folders, one slice
# ---------------------------------------------------------------------------

def open_time_series(data_root, dir_range, fields=("Ne",), y_idx=None,
                     filename="boutdata.nc", keep_yboundaries=True,
                     include_exb=True, include_y_derivs=True,
                     exb_z_deriv_fft=True,
                     dedup_times=True, verbose=True):
    """Stitch one poloidal slice across many restart folders.

    Opens ``<data_root>/<n>/<filename>`` for each ``n`` in ``dir_range``,
    extracts the requested fields at a single poloidal index, and concatenates
    them in time. Only the slice is retained, so memory stays flat no matter
    how many folders are given — this is what makes a long enough baseline for
    blob tracking affordable.

    Parameters
    ----------
    y_idx : int, optional
        Poloidal index. Defaults to the outboard midplane (:func:`omp_index`).
    include_exb : bool
        Also evaluate :func:`exb_radial_velocity_slice` per folder (needs
        ``phi``). This is what makes the tracking-free velocity cross-check
        possible.
    exb_z_deriv_fft : bool
        Additionally store ``v_ExB_r_fft``, the same velocity with ``∂φ/∂ζ``
        taken spectrally instead of by centred difference. The ratio of the
        two is the binormal-resolution headroom on the blob velocity — see
        :func:`exb_radial_velocity_slice`. Costs one extra FFT per folder.
    dedup_times : bool
        Drop repeated/backwards time points where consecutive restart folders
        overlap (a dump written as ``[0,1,2]``, ``[2,3,4]`` shares ``t=2``).
        A duplicated frame is a zero-displacement step, which biases any
        velocity estimate **low**, so this defaults to on.

    Returns
    -------
    dict with keys: ``t`` [s], ``r`` [m], ``b`` [m], ``b_period`` [m],
    ``y_idx``, ``ixseps1``, one entry per requested field with shape
    ``(t, x, z)``, and ``v_ExB_r`` if requested.
    """
    import gc

    from xhermes import open_hermesdataset

    dirs = list(dir_range)
    out = {name: [] for name in fields}
    if include_exb:
        out["v_ExB_r"] = []
        out["v_ExB_b"] = []
        if exb_z_deriv_fft:
            out["v_ExB_r_fft"] = []
    times = []
    meta = {}

    for n in dirs:
        path = f"{data_root.rstrip('/')}/{n}/{filename}"
        if verbose:
            print(f"  loading {path}")
        ds = open_hermesdataset(path, keep_yboundaries=keep_yboundaries)
        pdim = detect_poloidal_dim(ds)
        if y_idx is None:
            y_idx = omp_index(ds)
        if not meta:
            meta = dict(
                y_idx=int(y_idx),
                ixseps1=int(ds.metadata["ixseps1"]),
                r=radial_coordinate(ds, y_idx),
                rho_s0=float(ds.metadata["rho_s0"]),
                omega_ci=float(ds.metadata["Omega_ci"]),
            )
            meta["b"], meta["b_period"] = binormal_coordinate(ds, y_idx)
            # Inputs for blob_velocity_regimes: local field, separatrix
            # temperatures and the one-way connection length to the sheath.
            _ix = meta["ixseps1"] + 2
            try:
                meta["B_slice"] = float(
                    np.asarray(ds["Bxy"].isel({pdim: y_idx}).values,
                               dtype=float).ravel()[_ix])
            except Exception:
                meta["B_slice"] = np.nan
            # Bp/B at the slice. REQUIRED to convert a detected `size_b` into
            # a physical perpendicular width: `binormal_coordinate` returns the
            # TOROIDAL arc by default (project_to_binormal=False), which is
            # right for tracking in the stored (x,zeta) plane but is NOT the
            # binormal distance. On MAST-U at the OMP Bp/B = 0.44, so a
            # `size_b` of 35.5 mm is a perpendicular 15.6 mm -- feeding the
            # unprojected value into a_b = sqrt(a_rho a_chi) overstates it by
            # sqrt(B/Bp) = 1.5x and so overstates a_hat.
            try:
                _, _bp = binormal_coordinate(ds, y_idx, x_idx=_ix,
                                             project_to_binormal=True)
                _, _tor = binormal_coordinate(ds, y_idx, x_idx=_ix)
                meta["BpB_slice"] = float(_bp / _tor) if _tor else np.nan
            except Exception:
                meta["BpB_slice"] = np.nan
            for _nm, _key in (("Te", "Te_sep"), ("Ti", "Ti_sep")):
                try:
                    meta[_key] = float(
                        ds[_nm].isel({pdim: y_idx}).values[..., _ix, :].mean())
                except Exception:
                    meta[_key] = np.nan
            try:
                _ld, _lu, _lt = parallel_half_length(ds, y_idx, x_idx=_ix)
                meta["L_par"] = float(min(_ld, _lu))
                meta["L_par_total"] = float(_lt)
            except Exception:
                meta["L_par"] = meta["L_par_total"] = np.nan
        times.append(np.asarray(ds["t"].values, dtype=float))
        for name in fields:
            out[name].append(np.asarray(ds[name].isel({pdim: y_idx}).values,
                                        dtype=float))
        if include_exb:
            out["v_ExB_r"].append(
                exb_radial_velocity_slice(ds, y_idx,
                                          include_y_derivs=include_y_derivs))
            out["v_ExB_b"].append(exb_binormal_velocity_slice(ds, y_idx))
            if exb_z_deriv_fft:
                out["v_ExB_r_fft"].append(
                    exb_radial_velocity_slice(
                        ds, y_idx, include_y_derivs=include_y_derivs,
                        z_deriv="fft"))
        # Release the netCDF/HDF5 handle before opening the next folder.
        # Left to the garbage collector these accumulate and eventually
        # segfault h5netcdf part-way through a long folder list.
        ds.close()
        del ds
        gc.collect()

    t = np.concatenate(times)
    res = {k: np.concatenate(v, axis=0) for k, v in out.items()}

    if dedup_times and t.size > 1:
        # Keep a frame only if it is later than every frame kept so far.
        # Comparing against the immediately preceding frame is not enough: a
        # folder that starts well before the previous one ends would then have
        # its first point dropped but the rest kept, leaving a non-monotonic
        # series and a negative dt that silently corrupts every velocity.
        #
        # This is by time VALUE, never by position: dropping each folder's
        # first frame assumes BOUT++'s default dump_at_restart=true, and with
        # dump_at_restart=false that frame is a genuinely NEW time. Same logic
        # as load.dedup_restart_times (which takes a Dataset; this works on
        # the concatenated numpy arrays), including its tolerance -- a restart
        # time differing only in the last bits is not caught by a strict `>`
        # and leaves a near-zero dt.
        _d = np.diff(t)
        _med = float(np.median(_d[_d > 0])) if np.any(_d > 0) else 0.0
        _tol = 1e-3 * _med
        keep = np.ones(t.size, dtype=bool)
        _run = -np.inf
        for _i, _ti in enumerate(t):
            if _ti > _run + _tol:
                _run = _ti
            else:
                keep[_i] = False
        ndrop = int((~keep).sum())
        if ndrop and verbose:
            print(f"  dropped {ndrop} duplicated/out-of-order time point(s) "
                  f"at restart boundaries")
        t = t[keep]
        res = {k: v[keep] for k, v in res.items()}
        if t.size > 1 and not np.all(np.diff(t) > 0):
            raise RuntimeError("time series is still not monotonic after "
                               "de-duplication — check the folder order")

    res.update(t=t, **{k: meta[k] for k in
                       ("r", "b", "b_period", "y_idx", "ixseps1", "rho_s0",
                        "omega_ci", "B_slice", "BpB_slice", "Te_sep",
                        "Ti_sep", "L_par", "L_par_total") if k in meta})
    return res


# ---------------------------------------------------------------------------
# Background removal
# ---------------------------------------------------------------------------

def fluctuation_field(n, normalise="sigma", detrend_window=None, eps=1e-30):
    """Background-subtracted, radially flattened fluctuation field.

    Parameters
    ----------
    n : ndarray ``(t, x, z)``
        Raw field (e.g. density) on the analysis slice.
    normalise : {'sigma', 'mean', 'none'}
        ``'sigma'`` returns ``(n − ⟨n⟩)/σ(x)`` — the recommended choice. The
        SOL profile falls steeply, so without dividing by the local RMS a
        single threshold selects only the near-separatrix region. With it, a
        threshold of 2–3 has the same meaning at every radius.
        ``'mean'`` returns ``(n − ⟨n⟩)/⟨n⟩`` (relative fluctuation);
        ``'none'`` returns the bare ``n − ⟨n⟩``.
    detrend_window : int, optional
        If given, the background ``⟨n⟩`` is a centred moving average over this
        many frames instead of the mean of the whole series. Use it when the
        stitched series is long enough that the mean profile itself evolves —
        otherwise a slow profile drift leaks into the fluctuation field and
        fakes structures.

    References
    ----------
    Thresholding the amplitude-normalised fluctuation is the standard
    convention for blob/filament identification, with 2.5 sigma the common
    trigger level:

    * D. A. D'Ippolito, J. R. Myra & S. J. Zweben, "Convective transport by
      intermittent blob-filaments: Comparison of theory and experiment",
      Phys. Plasmas **18**, 060501 (2011) — the review of the field.
    * J. A. Boedo et al., "Transport by intermittent convection in the
      boundary of the DIII-D tokamak", Phys. Plasmas **8**, 4826 (2001),
      doi:10.1063/1.1406940 — conditional averaging of probe signals.
    * G. Decristoforo et al., "Blob interaction in 2D scrape off layer
      simulations", arXiv:2007.04260 — the simulation-side equivalent, and
      the basis of the ``xblobs`` package; blobs are ``n > mean + 2.5 sigma``.

    Note the literature convention normalises a *time series at one probe
    position* by its own sigma. Dividing a 2-D field by ``sigma(x)`` at every
    radius, as here, is the natural extension of that to a whole radial
    profile — it is what makes one threshold valid across a SOL that falls by
    orders of magnitude — rather than something taken verbatim from a paper.

    Returns
    -------
    f, background, sigma : the flattened field ``(t,x,z)``, the background
    (``(x,)`` or ``(t,x)`` when detrended) and the RMS profile ``(x,)``.
    """
    n = np.asarray(n, dtype=float)
    if detrend_window:
        w = int(detrend_window)
        prof = n.mean(axis=2)                       # (t, x)
        kern = np.ones(w) / w
        pad = w // 2
        padded = np.pad(prof, ((pad, pad), (0, 0)), mode="edge")
        bg = np.apply_along_axis(
            lambda a: np.convolve(a, kern, mode="same"), 0, padded)[pad:pad + n.shape[0]]
        dn = n - bg[:, :, None]
    else:
        bg = n.mean(axis=(0, 2))                    # (x,)
        dn = n - bg[None, :, None]

    sigma = dn.std(axis=(0, 2))                     # (x,)
    if normalise == "sigma":
        f = dn / (sigma[None, :, None] + eps)
    elif normalise == "mean":
        ref = bg if bg.ndim == 1 else bg.mean(axis=0)
        f = dn / (np.abs(ref)[None, :, None] + eps)
    elif normalise == "none":
        f = dn
    else:
        raise ValueError(f"unknown normalise={normalise!r}")
    return f, bg, sigma


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------

class Blob:
    """One detected structure in one frame.

    Attributes are all physical. ``r``/``b`` are the intensity-weighted
    centroid ("centre of mass") in metres — the position everything downstream
    uses, because it is sub-cell accurate. ``r_front`` / ``r_back`` are the
    outermost / innermost cell of the region and ``r_peak`` the location of
    the maximum, all in metres: these are quantised to the grid, so they are
    noisier trackers, but ``r_front`` is the quantity blob theory usually
    means by the filament front. ``major``/``minor``/``tilt`` describe the
    second-moment ellipse (no circularity assumed), ``amplitude`` is the peak
    value of the flattened field and ``mass`` its integral over the region.
    """

    __slots__ = ("it", "time", "r", "b", "amplitude", "npix", "area",
                 "size_r", "size_b", "major", "minor", "tilt", "mass",
                 "ix_peak", "iz_peak", "r_front", "r_back", "r_peak")

    def __init__(self, **kw):
        for k in self.__slots__:
            setattr(self, k, kw.get(k, np.nan))

    def __repr__(self):
        return (f"Blob(t={self.time:.3e}s r={self.r*1e3:.2f}mm "
                f"b={self.b*1e3:.1f}mm A={self.amplitude:.2f} "
                f"npix={self.npix})")


def _label_periodic(mask):
    """Connected-component label of a 2-D mask, periodic in the second axis."""
    from scipy import ndimage
    lab, _ = ndimage.label(mask)
    # Merge labels that touch across the z wrap-around.
    for i in range(lab.shape[0]):
        a, b = lab[i, 0], lab[i, -1]
        if a > 0 and b > 0 and a != b:
            lab[lab == b] = a
    return lab


def detect_frame(f2d, r, b, b_period, threshold=2.5, min_pixels=4,
                 it=0, time=np.nan, weight="excess"):
    """Find structures in one ``(x, z)`` frame of the flattened field.

    Regions are the connected components of ``f2d > threshold`` (periodic in
    z) with at least ``min_pixels`` cells. Each is reduced to moments rather
    than fitted with a circle, so irregular and elongated shapes are handled
    without bias.

    ``weight='excess'`` centroids on ``f − threshold`` (clipped at zero), which
    is the standard choice: it puts the centroid on the structure's core and is
    insensitive to exactly where the boundary contour falls. ``weight='value'``
    uses ``f`` itself.

    The z centroid is computed as a **circular** mean so a structure straddling
    the periodic boundary is not split or pulled to the middle of the domain.
    """
    mask = f2d > threshold
    if not mask.any():
        return []
    lab = _label_periodic(mask)
    dr = np.gradient(r)
    db = b_period / len(b)

    blobs = []
    for lid in np.unique(lab):
        if lid == 0:
            continue
        sel = lab == lid
        npix = int(sel.sum())
        if npix < min_pixels:
            continue
        ix, iz = np.nonzero(sel)
        w = (f2d[ix, iz] - threshold) if weight == "excess" else f2d[ix, iz]
        w = np.clip(w, 0.0, None)
        if w.sum() <= 0:
            continue

        r_c = float(np.average(r[ix], weights=w))
        # circular mean over the periodic binormal coordinate
        ang = 2.0 * np.pi * b[iz] / b_period
        cx = float(np.average(np.cos(ang), weights=w))
        cy = float(np.average(np.sin(ang), weights=w))
        b_c = float(np.arctan2(cy, cx) % (2.0 * np.pi) * b_period / (2.0 * np.pi))

        # second moments (binormal offsets taken on the shortest periodic path)
        dr_i = r[ix] - r_c
        db_i = (b[iz] - b_c + 0.5 * b_period) % b_period - 0.5 * b_period
        wsum = w.sum()
        mrr = float(np.sum(w * dr_i ** 2) / wsum)
        mbb = float(np.sum(w * db_i ** 2) / wsum)
        mrb = float(np.sum(w * dr_i * db_i) / wsum)
        cov = np.array([[mrr, mrb], [mrb, mbb]])
        evals, evecs = np.linalg.eigh(cov)
        evals = np.clip(evals, 0.0, None)
        major = 2.0 * float(np.sqrt(evals[1]))
        minor = 2.0 * float(np.sqrt(evals[0]))
        tilt = float(np.degrees(np.arctan2(evecs[1, 1], evecs[0, 1])))

        pk = int(np.argmax(f2d[ix, iz]))
        area = float(np.sum(dr[ix]) * db)
        blobs.append(Blob(
            it=it, time=time, r=r_c, b=b_c,
            amplitude=float(f2d[ix, iz].max()),
            npix=npix, area=area,
            size_r=2.0 * float(np.sqrt(mrr)), size_b=2.0 * float(np.sqrt(mbb)),
            major=major, minor=minor, tilt=tilt,
            mass=float(np.sum(f2d[ix, iz] * dr[ix]) * db),
            ix_peak=int(ix[pk]), iz_peak=int(iz[pk]),
            r_front=float(r[ix].max()), r_back=float(r[ix].min()),
            r_peak=float(r[ix[pk]]),
        ))
    return blobs


def detect_blobs(f, r, b, b_period, t=None, threshold=2.5, min_pixels=4,
                 x_slice=None, weight="excess"):
    """Run :func:`detect_frame` over every frame. Returns a list per frame.

    ``x_slice`` restricts detection to a radial window (e.g. the SOL only);
    the returned centroids are still in the global physical coordinate.
    """
    f = np.asarray(f)
    if t is None:
        t = np.arange(f.shape[0], dtype=float)
    xs = x_slice or slice(None)
    rr = r[xs]
    return [detect_frame(f[i][xs], rr, b, b_period, threshold=threshold,
                         min_pixels=min_pixels, it=i, time=float(t[i]),
                         weight=weight)
            for i in range(f.shape[0])]


# ---------------------------------------------------------------------------
# Tracking
# ---------------------------------------------------------------------------

class Track:
    """A blob followed across frames. ``blobs`` is ordered in time."""

    __slots__ = ("blobs", "tid")

    def __init__(self, blobs, tid=-1):
        self.blobs = list(blobs)
        self.tid = tid

    def __len__(self):
        return len(self.blobs)

    @property
    def times(self):
        return np.array([bl.time for bl in self.blobs])

    @property
    def r(self):
        return np.array([bl.r for bl in self.blobs])

    def b_unwrapped(self, b_period):
        """Binormal position [m] with the periodic wrap removed.

        Each step is taken along the shortest periodic path, so a blob that
        crosses the z boundary gives a continuous trajectory instead of a
        jump of one period.
        """
        raw = np.array([bl.b for bl in self.blobs])
        if raw.size < 2:
            return raw
        steps = (np.diff(raw) + 0.5 * b_period) % b_period - 0.5 * b_period
        return np.concatenate(([raw[0]], raw[0] + np.cumsum(steps)))

    @property
    def amplitude(self):
        return np.array([bl.amplitude for bl in self.blobs])

    def __repr__(self):
        return (f"Track(id={self.tid}, n={len(self)}, "
                f"r {self.r[0]*1e3:.1f}→{self.r[-1]*1e3:.1f} mm)")


def track_blobs(frames, b_period, max_dr=None, max_db=None, max_gap=1,
                min_length=3, v_r_guess=0.0, v_b_guess=0.0, dt=None):
    """Link per-frame detections into tracks by optimal assignment.

    Consecutive frames are matched with a global optimal assignment
    (Hungarian algorithm) rather than greedy nearest-neighbour, so two blobs
    that pass close to one another are not swapped. Binormal distance is
    measured on the shortest periodic path.

    Parameters
    ----------
    max_dr, max_db : float
        Gate on the frame-to-frame displacement [m]. A pair further apart than
        this is never linked. Set them from the largest plausible velocity
        times ``dt`` — too generous and unrelated structures get stitched into
        a fake fast track; too tight and real blobs are cut into fragments.
    max_gap : int
        How many frames a blob may drop below the detection threshold and
        still be picked up again by the same track.
    v_r_guess, v_b_guess : float
        Optional expected drift [m/s], used with ``dt`` to predict where a
        blob should be before matching. Helps when the binormal motion is fast
        (the usual case: blobs move much faster along z than in x).
    min_length : int
        Discard tracks shorter than this many frames.
    """
    from scipy.optimize import linear_sum_assignment

    if max_dr is None:
        max_dr = np.inf
    if max_db is None:
        max_db = np.inf
    shift_r = (v_r_guess * dt) if dt else 0.0
    shift_b = (v_b_guess * dt) if dt else 0.0

    open_tracks, done = [], []
    for frame in frames:
        # age out tracks that have been unmatched for too long
        still, closed = [], []
        for tr, miss in open_tracks:
            (still if miss <= max_gap else closed).append((tr, miss))
        open_tracks = still
        done.extend(tr for tr, _ in closed)

        if not frame:
            open_tracks = [(tr, miss + 1) for tr, miss in open_tracks]
            continue
        if not open_tracks:
            open_tracks = [(Track([bl]), 0) for bl in frame]
            continue

        cost = np.full((len(open_tracks), len(frame)), np.inf)
        for i, (tr, miss) in enumerate(open_tracks):
            last = tr.blobs[-1]
            steps = miss + 1
            pr = last.r + shift_r * steps
            pb = last.b + shift_b * steps
            for j, bl in enumerate(frame):
                ddr = abs(bl.r - pr)
                ddb = abs((bl.b - pb + 0.5 * b_period) % b_period - 0.5 * b_period)
                if ddr > max_dr * steps or ddb > max_db * steps:
                    continue
                # normalise each axis by its gate so neither dominates
                cost[i, j] = ((ddr / (max_dr * steps if np.isfinite(max_dr) else 1.0)) ** 2
                              + (ddb / (max_db * steps if np.isfinite(max_db) else 1.0)) ** 2)

        big = 1e6
        rows, cols = linear_sum_assignment(np.where(np.isfinite(cost), cost, big))
        matched_t, matched_b = set(), set()
        for i, j in zip(rows, cols):
            if np.isfinite(cost[i, j]):
                open_tracks[i][0].blobs.append(frame[j])
                matched_t.add(i)
                matched_b.add(j)

        new = []
        for i, (tr, miss) in enumerate(open_tracks):
            new.append((tr, 0 if i in matched_t else miss + 1))
        for j, bl in enumerate(frame):
            if j not in matched_b:
                new.append((Track([bl]), 0))
        open_tracks = new

    done.extend(tr for tr, _ in open_tracks)
    tracks = [tr for tr in done if len(tr) >= min_length]
    for k, tr in enumerate(tracks):
        tr.tid = k
    return tracks


def track_radial_velocity(track, min_length=3, dr_cell=None, min_cells=2.0,
                          r_grid=None):
    """Radial velocity of one track from a least-squares fit of ``r(t)``.

    A straight-line fit over the whole track is used rather than a first/last
    difference, so centroid jitter averages down instead of setting the answer.

    ``sigma_v`` is only the *statistical* error of the straight-line fit. It
    does **not** include the systematic error from the centroid shifting
    because the blob changed shape rather than moved, which dominates when the
    total displacement is a small number of cells. Pass ``dr_cell`` (the local
    radial cell width) to get a ``resolved`` flag that is True only when the
    track moved at least ``min_cells`` cells in total — treat an unresolved
    track's velocity as noise regardless of how small ``sigma_v`` looks.

    Returns ``dict`` with ``v_r`` [m/s], ``sigma_v``, ``r_mean``,
    ``dr_total``, ``duration``, ``n``, ``amplitude``, ``cells_moved`` and
    ``resolved``. ``v_r`` is NaN for tracks shorter than ``min_length``.
    """
    t = track.times
    r = track.r
    n = len(t)
    if n < min_length or np.ptp(t) <= 0:
        return dict(v_r=np.nan, sigma_v=np.nan, r_mean=float(np.mean(r)),
                    dr_total=np.nan, duration=0.0, n=n,
                    amplitude=float(np.mean(track.amplitude)), r2=np.nan,
                    cells_moved=np.nan, dr_cell_local=np.nan, resolved=False)
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, res, *_ = np.linalg.lstsq(A, r, rcond=None)
    v = float(coef[0])
    if n > 2:
        pred = A @ coef
        s2 = float(np.sum((r - pred) ** 2) / (n - 2))
        sxx = float(np.sum((t - t.mean()) ** 2))
        sig = float(np.sqrt(s2 / sxx)) if sxx > 0 else np.nan
    else:
        sig = np.nan

    # R^2 of the straight-line fit. A clean filament moving steadily gives
    # R^2 > 0.98; a value well below that means the centroid wandered back and
    # forth, which is the signature of the tracker hopping between merging and
    # splitting structures rather than following one. The fitted slope is then
    # a difference between unrelated endpoints, not a velocity.
    ss_tot = float(np.sum((r - r.mean()) ** 2))
    r2 = float(1.0 - np.sum((r - (A @ coef)) ** 2) / ss_tot) if ss_tot > 0 else np.nan

    out = dict(v_r=v, sigma_v=sig, r_mean=float(np.mean(r)),
               dr_total=float(r[-1] - r[0]), duration=float(t[-1] - t[0]),
               n=n, amplitude=float(np.mean(track.amplitude)), r2=r2)
    dr_local = None
    if r_grid is not None:
        rg = np.asarray(r_grid, dtype=float)
        dr_local = float(np.gradient(rg)[int(np.argmin(np.abs(rg - out["r_mean"])))])
    elif dr_cell is not None:
        dr_local = float(dr_cell)

    if dr_local is not None and np.isfinite(dr_local) and dr_local > 0:
        out["dr_cell_local"] = dr_local
        out["cells_moved"] = abs(out["dr_total"]) / dr_local
        out["resolved"] = bool(out["cells_moved"] >= min_cells)
    else:
        out["dr_cell_local"] = np.nan
        out["cells_moved"] = np.nan
        out["resolved"] = True
    return out


# ---------------------------------------------------------------------------
# Tracking-free estimators
# ---------------------------------------------------------------------------

def track_binormal_velocity(track, b_period, min_length=3):
    """Binormal velocity of one track [m/s], from the unwrapped trajectory.

    Worth computing even when only the radial velocity is wanted: blobs move
    much faster along the binormal than radially, so this component *is* well
    resolved by the frame rate. If it comes out at a sensible fraction of the
    ExB binormal drift, the frame-to-frame linking is sound and a poor radial
    number is a resolution problem, not a tracking bug.
    """
    t = track.times
    b = track.b_unwrapped(b_period)
    n = len(t)
    if n < min_length or np.ptp(t) <= 0:
        return dict(v_b=np.nan, sigma_v=np.nan, db_total=np.nan, n=n)
    A = np.vstack([t - t.mean(), np.ones_like(t)]).T
    coef, *_ = np.linalg.lstsq(A, b, rcond=None)
    sig = np.nan
    if n > 2:
        s2 = float(np.sum((b - A @ coef) ** 2) / (n - 2))
        sxx = float(np.sum((t - t.mean()) ** 2))
        sig = float(np.sqrt(s2 / sxx)) if sxx > 0 else np.nan
    return dict(v_b=float(coef[0]), sigma_v=sig,
                db_total=float(b[-1] - b[0]), n=n)


def exb_binormal_velocity_slice(ds, y_idx, phi=None):
    """Binormal ExB drift [m/s] on one slice, ``(t,x,z)``.

    ``v_E,binormal = −(√g11 · ∂φ/∂x) / B = −E_r/B`` — the component
    perpendicular to both B and the radial direction. This is the *fast* drift:
    it is driven by the **mean radial electric field**, whereas the radial ExB
    velocity is driven by the much smaller fluctuating binormal field. A
    binormal velocity one to two orders of magnitude above the radial one is
    therefore expected, not a bug — a few kV/m of E_r over a sub-tesla field is
    already ~10 km/s.

    The overall sign depends on the orientation convention for b̂, so compare
    magnitudes when checking a tracked velocity. The **variation with radius**
    is convention-independent, and a sign reversal across the SOL is the
    familiar E_r shear layer.
    """
    pdim = detect_poloidal_dim(ds)
    if phi is None:
        phi = ds["phi"]
    p0 = np.asarray(phi.isel({pdim: y_idx}).values, dtype=float)   # (t,x,z)
    dx = np.asarray(ds["dx"].isel({pdim: y_idx}).values, dtype=float)
    g11 = np.asarray(ds["g11"].isel({pdim: y_idx}).values, dtype=float)
    B = np.asarray(ds["Bxy"].isel({pdim: y_idx}).values, dtype=float)
    dphi_dx = np.gradient(p0, axis=1) / dx[None, :, None]
    return -np.sqrt(g11)[None, :, None] * dphi_dx / B[None, :, None]


def velocity_statistics(tracks, b_period=None, dr_cell=None, min_cells=2.0,
                        min_length=3, r_grid=None, n_sigma=2.0):
    """Ensemble summary of tracked velocities.

    Returns a dict of arrays (one entry per track) plus summary scalars. The
    summary deliberately reports the median and the inter-quartile range of the
    **resolved** tracks only: with irregular blobs the distribution has heavy
    tails from mis-linked fragments, and a mean is not robust against them.
    """
    if r_grid is None and dr_cell is not None and tracks:
        # Scoring every track against a single cell width is only correct on a
        # uniform radial mesh. SOL meshes are not uniform (a factor 16 across
        # the SOL in one STORM case), so this silently overstates the
        # displacement of tracks away from the reference radius and marks
        # under-resolved tracks as resolved. Warn rather than fail, because
        # the scalar path is still the right answer on a uniform grid — but a
        # caller that simply has not been updated to pass r_grid should hear
        # about it instead of getting quietly wrong `resolved` flags.
        import warnings
        warnings.warn(
            "velocity_statistics: r_grid not given, so 'cells_moved' uses a "
            "single cell width for every track. On a non-uniform radial mesh "
            "this overstates the displacement of tracks away from the "
            "reference radius. Pass r_grid=<radial coordinate array>.",
            RuntimeWarning, stacklevel=2)

    rows = [track_radial_velocity(tr, min_length=min_length, dr_cell=dr_cell,
                                  min_cells=min_cells, r_grid=r_grid)
            for tr in tracks]
    keys = ("v_r", "sigma_v", "r_mean", "dr_total", "duration", "n",
            "amplitude", "r2", "cells_moved", "dr_cell_local", "resolved")
    out = {k: np.array([row[k] for row in rows]) for k in keys}
    if b_period is not None:
        vb = [track_binormal_velocity(tr, b_period, min_length=min_length)
              for tr in tracks]
        out["v_b"] = np.array([q["v_b"] for q in vb])
        out["db_total"] = np.array([q["db_total"] for q in vb])

    ok = out["resolved"].astype(bool) & np.isfinite(out["v_r"])
    fin = np.isfinite(out["v_r"])
    out["n_tracks"] = len(tracks)
    out["n_resolved"] = int(ok.sum())

    def _summary(mask, tag):
        if mask.any():
            v = out["v_r"][mask]
            out[f"v_r_median{tag}"] = float(np.median(v))
            out[f"v_r_q1{tag}"] = float(np.percentile(v, 25))
            out[f"v_r_q3{tag}"] = float(np.percentile(v, 75))
            out[f"v_r_sem{tag}"] = float(np.std(v) / np.sqrt(v.size))
        else:
            for k in ("v_r_median", "v_r_q1", "v_r_q3", "v_r_sem"):
                out[f"{k}{tag}"] = np.nan

    # `_all` is the headline estimator. `_resolved` is kept for comparison but
    # is BIASED HIGH: cells_moved = |v_r|·duration/dr_cell is proportional to
    # the very quantity being measured, so cutting on it preferentially
    # removes slow tracks. Centroid noise, by contrast, is roughly symmetric,
    # so keeping noisy tracks inflates the spread of the median without
    # shifting it. On the STORM sxd_n1 case the unfiltered median (134 m/s)
    # matches the selection-free conditional ExB (132 m/s) to 2 %, while the
    # resolved-only median (172 m/s) is 30 % high.
    _summary(fin, "")
    _summary(ok, "_resolved")

    # Is v_r distinguishable from zero? This is the physically meaningful
    # question, and it is NOT the same as `resolved`. `cells_moved` mixes
    # velocity with track lifetime, so a short-lived blob with a perfectly
    # well-measured velocity fails it, while a genuinely slow blob (v_r ~ 0
    # with large binormal motion — entirely normal in a sheared E_r layer)
    # gets described as unmeasurable when in fact it is measured, and small.
    # `sigma_v` is the standard error of the fitted slope, so |v_r|/sigma_v is
    # a proper significance ratio. Caveat: it captures scatter about the line
    # only, not a systematic centroid bias from shape change.
    with np.errstate(divide="ignore", invalid="ignore"):
        snr = np.abs(out["v_r"]) / out["sigma_v"]
    out["v_snr"] = snr
    out["v_significant"] = np.where(np.isfinite(snr), snr > n_sigma, False)
    out["n_significant"] = int(out["v_significant"].sum())
    return out


def ccf_radial_velocity(f, r, t, x_ref, x_offsets=(2, 4, 6, 8), max_lag=None,
                        z_slice=None, b=None, b_period=None, max_zlag=None,
                        binormal_scan=True):
    """Propagation velocity from the two-point time-delay cross-correlation.

    The numerical version of a two-probe experimental measurement: the
    fluctuation signal at ``x_ref`` is correlated against the signal a few
    cells further out, and the lag of the correlation peak gives the transit
    time over the known radial separation. No structure identification is
    involved, so it survives the irregular, fragmented blobs that defeat
    tracking.

    **The correlation must be scanned over the BINORMAL shift as well, not
    just the time lag.** A SOL filament moves binormally 10-100x faster than
    radially (``exb_binormal_velocity_slice`` is driven by the *mean* E_r,
    the radial drift only by the small fluctuating binormal field), so by the
    time it reaches ``x_ref + offset`` it has left the z column it started
    in. Correlating at *matched z* — which is what a 1-D lag scan does — then
    measures the stationary radial shape, not the propagation.

    Measured on synthetic blobs with a known ``v_r = 400 m/s``: at
    ``v_b = 0`` the 1-D scan gives 470 m/s, but at ``v_b = 6000 m/s`` every
    offset lands at zero lag (no answer at all), and for a narrower blob it
    returns **113 m/s with ``resolved=True``** — a silent 3.5x
    *under*-estimate, i.e. exactly the "blobs are too slow" symptom,
    manufactured by the diagnostic. The 2-D scan recovers 337-487 m/s in
    every one of those cases and ``v_b`` to 0.3 %. This is also why a real
    two-probe measurement uses a *poloidally* separated probe pair, or 2-D
    GPI velocimetry, rather than one radial pair.

    ``binormal_scan=True`` (the default) therefore scans ``C(dt_lag, dz_lag)``
    and takes the joint peak, giving ``v_b`` for free. Pass ``b``/``b_period``
    to get ``v_b`` in m/s (without them the binormal shift is still scanned,
    so ``v_r`` is correct, but ``v_b`` comes back in cells/s). Set
    ``binormal_scan=False`` to reproduce the old matched-z behaviour.

    The peak is refined by a parabolic fit through the three samples around
    the maximum in each direction, giving sub-frame and sub-cell resolution —
    necessary because the true delay is often a fraction of ``dt``.

    Returns ``dict`` with ``dr`` [m], ``lag`` [s], ``v_r`` [m/s],
    ``peak_corr``, ``zlag`` [m] and ``v_b`` [m/s] per offset, plus a
    ``resolved`` flag that is False when the correlation peak sits at zero
    lag (transit time below one frame — the measurement is then an upper
    bound on the delay, i.e. a *lower* bound on the velocity).
    """
    f = np.asarray(f, dtype=float)
    zs = z_slice or slice(None)
    dt = float(np.median(np.diff(t)))
    nt = f.shape[0]
    if max_lag is None:
        max_lag = max(1, nt // 3)

    a = f[:, int(x_ref), zs]
    a = a - a.mean(axis=0, keepdims=True)
    nz = a.shape[1]
    db = (float(b_period) / nz) if b_period else 1.0
    if max_zlag is None:
        max_zlag = nz // 2
    zlags = (np.arange(-int(max_zlag), int(max_zlag) + 1) if binormal_scan
             else np.array([0]))

    out = {"dr": [], "lag": [], "v_r": [], "peak_corr": [], "resolved": [],
           "offset": [], "zlag": [], "v_b": []}
    for off in x_offsets:
        ix = int(x_ref) + int(off)
        if ix >= f.shape[1]:
            continue
        c = f[:, ix, zs]
        c = c - c.mean(axis=0, keepdims=True)
        lags = np.arange(-max_lag, max_lag + 1)
        corr = np.zeros((lags.size, zlags.size))
        for k, L in enumerate(lags):
            if L >= 0:
                x1, x2 = a[: nt - L], c[L:]
            else:
                x1, x2 = a[-L:], c[: nt + L]
            n1 = np.sqrt((x1 ** 2).sum())
            for m, Z in enumerate(zlags):
                # roll the OUTER signal back by Z so a structure that has
                # drifted +Z cells binormally is brought back onto the inner
                # one; the peak then sits at (transit time, binormal drift).
                x2z = np.roll(x2, -int(Z), axis=1) if Z else x2
                denom = n1 * np.sqrt((x2z ** 2).sum())
                corr[k, m] = (x1 * x2z).sum() / denom if denom > 0 else 0.0
        kpk, mpk = np.unravel_index(int(np.argmax(corr)), corr.shape)
        lag_frames = float(lags[kpk])
        zlag_cells = float(zlags[mpk])
        # parabolic sub-sample refinement, independently in each direction
        tline = corr[:, mpk]
        if 0 < kpk < tline.size - 1:
            y0, y1, y2 = tline[kpk - 1], tline[kpk], tline[kpk + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                lag_frames += 0.5 * (y0 - y2) / den
        zline = corr[kpk, :]
        if 0 < mpk < zline.size - 1:
            y0, y1, y2 = zline[mpk - 1], zline[mpk], zline[mpk + 1]
            den = y0 - 2 * y1 + y2
            if den != 0:
                zlag_cells += 0.5 * (y0 - y2) / den
        dr = float(r[ix] - r[int(x_ref)])
        lag = lag_frames * dt
        out["offset"].append(int(off))
        out["dr"].append(dr)
        out["lag"].append(lag)
        out["zlag"].append(zlag_cells * db)
        out["v_r"].append(dr / lag if lag != 0 else np.inf)
        out["v_b"].append(zlag_cells * db / lag if lag != 0 else np.nan)
        out["peak_corr"].append(float(corr[kpk, mpk]))
        out["resolved"].append(bool(abs(lags[kpk]) >= 1))
    return {k: np.asarray(v) for k, v in out.items()}


def ccf_feasibility(v_r, tau_c, size_r, dr_cell, offsets=None, verbose=True):
    """Can a time-delay cross-correlation measure `v_r` on this case at all?

    A TDE/CCF measures a propagation delay between two radii. For that to work
    the structure must travel **further than its own radial width** while
    staying correlated, because the offset has to satisfy two bounds at once:

        offset >  size_r      -- else the zero-lag correlation of the
                                 stationary radial SHAPE dominates and the
                                 peak pins at lag 0
        offset <  v_r * tau_c -- else the structure has decorrelated by the
                                 time it arrives and there is no peak to find

    so the method needs ``L_dec = v_r*tau_c > size_r``. That is a property of
    the FLOW, not of the analysis, and no choice of ``x_offsets`` fixes it.

    Measured on `from_archer_6kW/609-614`: `v_r = 90 m/s`, `tau_c = 2.09 us`,
    `size_r = 1.79 mm` give `L_dec = 0.188 mm = 0.68 cells` against a
    structure of **6.5 cells** — a ratio of **0.11**, i.e. the filament moves
    11 % of its own width before decorrelating. The CCF is inapplicable there
    and returned "all peaks at zero lag" at every offset tried, which is the
    correct answer rather than a tuning failure.

    That ratio is also physics: a structure that decorrelates in place instead
    of propagating is not a ballistically moving filament, which is consistent
    with the same case showing a packing fraction of 0.004 and only 3-6 % of
    the turbulent flux carried by detected structures.

    Returns ``dict`` with ``L_dec``, ``cells_dec``, ``ratio`` and ``feasible``.
    """
    L_dec = float(v_r) * float(tau_c)
    ratio = L_dec / float(size_r) if size_r > 0 else np.inf
    out = dict(L_dec=L_dec, cells_dec=L_dec / dr_cell if dr_cell else np.nan,
               size_r=float(size_r), ratio=ratio, feasible=bool(ratio > 1.0))
    if verbose:
        print("\n===== can a CCF measure v_r here? =====")
        print(f"  v_r assumed               : {v_r:8.1f} m/s")
        print(f"  tau_c                     : {tau_c*1e6:8.2f} us")
        print(f"  radial decorrelation len  : {L_dec*1e3:8.3f} mm "
              f"= {out['cells_dec']:.2f} cells")
        print(f"  radial structure size     : {size_r*1e3:8.2f} mm "
              f"= {size_r/dr_cell if dr_cell else np.nan:.1f} cells")
        print(f"  L_dec / size_r            : {ratio:8.2f}   "
              f"(needs > 1 for a CCF)")
        if not out["feasible"]:
            print(f"  VERDICT: CCF NOT APPLICABLE — the structure travels "
                  f"{100*ratio:.0f} % of its own\n           width before "
                  f"decorrelating, so no offset satisfies both\n           "
                  f"offset > size_r and offset < L_dec. Zero-lag peaks are the"
                  f"\n           correct result, not a tuning problem. Use "
                  f"tracking and the\n           conditional ExB instead.")
            if offsets is not None and dr_cell:
                print(f"           (offsets tried: {tuple(offsets)} cells = "
                      f"{tuple(round(o*dr_cell*1e3, 2) for o in offsets)} mm)")
        else:
            lo = int(np.ceil(size_r / dr_cell)) if dr_cell else 1
            hi = int(np.floor(L_dec / dr_cell)) if dr_cell else 1
            print(f"  VERDICT: feasible — use offsets between ~{lo} and "
                  f"~{hi} cells")
    return out


def conditional_exb_velocity(v_exb, f, threshold=2.5, x_slice=None,
                             weighted=False):
    """ExB radial velocity averaged over the cells that are inside a blob.

    Within the model this *is* the blob radial velocity — no identification, no
    tracking and no time baseline required — so it is the reference value that
    a tracked velocity should reproduce. If tracking gives something far
    smaller, the tracking is at fault.

    ``weighted=True`` weights by the fluctuation amplitude, giving the
    density-weighted transport velocity ``⟨δn·v⟩/⟨δn⟩`` rather than the plain
    conditional mean.

    Returns ``dict`` with the conditional mean ``v_mean``, its spread
    ``v_std``, the standard error ``v_sem`` (computed from the number of
    independent *structures*, not cells, since cells within one blob are
    correlated), the background all-cell mean ``v_all``, and the counts
    ``n_cells`` / ``n_structures``. Compare ``v_mean`` against ``v_sem``: if
    the two are comparable, the conditional signal is not significant and more
    frames are needed.
    """
    f = np.asarray(f, dtype=float)
    v = np.asarray(v_exb, dtype=float)
    xs = x_slice or slice(None)
    ff, vv = f[:, xs, :], v[:, xs, :]
    m = ff > threshold
    n = int(m.sum())
    if n == 0:
        # Same keys as the normal return, so callers can index the result
        # unconditionally (a frame with no structures is common).
        return dict(v_mean=np.nan, v_std=np.nan, v_sem=np.nan,
                    v_all=float(vv.mean()), n_cells=0, n_structures=0,
                    threshold=threshold)
    if weighted:
        w = ff[m]
        vm = float(np.average(vv[m], weights=w))
        vs = float(np.sqrt(np.average((vv[m] - vm) ** 2, weights=w)))
    else:
        vm = float(vv[m].mean())
        vs = float(vv[m].std())

    # Cells inside one blob are strongly correlated, so the naive std/sqrt(n)
    # badly understates the error. Count independent structures instead: label
    # the mask per frame and use the number of connected regions.
    nblob = 0
    for i in range(ff.shape[0]):
        if m[i].any():
            lab = _label_periodic(m[i])
            nblob += int(len(np.unique(lab)) - 1)
    sem = vs / np.sqrt(nblob) if nblob > 0 else np.nan

    return dict(v_mean=vm, v_std=vs, v_sem=float(sem), v_all=float(vv.mean()),
                n_cells=n, n_structures=int(nblob), threshold=threshold)


def turbulent_flux_slice(n, v_r, zdim_axis=2):
    """Turbulent radial particle flux ``⟨δn·δv_r⟩(x)`` on the analysis slice.

    Uses the package convention: the mean is the full ensemble average over
    ζ **and** t taken inside the product, so

        Γ_turb(x) = ⟨⟨n·v⟩_ζ⟩_t − ⟨⟨n⟩_ζ⟩_t·⟨⟨v⟩_ζ⟩_t

    This is the decisive physics check on a blob analysis: in a developed SOL
    the turbulent flux must be **outward (positive)** across the open field
    line region, because that is what carries particles from the separatrix to
    the wall. A negative or sign-changing profile means either the turbulence
    is not developed, the sample is too short for the average to converge, or
    the detected structures are not transporting blobs. Check it before
    interpreting any conditional velocity.

    Returns the profile in ``[m^-2 s^-1]``, one value per x index.
    """
    n = np.asarray(n, dtype=float)
    v = np.asarray(v_r, dtype=float)
    ax = (0, zdim_axis)
    return (n * v).mean(axis=ax) - n.mean(axis=ax) * v.mean(axis=ax)


def blob_flux_closure(n, v_r, f, r, b, b_period, threshold=2.5, x_slice=None,
                      min_pixels=4, verbose=True):
    """Packing-fraction flux closure: do the detected blobs carry the flux?

    **This is the best end-to-end validation of a blob analysis available, and
    it has a published expected value.** Mancini et al., Plasma Phys. Control.
    Fusion **68**, 055029 (2026) (the TCV-X21/X23 GBS validation) estimates
    the turbulent flux twice and compares:

        eq (3)  Gamma_ExB = <n v_r> - <n><v_r>          (the fluctuating flux)
        eq (5)  Gamma_fil = f_p <n>_fil <v_r>_fil       (from detection alone)

    with ``f_p`` the **packing fraction** — mean blob cross-sectional area
    divided by the total SOL area. They find ``Gamma_fil/Gamma_ExB`` close to
    **unity at the outer midplane**, with ``f_p = (5.9 ± 0.2)e-2`` for
    TCV-X21/X23-low and ``(7.2 ± 0.3)e-2`` for X23-high. So a run whose ratio
    is far from 1 at the OMP has a detection/threshold/velocity problem, and
    the number to aim at is not invented here.

    Their stated failure modes are worth knowing before over-reading it: in
    the **divertor** the detection *overestimates* filament properties near
    the separatrix and badly *underestimates* in the far SOL (detection events
    become rare), and they conclude the fluctuating ExB flux is the more
    reliable of the two there. Use this closure at the midplane.

    Two variants are returned because eq (5) does not say whether ``<n>_fil``
    is the total density or the density excess inside a filament, and they
    differ by O(1) for ``dn/n ~ 1`` blobs: ``ratio_dn`` uses the excess
    ``n - <n>`` (the quantity that actually constitutes the *turbulent* flux)
    and ``ratio_n`` the total ``n``. Quote whichever you state.

    Parameters
    ----------
    n, v_r, f : ndarray ``(t,x,z)``
        Raw field, radial velocity, and the sigma-normalised fluctuation that
        detection runs on.
    r, b, b_period :
        Physical coordinates, as for :func:`detect_blobs`.

    Returns
    -------
    dict with ``f_p``, ``n_fil``, ``dn_fil``, ``v_fil``, ``gamma_fil``,
    ``gamma_fil_dn``, ``gamma_exb``, ``ratio_n``, ``ratio_dn`` and the
    turbulent fraction ``turb_fraction`` of eq (4).
    """
    n = np.asarray(n, dtype=float)
    v = np.asarray(v_r, dtype=float)
    f = np.asarray(f, dtype=float)
    xs = x_slice or slice(None)
    nn, vv, ff = n[:, xs, :], v[:, xs, :], f[:, xs, :]
    rr = np.asarray(r, dtype=float)[xs]

    m = ff > threshold
    nt = ff.shape[0]

    # --- packing fraction: mean blob area / SOL area, per frame then averaged
    dr = np.gradient(rr)
    db = b_period / ff.shape[-1]
    cell_area = dr[:, None] * db                      # (x, 1), broadcasts over z
    sol_area = float(np.sum(dr) * b_period)
    blob_area = np.array([float((cell_area * m[i]).sum()) for i in range(nt)])
    f_p = float(blob_area.mean() / sol_area)

    # --- conditional means inside filaments
    bg = nn.mean(axis=(0, 2))
    dn = nn - bg[None, :, None]
    if m.any():
        n_fil = float(nn[m].mean())
        dn_fil = float(dn[m].mean())
        v_fil = float(vv[m].mean())
    else:
        n_fil = dn_fil = v_fil = np.nan

    g_fil = f_p * n_fil * v_fil
    g_fil_dn = f_p * dn_fil * v_fil

    # --- eq (3), SOL-averaged, and eq (4)
    gt = (nn * vv).mean(axis=(0, 2)) - nn.mean(axis=(0, 2)) * vv.mean(axis=(0, 2))
    g_mean = nn.mean(axis=(0, 2)) * vv.mean(axis=(0, 2))
    g_exb = float(gt.mean())
    with np.errstate(divide="ignore", invalid="ignore"):
        turb_frac = float(np.nanmean(np.abs(gt) / (np.abs(g_mean) + np.abs(gt))))

    out = dict(f_p=f_p, n_fil=n_fil, dn_fil=dn_fil, v_fil=v_fil,
               gamma_fil=g_fil, gamma_fil_dn=g_fil_dn, gamma_exb=g_exb,
               ratio_n=g_fil / g_exb if g_exb else np.nan,
               ratio_dn=g_fil_dn / g_exb if g_exb else np.nan,
               turb_fraction=turb_frac, sol_area=sol_area, threshold=threshold)
    if verbose:
        print("\n===== blob flux closure (Mancini 2026 PPCF eqs 3-5) =====")
        print(f"  packing fraction f_p        : {f_p:8.4f}   "
              f"(TCV: 0.059 attached / 0.072 high-density)")
        print(f"  <n>_fil / <dn>_fil          : {n_fil:.3e} / {dn_fil:.3e} m^-3")
        print(f"  <v_r>_fil                   : {v_fil:8.1f} m/s")
        print(f"  Gamma_fil (eq 5, total n)   : {g_fil:+.3e} m^-2 s^-1")
        print(f"  Gamma_fil (eq 5, excess dn) : {g_fil_dn:+.3e}")
        print(f"  Gamma_ExB (eq 3, SOL mean)  : {g_exb:+.3e}")
        print(f"  RATIO fil/ExB               : {out['ratio_n']:8.3f} (total n) "
              f"/ {out['ratio_dn']:8.3f} (excess)")
        print(f"  turbulent fraction (eq 4)   : {turb_frac:8.3f}   "
              f"(TCV: ~0.30 attached, >0.50 detached)")
        if np.isfinite(out["ratio_dn"]) and not (0.3 < abs(out["ratio_dn"]) < 3):
            print("  WARNING: far from unity at the midplane. Mancini 2026 gets "
                  "~1 there, so this\n           points at the detection "
                  "threshold, the conditional velocity or the\n           "
                  "packing fraction rather than at the physics. (In the "
                  "DIVERTOR they find\n           detection overestimates near "
                  "the separatrix and underestimates in the\n           far "
                  "SOL, and prefer the fluctuating ExB flux -- so only trust "
                  "this at\n           the midplane.)")
        print("=" * 56 + "\n")
    return out


def conditional_average(f, x_trigger, threshold=2.5, window=8, z_slice=None):
    """Conditionally averaged blob shape in ``(lag, z)`` at one radius.

    Every threshold crossing at ``x_trigger`` is used as a trigger and the
    surrounding time window averaged. The result is the mean structure — the
    standard way to get a representative shape and lifetime when individual
    blobs are too irregular to characterise one at a time.

    Returns ``(lags, ca)`` where ``ca`` has shape ``(2*window+1, nz)``.
    """
    f = np.asarray(f, dtype=float)
    zs = z_slice or slice(None)
    sig = f[:, int(x_trigger), zs]
    nt, nz = sig.shape
    acc = np.zeros((2 * window + 1, nz))
    cnt = 0
    for iz in range(nz):
        s = sig[:, iz]
        above = s > threshold
        # rising edges only, so one blob contributes one trigger
        trig = np.nonzero(above[1:] & ~above[:-1])[0] + 1
        for it in trig:
            if it - window < 0 or it + window >= nt:
                continue
            acc += f[it - window: it + window + 1, int(x_trigger), zs]
            cnt += 1
    lags = np.arange(-window, window + 1)
    return lags, (acc / cnt if cnt else acc), cnt


def structure_size_report(blobs, r, b, b_period, verbose=True):
    """How well are the detected structures resolved by the grid?

    A centroid is only meaningful if the structure covers several cells. SOL
    grids are strongly anisotropic — a fine radial mesh near the separatrix
    combined with a coarse toroidal mesh — so filaments are routinely a dozen
    cells wide radially but only one or two cells wide binormally. When that
    happens the binormal centroid, the shape moments and the aspect ratio are
    all dominated by the discretisation, which is a large part of why blobs in
    a real run look ragged compared with a seeded single-blob simulation.

    ``blobs`` is the flat list of **per-frame detections** — one physical
    structure present in several frames contributes one entry per frame. That
    is the right sample for a size distribution; use :func:`track_blobs` when
    you need distinct objects.

    Returns a dict of medians; set ``verbose`` for the printed summary.
    """
    if not blobs:
        if verbose:
            print("\nno structures detected — lower the threshold or check "
                  "the background subtraction")
        return {}
    db = b_period / len(b)
    dr = np.gradient(r)

    size_r = np.array([bl.size_r for bl in blobs])
    size_b = np.array([bl.size_b for bl in blobs])
    dr_at = np.array([dr[int(np.argmin(np.abs(r - bl.r)))] for bl in blobs])
    cells_r = size_r / dr_at
    cells_b = size_b / db
    frac_1col = float((size_b < 0.25 * db).mean())
    rep = dict(n=len(blobs), frac_single_z_column=frac_1col,
               size_r=float(np.median(size_r)), size_b=float(np.median(size_b)),
               cells_r=float(np.median(cells_r)), cells_b=float(np.median(cells_b)),
               npix=float(np.median([bl.npix for bl in blobs])),
               dr_cell=float(np.median(dr_at)), db_cell=float(db))
    if verbose:
        print("\n===== structure size vs grid =====")
        print(f"  detections (all frames)   : {rep['n']}  "
              f"(per-frame regions, not distinct blobs)")
        print(f"  radial   size (median)    : {rep['size_r']*1e3:7.2f} mm "
              f"= {rep['cells_r']:.1f} cells   (cell {rep['dr_cell']*1e3:.3f} mm)")
        print(f"  binormal size (median)    : {rep['size_b']*1e3:7.2f} mm "
              f"= {rep['cells_b']:.1f} cells   (cell {rep['db_cell']*1e3:.2f} mm)")
        print(f"  cells per structure       : {rep['npix']:.0f}")
        print(f"  span a single z column    : {100*frac_1col:.0f} %  "
              f"(these return size_b = 0 exactly)")
        aspect = rep["db_cell"] / rep["dr_cell"] if rep["dr_cell"] else np.nan
        print(f"  perpendicular cell aspect : {aspect:.0f}:1 (binormal:radial)")
        worst = min(rep["cells_r"], rep["cells_b"])
        if worst < 2:
            print("  VERDICT: structures are at or below the grid scale in one "
                  "direction.\n           Centroids and shapes are dominated by "
                  "discretisation; treat\n           positions as indicative "
                  "only.")
            if rep["cells_b"] < 2:
                print("           The binormal extent is UNRESOLVED: a structure "
                      "spanning one z\n           column returns size_b ~ 0, so "
                      "the quoted binormal size is the cell\n           size, "
                      "not a measurement, and major/minor/tilt are meaningless.\n"
                      "           The 'streamer' look comes from this cell aspect "
                      "ratio, NOT from\n           needing field-aligned "
                      "coordinates (see to_field_aligned docs).")
        elif worst < 4:
            print("  VERDICT: marginally resolved (< 4 cells in one direction) "
                  "— shape\n           moments and the aspect ratio are not "
                  "reliable.")
        else:
            print("  VERDICT: adequately resolved.")
        print("==================================\n")
    return rep


# ---------------------------------------------------------------------------
# The diagnostic to run first
# ---------------------------------------------------------------------------

def parallel_half_length(ds, y_idx, x_idx=None):
    """Parallel distance [m] from one poloidal slice to each end of its flux tube.

    ``L_par`` is what sets the sheath-connection branch of the blob velocity
    (:func:`blob_velocity_regimes`), and it must be the distance from the
    measurement plane to the *sheath*, not the whole field line.

    In the BOUT++ double-null layout the two poloidal regions are each a
    complete field line: region A runs lower-inner target → inner core/SOL →
    upper-inner target, region B upper-outer target → outer core/SOL →
    lower-outer target. So for an open flux surface the sum of
    ``dy·√g_22`` over the region containing ``y_idx`` IS the full connection
    length, and the two partial sums either side of ``y_idx`` are the two
    one-way lengths. (``√g_22 = hthe·B/Bp`` is the parallel arc length per
    unit y — the identity ``connection_length.py`` verifies on these grids.)

    Returns ``(L_down, L_up, L_total)`` in metres, where "down"/"up" are
    decreasing/increasing y. Use ``min(L_down, L_up)`` for the sheath branch
    on an open field line; the result is meaningless on a *closed* surface
    (``x_idx < ixseps1``), where there is no sheath.
    """
    m = ds.metadata
    pdim = detect_poloidal_dim(ds)
    ny_full = ds.sizes[pdim]
    ny = int(m.get("ny", ny_full))
    MYG = int(m.get("MYG", 2))
    g = MYG if ny_full > ny else 0
    if x_idx is None:
        x_idx = int(m["ixseps1"]) + 2

    ny_inner = int(m.get("ny_inner", 0))
    if ny_inner and ny_inner < ny:            # double null: two regions
        split = ny_inner + 2 * g
        lo, hi = (0, split) if y_idx < split else (split, ny_full)
    else:                                      # single null: one region
        lo, hi = 0, ny_full

    dy = np.asarray(ds["dy"].isel({pdim: slice(lo, hi)}).values, dtype=float)
    g22 = np.asarray(ds["g_22"].isel({pdim: slice(lo, hi)}).values, dtype=float)
    if _is_unnormalised(ds):
        # g_22 IS unnormalised by xhermes (x rho_s0^2), dy is in radians, so
        # sqrt(g_22)*dy is already metres. Nothing to do -- but check the
        # other way round if a raw dataset is passed.
        pass
    else:
        g22 = g22 * float(m["rho_s0"]) ** 2
    dy = dy[..., int(x_idx)] if dy.ndim > 1 else dy
    g22 = g22[..., int(x_idx)] if g22.ndim > 1 else g22
    dl = dy * np.sqrt(g22)
    # guard rows carry no physical length
    if g:
        dl = dl.copy()
        dl[:g] = 0.0
        dl[-g:] = 0.0
    j = int(y_idx) - lo
    return float(dl[:j].sum()), float(dl[j:].sum()), float(dl.sum())


def blob_velocity_regimes(Te_eV, L_par, R, delta, B=None, Ti_eV=None,
                          AA=2.0, ne=1e19, Lambda=None, eps_x=0.3,
                          f_x_max=None, coulomb_log=12.0, verbose=True):
    """Myra two-region blob velocity: all FOUR regimes, with the Λ axis.

    Implements the regime diagram of Myra, Russell & D'Ippolito, Phys.
    Plasmas **13**, 112502 (2006), in the form tabulated by Tsui et al.,
    "Filamentary velocity scaling validation in the TCV tokamak", Phys.
    Plasmas **25**, 072506 (2018), eqs. (1.1)-(1.9) — both on disk under
    ``Fusion/Filament_blob/``. Normalisations::

        a_hat = a_b · R^(1/5) / (L_par^(2/5) ρ_s^(4/5))      (Tsui 1.2)
        Theta = a_hat^(5/2)
        v_hat = v_r / [ c_s (2 L_par ρ_s² / R³)^(1/5) ]      (Tsui 1.1)
        Lambda = ∫ ν_ei/(Ω_e ρ_s) ds_par  ≈  L_par ν_ei/(Ω_e ρ_s)  (1.3/1.4)

    and the four branches::

        Cs  sheath-connected          v_hat ~ 1/a_hat²        (Λ < 1, large a)
        Ci  connected ideal-interch.  v_hat ~ eps_x·a_hat^½
        RX  resistive X-point         v_hat ~ Λ/a_hat²    <- only Λ-dependent
        RB  resistive ballooning      v_hat ~ a_hat^½         (Λ > Theta)

    **`v_hat` is an UPPER ENVELOPE, not a central value.** Tsui's thousands
    of TCV samples "are mainly scattered between zero and a maximum velocity
    which varies as a function of size and collisionality", the scatter being
    "consistent with mechanisms that tend to slow the velocity of individual
    filaments" (background density, dipole mixing between neighbouring
    filaments, monopole rotation of hot filaments, magnetic-shear compression
    of the return-current path). So a measured velocity *below* the branch is
    expected; the quantitative comparison is **how far below, at matched
    `a_hat`**, against TCV's factor 1.6-6.4. Measured there: `v_r` clustered
    **0.5-2 km/s** (minority to +5 / −4 km/s), `a_b` **3-11 mm**, `a_hat`
    **0.3-1.1**, "most filaments populate the resistive-ballooning regime".

    ``eps_x`` is the field-line fanning parameter, ``~1/f_x_max`` for
    ``f_x_max > 1`` (Tsui eq. 1.5); 0.3 is their quoted typical diverted
    value and is used only on the **Ci** branch. Pass ``f_x_max`` to derive
    it instead — relevant for a Super-X, where `f_x` reaches 5-14.

    ``Lambda`` is computed from ``ne``/``Te_eV``/``B``/``L_par`` when not
    given, with ``ν_ei = 2.91e-12 n_e[m⁻³] lnΛ_c T_e[eV]^(-3/2)`` (NRL).
    That is Tsui's ``Λ_mid``: the full connection length with *upstream*
    parameters. Their ``Λ_div`` uses the X-point-to-target length with target
    parameters and "can dominate the entire system", so a divertor-relevant
    answer needs that variant passed in explicitly.

    **Scope**: isothermal, electrostatic, no ion-kinetic or strong-gradient
    effects, and Tsui stresses the scalings "utilize asymptotic expressions
    and were developed by approximating gradients as inverse scale lengths",
    so "the scalings and the normalized quantities have uncertainties greater
    than order unity". Treat a factor ~2 as agreement. Note also that Tsui's
    ``v*`` carries a 2 that Myra's eq. (28) does not — a 2^(1/5) = 1.15
    convention difference, far inside that bound; this follows Tsui.

    Returns a dict with ``c_s``, ``rho_s``, ``nu_ei``, ``Lambda``,
    ``a_star``, ``v_star``, ``a_hat``, ``Theta``, per-branch ``v_Cs``,
    ``v_Ci``, ``v_RX``, ``v_RB``, the selected ``regime`` and ``v_envelope``.
    """
    qe, Mp, me, eps0 = 1.602176634e-19, 1.67262192369e-27, 9.1093837015e-31, 8.8541878128e-12
    mi = AA * Mp
    T = float(Te_eV) + (float(Ti_eV) if Ti_eV is not None else 0.0)
    c_s = np.sqrt(qe * T / mi)
    if B is None:
        raise ValueError("B [T] is required")
    B = float(B)
    rho_s = c_s * mi / (qe * B)

    # NRL electron-ion collision frequency [s^-1]
    nu_ei = 2.91e-12 * float(ne) * float(coulomb_log) * float(Te_eV) ** -1.5
    Omega_e = qe * B / me
    if Lambda is None:
        Lambda = float(L_par) * nu_ei / (Omega_e * rho_s)
    Lambda = float(Lambda)
    if f_x_max is not None and float(f_x_max) > 1.0:
        eps_x = 1.0 / float(f_x_max)
    eps_x = float(eps_x)

    delta = np.atleast_1d(np.asarray(delta, dtype=float))
    a_star = float(L_par) ** 0.4 * rho_s ** 0.8 / R ** 0.2
    v_star = c_s * (2.0 * float(L_par) * rho_s ** 2 / R ** 3) ** 0.2
    a_hat = delta / a_star
    Theta = a_hat ** 2.5

    vh_Cs = 1.0 / a_hat ** 2
    vh_Ci = eps_x * np.sqrt(a_hat)
    vh_RX = Lambda / a_hat ** 2
    vh_RB = np.sqrt(a_hat)

    # Regime selection on (Lambda, Theta). NOT min() over branches -- which
    # branch applies is set by where the return current closes, and a minimum
    # silently picks Cs for a_hat > 1 even when the filament has disconnected
    # from the sheath (Paruta et al 2019, PoP 26, 032302: their diverted GBS
    # blobs are RB and RX, and "the sheath connected scaling significantly
    # underestimates the blob velocity").
    #
    # FIXED 2026-10-10: the first version tested `Lambda > Theta` FIRST and so
    # returned RB for ANY small blob, because Theta = a_hat^(5/2) -> 0. That is
    # backwards: RB is the HIGH-collisionality, divertor-DISCONNECTED regime
    # ("when the collisionality is sufficiently high the filaments are not
    # electrically connected to the divertor region", Tsui 2018). A low-Lambda
    # case is sheath-connected by construction, and the user confirmed
    # independently that the large pure-plasma SXD run "should be in the
    # sheath limited region" -- which the selector was contradicting.
    #
    # Low Lambda (< 1) = connected to the target, so the Cs/Ci pair applies and
    # the split is by SIZE: the sheath current wins for large blobs, the ion
    # polarisation current for small ones, crossing where
    # 1/a_hat^2 = eps_x a_hat^(1/2), i.e. Theta = 1/eps_x (Tsui's
    # "a_hat > 1/eps_x^(2/5) = 1.6, equivalent to Theta > 1/eps_x = 3.3").
    # High Lambda (>= 1) = the resistive family, split by Lambda vs Theta.
    regime, v_env = [], []
    for ah, th, c, i, x, b in zip(a_hat, Theta, vh_Cs, vh_Ci, vh_RX, vh_RB):
        if Lambda < 1.0:                           # sheath-connected family
            if th > 1.0 / eps_x:
                regime.append("Cs"); v_env.append(c)
            else:
                regime.append("Ci"); v_env.append(i)
        else:                                      # resistive family
            if Lambda > th:
                regime.append("RB"); v_env.append(b)
            else:
                regime.append("RX"); v_env.append(x)
    v_env = np.asarray(v_env) * v_star

    out = dict(c_s=c_s, rho_s=rho_s, nu_ei=nu_ei, Lambda=Lambda,
               a_star=a_star, v_star=v_star, delta=delta, a_hat=a_hat,
               Theta=Theta, eps_x=eps_x,
               v_Cs=vh_Cs * v_star, v_Ci=vh_Ci * v_star,
               v_RX=vh_RX * v_star, v_RB=vh_RB * v_star,
               regime=regime, v_envelope=v_env)
    if verbose:
        print("\n===== blob velocity regimes (Myra 2006 / Tsui 2018) =====")
        print(f"  c_s (T={T:.1f} eV, AA={AA})    : {c_s/1e3:8.2f} km/s")
        print(f"  rho_s (B={B:.3f} T)            : {rho_s*1e3:8.3f} mm")
        print(f"  nu_ei (ne={ne:.2e}, lnL={coulomb_log:.0f}) : {nu_ei:.3e} s^-1")
        print(f"  Lambda (Tsui 1.4, mid)         : {Lambda:8.4f}")
        print(f"  eps_x (fanning)                : {eps_x:8.3f}")
        print(f"  a* (Tsui 1.2)                  : {a_star*1e3:8.2f} mm "
              f"= {a_star/rho_s:.1f} rho_s")
        print(f"  v* (Tsui 1.1)                  : {v_star/1e3:8.2f} km/s")
        print(f"  {'a_b [mm]':>9s}{'a_hat':>8s}{'Theta':>9s}"
              f"{'v_RB':>9s}{'v_Ci':>9s}{'v_RX':>10s}{'v_Cs':>10s}"
              f"{'ENVELOPE':>10s}  regime")
        for k in range(delta.size):
            print(f"  {delta[k]*1e3:9.2f}{a_hat[k]:8.3f}{Theta[k]:9.4f}"
                  f"{out['v_RB'][k]:9.0f}{out['v_Ci'][k]:9.0f}"
                  f"{out['v_RX'][k]:10.0f}{out['v_Cs'][k]:10.0f}"
                  f"{v_env[k]:10.0f}  {regime[k]}")
        print("  NB the envelope is an UPPER BOUND -- Tsui's TCV filaments")
        print("     scatter between ~0 and it, landing a factor 1.6-6.4 below")
        print("     (measured 0.5-2 km/s, a_hat 0.3-1.1, mostly RB).")
        print("=" * 56 + "\n")
    return out


def resolvability_report(t, r, ixseps1, v_expected, f=None, x_slice=None,
                         omega_ci=None, frames_target=30, verbose=True):
    """Can this time series measure a blob velocity at all?

    Blob tracking fails silently when the frame-to-frame displacement is a
    small fraction of a grid cell: the centroid is then dominated by shape
    changes and threshold quantisation, and the fitted velocity comes out far
    too small. This function turns that into an explicit number *before* any
    tracking is attempted.

    It reports the displacement per frame in cells, the displacement over the
    whole series, and how many frames would be needed for a blob to cross a
    few cells. When ``f`` (the flattened fluctuation field) is supplied it also
    compares the observed peak amplitude against the largest value expected
    from Gaussian noise of the same sample size, ``√(2 ln N)`` — if the
    observed peak is not clearly above that, the "blobs" are not statistically
    distinguishable from background fluctuations and no threshold will separate
    them.

    Every quantity that carries the verdict is **dimensionless** — a
    displacement in grid cells, or a number of frames per SOL transit. Those
    ratios cancel ``Omega_ci``, ``Cs0`` and ``rho_s0`` exactly (since
    ``rho_s0 = Cs0/Omega_ci``), so the verdict is unaffected even if the
    normalisation constants were wrong; only the m/s and µs values quoted
    alongside them would move. Use :func:`hermes3_turb.load.check_normalisation`
    to confirm those separately.

    They are also insensitive to grid resolution in the helpful direction: a
    finer radial mesh makes the displacement a *larger* number of cells, so a
    higher-resolution run is easier to track, not harder. What does not improve
    with resolution is ``frames_per_sol_crossing`` — that is set by the dump
    cadence against the blob transit time.

    Returns a dict of the same quantities.
    """
    t = np.asarray(t, dtype=float)
    dt = float(np.median(np.diff(t))) if t.size > 1 else np.nan
    span = float(t[-1] - t[0]) if t.size > 1 else 0.0
    dr_sep = float(np.gradient(r)[int(ixseps1)])

    per_frame = v_expected * dt
    per_series = v_expected * span
    frames_per_cell = dr_sep / per_frame if per_frame > 0 else np.inf

    xs = x_slice or slice(int(ixseps1), None)
    dr_all = np.gradient(r)[xs]
    dr_med = float(np.median(dr_all))
    sol_width = float(np.sum(dr_all))
    frames_cross = sol_width / per_frame if per_frame > 0 else np.inf

    # What dt_dump would put ~`frames_target` frames across one SOL crossing?
    dt_want = (sol_width / v_expected / frames_target) if v_expected > 0 else np.nan

    rep = dict(dt=dt, n_frames=int(t.size), span=span, dr_cell=dr_sep,
               dt_recommended=dt_want,
               dt_recommended_norm=(dt_want * omega_ci
                                    if omega_ci else np.nan),
               dr_cell_median=dr_med, sol_width=sol_width,
               cells_per_frame_median=per_frame / dr_med if dr_med else np.nan,
               frames_per_sol_crossing=frames_cross,
               folder_fraction_of_transit=(t.size / frames_cross
                                           if np.isfinite(frames_cross) else np.nan),
               v_expected=v_expected, disp_per_frame=per_frame,
               cells_per_frame=per_frame / dr_sep if dr_sep else np.nan,
               disp_per_series=per_series,
               cells_per_series=per_series / dr_sep if dr_sep else np.nan,
               frames_per_cell=frames_per_cell,
               frames_for_5_cells=5 * frames_per_cell)

    if f is not None:
        f = np.asarray(f)
        nsamp = f.shape[0] * f.shape[-1]
        rep["n_samples_per_x"] = int(nsamp)
        rep["gaussian_max"] = float(np.sqrt(2.0 * np.log(max(nsamp, 2))))
        rep["observed_max"] = float(np.nanmax(f))

        # ---- decorrelation time and the EFFECTIVE sample size -------------
        # A fine dump cadence buys tracking but NOT statistics: frames closer
        # together than tau_c are the same realisation. The independent sample
        # count is min(nt, span/(2 tau_c)), and it is what sets the error bar
        # on every same-time quantity (conditional velocity, turbulent flux,
        # spectra) -- none of which care about dt at all. So the two
        # requirements are different and a single dt_dump trades them off:
        #   tracking  wants  dt <~ tau_c
        #   statistics want  span >> tau_c, i.e. MANY frames however spaced
        # Measured on sxd_MU/from_archer_6kW: tau_c = 2.1 us, so folder 396
        # (11 frames over 10.4 us) holds only ~2.5 independent samples and its
        # error bars were understated ~2x, while dirs 403-408 (8 frames at
        # 52 us) are 8 FULLY independent samples.
        if f.ndim == 3 and f.shape[0] > 2:
            # Sample at the SEPARATRIX, not the middle of the array. `f` is
            # already x-sliced by the caller, so locate ixseps1 inside it.
            # Measured on from_archer_6kW/609-614 (60 frames, 1.044 us), the
            # integral time of v_r runs 1.60 us at the separatrix to 3.48 us
            # in the far SOL, and dn is 2-3x slower again (3.79 -> 5.42 us).
            # `(sliced nx)//2` landed at x~153 of 244 -- far SOL -- and
            # returned "never decorrelates in this window", i.e. N_eff = 1 of
            # 60, when the separatrix value is N_eff = 19.
            _x0 = getattr(x_slice, "start", None) if x_slice is not None else None
            _ic = int(ixseps1) - int(_x0) if _x0 is not None else 0
            _ic = int(np.clip(_ic, 0, f.shape[1] - 1))
            a = f[:, _ic, :]
            a = a - a.mean(axis=0, keepdims=True)
            ntf = a.shape[0]
            ac = []
            for L in range(1, min(ntf - 1, 6)):
                d0, d1 = a[: ntf - L], a[L:]
                den = np.sqrt((d0 ** 2).sum() * (d1 ** 2).sum())
                ac.append(float((d0 * d1).sum() / den) if den > 0 else 0.0)
            ac = np.asarray(ac)
            below = np.flatnonzero(ac < np.exp(-1.0))
            # lag index 1 == one dt; interpolate nothing, report the crossing
            tau_c = float((below[0] + 1) * dt) if below.size else np.nan
            rep["tau_c"] = tau_c
            rep["autocorr"] = ac
            # The error bar on a mean needs the INTEGRAL time, not the 1/e
            # crossing: Var(mean) = sigma^2 * 2*tau_int/T. Sum only up to the
            # first zero crossing -- beyond it a mean-subtracted series of
            # length N is forced towards sum(rho) = -1/2, so the tail is
            # finite-N bias rather than signal. Using the 1/e time instead
            # overstates tau (2.09 vs 1.60 us here) and so understates N_eff.
            _zc = np.flatnonzero(ac <= 0.0)
            _cut = _zc[0] if _zc.size else ac.size
            tau_int = float(dt * (0.5 + ac[:_cut].sum()))
            rep["tau_int"] = tau_int
            if tau_int > 0:
                rep["n_independent"] = float(min(t.size, span / (2.0 * tau_int)))
            else:
                rep["n_independent"] = float(t.size)

    if verbose:
        print("\n===== blob velocity resolvability =====")
        print(f"  frames                    : {rep['n_frames']}")
        print(f"  dt                        : {dt*1e6:.4f} us")
        print(f"  time baseline             : {span*1e6:.3f} us")
        print(f"  radial cell at separatrix : {dr_sep*1e3:.4f} mm")
        print(f"  assumed blob speed        : {v_expected:.1f} m/s")
        print(f"  displacement per frame    : {per_frame*1e3:.5f} mm "
              f"= {rep['cells_per_frame']:.4f} cells")
        print(f"  displacement over series  : {per_series*1e3:.3f} mm "
              f"= {rep['cells_per_series']:.2f} cells")
        if frames_per_cell >= 1:
            print(f"  frames to cross one cell  : {frames_per_cell:.1f}")
        else:
            print(f"  cells crossed per frame   : "
                  f"{rep['cells_per_frame']:.1f}  (more than one cell per dump)")
        print(f"  -- normalisation-independent ratios "
              f"(unchanged by Omega_ci / rho_s0) --")
        print(f"  cells/frame, median cell  : "
              f"{rep['cells_per_frame_median']:.4f}  "
              f"(median cell {dr_med*1e3:.3f} mm)")
        print(f"  frames per SOL crossing   : {frames_cross:.0f} "
              f"(SOL {sol_width*1e3:.1f} mm)")
        _tr = rep["folder_fraction_of_transit"]
        if _tr < 1:
            print(f"  this series covers        : {100*_tr:.2f} % of one blob "
                  f"transit")
        else:
            print(f"  this series covers        : {_tr:.1f} blob transits "
                  f"(sampled every {frames_cross/1:.1f} frames)")
        if "gaussian_max" in rep:
            print(f"  peak |f| observed         : {rep['observed_max']:.2f} sigma")
            print(f"  Gaussian expectation      : {rep['gaussian_max']:.2f} sigma "
                  f"(N={rep['n_samples_per_x']} samples/x)")
        if "n_independent" in rep:
            _tc, _ni = rep["tau_c"], rep["n_independent"]
            print(f"  -- STATISTICS (independent of dt; sets every error bar) --")
            _ti = rep.get("tau_int", np.nan)
            if np.isfinite(_tc):
                print(f"  tau_c (1/e crossing)      : {_tc*1e6:.2f} us")
            else:
                print(f"  tau_c (1/e crossing)      : > {span*1e6:.1f} us "
                      f"(not reached in this window)")
            print(f"  tau_int (sets the error)  : {_ti*1e6:.2f} us "
                  f"at the separatrix")
            print(f"  NB tau_c varies with radius and field: v_r is ~2x faster "
                  f"than dn,\n     and both slow by ~2x into the far SOL.")
            print(f"  INDEPENDENT samples       : {_ni:.1f} of {t.size} frames "
                  f"= min(nt, span/2tau_c)")
            if _ni < 0.5 * t.size:
                print(f"  NOTE: frames are CORRELATED -- the error bar on a "
                      f"conditional velocity,\n        turbulent flux or "
                      f"spectrum scales as 1/sqrt({_ni:.1f}), not "
                      f"1/sqrt({t.size}).\n        A finer dt_dump does NOT "
                      f"help here; a LONGER span does. These same-time\n"
                      f"        quantities do not care about dt at all.")
            if _ni < 50 and np.isfinite(_ti) and _ti > 0:
                for _tgt in (20, 50, 80):
                    _need = 2.0 * _ti * _tgt
                    print(f"  for N_eff = {_tgt:<3d} need span "
                          f"{_need*1e6:7.1f} us "
                          f"({_need/max(span,1e-30):.1f}x the present window)")
                print(f"  frames closer than 2*tau_int = {2*_ti*1e6:.1f} us add "
                      f"TRACKING, not error bars")
        # The dump cadence can fail in BOTH directions, and the cures are
        # opposite, so the two cases are reported separately.
        if frames_cross < 10:
            print("  VERDICT: SAMPLING TOO COARSE for tracking — a blob crosses "
                  "the whole SOL\n           in "
                  f"{frames_cross:.1f} frames, moving "
                  f"{rep['cells_per_frame_median']:.0f} cells between dumps. "
                  "Frame-to-frame\n           linking connects unrelated "
                  "structures, so tracked velocities are\n           meaningless "
                  "and take either sign. Adding folders does NOT help at\n"
                  "           this cadence — you need a SMALLER dt_dump (aim for "
                  "~20-50 frames\n           per SOL crossing). Use the "
                  "conditional ExB value instead.")
            _rec = (f"  SUGGESTION: set dt_dump ~ {dt_want*1e6:.2f} us "
                    f"(currently {dt*1e6:.2f} us, a factor "
                    f"{dt/dt_want:.0f} too coarse)")
            if omega_ci:
                _rec += (f"\n              = {dt_want*omega_ci:.3g} in BOUT.inp "
                         f"normalised time units")
            print(_rec)
            print("              Only the TRACKING needs this. "
                  "conditional_exb_velocity and\n              "
                  "turbulent_flux_slice are same-time correlations and are "
                  "already\n              valid at the present cadence.")
        elif rep["cells_per_series"] < 1:
            print("  VERDICT: NOT MEASURABLE by tracking — a blob moves less than "
                  "one radial\n           cell over the entire series. Add more "
                  "folders; you need about\n           "
                  f"{int(np.ceil(rep['frames_for_5_cells']))} frames for a 5-cell "
                  "displacement.")
        elif rep["cells_per_series"] < 5:
            print("  VERDICT: MARGINAL — tracked velocities will be noisy. Prefer "
                  "the ExB\n           conditional average, and add folders.")
            if np.isfinite(dt_want):
                print(f"  SUGGESTION: dt_dump ~ {dt_want*1e6:.3g} us would give "
                      f"~{frames_target} frames per\n              SOL crossing "
                      f"(currently {dt*1e6:.3g} us).")
        else:
            print("  VERDICT: OK for tracking.")
        if "gaussian_max" in rep and rep["observed_max"] < rep["gaussian_max"] + 0.5:
            print("  WARNING: peak amplitude is not above the Gaussian-noise "
                  "expectation.\n           Structures are not statistically "
                  "distinguishable from background;\n           lower the "
                  "threshold only with that caveat, or use more frames.")
        print("=======================================\n")
    return rep

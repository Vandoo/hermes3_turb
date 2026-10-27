"""
Utility helpers for working with Hermes-3 / BOUT++ datasets.

This module does NOT wrap xhermes or xbout loaders. Load data directly:

    import xhermes
    ds = xhermes.open(path, gridfilepath=gridfile, geometry="toroidal",
                      keep_yboundaries=True)

    # or without xhermes:
    from xbout import open_boutdataset
    ds = open_boutdataset(path+"BOUT.dmp.*.nc", inputfilepath=path+"BOUT.settings",
                          gridfilepath=gridfile, geometry="toroidal",
                          keep_yboundaries=True)

After loading, ds has dimensions (t, x, theta, zeta) with metric components
(J, g11, g_22, g_33, g_23, g_12, g_13, Bxy, Bpxy, dx, dy, dz) available.

Hermes-3 variable naming (after xhermes unnormalisation to SI):
  Ne, Pe, NVe  : electron density [m⁻³], pressure [Pa], momentum [kg/m²/s]
  Pi, NVi      : ion pressure [Pa], momentum [kg/m²/s]
  phi          : electrostatic potential [V]
  Te, Ti       : temperatures [eV]   (requires diagnose = true in BOUT.inp)
  Ve, Vi       : parallel velocities [m/s]  (requires diagnose = true)
"""


def _ensure_z_single_chunk(da, zdim=None):
    """Rechunk ``da`` so the toroidal dim has a single chunk; no-op if not
    dask-backed or already single-chunk.

    xBOUT's ``.bout.ddy()`` → ``to_field_aligned()`` does an FFT along the
    toroidal axis (``_shift_z``). Dask's FFT can't span multiple chunks
    along the transform axis, so a multi-chunked ζ raises::

        ValueError: Dask array only supports taking an FFT along an axis
        that has a single chunk.

    Call this on any field passed to ``.bout.ddy()``.
    """
    if not hasattr(da, "chunks") or da.chunks is None:
        return da
    if zdim is None:
        zdim = detect_toroidal_dim(da)
    chunks_by_dim = dict(zip(da.dims, da.chunks))
    if zdim in chunks_by_dim and len(chunks_by_dim[zdim]) > 1:
        return da.chunk({zdim: -1})
    return da


def _no_t(da):
    """
    Drop a phantom ``t`` axis if present.

    BOUT++ writes the metric tensor (J, g_ij, dx, dy, dz) into every dump file,
    so xBOUT/xhermes sometimes leaves a ``t`` dimension on these arrays even
    though they are physically time-independent. Multiplying a 2-D field
    against such a metric silently broadcasts back to 3-D and `.sum()` over the
    result then sums over time too, inflating volume / surface integrals by
    exactly ``ds.sizes['t']``. Apply this helper at every metric access to be
    robust against either case.

    Returns the input unchanged when no ``t`` axis is present.
    """
    if "t" in getattr(da, "dims", ()):
        return da.isel(t=0)
    return da


def load_curvature_vector(gridfile, ds=None, pdim="theta", normalise=None):
    """
    Full contravariant curvature vector ``Curlb_B = ∇×(b/B)``, all three
    components, reproducing Hermes-3's own assembly.

    Hermes-3 builds it in ``braginskii_ion_viscosity.cxx:82–112``::

        mesh->get(Curlb_B, "bxcv")           # contravariant, from the grid
        if paralleltransform == "shifted":
            mesh->get(I, "sinty")
            Curlb_B.z += I * Curlb_B.x       # shifted-metric twist, on RAW values
        Curlb_B.x /= Bnorm
        Curlb_B.y *= SQ(Lnorm);  Curlb_B.z *= SQ(Lnorm)
        Curlb_B *= 2. / coord->Bxy           # Bxy here is NORMALISED

    so the returned components are ``2·bxcv_i/B`` with the ``2/B`` already
    applied — the caller does not reapply it.

    **The twist is only as good as the grid.** Hermes-3 calls
    ``mesh->get(I, "sinty")`` without checking the return value, so on a grid
    that has no ``sinty`` field BOUT++ leaves ``I = 0`` (with a log warning)
    and the run gets **no twist at all**. This function mirrors that exactly:
    it applies the twist only if ``sinty`` is in the grid file, and reports
    which happened via the returned ``'twist'`` key — so the result always
    matches what the simulation actually used, rather than what it arguably
    should have used. Hypnotoad grids for these MAST-U cases carry ``zShift``
    and ``ShiftAngle`` but **not** ``sinty``, so the twist is inoperative
    there. (``sinty`` is the integrated shear ``∂zShift/∂x``; deriving it from
    ``zShift`` would NOT reproduce the run.)

    Always read from the **grid file**, never from the dump: BOUT++ passes
    ``bxcv*`` through to the dump as the mesh read them, which is not
    necessarily the normalisation the component ends up using, and ``bxcvz``
    is frequently absent from the dump altogether.

    Returns
    -------
    dict
        ``{'x', 'y', 'z'}`` as xarray.DataArray on dims ``(x, pdim)``,
        plus ``'Bxy'`` and the bool ``'twist'``.
    """
    import netCDF4 as nc
    import xarray as xr
    import numpy as np

    with nc.Dataset(gridfile) as g:
        raw = {c: np.array(g[f"bxcv{c}"][:], dtype=float) for c in "xyz"}
        Bxy = np.array(g["Bxy"][:], dtype=float)
        twist = "sinty" in g.variables
        if twist:
            raw["z"] = raw["z"] + np.array(g["sinty"][:], dtype=float) * raw["x"]

    if normalise is None:
        normalise = (ds is not None and "rho_s0" in ds.metadata
                     and "Bnorm" in ds.metadata)
    if normalise:
        if ds is None:
            raise ValueError("load_curvature_vector(normalise=True) requires "
                             "`ds` for rho_s0 / Bnorm")
        rho_s0 = float(ds.metadata["rho_s0"])
        Bnorm = float(ds.metadata["Bnorm"])
        raw["x"] = raw["x"] / Bnorm
        raw["y"] = raw["y"] * rho_s0 ** 2
        raw["z"] = raw["z"] * rho_s0 ** 2
        Bxy = Bxy / Bnorm

    out = {c: xr.DataArray(2.0 * raw[c] / Bxy, dims=["x", pdim]) for c in "xyz"}
    out["Bxy"] = xr.DataArray(Bxy, dims=["x", pdim])
    out["twist"] = twist
    return out


def load_grid_vars(gridfile, ds=None, pdim="theta", normalise=None):
    """
    Read geometric variables (bxcvx, bxcvy, Bxy) from the BOUT++ grid file
    and, by default, rescale them into the dataset's BOUT++-normalised units.

    The Hypnotoad grid file stores these in physical SI:
        bxcvx [T]     — B/2·Curl(b/B)·Grad(ψ)
        bxcvy [m⁻²]   — B/2·Curl(b/B)·Grad(y)
        Bxy   [T]
    A BOUT++ dump opened with ``xhermes(unnormalise=False)`` (or
    ``open_stormdataset(unnormalise=False)``) has these in the BOUT++
    normalisation, related to physical by the same factors xstorm applies
    at load time (xstorm/load.py:783–803):
        bxcvx_norm = bxcvx_phys / B_0
        bxcvy_norm = bxcvy_phys · rho_s0²
        Bxy_norm   = Bxy_phys   / B_0

    The flux/drift formulas in :mod:`hermes3_turb.fluxes` combine these with
    metric components ``g_ij``, ``J`` taken from the dataset, so the two must
    be in the same normalisation. Reading bxcvy raw from the grid (physical
    m⁻²) and combining it with normalised metric over-counts the poloidal
    magnetic-drift by 1/rho_s0² ≈ 10⁶ for rho_s0 ~ mm — the cause of an early
    bug in this package, see [[bug-poloidal-magdrift]].

    Parameters
    ----------
    gridfile : str
        Path to Hypnotoad grid file (.grd.nc).
    ds : xarray.Dataset, optional
        Dataset providing ``rho_s0`` / ``Bnorm`` via ``ds.metadata``. Used to
        rescale the raw grid values to match the dataset's normalisation
        when ``normalise=True`` (or auto).
    pdim : str
        Name of the poloidal dimension in the dataset ('theta' or 'y').
    normalise : bool or None
        If True: apply the xstorm-style BOUT++ normalisation
        (bxcvx /= Bnorm, bxcvy *= rho_s0², Bxy /= Bnorm) — correct for data
        loaded with ``unnormalise=False``.
        If False: return raw physical SI values — correct for data already
        unnormalised (``unnormalise=True``) or for direct comparison with
        the grid file.
        If None (default): apply normalisation when ``ds`` is given and
        carries ``rho_s0`` / ``Bnorm``; otherwise return raw.

    Returns
    -------
    dict of xarray.DataArray
        Keys: 'bxcvx', 'bxcvy', 'Bxy'.
    """
    import netCDF4 as nc
    import xarray as xr
    import numpy as np

    # First preference: read from `ds` if all three are present there. This
    # path keeps the y-dim consistent with the rest of the metric (g_ij, J,
    # etc.) — important when the dump retains y-guards but the grid file
    # doesn't, in which case mixing grid-file arrays (y = ny_grid) with the
    # dump's metric (y = ny_grid + 4·MYG in DN) raises an AlignmentError in
    # downstream multiplications. Hermes-3 writes bxcvx/bxcvy/Bxy into the
    # dump alongside the metric, with the guards already extrapolated, so
    # the sizes match and the normalisation is already consistent with the
    # dataset's units (no further rescaling needed).
    if (ds is not None
            and all(v in ds.data_vars for v in ("bxcvx", "bxcvy", "Bxy"))):
        return {v: _no_t(ds[v]) for v in ("bxcvx", "bxcvy", "Bxy")}

    out = {}
    with nc.Dataset(gridfile) as g:
        for var in ("bxcvx", "bxcvy", "Bxy"):
            arr = np.array(g[var][:])
            out[var] = xr.DataArray(arr, dims=["x", pdim])

    if normalise is None:
        normalise = ds is not None and "rho_s0" in ds.metadata and "Bnorm" in ds.metadata

    if normalise:
        if ds is None:
            raise ValueError("load_grid_vars(normalise=True) requires `ds` "
                             "for rho_s0 / Bnorm")
        rho_s0 = float(ds.metadata["rho_s0"])
        Bnorm  = float(ds.metadata["Bnorm"])
        out["bxcvx"] = out["bxcvx"] / Bnorm
        out["bxcvy"] = out["bxcvy"] * (rho_s0 ** 2)
        out["Bxy"]   = out["Bxy"]   / Bnorm

    # If the grid-file y-size doesn't match the dataset, pad with the BOUT++
    # `extrapolate_y = false` convention used by Hermes-3 for the metric:
    # zero-gradient copy of the boundary cells into the guards. Works for
    # single-null (2 guard regions: bottom + top) and double-null (4 guard
    # regions: bottom + inner X-pt × 2 + top), with the inter-region split
    # taken from `ny_inner` in the dataset metadata.
    if ds is not None:
        ds_pdim = detect_poloidal_dim(ds)
        n_ds   = ds.sizes[ds_pdim]
        n_grid = out["bxcvx"].sizes[pdim]
        if n_ds != n_grid:
            import numpy as _np
            MYG = int(ds.metadata.get("MYG", 2))
            ny_inner = ds.metadata.get("ny_inner", None)
            is_dn = ny_inner is not None and 0 < int(ny_inner) < n_grid

            # Expected guard layouts:
            #   SN: dump = grid + 2·MYG  (lower + upper)
            #   DN: dump = grid + 4·MYG  (lower + inner upper + outer lower + upper)
            expected_extra = (4 if is_dn else 2) * MYG

            # Mirror case: the GRID carries the guards and the dump does not.
            # A grid built with y_boundary_guards=2 stores ny+4·MYG rows, so a
            # guardless (or guard-trimmed) dump is *smaller* than the grid.
            # Drop the grid's guard rows rather than erroring — the same
            # layout as the padding below, read the other way.
            if n_grid - n_ds == expected_extra:
                g_rows = (list(range(MYG))
                          + list(range(int(ny_inner) + MYG,
                                       int(ny_inner) + 3 * MYG))
                          + list(range(n_ds + 3 * MYG, n_ds + 4 * MYG))
                          ) if is_dn else (
                    list(range(MYG)) + list(range(n_ds + MYG, n_ds + 2 * MYG)))
                keep = [i for i in range(n_grid) if i not in set(g_rows)]
                for k in out:
                    out[k] = out[k].isel({pdim: keep})
                return out

            if n_ds - n_grid != expected_extra:
                raise ValueError(
                    f"load_grid_vars: y-dim mismatch — gridfile has "
                    f"{n_grid} cells, dump has {n_ds}; expected an extra "
                    f"{expected_extra} guard cells for "
                    f"{'DN' if is_dn else 'SN'} (MYG={MYG}). "
                    "Either re-squash with `keep_yboundaries=False` so the "
                    "dump matches the grid, OR save bxcvx/bxcvy/Bxy into the "
                    "dump itself (Hermes-3 normally writes them)."
                )

            def _pad_grid_to_dump(arr_grid):
                """Zero-gradient (Neumann) extrapolation into y-guards."""
                a = arr_grid.values
                # Build the padded array by stacking [guards | region | guards].
                if is_dn:
                    split = int(ny_inner)
                    # Region A: [0:split] → pad MYG on each side
                    A = a[:, :split]
                    A_pad = _np.concatenate(
                        [_np.repeat(A[:, :1], MYG, axis=1),
                         A,
                         _np.repeat(A[:, -1:], MYG, axis=1)], axis=1)
                    # Region B: [split:] → pad MYG on each side
                    B = a[:, split:]
                    B_pad = _np.concatenate(
                        [_np.repeat(B[:, :1], MYG, axis=1),
                         B,
                         _np.repeat(B[:, -1:], MYG, axis=1)], axis=1)
                    padded = _np.concatenate([A_pad, B_pad], axis=1)
                else:
                    padded = _np.concatenate(
                        [_np.repeat(a[:, :1], MYG, axis=1),
                         a,
                         _np.repeat(a[:, -1:], MYG, axis=1)], axis=1)
                return xr.DataArray(padded, dims=arr_grid.dims)

            for key in ("bxcvx", "bxcvy", "Bxy"):
                out[key] = _pad_grid_to_dump(out[key])
            # Rename the y-dim to match ds's poloidal dim name if different
            if pdim != ds_pdim:
                for key in ("bxcvx", "bxcvy", "Bxy"):
                    out[key] = out[key].rename({pdim: ds_pdim})
            print(f"  load_grid_vars: padded grid-file y={n_grid} → "
                  f"dump y={n_ds} via zero-gradient extrapolation "
                  f"(MYG={MYG}, {'DN' if is_dn else 'SN'})")

    return out


def to_field_aligned(da, ds=None, direction=+1):
    """
    Shift a DataArray to field-aligned coordinates by the toroidal angle
    ``zShift``, using FFTs in the periodic toroidal direction.

    This replicates ``xbout``'s ``DataArray.bout.to_field_aligned()`` /
    ``_shift_z`` but works **without** ``geometry="toroidal"`` applied — i.e.
    on a guard-retaining squashed dump that cannot be loaded with a (guardless)
    grid file. ``zShift``, ``dz`` and ``nz`` are read from the dump.

    Why this matters: with ``[mesh:paralleltransform] type = shifted``
    (ShiftedMetric) the data is stored in *non-aligned* coordinates, so cells at
    the same ``(x, z)`` index but adjacent ``y`` are NOT connected along the
    field line — they are offset toroidally by ``zShift``. A parallel
    derivative ``∂/∂y`` (e.g. for heat conduction ``−κ_∥ ∂T/∂s_∥``) is only
    correct after aligning, otherwise ``zShift`` variation produces spurious
    jumps. Align → take the ``y`` difference → (optionally) shift back with
    ``direction=-1``.

    Parameters
    ----------
    da : xarray.DataArray
        Field with a toroidal dimension (``zeta``/``z``); typically (x, θ, ζ).
    ds : xarray.Dataset, optional
        Source of ``zShift``/``dz``/``nz`` (defaults to ``da`` itself).
    direction : int
        ``+1`` for to-field-aligned (default), ``-1`` for from-field-aligned.

    Returns
    -------
    xarray.DataArray
        ``da`` shifted in the toroidal direction, same dims/coords/attrs.

    Notes
    -----
    A single global FFT shift is invalid across topological branch cuts
    (X-points, the inter-region guard rows of a double null), so derivatives at
    those poloidal indices remain unreliable and should be masked.
    """
    import numpy as np

    src = ds if ds is not None else da
    zdim = detect_toroidal_dim(da)
    # Rechunk before .values so the numpy FFT below sees the full ζ axis.
    da = _ensure_z_single_chunk(da, zdim)
    zsh = _no_t(src["zShift"])
    nz = da.sizes[zdim]
    dz = float(np.asarray(_no_t(src["dz"]).values).flat[0])
    zlength = nz * dz
    nmodes = nz // 2 + 1
    kz = 2.0 * np.pi * np.arange(nmodes) / zlength

    order = [d for d in da.dims if d != zdim] + [zdim]
    arr = da.transpose(*order).values
    # zShift only has (x, θ); `da` may have extra leading dims like 't'.
    # Transpose to the subset of `order` that zsh actually has, then reshape
    # with singleton 1s in the positions of dims zsh doesn't carry so numpy
    # broadcasts correctly against `arr` (whose non-ζ axes are `order[:-1]`).
    zsh_dims_in_order = [d for d in order if d != zdim and d in zsh.dims]
    zsh_t = zsh.transpose(*zsh_dims_in_order).values
    shape_for_zsh = []
    j = 0
    for d in order[:-1]:           # all non-ζ dims of `arr`, in order
        if d in zsh.dims:
            shape_for_zsh.append(zsh_t.shape[j])
            j += 1
        else:
            shape_for_zsh.append(1)
    zsh_b = zsh_t.reshape(tuple(shape_for_zsh))
    phase = np.exp(direction * 1.0j * kz * zsh_b[..., None])
    out = np.fft.irfft(np.fft.rfft(arr, axis=-1) * phase, n=nz, axis=-1)

    return da.transpose(*order).copy(data=out).transpose(*da.dims)


#: Atomic mass number per Hermes-3 species label, used only for the NV/N
#: fallback in :func:`get_parallel_velocity`. Hermes-3 does not save ``AA`` to
#: the dump, so it has to be supplied here or by the caller.
SPECIES_AA = {
    "e": 1.0 / 1836.152674,
    "i": 2.0,          # deuterium main ion in these cases
    "h+": 1.0,
    "d+": 2.0,
    "d": 2.0,
    "t+": 3.0,
    "t": 3.0,
    "he+": 4.0,
    "he+2": 4.0,
    "he": 4.0,
}


def get_parallel_velocity(ds, species, AA=None):
    """
    Return parallel velocity, deriving it from momentum when V is absent.

    Hermes-3 stores momentum as ``NV = AA * N * V`` (mass density times
    velocity), so the fallback **must** divide by the atomic mass number as
    well as the density. Verified against dumps that save both: for deuterium
    ``(NVi/Ni)/Vi = 2.0`` exactly, and for electrons ``(NVe/Ne)/Ve = 1/1836``.
    This relation applies to normalised fields. For SI fields momentum is
    ``NV = AA * proton_mass * N * V``. Both momentum and density must have
    matching ``units_type`` annotations. Unannotated fields retain the legacy
    assumption of normalised data. If available, the momentum conversion and
    Nnorm/Cs0 metadata determine the proton mass used by the loader; otherwise
    the CODATA proton mass is used.

    ``AA`` is not saved to the dump, so it comes from ``SPECIES_AA`` unless the
    caller passes it explicitly (needed for any species not in that table, or
    for a non-deuterium main ion labelled ``'i'``).

    Parameters
    ----------
    ds : xarray.Dataset
    species : str
        Species label as used in Hermes-3, e.g. 'e', 'i', 'd+'.
    AA : float, optional
        Atomic mass number. Defaults to ``SPECIES_AA[species]``. Only used by
        the NV/N fallback; ignored when ``V<species>`` is present.

    Returns
    -------
    xarray.DataArray
        Parallel velocity V_{∥s} [m/s if unnormalised].
    """
    v_name = f"V{species}"
    nv_name = f"NV{species}"
    n_name = f"N{species}"

    if v_name in ds:
        return ds[v_name]

    if nv_name in ds and n_name in ds:
        from math import isfinite
        if AA is None:
            try:
                AA = SPECIES_AA[species]
            except KeyError:
                raise KeyError(
                    f"No atomic mass number known for species '{species}', and "
                    f"'{v_name}' is not in the dataset. Pass AA=... explicitly "
                    f"(Hermes-3 stores NV = AA*N*V and does not save AA)."
                ) from None
        momentum, density = ds[nv_name], ds[n_name]
        momentum_si = str(momentum.attrs.get("units_type", "")).upper() == "SI"
        density_si = str(density.attrs.get("units_type", "")).upper() == "SI"
        if momentum_si != density_si:
            raise ValueError("Momentum and density must use matching units_type "
                             "annotations (SI or normalised)")
        mass = float(AA)
        if not isfinite(mass) or mass <= 0:
            raise ValueError("AA must be finite and positive")
        if momentum_si:
            meta = ds.attrs.get("metadata", {})
            proton_mass = 1.67262192369e-27  # kg; fallback when conversions are absent
            nscale = density.attrs.get("conversion", meta.get("Nnorm"))
            vscale = meta.get("Cs0")
            pscale = momentum.attrs.get("conversion")
            if nscale is not None and vscale is not None and pscale is not None:
                scales = tuple(float(s) for s in (pscale, nscale, vscale))
                if not all(isfinite(s) and s > 0 for s in scales):
                    raise ValueError("Invalid momentum/density conversion factors")
                proton_mass = scales[0] / (scales[1] * scales[2])
            if not isfinite(proton_mass) or proton_mass <= 0:
                raise ValueError("Invalid momentum/density conversion factors")
            mass *= proton_mass
        return momentum / (mass * density)

    raise KeyError(
        f"Cannot find parallel velocity for species '{species}'. "
        f"Expected '{v_name}' or both '{nv_name}' and '{n_name}' in dataset."
    )


def dedup_restart_times(ds, rtol=1e-3, verbose=True):
    """Drop repeated / out-of-order output times after concatenating folders.

    **Use this instead of dropping each folder's first frame.** The positional
    rule assumes the first output of a restarted run duplicates the last
    output of the previous one, which is only true for BOUT++'s default
    ``dump_at_restart = true``. With ``dump_at_restart = false`` the first
    frame is a **genuinely new time**, so the positional drop silently deletes
    real data — and a folder squashed to a *single* output time then
    contributes nothing at all, so N such folders collapse to one frame.
    Measured on ``sxd_MU/from_archer_6kW``: folders 404/405/407/408 hold one
    frame each and vanished entirely, while 403/404 really do share
    ``t = 726111.1``. Comparing time *values* handles both conventions with
    no knowledge of the flag, and mixed settings across one ``dir_range`` as
    well.

    The comparison is a **running max**, not against the previous frame only:
    a folder that starts before the previous one ends would otherwise leave a
    non-monotonic series and a negative ``dt``.

    ``rtol`` guards float noise. A restart time that differs in the last bits
    (``726111.1`` vs ``726111.100001``) is not caught by a strict ``>`` and
    leaves a near-zero ``dt`` — which is the 0.1 µs trap that once made
    parallel flows look steady to 0.5 % when the real frame spacing was
    104 µs. A frame is kept only if it is later than the running max by more
    than ``rtol`` of the median positive step. Any surviving step below
    ``rtol`` of the median is reported.

    Returns the dataset with the offending time points removed.
    """
    import numpy as np

    if "t" not in ds.dims or ds.sizes["t"] < 2:
        return ds
    t = np.asarray(ds["t"].values, dtype=float)
    d = np.diff(t)
    med = float(np.median(d[d > 0])) if np.any(d > 0) else 0.0
    tol = rtol * med

    keep = np.ones(t.size, dtype=bool)
    run = -np.inf
    for i, ti in enumerate(t):
        if ti > run + tol:
            run = ti
        else:
            keep[i] = False
    ndrop = int((~keep).sum())
    if ndrop and verbose:
        print(f"  dropped {ndrop} duplicated/out-of-order time point(s) "
              f"at restart boundaries")
    ds = ds.isel(t=np.flatnonzero(keep))

    tk = np.asarray(ds["t"].values, dtype=float)
    if tk.size > 1 and med > 0:
        dk = np.diff(tk)
        bad = dk < rtol * med
        if bad.any() and verbose:
            print(f"  WARNING: {int(bad.sum())} retained step(s) below "
                  f"{rtol:g} x the median dt ({med:.4g}) — a near-duplicate "
                  f"frame survived\n           and any velocity or rate from "
                  f"this series will be corrupted.")
    return ds


def existing_dirs(data_root, dir_range, names=("boutdata.nc",)):
    """The folders of ``dir_range`` that hold a dump, printing the ones skipped.

    A folder counts if any of ``names`` (glob patterns allowed) exists in
    ``data_root/<dir>``. Exits if none does.
    """
    import glob
    import os
    import sys
    names = (names,) if isinstance(names, str) else tuple(names)
    keep, skip = [], []
    for d in dir_range:
        folder = os.path.join(data_root, str(d))
        found = any(glob.glob(os.path.join(folder, n)) for n in names)
        (keep if found else skip).append(d)
    if skip:
        print(f"skipping {len(skip)} folder(s) with no {' / '.join(names)}: {skip}")
    if not keep:
        sys.exit(f"no {' / '.join(names)} in any of {list(dir_range)} under {data_root}")
    return keep


def cache_path(name, dir_range, *extra, cache_dir="./cache"):
    """Build a pickle cache filename inside ``cache_dir`` for a script run.

    Example:
        cache_path("v3_radial_flux", range(146, 148), "e")
        → Path("./cache/v3_radial_flux_dirs146-147_e.npz")

    The directory is created if missing.
    """
    from pathlib import Path
    p = Path(cache_dir)
    p.mkdir(parents=True, exist_ok=True)
    drs = list(dir_range) if hasattr(dir_range, "__iter__") else [dir_range]
    rng = f"{min(drs)}-{max(drs)}" if len(drs) > 1 else f"{drs[0]}"
    tail = "_".join(str(e) for e in extra if e not in (None, "", False))
    return p / (f"{name}_dirs{rng}" + (f"_{tail}" if tail else "") + ".npz")


def save_cache(path, **arrays):
    """Save a dict of arrays/scalars to ``path`` in numpy's portable ``.npz``.

    The suffix is forced to ``.npz``. Unlike a pickle, this format is readable
    across numpy major versions and machines, so a cache written on the
    cluster loads on a laptop with a different numpy (a pickle written under
    numpy>=2 fails on numpy 1.x with ``ModuleNotFoundError: numpy._core``).

    xarray objects are stored as plain numpy via ``.values``. Non-array values
    (``None``, ``slice``, strings, numbers, and JSON-able dicts/lists/tuples)
    go into a single JSON side entry so nothing needs ``allow_pickle``.
    """
    import json
    from pathlib import Path
    import numpy as np

    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)

    npz, meta = {}, {}
    for k, v in arrays.items():
        if isinstance(v, np.generic):          # numpy scalar -> python scalar
            v = v.item()
        if v is None:
            meta[k] = {"none": True}
        elif isinstance(v, slice):
            meta[k] = {"slice": [v.start, v.stop, v.step]}
        elif isinstance(v, (str, bool, int, float)):
            meta[k] = {"value": v}
        elif isinstance(v, (dict, list, tuple)):
            try:
                json.dumps(v)
                meta[k] = {"value": list(v) if isinstance(v, tuple) else v,
                           "tuple": isinstance(v, tuple)}
            except TypeError:                  # not JSON-able: store as array
                npz[k] = np.asarray(v)
        elif hasattr(v, "values"):             # xarray DataArray
            npz[k] = np.asarray(v.values)
        else:
            npz[k] = np.asarray(v)

    npz["__meta__"] = np.asarray(json.dumps(meta))
    np.savez(path, **npz)
    return path


def load_cache(path):
    """Load a cache written by :func:`save_cache`. Returns a dict.

    Prefers the portable ``.npz``; falls back to a legacy ``.pkl`` of the same
    stem so old caches keep working. A legacy pickle written under a different
    numpy major version cannot be read at all — that raises a clear error
    naming the remedy rather than a bare ``ModuleNotFoundError``.
    """
    import json
    from pathlib import Path
    import numpy as np

    p = Path(path)
    npz = p.with_suffix(".npz")
    if npz.exists():
        with np.load(npz, allow_pickle=False) as z:
            out = {k: z[k] for k in z.files if k != "__meta__"}
            meta = json.loads(z["__meta__"].item()) if "__meta__" in z.files else {}
        for k, spec in meta.items():
            if spec.get("none"):
                out[k] = None
            elif "slice" in spec:
                out[k] = slice(*spec["slice"])
            else:
                v = spec["value"]
                out[k] = tuple(v) if spec.get("tuple") else v
        return out

    pkl = p.with_suffix(".pkl")
    if not pkl.exists():
        raise FileNotFoundError(f"no cache at {npz} or {pkl}")
    import pickle
    try:
        with open(pkl, "rb") as f:
            return pickle.load(f)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise RuntimeError(
            f"legacy pickle cache {pkl} was written by an incompatible "
            f"numpy/python (original error: {exc}).\nDelete it and recompute "
            f"— new caches are written as portable .npz and will not have "
            f"this problem."
        ) from exc


def check_normalisation(ds, verbose=True, rtol=1e-6):
    """Verify the dump's normalisation constants are mutually consistent.

    Hermes-3 defines (``hermes-3.cxx``)

        Omega_ci = qe·Bnorm/Mp        <- PROTON mass, no AA factor
        Cs0      = sqrt(qe·Tnorm/Mp)
        rho_s0   = Cs0/Omega_ci

    Note the absence of the ion mass number: assuming an ``AA`` divisor is the
    classic error (a factor 6.6 in one earlier script). This helper recomputes
    all three from ``Bnorm``/``Tnorm`` and compares them with what the dump
    actually stores, so a script never has to trust a hard-coded fallback.

    Any quantity converted to SI with these constants is only as good as they
    are, so run this whenever an SI number looks surprising. Note, however,
    that *dimensionless* diagnostics — a displacement expressed in grid cells,
    or a number of frames per transit — cancel the normalisation exactly and
    are therefore immune to an error here.

    Returns a dict of ``{name: (dump, recomputed, ratio)}`` plus ``'consistent'``.
    """
    import numpy as np

    qe = 1.602176634e-19
    Mp = 1.67262192369e-27
    m = ds.metadata
    Bnorm = float(m["Bnorm"])
    Tnorm = float(m["Tnorm"])

    expect = {
        "Omega_ci": qe * Bnorm / Mp,
        "Cs0": np.sqrt(qe * Tnorm / Mp),
    }
    expect["rho_s0"] = expect["Cs0"] / expect["Omega_ci"]

    out, ok = {}, True
    for k, want in expect.items():
        if k not in m:
            continue
        got = float(m[k])
        ratio = got / want if want else np.nan
        out[k] = (got, want, ratio)
        if not np.isclose(ratio, 1.0, rtol=rtol):
            ok = False
    out["consistent"] = ok

    if verbose:
        print("\n===== normalisation check =====")
        print(f"  Bnorm = {Bnorm:g} T   Tnorm = {Tnorm:g} eV")
        for k in expect:
            if k not in out:
                continue
            got, want, ratio = out[k]
            flag = "ok" if np.isclose(ratio, 1.0, rtol=rtol) else "MISMATCH"
            print(f"  {k:9s} dump={got:<20.10g} qe,Mp form={want:<20.10g} "
                  f"ratio={ratio:.9f}  {flag}")
        if ok:
            print("  all constants self-consistent (proton mass, no AA factor)")
        else:
            print("  WARNING: a constant disagrees with the qe/Mp definition — "
                  "SI values\n           from this dump are suspect. "
                  "Dimensionless diagnostics are not.")
        print("===============================\n")
    return out


def detect_toroidal_dim(ds):
    """Return the name of the toroidal dimension ('zeta' or 'z')."""
    for candidate in ("zeta", "z"):
        if candidate in ds.dims:
            return candidate
    raise ValueError("Cannot detect toroidal dimension. Expected 'zeta' or 'z'.")


def detect_poloidal_dim(ds):
    """Return the name of the poloidal dimension ('theta' or 'y')."""
    for candidate in ("theta", "y"):
        if candidate in ds.dims:
            return candidate
    raise ValueError("Cannot detect poloidal dimension. Expected 'theta' or 'y'.")

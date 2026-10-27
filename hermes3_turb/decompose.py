"""
Toroidal mean / fluctuation decomposition.

For 3D turbulence, the toroidal (zeta) direction is used to separate
the equilibrium background from turbulent fluctuations:

    f(x, θ, ζ) = f̄(x, θ) + δf(x, θ, ζ)

where f̄ = <f>_ζ is the toroidal mean and δf is the fluctuation.

The turbulent radial transport then comes from correlations:
    <δn δv_ExB_r>_ζ

compared with the mean-flow contribution:
    n̄ · v̄_ExB_r
"""

import xarray as xr

from .load import detect_toroidal_dim, detect_poloidal_dim


def toroidal_mean(da, dim=None):
    """
    Toroidal (ζ) average of a DataArray.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str, optional
        Name of the toroidal dimension. Auto-detected if not given.

    Returns
    -------
    xarray.DataArray
        Toroidal mean with the zeta dimension collapsed.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    return da.mean(dim=dim)


def fluctuation(da, dim=None):
    """
    Toroidal fluctuation: δf = f - <f>_ζ.

    Returns
    -------
    xarray.DataArray
        Same shape as input; toroidal mean is zero.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    return da - da.mean(dim=dim)


def reynolds_decompose(da, dim=None):
    """
    Split a field into its toroidal mean and fluctuation.

    Returns
    -------
    mean : xarray.DataArray
        <f>_ζ  (toroidal mean, zeta dimension collapsed)
    fluc : xarray.DataArray
        δf = f - <f>_ζ  (same shape as input)
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    mean = da.mean(dim=dim)
    fluc = da - mean
    return mean, fluc


def turbulent_correlation(a, b, dim=None):
    """
    Turbulent (Reynolds) correlation: <δa · δb>_ζ.

    Computes the toroidal mean of the product of fluctuations, which gives
    the turbulent transport when a=n and b=v_ExB_r, for example.

    Parameters
    ----------
    a, b : xarray.DataArray
        Fields to correlate. Must be broadcastable.
    dim : str, optional
        Toroidal dimension name. Auto-detected if not given.

    Returns
    -------
    xarray.DataArray
        <δa δb>_ζ with the toroidal dimension collapsed.
    """
    if dim is None:
        dim = detect_toroidal_dim(a)
    delta_a = a - a.mean(dim=dim)
    delta_b = b - b.mean(dim=dim)
    return (delta_a * delta_b).mean(dim=dim)


def fluctuation_rms(da, dim=None):
    """
    RMS amplitude of the toroidal fluctuation:  √(⟨(f − ⟨f⟩_ζ)²⟩_ζ).

    This is the quantity Ben Dudson recommends plotting against time to watch
    turbulence grow and saturate (BOUT++ Zulip, Hermes-3 / "Initializing
    Turbulence Runs"): the toroidal average is removed first so the metric is
    not swamped by slow changes to the background profile.

    Returns
    -------
    xarray.DataArray
        Toroidal (ζ) dimension collapsed; same units as ``da``.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    mean = da.mean(dim=dim)
    return (((da - mean) ** 2).mean(dim=dim)) ** 0.5


def relative_fluctuation(da, dim=None, floor=None):
    """
    Relative toroidal-fluctuation level:  √(⟨δf²⟩_ζ) / |⟨f⟩_ζ|.

    Multiply by 100 for the "few to 10s of %" scale that signals a saturated
    turbulent state. Only meaningful for sign-definite fields with a non-zero
    toroidal mean (density, pressure, temperature) — for a zero-mean field such
    as vorticity use :func:`fluctuation_rms` instead.

    Parameters
    ----------
    floor : float, optional
        Values of |⟨f⟩_ζ| below this are masked (set to NaN) to avoid divide-by-
        near-zero blow-up. Defaults to a tiny fraction of the field's own max.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    mean = da.mean(dim=dim)
    rms = (((da - mean) ** 2).mean(dim=dim)) ** 0.5
    denom = abs(mean)
    if floor is None:
        floor = 1e-6 * abs(mean).max()
    denom = denom.where(denom > floor)
    return rms / denom


def time_mean(da, dim="t", tstart=None):
    """
    Time average, optionally over a saturation window [tstart:].

    Parameters
    ----------
    da : xarray.DataArray
    dim : str
        Time dimension name.
    tstart : int or float, optional
        If int: index of the first time step to include.
        If float: treated as fraction of total time steps (0.5 = last half).

    Returns
    -------
    xarray.DataArray
    """
    if tstart is not None:
        if isinstance(tstart, float) and 0.0 < tstart < 1.0:
            tstart = int(tstart * da.sizes[dim])
        da = da.isel({dim: slice(tstart, None)})
    return da.mean(dim=dim)


def skewness(da, dim=None):
    """
    Skewness of the toroidal (ζ) fluctuation:  ⟨δf³⟩ / σ³,  σ = √⟨δf²⟩.

    Third standardised moment — the intermittency asymmetry of the turbulence
    (positive skew ⇒ bursty positive excursions, e.g. blobs; negative ⇒ holes).
    Population (biased) convention, matching ``scipy.stats.skew`` used in the
    STORM analysis; a Gaussian field gives 0.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str or list of str, optional
        Dimension(s) to take the moment over. Defaults to the toroidal (ζ)
        dimension; pass e.g. ``["zeta", "t"]`` to pool toroidal and time samples
        into one PDF per (x, θ).

    Returns
    -------
    xarray.DataArray
        Dimensionless; the moment dimension(s) collapsed.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    delta = da - da.mean(dim=dim)
    var = (delta ** 2).mean(dim=dim)
    return (delta ** 3).mean(dim=dim) / var ** 1.5


def kurtosis(da, dim=None, excess=True):
    """
    Kurtosis of the toroidal (ζ) fluctuation:  ⟨δf⁴⟩ / σ⁴  (− 3 if ``excess``).

    Fourth standardised moment — the "tailedness"/intermittency of the
    turbulence: excess kurtosis > 0 means heavy tails (rare large bursts).
    ``excess=True`` (default) subtracts 3 so a Gaussian gives 0, matching
    ``scipy.stats.kurtosis`` (Fisher, biased) used in the STORM analysis.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str or list of str, optional
        Dimension(s) to take the moment over. Defaults to the toroidal (ζ)
        dimension; pass e.g. ``["zeta", "t"]`` to pool toroidal and time samples.
    excess : bool, default True
        Subtract 3 (Fisher). ``False`` returns the raw 4th-moment ratio.

    Returns
    -------
    xarray.DataArray
        Dimensionless; the moment dimension(s) collapsed.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    delta = da - da.mean(dim=dim)
    var = (delta ** 2).mean(dim=dim)
    k = (delta ** 4).mean(dim=dim) / var ** 2
    return k - 3.0 if excess else k


def radial_statistics(da, dim=None, theta=None, excess=True):
    """
    The four PDF moments of the toroidal fluctuation bundled into one Dataset:
    ``mean``, ``std``, ``skewness``, ``kurtosis`` — the "statistics vs radius".

    The moments are taken over the toroidal (ζ) dimension by default (pass
    ``dim=["zeta", "t"]`` to pool toroidal + time into one PDF per (x, θ)), then
    optionally reduced over the poloidal dimension to give a pure radial profile.
    `delta` and the variance are formed once and shared, so this is cheaper than
    calling the four moment functions separately.

    Parameters
    ----------
    da : xarray.DataArray
    dim : str or list of str, optional
        Toroidal moment dimension(s). Auto-detected (ζ) if not given.
    theta : None | "mean" | int, optional
        Poloidal reduction. ``None`` keeps the poloidal dimension; ``"mean"``
        averages over it; an int selects that poloidal index — e.g. the OMP index
        from ``target.get_midplane_y_indices(ds)`` — to give a radial profile.
    excess : bool, default True
        Fisher (excess) kurtosis, matching ``scipy.stats``.

    Returns
    -------
    xarray.Dataset
        Variables ``mean``, ``std``, ``skewness``, ``kurtosis``.
    """
    if dim is None:
        dim = detect_toroidal_dim(da)
    mean = da.mean(dim=dim)
    delta = da - mean
    var = (delta ** 2).mean(dim=dim)
    skew = (delta ** 3).mean(dim=dim) / var ** 1.5
    kurt = (delta ** 4).mean(dim=dim) / var ** 2
    if excess:
        kurt = kurt - 3.0
    out = xr.Dataset(
        {"mean": mean, "std": var ** 0.5, "skewness": skew, "kurtosis": kurt}
    )
    if theta is not None:
        pdim = detect_poloidal_dim(out)
        out = out.mean(dim=pdim) if theta == "mean" else out.isel({pdim: theta})
    return out

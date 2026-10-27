"""
Geometric area and volume elements from BOUT++ metric tensors.

All area/volume elements are in physical SI units when the dataset has
been unnormalised by xhermes (dx, dy, dz, J, g_ij all in SI).

BOUT++ metric conventions (xhermes notation):
  g11  = g^{xx}  : contravariant radial-radial          [T²m²]
  g_22 = g_{θθ}  : covariant poloidal-poloidal           [m²]
  g_33 = g_{ζζ}  : covariant toroidal-toroidal           [m²]  (≈ R²)
  g_23 = g_{θζ}  : covariant poloidal-toroidal cross     [m²]
  g_11 = g_{xx}  : covariant radial-radial               [m⁻²·T⁻²] (rarely needed)
  g_13 = g_{xζ}  : covariant radial-toroidal cross       [m²]  (often ≈0)
  J    : Jacobian of the coordinate transformation       [m/T]
  dx   : radial coordinate width     [Wb]  (flux space)
  dy   : poloidal coordinate width   [rad]
  dz   : toroidal coordinate width   [rad]

These are the same conventions used by the STORM post-processing scripts
(v2_fluxX.py, v2_fluxY.py, etc.), so area elements match those scripts.
"""

import numpy as np

from .load import _no_t


def radial_area_element(ds):
    """
    Area element for a flux surface (x = const) per unit (dθ · dζ).

    dA_x = sqrt(g_{θθ} · g_{ζζ} - g_{θζ}²) [m²/rad²]

    This is the metric factor that, multiplied by dy*dz (in radians),
    gives the physical area [m²].

    In the STORM convention, the 'elem_sur' for radial flux integration is:
        elem_sur = ds['dy'] * sqrt(g_22 * g_33 - g_23²)
    i.e., `dy` is already included for easy summation over theta.

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.DataArray [m²/rad²]
        Multiply by dy*dz or sum(dy * this) over theta to get area [m²].
    """
    g_22, g_33, g_23 = (_no_t(ds[k]) for k in ("g_22", "g_33", "g_23"))
    return np.sqrt(g_22 * g_33 - g_23 ** 2)


def radial_area_element_dtheta(ds):
    """
    Area element for a flux surface pre-multiplied by dθ.

    elem_sur = dy * sqrt(g_22 * g_33 - g_23²)  [m²/rad]

    Summing over the theta dimension gives the flux-surface area per unit dζ.
    Following the STORM script convention used for conservation checks.
    """
    dy, g_22, g_33, g_23 = (_no_t(ds[k]) for k in ("dy", "g_22", "g_33", "g_23"))
    return dy * np.sqrt(g_22 * g_33 - g_23 ** 2)


def poloidal_area_element(ds):
    """
    Area element for a poloidal cross-section (θ = const) per unit (dx · dζ).

    dA_θ = sqrt(g_{xx} · g_{ζζ} - g_{xζ}²) [m²]

    In the STORM convention, the 'elem_sur' for poloidal flux integration is:
        elem_sur = ds['dx'] * sqrt(g_11 * g_33 - g_13²)

    Parameters
    ----------
    ds : xarray.Dataset

    Returns
    -------
    xarray.DataArray [m²]
    """
    g_11, g_33, g_13 = (_no_t(ds[k]) for k in ("g_11", "g_33", "g_13"))
    return np.sqrt(g_11 * g_33 - g_13 ** 2)


def poloidal_area_element_dx(ds):
    """
    Area element for a poloidal cross-section pre-multiplied by dx.

    elem_sur = dx * sqrt(g_11 * g_33 - g_13²)

    Summing over x gives the poloidal-surface area per unit dζ.
    """
    dx, g_11, g_33, g_13 = (_no_t(ds[k]) for k in ("dx", "g_11", "g_33", "g_13"))
    return dx * np.sqrt(g_11 * g_33 - g_13 ** 2)


def cell_volume(ds):
    """
    Volume element for each computational cell: dV = J · dx · dy · dz [m³].

    Multiply a volumetric source density [X/m³/s] by this and sum over all
    cells to get the total source rate [X/s].
    """
    J, dx, dy, dz = (_no_t(ds[k]) for k in ("J", "dx", "dy", "dz"))
    return J * dx * dy * dz

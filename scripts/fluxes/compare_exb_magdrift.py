#!/usr/bin/env python3
"""
Cross-validate h3t ExB and magnetic drift formulas against xstorm-equivalent
expressions at a single time, z-slice, and poloidal location (OMP).

Both are evaluated in BOUT++ normalised units so the numbers can be directly
compared against v2_fluxX.py results on the same or equivalent STORM data.

Usage
-----
    python compare_exb_magdrift.py ./146 ../grid.nc
"""
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import xarray as xr
from xhermes import open_hermesdataset
import hermes3_turb as h3t

# ── CONFIG ────────────────────────────────────────────────────────────────────
casepath = './146' #sys.argv[1] if len(sys.argv) > 1 else "."
gridfile  = '../exp45470_x132_y28.nc' #sys.argv[2] if len(sys.argv) > 2 else None
t_idx     = 0     # time index to compare
z_idx     = 0     # toroidal index to compare
# ─────────────────────────────────────────────────────────────────────────────

# ── Load (normalised units, same as STORM post-processing) ───────────────────
print(f"Loading: {casepath}")
_nc = os.path.join(casepath, "boutdata.nc")
datapath = _nc if os.path.exists(_nc) else os.path.join(casepath, "BOUT.dmp.*.nc")
ds = open_hermesdataset(
    datapath,
    gridfilepath=gridfile,
    geometry="toroidal",
    keep_yboundaries=False,
    unnormalise=False,
)

pdim = h3t.detect_poloidal_dim(ds)
zdim = h3t.detect_toroidal_dim(ds)

# Grid vars from file
gv    = nc.Dataset(gridfile)
bxcvx = xr.DataArray(np.array(gv["bxcvx"][:]), dims=["x", pdim])
Bxy   = xr.DataArray(np.array(gv["Bxy"][:]),   dims=["x", pdim])
gv.close()

# Topology
m       = ds.metadata
ixseps1 = int(m.get("ixseps1", ds.sizes["x"] // 2))
ixseps2 = int(m.get("ixseps2", ixseps1))
if ixseps1 > ixseps2:
    ixseps1, ixseps2 = ixseps2, ixseps1
# Detect guard cells: compare dataset y-size with grid file ny
with nc.Dataset(gridfile) as _grd:
    _ny_grid = int(_grd["ny"][:])
_has_guards = ds.sizes[pdim] > _ny_grid
MYG = int(m.get("MYG", 0)) if _has_guards else 0
print(f"ny_grid={_ny_grid}, ny_data={ds.sizes[pdim]}, guard cells in data: {_has_guards}, MYG={MYG}")

# OMP: middle of outer main SOL in theta
j1  = int(m.get("jyseps1_2", ds.sizes[pdim] // 4)) + MYG * 3
j2  = int(m.get("jyseps2_2", 3 * ds.sizes[pdim] // 4)) + MYG * 3
y_omp = (j1 + j2) // 2
print(f"OMP theta index: {y_omp}  (jyseps1_2+guard={j1}, jyseps2_2+guard={j2})")

# ── Slice: single time, single z, OMP ────────────────────────────────────────
# Keep full (x, theta, zeta) for derivative operators, then slice after
ds_t = ds.isel(t=t_idx)   # (x, theta, zeta)

ion_species = "i" if "Ni" in ds.data_vars else "d+"

phi = ds_t["phi"]
Ne  = ds_t["Ne"]
Pe  = ds_t["Pe"]
Ni  = ds_t[f"N{ion_species}"]
Pi  = ds_t[f"P{ion_species}"]
Te  = ds_t["Te"]         if "Te"              in ds_t else Pe / Ne.clip(min=1e-10)
Ti  = ds_t[f"T{ion_species}"].fillna(Pi / Ni.clip(min=1e-10)) if f"T{ion_species}" in ds_t else Pi / Ni.clip(min=1e-10)

# ── ExB radial velocity ───────────────────────────────────────────────────────
# Formula A — h3t default (include_y_derivs=False, matches STORM)
vExB_no_y   = phi.bout.ddz() / np.sqrt(ds_t["g11"])

# Formula B — include y-derivs (Hermes-3 with poloidal_flows=true)
vExB_with_y = (phi.bout.ddz() - ds_t["g_23"] / ds_t["g_22"] * phi.bout.ddy()) \
              / np.sqrt(ds_t["g11"])

# ── Magnetic drift radial velocity (electrons and ions) ──────────────────────
# v_mag_s = -(T_s / q_s) * 2*bxcvx / (Bxy * sqrt(g11))
# electrons (q=-e): negative sign → v_mag_e = -Te * 2*bxcvx / (Bxy*sqrt(g11))
# ions      (q=+e): positive sign → v_mag_i = +Ti * 2*bxcvx / (Bxy*sqrt(g11))
# (mass-independent in drift-fluid limit: m*v_th^2 = T cancels m)
geom  = 2.0 * bxcvx / (Bxy * np.sqrt(ds_t["g11"]))
v_mag_e = -Te * geom
v_mag_i = +Ti * geom

# ── Slice to (x,) at OMP and z=z_idx ─────────────────────────────────────────
def omp_profile(da):
    return da.isel({pdim: y_omp, zdim: z_idx}).values

x = ds_t["x"].values

vExB_A  = omp_profile(vExB_no_y)
vExB_B  = omp_profile(vExB_with_y)
vmag_e  = omp_profile(v_mag_e)
vmag_i  = omp_profile(v_mag_i)
phi_x   = omp_profile(phi)
Te_x    = omp_profile(Te)
Ti_x    = omp_profile(Ti)

# ── Plot ─────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
fig.suptitle(f"t={t_idx}, z={z_idx}, OMP (y={y_omp})  —  normalised units")

ax = axes[0, 0]
ax.plot(x, vExB_A, label=r"$v_{ExB}$ no y-derivs (STORM default)")
ax.plot(x, vExB_B, "--", label=r"$v_{ExB}$ with y-derivs (Hermes-3)")
ax.axvline(ixseps1, color="k", ls="--", lw=0.8, label="sep")
ax.set_xlabel("x index"); ax.set_ylabel(r"$v_{ExB,r}$ [norm]")
ax.set_title("ExB radial velocity"); ax.legend(fontsize=8)

ax = axes[0, 1]
ax.plot(x, vExB_B - vExB_A, color="tab:orange")
ax.axvline(ixseps1, color="k", ls="--", lw=0.8)
ax.set_xlabel("x index"); ax.set_ylabel(r"$\Delta v_{ExB,r}$")
ax.set_title("Difference: with\_y $-$ no\_y")

ax = axes[1, 0]
ax.plot(x, vmag_e, color="tab:red",  label=r"$v_{mag,e}$ (electrons, $q=-e$)")
ax.plot(x, vmag_i, color="tab:blue", label=rf"$v_{{mag,i}}$ ({ion_species}, $q=+e$)")
ax.plot(x, vmag_e + vmag_i, "k--", lw=1, label=r"$v_{mag,e}+v_{mag,i}$ $\propto T_i-T_e$")
ax.axvline(ixseps1, color="k", ls="--", lw=0.8)
ax.set_xlabel("x index"); ax.set_ylabel(r"$v_{mag,r}$ [norm]")
ax.set_title("Magnetic drift (curvature/grad-B) — opposite signs for e/i")
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(x, phi_x, label=r"$\phi$ [norm]")
ax.plot(x, Te_x,  label=r"$T_e$ [norm]")
ax.plot(x, Ti_x,  label=rf"$T_i$ ({ion_species}) [norm]")
ax.axvline(ixseps1, color="k", ls="--", lw=0.8)
ax.set_xlabel("x index"); ax.legend(fontsize=8)
ax.set_title(r"$\phi$, $T_e$, $T_i$ at OMP")

fig.tight_layout()
plt.show()

# ── Summary at separatrix ─────────────────────────────────────────────────────
print(f"\n── Values at separatrix (x={ixseps1}) ──")
print(f"  v_ExB (no y-derivs)   = {vExB_A[ixseps1]:.4f}")
print(f"  v_ExB (with y-derivs) = {vExB_B[ixseps1]:.4f}")
print(f"  v_mag_e               = {vmag_e[ixseps1]:.4f}")
print(f"  v_mag_i               = {vmag_i[ixseps1]:.4f}")
print(f"  v_mag_e + v_mag_i     = {(vmag_e+vmag_i)[ixseps1]:.4f}  (net, propto T_i-T_e)")
if vExB_A[ixseps1] != 0:
    print(f"  |v_mag_e / v_ExB|     = {abs(vmag_e[ixseps1]/vExB_A[ixseps1]):.4f}")
    print(f"  |v_mag_i / v_ExB|     = {abs(vmag_i[ixseps1]/vExB_A[ixseps1]):.4f}")
else:
    print("  (v_ExB = 0 at separatrix)")

#!/usr/bin/env python3
"""
Conservation closure diagnostic for a Hermes-3 3D turbulence simulation.

In steady state:  Source  =  LCFS flux  +  d(core inventory)/dt
If d/dt > 0 the simulation hasn't relaxed yet; the "missing" part of the
source is being absorbed into the core inventory and the balance only closes
once that storage term is included.

For the chosen species this script reports

    S_part = ∫ S_n   dV                 (electron particle source)
    F_part = ∮ Γ_r   dA   at  x=ixseps1 (ExB + mag-drift + anomalous + flutter)
    dN/dt  = ΔN_core / Δt               (loaded time window)
    closure = (F + dN/dt) / S

and the analogous quantities for energy: the source is 3/2·Pe_src (Hermes-3
evolves the PRESSURE, the internal energy is 3/2 P), the flux is the 5/2
enthalpy flux, and the storage is 3/2 d∫P dV/dt. All three factors are needed
for the closure to mean anything. It also prints a geometry sanity check (torus
volume, LCFS area) in physical SI by rescaling with rho_s0, and plots the
per-frame LCFS flux to expose transients and statistical scatter.

Edit the CONFIG block below and run.
"""

import gc
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from hermes3_turb.load import _no_t, get_parallel_velocity
from xhermes import open_hermesdataset


# ── Cleanup from any previous %run in the same IPython session ───────────────
# Stale netCDF/HDF5 file handles from a previous load cause segfaults when
# xhermes/h5netcdf re-opens the same files; explicitly close and garbage-
# collect anything left behind before loading again.
plt.close("all")
for _name in ("ds", "datasets", "ds_i", "n", "P", "phi", "Ne", "Ni", "Pe", "Pi",
              "Te_eV", "Ti_eV", "v_par", "Gamma_r", "q_r",
              "G_per_t", "G_lcfs_t", "N_core_t", "U_core_t"):
    _v = globals().pop(_name, None)
    if _v is None:
        continue
    if hasattr(_v, "close"):
        try:
            _v.close()
        except Exception:
            pass
    elif isinstance(_v, list):
        for _d in _v:
            if hasattr(_d, "close"):
                try:
                    _d.close()
                except Exception:
                    pass
    del _v
gc.collect()


# ── CONFIG ────────────────────────────────────────────────────────────────────
data_root    = "./"
gridfile     = data_root+"../exp45470_x132_y28.nc"
dir_range    = range(140, 141)    # end exclusive
keep_yguards = False
species      = "e"
include_y_derivs = True           # full curvilinear ExB (matches Hermes-3); False = xstorm convention
# ─────────────────────────────────────────────────────────────────────────────
dir_range = h3t.existing_dirs(data_root, dir_range, "boutdata.nc")


# ── Load ──────────────────────────────────────────────────────────────────────
datasets = []
for i, dirnum in enumerate(dir_range):
    print(f"loading {data_root}{dirnum}")
    ds_i = open_hermesdataset(
        f"{data_root}/{dirnum}/boutdata.nc",
        geometry="toroidal",
        inputfilepath=f"{data_root}/{dirnum}/BOUT.inp",   # in case/data/ alongside the script
        gridfilepath=gridfile,
        keep_xboundaries=True,
        keep_yboundaries=keep_yguards,
        unnormalise=False,
        info=False,
    )
    datasets.append(ds_i)
# NB do NOT drop each folder's first frame: that assumes BOUT++'s default
# dump_at_restart=true, and with dump_at_restart=false the first frame is a
# genuinely NEW time, so the positional rule deletes real data (and a folder
# squashed to a single output time contributes nothing at all). Restart
# overlap is removed after concatenation by time VALUE -- see
# load.dedup_restart_times, which handles both settings and mixed ones.
ds = xr.concat(datasets, dim="t")
ds = h3t.dedup_restart_times(ds)

zdim = h3t.detect_toroidal_dim(ds)
pdim = h3t.detect_poloidal_dim(ds)

# Remove retained y-guards before poloidal derivatives.
# xBOUT derivative coordinates require the interior poloidal layout.
import netCDF4 as _nc_g
with _nc_g.Dataset(gridfile) as _g0:
    _ny_g = int(_g0["ny"][:])
# The guard count per band is derived from the dump (npol - ny), NOT from
# MYG: a grid with y_boundary_guards=1 gives ny+4 rows while MYG reports
# 2, so a `npol == ny + 4*MYG` test matches neither layout and strips
# nothing. One implementation, in h3t.guard_rows.
_npol = ds.sizes[pdim]
_guard_rows = h3t.guard_rows(ds, ny_grid=_ny_g)
if _guard_rows:
    _keep_rows = [i for i in range(_npol) if i not in set(_guard_rows)]
    print(f"  dataset kept y-guards despite keep_yguards={keep_yguards} "
          f"(grid y_boundary_guards>0): dropping rows {_guard_rows}, "
          f"{_npol} -> {len(_keep_rows)} poloidal cells")
    ds = ds.isel({pdim: _keep_rows})
m    = ds.metadata
print(f"Loaded {ds.sizes['t']} time steps")


# ── Topology ─────────────────────────────────────────────────────────────────
# Robust MYG detection: metadata can report MYG=2 even when guards are stripped.
import netCDF4 as _nc
with _nc.Dataset(gridfile) as _grd:
    _ny_grid = int(_grd["ny"][:])
_has_yguards = ds.sizes[pdim] > _ny_grid
MYG = int(m.get("MYG", 2)) if _has_yguards else 0
MXG = int(m.get("MXG", 2))

ixsep1 = m.get("ixseps1", ds.sizes["x"] // 2)
j1_1   = m.get("jyseps1_1", MYG) + MYG + 1
j2_1   = m.get("jyseps2_1", ds.sizes[pdim] // 4) + MYG + 1
j1_2   = m.get("jyseps1_2", ds.sizes[pdim] // 2) + MYG * 3 + 1
j2_2   = m.get("jyseps2_2", 3 * ds.sizes[pdim] // 4) + MYG * 3 + 1

inner_sol = slice(j1_1, j2_1)
outer_sol = slice(j1_2, j2_2)
print(f"ixseps1={ixsep1}, inner SOL=[{j1_1}:{j2_1}], outer SOL=[{j1_2}:{j2_2}]")


# ── Field accessors ──────────────────────────────────────────────────────────
n_name = f"N{species}"
P_name = f"P{species}"
src_n_name = f"S{species}_src"
src_P_name = f"P{species}_src"

is_electron = (species == "e")
ion_species = "i" if "Ni" in ds.data_vars else "d+"

n   = ds[n_name]
P   = ds[P_name]
phi = ds["phi"]
Ne  = ds["Ne"]; Ni = ds[f"N{ion_species}"]
Pe  = ds["Pe"]; Pi = ds[f"P{ion_species}"]
# Te/Ti are in the dump only if the input asks for them; a folder that
# did not leaves whole NaN frames after the concat (nx32 dirs 23-28:
# folder 23 has no Ti), so fill those frames from P/N.
Te_eV = Pe / Ne.clip(min=1e-10)
Ti_eV = Pi / Ni.clip(min=1e-10)
if "Te" in ds:
    Te_eV = ds["Te"].fillna(Te_eV)
if f"T{ion_species}" in ds:
    Ti_eV = ds[f"T{ion_species}"].fillna(Ti_eV)
v_par = get_parallel_velocity(ds, species)

J, dx, dy = (_no_t(ds[k]) for k in ("J", "dx", "dy"))
elem = h3t.radial_area_element_dtheta(ds)


# ── Total radial fluxes (all channels, time-mean) ────────────────────────────
def tmean(da):
    return da.mean("t") if "t" in getattr(da, "dims", ()) else da

print("Computing radial fluxes (ExB + mag-drift + anomalous + flutter) ...")
_, _, Gamma_ExB = h3t.decompose_radial_particle_flux(ds, n, phi, include_y_derivs=include_y_derivs)
_, _, q_ExB     = h3t.decompose_radial_heat_flux(ds, P, phi, include_y_derivs=include_y_derivs)

grid_vars = h3t.load_grid_vars(gridfile, ds=ds, pdim=pdim)
bxcvx, Bxy = grid_vars["bxcvx"], grid_vars["Bxy"]
Gmag_e, Gmag_i    = h3t.radial_magnetic_drift_particle_flux(ds, Ne, Ni, bxcvx, Bxy, Te_eV, Ti_eV)
qmag_e, qmag_i, _ = h3t.radial_magnetic_drift_heat_flux(ds, Pe, Pi, bxcvx, Bxy, Te_eV=Te_eV, Ti_eV=Ti_eV)
Gamma_mag = tmean(Gmag_e if is_electron else Gmag_i)
q_mag     = tmean(qmag_e if is_electron else qmag_i)

# Anomalous chi/D from BOUT.inp (re-normalised to match unnormalise=False fields).
def _opt(section, key, default=0.0):
    try:
        return float(ds.options[section][key])
    except (KeyError, TypeError, AttributeError):
        return default
chi_s = _opt(P_name, "anomalous_chi") or _opt(species, "anomalous_chi")
D_s   = _opt(n_name, "anomalous_D")   or _opt(species, "anomalous_D")
_dn   = float(m.get("rho_s0", 1.0))**2 * float(m.get("Omega_ci", 1.0))
chi_n = chi_s / _dn if _dn > 0 else chi_s
D_n   = D_s   / _dn if _dn > 0 else D_s
T_s   = Te_eV if is_electron else Ti_eV
Gamma_anom = tmean(h3t.radial_anomalous_particle_flux(ds, n, D_n)) if D_s > 0 else 0.0
q_anom     = tmean(h3t.radial_anomalous_heat_flux(ds, n, T_s, chi_n, D=D_n)) if (chi_s > 0 or D_s > 0) else 0.0

try:
    _flutter = str(ds.options["electromagnetic"]["magnetic_flutter"]).lower() == "true"
except (KeyError, TypeError, AttributeError):
    _flutter = False
if _flutter and "Apar" in ds:
    Ve = get_parallel_velocity(ds, "e"); Vi = get_parallel_velocity(ds, ion_species)
    Gf, qf = h3t.radial_flutter_fluxes(ds, Ne, Pe, Ve, Pi, Vi, ds["Apar"])
    Gamma_flutter, q_flutter = tmean(Gf), tmean(qf)
else:
    Gamma_flutter, q_flutter = 0.0, 0.0

Gamma_r = Gamma_ExB + Gamma_mag + Gamma_anom + Gamma_flutter
q_r     = q_ExB     + q_mag     + q_anom     + q_flutter


def lcfs_flux(F):
    """Integrate a 2-D (x,θ) toroidal-mean flux density over the LCFS poloidal
    range (HFS main SOL + LFS main SOL), full torus."""
    integrand = (F * elem).isel(x=ixsep1)
    r = integrand.isel({pdim: inner_sol}).sum(pdim) + integrand.isel({pdim: outer_sol}).sum(pdim)
    leftover = [d for d in r.dims if d not in ("x",)]
    if leftover:
        r = r.mean(dim=leftover)
    return float(r) * 2 * np.pi


F_part = lcfs_flux(Gamma_r)
F_eng  = lcfs_flux(q_r)


# ── Volume-integrated sources ────────────────────────────────────────────────
# `energy_source` carries the 3/2 (Hermes-3 evolves P; the internal energy is
# 3/2 P), so S_eng is comparable with the 5/2 enthalpy flux and the 3/2
# storage term. `volume_integral` sums interior cells only (x- and y-guards
# dropped).
_src_n = h3t.particle_source(ds, species)
_src_q = h3t.energy_source(ds, species)          # = 3/2 * P<sp>_src
if _src_n is None or _src_q is None:
    _missing = [nm for nm, v in ((src_n_name, _src_n), (src_P_name, _src_q))
                if v is None]
    print(f"\n  !! WARNING: {', '.join(_missing)} not in the dump for "
          f"species '{species}'. The corresponding source is taken as 0, so "
          f"its closure is meaningless, not 'bad'. Save the source(s) or pick "
          f"a species the dump carries "
          f"(present: {sorted(v for v in ds.data_vars if v.endswith('_src'))}).")
S_part = float(h3t.volume_integral(ds, _src_n)) if _src_n is not None else 0.0
S_eng  = float(h3t.volume_integral(ds, _src_q)) if _src_q is not None else 0.0


# ── Inventory time series (closed-core volume: x < ixseps1, all θ) ───────────
# Storage of the entire core (PFR contributions at x<ixseps1 in the legs are
# small; restricting to y∈(j1_1,j2_2) would tighten this but rarely changes
# the answer at the % level for typical DN setups).
core_x = slice(MXG, ixsep1)

def inventory_t(density):
    """Volume-summed (closed-core x-range), toroidal-mean. Returns 1-D in t."""
    fz = density.mean(zdim) if zdim in density.dims else density
    return (fz * J * dx * dy).isel(x=core_x).sum(("x", pdim)) * 2 * np.pi

N_core_t = inventory_t(n)
U_core_t = inventory_t(1.5 * P)   # internal energy density = 3/2 P


def inventory_rate(times, inventory):
    """Endpoint storage rate, or NaN without a finite increasing time interval."""
    times = np.asarray(times, dtype=float)
    values = np.asarray(inventory, dtype=float)
    if times.ndim != 1 or values.shape != times.shape:
        raise ValueError("Inventory must have one value per output time")
    if (times.size < 2 or not np.all(np.isfinite(times))
            or not np.all(np.diff(times) > 0)
            or not np.isfinite(values[[0, -1]]).all()):
        return float("nan")
    return float((values[-1] - values[0]) / (times[-1] - times[0]))


def closure_percent(source, flux, storage):
    """Closure is undefined without a nonzero source and a valid storage rate."""
    if source == 0 or not np.isfinite([source, flux, storage]).all():
        return float("nan")
    return (flux + storage) / source * 100


def closure_label(value):
    return f"{value:.1f}%" if np.isfinite(value) else "unavailable"


t  = ds["t"].values
dt = float(t[-1] - t[0]) if t.size > 1 else 0.0
dN_dt = inventory_rate(t, N_core_t.values)
dU_dt = inventory_rate(t, U_core_t.values)
if not np.isfinite([dN_dt, dU_dt]).all():
    print("Storage rates/closure unavailable: need at least two finite, "
          "strictly increasing output times and finite endpoint inventories.")


# ── Per-frame LCFS ExB particle flux (transient / noise diagnostic) ──────────
v_r = h3t.ExB_radial_velocity(ds, phi, include_y_derivs=include_y_derivs)
G_per_t = (n * v_r).mean(zdim)                              # (t, x, θ)
elem_lcfs = elem.isel(x=ixsep1)
G_lcfs_t = ((G_per_t.isel(x=ixsep1) * elem_lcfs).isel({pdim: inner_sol}).sum(pdim)
          + (G_per_t.isel(x=ixsep1) * elem_lcfs).isel({pdim: outer_sol}).sum(pdim)) * 2 * np.pi
# collapse any leftover dims (e.g. residual zeta on metric)
_extra = [d for d in G_lcfs_t.dims if d != "t"]
if _extra:
    G_lcfs_t = G_lcfs_t.mean(dim=_extra)


# ── Print closure table ──────────────────────────────────────────────────────
def print_row(label, S, F, dInv_dt):
    sumFI   = F + dInv_dt
    closure = closure_percent(S, F, dInv_dt)
    print(f"  {label:8s}  S={S:.3e}   F={F:.3e}   dInv/dt={dInv_dt:.3e}   "
          f"F+dInv/dt={sumFI:.3e}   closure={closure_label(closure)}")

print(f"\n===== Conservation closure  (species '{species}', Δt = {dt:.3e}) =====")
print_row("particle", S_part, F_part,       dN_dt)
print_row("energy",   S_eng,  F_eng,        dU_dt)
print("  (closure → 100% in steady state; if <100% with dInv/dt > 0, the core is")
print("   still filling — flux has not yet caught up with the source)")


# Geometry in SI: volume excludes guards; LCFS area excludes divertor legs.
rho_s0 = float(m.get("rho_s0", 1.0))
_interior = h3t.interior_mask(ds)                 # (x, θ), guards False
_dV = (J * dx * dy).where(xr.DataArray(_interior, dims=("x", pdim)), other=0.0)
V_norm = float(_dV.sum(("x", pdim))) * 2 * np.pi

def _area_at(x_idx, rows):
    """Flux-surface area per 2π at x=x_idx over the given poloidal rows."""
    a = elem.isel(x=x_idx)
    a = a.isel({pdim: list(rows)}) if rows is not None else a
    a = a.sum(pdim)
    _left = list(a.dims)
    if _left:
        a = a.mean(dim=_left)
    return float(a) * 2 * np.pi

_bands = h3t.sol_band_regions(ds)                 # {name: (y0, y1)}, legs excluded
_lcfs_rows = [j for y0, y1 in _bands.values() for j in range(y0, y1)]
A_norm     = _area_at(ixsep1, _lcfs_rows)         # true LCFS: HFS + LFS bands
A_all_norm = _area_at(ixsep1, None)               # every θ, legs included
print(f"\n===== Geometry sanity =====")
print(f"  rho_s0  = {rho_s0:.3e} m   (from ds.metadata)")
print(f"  V_torus = {V_norm:.3e} (normalised)  →  {V_norm * rho_s0**3:.3f} m³"
      f"   (interior cells only)")
print(f"  A_LCFS  = {A_norm:.3e} (normalised)  →  {A_norm * rho_s0**2:.3f} m²"
      f"   ({', '.join(f'{k} {y0}:{y1}' for k, (y0, y1) in _bands.items())})")
print(f"  A(x=ixseps1), all θ = {A_all_norm:.3e}  →  "
      f"{A_all_norm * rho_s0**2:.3f} m²   (adds the four divertor legs — "
      f"NOT the LCFS area)")


# ── Plots ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
fig.suptitle(f"Conservation diagnostic — species '{species}'  "
             f"(closure: particle {closure_label(closure_percent(S_part, F_part, dN_dt))}, "
             f"energy {closure_label(closure_percent(S_eng, F_eng, dU_dt))})")

ax = axes[0]
ax.plot(t, G_lcfs_t.values, "o-", label="ExB flux at LCFS, per frame")
ax.axhline(S_part, ls="--", color="red", label=f"source = {S_part:.2e}")
ax.axhline(float(G_lcfs_t.mean()), ls=":", color="gray",
           label=f"mean = {float(G_lcfs_t.mean()):.2e}")
ax.set_xlabel("t [normalised]")
ax.set_ylabel("particle rate")
ax.set_title("LCFS ExB particle flux per frame")
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(t, N_core_t.values, "o-", label="N_core(t)")
ax.set_xlabel("t [normalised]")
ax.set_ylabel("N_core (normalised)")
ax.set_title(f"Core particle inventory  (dN/dt = {dN_dt:.2e})")
ax.legend(fontsize=8)

ax = axes[2]
ax.plot(t, U_core_t.values, "o-", label="U_core(t) = 3/2 ∫P dV")
ax.set_xlabel("t [normalised]")
ax.set_ylabel("U_core (normalised)")
ax.set_title(f"Core energy inventory  (dU/dt = {dU_dt:.2e})")
ax.legend(fontsize=8)

fig.tight_layout()
plt.show()

#!/usr/bin/env python3
"""
Conservation check: target sheath heat flux vs. poloidal heat flux near the
target.

Physics
-------
With no particle/energy source or sink in the divertor leg, the heat flowing
poloidally toward a target plate must be deposited on it (steady state). So the
poloidal power flowing through cross-sections just upstream of the plate should
(a) be roughly constant from cell to cell and (b) equal the power the target
heat-flux model deposits. This is a direct cross-check of the target machinery
(plate-index detection, sheath interpolation, Bp/B projection, area element).

Three measures of the same near-target power are compared:

  1. Target SHEATH power  — Σ_s γ_s P_s |V_∥s| (Bp/B), integrated over the plate
     (hermes3_turb.target). γ_s carries conduction + kinetic + sheath-potential
     energy on top of convection.
  2. CONVECTIVE poloidal power — Σ_s (5/2) P_s V_∥s (Bp/B), integrated over the
     poloidal surface at each near-target θ slice. Uses the identity
     v_pol = v_∥·(Bp/B) (so this is robust to the metric normalisation, unlike
     parallel_to_poloidal under unnormalise=True). This captures *only* the
     convective enthalpy channel — expect it to be a fraction (~2.5/γ) of (1).
  3. Hermes-3 native poloidal power — `efe_tot_ylow` (+ ion equivalent if the
     ion species was run with `diagnose = true`), the code's own Y-face energy
     flow including conduction. This is the most complete and should match (1).

NOTE this is the most reliable as a *near-target* check only: upstream of the
last few cells, turbulent radial transport removes power from the flux bundle,
so the poloidal power is not flat far from the plate. A single time slice is
also noisy — average over a saturated turbulence window when possible.

Usage
-----
    python target_poloidal_check.py <casepath> [options in CONFIG block]

Loads a squashed boutdata.nc (or BOUT.dmp.*.nc) that **retains y-boundary
guard cells** (keep_yboundaries=True). No grid file is needed (and a guardless
grid will not line up with a guard-retaining dump under geometry="toroidal", so
geometry processing is skipped — all metric/Bp/B come from the dump).
"""

import os
import sys
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xhermes import open_hermesdataset
import hermes3_turb as h3t
from hermes3_turb.target import (
    get_target_boundaries,
    field_line_incidence,
    target_heat_flux_density,
    total_target_power,
)

# ============================================================
# CONFIG
# ============================================================
#casepath   = sys.argv[1] if len(sys.argv) > 1 else "."
casepath = './140/'
#casepath = '/home/qixia/remote_archer2/tmp_storage/data/300/'
ion_species = "i"
AA_ion      = 2.0      # ion mass number (for the ion kinetic-energy flux term)
n_upstream  = 6        # number of cells to profile upstream of each plate
include_conduction = True   # add parallel electron conduction (κ_∥e ∂T/∂s).
                            # Reconstructed by finite difference, so it is
                            # sensitive to steep/odd T near crashes — disable
                            # for noisy runs.
include_ion_kinetic = True  # add the ion kinetic-energy flux ½ m_i n_i V_∥i³
                            # (Helander 2.14) to the total at every cell of the
                            # upstream sweep. The M_∥ diagnostic below flags
                            # cells where the time-averaged ⟨V_∥³⟩ has been
                            # inflated by turbulent V-bursts (apparent M > 1
                            # at the cell centre is a numerical artefact, not
                            # the physical Bohm condition — Bohm is at the
                            # sheath FACE between i_dom and i_gc, not at i_dom
                            # itself).
bohm_cap_v = False    # DEPRECATED. Wrong direction — Hermes-3 enforces
                      # |V_∥| ≥ c_s (floor, not cap). Setting True is a no-op
                      # and emits a DeprecationWarning. Apparent M > 1 in the
                      # M_∥ diagnostic is either physical (sheath_boundary_simple
                      # with supersonic interior) or a Favre artefact — see
                      # the diagnostic print to judge.
use_phi_gamma = True  # sheath γ from φ (eqs. 12a/12b) vs FIXED γ_e = γ_i = 3.5
                      # (the fixed values match sheath_boundary_simple's defaults).
                       # φ-based γ_e is the exact Hermes-3 sheath_boundary form
                       # (γ_e = 2 + (φ_sheath−φ_wall)/T_e); it only "blows up"
                       # when the data itself has large φ/T_e (e.g. near a crash).
phi_wall    = 0.0      # matches Hermes-3 sheath_boundary 'wall_potential' default
# ============================================================

# ── Load (SI; no geometry so guard-retaining dumps load without a matching grid)
print(f"Loading: {casepath}")
if os.path.isdir(casepath):
    #_nc = os.path.join(casepath, "boutdata.nc")
    _nc = os.path.join(casepath, "boutdata_guards.nc")
    datapath = _nc if os.path.exists(_nc) else os.path.join(casepath, "BOUT.dmp.*.nc")
else:
    datapath = casepath
_data_dir = os.path.dirname(os.path.abspath(casepath))
_settings = next((p for p in (os.path.join(_data_dir, "BOUT.settings"),
                              os.path.join(casepath, "BOUT.settings"),
                              os.path.join(casepath, "BOUT.inp"))
                  if os.path.exists(p)), None)

ds = open_hermesdataset(datapath, inputfilepath=_settings, keep_yboundaries=True)

zdim = h3t.detect_toroidal_dim(ds)
pdim = h3t.detect_poloidal_dim(ds)
zperiod = int(ds.metadata.get("zperiod", 1))

boundaries, has_guards = get_target_boundaries(ds)
if not has_guards:
    print("WARNING: no y-boundary guard cells — plate interpolation falls back "
          "to the last domain cell.")

# Time-average over all loaded frames (assumed saturated; pre-trim before squashing)
print(f"Averaging over all {ds.sizes['t']} loaded frame(s)")
dsa = ds.mean("t")

Pe = dsa["Pe"]
Pi = dsa[f"P{ion_species}"]
Ve = h3t.get_parallel_velocity(dsa, "e")
Vi = h3t.get_parallel_velocity(dsa, ion_species)
# NV/N can be ±inf where N→0 in far guard corners; neutralise (off the plate)
Ve = Ve.where(np.isfinite(Ve), 0.0)
Vi = Vi.where(np.isfinite(Vi), 0.0)
phi = dsa["phi"] if use_phi_gamma else None

# ── Poloidal heat flux FROM DEFINITIONS (no efe_tot_ylow needed) ──────────────
# Total parallel heat flux toward the plate, projected to poloidal by ×(Bp/B):
#   advection   (5/2) P_s V_∥s                         (both species)
#   conduction  −κ_∥e ∂T_e/∂s_∥,  s_∥ = √g_22 · y      (electron; opt.)
#   ion kinetic  ½ m_i n_i V_∥i³                        (opt.)
# matching the channels in v3_poloidal_flux.py (which uses the advective part)
# plus the conduction that the sheath γ carries. Computed for both species so
# no `diagnose = true` / efe_*_ylow diagnostic is required.
from hermes3_turb.load import _no_t
_mp = 1.673e-27
MXG = int(ds.metadata.get("MXG", 2))
nx  = ds.sizes["x"]
_xmask = xr.DataArray((np.arange(nx) >= MXG) & (np.arange(nx) < nx - MXG), dims="x")
g22 = _no_t(dsa["g_22"]); dyf = _no_t(dsa["dy"])
Ni  = dsa[f"N{ion_species}"] if f"N{ion_species}" in dsa else None
# Conduction needs the parallel gradient in field-aligned coords (shifted
# parallel transform) — align T and κ_∥ first, else ∂T/∂y has spurious jumps.
Te = kappa_e = None
if include_conduction and "Te" in dsa and "kappa_par_e" in dsa:
    Te = h3t.to_field_aligned(dsa["Te"].clip(min=0.1), dsa)   # floor tames crash T
    kappa_e = h3t.to_field_aligned(dsa["kappa_par_e"], dsa)

# Optional cross-check against Hermes-3's own Y-face energy flow (if present).
# efe_tot_ylow is the **electron** Y-face flow — for a fair apples-to-apples
# comparison with the e+i `adv` line we need the ion side too. Look for the
# matching ion diagnostic (efi_tot_ylow / efd+_tot_ylow / …); add if found.
efe = dsa.get("efe_tot_ylow", None)
efi_name = next((k for k in (f"ef{ion_species}_tot_ylow",
                             "efi_tot_ylow", "efd+_tot_ylow")
                 if k in dsa.data_vars), None)
efi = dsa.get(efi_name) if efi_name else None
have_native = efe is not None
native_label = "efe_tot_ylow (e)"
if efi is not None:
    native_label = f"ef_tot_ylow (e+{ion_species})"
    print(f"native cross-check includes ion diagnostic '{efi_name}'")
elif efe is not None:
    print("native cross-check uses 'efe_tot_ylow' only — no matching ion "
          "ef*_tot_ylow found; the green line is electron-only and will sit "
          "below the e+i 'adv' line by ~½ in a Te≈Ti plasma. "
          "Set diagnose=true on the ion species to get the full comparison.")
_bp = {}
def bp(y):
    if y not in _bp:
        _bp[y] = field_line_incidence(dsa, y, pdim=pdim)
    return _bp[y]


def _integrate(q_xz, y):
    """Toroidal-mean a (x,ζ) density, drop x-guards, integrate over the
    poloidal surface at y → power [W], full torus."""
    q = h3t.toroidal_mean(q_xz, dim=zdim).where(_xmask, 0.0)
    return float(total_target_power(dsa, q, y))


def P_adv(y):
    """Advective poloidal power (e+i) [W] at poloidal index y."""
    q = 2.5 * (Pe.isel({pdim: y}) * Ve.isel({pdim: y})
               + Pi.isel({pdim: y}) * Vi.isel({pdim: y})) * bp(y)
    return _integrate(q, y)


def P_cond(y):
    """Electron parallel-conduction poloidal power [W] at y (field-aligned,
    centred ∂T/∂s_∥). Unreliable across X-points / DN inter-region guards."""
    if Te is None or kappa_e is None:
        return np.nan
    if not (0 < y < ds.sizes[pdim] - 1):
        return np.nan
    dTds = (Te.isel({pdim: y + 1}) - Te.isel({pdim: y - 1})) \
        / (2.0 * np.sqrt(g22) * dyf).isel({pdim: y})
    q = -kappa_e.isel({pdim: y}) * dTds * bp(y)
    return _integrate(q, y)


def q_pol_profile(y):
    """Radial profile of the total poloidal heat flux density [W/m²] at
    poloidal index y, toroidal-mean.

      q_pol(x) = [(5/2)(P_e V_e + P_i V_i) − κ_∥e ∂T_e/∂s_∥ + ½ m_i n_i V_∥i³] · (Bp/B)

    The signed contributions are summed (no abs), then toroidal-averaged and
    returned as a 1-D array along x. Use np.abs() on output if you only care
    about magnitude. Returns NaN where the cell would require y±1 outside the
    domain (last/first cell)."""
    if not (0 <= y < ds.sizes[pdim]):
        return np.full(nx, np.nan)
    bpy = bp(y)
    q_adv = 2.5 * (Pe.isel({pdim: y}) * Ve.isel({pdim: y})
                   + Pi.isel({pdim: y}) * Vi.isel({pdim: y})) * bpy
    q_total = q_adv
    if include_conduction and Te is not None and kappa_e is not None \
       and 0 < y < ds.sizes[pdim] - 1:
        dTds = (Te.isel({pdim: y + 1}) - Te.isel({pdim: y - 1})) \
            / (2.0 * np.sqrt(g22) * dyf).isel({pdim: y})
        q_total = q_total + (-kappa_e.isel({pdim: y}) * dTds * bpy)
    if include_ion_kinetic and Ni is not None:
        q_total = q_total + _ion_kin_density(y) * bpy
    return h3t.toroidal_mean(q_total, dim=zdim).where(_xmask, np.nan).values


def _ion_kin_density(y):
    """Ion parallel kinetic-energy flux density [W/m²] before Bp/B projection.

    Returns ½ m_i n_i V_∥i³ from the time-averaged data (Helander 2.14, direct
    reconstruction from ⟨NV/N⟩³). No Bohm cap is applied — Bohm sets the
    velocity at the sheath FACE (between i_dom and i_gc), not at the cell
    centre i_dom, so a cap on the cell-centre value would only mask the
    time-averaging artefact, not enforce physics. The M_∥ sanity print
    flags those artefacts so the user can interpret the kin column.
    """
    Vi_y = Vi.isel({pdim: y})
    return 0.5 * AA_ion * _mp * Ni.isel({pdim: y}) * Vi_y ** 3


def P_kin(y):
    """Ion kinetic-energy poloidal power [W] at y."""
    if not include_ion_kinetic or Ni is None:
        return np.nan
    q = _ion_kin_density(y) * bp(y)
    return _integrate(q, y)


def _sum_face(da, y):
    v = da.isel({pdim: y}).where(_xmask).values
    return float(np.where(np.isfinite(v), v, 0.0).sum())

def P_native(y):
    """Σ_{x,ζ}|ef*_tot_ylow| at face y (e + i if both available; cross-check
    against the sheath. Matches the e+i `adv` line when both species are
    diagnosed)."""
    if not have_native:
        return np.nan
    tot = _sum_face(efe, y)
    if efi is not None:
        tot = tot + _sum_face(efi, y)
    return abs(tot) * zperiod


targets = [t for t in ("outer_lower", "inner_lower", "outer_upper", "inner_upper")
           if t in boundaries]

# Guarded X-point (jyseps) index bounding each divertor leg, so the upstream
# profile stops at the X-point instead of marching across the branch cut into
# the main SOL (where the field-aligned shift is invalid and conduction spikes).
_m = ds.metadata
_g = int(_m.get("MYG", 2)) if has_guards else 0
_is_dn = _m.get("ny_inner", None) is not None and int(_m.get("ny_inner", 0)) > 0
if _is_dn:                                  # region A: +MYG, region B: +3·MYG
    xpt = {"inner_lower": int(_m["jyseps1_1"]) + _g,
           "inner_upper": int(_m["jyseps2_1"]) + _g,
           "outer_upper": int(_m["jyseps1_2"]) + 3 * _g,
           "outer_lower": int(_m["jyseps2_2"]) + 3 * _g}
else:
    xpt = {"inner_lower": int(_m["jyseps1_1"]) + _g,
           "outer_lower": int(_m["jyseps2_2"]) + _g}

# ── M_∥ diagnostic at the last plasma cell ───────────────────────────────────
# Bohm's criterion |V_∥| ≥ c_s applies at the SHEATH FACE (between i_dom and
# i_gc), where Hermes-3 enforces it via the sheath BC. The CELL-CENTRE i_dom
# sits half a cell upstream of that face and is physically subsonic.
# This diagnostic reports the time-averaged ⟨NV/N⟩/c_s at the cell centre —
# values > 1 are a numerical artefact of averaging (correlation between N and
# V fluctuations inflates ⟨NV/N⟩³ relative to ⟨V⟩³), not a violation of Bohm.
# Use it to know how much to trust the `kin` column of the upstream table.
print("\nM_∥ diagnostic at i_dom (toroidal+x-mean over SOL x range; cell centre,"
      " not the sheath face — values >1 indicate ⟨V³⟩ inflation, not violation"
      " of Bohm):")
_Ti = dsa[f"T{ion_species}"].fillna(Pi / Ni.clip(min=1e-10)) if f"T{ion_species}" in dsa else Pi / Ni.clip(min=1e-10)
_Te = dsa["Te"] if "Te" in dsa else Pe / dsa["Ne"].clip(min=1e-10)
for _tgt, (_idom, _) in boundaries.items():
    _Vi   = float(Vi.isel({pdim: _idom}).where(_xmask, np.nan).mean(skipna=True))
    _Tit  = float(_Ti.isel({pdim: _idom}).where(_xmask, np.nan).mean(skipna=True))
    _Tet  = float(_Te.isel({pdim: _idom}).where(_xmask, np.nan).mean(skipna=True))
    _cs   = np.sqrt(1.602e-19 * max(_Tet + _Tit, 1e-6) / (AA_ion * _mp))
    _M    = abs(_Vi) / _cs if _cs > 0 else float("nan")
    flag  = "" if _M <= 1.0 else "  ← ⟨V³⟩ inflated, kin over-estimated"
    print(f"  {_tgt:<14}  M_∥ = {_M:.2f}    "
          f"(Vi={_Vi:+.2e} m/s, T_e+T_i={_Tet+_Tit:.1f} eV){flag}")

print("\n" + "=" * 86)
print("Near-target poloidal power [kW] from definitions  (plate → upstream; "
      "x-guards excluded)")
print("=" * 86)

fig, axes = plt.subplots(1, len(targets), figsize=(5.5 * len(targets), 5), squeeze=False)
axes = axes[0]
summary = {}

for ax, tgt in zip(axes, targets):
    i_dom, i_gc = boundaries[tgt]
    # Upstream marches away from the plate (guard side); direction = sign(i_dom−i_gc).
    up = int(np.sign(i_dom - i_gc)) or 1
    # March from the plate up the leg, stopping BEFORE the X-point (branch cut).
    x_lim = xpt.get(tgt)
    ys = []
    y = i_dom
    for _ in range(n_upstream):
        if not (0 <= y < ds.sizes[pdim]):
            break
        ys.append(y)
        y_next = y + up
        if x_lim is not None and ((up < 0 and y_next <= x_lim)
                                  or (up > 0 and y_next >= x_lim)):
            break
        y = y_next

    adv  = np.array([P_adv(y)  for y in ys]) / 1e3
    cond = np.array([P_cond(y) for y in ys]) / 1e3
    kin  = np.array([P_kin(y)  for y in ys]) / 1e3
    nat  = np.array([P_native(y) for y in ys]) / 1e3
    # Total poloidal heat flux from definitions (deposited magnitude).
    tot  = np.abs(adv + np.nan_to_num(cond) + np.nan_to_num(kin))

    _, q_avg, _ = target_heat_flux_density(dsa, Pe, Pi, Ve, Vi, phi=phi,
                                           target=tgt, phi_wall=phi_wall,
                                           AA_ion=AA_ion,
                                           bohm_cap_v=bohm_cap_v)
    P_sheath = float(total_target_power(dsa, q_avg.where(_xmask, 0.0), i_dom)) / 1e3
    print(f"  γ_e implied = {P_sheath / (P_adv(i_dom)/1e3 * 2/5):.2f}  (compare 5 or 2 + φ/T)")

    summary[tgt] = dict(sheath=P_sheath, tot=tot[0], adv=abs(adv[0]),
                        nat=nat[0])
    print(f"\n{tgt}:  target SHEATH (γ, total) = {P_sheath:7.2f} kW")
    hdr = f"  {'y':>4}{'adv(e+i)':>11}{'cond(e)':>10}{'kin(i)':>9}{'TOTAL':>10}"
    if have_native:
        hdr += f"{native_label.split()[0]:>14}"
    print(hdr)
    for j, y in enumerate(ys):
        row = f"  {y:>4}{adv[j]:>11.2f}{cond[j]:>10.2f}{kin[j]:>9.2f}{tot[j]:>10.2f}"
        if have_native:
            row += f"{nat[j]:>14.2f}"
        print(row)

    d = np.arange(len(ys))
    ax.plot(d, np.abs(adv),               "o-", label="advection (5/2 PV)")
    if include_conduction:
        ax.plot(d, tot,                   "^-", label="adv+cond(+kin) total")
    if have_native:
        ax.plot(d, nat,                   "s-", label=native_label)
    ax.axhline(P_sheath, color="r", ls="--", label=f"target sheath = {P_sheath:.0f} kW")
    ax.set_xlabel("cells upstream of plate")
    ax.set_ylabel("poloidal power [kW]")
    ax.set_title(tgt.replace("_", " ").title())
    ax.legend(fontsize=8)

print("\n" + "=" * 86)
print(f"{'target':<14}{'sheath[kW]':>12}{'def-total[kW]':>14}{'advection[kW]':>14}"
      f"{'sheath/total':>14}")
print("-" * 86)
for tgt, s in summary.items():
    ratio = s["sheath"] / s["tot"] if s["tot"] else float("nan")
    print(f"{tgt:<14}{s['sheath']:>12.2f}{s['tot']:>14.2f}{s['adv']:>14.2f}"
          f"{ratio:>14.2f}")
print("=" * 86)

# ── Power balance: total target load vs input source power ───────────────────
# Input source = 3/2 · Σ(P_src · J·dx·dy) · 2π per species (same as
# inspect_hermes_run.py). In steady state Σ_targets ≈ Σ_input; a large mismatch
# flags a transient/crash, heating ≠ assumed, or an UNDER-COUNTED input (the ion
# pressure source is only saved when the ion species has diagnose=true).
tot_sheath = sum(s["sheath"] for s in summary.values())
tot_def    = sum(s["tot"]    for s in summary.values())
elem_vol   = _no_t(dsa["J"]) * _no_t(dsa["dx"]) * _no_t(dsa["dy"])     # m³/rad
P_in = {}
for sp, src in (("e", "Pe_src"), (ion_species, f"P{ion_species}_src")):
    if src in dsa:
        S = dsa[src].isel({zdim: 0}) if zdim in dsa[src].dims else dsa[src]
        P_in[sp] = float((1.5 * S * elem_vol * 2 * np.pi).where(_xmask, 0.0).sum())

print("\nPOWER BALANCE  (full torus)")
print(f"  total target load   : sheath {tot_sheath:8.1f} kW    def {tot_def:8.1f} kW")
if P_in:
    print("  input source power  : "
          + ", ".join(f"{k}={v/1e3:.1f}" for k, v in P_in.items()) + " kW"
          + f"   → total {sum(P_in.values())/1e3:8.1f} kW")
    print(f"  target / input      : {tot_sheath / (sum(P_in.values())/1e3):.2f}")
    missing = [f"P{s}_src" for s in ("e", ion_species) if f"P{s}_src" not in dsa]
    if missing:
        print(f"  WARNING: {missing} not saved (needs diagnose=true) → input "
              "UNDER-counts those species; target/input is then overstated.")
    print("  Residual (input − target) = dW/dt + e–i exchange + radial-wall loss;\n"
          "  target ≫ input ⇒ transient/crash, higher heating than assumed, or\n"
          "  under-counted input (see warning). Time-average a saturated window.")
else:
    print("  (no P*_src in the dump — cannot compute input power)")
print("=" * 86)
print("def-total = advection + electron conduction + ion kinetic (from definitions).")
print("Match to the target sheath power = energy conservation toward the plate.")
print("Advection alone is ~2.5/γ of the total; the rest is parallel conduction.")
if not (include_conduction):
    print("NOTE: conduction OFF — def-total is advection-only and will undershoot.")

fig.suptitle("Target vs poloidal heat-flux conservation check (from definitions)", y=1.02)
fig.tight_layout()

# ── Radial profile: target q⊥(x) vs poloidal q_pol(x) at i_dom and i_dom+up ─
# How the radial shape of the poloidal heat flux just upstream of the plate
# compares with the heat-flux profile the target actually sees. A perfect
# attached SOL would have q_pol(s) at the last cell ≈ q_target(s) (same s = OMP
# coordinate via flux-surface mapping isn't applied here — both are plotted vs
# the physical radial coord at the plate to keep the comparison local).
from hermes3_turb.target import target_physical_distance, fit_eich, fit_double_exp

# Eich / 2-exp fit window in the same coord as the radial profile (s from
# separatrix at the plate, [mm]). Tweak if the SOL extent of your case differs.
fit_s_range_mm = (-3, 30)
fit_s0         = False      # see hermes3_turb.target.fit_eich; True for shifted-peak cases

def _fit_lambdas(s_mm, q_arr, s_range, fit_s0=False):
    """Run Eich + double-exp on (s, q[MW/m²]); return concise λ-summary string."""
    mask = (s_mm >= s_range[0]) & (s_mm <= s_range[1]) & np.isfinite(q_arr) & (q_arr > 0)
    if mask.sum() < 4:
        return None, None, "λ-fit skipped: too few points"
    s_fit, q_fit = s_mm[mask], q_arr[mask]
    try:
        ep, _, s_fine, q_eich = fit_eich(s_fit, q_fit, fit_s0=fit_s0)
    except Exception as exc:
        ep, s_fine, q_eich = None, None, None
    try:
        dp, _, s_de, q_de = fit_double_exp(s_fit, q_fit, s_min=s_range[0], fit_s0=fit_s0)
    except Exception:
        dp, s_de, q_de = None, None, None
    parts = []
    if ep is not None:
        parts.append(rf"$\lambda_q={ep['lambda_q']:.2f}$, $S={ep['S']:.2f}$")
    if dp is not None:
        parts.append(rf"$\lambda_n={dp['lambda_near']:.2f}/\lambda_f={dp['lambda_far']:.2f}$")
    label = " | ".join(parts) if parts else "λ-fit failed"
    return (ep, s_fine, q_eich), (dp, s_de, q_de), label

fig2, axes2 = plt.subplots(1, len(targets), figsize=(5.8 * len(targets), 5.0),
                           squeeze=False)
axes2 = axes2[0]

print("\n" + "=" * 86)
print("Eich / 2-exp fits at the plate, i_dom, and i_dom+1 — λ_q comparison")
print("=" * 86)
print(f"{'target':<14}{'where':<22}{'λ_q [mm]':>10}{'S [mm]':>8}"
      f"{'λ_int [mm]':>12}{'λ_near':>10}{'λ_far':>9}")
print("-" * 86)
target_lambdas = {}

for ax, tgt in zip(axes2, targets):
    i_dom, i_gc = boundaries[tgt]
    up = int(np.sign(i_dom - i_gc)) or 1

    # Plate q_target(x) (textbook γ·P·V, e+i, ×Bp/B), unit MW/m²
    _, q_avg, _ = target_heat_flux_density(dsa, Pe, Pi, Ve, Vi, phi=phi,
                                           target=tgt, phi_wall=phi_wall,
                                           AA_ion=AA_ion,
                                           bohm_cap_v=bohm_cap_v)
    q_target  = q_avg.values * 1e-6
    q_pol_last = np.abs(q_pol_profile(i_dom)        * 1e-6)
    q_pol_up1  = np.abs(q_pol_profile(i_dom + up)   * 1e-6)
    s_mm = target_physical_distance(dsa, i_dom,
                                    ixseps1=int(dsa.metadata["ixseps1"])) * 1e3

    # ── Fits ────────────────────────────────────────────────────────────────
    fits_tgt  = _fit_lambdas(s_mm, q_target,   fit_s_range_mm, fit_s0=fit_s0)
    fits_last = _fit_lambdas(s_mm, q_pol_last, fit_s_range_mm, fit_s0=fit_s0)
    fits_up1  = _fit_lambdas(s_mm, q_pol_up1,  fit_s_range_mm, fit_s0=fit_s0)

    target_lambdas[tgt] = {
        "target":  fits_tgt[0][0] if fits_tgt[0] else None,
        "i_dom":   fits_last[0][0] if fits_last[0] else None,
        "i_dom+1": fits_up1[0][0] if fits_up1[0] else None,
        "target_dexp":  fits_tgt[1][0]  if fits_tgt[1] else None,
        "i_dom_dexp":   fits_last[1][0] if fits_last[1] else None,
        "i_dom+1_dexp": fits_up1[1][0]  if fits_up1[1] else None,
    }

    def _row(where, eich, dexp):
        lq  = f"{eich['lambda_q']:.2f}" if eich else "—"
        S_  = f"{eich['S']:.2f}"        if eich else "—"
        lint= f"{eich['lambda_int']:.2f}" if eich else "—"
        ln  = f"{dexp['lambda_near']:.2f}" if dexp else "—"
        lf  = f"{dexp['lambda_far']:.2f}"  if dexp else "—"
        print(f"{'':<14}{where:<22}{lq:>10}{S_:>8}{lint:>12}{ln:>10}{lf:>9}")

    print(f"{tgt:<14}")
    _row("target plate",       fits_tgt[0][0]  if fits_tgt[0]  else None,
                               fits_tgt[1][0]  if fits_tgt[1]  else None)
    _row("y = i_dom",          fits_last[0][0] if fits_last[0] else None,
                               fits_last[1][0] if fits_last[1] else None)
    _row(f"y = i_dom{'+' if up>0 else '-'}1",
                               fits_up1[0][0]  if fits_up1[0]  else None,
                               fits_up1[1][0]  if fits_up1[1]  else None)

    # ── Plot data + Eich fit overlays (no double-exp curve, table only) ────
    ax.plot(s_mm, q_target,   "r-",  lw=1.8,
            label=fr"target  ({fits_tgt[2]})")
    ax.plot(s_mm, q_pol_last, "b-",  lw=1.4,
            label=fr"$q_\mathrm{{pol}}$@i_dom  ({fits_last[2]})")
    ax.plot(s_mm, q_pol_up1,  "g--", lw=1.4,
            label=fr"$q_\mathrm{{pol}}$@i_dom{'+' if up > 0 else '-'}1  ({fits_up1[2]})")
    # Eich fit curves (mm/MW/m²) overlaid as thin lines
    for fits, color in [(fits_tgt, "r"), (fits_last, "b"), (fits_up1, "g")]:
        ep_tuple, _, _ = fits
        if ep_tuple and ep_tuple[1] is not None:
            ax.plot(ep_tuple[1], ep_tuple[2], color=color, ls=":", lw=0.9, alpha=0.7)
    ax.axvline(0, color="k", ls=":", lw=0.6)
    # No explicit xlim — let matplotlib autoscale to the full radial extent.
    # The Eich fit curve is restricted to the fit window only because that's
    # the s-range the fit was performed over; the data lines extend the full
    # grid range.
    ax.set_xlabel("s from separatrix at plate [mm]")
    ax.set_ylabel(r"heat flux density [MW/m²]")
    ax.set_title(tgt.replace("_", " ").title())
    ax.legend(fontsize=7, loc="upper right")

print("=" * 86)
print("Eich fits use the SAME s-coordinate (target frame) at all three locations,\n"
      "so λ_q from i_dom vs target measures how the radial profile *broadens or*\n"
      "*narrows* in the half-cell to the plate. Large discrepancy ⇒ significant\n"
      "near-target radial transport between the last plasma cell and the sheath.")

fig2.suptitle(r"Radial profiles + Eich/2-exp fits — $q_\mathrm{target}$ vs "
              r"$q_\mathrm{pol}$ at i_dom & i_dom±1", y=1.02)
fig2.tight_layout()

plt.show()

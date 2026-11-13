#!/usr/bin/env python3
"""
Target heat flux density, total power, and Eich λ_q fit for a Hermes-3
3D turbulence simulation.

Produces:
  1. Target heat flux profiles q(x) for all divertor targets
  2. Eich profile fit to extract λ_q and spreading factor S
  3. Total deposited power per target [kW]
  4. Optional: OMP-mapped profile for comparison with experiment

Usage
-----
    python target_heatflux.py <casepath> [gridfile] [options]

Arguments
---------
    casepath : directory containing the squashed ``boutdata.nc`` (preferred)
               or a set of ``BOUT.dmp.*.nc`` files, or a path to a single file.
    gridfile : (optional) Hypnotoad .grd.nc file, read with netCDF4 only, for
               the true-midplane "omp" coordinate (omp_mapped_distance). It is
               NOT passed to the loader: the squashed dump already carries the
               full metric and Bp/B is derived from it. A guardless grid also
               cannot be loaded alongside a guard-retaining dump under
               geometry="toroidal" (dimension-'y' size conflict), so geometry
               processing is skipped entirely.

IMPORTANT — guard cells
-----------------------
Target heat flux is evaluated *on the divertor plate*, which lies on the cell
boundary between the last plasma cell and the first y-boundary guard cell. The
plasma profiles must therefore be interpolated across that boundary, which is
only possible when the dump retains the (Hermes-3-extrapolated) y-boundary
guard cells. Squash the data with the y-boundaries kept, e.g.

    squashoutput(..., keep_yboundaries=True)

If the guard cells are absent the script still runs but falls back to the last
domain-cell value (a small underestimate at the plate).

Options (edit the CONFIG block below)
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from xhermes import open_hermesdataset

from hermes3_turb.target import (
    get_target_y_indices,
    get_target_boundaries,
    target_heat_flux_density,
    recover_target_ion_velocity,
    target_heat_flux_from_diagnostics,
    target_physical_distance,
    get_midplane_y_indices,
    omp_mapped_distance,
    field_line_incidence,
    eich_profile,
    fit_eich,
    integral_width,
    double_exponential_profile,
    fit_double_exp,
    total_target_power,
)
from hermes3_turb.load import (get_parallel_velocity, detect_poloidal_dim,
                              detect_toroidal_dim)

# ============================================================
# CONFIG – case settings
# ============================================================
#casepath = sys.argv[1] if len(sys.argv) > 1 else "."
#gridfile  = sys.argv[2] if len(sys.argv) > 2 else None
casepath = './161/'
gridfile = "../exp45470_x132_y28.nc"

AA_ion     = 2.0     # deuterium
Z_ion      = 1.0
phi_wall   = 0.0     # [V] wall potential
bohm_cap_v = False   # DEPRECATED (no-op). Wrong direction; see target.py docstring.

# Which targets to analyse. None = all targets present in the topology
# (inner_lower + outer_lower for single-null; + inner_upper + outer_upper for
# double-null). Or set an explicit subset, e.g. ["outer_lower", "outer_upper"].
targets_to_plot = None

# Coordinate the Eich / double-exp fits are performed in:
#   "omp"    — upstream-midplane-mapped (Eich convention; λ_q comparable to
#              Eich scaling & experiment). Inner targets map to IMP, outer to
#              OMP. With `gridfile` set, each flux surface (the leg's own ψ)
#              is mapped to R at the TRUE midplane through the geqdsk in the
#              grid (omp_mapped_distance); without it, √g11 along the
#              "midplane row", which is off the midplane on coarse ny.
#   "target" — flux-surface-perpendicular distance at the target plate.
#   "wetted" — distance along the inclined plate (s_target / sinα = s·B/Bp).
fit_coordinate = "omp"

# Eich fit: x range to use [fraction of ixseps1 offset in SOL, core]
fit_s_range_mm = (-5, 30)   # [mm] window around separatrix, in the fit coordinate
use_omp_mapped = False       # extra overlay of the target-coord profile (only
                             # meaningful when fit_coordinate != "omp")

# Electron/ion channel split of q_t. A third figure with, per target:
#   top    q_t, q_e, q_i                    — which channel carries the feature
#   bottom q_e/q_t, and gamma_e/gamma_i and phi/T_e on twin axes
# This is the diagnostic for the measured "two-humped target profile" of
# findings_target.md: the feature is the e/i channel crossover,
# controlled by phi/T_e through |V_e| ~ exp(-(phi-phi_wall)/T_e), NOT by
# gamma_e. It also separates gamma from P.|V| (pv_* in the components dict),
# which settles whether a narrow spike comes from the transmission factor or
# from the flow. Needs phi; with phi absent gamma is the fixed 3.5 fallback
# and the gamma panel is flat by construction.
split_species = True
# ============================================================

# ── Load ─────────────────────────────────────────────────────────────────────
# Locate the data: a squashed boutdata.nc (preferred) or BOUT.dmp.*.nc set.
print(f"Loading: {casepath}")
if os.path.isdir(casepath):
    datapath = next((os.path.join(casepath, name) for name in
                     ("boutdata_guards.nc", "boutdata_yguards.nc", "boutdata.nc")
                     if os.path.exists(os.path.join(casepath, name))),
                    os.path.join(casepath, "BOUT.dmp.*.nc"))
    _dump_dir = os.path.abspath(casepath)
else:
    datapath = casepath
    _dump_dir = os.path.dirname(os.path.abspath(casepath))
_data_dir = os.path.dirname(_dump_dir)
_settings = next((os.path.join(folder, name)
                  for folder in (_dump_dir, _data_dir)
                  for name in ("BOUT.settings", "BOUT.inp")
                  if os.path.exists(os.path.join(folder, name))), None)

# NOTE: geometry="toroidal" + a Hypnotoad grid are deliberately NOT used here.
# A guardless grid (ny without y-guards) conflicts with a guard-retaining dump
# (ny + y-guards) under geometry="toroidal" → "conflicting sizes for dimension
# 'y'". Everything the target analysis needs — metric, Bxy, dx, and Bp/B
# (derived from the metric) — is already in the dump, so geometry processing is
# skipped. `gridfile` is read with netCDF4 only, for the "omp" coordinate.
ds = open_hermesdataset(
    datapath,
    inputfilepath=_settings,
    keep_yboundaries=True,
)

m       = ds.metadata
ixseps1 = int(m.get("ixseps1", ds.sizes["x"] // 2))
pdim    = detect_poloidal_dim(ds)

# Report guard-cell status (needed to interpolate profiles onto the plate)
_bnd, _has_guards = get_target_boundaries(ds)
if _has_guards:
    print(f"y-boundary guard cells present (ny={ds.sizes[pdim]}) — "
          "profiles interpolated to the target plate.")
else:
    print("WARNING: no y-boundary guard cells in this dump — target values "
          "fall back to the last domain cell. Re-squash with "
          "keep_yboundaries=True for plate-accurate heat fluxes.")

if "Ve" not in ds:
    # The sheath sets the electron guard velocity too; NVe/Ne evaluated in
    # that guard does not generally recover the imposed sheath velocity.
    raise SystemExit("Ve is not in the dump: the electron sheath power "
                     "cannot be taken from NVe/Ne in the guard cells. Save Ve "
                     "([e] diagnose = true) and re-run.")

# Missing Vi: rebuild the sheath guard from interior momentum BEFORE averaging
# (NVi/Ni in momentum guards is not the sheath velocity; see docs/target.md).
if "Vi" not in ds:
    print("  Vi absent: reconstructing the full sheath from interior momentum")
    ds = recover_target_ion_velocity(ds, "i", AA_ion, Z_ion)
# Time-average over all loaded frames (assumed saturated; pre-trim before squashing)
print(f"Time-averaging over all {ds.sizes['t']} loaded frames")
ds_avg = ds.mean("t", keep_attrs=True)
for _name in ds.coords:
    if _name not in ds_avg and _name != "t":
        _coord = ds[_name]
        ds_avg = ds_avg.assign_coords({
            _name: _coord.isel(t=0, drop=True) if "t" in _coord.dims else _coord})

# Fields
phi = ds_avg["phi"]
Pe  = ds_avg["Pe"]
Pi  = ds_avg["Pi"]
Ve  = get_parallel_velocity(ds_avg, "e")
Vi  = get_parallel_velocity(ds_avg, "i")

# ── Compute and plot ──────────────────────────────────────────────────────────
all_targets = get_target_y_indices(ds_avg)
print(f"Targets found: {list(all_targets.keys())}")
zdim_name = detect_toroidal_dim(ds_avg)

# Default to every target the topology has (4 for double-null); otherwise keep
# only the requested ones that actually exist, warning about any that don't.
if targets_to_plot is None:
    targets_to_plot = list(all_targets.keys())
else:
    missing = [t for t in targets_to_plot if t not in all_targets]
    if missing:
        print(f"  (requested targets not in this topology, skipping: {missing})")
    targets_to_plot = [t for t in targets_to_plot if t in all_targets]

n_targets = len(targets_to_plot)
fig_hf, axes_hf = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), sharey=False)
if n_targets == 1:
    axes_hf = [axes_hf]

fig_eich, axes_eich = plt.subplots(1, n_targets, figsize=(6 * n_targets, 5), sharey=False)
if n_targets == 1:
    axes_eich = [axes_eich]

if split_species:
    fig_split, axes_split = plt.subplots(
        2, n_targets, figsize=(6 * n_targets, 8), sharex="col", squeeze=False)
else:
    fig_split = axes_split = None

results = {}

for i_col, (ax_hf, ax_eich, tgt) in enumerate(
        zip(axes_hf, axes_eich, targets_to_plot)):
    print(f"\n── Target: {tgt} ──")

    # ── Try diagnostic variables first ───────────────────────────────────────
    q_avg = target_heat_flux_from_diagnostics(ds_avg, target=tgt)

    if q_avg is not None:
        print("  Using Hermes-3 sheath diagnostic variables")
    else:
        # ── Compute from fields ───────────────────────────────────────────────
        print("  No sheath diagnostics found — computing from (γ P v_∥)(Bp/B)")
        _, q_avg, y_idx = target_heat_flux_density(
            ds_avg, Pe, Pi, Ve, Vi,
            phi=phi, target=tgt,
            phi_wall=phi_wall,
            AA_ion=AA_ion, Z_ion=Z_ion,
            bohm_cap_v=bohm_cap_v,
        )
        print(f"  Target y-index = {y_idx}")

    y_idx = all_targets[tgt]

    # ── Distance from separatrix, in the chosen fit coordinate ────────────────
    if fit_coordinate == "omp":
        mids   = get_midplane_y_indices(ds_avg)
        is_imp = tgt.startswith("inner") and "imp" in mids
        if gridfile:
            # TRUE midplane, mapped by the leg's own ψ (omp_mapped_distance):
            # the midplane ROW is off the midplane on coarse ny (SXD ny32:
            # Z=+0.475 m, ~1.2x too wide) and its x-columns gave PFR cells the
            # core ψ.
            coord_label = "IMP-mapped (ψ)" if is_imp else "OMP-mapped (ψ)"
            s_m = omp_mapped_distance(gridfile, tgt)
            if s_m.size != ds_avg.sizes["x"]:
                sys.exit(f"gridfile nx={s_m.size} but dump nx={ds_avg.sizes['x']}"
                         f" — wrong grid for this run")
        else:
            print("  WARNING: no gridfile — mapping by the midplane ROW, which is "
                  "off the midplane on coarse-ny grids (ny32: s ~1.2x too wide)")
            y_ref  = mids["imp"] if is_imp else mids["omp"]
            coord_label = "IMP-mapped" if is_imp else "OMP-mapped"
            s_m = target_physical_distance(ds_avg, y_ref, ixseps1=ixseps1)
    elif fit_coordinate == "wetted":
        coord_label = "wetted (along plate)"
        s_perp = target_physical_distance(ds_avg, y_idx, ixseps1=ixseps1)
        sin_a  = float(field_line_incidence(ds_avg, y_idx).mean())  # ≈ Bp/B at plate
        s_m    = s_perp / max(sin_a, 1e-3)
    else:  # "target"
        coord_label = "target (⊥ flux surfaces)"
        s_m = target_physical_distance(ds_avg, y_idx, ixseps1=ixseps1)
    s_mm = s_m * 1e3   # [mm]

    q_Wm2  = q_avg.values             # [W/m²]
    q_MWm2 = q_Wm2 * 1e-6            # [MW/m²]

    # ── Eich fit ──────────────────────────────────────────────────────────────
    # Restrict to fit window
    fit_mask = (s_mm >= fit_s_range_mm[0]) & (s_mm <= fit_s_range_mm[1])
    s_fit    = s_mm[fit_mask]
    q_fit    = q_MWm2[fit_mask]

    try:
        params, perr, s_fine, q_eich = fit_eich(s_fit, q_fit)
        fit_ok = True
        print(f"  Eich fit  [{coord_label}]:")
        print(f"    q₀        = {params['q0']:.3f} ± {perr['q0']:.3f}  MW/m²")
        print(f"    λ_q       = {params['lambda_q']:.2f} ± {perr['lambda_q']:.2f}  mm")
        print(f"    S         = {params['S']:.2f} ± {perr['S']:.2f}  mm")
        print(f"    q_bg      = {params['q_bg']:.4f} ± {perr['q_bg']:.4f}  MW/m²")
    except Exception as exc:
        fit_ok = False
        print(f"  Eich fit failed: {exc}")

    # ── Integral power decay length ───────────────────────────────────────────
    # The DEFINITION is the integral (Loarte 1999; Eich 2013 §2):
    #   λ_int = ∫(q − q_BG) ds / q_max
    # `λ_q + 1.64·S` is only Makowski's approximation to it in terms of the
    # fitted parameters, so it is reported second and inherits any badness of
    # the fit. The integral is taken over the WHOLE measured profile including
    # the PFR side (s < 0) — power deposited there is still deposited power,
    # and Thornton 2015 shows the Eich form misses that spreading. It is
    # computed even when the fit fails, which is the point of it.
    # Two honest numbers, because they answer different questions:
    #  * over the FIT WINDOW with the fitted q_bg — apples-to-apples with
    #    Makowski's λ_q+1.64·S, so any gap is genuinely profile shape.
    #  * over the WHOLE profile with no background — the width carrying the
    #    total deposited power, which is what matters for power handling.
    # Never subtract a fitted q_bg over the whole domain: it is a local
    # nuisance parameter, and extrapolating it across 100+ mm of PFR removes a
    # large spurious area.
    iw_full = integral_width(s_mm, q_MWm2)
    print(f"    λ_int     = {iw_full['lambda_int']:.2f}  mm  "
          f"(integral over full profile, no q_bg — total deposited power)")
    if fit_ok:
        iw_win = integral_width(s_mm, q_MWm2, q_bg=params["q_bg"],
                                s_range=fit_s_range_mm)
        _approx = params["lambda_int"]
        _dev = (100.0 * (_approx / iw_win["lambda_int"] - 1.0)
                if iw_win["lambda_int"] else float("nan"))
        print(f"    λ_int     = {iw_win['lambda_int']:.2f}  mm  "
              f"(integral over fit window, q_bg removed)")
        print(f"    λ_int     = {_approx:.2f}  mm  (= λ_q + 1.64·S, Makowski "
              f"approx; {_dev:+.1f} % vs the fit-window integral)")
        if not params.get("lambda_int_valid", True):
            print(f"                 NOTE S/λ_q = {params['S']/params['lambda_q']:.1f} "
                  f">= 10 — outside the <4 % validity range of the approximation")
        elif abs(_dev) > 10.0:
            print("                 NOTE >10 % apart — the Eich form is not "
                  "describing this profile\n                 (known for MAST; "
                  "Horacek 2020). Trust the integral.")
    else:
        iw_win = None

    # ── Double-exponential fit (near + far SOL, s ≥ 0) ─────────────────────────
    try:
        de_params, de_perr, s_fine_de, q_de = fit_double_exp(s_fit, q_fit, s_min=0.0)
        de_ok = True
        print(f"  Double-exponential fit (SOL side):")
        print(f"    λ_near    = {de_params['lambda_near']:.2f} ± {de_perr['lambda_near']:.2f}  mm"
              f"  (q_near = {de_params['q_near']:.3f} MW/m²)")
        print(f"    λ_far     = {de_params['lambda_far']:.2f} ± {de_perr['lambda_far']:.2f}  mm"
              f"  (q_far  = {de_params['q_far']:.3f} MW/m²)")
        print(f"    λ_eff     = {de_params['lambda_eff']:.2f}  mm  (power-weighted)")
        print(f"    far power = {de_params['power_frac_far'] * 100:.0f} %  of SOL")
    except Exception as exc:
        de_ok = False
        print(f"  Double-exponential fit failed: {exc}")

    # ── Electron / ion channel split ──────────────────────────────────────────
    # Computed from the formula even when q_avg above came from the sheath
    # diagnostics: E<sp>_sheath is a total, so it carries no split.
    split = None
    if split_species:
        _, q_avg_f, _, comp = target_heat_flux_density(
            ds_avg, Pe, Pi, Ve, Vi,
            phi=phi, target=tgt,
            phi_wall=phi_wall,
            AA_ion=AA_ion, Z_ion=Z_ion,
            bohm_cap_v=bohm_cap_v,
            components=True,
        )
        qe = comp["q_e_avg"].values * 1e-6         # [MW/m²]
        qi = comp["q_i_avg"].values * 1e-6
        qt = q_avg_f.values * 1e-6
        frac = comp["q_e_frac"].values
        g_e  = comp["gamma_e_avg"].values
        g_i  = comp["gamma_i_avg"].values
        phiT = (comp["phi_over_Te"].mean(dim=zdim_name).values
                if comp["phi_over_Te"] is not None else None)
        phi_V = (comp["phi"].mean(dim=zdim_name).values
                 if comp["phi"] is not None else None)

        # Report the split where it matters: at the separatrix and at the peak.
        i_sep  = int(np.nanargmin(np.abs(s_mm)))
        i_peak = int(np.nanargmax(np.where(np.isfinite(qt), qt, -np.inf)))
        print(f"  e/i split: q_e/q = {frac[i_sep]:.2f} at the separatrix, "
              f"{frac[i_peak]:.2f} at the peak (s={s_mm[i_peak]:.1f} mm)")
        print(f"             γ_e {g_e[i_sep]:.2f}→{g_e[i_peak]:.2f}, "
              f"γ_i {g_i[i_sep]:.2f}→{g_i[i_peak]:.2f} over the same span"
              + ("" if phi is not None else "   [fixed-3.5 fallback: no φ]"))
        if phiT is not None:
            print(f"             φ/T_e {phiT[i_sep]:.2f}→{phiT[i_peak]:.2f}  "
                  f"(exp(−φ/T_e) {np.exp(-phiT[i_sep]):.3f}→"
                  f"{np.exp(-phiT[i_peak]):.3f}), φ = {phi_V[i_sep]:.1f} V "
                  f"at the separatrix")
        # The two channels must sum to the total: a free check on the split.
        _resid = np.nanmax(np.abs(qe + qi - qt)) / max(np.nanmax(np.abs(qt)), 1e-30)
        if _resid > 1e-10:
            print(f"  !! q_e + q_i differs from q by {_resid:.2e} of peak")

        split = {"q_e_MWm2": qe, "q_i_MWm2": qi, "q_tot_MWm2": qt,
                 "q_e_frac": frac, "gamma_e": g_e, "gamma_i": g_i,
                 "phi_over_Te": phiT, "phi_V": phi_V,
                 "i_sep": i_sep, "i_peak": i_peak}

        ax_t, ax_b = axes_split[0][i_col], axes_split[1][i_col]
        ax_t.plot(s_mm, qt, "k.-", ms=4, lw=1.4, label=r"$q_t$ (total)")
        ax_t.plot(s_mm, qe, "C0.-", ms=3, lw=1.2, label=r"$q_e=\gamma_e P_e|V_e|B_p/B$")
        ax_t.plot(s_mm, qi, "C3.-", ms=3, lw=1.2, label=r"$q_i=\gamma_i P_i|V_i|B_p/B$")
        ax_t.axvline(0, color="k", ls="--", lw=0.8)
        ax_t.set_ylabel(r"$q_\perp$ [MW/m²]")
        ax_t.set_title(f"{tgt.replace('_', ' ').title()} — channel split")
        ax_t.legend(fontsize=8)

        ax_b.plot(s_mm, frac, "C2.-", ms=4, lw=1.4, label=r"$q_e/q_t$")
        ax_b.axhline(0.5, color="C2", ls=":", lw=0.8)
        ax_b.axvline(0, color="k", ls="--", lw=0.8)
        ax_b.set_ylim(0, 1.05)
        ax_b.set_ylabel(r"$q_e/q_t$", color="C2")
        ax_b.set_xlabel(f"s from separatrix [mm] — {coord_label}")
        ax_b.legend(fontsize=8, loc="upper left")

        ax_g = ax_b.twinx()
        ax_g.plot(s_mm, g_e, "C0--", lw=1, label=r"$\gamma_e$")
        ax_g.plot(s_mm, g_i, "C3--", lw=1, label=r"$\gamma_i$")
        if phiT is not None:
            ax_g.plot(s_mm, phiT, "C4:", lw=1.4, label=r"$\phi/T_e$")
        ax_g.set_ylabel(r"$\gamma$,  $\phi/T_e$")
        ax_g.legend(fontsize=8, loc="upper right")
        ax_b.set_xlim(fit_s_range_mm[0] - 2, fit_s_range_mm[1])

    # ── Total target power ────────────────────────────────────────────────────
    P_total = total_target_power(ds_avg, q_avg, y_idx)
    print(f"  Total target power = {P_total / 1e3:.2f} kW")

    results[tgt] = {
        "s_mm": s_mm, "q_MWm2": q_MWm2,
        "P_total_kW": P_total / 1e3,
    }
    if split is not None:
        results[tgt]["split"] = split
    if fit_ok:
        results[tgt]["eich_params"] = params
        results[tgt]["eich_perr"]   = perr
    if de_ok:
        results[tgt]["dexp_params"] = de_params
        results[tgt]["dexp_perr"]   = de_perr

    # ── Target-coordinate overlay (optional; redundant when fitting in OMP) ───
    # When the fit is in target coordinates, optionally overlay the midplane-
    # mapped profile for comparison (inner→IMP, outer→OMP).
    if use_omp_mapped and fit_coordinate != "omp":
        mids = get_midplane_y_indices(ds_avg)
        if tgt.startswith("inner") and "imp" in mids:
            y_mid, mid_name = mids["imp"], "IMP"
        else:
            y_mid, mid_name = mids["omp"], "OMP"
        s_mid_mm = (omp_mapped_distance(gridfile, tgt) if gridfile else
                    target_physical_distance(ds_avg, y_mid, ixseps1=ixseps1)) * 1e3
        ax_hf.plot(s_mid_mm, q_MWm2, "k--", lw=1,
                   label=f"{mid_name}-mapped", alpha=0.6)

    # ── Heat flux profile plot ────────────────────────────────────────────────
    ax_hf.plot(s_mm, q_MWm2, "b.-", ms=4, label="simulation (toroidal avg)")
    ax_hf.axvline(0, color="k", ls="--", lw=0.8, label="separatrix (s=0)")
    ax_hf.set_xlabel(f"s from separatrix [mm] — {coord_label}")
    ax_hf.set_ylabel(r"$q_\perp$ [MW/m²]")
    ax_hf.set_title(f"{tgt.replace('_', ' ').title()}\n"
                    f"$P_{{total}}$ = {P_total/1e3:.1f} kW")
    ax_hf.legend(fontsize=8)
    ax_hf.set_xlim(fit_s_range_mm[0] - 2, fit_s_range_mm[1])

    # ── Eich fit plot ─────────────────────────────────────────────────────────
    ax_eich.plot(s_mm[fit_mask], q_fit, "b.", ms=5, label="simulation (fit range)")
    ax_eich.plot(s_mm[~fit_mask], q_MWm2[~fit_mask], "b.", ms=3, alpha=0.3)
    if fit_ok:
        ax_eich.plot(s_fine, q_eich, "r-", lw=2, label="Eich fit")
        # Mark λ_q
        lq = params["lambda_q"]
        ax_eich.axvline(lq, color="r", ls=":", lw=1,
                        label=rf"$\lambda_q$ = {lq:.1f} mm")
        ax_eich.axvline(params["lambda_int"], color="m", ls=":", lw=1,
                        label=rf"$\lambda_{{int}}$ (fit) = {params['lambda_int']:.1f} mm")
    ax_eich.axvline(iw_full["lambda_int"], color="m", ls="-", lw=1.4, alpha=0.75,
                    label=rf"$\lambda_{{int}}$ (integral) = {iw_full['lambda_int']:.1f} mm")
    if de_ok:
        ax_eich.plot(s_fine_de, q_de, "g--", lw=2,
                     label=(rf"double-exp ($\lambda_n$={de_params['lambda_near']:.1f}, "
                            rf"$\lambda_f$={de_params['lambda_far']:.1f} mm)"))
    if fit_ok:
        ax_eich.set_title(
            f"{tgt.replace('_', ' ').title()} — Eich + double-exp\n"
            rf"$\lambda_q$ = {lq:.2f} mm, $S$ = {params['S']:.2f} mm"
        )
    else:
        ax_eich.set_title(f"{tgt.replace('_', ' ').title()} — Eich fit failed")

    ax_eich.axvline(0, color="k", ls="--", lw=0.8)
    ax_eich.set_xlabel(f"s from separatrix [mm] — {coord_label}")
    ax_eich.set_ylabel(r"$q_\perp$ [MW/m²]")
    ax_eich.legend(fontsize=8)
    ax_eich.set_xlim(fit_s_range_mm[0] - 2, fit_s_range_mm[1])

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n" + "=" * 86)
print(f"Widths in '{fit_coordinate}' coordinate"
      + ("  (OMP/IMP-mapped — Eich convention)" if fit_coordinate == "omp" else ""))
print(f"{'Target':<18} {'P [kW]':>9} {'λ_q [mm]':>10} {'S [mm]':>9} "
      f"{'λ_near [mm]':>12} {'λ_far [mm]':>11} {'far %':>7}")
print("-" * 86)
for tgt, res in results.items():
    e = res.get("eich_params")
    d = res.get("dexp_params")
    lq_str = f"{e['lambda_q']:.2f}" if e else "N/A"
    S_str  = f"{e['S']:.2f}"        if e else "N/A"
    ln_str = f"{d['lambda_near']:.2f}" if d else "N/A"
    lf_str = f"{d['lambda_far']:.2f}"  if d else "N/A"
    ff_str = f"{d['power_frac_far'] * 100:.0f}" if d else "N/A"
    print(f"{tgt:<18} {res['P_total_kW']:>9.1f} {lq_str:>10} {S_str:>9} "
          f"{ln_str:>12} {lf_str:>11} {ff_str:>7}")
print("=" * 86)

# ── Channel-split summary ─────────────────────────────────────────────────────
if split_species and any("split" in r for r in results.values()):
    print("")
    print(f"{'Target':<18} {'q_e/q sep':>10} {'q_e/q peak':>11} "
          f"{'γ_e sep':>8} {'φ/T_e sep':>10} {'φ [V] sep':>10}")
    print("-" * 72)
    for tgt, res in results.items():
        sp = res.get("split")
        if sp is None:
            continue
        i0, i1 = sp["i_sep"], sp["i_peak"]
        pT = f"{sp['phi_over_Te'][i0]:.2f}" if sp["phi_over_Te"] is not None else "N/A"
        pV = f"{sp['phi_V'][i0]:.1f}" if sp["phi_V"] is not None else "N/A"
        print(f"{tgt:<18} {sp['q_e_frac'][i0]:>10.2f} {sp['q_e_frac'][i1]:>11.2f} "
              f"{sp['gamma_e'][i0]:>8.2f} {pT:>10} {pV:>10}")
    print("-" * 72)
    print("  A target whose q_e/q falls from ~1 at the separatrix to ~0.2 at the")
    print("  peak is the e/i crossover: two channels with different radial")
    print("  profiles, not one broadened channel. Eich cannot fit that at all —")
    print("  quote λ_int. The control parameter is φ/T_e, through")
    print("  |V_e| ∝ exp(−(φ−φ_wall)/T_e); a high-φ target (φ/T_e ≳ 3) is")
    print("  ion-dominated everywhere and smooth. See findings_target.md.")
    print("=" * 72)

fig_hf.suptitle("Target heat flux profiles", y=1.01)
fig_hf.tight_layout()

fig_eich.suptitle("Eich (red) + double-exponential (green) profile fits", y=1.01)
fig_eich.tight_layout()

if fig_split is not None:
    fig_split.suptitle(
        "Electron / ion channel split of the target heat flux", y=1.00)
    fig_split.tight_layout()

plt.show()

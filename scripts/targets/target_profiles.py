#!/usr/bin/env python3
"""Plot divertor-plate density, temperatures, pressure, and ion saturation current.

Configure one or more cases in CASES. Profiles use the sheath plane between
the final domain cell and its guard; guardless input is identified as a
last-domain-cell estimate. Each folder's archived settings are used for ion
velocity reconstruction before folders are combined.

fit_coordinate selects target distance or OMP/IMP-mapped distance. A grid
file provides the true-midplane flux mapping. x_axis='index' instead shows
radial cell index relative to the separatrix. third_row selects pressure,
ion saturation current, Ti/Te, or no third row.

This script uses separate caches from the target heat-flux scripts. Derived
quantities use time-averaged fields, so nonlinear products can differ from
averages formed per snapshot.
"""
import os
import warnings

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

import hermes3_turb as h3t
from hermes3_turb.load import get_parallel_velocity
from hermes3_turb.target import (
    get_target_boundaries, get_target_y_indices, get_midplane_y_indices,
    omp_mapped_distance, target_physical_distance, target_heat_flux_density,
    recover_target_ion_velocity,
)
from xhermes import open_hermesdataset

# ── CONFIG ────────────────────────────────────────────────────────────────────
# One entry per case: (label, data_root, dir_range, gridfile or None).
# data_root holds the numbered restart folders, each with a squashed dump that
# KEEPS THE Y-GUARDS (boutdata_guards.nc / boutdata_yguards.nc / boutdata.nc).
CASES = [
    # The gridfile is read with netCDF4 ONLY, for the true-midplane psi
    # mapping. Pass None to fall back to the midplane ROW (and get a warning).
    ("SXD 6kW", "./", range(422, 426), "../bout_sxd_v4_nx244_new.grd.nc"),
    # ("CD ny28", "../../cd_MU/new_ny28/fresh_ES/data/", range(130, 135),
    #  "../../cd_MU/new_ny28/exp45470_x132_y28.nc"),
]
targets_to_plot = None     # None = every target in the topology
ion_species     = "i"      # "i" or "d+"
AA_ion, Z_ion   = 2.0, 1.0
fit_coordinate  = "omp"    # "omp" (midplane-mapped, matches target_heatflux.py)
                           # or "target" (perpendicular distance at the plate)
xlim_mm         = (-10, 40)
x_axis          = "s"      # "s"     : mapped distance s [mm] (fit_coordinate)
                           # "index" : radial cell index from the separatrix,
                           #           0 = first SOL cell. The mapped s squeezes
                           #           the near-separatrix cells (0.05 mm/cell on
                           #           the SXD outer legs), so a smooth rise over
                           #           20+ cells draws as a vertical step at s=0;
                           #           this axis shows whether it is one.
xlim_index      = (-40, 80)  # window in cells for x_axis="index"; None = all
mxg             = 2        # x-guard cells dropped at both radial ends
logy_density    = False
third_row       = "pressure"   # "pressure" : static plate pressure Pe + Pi
                               # "jsat"     : ion saturation current density,
                               #              Z e N_i |V_i| (Bp/B) — what a
                               #              flush-mounted LP collects
                               # "Ti_Te"    : the temperature ratio
                               # None       : two rows only
normalise       = None     # None | "sep" (divide by the separatrix value)
use_cache       = True
save_png        = None     # e.g. "target_profiles.png"
# ──────────────────────────────────────────────────────────────────────────────

THIRD = {
    "pressure": ("p_Pa",  r"$p_e + p_i$ at the plate [Pa]"),
    "jsat":     ("jsat",  r"$j_\mathrm{sat}$ [A m$^{-2}$]"),
    "Ti_Te":    ("Ti_Te", r"$T_i/T_e$"),
}
# Te and Ti need their own colours, not one per case with a linestyle: with a
# single case the two curves are otherwise near-indistinguishable.
TE_COLOUR, TI_COLOUR = "C0", "C3"


def load_case(label, data_root, dir_range, gridfile):
    """Plate Ne/Ni/Te/Ti per target for one case, cached."""
    cache = h3t.cache_path("target_profiles", dir_range, ion_species,
                           fit_coordinate, "v3", cache_dir=os.path.join(
                               data_root, "cache"))
    if use_cache and cache.exists():
        print(f"[{label}] cache hit: {cache}")
        return h3t.load_cache(cache)

    dirs = h3t.existing_dirs(data_root, dir_range,
                             ("boutdata_guards.nc", "boutdata_yguards.nc",
                              "boutdata.nc"))
    if not dirs:
        raise SystemExit(f"[{label}] no squashed dumps found in {data_root}")
    dss = []
    for d in dirs:
        for name in ("boutdata_guards.nc", "boutdata_yguards.nc", "boutdata.nc"):
            path = os.path.join(data_root, str(d), name)
            if os.path.exists(path):
                break
        # BOUT.settings / BOUT.inp usually sits one level above the numbered dir.
        settings = next(
            (c for c in (os.path.join(data_root, str(d), "BOUT.settings"),
                         os.path.join(data_root, str(d), "BOUT.inp"),
                         os.path.join(data_root, "BOUT.settings"),
                         os.path.join(data_root, "BOUT.inp"))
             if os.path.exists(c)), None)
        print(f"[{label}] loading {path}")
        # NO geometry="toroidal" and NO gridfilepath — the same deliberate
        # choice target_heatflux.py documents. Two reasons, and they bite in
        # opposite directions:
        #   * without a grid, geometry="toroidal" raises
        #     "Grid file is required to provide psixy";
        #   * with one, a guardless grid (ny) conflicts with this
        #     guard-retaining dump (ny + y-guards) -> "conflicting sizes for
        #     dimension 'y'".
        # Everything needed here — the metric, Bxy, dx — is already in the
        # dump. `gridfile` is used ONLY by omp_mapped_distance, which reads it
        # with netCDF4 directly.
        folder_ds = open_hermesdataset(
            path, inputfilepath=settings, keep_yboundaries=True)
        if "Ve" not in folder_ds:
            raise SystemExit(
                f"[{label}] Ve is not in {path}: save Ve "
                "([e] diagnose = true); NVe/Ne in a guard is not the sheath velocity.")
        if f"V{ion_species}" not in folder_ds:
            folder_ds = recover_target_ion_velocity(
                folder_ds, ion_species, AA_ion, Z_ion)
        dss.append(folder_ds)
    pdim = h3t.detect_poloidal_dim(dss[0])
    if len({part.sizes[pdim] for part in dss}) != 1:
        raise SystemExit(f"[{label}] mixed poloidal sizes: re-squash with consistent y-guards")
    ds = h3t.dedup_restart_times(xr.concat(dss, dim="t", join="exact"))
    t_s = np.asarray(ds["t"].values, dtype=float)
    _, has_guards = get_target_boundaries(ds)
    if not has_guards:
        warnings.warn(
            f"[{label}] no y-boundary guards: plate profiles are evaluated at "
            "the last domain cell. Re-squash with keep_yboundaries=True.",
            stacklevel=2)
    ds_avg = ds.mean("t", keep_attrs=True)
    for name in ds.coords:
        if name not in ds_avg and name != "t":
            coord = ds[name]
            ds_avg = ds_avg.assign_coords({
                name: coord.isel(t=0, drop=True) if "t" in coord.dims else coord})

    Pe, Pi = ds_avg["Pe"], ds_avg[f"P{ion_species}"]
    Ve = get_parallel_velocity(ds_avg, "e")
    Vi = get_parallel_velocity(ds_avg, ion_species)
    Ve = Ve.where(np.isfinite(Ve), 0.0)
    Vi = Vi.where(np.isfinite(Vi), 0.0)
    phi = ds_avg["phi"] if "phi" in ds_avg else None

    m = ds_avg.metadata
    ixseps1 = int(m["ixseps1"])
    mids = get_midplane_y_indices(ds_avg)
    targets = get_target_y_indices(ds_avg)
    want = targets_to_plot or list(targets)

    out = {"target_names": [], "target_scalars": {},
           "t_s": t_s, "has_guards": bool(has_guards), "label": label}
    for tgt in want:
        if tgt not in targets:
            warnings.warn(f"[{label}] no target {tgt!r}", stacklevel=2)
            continue
        # The components dict carries the plate plasma AT THE SHEATH.
        _, _, _, comp = target_heat_flux_density(
            ds_avg, Pe, Pi, Ve, Vi, phi=phi, target=tgt,
            AA_ion=AA_ion, Z_ion=Z_ion, ion_species=ion_species,
            bohm_cap_v=False, components=True)

        # Same coordinate convention as target_heatflux.py, so a feature here
        # lines up with a feature there.
        is_imp = tgt.startswith("inner") and "imp" in mids
        if fit_coordinate == "omp" and gridfile:
            # TRUE midplane, mapped by the leg's own psi.
            coord = "IMP-mapped (ψ)" if is_imp else "OMP-mapped (ψ)"
            s_m = omp_mapped_distance(gridfile, tgt)
            if np.asarray(s_m).size != ds_avg.sizes["x"]:
                raise SystemExit(
                    f"[{label}] gridfile nx={np.asarray(s_m).size} but dump "
                    f"nx={ds_avg.sizes['x']} — wrong grid for this run")
        elif fit_coordinate == "omp":
            # The midplane ROW is off the midplane on coarse ny (SXD ny32:
            # Z=+0.475 m, s ~1.2x too wide), so say so rather than silently
            # plotting a stretched coordinate.
            warnings.warn(
                f"[{label}] no gridfile — mapping by the midplane ROW, which "
                "is off the midplane on coarse-ny grids (ny32: s ~1.2x too "
                "wide). Give the grid for the true psi mapping.", stacklevel=2)
            coord = "IMP-mapped" if is_imp else "OMP-mapped"
            s_m = target_physical_distance(
                ds_avg, mids["imp"] if is_imp else mids["omp"],
                ixseps1=ixseps1)
        else:
            coord = "target (⊥ flux surf)"
            s_m = target_physical_distance(ds_avg, targets[tgt],
                                           ixseps1=ixseps1)
        s_mm = np.asarray(s_m, dtype=float) * 1e3

        out["target_names"].append(tgt)
        out["target_scalars"][tgt] = {"coord_label": coord,
                                      "y_idx": int(targets[tgt])}
        for key, src in (("s_mm", s_mm),
                         ("Ne_m3", comp["Ne_avg"].values),
                         ("Ni_m3", comp["Ni_avg"].values),
                         ("Te_eV", comp["Te_eV_avg"].values),
                         ("Ti_eV", comp["Ti_eV_avg"].values),
                         ("Pe_Pa", comp["Pe_avg"].values),
                         ("Pi_Pa", comp["Pi_avg"].values),
                         ("jsat", comp["jsat_avg"].values)):
            out[f"tgt_{tgt}__{key}"] = np.asarray(src, dtype=float)
    h3t.save_cache(cache, **out)
    print(f"[{label}] saved cache: {cache}")
    return h3t.load_cache(cache)


def profiles_from(cached, guards):
    """{target: dict of 1-D arrays}, x-guards dropped."""
    names = np.atleast_1d(cached.get("target_names", [])).tolist()
    scalars = cached.get("target_scalars", {}) or {}
    sl = slice(guards, -guards if guards else None)
    out = {}
    for name in names:
        name = name.decode() if isinstance(name, bytes) else str(name)
        d = {k[len(f"tgt_{name}__"):]: np.asarray(v, dtype=float)
             for k, v in cached.items() if k.startswith(f"tgt_{name}__")}
        if "s_mm" not in d:
            continue
        rec = {k: v[sl] for k, v in d.items()}
        # Cell index from the separatrix: s = 0 is the face between ixseps1-1
        # and ixseps1, so the first cell with s > 0 is ixseps1 -> index 0.
        sol = np.flatnonzero(d["s_mm"] > 0)
        if sol.size:
            rec["ix_sep"] = np.arange(d["s_mm"].size)[sl] - int(sol[0])
        te = rec["Te_eV"]
        rec["Ti_Te"] = rec["Ti_eV"] / np.where(te > 0, te, np.nan)
        if {"Pe_Pa", "Pi_Pa"} <= rec.keys():
            rec["p_Pa"] = rec["Pe_Pa"] + rec["Pi_Pa"]
        rec["coord"] = (scalars.get(name) or {}).get("coord_label", "?")
        out[name] = rec
    return out


def main():
    loaded = []
    for label, root, dirs, grid in CASES:
        cached = load_case(label, root, dirs, grid)
        loaded.append((label, profiles_from(cached, mxg)))

    targets = targets_to_plot or list(dict.fromkeys(
        t for _, p in loaded for t in p))
    if not targets:
        raise SystemExit("no targets found in any case")

    if x_axis not in ("s", "index"):
        raise SystemExit('x_axis must be "s" or "index"')
    if third_row is not None and third_row not in THIRD:
        raise SystemExit(f"third_row must be one of {sorted(THIRD)} or None")
    key3, ylab3 = THIRD[third_row] if third_row else (None, None)
    nrows = 3 if third_row else 2
    multi = len(loaded) > 1

    fig, axes = plt.subplots(nrows, len(targets),
                             figsize=(4.6 * len(targets), 3.3 * nrows),
                             sharex="col", squeeze=False)
    hdr3 = {"pressure": "p sep [Pa]", "jsat": "jsat sep", "Ti_Te": "Ti/Te sep"}
    print(f"\n{'case':16s} {'target':14s} {'Ne sep':>11s} {'Te sep':>8s} "
          f"{'Ti sep':>8s} {'Te peak':>8s} {'Ti peak':>8s}"
          + (f" {hdr3[third_row]:>12s}" if third_row else ""))
    print("-" * (92 + (13 if third_row else 0)))
    for col, tgt in enumerate(targets):
        for k, (label, prof) in enumerate(loaded):
            if tgt not in prof:
                warnings.warn(f"{label}: {tgt} absent", stacklevel=2)
                continue
            p = prof[tgt]
            s_mm = p["s_mm"]
            if x_axis == "index":
                if "ix_sep" not in p:
                    warnings.warn(f"{label}: {tgt}: no cell with s > 0, cannot "
                                  "place the separatrix", stacklevel=2)
                    continue
                xv = p["ix_sep"]
                lo, hi = xlim_index or (xv[0], xv[-1])
            else:
                xv = s_mm
                lo, hi = xlim_mm
            mask = (xv >= lo) & (xv <= hi)
            if not mask.any():
                warnings.warn(f"{label}: {tgt}: nothing in the plotted window",
                              stacklevel=2)
                continue
            i_sep = int(np.nanargmin(np.abs(s_mm)))
            colour = f"C{k % 10}"

            def scaled(arr):
                if normalise == "sep" and np.isfinite(arr[i_sep]) and arr[i_sep]:
                    return arr / arr[i_sep]
                return arr

            axes[0][col].plot(xv[mask], scaled(p["Ne_m3"])[mask], ".-",
                              ms=3, color=colour, label=label)
            # Te and Ti get their OWN colours. With several cases the case is
            # distinguished by linestyle instead, so the pair stays readable
            # either way.
            ls_e, ls_i = ("-", "--") if not multi else (["-", "--", ":", "-."][k % 4],) * 2
            axes[1][col].plot(xv[mask], scaled(p["Te_eV"])[mask], ls_e, lw=1.4,
                              marker=".", ms=2, color=TE_COLOUR,
                              label=(f"{label}  " if multi else "") + r"$T_e$")
            axes[1][col].plot(xv[mask], scaled(p["Ti_eV"])[mask], ls_i, lw=1.4,
                              marker=".", ms=2, color=TI_COLOUR,
                              label=(f"{label}  " if multi else "") + r"$T_i$")

            v3 = p.get(key3) if key3 else None
            if key3 and v3 is None:
                warnings.warn(f"{label}: {tgt}: {key3} not in this cache — "
                              "delete it and re-run", stacklevel=2)
            elif key3:
                axes[2][col].plot(xv[mask], scaled(v3)[mask], ".-", ms=3,
                                  color=colour, label=label)

            i_pk = np.flatnonzero(mask)[int(np.nanargmax(p["Te_eV"][mask]))]
            line = (f"{label:16s} {tgt:14s} {p['Ne_m3'][i_sep]:11.3e} "
                    f"{p['Te_eV'][i_sep]:8.2f} {p['Ti_eV'][i_sep]:8.2f} "
                    f"{p['Te_eV'][i_pk]:8.2f} {p['Ti_eV'][i_pk]:8.2f}")
            if key3 and v3 is not None:
                line += f" {v3[i_sep]:12.4g}"
            print(line)

        ylabels = ["value / separatrix" if normalise == "sep"
                   else r"$N_e$ [m$^{-3}$]",
                   "value / separatrix" if normalise == "sep" else "T [eV]"]
        if third_row:
            ylabels.append("value / separatrix" if normalise == "sep" else ylab3)
        for r in range(nrows):
            ax = axes[r][col]
            # The separatrix: s = 0, or the face between cells -1 and 0.
            ax.axvline(0.0 if x_axis == "s" else -0.5, color="k", ls="--", lw=0.8)
            ax.grid(alpha=0.3)
            if x_axis == "s":
                ax.set_xlim(*xlim_mm)
            elif xlim_index:
                ax.set_xlim(*xlim_index)
            if col == 0:
                ax.set_ylabel(ylabels[r])
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7)
        if third_row == "Ti_Te":
            axes[2][col].axhline(1.0, color="gray", ls=":", lw=0.8)
        axes[0][col].set_title(tgt.replace("_", " "))
        if logy_density and normalise is None:
            axes[0][col].set_yscale("log")
        axes[nrows - 1][col].set_xlabel(
            f"s [mm], {loaded[0][1].get(tgt, {}).get('coord', '?')}"
            if x_axis == "s" else "radial cell from separatrix (0 = first SOL cell)")
    print("-" * (92 + (13 if third_row else 0)))
    print("  'sep' is s = 0 (the magnetic separatrix); 'peak' is the Te maximum")
    print("  in the window. Values are AT THE SHEATH, 0.5*(domain + guard) —")
    print("  the same cells target.py uses for the heat flux.")
    if third_row == "jsat":
        print("  jsat = Z e N_i |V_i| (Bp/B): the simulation's own ion flux to")
        print("  the plate, no assumed Bohm speed, as a flush LP collects it.")

    fig.suptitle("Plate plasma profiles")
    fig.tight_layout()
    if save_png:
        fig.savefig(save_png, dpi=150)
        print(f"saved {save_png}")
    plt.show()


if __name__ == "__main__":
    main()

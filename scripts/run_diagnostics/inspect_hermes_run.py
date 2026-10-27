"""
inspect_hermes_run_new.py
-------------------------
Inspect and plot diagnostics from a BOUT++ / Hermes-3 run.

opt_plt options:
  check_source   — plot particle/energy source profiles and totals
  inspect_quick  — load normalised dataset for interactive inspection
  connLen        — (reserved)
  flux           — load dataset for flux analysis
  extreme_values — print max/min/avg/CV for key variables; optionally
                   open xhermes dataset for further investigation

plot_std (only active when opt_plt == 'extreme_values'):
  True  — show interactive CV slider for relative fluctuation strength
  False — skip
"""

from __future__ import annotations

import logging
import numpy as np
import netCDF4 as nc
from matplotlib import pyplot as plt
import sys, os, fnmatch, configparser

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
cfg = configparser.ConfigParser()
cfg.read(os.path.join(os.getcwd(), 'config_analysis.ini'))
prepath  = cfg.get('inspect', 'prepath',  fallback='./1/')
opt_plt  = cfg.get('inspect', 'opt_plt',  fallback='check_source')
plot_std = cfg.get('inspect', 'plot_std',  fallback='False')
opt_plt = "check_source"
# opt_plt = "inspect_quick"
# opt_plt = "connLen"
# opt_plt = "flux"
# opt_plt = "extreme_values"

DATANAME = "boutdata.nc"
# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def find_file(pattern: str, search_path: str) -> str | list:
    """Return the first file matching *pattern* under *search_path*, or []."""
    for root, _dirs, files in os.walk(search_path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return []


def _open_xhermes(path: str, gridname: str, yguard: bool,
                  unnormalise: bool = True):
    """Open a Hermes dataset via xhermes, with or without a grid file."""
    import xhermes
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    if len(gridname) == 0:
        return xhermes.open_hermesdataset(path, inputfilepath="./BOUT.inp")
    return xhermes.open_hermesdataset(
        path,
        geometry="toroidal",
        gridfilepath=gridname,
        keep_xboundaries=True,
        keep_yboundaries=yguard,
        unnormalise=unnormalise,
        info=False,
    )


def _make_cv(ds_fa, var: str, t: int, x_slice, ybndry_id: list) -> np.ndarray:
    """Coefficient of variation (std/|mean|) at timestep *t* for *var*."""
    data = np.concatenate(
        [ds_fa[var][t, x_slice, ybndry_id[0]:ybndry_id[1], :],
         ds_fa[var][t, x_slice, ybndry_id[2]:ybndry_id[3], :]],
        axis=1,
    )
    mean_xy = np.mean(data, axis=2)
    std_xy  = np.std(data,  axis=2)
    return std_xy / (np.abs(mean_xy) + 1e-10)


def plot_cv_slider(ds_fa, vars_to_plot: list[str], x_slice, ybndry_id: list) -> None:
    """
    Interactive 2x2 grid of CV (relative fluctuation) maps with a time slider.
    Only plots variables that exist in *ds_fa*.
    """
    from matplotlib.widgets import Slider

    nt        = ds_fa[vars_to_plot[0]].shape[0]
    fig, axes = plt.subplots(2, 2, figsize=(10, 9))
    plt.subplots_adjust(bottom=0.15)
    axes = axes.flatten()

    ims   = []
    valid = []  # (im, var) pairs that got a real plot

    for ax, var in zip(axes, vars_to_plot):
        if var not in list(ds_fa.variables):
            ax.set_title(f"{var} — not found")
            ax.axis("off")
            ims.append(None)
            continue

        cv_xy           = _make_cv(ds_fa, var, 0, x_slice, ybndry_id)
        cv_mean, cv_std = np.mean(cv_xy), np.std(cv_xy)
        vmin = max(0.0, cv_mean - 3 * cv_std)
        vmax = cv_mean + 3 * cv_std

        im = ax.pcolormesh(cv_xy.T, cmap="hot", vmin=vmin, vmax=vmax)
        fig.colorbar(im, ax=ax, label="CV", extend="max")
        ax.set_title(var)
        ax.set_xlabel("x, radial")
        ax.set_ylabel("y, theta")
        ims.append(im)
        valid.append((im, var))

    title    = fig.suptitle("Relative fluctuations CV at t=0", fontsize=13)
    ax_slide = fig.add_axes([0.15, 0.05, 0.7, 0.03])
    slider   = Slider(ax_slide, "timestep", 0, nt - 1, valinit=0, valstep=1)

    def update(val: float) -> None:
        t = int(slider.val)
        for im, var in valid:
            cv_xy           = _make_cv(ds_fa, var, t, x_slice, ybndry_id)
            cv_mean, cv_std = np.mean(cv_xy), np.std(cv_xy)
            im.set_array(cv_xy.T.ravel())
            im.set_clim(max(0.0, cv_mean - 3 * cv_std), cv_mean + 3 * cv_std)
        title.set_text(f"Relative fluctuations CV at t={t}")
        fig.canvas.draw_idle()

    slider.on_changed(update)
    plt.show()



gridname = find_file("*.nc", "../")
gr       = nc.Dataset(gridname)
MYG      = int(gr["y_boundary_guards"][:])
yguard   = MYG == 2

# ---------------------------------------------------------------------------
# Load dataset (where required by opt_plt)
# ---------------------------------------------------------------------------
ds = None

if opt_plt in ("check_source", "flux"):
    ds = _open_xhermes(prepath + DATANAME, gridname, yguard, unnormalise=True)

elif opt_plt == "inspect_quick":
    log.info("normalised boutdata")
    import xhermes
    ds = xhermes.open_hermesdataset(
        prepath + DATANAME,
        geometry="toroidal",
        gridfilepath=gridname,
        keep_xboundaries=True,
        keep_yboundaries=yguard,
        unnormalise=False,
        info=False,
    )

elif opt_plt == "extreme_values":
    pass  # uses raw netCDF — opened below

else:
    log.error("Unknown opt_plt=%r — nothing to do.", opt_plt)
    raise SystemExit(1)

# ---------------------------------------------------------------------------
# Derive geometry from xhermes dataset (when loaded)
# ---------------------------------------------------------------------------
if ds is not None:
    sim_dim = "2D"
    if ds.dims["zeta"] > 1:
        sim_dim = "3D_nn" if "Nd" in list(ds.variables) else "3D"

    SZxy    = np.array(ds["g11"].shape)
    ixseps1 = ds.metadata["ixseps1"]
    ixseps2 = ds.metadata["ixseps2"]
    if ixseps1 > ixseps2:
        ixseps1, ixseps2 = ixseps2, ixseps1

    jyseps1_1 = ds.metadata["jyseps1_1"] + MYG
    jyseps2_1 = ds.metadata["jyseps2_1"] + MYG
    jyseps1_2 = ds.metadata["jyseps1_2"] + MYG * 3
    jyseps2_2 = ds.metadata["jyseps2_2"] + MYG * 3
    if jyseps2_1 != jyseps1_2:
        ny_inner = ds.metadata["ny_inner"]

    # jyseps1_2/jyseps2_2 already carry the +3*MYG region-B offset, so the
    # midpoint must NOT add it again (it did, double-counting the guards by
    # 3*MYG — harmless only because these grids have y_boundary_guards=0).
    y_OMP = int((jyseps1_2 + jyseps2_2) / 2 + 1)
    ny_inner_g = (ds.metadata["ny_inner"] + MYG * 2
                  if jyseps2_1 != jyseps1_2 else None)
    gr.close()

# ---------------------------------------------------------------------------
# check_source
# ---------------------------------------------------------------------------
if opt_plt == "check_source":
    elem_vol = ds["J"] * ds["dx"] * ds["dy"]
    if "t" in elem_vol.dims:          # J/dx/dy sometimes carry a t dim; a
        elem_vol = elem_vol.isel(t=0)  # product with it inflates every total by Nt
    elem_vol = np.asarray(elem_vol.values)
    var_list = list(ds.variables)
    MXG = int(ds.metadata.get("MXG", 2))

    # ---------------------------------------------------------------- regions
    # A source injected near the target does not show up in a single total, so
    # split the domain. The masks tile it exactly once, and the split is
    # checked against the total below rather than assumed.
    nx, ny = elem_vol.shape
    xs = np.arange(nx)[:, None]
    ys = np.arange(ny)[None, :]

    # Full (nx, ny) so the masks can be used to index the 2-D arrays directly.
    # Drops the x-guards AND, on a guard-retaining squash, the y-guard rows.
    # The y-guards were previously swept in: they carry a nonzero grid source
    # and a full volume element, worth +5.6 kW of 619.7 (0.9 %) on
    # from_archer_6kW/408. Small, but it is one of the two reasons this script
    # and v3_radial_conservation.py disagreed on the input power (that one
    # also kept the x-guards, +32.6 kW / 5.3 %).
    _ny_grid = int(ds.metadata.get("ny", ny))
    _nyi = int(ds.metadata.get("ny_inner", 0))
    _MYG = int(ds.metadata.get("MYG", 2))
    _g = _MYG if ny > _ny_grid else 0
    y_interior = np.ones(ny, dtype=bool)
    if _g:
        for _a, _b in ((0, _g), (_nyi + _g, _nyi + 3 * _g),
                       (_ny_grid + 3 * _g, _ny_grid + 4 * _g)):
            y_interior[_a:min(_b, ny)] = False
        print(f"  dropping {int((~y_interior).sum())} y-guard rows from the "
              f"volume integrals")
    inside = ((xs >= MXG) & (xs < nx - MXG)) & y_interior[None, :]
    core_side, sol_side = (xs < ixseps1), (xs >= ixseps1)

    upstream = np.zeros((1, ny), dtype=bool)        # between the X-points
    upstream[0, jyseps1_1 + 1:jyseps2_1 + 1] = True
    upstream[0, jyseps1_2 + 1:jyseps2_2 + 1] = True  # empty for a single null

    legs = {"inner lower leg": (0, jyseps1_1 + 1),
            "outer lower leg": (jyseps2_2 + 1, ny)}
    if ny_inner_g is not None:                      # connected double null
        legs["inner upper leg"] = (jyseps2_1 + 1, ny_inner_g)
        legs["outer upper leg"] = (ny_inner_g, jyseps1_2 + 1)

    regions = {"core": inside & core_side & upstream,
               "SOL (upstream)": inside & sol_side & upstream}
    for name, (y0, y1) in legs.items():
        m = np.zeros((1, ny), dtype=bool)
        m[0, y0:y1] = True
        if m.any():
            regions[name] = inside & m
    # "core" is the only closed-field-line region; everything else is edge.
    edge = {k: v for k, v in regions.items() if k != "core"}

    def integrate(field, mask, factor):
        return float((np.asarray(field) * elem_vol * 2 * np.pi
                      * factor)[mask].sum())

    Sp_list = ["Se_src", "Si_src", "Sd+_src", "Sd_src"]
    Se_list = ["Pe_src", "Pi_src", "Pd+_src", "Pd_src"]

    # Every source variable present gets its own column. The original code
    # kept only the LAST match in `Sp`/`Se`, so the figure and any summary
    # described one species while the totals listed several — and the two
    # disagree whenever the species carry different sources.
    found = []          # (name, field, factor, unit)
    for var in Sp_list:
        if var in var_list:
            found.append((var, ds[var][0, :, :, 0], 1.0, "/s"))
    for var in Se_list:
        if var in var_list:
            # 1.5e-3 = (3/2) * 1e-3, NOT a botched W->kW conversion.
            # P<sp>_src is a PRESSURE source [Pa/s]; the energy density is
            # 3/2 P, so the injected power density is (3/2)*P_src [W/m^3].
            # Hermes-3 states the factor itself -- evolve_pressure.cxx:391:
            #   ddtP_component_source = (2./3) * energy_source
            # so energy->pressure is x2/3 and pressure->power is x3/2.
            # Dropping the 1.5 under-reports every energy source by 33%
            # (303.1 -> 202.1 kW on the SXD nx32 case). The giveaway that
            # 1e-3 is wrong: the radial heat flux at the LCFS there is
            # 568+/-47 kW, which a 404 kW total input could not supply.
            found.append((var, ds[var][0, :, :, 0], 1.5e-3, "kW"))

    # Kept for the profile plots: the first particle and first energy source.
    Sp = next((f for n, f, k, u in found if u == "/s"), None)
    Se = next((f for n, f, k, u in found if u == "kW"), None)
    Sp_name = next((n for n, f, k, u in found if u == "/s"), "Sp")
    Se_name = next((n for n, f, k, u in found if u == "kW"), "Se")

    for name, field, fac, unit in found:
        log.info("total %8s source = %.4e %s", name,
                 integrate(field, inside, fac), unit)

    if not found:
        log.warning("No Se/Sp source variables found in dataset.")
    else:
        totals = [integrate(f, inside, k) for _n, f, k, _u in found]
        pct = lambda v, t: (100 * v / t if t else float("nan"))

        hdr = "%-20s" % "region"
        for (name, _f, _k, unit), _t in zip(found, totals):
            hdr += " %14s %6s" % (f"{name} [{unit}]", "%")
        log.info("\n" + hdr)

        def row(label, mask, indent=""):
            line = "%-20s" % (indent + label)
            for (_n, f, k, _u), tot in zip(found, totals):
                v = integrate(f, mask, k)
                line += " %14.4e %5.1f%%" % (v, pct(v, tot))
            log.info(line)

        row("core", regions["core"])
        edge_mask = np.zeros_like(inside)
        for m in edge.values():
            edge_mask |= m
        row("edge", edge_mask)
        for name, m in edge.items():
            row(name, m, indent="  ")
        line = "%-20s" % "total"
        for t in totals:
            line += " %14.4e %5.1f%%" % (t, 100.0)
        log.info(line)

        # The split is only trustworthy if the parts add up: any overlap or
        # gap in the masks shows here rather than silently mis-attributing
        # a near-target source.
        worst = 0.0
        for (_n, f, k, _u), tot in zip(found, totals):
            s = sum(integrate(f, m, k) for m in regions.values())
            worst = max(worst, abs(s - tot) / abs(tot) if tot else 0.0)
        log.info("regions tile the domain: worst parts-vs-total %.2e", worst)
        if worst > 1e-10:
            log.warning("  ^ regions do NOT tile the domain — the split above "
                        "is unreliable (check jyseps/ny_inner guard offsets)")

    if Se is not None and Sp is not None:
        # Two panels: the radial shape at the OMP (as before) and the
        # POLOIDAL distribution, which is where a near-target source lives and
        # what a single total hides completely.
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))
        ax.plot(Se[:, y_OMP] / Se[MXG:-MXG, y_OMP].max().values, ":",
            label=Se_name)
        ax.plot(Sp[:, y_OMP] / Sp[MXG:-MXG, y_OMP].max().values, "--",
            label=Sp_name)
        ax.axvline(ixseps1, label="ixseps1", color="k", lw=0.8)
        ax.set_xlabel("radial x")
        ax.set_ylabel("/ max amplitude")
        ax.set_title(f"radial shape at OMP (y={y_OMP})")
        ax.legend()

        # Radially integrated, so the curve is the power/particles per
        # poloidal cell — the quantity the region table sums up.
        for field, style, lab, fac in ((Sp, "--", f"{Sp_name} [/s]", 1.0),
                                       (Se, ":", f"{Se_name} [kW]", 1.5e-3)):  # 3/2 * 1e-3, see check_source above
            prof = (np.asarray(field) * elem_vol * 2 * np.pi * fac)
            prof = np.where(inside, prof, 0.0).sum(axis=0)
            ax2.plot(prof / np.abs(prof).max(), style, label=lab)
        for yv, lab in ((jyseps1_1, "X-pt"), (jyseps2_1, None),
                        (jyseps1_2, None), (jyseps2_2, None)):
            ax2.axvline(yv + 0.5, color="k", lw=0.8, ls="-",
                        label=lab if lab else None)
        if ny_inner_g is not None:
            ax2.axvline(ny_inner_g - 0.5, color="0.5", lw=0.8, ls="--",
                        label="upper target")
        ax2.set_xlabel("poloidal y")
        ax2.set_ylabel("/ max, radially integrated")
        ax2.set_title("poloidal distribution")
        ax2.legend(fontsize=8)
        fig.suptitle("check source")
        plt.tight_layout()
        plt.show()

# ---------------------------------------------------------------------------
# extreme_values
# ---------------------------------------------------------------------------
elif opt_plt == "extreme_values":
    # Re-read grid scalars (gr was not closed for this branch)
    jyseps1_1 = int(gr["jyseps1_1"][:]) + MYG
    jyseps2_1 = int(gr["jyseps2_1"][:]) + MYG
    jyseps1_2 = int(gr["jyseps1_2"][:]) + MYG * 3
    jyseps2_2 = int(gr["jyseps2_2"][:]) + MYG * 3
    if jyseps2_1 != jyseps1_2:
        ny_inner = int(gr["ny_inner"][:])
    ny_grid = gr.dimensions["y"].size
    ixseps1 = int(gr["ixseps1"][:])
    gr.close()

    var_list = [
        "Ne", "Pe", "NVe",
        "Ni", "Pi", "NVi",
        "Nd+", "Pd+", "NVd+",
        "Nd", "Pd", "NVd",
        "Vort", "phi",
    ]

    log.info("netCDF data loads")
    ds_raw = nc.Dataset(prepath + DATANAME)

    if ds_raw.dimensions["y"].size > ny_grid:
        log.info("*** y_guard_cells saved in boutdata but not in gridfile ***")
        ybndry_id = [1, ny_inner + 3, ny_inner + 5, ds_raw.dimensions["y"].size - 1]
    else:
        if MYG == 2:
            ybndry_id = [1, ny_inner + 3, ny_inner + 5, ny_grid - 1]
        else:
            ybndry_id = [0, ny_inner + 1, ny_inner, ny_grid]

    for var in var_list:
        if var not in ds_raw.variables:
            continue
        data_tmp = np.concatenate(
            [ds_raw[var][:, 2:-2, ybndry_id[0]:ybndry_id[1], :],
             ds_raw[var][:, 2:-2, ybndry_id[2]:ybndry_id[3], :]],
            axis=2,
        )
        max_v = float(np.max(data_tmp))
        min_v = float(np.min(data_tmp))
        avg_v = float(np.mean(data_tmp))
        std_v = float(np.std(data_tmp))
        cv    = std_v / abs(avg_v) if abs(avg_v) > 1e-10 else float("inf")

        # Per-axis CV diagnostics (uncomment to enable):
        # cv_x = np.std(data_tmp, axis=(1,2)) / (np.abs(np.mean(data_tmp, axis=(1,2))) + 1e-10)
        # cv_y = np.std(data_tmp, axis=(0,2)) / (np.abs(np.mean(data_tmp, axis=(0,2))) + 1e-10)

        id_max = np.unravel_index(np.argmax(data_tmp), data_tmp.shape)
        id_min = np.unravel_index(np.argmin(data_tmp), data_tmp.shape)

        print(
            f"{var:6s}: "
            f"max={max_v:.2e} @(t={id_max[0]},x={id_max[1]+2},y={id_max[2]},z={id_max[3]}), "
            f"min={min_v:.2e} @(t={id_min[0]},x={id_min[1]+2},y={id_min[2]},z={id_min[3]}), "
            f"avg={avg_v:.2e}, CV={cv:.2f}"
        )

    ds_raw.close()

    # Open xhermes dataset for interactive follow-up inspection
    log.info("xarray data loads for inspecting large values")
    ds = _open_xhermes(prepath + DATANAME, gridname, yguard, unnormalise=False)
    log.info("--------------------------------------------------------")
    log.info("Find max/min location with:  (dvar = ds['var'])")
    log.info("  id_max = np.unravel_index(dvar.argmax(), dvar.shape)")
    log.info("  id_min = np.unravel_index(dvar.argmin(), dvar.shape)")

    if plot_std:
        x_slice   = slice(2, -2)   # standard x guard exclusion
        cv_vars   = ["Vort", "phi", "Pi", "Ne"]
        plot_cv_slider(ds, cv_vars, x_slice, ybndry_id)

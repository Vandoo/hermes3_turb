"""
h2time_hermes.py
----------------
Reads and plots time-trace diagnostics from BOUT++ Hermes-3 runs.

- read_h2time: handles quasi-neutral fallback, optional Nd/Td neutral species
- sum_plt: uses time_dict, preserves t_shift stitching from prev version,
           conditionally plots Nd/Td subplots, fixes final pickle dump bug
"""

from __future__ import annotations

import copy
import configparser
import fnmatch
import logging
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import netCDF4 as nc
import pickle as pkl

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
cfg = configparser.ConfigParser()
cfg.read(Path.cwd() / "config_analysis.ini")

dir_1    = cfg.getint    ("h2time", "dir_1",    fallback=1)
dir_2    = cfg.getint    ("h2time", "dir_2",    fallback=10)
pout_opt = cfg.get       ("h2time", "pout_opt", fallback="sum_plt")
jump_old = cfg.getboolean("h2time", "jump_old", fallback=True)

# ---------------------------------------------------------------------------
# Physical normalisations
# ---------------------------------------------------------------------------
# Fallbacks only - these are overwritten below from the dump's own metadata
# (Omega_ci/Nnorm/Tnorm).  Hard-coding them silently rescales the "t (ms)" axis
# whenever a case uses a different Bnorm/Tnorm.
OMEGA_I = 1.447280e+07          # ion gyrofrequency [rad/s]
DT_MS   = 1.0 / OMEGA_I * 1e3  # time step in ms
N_NORM  = 1e+19                 # density normalisation [m^-3]
T_NORM  = 50                    # temperature normalisation [eV]


def read_normalisations(ds: nc.Dataset) -> None:
    """Take Omega_ci / Nnorm / Tnorm from the dump instead of trusting constants."""
    global OMEGA_I, DT_MS, N_NORM, T_NORM
    for var, name in (("Omega_ci", "OMEGA_I"), ("Nnorm", "N_NORM"), ("Tnorm", "T_NORM")):
        if var in ds.variables:
            globals()[name] = float(np.asarray(ds[var][:]).ravel()[0])
        else:
            log.warning("!!! %s not in dump - keeping fallback %s", var, globals()[name])
    DT_MS = 1.0 / OMEGA_I * 1e3
    log.info("normalisations: Omega_ci=%.6e s^-1  Nnorm=%.3e m^-3  Tnorm=%g eV",
             OMEGA_I, N_NORM, T_NORM)

# ---------------------------------------------------------------------------
# Grid helpers
# ---------------------------------------------------------------------------
def find_file(pattern: str, search_path: str) -> str | list:
    """Return the first file matching *pattern* found under *search_path*."""
    for root, _dirs, files in os.walk(search_path):
        for name in files:
            if fnmatch.fnmatch(name, pattern):
                return os.path.join(root, name)
    return []


def load_grid(gridname: str) -> dict:
    """Read separator/boundary indices from the grid NetCDF file."""
    grid_nc = nc.Dataset(gridname)
    info = {
        "SZxy":      np.array(grid_nc["Rxy"].shape),
        "jyseps1_1": int(grid_nc["jyseps1_1"][:]),
        "jyseps2_1": int(grid_nc["jyseps2_1"][:]),
        "jyseps1_2": int(grid_nc["jyseps1_2"][:]),
        "jyseps2_2": int(grid_nc["jyseps2_2"][:]),
        "ixseps1":   int(grid_nc["ixseps1"][:]),
        "ixseps2":   int(grid_nc["ixseps2"][:]),
        "MYG":       int(grid_nc["y_boundary_guards"][:]),
    }
    grid_nc.close()
    # ensure ixseps1 <= ixseps2
    if info["ixseps1"] > info["ixseps2"]:
        info["ixseps1"], info["ixseps2"] = info["ixseps2"], info["ixseps1"]
    return info


# ---------------------------------------------------------------------------
# Load grid
# ---------------------------------------------------------------------------
gridname = find_file("*.nc", "../")
grid     = load_grid(gridname)

ixseps1   = grid["ixseps1"]
MYG       = grid["MYG"]

# Check whether squashed data has extra y guards
_ds_check = nc.Dataset(f"{dir_2}/boutdata.nc", "r")
new_ny    = _ds_check.dimensions["y"].size
read_normalisations(_ds_check)
_ds_check.close()
if MYG == 0 and new_ny > grid["SZxy"][1]:
    log.info("--- squashed data has extra y guards ---")
    MYG = 2

y_IMP = int((grid["jyseps1_1"] + grid["jyseps2_1"]) / 2 + 1) + MYG
y_OMP = int((grid["jyseps1_2"] + grid["jyseps2_2"]) / 2 + 1) + MYG * 3

if not os.path.exists(f"./{dir_2}/sat_time.p"):
    log.info("new data to check out in dir_2")
    pout_opt = "sum_load"


# ---------------------------------------------------------------------------
# Data extraction
# ---------------------------------------------------------------------------
def _zonal_mean(arr, ix: int, iy: int) -> np.ndarray:
    """Return toroidal (z) mean of arr[:, ix, iy, :]."""
    return arr[:, ix, iy, :].mean(axis=-1)


def read_h2time(ds: nc.Dataset, dir_name: str = "./") -> None:
    """
    Extract key time-trace diagnostics from a BOUT++ dataset and pickle them.

    Handles:
    - Alternative time/z dimension names (t_array, zeta)
    - Quasi-neutral fallback: Ne derived from Ni if Ne absent
    - Te derived from Pe/Ne if Te absent
    - Optional ion species: Ni/Ti or Nd+/Td+ 
    - Optional neutral species: Nd/Td
    """
    # Resolve dimension names
    tname = "t_array" if "t_array" in ds.dimensions else "t"

    sel: dict[str, np.ndarray | int] = {"t": ds[tname][:]}

    # Per-folder guard detection: a squashed dump may or may not retain y-guards.
    # Recompute the sampling rows from THIS dump's ny so that mixing guard-retaining
    # (boutdata_guards.nc) and guardless (boutdata.nc) folders does not mislocate the
    # OMP/IMP/target rows (the module-level y_OMP was fixed from dir_2 only).
    ny_dump   = ds.dimensions["y"].size
    ny_grid   = int(grid["SZxy"][1])
    n_targets = 2 if grid["jyseps2_1"] == grid["jyseps1_2"] else 4
    myg = int(round((ny_dump - ny_grid) / n_targets)) if ny_dump > ny_grid else int(grid["MYG"])
    yIMP = int((grid["jyseps1_1"] + grid["jyseps2_1"]) / 2 + 1) + myg
    yOMP = int((grid["jyseps1_2"] + grid["jyseps2_2"]) / 2 + 1) + myg * 3
    idx  = [("imp", yIMP), ("omp", yOMP), ("IL", myg), ("OL", -myg - 1)]

    # ---- Ion species names --------------------------------------------------
    if "Ni" in ds.variables:
        Ni_name, Ti_name, Pi_name = "Ni",  "Ti",  "Pi"
    else:
        Ni_name, Ti_name, Pi_name = "Nd+", "Td+", "Pd+"

    # ---- Electron density (quasi-neutral fallback) --------------------------
    ne_arr = ds["Ne"][:] if "Ne" in ds.variables else ds[Ni_name][:]
    for key, iy in idx:
        sel[f"Ne_{key}"] = _zonal_mean(ne_arr, ixseps1, iy)

    # ---- Electron temperature -----------------------------------------------
    Te_arr = ds["Te"][:] if "Te" in ds.variables else ds["Pe"][:] / ne_arr
    for key, iy in idx:
        sel[f"Te_{key}"] = _zonal_mean(Te_arr, ixseps1, iy)

    # ---- Ion density --------------------------------------------------------
    ni_arr = ds[Ni_name][:]
    for key, iy in idx:
        sel[f"Ni_{key}"] = _zonal_mean(ni_arr, ixseps1, iy)

    # ---- Ion temperature ----------------------------------------------------
    Ti_arr = ds[Ti_name][:] if Ti_name in ds.variables else ds[Pi_name][:] / ni_arr
    for key, iy in idx:
        sel[f"Ti_{key}"] = _zonal_mean(Ti_arr, ixseps1, iy)

    # ---- Neutral density/temperature (optional) -----------------------------
    if "Nd" in ds.variables:
        nd_arr = ds["Nd"][:]
        for key, iy in idx:
            sel[f"Nd_{key}"] = _zonal_mean(nd_arr, ixseps1, iy)
        if "Td" in ds.variables:
            td_arr = ds["Td"][:]
            for key, iy in idx:
                sel[f"Td_{key}"] = _zonal_mean(td_arr, ixseps1, iy)
        sel["include_nn"] = 1
    else:
        sel["include_nn"] = 0

    pkl.dump(sel, open(f"{dir_name}/sat_time.p", "wb"))


# ---------------------------------------------------------------------------
# Helper: initialise an empty time_dict (or from legacy list pickle)
# ---------------------------------------------------------------------------
_STANDARD_KEYS = [
    "t_arr",
    "ne_OMP_arr", "Te_OMP_arr", "ne_IMP_arr", "Te_IMP_arr",
    "ne_IL_arr",  "Te_IL_arr",  "ne_OL_arr",  "Te_OL_arr",
    "ni_OMP_arr", "Ti_OMP_arr", "ni_IMP_arr", "Ti_IMP_arr",
    "ni_IL_arr",  "Ti_IL_arr",  "ni_OL_arr",  "Ti_OL_arr",
]
_NEUTRAL_KEYS  = ["Nd_imp", "Nd_omp", "Nd_IL", "Nd_OL"]
_NEUTRAL_T_KEYS = ["Td_imp", "Td_omp", "Td_IL", "Td_OL"]

_LEGACY_KEY_MAP = {
    "t_arr":      "t_arr",
    "ne_OMP_arr": "ne_OMP_arr", "Te_OMP_arr": "Te_OMP_arr",
    "ne_IMP_arr": "ne_IMP_arr", "Te_IMP_arr": "Te_IMP_arr",
    "ni_OMP_arr": "ni_OMP_arr", "Ti_OMP_arr": "Ti_OMP_arr",
    "ni_IMP_arr": "ni_IMP_arr", "Ti_IMP_arr": "Ti_IMP_arr",
    "ne_IL_arr":  "ne_IL_arr",  "Te_IL_arr":  "Te_IL_arr",
    "ne_OL_arr":  "ne_OL_arr",  "Te_OL_arr":  "Te_OL_arr",
    "ni_IL_arr":  "ni_IL_arr",  "Ti_IL_arr":  "Ti_IL_arr",
    "ni_OL_arr":  "ni_OL_arr",  "Ti_OL_arr":  "Ti_OL_arr",
}


def _empty_time_dict() -> dict:
    return {k: [] for k in _STANDARD_KEYS}


def _from_legacy_list(data: list) -> dict:
    """Convert old list-format pickle into time_dict."""
    (t_prev,
     ne_OMP, Te_OMP, ne_IMP, Te_IMP,
     ne_IL,  Te_IL,  ne_OL,  Te_OL,
     ni_OMP, Ti_OMP, ni_IMP, Ti_IMP,
     ni_IL,  Ti_IL,  ni_OL,  Ti_OL) = data
    return {
        "t_arr":      t_prev,
        "ne_OMP_arr": ne_OMP, "Te_OMP_arr": Te_OMP,
        "ne_IMP_arr": ne_IMP, "Te_IMP_arr": Te_IMP,
        "ne_IL_arr":  ne_IL,  "Te_IL_arr":  Te_IL,
        "ne_OL_arr":  ne_OL,  "Te_OL_arr":  Te_OL,
        "ni_OMP_arr": ni_OMP, "Ti_OMP_arr": Ti_OMP,
        "ni_IMP_arr": ni_IMP, "Ti_IMP_arr": Ti_IMP,
        "ni_IL_arr":  ni_IL,  "Ti_IL_arr":  Ti_IL,
        "ni_OL_arr":  ni_OL,  "Ti_OL_arr":  Ti_OL,
    }


def _concat(time_dict: dict, key: str, new_data: np.ndarray) -> None:
    time_dict[key] = np.concatenate([time_dict[key], new_data])


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _add_ms_axis(ax: plt.Axes, t_arr: np.ndarray, n_ticks: int = 5) -> None:
    """Convert the plotted time coordinates to ms, preserving restart history.

    t_arr already includes any history loaded from prev_time_sat.p. Neither
    reset its origin nor add an offset here: the upper axis must describe the
    same snapshots as the lower axis, including when the first time is nonzero.
    """
    from matplotlib.ticker import ScalarFormatter

    dt_ms = DT_MS
    ax2 = ax.secondary_xaxis(
        'top', functions=(lambda t: t * dt_ms, lambda t_ms: t_ms / dt_ms))
    ax2.set_xticks(np.linspace(float(t_arr[0]) * dt_ms,
                              float(t_arr[-1]) * dt_ms, n_ticks))
    ax2.xaxis.set_major_formatter(ScalarFormatter(useOffset=False))
    ax2.set_xlabel("physical simulation time (ms)", fontsize=14)


def _plot_locations(ax: plt.Axes, t: np.ndarray, scale: float,
                    d: dict, il_key: str, omp_key: str,
                    imp_key: str, ol_key: str, ylabel: str) -> None:
    ax.plot(t, scale * d[il_key],  ":",  label=r"$\mathrm{IL}$")
    ax.plot(t, scale * d[omp_key], "-",  label=r"$\mathrm{OMP}$")
    ax.plot(t, scale * d[imp_key], "-",  label=r"$\mathrm{IMP}$")
    ax.plot(t, scale * d[ol_key],  "--", label=r"$\mathrm{OL}$")
    ax.set_ylabel(ylabel, fontsize=12)
    ax.legend()


# ---------------------------------------------------------------------------
# Main logic
# ---------------------------------------------------------------------------
if pout_opt == "isel":
    pass  # reserved for future interactive selection

elif pout_opt == "sum_load":
    for tmp_name in range(dir_1, dir_2 + 1):
        dir_name = f"./{tmp_name}"
        if not os.path.exists(f"{dir_name}/boutdata.nc"):
            log.info("no boutdata in folder: %s", tmp_name)
            continue
        if not os.path.exists(f"{dir_name}/sat_time.p") or not jump_old:
            log.info("Working in folder: %s", tmp_name)
            ds = nc.Dataset(f"{dir_name}/boutdata.nc")
            read_h2time(ds, dir_name)
            ds.close()

elif pout_opt == "sum_plt":
    log.info("start reading...")
    time_dict: dict = {}
    t_prev_last: float | None = None  # tracks whether we've seen a previous folder
    has_previous_history = False
    continuation_start: float | None = None

    for tmp_name in range(dir_1, dir_2 + 1):
        dir_name = f"./{tmp_name}"

        # ------------------------------------------------------------------ #
        # Initialise time_dict on first iteration                             #
        # ------------------------------------------------------------------ #
        if tmp_name == dir_1:
            prev_pkl = f"{dir_name}/prev_time_sat.p"
            if os.path.exists(prev_pkl):
                raw = pkl.load(open(prev_pkl, "rb"))
                time_dict = _from_legacy_list(raw) if isinstance(raw, list) \
                            else copy.deepcopy(raw)
                t_prev_last = float(time_dict["t_arr"][-1])
                has_previous_history = True
            else:
                time_dict = _empty_time_dict()

            # Pre-allocate neutral keys if the next folder has them
            next_pkl = f"./{tmp_name + 1}/sat_time.p"
            if os.path.exists(next_pkl):
                A_peek = pkl.load(open(next_pkl, "rb"))
                if "Nd_imp" in A_peek and "Nd_imp" not in time_dict:
                    for k in _NEUTRAL_KEYS:
                        time_dict[k] = []
                    if "Td_imp" in A_peek:
                        for k in _NEUTRAL_T_KEYS:
                            time_dict[k] = []

        # ------------------------------------------------------------------ #
        # Append data from this folder                                        #
        # ------------------------------------------------------------------ #
        sat_pkl = f"{dir_name}/sat_time.p"
        if not os.path.exists(sat_pkl):
            log.info("no reading in folder: %s", dir_name)
            continue

        A = pkl.load(open(sat_pkl, "rb"))

        # -- Time stitching ---------------------------------------------------
        # Drop EVERY frame whose time is <= the last already-stitched time, not
        # just the single restart-duplicate first frame. This keeps t_arr
        # strictly increasing even when folders were rolled back / rerun, or a
        # finer-cadence diagnostic run overlaps a production segment (both of
        # which otherwise produce backward/duplicate timestamps in the plot).
        t_full = A["t"].astype(float)
        # Mark the first available continuation output BEFORE removing overlap:
        # previous history may extend beyond the actual restart time.
        if has_previous_history and continuation_start is None and len(t_full):
            continuation_start = float(t_full[0])
        if t_prev_last is None:
            skip = 0
        else:
            skip = int(np.searchsorted(t_full, t_prev_last, side="right"))
        if skip >= len(t_full):
            log.info("%s: fully overlaps existing timeline (t<=%.3f) - skipped",
                     tmp_name, t_prev_last)
            continue
        if skip > 1:
            log.info("%s: dropped %d overlapping frames (t<=%.3f)",
                     tmp_name, skip, t_prev_last)
        t_new  = t_full[skip:]
        _concat(time_dict, "t_arr", t_new)
        t_prev_last = float(t_new[-1])

        log.info("%s: t_array = %s", tmp_name, A["t"])

        # -- Standard fields ------------------------------------------------
        _concat(time_dict, "ne_OMP_arr", A["Ne_omp"][skip:])
        _concat(time_dict, "Te_OMP_arr", A["Te_omp"][skip:])
        _concat(time_dict, "ne_IMP_arr", A["Ne_imp"][skip:])
        _concat(time_dict, "Te_IMP_arr", A["Te_imp"][skip:])
        _concat(time_dict, "ne_IL_arr",  A["Ne_IL"][skip:])
        _concat(time_dict, "Te_IL_arr",  A["Te_IL"][skip:])
        _concat(time_dict, "ne_OL_arr",  A["Ne_OL"][skip:])
        _concat(time_dict, "Te_OL_arr",  A["Te_OL"][skip:])
        _concat(time_dict, "ni_OMP_arr", A["Ni_omp"][skip:])
        _concat(time_dict, "Ti_OMP_arr", A["Ti_omp"][skip:])
        _concat(time_dict, "ni_IMP_arr", A["Ni_imp"][skip:])
        _concat(time_dict, "Ti_IMP_arr", A["Ti_imp"][skip:])
        _concat(time_dict, "ni_IL_arr",  A["Ni_IL"][skip:])
        _concat(time_dict, "Ti_IL_arr",  A["Ti_IL"][skip:])
        _concat(time_dict, "ni_OL_arr",  A["Ni_OL"][skip:])
        _concat(time_dict, "Ti_OL_arr",  A["Ti_OL"][skip:])

        # -- Neutral fields (optional) --------------------------------------
        if "Nd_imp" in A:
            _concat(time_dict, "Nd_imp", A["Nd_imp"][skip:])
            _concat(time_dict, "Nd_omp", A["Nd_omp"][skip:])
            _concat(time_dict, "Nd_IL",  A["Nd_IL"][skip:])
            _concat(time_dict, "Nd_OL",  A["Nd_OL"][skip:])
            if "Td_imp" in A:
                _concat(time_dict, "Td_imp", A["Td_imp"][skip:])
                _concat(time_dict, "Td_omp", A["Td_omp"][skip:])
                _concat(time_dict, "Td_IL",  A["Td_IL"][skip:])
                _concat(time_dict, "Td_OL",  A["Td_OL"][skip:])

    # ---------------------------------------------------------------------- #
    # Plotting                                                                #
    # ---------------------------------------------------------------------- #
    has_neutral_T = "Td_omp" in time_dict
    has_neutral   = "Nd_omp" in time_dict

    if has_neutral_T:
        fig = plt.figure(figsize=(16.2, 7.0))
        layout = {
            "ne": 231, "Te": 233, "Ti": 232,
            "Nd": 234, "Td": 235,
        }
    else:
        fig = plt.figure(figsize=(7.2, 7.0))
        layout = {"ne": 221, "Te": 222, "Ti": 224}

    t = time_dict["t_arr"]

    # -- ne --
    ax = fig.add_subplot(layout["ne"])
    _plot_locations(ax, t, N_NORM, time_dict,
                    "ne_IL_arr", "ne_OMP_arr", "ne_IMP_arr", "ne_OL_arr",
                    r"$n_e$")
    ax.set_xlabel(r"$t$", fontsize=14)
    _add_ms_axis(ax, t)

    # -- Te --
    ax = fig.add_subplot(layout["Te"])
    _plot_locations(ax, t, T_NORM, time_dict,
                    "Te_IL_arr", "Te_OMP_arr", "Te_IMP_arr", "Te_OL_arr",
                    r"$T_e$")

    # -- Ti --
    ax = fig.add_subplot(layout["Ti"])
    _plot_locations(ax, t, T_NORM, time_dict,
                    "Ti_IL_arr", "Ti_OMP_arr", "Ti_IMP_arr", "Ti_OL_arr",
                    r"$T_i$")

    # -- Nd (optional) --
    if has_neutral:
        tim_nd = len(time_dict["Nd_IL"])
        t_nd   = t[-tim_nd:]
        ax = fig.add_subplot(layout["Nd"])
        _plot_locations(ax, t_nd, N_NORM, time_dict,
                        "Nd_IL", "Nd_omp", "Nd_imp", "Nd_OL",
                        r"$N_d$")
        ax.set_xlabel(r"$t$", fontsize=14)

    # -- Td (optional) --
    if has_neutral_T:
        ax = fig.add_subplot(layout["Td"])
        _plot_locations(ax, t_nd, T_NORM, time_dict,
                        "Td_IL", "Td_omp", "Td_imp", "Td_OL",
                        r"$T_d$")

    if continuation_start is not None:
        for panel in fig.axes:
            limits = panel.get_xlim()
            panel.axvline(continuation_start, color='grey', ls='--', lw=1.2)
            panel.set_xlim(limits)
        fig.axes[0].annotate(
            f'Continued output starts\n{continuation_start * DT_MS:.3f} ms',
            xy=(continuation_start, .98), xycoords=('data', 'axes fraction'),
            xytext=(5, -5), textcoords='offset points', va='top',
            fontsize=8, color='dimgray',
            bbox=dict(facecolor='white', edgecolor='none', alpha=.8))

    fig.tight_layout()
    plt.show()

    # ---------------------------------------------------------------------- #
    # Dump (fixed: use time_dict keys, not bare variables)                   #
    # ---------------------------------------------------------------------- #
    pkl.dump(
        [time_dict["t_arr"],
         time_dict["ne_OMP_arr"], time_dict["Te_OMP_arr"],
         time_dict["ne_IMP_arr"], time_dict["Te_IMP_arr"],
         time_dict["ne_IL_arr"],  time_dict["Te_IL_arr"],
         time_dict["ne_OL_arr"],  time_dict["Te_OL_arr"],
         time_dict["ni_OMP_arr"], time_dict["Ti_OMP_arr"],
         time_dict["ni_IMP_arr"], time_dict["Ti_IMP_arr"],
         time_dict["ni_IL_arr"],  time_dict["Ti_IL_arr"],
         time_dict["ni_OL_arr"],  time_dict["Ti_OL_arr"],
         ],
        open("full_time_sat.p", "wb"),
    )

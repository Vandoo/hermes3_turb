#!/usr/bin/env python3
"""
turbulence_monitor.py — is the run turbulent yet, and has it saturated?


  1. Fluctuation growth/saturation (Ben Dudson's recipe)
       δf = f − ⟨f⟩_z            (remove the toroidal average so the metric
                                   is not swamped by slow profile drift)
       amplitude(t) = √(⟨δf²⟩_z)  — RMS over the toroidal direction
     Plotted vs time it shows the linear (exponential) growth phase and the
     saturation plateau. For density/pressure the *relative* level
     √⟨δf²⟩/|⟨f⟩| (%) is the "few to 10s of %" signal of a saturated state;
     for the zero-mean vorticity the absolute RMS is used.

  2. Profile steadiness (the nsep / Tsep check, as in h2_times.py)
     Toroidally-averaged separatrix density and temperature at the outer
     midplane vs time — a turbulent run is "done initializing" once these stop
     trending and just fluctuate about a steady mean.

  3. Monitor location: outer-midplane separatrix by default (OMP, x = ixseps1),
     which is where the interchange/drift-wave drive is strongest. A radial profile
     of the relative fluctuation level at the final saturated window is also shown so
     you can see *where* the turbulence lives.

Loading
-------
Loads one or more output folders and concatenates them in time (v3_* style),
WITHOUT geometry/gridfile and with unnormalise=True (SI). No y-boundary guard
cells are needed — this is a volume diagnostic, not a target one. A single
squashed dump with many time points works too (set dir_range = range(N, N+1)).

Configuration is read from ``config_analysis.ini`` in the current working
directory (same convention as debug_hermes_run.py), section ``[turbulence]``::

    [turbulence]
    data_root    = ./
    dir_start    = 140        ; output folders dir_start..dir_stop-1, concat in time
    dir_stop     = 141
    fields       = Ne, Pe, Vort   ; Ne/Pe use relative %, zero-mean Vort uses abs RMS
    location     = omp_sep    ; omp_sep (OMP separatrix point) | omp_sol (SOL mean)
    sat_window   = 0.5        ; the LAST this-fraction of the run = saturated window
                              ;   (0.5 = last half; 0.2 = last 20%)
    logy         = True       ; log-y on the growth panel (see exponential growth)
    save_png     = turbulence_monitor.png   ; filename, or false to only show (no save)

Every key has a fallback, so an absent file or section just runs the defaults.
Run from the case directory:  python turbulence_monitor.py
"""

import os
import matplotlib
if not os.environ.get("DISPLAY"):
    # Headless cluster node: use a non-interactive backend so plt.show() never
    # blocks forever waiting for a display that isn't there.
    matplotlib.use("Agg")
import configparser
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from xhermes import open_hermesdataset

import hermes3_turb as h3t
from hermes3_turb.load import cache_path, save_cache, load_cache
from hermes3_turb.target import get_midplane_y_indices

# ── CONFIG (from config_analysis.ini [turbulence], with fallbacks) ─────────────
_cfg = configparser.ConfigParser(inline_comment_prefixes=(";", "#"))
_cfg.read(os.path.join(os.getcwd(), "config_analysis.ini"))


def _get(key, fallback):
    return _cfg.get("turbulence", key, fallback=fallback)


data_root    = _get("data_root", "./")
dir_range    = range(int(_get("dir_start", "140")), int(_get("dir_stop", "141")))
fields       = [f.strip() for f in _get("fields", "Ne, Pe, Vort").split(",")
                if f.strip()]
location     = _get("location", "omp_sep")
# sat_window: fraction of the time series, measured FROM THE END, that counts as
# the saturated window (0.5 = last half, 0.2 = last 20%). tstart derives from it.
sat_window   = float(_get("sat_window", "0.5"))
t_stride     = int(_get("t_stride", "1"))           # subsample time every Nth
                                                    # frame on load (1 = keep all)
                                                    # — speeds up the reduction on
                                                    # long runs; growth/saturation
                                                    # trends survive coarse t.
bg_drift_max = float(_get("bg_drift_max", "0.10"))  # max |background drift| over
                                                    # the sat window still called
                                                    # steady; above it → draining/
                                                    # filling (blocks SATURATED).
logy         = _cfg.getboolean("turbulence", "logy", fallback=True)
use_cache    = _cfg.getboolean("turbulence", "use_cache", fallback=True)
# save_png: a filename to save, or false/no/none/off/blank to just show it.
save_png     = _get("save_png", "turbulence_monitor.png")
if save_png.strip().lower() in ("false", "no", "none", "off", ""):
    save_png = None
# ──────────────────────────────────────────────────────────────────────────────

# Fields whose toroidal mean is ~zero → report absolute RMS, not relative %.
ZERO_MEAN_FIELDS = {"Vort", "vorticity", "phi"}


def _find_datapath(folder):
    """Return a loadable data path in *folder*, or None if the folder has no
    data (so the caller can skip it and move on)."""
    import glob
    for c in (os.path.join(folder, "boutdata.nc"),
              os.path.join(folder, "boutdata_guards.nc")):
        if os.path.exists(c):
            return c
    if glob.glob(os.path.join(folder, "BOUT.dmp.*.nc")):
        return os.path.join(folder, "BOUT.dmp.*.nc")
    return None


def _load():
    datasets = []
    for dirnum in dir_range:
        folder = os.path.join(data_root, str(dirnum))
        datapath = _find_datapath(folder)
        if datapath is None:
            print(f"  [skip] no data in {folder}")
            continue
        settings = next((s for s in (os.path.join(data_root, "BOUT.settings"),
                                     os.path.join(folder, "BOUT.settings"),
                                     os.path.join(folder, "BOUT.inp"))
                         if os.path.exists(s)), None)
        print(f"loading {datapath}")
        try:
            dsi = open_hermesdataset(datapath, inputfilepath=settings,
                                     keep_yboundaries=False, unnormalise=True)
        except Exception as e:                                   # noqa: BLE001
            print(f"  [skip] failed to load {datapath}: {e}")
            continue
        # NB do NOT drop each folder's first frame here. A folder squashed to
        # a SINGLE output time then contributes nothing at all, and N such
        # folders collapse to one frame -- measured on
        # sxd_MU/from_archer_6kW/403-408, where folders 404/405/407/408 hold
        # exactly 1 frame each and would vanish, halving the series silently.
        # Restart overlap is removed after concatenation by time VALUE with a
        # running max, as blobs.open_time_series and v3_target_heatflux do
        # (403 and 404 there really do share t=726111.1).
        datasets.append(dsi)

    if not datasets:
        raise SystemExit(f"No loadable folders in {list(dir_range)} "
                         f"under {data_root!r}.")
    if len(datasets) == 1:
        return datasets[0]

    # Concatenate over the intersection of data variables only. Optional
    # diagnostics (e.g. efe_kin_ylow, only written with diagnose=true) may be
    # present in some folders and absent in others; xr.concat's default requires
    # every variable in every dataset, so restrict to the common set first.
    # data_vars/coords="minimal" + compat="override" then concatenate only the
    # time-dependent fields and take time-independent metric from the first.
    common = set(datasets[0].data_vars)
    for d in datasets[1:]:
        common &= set(d.data_vars)
    dropped = set(datasets[0].data_vars) - common
    if dropped:
        print(f"  (concat over {len(common)} common vars; dropped "
              f"{len(dropped)} not in all folders, e.g. "
              f"{', '.join(sorted(dropped)[:4])}...)")
    datasets = [d[sorted(common)] for d in datasets]
    ds = xr.concat(datasets, dim="t", data_vars="minimal",
                   coords="minimal", compat="override")

    # Shared implementation -- handles dump_at_restart true/false and
    # float-noise near-duplicates. See load.dedup_restart_times.
    return h3t.dedup_restart_times(ds)


# Standard background fields always reduced (needed for panel 2 / steadiness),
# on top of whatever `fields` the user configured.
STD_FIELDS = ("Ne", "Pe", "Te", "Ti")


def _reduce_fields():
    """Ordered, de-duplicated list of fields to reduce & cache: the configured
    `fields` plus the standard background set."""
    seen = []
    for f in list(fields) + list(STD_FIELDS):
        if f not in seen:
            seen.append(f)
    return seen


def _compute():
    """Load the folders and reduce each field at the OMP poloidal plane to two
    small ``(t, x)`` arrays — the toroidal mean ``⟨f⟩_ζ(t,x)`` and the
    fluctuation RMS ``√⟨δf²⟩_ζ(t,x)``.

    THIS is the expensive step (it reads every timestep of the 3-D fields). The
    verdict and all three panels derive from these 2-D arrays alone — the
    monitor-location series is a column/slice, the radial profile a time-average,
    etc. — so they are what we cache. A rerun to change sat_window, the
    thresholds, or location is then instant and needs no reload.
    """
    ds = _load()
    pdim = h3t.detect_poloidal_dim(ds)
    zname = h3t.detect_toroidal_dim(ds)
    if t_stride > 1 and "t" in ds.dims:
        ds = ds.isel(t=slice(None, None, t_stride))
        print(f"  time-subsampled by {t_stride}× → {ds.sizes['t']} frames")
    m = ds.metadata
    ixsep = int(m.get("ixseps1", ds.sizes["x"] // 2))
    y_omp = get_midplane_y_indices(ds)["omp"]
    t = np.asarray(ds["t"].values, dtype=float)

    out = {"t": t, "ixsep": ixsep, "y_omp": y_omp,
           "nx": int(ds.sizes["x"]), "reduced": []}
    for f in _reduce_fields():
        if f not in ds.data_vars:
            continue
        # Read the OMP y-plane ONCE into memory (zeta last), then reduce with
        # numpy. Calling toroidal_mean() and fluctuation_rms() on the lazy array
        # separately would re-read it from disk each time (and recompute the mean
        # a third time inside fluctuation_rms) — the source of the slowness.
        at_omp = ds[f].isel({pdim: y_omp})                    # (t, x, zeta) lazy
        order = [d for d in at_omp.dims if d != zname] + [zname]
        arr = np.asarray(at_omp.transpose(*order).values)     # single disk read
        mean = arr.mean(axis=-1)                              # (t, x)
        rms = np.sqrt(np.mean((arr - mean[..., None]) ** 2, axis=-1))
        out[f"mean_{f}"] = mean
        out[f"rms_{f}"] = rms
        out["reduced"].append(f)
        print(f"  reduced {f} at OMP plane → (t,x)={mean.shape}")
    return out


def _get_data():
    """Return the reduced-data dict, from ./cache if present else by computing
    (and then caching) it. The cache key includes dir_range and the reduced-field
    list, so changing `fields` transparently invalidates it."""
    cf = cache_path("turbulence_monitor_d2", dir_range,
                    "-".join(_reduce_fields()), f"t{t_stride}")
    if use_cache and os.path.exists(cf):
        print(f"using cached reductions: {cf}")
        print("  (set use_cache = false in [turbulence] to force recompute)")
        return load_cache(cf)
    data = _compute()
    save_cache(cf, **data)
    print(f"cached reductions → {cf}")
    return data


def _reduce_loc(arr_tx, ixsep):
    """Collapse a cached ``(t, x)`` array to a ``(t,)`` monitor-location series."""
    if location == "omp_sep":
        return arr_tx[:, ixsep]
    # "omp_sol": mean over the SOL side (x ≥ separatrix), dropping x-guards.
    return arr_tx[:, ixsep:-2].mean(axis=1)


def _slope_per_efold(t, y):
    """Growth rate γ [1/time] from a log-linear fit over the growth phase, and
    the fractional change over the saturated window (for the plateau test)."""
    good = np.isfinite(y) & (y > 0)
    if good.sum() < 3:
        return np.nan
    lt, ly = t[good], np.log(y[good])
    # fit over the first half (growth), where it is roughly exponential
    n = max(3, len(lt) // 2)
    return np.polyfit(lt[:n], ly[:n], 1)[0]


def main():
    data = _get_data()
    t = np.asarray(data["t"], dtype=float)
    ixsep = int(data["ixsep"])
    y_omp = int(data["y_omp"])
    nx = int(data["nx"])
    reduced = list(data["reduced"])
    tstart = int((1.0 - sat_window) * len(t))

    present = [f for f in fields if f in reduced]
    missing = [f for f in fields if f not in reduced]
    if missing:
        print(f"  (skipping absent fields: {', '.join(missing)})")
    if not present:
        raise SystemExit(f"None of {fields} present in dataset.")

    print(f"\nmonitor location: {location}  (OMP y={y_omp}, separatrix x={ixsep})")
    print(f"time points: {len(t)}   saturated window: t ≥ {t[tstart]:.3g} "
          f"(last {100*sat_window:.0f}%)\n")

    series = {}   # name -> dict(abs, rel, mean, is_zero_mean)
    for f in present:
        abs_rms = _reduce_loc(data[f"rms_{f}"], ixsep)
        mean = _reduce_loc(data[f"mean_{f}"], ixsep)
        zm = f in ZERO_MEAN_FIELDS
        rel = np.full_like(t, np.nan) if zm else 100.0 * abs_rms / np.abs(mean)
        series[f] = dict(abs=abs_rms, rel=rel, mean=mean, zm=zm)

    # ── verdict ────────────────────────────────────────────────────────────────
    # Two independent conditions must BOTH hold for "SATURATED":
    #   (1) Δ_fluc  — the relative-fluctuation amplitude has plateaued
    #   (2) bg drift — the background ⟨f⟩ itself has stopped trending
    # Checking only (1) gives a false positive when ⟨f⟩ drains and the absolute
    # fluctuation drains with it (their ratio stays flat) — hence (2).
    print(f"field   growth γ[1/t]   sat.level    Δ_fluc    bg drift   "
          f"verdict  (bg steady if |drift|<{100*bg_drift_max:.0f}%)")
    print("-" * 84)
    for f, s in series.items():
        amp = s["abs"] if s["zm"] else s["rel"]
        gamma = _slope_per_efold(t, amp)
        sat = amp[tstart:]
        sat = sat[np.isfinite(sat)]
        if sat.size == 0:
            continue
        level = np.nanmean(sat)
        drift = (np.nanmean(sat[len(sat)//2:]) - np.nanmean(sat[:len(sat)//2])) \
            / (level + 1e-30)
        # background drift over the saturated window (the panel-2 quantity)
        bg = s["mean"]
        bg_sw = np.nanmean(bg[tstart:])
        bg_drift = (bg[-1] - bg_sw) / (abs(bg_sw) + 1e-30)

        unit = "" if s["zm"] else "%"
        if s["zm"]:
            # zero-mean field (Vort): no meaningful background level to track,
            # so the bg-drift gate does not apply.
            growing = np.isfinite(gamma) and gamma > 0 and amp[-1] > 5 * amp[0]
            verdict = ("growing" if growing else
                       "steady/saturated" if abs(drift) < 0.15 else "drifting")
            bg_str = "     —  "
        else:
            bg_str = f"{100*bg_drift:+6.1f}%"
            if level < 1.0:
                verdict = "laminar (no turbulence yet)"
            elif abs(bg_drift) >= bg_drift_max:
                verdict = ("draining (bg↓)" if bg_drift < 0 else "filling (bg↑)")
            elif abs(drift) < 0.15:
                verdict = "SATURATED" if level > 2 else "weakly turbulent"
            else:
                verdict = "still growing" if drift > 0 else "decaying"
        print(f"{f:6s}  {gamma:11.4g}   {level:7.3g}{unit:<2s}  "
              f"{100*drift:+6.1f}%   {bg_str:>8s}   {verdict}")

    # steadiness of the background profile (nsep / Tsep style), always at the
    # OMP separatrix regardless of the monitor `location`.
    print("\nprofile steadiness at OMP separatrix (last vs sat-window mean):")
    for f in STD_FIELDS:
        if f not in reduced:
            continue
        mean = data[f"mean_{f}"][:, ixsep]
        sw = np.nanmean(mean[tstart:])
        print(f"  {f:4s} = {mean[-1]:.4g}   (sat-window mean {sw:.4g}, "
              f"drift {100*(mean[-1]-sw)/(sw+1e-30):+.1f}%)")

    # ── figure ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))

    # panel 1: fluctuation growth/saturation.
    # Relative-% fields (Ne/Pe) and absolute-RMS fields (Vort) are DIFFERENT
    # quantities, so they get separate y-axes: relative % on the left, absolute
    # RMS on a twin right axis. Never share one axis / one label between them.
    ax0r = None
    lines, labels = [], []
    for f, s in series.items():
        if s["zm"]:
            if ax0r is None:
                ax0r = ax[0].twinx()
            (ln,) = ax0r.plot(t, s["abs"], marker=".", ms=3, ls="--",
                              color="0.4", label=f + "  √⟨δf²⟩ (abs)")
        else:
            (ln,) = ax[0].plot(t, s["rel"], marker=".", ms=3,
                               label=f + "  √⟨δf²⟩/|⟨f⟩| [%]")
        lines.append(ln); labels.append(ln.get_label())
    ax[0].axvspan(t[tstart], t[-1], color="0.85", zorder=0)
    ax[0].set_xlabel("time")
    ax[0].set_ylabel("relative fluctuation  √⟨δf²⟩/|⟨f⟩|  [%]")
    ax[0].set_title("turbulence growth / saturation")
    if logy:
        ax[0].set_yscale("log")
    if ax0r is not None:
        ax0r.set_ylabel("absolute RMS  √⟨δf²⟩  (zero-mean fields)")
        if logy:
            ax0r.set_yscale("log")
    ax[0].legend(lines, labels, fontsize=8); ax[0].grid(alpha=0.3)

    # panel 2: background profile steadiness (nsep / Tsep). Curves are
    # normalised to their own sat-window mean so trends are comparable, but that
    # hides the physical magnitude — so annotate each line with its absolute
    # last value (and units) at the right edge.
    _UNITS = {"Ne": "m⁻³", "Pe": "Pa", "Te": "eV", "Ti": "eV"}
    for f in STD_FIELDS:
        if f in reduced:
            mean = data[f"mean_{f}"][:, ixsep]
            norm = mean / (np.nanmean(mean[tstart:]) + 1e-30)
            (ln,) = ax[1].plot(t, norm, marker=".", ms=3, label=f)
            ax[1].annotate(f" {mean[-1]:.3g} {_UNITS[f]}",
                           xy=(t[-1], norm[-1]), xytext=(3, 0),
                           textcoords="offset points", va="center",
                           fontsize=7, color=ln.get_color(),
                           annotation_clip=False)
    ax[1].axvspan(t[tstart], t[-1], color="0.85", zorder=0)
    ax[1].axhline(1.0, color="k", lw=0.6, ls="--")
    ax[1].set_xlabel("time")
    ax[1].set_ylabel("⟨f⟩_z  / sat-window mean")
    ax[1].set_title("OMP-separatrix profile steadiness")
    ax[1].legend(fontsize=8, loc="best"); ax[1].grid(alpha=0.3)
    # leave headroom on the right for the absolute-value annotations
    ax[1].margins(x=0.18)

    # panel 3: radial profile of relative fluctuation at OMP, saturated window.
    # Time-average the cached (t,x) reductions over the saturated window, then
    # form the ratio — matches ⟨√⟨δf²⟩⟩_t / |⟨⟨f⟩⟩_t| along x.
    xr_ = np.arange(nx) - ixsep
    for f, s in series.items():
        if s["zm"]:
            continue
        rms_x = np.asarray(data[f"rms_{f}"])[tstart:].mean(axis=0)
        mean_x = np.abs(np.asarray(data[f"mean_{f}"])[tstart:].mean(axis=0))
        ax[2].plot(xr_, 100.0 * rms_x / mean_x, marker=".", ms=3, label=f + " (%)")
    ax[2].axvline(0, color="k", lw=0.6, ls="--", label="separatrix")
    ax[2].set_xlabel("x − ixseps1  (radial index from separatrix)")
    ax[2].set_ylabel("relative fluctuation [%]")
    ax[2].set_title(f"radial fluctuation profile (last {100*sat_window:.0f}%)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=0.3)

    fig.tight_layout()
    if save_png:
        fig.savefig(save_png, dpi=130)
        print(f"\nsaved {save_png}")
    # Headless nodes: matplotlib fell back to Agg, so plt.show() would return
    # immediately without ever displaying — save a file instead so the run isn't
    # silently useless, and never block.
    if matplotlib.get_backend().lower() == "agg":
        if not save_png:
            fallback = "turbulence_monitor.png"
            fig.savefig(fallback, dpi=130)
            print(f"\nno display detected (Agg backend) — saved {fallback} "
                  "instead of showing.")
    else:
        plt.show()


if __name__ == "__main__":
    main()

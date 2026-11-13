#!/usr/bin/env python3
"""Overlay target heat-flux profiles from v3_target_heatflux caches.

Edit CONFIG, as in compare_omp_profiles.py, or supply repeated --case LABEL CACHE
arguments. No dumps or xhermes required. Each target gets its own panel; no fits
are made. The mean is formed from cached per-snapshot heat fluxes, with optional
sample standard deviation over time (ddof=1), not an uncertainty on the mean.
The cached heat-flux definition and averaging window are preserved.
"""
import argparse
from pathlib import Path
import warnings

import matplotlib.pyplot as plt
import numpy as np

import hermes3_turb as h3t
from hermes3_turb.profile_time import profile_times, time_label

time_label_mode = "span"  # "span": Δt in ms; "range": start–end in ms

# ── CONFIG ────────────────────────────────────────────────────────────────────
root = "/pitagora_work/FUPA2_SOLBOU10/hermes3/"
CASES = [
    # (label, target cache relative to root or absolute)
    # ("nx32 ny32", "case/data/cache/v3_target_heatflux_dirs23-28_i_omp_..._fxv8.npz"),
    # ("nx68 ny32", "other/data/cache/v3_target_heatflux_dirs32-33_i_omp_..._fxv8.npz"),
    ("nx32 ny32",  "sxd_MU/nx32_ny32_45456/data/cache/v3_target_heatflux_dirs28_i_omp_phig_nocap_diag_Z1_rowmid_fxv8.npz"),
    #("nx68 ny32",  "sxd_MU/nx68_ny32_45456/data/cache/v3_target_heatflux_dirs30-33_i_omp_phig_nocap_diag_Z1_rowmid_fxv8.npz"),
    #("nx132 ny32", "sxd_MU/nx132_ny32_45456/data/cache/v3_target_heatflux_dirs30-34_i_omp_phig_nocap_diag_Z1_rowmid_fxv8.npz"),
    ("nx68 ny44", "sxd_MU/nx68_ny44_45456/data/cache/v3_target_heatflux_dirs24-30_i_omp_phig_nocap_diag_Z1_rowmid_fxv8.npz"),
    ("nx68 ny52", "sxd_MU/nx68_ny52_45456/data/cache/v3_target_heatflux_dirs26-27_i_omp_phig_nocap_diag_Z1_rowmid_fxv8.npz"),
    ("nx68 ny80", "sxd_MU/nx68_ny80_45456/data/cache/v3_target_heatflux_dirs26-27_i_omp_phig_nocap_diag_Z1_rowmid_fxv8.npz")
]
mxg = 2                       # x-guards removed at BOTH ends; 0 keeps all cells
targets_to_plot = None        # None = union of available targets
xlim_mm = (-5, 40)            # distance from separatrix in the CACHED coordinate
band = True
band_n = 1.0                  # ±band_n sigma_t
normalise = None              # None | "peak" (peak within xlim_mm)
logy = False
save_png = None               # e.g. "compare_target_profiles.png"
# ──────────────────────────────────────────────────────────────────────────────


def read_profiles(path, guards=2):
    """Read flattened target caches without importing the dump-loading script."""
    if guards < 0:
        raise ValueError("mxg must be nonnegative")
    cache = h3t.load_cache(path)
    names = np.atleast_1d(cache.get("target_names", [])).tolist()
    scalars = cache.get("target_scalars", {}) or {}
    profiles = {}
    time_s = None
    time_checked = False
    for name in names:
        name = name.decode() if isinstance(name, bytes) else str(name)
        prefix = f"tgt_{name}__"
        data = dict(scalars.get(name, {}))
        data.update({k[len(prefix):]: v for k, v in cache.items()
                     if k.startswith(prefix)})
        s = np.asarray(data["s_mm"], dtype=float)
        q = np.asarray(data["q_MWm2"], dtype=float)
        if s.ndim != 1 or q.shape != s.shape or s.size <= 2 * guards:
            raise ValueError(f"{path}: {name}: invalid radial shape or too many x-guards")
        sl = slice(guards, -guards if guards else None)
        s, q = s[sl], q[sl]
        frames = data.get("q_t_MWm2")
        sd, nt = None, 0
        if frames is not None:
            frames = np.asarray(frames, dtype=float)
            if frames.ndim != 2 or frames.shape[1] != len(data["s_mm"]) or not len(frames):
                raise ValueError(f"{path}: {name}: invalid per-frame heat-flux shape")
            frames = frames[:, sl]
            if not np.isfinite(frames).all():
                raise ValueError(f"{path}: {name}: nonfinite per-frame heat flux")
            nt = len(frames)
            q = frames.mean(axis=0)
            if nt > 1:
                sd = frames.std(axis=0, ddof=1)
        else:
            warnings.warn(f"{path}: {name}: no per-frame flux; using cached mean "
                          "(may be q of mean fields); no time band", stacklevel=2)
        if not np.isfinite(s).all() or not np.isfinite(q).all() or not np.all(np.diff(s) > 0):
            raise ValueError(f"{path}: {name}: nonfinite profile or non-increasing coordinate")
        coord = data.get("coord_label")
        if coord not in ("OMP-mapped (ψ)", "IMP-mapped (ψ)", "OMP-mapped", "IMP-mapped",
                         "target (⊥ flux surf)", "wetted (along plate)"):
            raise ValueError(f"{path}: {name}: missing or unsupported coord_label: {coord!r}")
        if coord in ("OMP-mapped", "IMP-mapped"):
            warnings.warn(f"{path}: {name}: midplane-row mapping can distort distances "
                          "on coarse grids; regenerate with gridfile for true ψ mapping",
                          stacklevel=2)
        if not time_checked:
            time_s = profile_times(cache, path, expected_frames=nt or None)
            time_checked = True
        elif time_s is not None and nt and len(time_s) != nt:
            raise ValueError(f"{path}: {name}: frame count differs from timestamps")
        profiles[name] = dict(s=s, q=q, sd=sd, nt=nt, coord=coord, time_s=time_s)
    if not profiles:
        raise ValueError(f"{path}: no flattened target profiles; regenerate with v3_target_heatflux.py")
    return profiles


def compare(cases, *, root_dir=".", guards=2, targets=None, limits=(-5, 40),
            show_band=True, nsigma=1., normalization=None, logarithmic=False):
    """Return a figure; reject different cached coordinate definitions per target."""
    if normalization not in (None, "peak"):
        raise ValueError("normalization must be None or 'peak'")
    if not np.isfinite(limits).all() or limits[0] >= limits[1]:
        raise ValueError("x limits must be finite and increasing")
    if not np.isfinite(nsigma) or nsigma < 0:
        raise ValueError("band multiplier must be finite and nonnegative")
    loaded = [(label, read_profiles(Path(root_dir) / path, guards)) for label, path in cases]
    if not loaded:
        raise ValueError("Add entries to CASES or supply --case LABEL CACHE")
    for label, profiles in loaded:
        first_profile = next(iter(profiles.values()))
        print(f"  {label}: averaging interval {time_label(first_profile['time_s'], mode='range')}")
    available = list(dict.fromkeys(t for _, profiles in loaded for t in profiles))
    if targets is None:
        order = ("inner_upper", "outer_upper", "inner_lower", "outer_lower")
        targets = [t for t in order if t in available]
        targets += [t for t in available if t not in order]
    else:
        targets = list(dict.fromkeys(targets))
    if not targets or any(t not in available for t in targets):
        raise ValueError(f"Choose targets from: {', '.join(available)}")
    # Check before creating a figure, so a failed comparison leaves no blank plot.
    for target in targets:
        coords = {p[target]["coord"] for _, p in loaded if target in p}
        if len(coords) != 1:
            raise ValueError(f"{target}: incompatible coordinates: {sorted(coords)}; "
                             "regenerate caches with the same mapping")
    ncols = min(2, len(targets))
    fig, axes = plt.subplots((len(targets) + ncols - 1) // ncols, ncols,
                             figsize=(6 * ncols, 4.2 * ((len(targets) + ncols - 1) // ncols)),
                             squeeze=False)
    print(f"{'case':24s} {'target':14s} {'frames':>7s} {'peak MW/m²':>12s} {'s_peak mm':>10s}")
    for ax, target in zip(axes.flat, targets):
        for k, (label, profiles) in enumerate(loaded):
            if target not in profiles:
                warnings.warn(f"{label}: {target} absent", stacklevel=2)
                continue
            p = profiles[target]
            s, q, sd = p["s"], p["q"], p["sd"]
            mask = (s >= limits[0]) & (s <= limits[1])
            if not mask.any():
                plt.close(fig)
                raise ValueError(f"{label}: {target}: no cells in plotted window")
            peak_idx = np.flatnonzero(mask)[np.argmax(q[mask])]
            peak = q[peak_idx]
            scale = peak if normalization == "peak" else 1.
            if scale <= 0:
                plt.close(fig)
                raise ValueError(f"{label}: {target}: cannot normalize a nonpositive peak")
            color = f"C{k % 10}"
            count = time_label(p["time_s"], mode=time_label_mode)
            ax.plot(s[mask], q[mask] / scale, ".-", ms=3, color=color,
                    label=f"{label} ({count})")
            if show_band and sd is not None:
                lower, upper = (q - nsigma * sd) / scale, (q + nsigma * sd) / scale
                if logarithmic:
                    lower = np.where(lower > 0, lower, np.nan)
                ax.fill_between(s[mask], lower[mask], upper[mask], color=color, alpha=.2, lw=0)
            print(f"{label:24s} {target:14s} {p['nt']:7d} {peak:12.5g} {s[peak_idx]:10.4g}")
        ax.set_title(target.replace("_", " "))
        ax.axvline(0, color="k", lw=.6)
        ax.set_xlim(*limits)
        ax.set_xlabel(f"Distance from separatrix [mm], {p['coord']}")
        ax.set_ylabel("q / peak in window" if normalization == "peak" else r"$q$ [MW m$^{-2}$]")
        if logarithmic:
            ax.set_yscale("log")
        ax.grid(alpha=.3)
        ax.legend(fontsize=8)
    for ax in list(axes.flat)[len(targets):]:
        ax.set_visible(False)
    title = "Target heat-flux profiles"
    if show_band:
        title += rf" (bands: $\pm{nsigma:g}\sigma_t$ where available)"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", nargs=2, action="append", metavar=("LABEL", "CACHE"),
                        help="repeat for each case; replaces CONFIG CASES")
    parser.add_argument("--root", default=root)
    parser.add_argument("--targets", nargs="+", default=targets_to_plot)
    parser.add_argument("--mxg", type=int, default=mxg)
    parser.add_argument("--xlim", nargs=2, type=float, default=xlim_mm)
    parser.add_argument("--normalise", choices=("none", "peak"), default=normalise or "none")
    parser.add_argument("--no-band", action="store_true")
    parser.add_argument("--band-n", type=float, default=band_n)
    parser.add_argument("--logy", action="store_true", default=logy)
    parser.add_argument("--save", default=save_png)
    parser.add_argument("--no-show", action="store_true")
    args = parser.parse_args()
    try:
        fig = compare(args.case if args.case is not None else CASES, root_dir=args.root,
                      guards=args.mxg, targets=args.targets, limits=args.xlim,
                      show_band=band and not args.no_band, nsigma=args.band_n,
                      normalization=None if args.normalise == "none" else args.normalise,
                      logarithmic=args.logy)
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
    if args.save:
        fig.savefig(args.save, dpi=150)
        print(f"saved {args.save}")
    if not args.no_show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()

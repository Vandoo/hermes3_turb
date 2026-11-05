"""Physical-time labels for cached profile comparisons."""
from pathlib import Path
import re
import warnings

import numpy as np


def profile_times(cache, path, expected_frames=None):
    """Read seconds from the cache, or recover legacy times from squash metadata.

    Recovery follows the original script's file preference and restart handling.
    No fields are loaded and no cache is changed. A frame-count mismatch is
    refused rather than inferring duration from a nominal dump interval.
    """
    try:
        if "t_s" in cache:
            times = np.asarray(cache["t_s"], dtype=float)
        else:
            import netCDF4 as nc
            import xarray as xr
            from .load import dedup_restart_times

            path = Path(path)
            match = re.search(r"_dirs(\d+)(?:-(\d+))?_", path.name)
            if match is None:
                raise ValueError("cache name does not identify its output folders")
            first = int(match[1])
            last = int(match[2] or first)
            radial = "Ne_omp" in cache
            names = (("boutdata.nc", "boutdata_guards.nc", "boutdata_yguards.nc")
                     if radial else
                     ("boutdata_guards.nc", "boutdata_yguards.nc", "boutdata.nc"))
            chunks = []
            for number in range(first, last + 1):
                folder = path.parent.parent / str(number)
                source = next((folder / name for name in names if (folder / name).exists()), None)
                if source is None:
                    continue  # Original loaders skip missing output folders too.
                with nc.Dataset(source) as ds:
                    key = "t_array" if "t_array" in ds.variables else "t"
                    raw = np.asarray(ds[key][:], dtype=float)
                    omega = float(ds["Omega_ci"][...])
                    if not np.isfinite(omega) or omega <= 0:
                        raise ValueError(f"invalid Omega_ci in {source}")
                    time = raw / omega
                if not radial and chunks:
                    time = time[time > max(float(t.max()) for t in chunks)]
                if time.size:
                    chunks.append(time)
            if not chunks:
                raise ValueError("no source squash timestamps available")
            times = np.concatenate(chunks)
            if radial:
                ds = xr.Dataset(coords={"t": times})
                times = dedup_restart_times(ds, verbose=False)["t"].values
            warnings.warn(
                f"{path}: recovered legacy cache times from current squash metadata; "
                "recompute the cache to record its timestamps permanently",
                stacklevel=2)
        if times.ndim != 1 or not times.size or not np.isfinite(times).all():
            raise ValueError("timestamps must be a nonempty finite 1-D array")
        if times.size > 1 and not np.all(np.diff(times) > 0):
            raise ValueError("timestamps must increase strictly")
        if expected_frames is not None and len(times) != expected_frames:
            raise ValueError(f"{len(times)} timestamps but {expected_frames} cached frames; "
                             "source files may have changed since caching")
        return times
    except (OSError, KeyError, ValueError, ImportError) as exc:
        warnings.warn(f"{path}: averaging time unavailable ({exc})", stacklevel=2)
        return None


def time_label(times, mode="span"):
    """Format an averaging span or start/end interval in ms; never assume dt."""
    if mode not in ("span", "range"):
        raise ValueError("time label mode must be 'span' or 'range'")
    if times is None or not len(times):
        return "time unavailable"
    ms = np.asarray(times) * 1e3
    if len(ms) == 1:
        return f"t={ms[0]:.1f} ms, single snapshot"
    if mode == "range":
        return f"{ms[0]:.1f}–{ms[-1]:.1f} ms"
    return f"Δt={ms[-1] - ms[0]:.1f} ms"

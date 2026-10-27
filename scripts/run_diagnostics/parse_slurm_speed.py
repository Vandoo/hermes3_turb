#!/usr/bin/env python3
"""
parse_slurm_speed.py
--------------------
Parses slurm-*.out files within a given job-ID range to extract:
  - Simulation time range (first and last sim time on data lines)
  - Wall-clock run time (from "Run time : Xh Xm Xs")
  - Computing speed = delta_sim_time / run_time_seconds
    where delta_sim_time = sim_end - sim_start (physics time covered per job)

The sim-time BOUT++ prints is **normalised** (units of 1/Omega_ci).  This script
also converts it to physical time, the same way h2time_hermes.py does:

    t_physical [s] = t_normalised / Omega_ci        Omega_ci = qe*Bnorm/Mp

Note Hermes-3 defines Omega_ci with the **proton** mass and no AA factor
(hermes-3.cxx:196), so a deuterium case does NOT get an extra 1/2 - assuming it
does is the 6.6x error that was once baked into h2time_hermes.py.  Omega_ci is
therefore read from the dump (or derived from Bnorm) rather than hard-coded; if
it cannot be found the physical-time columns are omitted instead of guessed.

Two extra columns and a "real time" summary are reported:
  - dt [us]   : physical time covered by that job
  - us/day    : physical microseconds simulated per day of wall clock,
                the practical "how long until this run is useful?" number

Usage:
    python parse_slurm_speed.py --start 613958 --end 669679 [--dir /path/to/logs]

    --start      First job ID to include (inclusive)
    --end        Last job ID to include (inclusive)
    --dir        Directory containing slurm-*.out files (default: current dir)
    --omega-ci   Omega_ci [1/s] override (skips auto-detection)
    --bnorm      Bnorm [T]; Omega_ci is derived as qe*Bnorm/Mp
    --dump       Explicit BOUT dump / grid .nc to read Omega_ci or Bnorm from
    --target     Physical time goal [us]; adds an ETA to the summary

Omega_ci is searched for in --dir, --dir/data and --dir's parent, checking each
directory itself and then its numbered restart folders (newest first), then
BOUT.settings / BOUT.inp.  That covers the usual layout where the slurm logs sit
one level above the dumps:

    case_A/slurm-*.out
    case_A/data/146/boutdata.nc

The header line reports which file the value came from; if it says UNKNOWN, pass
--dump / --bnorm / --omega-ci.
"""

import re
import sys
import glob
import os
import argparse
import statistics

# ── Physical constants (SI, as used by Hermes-3) ──────────────────────────────
QE = 1.602176634e-19   # elementary charge [C]
MP = 1.67262192369e-27 # proton mass [kg]  -- Hermes-3 uses Mp, NOT AA*Mp


def omega_ci_from_bnorm(bnorm):
    """Hermes-3 reference frequency: Omega_ci = qe*Bnorm/Mp (hermes-3.cxx:196).

    Deliberately no AA factor - Hermes-3 normalises with the proton mass whatever
    the ion species. Dividing by AA here would slow every reported physical time
    by that factor (the 6.6x bug that once lived in h2time_hermes.py).
    """
    return QE * bnorm / MP


def _read_nc_scalar(path, names):
    """Return the first of *names* found as a scalar in the NetCDF file at
    *path*, or None. Silent on any failure - this is a best-effort lookup."""
    try:
        import netCDF4 as nc
    except ImportError:
        return None
    try:
        with nc.Dataset(path) as ds:
            for name in names:
                if name in ds.variables:
                    import numpy as np
                    return float(np.asarray(ds[name][:]).ravel()[0])
    except Exception:
        return None
    return None


def _read_settings_bnorm(path):
    """Pull 'Bnorm = <float>' out of a BOUT.settings / BOUT.inp file."""
    try:
        with open(path, "r", errors="replace") as fh:
            for line in fh:
                line = line.split("#", 1)[0].split("//", 1)[0]
                m = re.match(r"\s*Bnorm\s*=\s*([0-9.eE+\-]+)\s*$", line)
                if m:
                    return float(m.group(1))
    except Exception:
        pass
    return None


def find_omega_ci(search_dir, dump=None, omega_ci=None, bnorm=None):
    """Resolve Omega_ci [1/s], returning (value, provenance-string).

    Order: explicit --omega-ci, explicit --bnorm, explicit --dump, then a search
    of *search_dir* and its immediate subdirectories for a dump carrying
    Omega_ci (or Bnorm), then BOUT.settings / BOUT.inp. Returns (None, reason)
    if nothing is found, so the caller can drop the physical columns rather than
    print a fabricated number.
    """
    if omega_ci is not None:
        return omega_ci, "--omega-ci"
    if bnorm is not None:
        return omega_ci_from_bnorm(bnorm), f"--bnorm {bnorm} T"

    names = ("boutdata.nc", "boutdata_guards.nc", "BOUT.dmp.0.nc")

    def _numeric_subdirs(root):
        """Numbered restart folders (140/, 141/, ...), newest first."""
        try:
            return sorted((d for d in os.listdir(root)
                           if d.isdigit() and os.path.isdir(os.path.join(root, d))),
                          key=int, reverse=True)
        except OSError:
            return []

    candidates = []
    if dump:
        candidates.append(dump)
    else:
        # The usual BOUT++ layout puts the slurm logs one level ABOVE the dumps:
        #     case_A/slurm-*.out
        #     case_A/data/146/boutdata.nc
        # so search both the given directory and its data/ subdirectory (and, for
        # the case where --dir IS data/, its parent), each time checking the
        # directory itself and then its numbered restart folders.
        roots = [search_dir,
                 os.path.join(search_dir, "data"),
                 os.path.dirname(search_dir)]
        seen_roots = set()
        for root in roots:
            root = os.path.abspath(root)
            if root in seen_roots or not os.path.isdir(root):
                continue
            seen_roots.add(root)
            for name in names:
                candidates.append(os.path.join(root, name))
            for sub in _numeric_subdirs(root)[:20]:
                for name in names:
                    candidates.append(os.path.join(root, sub, name))

    for path in candidates:
        if not os.path.exists(path):
            continue
        val = _read_nc_scalar(path, ("Omega_ci",))
        if val:
            return val, f"Omega_ci from {os.path.relpath(path, search_dir)}"
        bn = _read_nc_scalar(path, ("Bnorm",))
        if bn:
            return (omega_ci_from_bnorm(bn),
                    f"Bnorm={bn:g} T from {os.path.relpath(path, search_dir)}")

    for root in (search_dir, os.path.join(search_dir, "data"),
                 os.path.dirname(os.path.abspath(search_dir))):
        for name in ("BOUT.settings", "BOUT.inp"):
            path = os.path.join(root, name)
            if os.path.exists(path):
                bn = _read_settings_bnorm(path)
                if bn:
                    rel = os.path.relpath(path, search_dir)
                    return omega_ci_from_bnorm(bn), f"Bnorm={bn:g} T from {rel}"

    return None, "not found (no dump with Omega_ci/Bnorm, no BOUT.settings)"


def fmt_hms(seconds):
    """Seconds -> compact 'XhYmZs'."""
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h {m}m {sec}s"

# ── Regex patterns ────────────────────────────────────────────────────────────

# Matches sim-time ONLY at the start of a line (after optional ^M carriage return).
# Data lines look like:   1.320e+04   51965   2.14e+04  ...
# The wall-time column (3rd field) would also match the old pattern — anchoring
# to ^ prevents that.
SIM_TIME_RE = re.compile(
    r'^\r?(\d+\.\d+e[+\-]\d+)\s+\d',
    re.MULTILINE
)

# Matches "Run time : 15 h 28 m 38 s"; runs under an hour print "19 m 16 s",
# so hours and minutes are optional
RUN_TIME_RE = re.compile(
    r'Run time\s*:\s*(?:(\d+)\s*h\s*)?(?:(\d+)\s*m\s*)?(\d+)\s*s'
)

# Extracts job ID from filename, e.g. slurm-669678.out -> 669678
JOB_ID_RE = re.compile(r'slurm-(\d+)\.out$')


# ── Per-file parser ───────────────────────────────────────────────────────────

def parse_file(filepath, omega_ci=None):
    sim_times    = []
    run_time_s   = None
    run_time_str = None

    with open(filepath, 'r', errors='replace') as fh:
        content = fh.read()

    # Collect sim-time values anchored to line starts only
    for m in SIM_TIME_RE.finditer(content):
        sim_times.append(float(m.group(1)))

    # Grab run time
    m = RUN_TIME_RE.search(content)
    if m:
        h, mn, s = (int(g or 0) for g in m.groups())
        run_time_s   = h * 3600 + mn * 60 + s
        run_time_str = f"{h}h {mn}m {s}s"

    if not sim_times:
        raise ValueError("No simulation time values found")
    if run_time_s is None:
        raise ValueError("No 'Run time' line found")
    if run_time_s == 0:
        raise ValueError("Run time is zero — cannot compute speed")

    sim_start = sim_times[0]
    sim_end   = sim_times[-1]
    delta_sim = sim_end - sim_start

    if delta_sim <= 0:
        raise ValueError(f"Non-positive sim delta: {sim_end} - {sim_start} = {delta_sim}")

    m = JOB_ID_RE.search(filepath)
    job_id = int(m.group(1)) if m else -1

    out = {
        "filename"    : os.path.basename(filepath),
        "job_id"      : job_id,
        "sim_start"   : sim_start,
        "sim_end"     : sim_end,
        "delta_sim"   : delta_sim,   # normalised time covered this job
        "run_time_s"  : run_time_s,
        "run_time_str": run_time_str,
        "speed"       : delta_sim / run_time_s,  # sim-time units per wall-clock second
    }

    if omega_ci:
        # Normalised -> physical:  t[s] = t_norm / Omega_ci
        out["t_start_us"]  = sim_start / omega_ci * 1e6
        out["t_end_us"]    = sim_end   / omega_ci * 1e6
        out["delta_t_us"]  = delta_sim / omega_ci * 1e6
        out["us_per_day"]  = out["delta_t_us"] / run_time_s * 86400.0

    return out


def speed_trace(results, omega_ci, width=34):
    """Per-job speed trace: is the run settling, speeding up, or degrading?

    The x-axis that matters is CUMULATIVE SIMULATED TIME, not job index —
    jobs differ in wall-clock length and in how far they advance, so a trace
    against job number stretches the slow ones and compresses the fast.

    What this measures is solver stiffness, not physics. Speed here is set by
    the timestep CVODE can take, so a falling trace means the problem is
    getting stiffer (in these cases usually the phi runaway) and one that
    rises then flattens means the numerics have relaxed. That correlates with
    a run settling but is NOT a steady-state test: for that use
    `turbulence_monitor.py` (fluctuation RMS saturation) or the nsep/Tsep
    traces in `h2time_hermes.py`.
    """
    key = "us_per_day" if omega_ci else "speed"
    unit = "us/day" if omega_ci else "sim/s"
    vals = [r[key] for r in results]
    if not vals:
        return
    vmax = max(vals)
    if vmax <= 0:
        return

    print("\n" + "=" * 60)
    print("SPEED TRACE  (per job, in order)")
    print("=" * 60)
    tcol = "t_end [us]" if omega_ci else "sim_end"
    print(f"  {'Job ID':>9}  {tcol:>12}  {'speed':>11}  {'':<{width}}")
    for i, r in enumerate(results):
        v = r[key]
        bar = "#" * int(round(width * v / vmax))
        tend = r["t_end_us"] if omega_ci else r["sim_end"]
        # Flag a restart that did not continue where the previous job stopped.
        gap = ""
        if i:
            prev_end = results[i - 1]["sim_end"]
            d = r["sim_start"] - prev_end
            if abs(d) > 1e-9 * max(1.0, abs(prev_end)):
                gap = "  <- overlap" if d < 0 else "  <- gap"
        print(f"  {r['job_id']:>9}  {tend:>12.4f}  {v:>11.4g}  "
              f"{bar:<{width}}{gap}")

    # Trend: median of the last third against the first third. Medians, because
    # one requeued or short job skews a mean badly.
    n = len(vals)
    if n >= 6:
        k = n // 3
        first = statistics.median(vals[:k])
        last = statistics.median(vals[-k:])
        if first > 0:
            drift = 100.0 * (last / first - 1.0)
            if abs(drift) < 15:
                verdict = "STEADY (numerically)"
            elif drift > 0:
                verdict = "SPEEDING UP — stiffness relaxing"
            else:
                verdict = "SLOWING DOWN — stiffness growing; check |phi|max"
            print(f"\n  first third median {first:.4g} {unit} -> "
                  f"last third median {last:.4g} {unit}  ({drift:+.1f} %)")
            print(f"  {verdict}")
    else:
        print(f"\n  (need >= 6 jobs for a trend verdict; have {n})")
    print("\n  NB this is SOLVER speed, i.e. stiffness. For physics steady "
          "state use\n     turbulence_monitor.py or h2time_hermes.py.")


def plot_trace(results, omega_ci, path):
    """Speed vs cumulative simulated time, with a rolling median."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print(f"\nmatplotlib not available — skipping {path}")
        return

    key = "us_per_day" if omega_ci else "speed"
    ylab = "speed [us/day]" if omega_ci else "speed [sim-time/s]"
    xlab = ("cumulative simulated time [us]" if omega_ci
            else "cumulative sim time [normalised]")
    xs = [(r["t_end_us"] if omega_ci else r["sim_end"]) for r in results]
    ys = [r[key] for r in results]

    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(9, 6.5), sharex=True)
    ax.plot(xs, ys, "o-", ms=4, lw=1, label="per job")
    if len(ys) >= 5:                       # rolling median, window 5
        roll = [statistics.median(ys[max(0, i - 2):i + 3])
                for i in range(len(ys))]
        ax.plot(xs, roll, "r-", lw=2, label="rolling median (5)")
    # Mark restarts that did not continue where the previous job stopped —
    # the same check the text trace makes, so the two agree.
    bad = [i for i in range(1, len(results))
           if abs(results[i]["sim_start"] - results[i - 1]["sim_end"])
           > 1e-9 * max(1.0, abs(results[i - 1]["sim_end"]))]
    if bad:
        ax.plot([xs[i] for i in bad], [ys[i] for i in bad], "kv", ms=9,
                mfc="none", mew=1.5, label="restart gap / overlap")
    ax.set_ylabel(ylab)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_title("Solver speed per job "
                 "(stiffness proxy, not a steady-state test)", fontsize=10)

    # Wall-clock per job: the same information inverted, and the number that
    # actually decides how long the campaign takes.
    ax2.plot(xs, [r["run_time_s"] / 3600.0 for r in results], "s-", ms=4, lw=1,
             color="0.3")
    ax2.set_ylabel("wall clock per job [h]")
    ax2.set_xlabel(xlab)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    print(f"\nTrace plot written to: {path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Parse slurm output files and compute simulation speed statistics."
    )
    parser.add_argument("--start", type=int, required=True,
                        help="First job ID to include (inclusive), e.g. 613958")
    parser.add_argument("--end",   type=int, required=True,
                        help="Last job ID to include (inclusive), e.g. 669679")
    parser.add_argument("--dir",   type=str, default=".",
                        help="Directory containing slurm-*.out files (default: current dir)")
    parser.add_argument("--omega-ci", type=float, default=None,
                        help="Omega_ci [1/s] override; skips auto-detection")
    parser.add_argument("--bnorm", type=float, default=None,
                        help="Bnorm [T]; Omega_ci derived as qe*Bnorm/Mp")
    parser.add_argument("--dump", type=str, default=None,
                        help="Explicit .nc dump to read Omega_ci / Bnorm from")
    parser.add_argument("--no-trace", action="store_true",
                        help="suppress the per-job speed trace")
    parser.add_argument("--plot", type=str, nargs="?", const="slurm_speed.png",
                        default=None, metavar="PATH",
                        help="also write a speed-vs-simulated-time plot "
                             "(default slurm_speed.png)")
    parser.add_argument("--target", type=float, default=None,
                        help="Physical time goal [us]; adds an ETA to the summary")
    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.dir, "slurm-*.out")))

    files = []
    for fp in all_files:
        m = JOB_ID_RE.search(fp)
        if m and args.start <= int(m.group(1)) <= args.end:
            files.append(fp)

    if not files:
        print(f"No slurm-*.out files found with job IDs in "
              f"[{args.start}, {args.end}] under '{os.path.abspath(args.dir)}'")
        sys.exit(1)

    omega_ci, provenance = find_omega_ci(
        os.path.abspath(args.dir), dump=args.dump,
        omega_ci=args.omega_ci, bnorm=args.bnorm,
    )

    print(f"Job ID range : {args.start} – {args.end}")
    print(f"Directory    : {os.path.abspath(args.dir)}")
    print(f"Files matched: {len(files)}")
    if omega_ci:
        print(f"Omega_ci     : {omega_ci:.6e} 1/s   ({provenance})")
        print(f"               1 sim-time unit = {1.0/omega_ci*1e6:.4f} us physical")
    else:
        print(f"Omega_ci     : UNKNOWN — {provenance}")
        print("               physical-time columns omitted; pass --omega-ci, "
              "--bnorm or --dump")
    print()

    results  = []
    failures = []

    if omega_ci:
        col = ("{:<22}  {:>9}  {:>12}  {:>12}  {:>12}  {:>11}  "
               "{:>13}  {:>14}  {:>11}")
        header = ("File", "Job ID", "Sim Start", "Sim End", "Δ Sim", "Δt [µs]",
                  "Run Time", "Speed (sim/s)", "µs/day")
    else:
        col = "{:<22}  {:>9}  {:>12}  {:>12}  {:>12}  {:>14}  {:>16}"
        header = ("File", "Job ID", "Sim Start", "Sim End", "Δ Sim",
                  "Run Time", "Speed (sim/s)")
    print(col.format(*header))
    print("-" * (len(col.format(*header)) + 2))

    for fp in sorted(files, key=lambda f: int(JOB_ID_RE.search(f).group(1))):
        try:
            r = parse_file(fp, omega_ci=omega_ci)
            results.append(r)
            row = [
                r["filename"],
                r["job_id"],
                f"{r['sim_start']:.4e}",
                f"{r['sim_end']:.4e}",
                f"{r['delta_sim']:.4e}",
            ]
            if omega_ci:
                row.append(f"{r['delta_t_us']:.4f}")
            row.append(r["run_time_str"])
            row.append(f"{r['speed']:.4e}")
            if omega_ci:
                row.append(f"{r['us_per_day']:.3f}")
            print(col.format(*row))
        except ValueError as e:
            n_cols = 9 if omega_ci else 7
            row = [os.path.basename(fp)] + ["—"] * (n_cols - 2) + [f"SKIP: {e}"]
            print(col.format(*row))
            failures.append((os.path.basename(fp), str(e)))

    if results and not args.no_trace:
        speed_trace(results, omega_ci)
    if results and args.plot:
        plot_trace(results, omega_ci, args.plot)

    if results:
        speeds     = [r["speed"]      for r in results]
        run_times  = [r["run_time_s"] for r in results]
        delta_sims = [r["delta_sim"]  for r in results]

        def stats_block(label, values, unit=""):
            n    = len(values)
            mean = statistics.mean(values)
            med  = statistics.median(values)
            mn   = min(values)
            mx   = max(values)
            sd   = statistics.stdev(values) if n > 1 else float("nan")
            min_file = results[values.index(mn)]["filename"]
            max_file = results[values.index(mx)]["filename"]
            print(f"\n  {label}  [{unit}]")
            print(f"    N        : {n}")
            print(f"    Mean     : {mean:.4e}")
            print(f"    Median   : {med:.4e}")
            print(f"    Std Dev  : {sd:.4e}")
            print(f"    Min      : {mn:.4e}  ({min_file})")
            print(f"    Max      : {mx:.4e}  ({max_file})")

        print("\n" + "=" * 60)
        print("SUMMARY STATISTICS")
        print("=" * 60)
        print("\n  'Sim Time Delta' = physics time covered per job (sim_end - sim_start)")
        print("  'Speed'          = Sim Time Delta / wall-clock seconds\n")
        stats_block("Computing Speed",     speeds,     unit="sim-time / wall-s")
        stats_block("Wall-clock Run Time", run_times,  unit="seconds")
        stats_block("Sim Time Delta",      delta_sims, unit="sim-time units")

        if omega_ci:
            stats_block("Physical Time Delta", [r["delta_t_us"] for r in results],
                        unit="us")
            stats_block("Throughput",          [r["us_per_day"] for r in results],
                        unit="us physical / wall-day")

            total_wall_s = sum(run_times)
            total_us     = sum(r["delta_t_us"] for r in results)
            # Aggregate rate, not the mean of per-job rates: long jobs should
            # weigh more than short ones.
            agg_us_day   = total_us / total_wall_s * 86400.0

            print("\n" + "=" * 60)
            print("REAL TIME TOTALS")
            print("=" * 60)
            print(f"  Physical time simulated : {total_us:.4f} us"
                  f"   ({total_us*1e-3:.5f} ms)")
            print(f"  Wall-clock consumed     : {fmt_hms(total_wall_s)}"
                  f"   ({total_wall_s/86400.0:.2f} days)")
            print(f"  Aggregate throughput    : {agg_us_day:.3f} us/day"
                  f"   ({agg_us_day/24.0:.4f} us/hour)")
            if agg_us_day > 0:
                print(f"  Wall time per 1 us      : {fmt_hms(86400.0/agg_us_day)}")
                print(f"  Wall time per 1 ms      : "
                      f"{1000.0/agg_us_day:.1f} days")

            if args.target:
                remaining = args.target - total_us
                if remaining <= 0:
                    print(f"\n  Target {args.target:g} us already reached "
                          f"({total_us:.4f} us simulated).")
                elif agg_us_day > 0:
                    print(f"\n  To reach {args.target:g} us: "
                          f"{remaining:.4f} us still needed "
                          f"-> {remaining/agg_us_day:.1f} more wall-days "
                          f"at the aggregate rate.")

    if failures:
        print(f"\n{len(failures)} file(s) skipped:")
        for fname, reason in failures:
            print(f"  {fname}: {reason}")

    #if results:
    #    csv_path = f"slurm_speed_{args.start}_{args.end}.csv"
    #    with open(csv_path, "w") as fh:
    #        cols = ("filename,job_id,sim_start,sim_end,delta_sim,"
    #                "run_time_s,run_time_str,speed_sim_per_s")
    #        if omega_ci:
    #            cols += ",t_start_us,t_end_us,delta_t_us,us_per_day"
    #        fh.write(cols + "\n")
    #        for r in results:
    #            line = (
    #                f"{r['filename']},{r['job_id']},"
    #                f"{r['sim_start']:.6e},{r['sim_end']:.6e},{r['delta_sim']:.6e},"
    #                f"{r['run_time_s']},{r['run_time_str']},{r['speed']:.6e}"
    #            )
    #            if omega_ci:
    #                line += (f",{r['t_start_us']:.6e},{r['t_end_us']:.6e},"
    #                         f"{r['delta_t_us']:.6e},{r['us_per_day']:.6e}")
    #            fh.write(line + "\n")
    #    print(f"\nResults written to: {csv_path}")


if __name__ == "__main__":
    main()

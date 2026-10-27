#!/usr/bin/env python3
"""
cvode_report.py
---------------
CVODE solver diagnostics from BOUT.log.0, for comparing preconditioners (or any
solver setting) between runs of the same case. Reads text logs only, so it is
safe to run through the sshfs mount.

What BOUT++ prints after every output (cvode.cxx, run()):

    CVODE: nsteps N, nfevals F, nniters NI, npevals P, nliters L
        -> Last step size: h, order: q
        -> Local error fails: LE, nonlinear convergence fails: NL
    <sim time>  <RHS evals>  <wall time>  Calc Inv Comm I/O SOLVER

  nsteps   accepted time steps                       CVodeGetNumSteps
  nfevals  RHS calls made by CVODE itself            CVodeGetNumRhsEvals
  nniters  Newton iterations                         CVodeGetNumNonlinSolvIters
  npevals  preconditioner APPLICATIONS (psolve calls, i.e. calls of
           Hermes::precon) - not setups               CVodeGetNumPrecSolves
  nliters  GMRES iterations                          CVodeGetNumLinIters
  LE       local-error test fails (accuracy layer: step rejected as inaccurate)
  NL       Newton convergence fails (step rejected because Newton did not converge)

All counters are cumulative since the run started, so this script differences
consecutive blocks to get per-output values. The RHS-evals column is per output
and equals d(nfevals) + d(nliters): every GMRES iteration costs one RHS call for
the difference-quotient J*v. The sim-time column has only 4 significant figures,
so the mean step is taken from the exact output timestep instead.

Derived per output (d = change over that output):
  h_mean     output_timestep / d(nsteps)  - THE number a preconditioner should raise
  newt/step  d(nniters)/d(nsteps)
  lin/newt   d(nliters)/d(nniters)        - near maxl means GMRES hits its cap
  ps/newt    d(npevals)/d(nniters)        - should be ~ lin/newt + 1 (one extra
                                            application per Newton iteration)
  NL/step, LE/step  fail rates            - NL >> LE: convergence-limited (a better
                                            preconditioner can help); LE dominant:
                                            accuracy-limited (it cannot)
  wall/RHS   wall time per RHS evaluation - the cost of one F(u)
  us/day     physical time simulated per wall-clock day (Omega_ci = qe*Bnorm/Mp,
             proton mass, no AA)

Do NOT compare lin/newt between runs on its own: CVODE re-chooses h, and a better
preconditioner shows up as a larger h_mean at a similar lin/newt (numerics notes,
'Testing a preconditioner'). The comparison that matters is us/day, read together
with h_mean and the fail rates. The script also lists every option that differs
between the runs and warns when the Hermes git version differs, so a binary
change cannot masquerade as a preconditioner effect.

Usage:
    python cvode_report.py RUN [RUN ...] [--skip N] [--plot out.png] [--per-output]

    RUN           run directory holding BOUT.log.0 (or the log file itself)
    --skip N      leave the first N CVODE outputs out of the summary (start-up
                  transient); default 0
    --per-output  print the per-output table for every run
    --plot PATH   save h_mean, lin/newt, fail rates and wall time vs output
"""

import argparse
import os
import re
import sys

from parse_slurm_speed import RUN_TIME_RE, omega_ci_from_bnorm

CVODE_RE = re.compile(
    r"CVODE: nsteps (\d+), nfevals (\d+), nniters (\d+), npevals (\d+), nliters (\d+)"
    r".*?Last step size:\s+([0-9.eE+\-]+), order:\s+(\d+)"
    r".*?Local error fails:\s+(\d+), nonlinear convergence fails:\s+(\d+)"
    r".*?\n\s*([0-9.]+e[+\-]\d+)\s+(\d+)\s+([0-9.eE+\-]+)"
    r"\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)\s+([0-9.\-]+)",
    re.DOTALL)
OUTPUT_DT_RE = re.compile(r"output timestep of ([0-9.eE+\-]+)")
OPTION_RE = re.compile(r"^\s*Option (\S+) = (.*?) \((?:default|[^)]*)\)\s*$", re.MULTILINE)
# Options that always differ between runs and say nothing about the setup
IGNORE_OPTIONS = {"datadir", "settingsfile", "restart", "append", "optionfile",
                  "wall_limit", "time_report:show"}
COUNTERS = ["nsteps", "nfevals", "nniters", "npevals", "nliters"]


def parse_log(path):
    """One run: provenance, options and the list of per-output CVODE records."""
    if os.path.isdir(path):
        path = os.path.join(path, "BOUT.log.0")
    with open(path, "r", errors="replace") as fh:
        text = fh.read().replace("\r", "\n")   # progress lines use bare \r

    run = {"path": path, "name": os.path.basename(os.path.dirname(os.path.abspath(path)))}
    m = re.search(r"Git Version of Hermes:\s*([0-9a-f]+)", text)
    run["hermes_git"] = m.group(1)[:10] if m else "?"
    m = re.search(r"Command line options for this run\s*:\s*(\S+)", text)
    run["binary"] = m.group(1) if m else "?"
    m = re.search(r"Processor number:\s*\d+\s+of\s+(\d+)", text)
    run["nprocs"] = m.group(1) if m else "?"
    m = RUN_TIME_RE.search(text)
    run["run_time_s"] = (sum(int(g or 0) * f for g, f in zip(m.groups(), (3600, 60, 1)))
                         if m else None)
    run["finished"] = "Run finished at" in text
    m = re.search(r"BoutException[^\n]*", text)
    run["exception"] = m.group(0)[:160] if m else ""

    # Last value wins: the log echoes some options twice
    run["options"] = {k: v.strip() for k, v in OPTION_RE.findall(text)
                      if k not in IGNORE_OPTIONS}
    m = OUTPUT_DT_RE.search(text)
    run["dt_out"] = float(m.group(1)) if m else None
    bnorm = run["options"].get("hermes:Bnorm")
    try:
        run["omega_ci"] = omega_ci_from_bnorm(float(bnorm))
    except (TypeError, ValueError):
        run["omega_ci"] = None   # Bnorm given as an expression: physical columns omitted
    maxl = run["options"].get("solver:maxl")
    run["maxl"] = int(float(maxl)) if maxl else None

    recs = []
    for g in CVODE_RE.findall(text):
        recs.append(dict(zip(COUNTERS, map(int, g[:5])),
                         h_last=float(g[5]), order=int(g[6]),
                         le=int(g[7]), nl=int(g[8]),
                         t=float(g[9]), rhs=int(g[10]), wall=float(g[11]),
                         calc=float(g[12]), inv=float(g[13]), comm=float(g[14]),
                         io=float(g[15]), solver=float(g[16])))
    run["records"] = recs
    return run


def per_output(run):
    """Per-output derived rows from the cumulative counters."""
    rows, prev = [], dict.fromkeys(COUNTERS + ["le", "nl"], 0)
    for i, r in enumerate(run["records"]):
        d = {k: r[k] - prev[k] for k in COUNTERS + ["le", "nl"]}
        prev = r
        row = {"i": i, "t": r["t"], "order": r["order"], "h_last": r["h_last"],
               "wall": r["wall"], "rhs": r["rhs"], "solver_pct": r["solver"], **d}
        row.update(ratios(d, r["wall"], r["rhs"], run["dt_out"]))
        rows.append(row)
    return rows


def ratios(d, wall, rhs, dt_out):
    def div(a, b):
        return a / b if b else float("nan")
    return {"h_mean": div(dt_out, d["nsteps"]) if dt_out else float("nan"),
            "newt_step": div(d["nniters"], d["nsteps"]),
            "lin_newt": div(d["nliters"], d["nniters"]),
            "ps_newt": div(d["npevals"], d["nniters"]),
            "nl_step": div(d["nl"], d["nsteps"]),
            "le_step": div(d["le"], d["nsteps"]),
            "gmres_rhs_frac": div(d["nliters"], rhs),
            "wall_rhs": div(wall, rhs),
            "wall_step": div(wall, d["nsteps"])}


def summary(run, rows, skip):
    """Totals over outputs skip.. end (counters differenced, not averaged)."""
    use = rows[skip:]
    if not use:
        return None
    d = {k: sum(r[k] for r in use) for k in COUNTERS + ["le", "nl"]}
    wall, rhs = sum(r["wall"] for r in use), sum(r["rhs"] for r in use)
    span = run["dt_out"] * len(use) if run["dt_out"] else None
    s = {"n_out": len(use), "wall": wall, "rhs": rhs, **d,
         **ratios(d, wall, rhs, span)}
    s["order_last"] = use[-1]["order"]
    s["solver_pct"] = sum(r["solver_pct"] * r["wall"] for r in use) / wall if wall else float("nan")
    s["us_day"] = (span / run["omega_ci"] * 1e6 / wall * 86400.0
                   if span and run["omega_ci"] and wall else float("nan"))
    s["h_mean_us"] = (s["h_mean"] / run["omega_ci"] * 1e6
                      if run["omega_ci"] else float("nan"))
    return s


def verdict(run, s, rows):
    """Plain-language reading of one run's summary, in the order to check things."""
    out = []
    if run["maxl"] is not None:
        # lin/newt == maxl in an output means every GMRES solve in it stopped at the cap;
        # the run average can hide this when capped and uncapped outputs alternate
        capped = sum(r["lin_newt"] >= run["maxl"] - 1e-9 for r in rows)
        if s["lin_newt"] > 0.8 * run["maxl"] or capped:
            out.append(f"GMRES at its cap in {capped}/{len(rows)} outputs (mean lin/newt "
                       f"{s['lin_newt']:.2f}, maxl {run['maxl']}): raise maxl before judging "
                       "the preconditioner")
    if s["nl"] > 3 * max(s["le"], 1):
        out.append(f"convergence-limited (NL {s['nl']} >> LE {s['le']}): Newton/GMRES sets h, "
                   "so a better preconditioner can raise it")
    elif s["le"] > s["nl"]:
        out.append(f"accuracy-limited (LE {s['le']} >= NL {s['nl']}): the error test sets h, "
                   "a preconditioner cannot raise it")
    else:
        out.append(f"mixed limits (NL {s['nl']}, LE {s['le']})")
    if abs(s["ps_newt"] - s["lin_newt"] - 1) > 0.2:
        out.append(f"ps/newt {s['ps_newt']:.2f} != lin/newt+1 {s['lin_newt'] + 1:.2f}: "
                   "unexpected preconditioner call pattern")
    if not run["finished"]:
        out.append("run did NOT finish (wall kill, crash or still running)"
                   + (f": {run['exception']}" if run["exception"] else ""))
    return out


def fmt(x, spec):
    return "-" if x is None or x != x else format(x, spec)


def print_per_output(run, rows):
    print(f"\n--- {run['name']}: per output ---")
    print(f"{'#':>3} {'steps':>7} {'h_mean':>9} {'h_last':>9} {'q':>1} {'newt/st':>7} "
          f"{'lin/nt':>6} {'ps/nt':>6} {'NL/st':>6} {'LE/st':>6} {'RHS':>8} "
          f"{'wall[s]':>8} {'ms/RHS':>7}")
    for r in rows:
        print(f"{r['i']:>3} {r['nsteps']:>7} {fmt(r['h_mean'], '9.3e')} {r['h_last']:9.3e} "
              f"{r['order']:>1} {fmt(r['newt_step'], '7.2f')} {fmt(r['lin_newt'], '6.2f')} "
              f"{fmt(r['ps_newt'], '6.2f')} {fmt(r['nl_step'], '6.3f')} "
              f"{fmt(r['le_step'], '6.3f')} {r['rhs']:>8} {r['wall']:8.1f} "
              f"{fmt(r['wall_rhs'] * 1e3, '7.2f')}")


def print_comparison(runs, sums):
    labels = [r["name"] for r in runs]
    w = max(12, *(len(x) for x in labels)) + 1
    lines = [
        ("hermes git", [r["hermes_git"] for r in runs]),
        ("nprocs", [r["nprocs"] for r in runs]),
        ("outputs used", [s["n_out"] for s in sums]),
        ("wall [s]", [fmt(s["wall"], ".0f") for s in sums]),
        ("us/day", [fmt(s["us_day"], ".3g") for s in sums]),
        ("h_mean [norm]", [fmt(s["h_mean"], ".3e") for s in sums]),
        ("h_mean [us]", [fmt(s["h_mean_us"], ".3e") for s in sums]),
        ("last order", [s["order_last"] for s in sums]),
        ("steps", [s["nsteps"] for s in sums]),
        ("Newton/step", [fmt(s["newt_step"], ".3f") for s in sums]),
        ("Lin/Newton", [fmt(s["lin_newt"], ".3f") for s in sums]),
        ("Psolve/Newton", [fmt(s["ps_newt"], ".3f") for s in sums]),
        ("NL fails/step", [fmt(s["nl_step"], ".4f") for s in sums]),
        ("LE fails/step", [fmt(s["le_step"], ".4f") for s in sums]),
        ("NL fails", [s["nl"] for s in sums]),
        ("LE fails", [s["le"] for s in sums]),
        ("RHS evals", [s["rhs"] for s in sums]),
        ("GMRES share of RHS", [fmt(s["gmres_rhs_frac"], ".2f") for s in sums]),
        ("wall/RHS [ms]", [fmt(s["wall_rhs"] * 1e3, ".3f") for s in sums]),
        ("wall/step [ms]", [fmt(s["wall_step"] * 1e3, ".2f") for s in sums]),
        ("SOLVER % of wall", [fmt(s["solver_pct"], ".1f") for s in sums]),
    ]
    print("\n=== CVODE summary ===")
    print(f"{'':<20}" + "".join(f"{x:>{w}}" for x in labels))
    for name, vals in lines:
        print(f"{name:<20}" + "".join(f"{str(v):>{w}}" for v in vals))
    if len(runs) > 1 and sums[0]["us_day"] == sums[0]["us_day"]:
        base = sums[0]["us_day"]
        print(f"{'speed vs ' + labels[0][:10]:<20}"
              + "".join(f"{fmt(s['us_day'] / base, '.2f') + 'x':>{w}}" for s in sums))
    if not runs[0]["omega_ci"]:
        print("(us columns omitted: hermes:Bnorm not a plain number in the log)")


def print_option_diff(runs):
    keys = sorted(set().union(*(r["options"] for r in runs)))
    diff = [k for k in keys if len({r["options"].get(k, "(unset)") for r in runs}) > 1]
    print("\n=== Options that differ between runs ===")
    if not diff:
        print("  none - identical option sets")
    for k in diff:
        print(f"  {k}: " + " | ".join(f"{r['name']}={r['options'].get(k, '(unset)')}"
                                      for r in runs))
    if len({r["hermes_git"] for r in runs}) > 1:
        print("  WARNING: different Hermes git versions - a speed difference may be the "
              "code change, not the setting you meant to test")
    if len({r["binary"] for r in runs}) > 1:
        print("  binaries: " + " | ".join(f"{r['name']}={r['binary']}" for r in runs))


def plot(runs, all_rows, path):
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for run, rows in zip(runs, all_rows):
        i = [r["i"] for r in rows]
        axs[0, 0].semilogy(i, [r["h_mean"] for r in rows], "-o", ms=3, label=run["name"])
        axs[0, 1].plot(i, [r["lin_newt"] for r in rows], "-o", ms=3)
        # zero fail rates cannot sit on a log axis: leave those points out
        axs[1, 0].semilogy(i, [r["nl_step"] or float("nan") for r in rows], "-o", ms=3)
        axs[1, 0].semilogy(i, [r["le_step"] or float("nan") for r in rows], "--x", ms=3,
                           color=axs[1, 0].lines[-1].get_color())
        axs[1, 1].plot(i, [r["wall"] for r in rows], "-o", ms=3)
    if runs[0]["maxl"]:
        axs[0, 1].axhline(runs[0]["maxl"], color="grey", ls=":", label="maxl")
        axs[0, 1].legend(frameon=False)
    axs[0, 0].set_ylabel("mean step h (normalised)")
    axs[0, 1].set_ylabel("GMRES its / Newton it")
    axs[1, 0].set_ylabel("fails per step (o NL, x LE)")
    axs[1, 1].set_ylabel("wall time per output [s]")
    for ax in axs[1]:
        ax.set_xlabel("output index")
    axs[0, 0].legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    print(f"\nplot saved: {path}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--skip", type=int, default=0)
    ap.add_argument("--per-output", action="store_true")
    ap.add_argument("--plot")
    args = ap.parse_args()

    runs, all_rows, sums = [], [], []
    for p in args.runs:
        run = parse_log(p)
        rows = per_output(run)
        s = summary(run, rows, args.skip)
        if s is None:
            print(f"{run['name']}: no CVODE output blocks after skip={args.skip} "
                  f"({len(rows)} found) - not a CVODE run, or it died before the first output",
                  file=sys.stderr)
            continue
        runs.append(run)
        all_rows.append(rows)
        sums.append(s)
    if not runs:
        sys.exit(1)

    if args.per_output:
        for run, rows in zip(runs, all_rows):
            print_per_output(run, rows)
    print_comparison(runs, sums)
    print("\n=== Reading ===")
    for run, s, rows in zip(runs, sums, all_rows):
        for line in verdict(run, s, rows[args.skip:]):
            print(f"  {run['name']}: {line}")
    if len(runs) > 1:
        print_option_diff(runs)
    if args.plot:
        plot(runs, all_rows, args.plot)


if __name__ == "__main__":
    main()

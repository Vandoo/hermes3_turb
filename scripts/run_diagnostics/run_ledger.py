#!/usr/bin/env python3
"""
run_ledger.py
-------------
One row per restart folder of a Hermes-3 case: which binary ran it, what changed
in BOUT.inp, how fast it went and how it ended. Built from the logs after the
fact, so nothing has to be remembered at submission time.

Columns (all read from <folder>/BOUT.log.0 and <folder>/BOUT.inp):
  case, folder      case directory and numbered restart folder
  started, run_time "Run started at" / "Run time" (blank if the job never finished)
  status            finished | unfinished (wall kill, crash or still running)
  nprocs            "Processor number: 0 of N"
  binary            executable path from "Command line options" - the build
                    directory names the branch checkout (e.g. .../anomalous_EM/...)
  hermes_git        "Git Version of Hermes" (first 8 chars)
  bout_version      first line of the log
  inp_md5           md5 of BOUT.inp (first 8 chars) - equal md5 = identical input
  inp_changes       options that differ from the previous folder's BOUT.inp
  t_start, t_end    first/last sim time in the output table (normalised)
  us_per_day        physical us simulated per wall-clock day (Omega_ci = qe*Bnorm/Mp)
  last_dt, order    CVODE last step size and order
  le_fails, nl_fails  CVODE local-error / nonlinear-convergence fails
  exception         first BoutException line in the log, if any

Compare runs in us_per_day, not by fail counts: a run with many NL fails can
still be the faster one (see memory bug_wall_limit_exceeds_slurm_time).

Cluster data: do NOT run this through an sshfs mount on .nc-heavy folders -
copy it (and parse_slurm_speed.py, which it imports) to the cluster and run it
there, then merge the printed CSV locally:

    scp run_ledger.py parse_slurm_speed.py <host>:/tmp/
    ssh -o BatchMode=yes <host> 'cd /tmp && python3 run_ledger.py <case_dir> --csv' \
        > /tmp/rows.csv
    python run_ledger.py --merge /tmp/rows.csv --out ../../run_ledger.csv

Usage:
    python run_ledger.py <case_dir> [--out ledger.csv] [--csv]

    <case_dir>  directory holding numbered restart folders (or a data/ with them)
    --out       merge rows into this CSV (keyed by case+folder; later runs overwrite)
    --csv       print CSV to stdout instead of a table
    --merge     CSV file of rows (from a remote --csv run) to merge into --out
"""

import argparse
import csv
import hashlib
import os
import re
import socket
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from parse_slurm_speed import SIM_TIME_RE, RUN_TIME_RE, find_omega_ci

COLUMNS = ["host", "case", "folder", "started", "run_time", "status", "nprocs",
           "binary", "hermes_git", "bout_version", "inp_md5", "inp_changes",
           "t_start", "t_end", "us_per_day", "last_dt", "order",
           "le_fails", "nl_fails", "exception"]

CVODE_FAILS_RE = re.compile(
    r"Local error fails:\s+(\d+), nonlinear convergence fails:\s+(\d+)")
CVODE_LAST_RE = re.compile(r"Last step size:\s+([0-9.eE+\-]+), order:\s+(\d+)")


def read_inp(path):
    """BOUT.inp -> {"section:key": value} with comments stripped."""
    opts, section = {}, ""
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = re.match(r"\[(.+)\]$", line)
            if m:
                section = m.group(1).strip()
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                opts[f"{section}:{key}" if section else key] = val.strip()
    return opts


def inp_changes(prev, curr):
    """Compact 'a: x->y; b: (new)=z; c: removed' between two option dicts."""
    out = []
    for key in sorted(set(prev) | set(curr)):
        a, b = prev.get(key), curr.get(key)
        if a == b:
            continue
        if a is None:
            out.append(f"{key}: (new)={b}")
        elif b is None:
            out.append(f"{key}: removed")
        else:
            out.append(f"{key}: {a}->{b}")
    return "; ".join(out)


def parse_log(path, omega_ci):
    """Provenance, speed and solver health from one BOUT.log.0."""
    with open(path, "r", errors="replace") as fh:
        content = fh.read().replace("\r", "\n")   # progress lines use bare \r

    row = {}
    first = content.split("\n", 1)[0]
    row["bout_version"] = first.replace("BOUT++ version", "").strip()

    m = re.search(r"Processor number:\s*\d+\s+of\s+(\d+)", content)
    row["nprocs"] = m.group(1) if m else ""
    m = re.search(r"Command line options for this run\s*:\s*(\S+)", content)
    row["binary"] = m.group(1) if m else ""
    m = re.search(r"Git Version of Hermes:\s*([0-9a-f]+)", content)
    row["hermes_git"] = m.group(1)[:8] if m else ""
    m = re.search(r"Run started at\s*:\s*(.+)", content)
    row["started"] = m.group(1).strip() if m else ""

    m = RUN_TIME_RE.search(content)
    run_time_s = None
    if m:
        h, mn, s = (int(g or 0) for g in m.groups())
        run_time_s = h * 3600 + mn * 60 + s
        row["run_time"] = f"{h}h{mn:02d}m"
    else:
        row["run_time"] = ""
    row["status"] = "finished" if "Run finished at" in content else "unfinished"

    sim_strs = SIM_TIME_RE.findall(content)
    sims = [float(x) for x in sim_strs]
    row["t_start"] = sim_strs[0] if sims else ""
    row["t_end"] = sim_strs[-1] if sims else ""
    row["us_per_day"] = ""
    if sims and run_time_s and omega_ci:
        if sims[-1] > sims[0]:
            dt_us = (sims[-1] - sims[0]) / omega_ci * 1e6
            row["us_per_day"] = f"{dt_us / run_time_s * 86400.0:.3g}"
        else:
            # Table prints 4 significant figures: the run advanced less than one
            # printed digit, so only an upper bound is known.
            mant, exp = sim_strs[-1].lower().split("e")
            ndig = len(mant.split(".")[1]) if "." in mant else 0
            step_us = 10.0 ** (int(exp) - ndig) / omega_ci * 1e6
            row["us_per_day"] = f"<{step_us / run_time_s * 86400.0:.2g}"

    fails = CVODE_FAILS_RE.findall(content)
    row["le_fails"], row["nl_fails"] = fails[-1] if fails else ("", "")
    last = CVODE_LAST_RE.findall(content)
    row["last_dt"], row["order"] = last[-1] if last else ("", "")

    m = re.search(r"BoutException[^\n]*", content)
    row["exception"] = m.group(0)[:160] if m else ""
    return row


def numbered_folders(case_dir):
    """Numbered restart folders in case_dir or case_dir/data, ascending."""
    for root in (case_dir, os.path.join(case_dir, "data")):
        if not os.path.isdir(root):
            continue
        subs = sorted((d for d in os.listdir(root)
                       if d.isdigit() and os.path.isdir(os.path.join(root, d))),
                      key=int)
        if subs:
            return root, subs
    return case_dir, []


def scan(case_dir):
    case_dir = os.path.abspath(case_dir)
    root, folders = numbered_folders(case_dir)
    omega_ci, _ = find_omega_ci(root)
    host = socket.gethostname().split(".")[0]
    rows, prev_opts = [], None
    for folder in folders:
        fdir = os.path.join(root, folder)
        log = os.path.join(fdir, "BOUT.log.0")
        inp = os.path.join(fdir, "BOUT.inp")
        if not os.path.exists(log):
            continue
        row = {c: "" for c in COLUMNS}
        row.update(host=host, case=case_dir, folder=folder)
        row.update(parse_log(log, omega_ci))
        if os.path.exists(inp):
            with open(inp, "rb") as fh:
                row["inp_md5"] = hashlib.md5(fh.read()).hexdigest()[:8]
            opts = read_inp(inp)
            if prev_opts is not None:
                row["inp_changes"] = inp_changes(prev_opts, opts)
            prev_opts = opts
        rows.append(row)
    return rows


def merge_into(path, rows):
    """Merge rows into the CSV at path, keyed by (case, folder)."""
    existing = {}
    if os.path.exists(path):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                existing[(r["case"], r["folder"])] = r
    for r in rows:
        existing[(r["case"], r["folder"])] = r
    ordered = sorted(existing.values(),
                     key=lambda r: (r["case"], int(r["folder"]) if r["folder"].isdigit() else 0))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(ordered)
    return len(rows), len(ordered)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    p.add_argument("case_dir", nargs="?")
    p.add_argument("--out")
    p.add_argument("--csv", action="store_true")
    p.add_argument("--merge")
    a = p.parse_args()

    if a.merge:
        with open(a.merge, newline="") as fh:
            rows = list(csv.DictReader(fh))
    elif a.case_dir:
        rows = scan(a.case_dir)
    else:
        p.error("give a case_dir or --merge")

    if a.csv:
        w = csv.DictWriter(sys.stdout, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    else:
        for r in rows:
            print(f"{r['folder']:>5}  {r['status']:<10} {r['run_time']:>7}  "
                  f"us/day={r['us_per_day']:<7} dt={r['last_dt']:<10} ord={r['order']:<2} "
                  f"LE={r['le_fails']:<6} NL={r['nl_fails']:<6} git={r['hermes_git']}")
            if r["inp_changes"]:
                print(f"       inp: {r['inp_changes']}")
            if r["exception"]:
                print(f"       !! {r['exception']}")
    if a.out:
        n, total = merge_into(a.out, rows)
        print(f"merged {n} rows into {a.out} ({total} total)", file=sys.stderr)


if __name__ == "__main__":
    main()

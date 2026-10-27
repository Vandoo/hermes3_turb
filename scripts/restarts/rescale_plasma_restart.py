"""
rescale_restart.py
------------------
Multiply the density of a Hermes-3 restart by a constant factor, so a run can be
started AT the density you want instead of being fuelled up to it.

Why: the particle fill time of an edge/SOL case is milliseconds (see
particle_fill_time.py), while a 3D turbulence run advances microseconds per day
of wall clock.  Waiting for a gas puff to raise n_e,sep is not possible.  What is
possible is to set n directly and let the turbulence re-find its saturated state,
which happens on the much shorter energy/turbulence time.

What gets scaled, and why
  N<species>   x f    density
  NV<species>  x f    momentum - keeps the parallel velocity V = NV/N unchanged.
  P<species>   x f    ONLY in --scaling keep-T (see below)

Two ways to handle the pressure, and the choice matters:

  keep-P (default)  scale N and NV, leave P alone.  The stored energy W is
                    unchanged, so no energy transient is created.  T = P/N falls
                    to T/f, which is roughly where fixed-power balance wants it
                    anyway (T ~ (P_in/n)^alpha).

  keep-T            scale P with N so T is unchanged.  Gentler on the divertor
                    (no sudden cooling, less risk of a cold/stiff crash), but it
                    raises W by f, and that excess has to decay over tau_E.
                    Check tau_E with particle_fill_time.py first: if tau_E is
                    long compared with what you can afford to run, keep-T just
                    swaps a particle transient you cannot outrun for an energy
                    transient you cannot outrun either.

What does NOT get scaled
  Vort, phi           Hermes-3 inverts the vorticity in the BOUSSINESQ
                      approximation: src/vorticity.cxx sets the Laplacian
                      coefficient to average_atomic_mass / B^2, a fixed reference
                      mass with NO density in it.  A uniform density rescale
                      therefore does not invalidate the vorticity inversion.
                      phi is only a solver initial guess and is re-solved anyway.
  T<species>, V<species>, geometry, metric, metadata

Neutrals (--scale-neutrals, on by default) are scaled too: in steady state the
neutral density tracks the recycling flux, which tracks the plasma density, so
leaving them un-scaled just means waiting out the (also slow) neutral fill.

This is initial-condition surgery, not a converged solution.  The rescaled state
is NOT a self-consistent equilibrium: expect a transient while the profiles,
flows and turbulence re-adjust, and check the run has re-saturated (e.g. with
turbulence_monitor.py) before trusting any output.

Usage
-----
    python rescale_restart.py <restart_dir> --factor 1.478
    python rescale_restart.py <restart_dir> --target-ne 5.0e18 --current-ne 3.383e18

    -o, --out DIR        write scaled copies here (default: <restart_dir>_scaled)
        --in-place       modify the files where they are (no copy) - be sure
        --no-scale-neutrals
        --dry-run        report what would change, write nothing
"""

from __future__ import annotations

import argparse
import glob
import os
import re
import shutil
import sys

import numpy as np
import netCDF4 as nc

# Never touch these, whatever their rank
SKIP = {
    "Vort", "phi",
    "J", "dx", "dy", "dz", "zShift", "Bxy",
    "G1", "G2", "G3",
    "g11", "g12", "g13", "g22", "g23", "g33",
    "g_11", "g_12", "g_13", "g_22", "g_23", "g_33",
}


def species_of(name: str) -> str | None:
    """'NVd+' -> 'd+', 'Pe' -> 'e', 'Nd' -> 'd'.  None if not a species field."""
    m = re.match(r"^(NV|N|P|T|V)(.+)$", name)
    if not m:
        return None
    return m.group(2)


def classify(ds: nc.Dataset, scale_neutrals: bool,
             scale_pressure: bool) -> tuple[list[str], list[str]]:
    """Return (to_scale, skipped_species_fields)."""
    to_scale, skipped = [], []
    for name, var in ds.variables.items():
        if name in SKIP or var.ndim < 3:
            continue
        kind = re.match(r"^(NV|N|P)", name)
        sp = species_of(name)
        if kind is None or sp is None:
            skipped.append(name)
            continue
        if kind.group(1) == "P" and not scale_pressure:
            skipped.append(name)
            continue
        is_neutral = ("+" not in sp) and sp != "e"
        if is_neutral and not scale_neutrals:
            skipped.append(name)
            continue
        to_scale.append(name)
    return sorted(to_scale), sorted(skipped)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Rescale the density of a Hermes-3 restart.")
    ap.add_argument("restart_dir")
    ap.add_argument("--factor", type=float, default=None)
    ap.add_argument("--target-ne", type=float, default=None)
    ap.add_argument("--current-ne", type=float, default=None)
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--in-place", action="store_true")
    ap.add_argument("--no-scale-neutrals", dest="scale_neutrals",
                    action="store_false", default=True)
    ap.add_argument("--scaling", choices=("keep-P", "keep-T"), default="keep-P",
                    help="keep-P (default): scale N and NV only, W unchanged, T -> T/f. "
                         "keep-T: also scale P, T unchanged, W -> W*f.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    # ---- factor ------------------------------------------------------------
    if args.factor is not None:
        factor = args.factor
    elif args.target_ne is not None and args.current_ne is not None:
        factor = args.target_ne / args.current_ne
        print(f"target n_e,sep {args.target_ne:.4e} / current {args.current_ne:.4e}"
              f"  ->  factor {factor:.6f}")
    else:
        ap.error("give --factor, or both --target-ne and --current-ne")
    if not np.isfinite(factor) or factor <= 0:
        ap.error(f"factor must be finite and positive (got {factor})")

    files = sorted(glob.glob(os.path.join(args.restart_dir, "BOUT.restart.*.nc")))
    if not files:
        print(f"no BOUT.restart.*.nc in {args.restart_dir}")
        return 1
    print(f"{len(files)} restart files in {args.restart_dir}")

    # ---- decide what to scale, from the first file -------------------------
    keep_T = args.scaling == "keep-T"
    with nc.Dataset(files[0]) as ds:
        to_scale, skipped = classify(ds, args.scale_neutrals, scale_pressure=keep_T)
    if not to_scale:
        print("nothing to scale - are these Hermes-3 restarts?")
        return 1
    print(f"factor      : {factor:.6f}")
    print(f"scaling     : {', '.join(to_scale)}")
    print(f"leaving     : {', '.join(skipped) if skipped else '(nothing)'}")
    print("  (Vort/phi deliberately untouched - Boussinesq inversion has no n in it)")
    if keep_T:
        print(f"  mode keep-T : T unchanged, stored energy W -> W x {factor:.4f}"
              f" (decays over tau_E - check you can afford that)")
    else:
        print(f"  mode keep-P : W unchanged, T -> T / {factor:.4f}"
              f" (no energy transient; watch for cold-divertor stiffness)")

    if args.dry_run:
        with nc.Dataset(files[0]) as ds:
            print("\ndry run, ranges in the first file:")
            for name in to_scale:
                a = np.array(ds[name][:])
                print(f"  {name:<8} [{a.min():+.4e}, {a.max():+.4e}]"
                      f"  ->  [{a.min()*factor:+.4e}, {a.max()*factor:+.4e}]")
        return 0

    # ---- destination -------------------------------------------------------
    if args.in_place:
        targets = files
        print("\n*** modifying files IN PLACE ***")
    else:
        outdir = args.out or (args.restart_dir.rstrip("/") + "_scaled")
        os.makedirs(outdir, exist_ok=True)
        print(f"\nwriting scaled copies to {outdir}")
        targets = []
        for f in files:
            dest = os.path.join(outdir, os.path.basename(f))
            shutil.copy2(f, dest)
            targets.append(dest)

    # ---- scale -------------------------------------------------------------
    ranges = {}
    for i, path in enumerate(targets):
        with nc.Dataset(path, "r+") as ds:
            for name in to_scale:
                if name not in ds.variables:
                    continue
                a = np.array(ds[name][:]) * factor
                if not np.all(np.isfinite(a)):
                    print(f"!!! non-finite values in {name} of {path} - aborting")
                    return 1
                ds[name][:] = a
                lo, hi = float(a.min()), float(a.max())
                if name in ranges:
                    ranges[name] = (min(ranges[name][0], lo), max(ranges[name][1], hi))
                else:
                    ranges[name] = (lo, hi)
        if (i + 1) % 50 == 0 or i + 1 == len(targets):
            print(f"  {i+1}/{len(targets)} files")

    print("\nresulting ranges over all files:")
    for name, (lo, hi) in sorted(ranges.items()):
        # NV* is momentum and is legitimately signed; only density and
        # pressure going negative is a problem.
        sign_definite = name.startswith(("N", "P")) and not name.startswith("NV")
        flag = "   <-- NEGATIVE" if lo < 0 and sign_definite else ""
        print(f"  {name:<8} [{lo:+.4e}, {hi:+.4e}]{flag}")

    print("\nNext steps:")
    print("  - point the run at these restarts (restart = true)")
    print("  - drop start_timestep (e.g. 1e-3) for the first job: the state is not")
    print("    a self-consistent solution and the solver will complain otherwise")
    print("  - size the puff and pump_recycle_multiplier for the NEW density, or the")
    print("    run will just resume drifting toward the old equilibrium")
    print("  - re-check saturation before using any output")
    return 0


if __name__ == "__main__":
    sys.exit(main())

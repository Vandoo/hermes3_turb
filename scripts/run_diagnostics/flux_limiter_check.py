#!/usr/bin/env python3
"""
flux_limiter_check.py — would kappa_limit_alpha / eta_limit_alpha act on this run?

Reads a squashed, guard-retaining Hermes-3 ``boutdata.nc`` with netCDF4 only
(no xbout — runs with the cluster's bare python) and evaluates, cell by cell,

  conduction : R1 = |kappa_par Grad_par T| / (N T sqrt(T/AA))
  viscosity  : R1 = |eta Grad_par V| / P,  eta = c P tau, P tau from kappa_par

For a limiter coefficient alpha the code multiplies kappa (eta) by
1/(1 + R1/alpha).  Prints, per region and species, the R1 percentiles, the
fraction of cells cut by >10 %, and the flux-weighted fraction of the
classical flux that would survive, for a scan of alpha; plus an OMP radial
profile.  Saves the toroidally averaged R1 maps to an .npz for plotting.

Gradients are taken in field-aligned coordinates (zShift FFT), rows whose
centred stencil crosses a guard or X-point cut are skipped.  The run itself
need not have any limiter on (it should NOT, or kappa_par is already limited).
See hermes3_turb/limiters.py for the formulas and assumptions.

Usage:
  python flux_limiter_check.py <run>/data/425/boutdata.nc [--frames -1]
         [--alphas 0.1 0.2 0.3 0.5 1] [--out limiter_check.npz]
         [--species e i] [--AA 0.000544662 2] [--kappa-coeff e=2.2345 i=3.9]

Species: plasma species only (electrons "e" + ions, e.g. "d+" in neutral runs);
neutrals are not handled by braginskii_conduction.  Needs T<sp>, P<sp> and
kappa_par_<sp> (written only with [<sp>] diagnose = true).  V<sp> is used if
saved, otherwise V = NV<sp> / (AA * N<sp>), with N<sp> = P<sp>/T<sp> for a
quasineutral ion that does not save its density.
"""
import argparse
import os
import sys

import numpy as np
import netCDF4 as nc

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
try:
    from hermes3_turb.limiters import (shift_z, dump_row_to_y, valid_parallel_rows,
                                       grad_par_aligned, conduction_ratio,
                                       viscosity_ratio, limiter_summary,
                                       species_kind, KAPPA_COEFF)
except ImportError:  # module copied next to the script on the cluster
    from limiters import (shift_z, dump_row_to_y, valid_parallel_rows,
                          grad_par_aligned, conduction_ratio,
                          viscosity_ratio, limiter_summary,
                          species_kind, KAPPA_COEFF)

p = argparse.ArgumentParser()
p.add_argument("dump")
p.add_argument("--frames", type=int, nargs="+", default=[-1])
p.add_argument("--alphas", type=float, nargs="+", default=[0.1, 0.2, 0.3, 0.5, 1.0])
p.add_argument("--out", default="limiter_check.npz")
p.add_argument("--species", nargs="+", default=["e", "i"],
               help="plasma species names as in the dump (neutral runs: e d+)")
p.add_argument("--AA", type=float, nargs="+", default=None,
               help="atomic mass per species, same order (default 1/1836 for e, 2 otherwise)")
p.add_argument("--kappa-coeff", nargs="*", default=[],
               help="override kappa_coefficient, e.g. d+=3.9 (only if BOUT.inp set it)")
args = p.parse_args()

species = args.species
AA = {sp: (args.AA[k] if args.AA else (1.0 / 1836.0 if species_kind(sp) == "e" else 2.0))
      for k, sp in enumerate(species)}
kcoef = {sp: KAPPA_COEFF[species_kind(sp)] for sp in species}
for item in args.kappa_coeff:
    name, val = item.split("=")
    kcoef[name] = float(val)

d = nc.Dataset(args.dump)
v = d.variables
g = lambda k: np.asarray(v[k][...])
gi = lambda k: int(g(k))

MXG, MYG = gi("MXG"), gi("MYG")
ny, ny_inner, ixs = gi("ny"), gi("ny_inner"), gi("ixseps1")
j11, j21, j12, j22 = (gi(k) for k in ("jyseps1_1", "jyseps2_1", "jyseps1_2", "jyseps2_2"))
nx, npol, nz = v["Te"].shape[1:]
dn = j21 != j12
ymap = dump_row_to_y(npol, ny, ny_inner, double_null=dn)
cuts = [j11, j21, j12, j22] + ([ny_inner - 1] if dn else [])
ok = valid_parallel_rows(ymap, cuts)

zsh, dz, dy, g22 = g("zShift"), float(g("dz").ravel()[0]), g("dy"), g("g_22")
Rmaj = g("Rxy") if "Rxy" in v else np.sqrt(g("g_33"))
print(f"{args.dump}\n nx={nx} npol={npol} nz={nz} MXG={MXG} MYG={MYG} ixseps1={ixs} "
      f"jyseps={j11},{j21},{j12},{j22} ny_inner={ny_inner} double_null={dn}")
print(f" y-guard rows in dump: {(ymap < 0).sum()} (per target {(npol - ny) // (4 if dn else 2)})"
      f"; valid parallel-gradient rows: {ok.sum()} of {(ymap >= 0).sum()} interior")

# ---- region masks on (x, y) -------------------------------------------------
y = ymap
in_main = ((y > j11) & (y <= j21)) | ((y > j12) & (y <= j22))
in_leg = (y >= 0) & ~in_main
xin = np.zeros(nx, bool); xin[MXG:nx - MXG] = True
sol = np.zeros(nx, bool); sol[ixs:nx - MXG] = True
core = xin & ~sol
regions = {
    "SOL main": np.outer(sol, in_main & ok),
    "SOL legs": np.outer(sol, in_leg & ok),
    "core": np.outer(core, in_main & ok),
    "PFR": np.outer(core, in_leg & ok),
}
interior_rows = np.where(y >= 0)[0]
omp = interior_rows[np.argmax(Rmaj[ixs, interior_rows] * ok[interior_rows])]
print(f" OMP row {omp} (y={y[omp]})")

t_arr = g("t_array")
frames = [f % len(t_arr) for f in args.frames]
print(f" frames {frames}  t = {t_arr[frames]}")
print(" species " + ", ".join(f"{sp} ({species_kind(sp)}, AA={AA[sp]:.4g}, kappa_coeff={kcoef[sp]:.4g})"
                              for sp in species))
missing = [k for sp in species for k in (f"T{sp}", f"P{sp}", f"kappa_par_{sp}") if k not in v]
if missing:
    sys.exit(f"missing variables {missing}: kappa_par_<sp> needs [<sp>] diagnose = true")


def velocity(sp, f):
    """V<sp> if saved, else NV/(AA N), N from N<sp> or P/T (quasineutral ion)."""
    if f"V{sp}" in v:
        return g(f"V{sp}")[f]
    if f"NV{sp}" not in v:
        sys.exit(f"neither V{sp} nor NV{sp} in the dump")
    N = g(f"N{sp}")[f] if f"N{sp}" in v else g(f"P{sp}")[f] / np.maximum(g(f"T{sp}")[f], 1e-10)
    return g(f"NV{sp}")[f] / (AA[sp] * np.maximum(N, 1e-10))


for sp in species:
    print(f" V{sp}: " + ("saved" if f"V{sp}" in v else
                         f"derived from NV{sp}/(AA*N{sp})" if f"N{sp}" in v else
                         f"derived from NV{sp}/(AA*P{sp}/T{sp})"))

cases = [(k, sp) for k in ("cond", "visc") for sp in species]
pool = {c: {r: ([], []) for r in regions} for c in cases}
zmean = {c: np.zeros((nx, npol)) for c in cases}
omp_T = {sp: np.zeros(nx) for sp in species}

for f in frames:
    for sp in species:
        T = g(f"T{sp}")[f]; P = g(f"P{sp}")[f]; V = velocity(sp, f)
        kap = g(f"kappa_par_{sp}")[f]
        gT = grad_par_aligned(shift_z(T, zsh, dz), dy, g22, ok)
        gV = grad_par_aligned(shift_z(V, zsh, dz), dy, g22, ok)
        # gradients back to the stored (non-aligned) frame so they sit on the
        # same cells as kappa/T/P; NaN rows stay NaN through the FFT per (x,y)
        gT = np.where(np.isnan(gT), np.nan, shift_z(np.nan_to_num(gT), zsh, dz, -1))
        gV = np.where(np.isnan(gV), np.nan, shift_z(np.nan_to_num(gV), zsh, dz, -1))
        for kind, (R1, q) in (("cond", conduction_ratio(kap, T, P, AA[sp], gT)),
                              ("visc", viscosity_ratio(kap, P, AA[sp], gV,
                                                       species_kind(sp), kcoef[sp]))):
            for rname, m in regions.items():
                m3 = np.broadcast_to(m[..., None], R1.shape)
                pool[(kind, sp)][rname][0].append(R1[m3])
                pool[(kind, sp)][rname][1].append(q[m3])
            zmean[(kind, sp)] += np.nanmean(R1, axis=-1) / len(frames)
        omp_T[sp] += T[:, omp, :].mean(-1) / len(frames)

label = {"cond": "CONDUCTION  kappa_limit_alpha", "visc": "VISCOSITY   eta_limit_alpha"}
for c in cases:
    print(f"\n=== {label[c[0]]}   species {c[1]}")
    print(f"{'region':>9} {'R1 p50':>8} {'p90':>8} {'p99':>8} | "
          + " | ".join(f"a={a:<4g} cut>10% q_kept" for a in args.alphas))
    for rname in regions:
        R1 = np.concatenate(pool[c][rname][0]); q = np.concatenate(pool[c][rname][1])
        s = limiter_summary(R1, q, np.ones(R1.shape, bool), args.alphas)
        print(f"{rname:>9} {s['R1_p50']:8.3g} {s['R1_p90']:8.3g} {s['R1_p99']:8.3g} | "
              + " | ".join(f"{100*s[a][0]:9.1f}% {s[a][1]:6.2f}" for a in args.alphas))

print(f"\n=== OMP radial profile (toroidal mean R1, alpha=1; divide by alpha)")
print(f"{'x-ixs':>6} " + " ".join(f"{'T'+sp+'[eV]':>8}" for sp in species) + " "
      + " ".join(f"{k+' '+sp:>9}" for k, sp in cases))
Tn = float(g("Tnorm"))
for x in list(range(max(MXG, ixs - 20), ixs, 5)) + list(range(ixs, nx - MXG, 6)):
    print(f"{x-ixs:6d} " + " ".join(f"{omp_T[sp][x]*Tn:8.1f}" for sp in species) + " "
          + " ".join(f"{zmean[c][x, omp]:9.3g}" for c in cases))

np.savez(args.out, ymap=ymap, ok=ok, omp=omp, ixseps1=ixs, Rmaj=Rmaj,
         **{f"R1_{k}_{s}": zmean[(k, s)] for k, s in cases})
print(f"\nsaved {args.out}")

#!/usr/bin/env python3
"""
Connection length, sound speed and parallel residence time — compare geometries.

Answers "does the longer divertor leg give a longer parallel residence time,
and by how much?" for two or more cases side by side (e.g. Super-X vs
conventional divertor). Works for **STORM and Hermes-3** dumps: the connection
length comes from the grid alone and is code-agnostic, and only the
temperature normalisation differs, which is auto-detected.

What it computes
----------------
Connection length, integrating the parallel arc element along the field:

    L_c = Σ dy·√g_22          [m]

from the outer midplane to each target (and the X-point on the way), per flux
surface. ``√g_22 = hthe·B/Bp`` is verified against the grid to machine
precision on every run — if that check fails the grid is not in SI and every
length below is wrong.

Parallel residence time, three ways, because they differ by a factor ~2 and
the difference is systematic:

    tau_omp  = L_c / c_s(T_e at the midplane)     single representative c_s
    tau_mean = L_c / c_s(<T_e> along the leg)     sound speed of the mean T
    tau_int  = ∫ dl / c_s(T_e(l))                 leg-resolved  <-- the honest one

``c_s = √(e·T_e/m_i)`` (cold ion), matching STORM's own normalisation
``c_s0 = √(e·T_e0/m_i)``. Ion temperature is not used; including it as
``√(e(T_e+T_i)/m_i)`` scales every c_s by a constant and leaves all ratios
unchanged.

**Why tau_int is not optional.** ``1/c_s ∝ T_e^{-1/2}`` is convex, so
``⟨1/√T⟩ ≥ 1/√⟨T⟩``: using the sound speed of the mean temperature always
*understates* tau_par, and understates it most where T varies most along the
leg. ``c_s(⟨T⟩)`` is not a shortcut for ``⟨c_s⁻¹⟩``. On the MAST-U SXD/CD pair
this is a factor 2.0 (SXD) vs 1.5 (CD) — i.e. it distorts the *ratio*, not
just the absolute values.

Caveats that apply to every number here
---------------------------------------
* ``L/c_s`` is a free-streaming estimate. The real parallel flow accelerates
  from ~0 at the stagnation point to c_s only at the sheath, so all three
  columns are **lower bounds** on the residence time. Applied consistently to
  every case, so the ratio between geometries is the robust part.
* A leg that reaches ~1 eV (detached) dominates ``∫dl/c_s``, so tau_int is
  sensitive to the detachment state of the coldest case.
* Single output folders with few time slices give an order-of-magnitude
  number; average over more folders before quoting a ratio as firm.

Usage
-----
Edit the CASES list and run. Each entry is (label, grid file, dump file).
The dump is only needed for T_e — omit it (None) to get connection lengths
alone.
"""

import os
import numpy as np
import netCDF4
import matplotlib.pyplot as plt

# ============================================================================
# CONFIG
# ============================================================================
_ROOT = "../.."          # only used to shorten the paths below

CASES = [
    ("SXD", "sxd_n1/bout_sxd_nx244_v6.grd.nc", "sxd_n1/data/705/boutdata.nc"),
    ("CD",  "cd_n1/bout_conv_v4.grd.nc",       "cd_n1/data/500/boutdata.nc"),
]

leg          = "outer_lower"   # 'outer_lower'|'outer_upper'|'inner_lower'|'inner_upper'
x_offsets    = (0, 5, 15)      # flux surfaces to tabulate, as cells outside ixseps1
x_ref        = 0               # the offset used for the tau_par table
T_e0         = 15.0            # [eV] STORM normalisation; ignored for Hermes-3
A_ion        = 2.0             # ion mass number
save_png     = False
# ============================================================================

E_CHARGE = 1.602176634e-19
U_AMU = 1.66053906660e-27


def _meta(d):
    g = lambda k: int(d[k][:].item()) if k in d.variables else None
    return {k: g(k) for k in ("nx", "ny", "ny_inner", "ixseps1", "ixseps2",
                              "jyseps1_1", "jyseps2_1", "jyseps1_2",
                              "jyseps2_2", "y_boundary_guards")}


def load_grid(path):
    """Parallel arc element dl = dy·√g_22 [m], plus topology, from the grid."""
    d = netCDF4.Dataset(path)
    m = _meta(d)
    dy = np.asarray(d["dy"][:], dtype=float)
    g22 = np.asarray(d["g_22"][:], dtype=float)

    # Verify the grid is SI: sqrt(g_22) must equal hthe·B/Bp exactly. A grid in
    # normalised units fails this, and every length below would be wrong.
    if all(v in d.variables for v in ("hthe", "Bxy", "Bpxy")):
        lhs = np.sqrt(g22)
        rhs = (np.asarray(d["hthe"][:], dtype=float)
               * np.asarray(d["Bxy"][:], dtype=float)
               / np.asarray(d["Bpxy"][:], dtype=float))
        dev = float(np.abs(lhs / rhs - 1.0).max())
        if dev > 1e-6:
            raise RuntimeError(
                f"{path}: sqrt(g_22) != hthe*B/Bp (max dev {dev:.2e}). The grid "
                f"is not in SI units, so connection lengths would be wrong.")
        m["metric_check"] = dev
    d.close()
    return dy * np.sqrt(g22), m


def leg_indices(m, which):
    """Guardless (start, stop) y-cells from the midplane to a target, and the
    X-point index, for the requested leg.

    BOUT++ double-null layout, guardless: the inner region is
    ``0 .. ny_inner-1`` and the outer ``ny_inner .. ny-1``; the four X-points
    are the ``jyseps`` indices. Single null collapses to the outer pair.
    """
    ny, nyi = m["ny"], m["ny_inner"]
    j11, j21, j12, j22 = (m["jyseps1_1"], m["jyseps2_1"],
                          m["jyseps1_2"], m["jyseps2_2"])
    is_dn = nyi is not None and 0 < nyi < ny
    mid_o = (j12 + j22) // 2
    mid_i = (j11 + j21) // 2
    if which == "outer_lower":
        return mid_o, ny, j22, "outer"
    if which == "outer_upper":
        return (nyi if is_dn else 0), mid_o, j12, "outer"
    if which == "inner_lower":
        return 0, mid_i, j11, "inner"
    if which == "inner_upper":
        return mid_i, (nyi if is_dn else ny), j21, "inner"
    raise ValueError(f"unknown leg {which!r}")


def load_temperature(path, m, region, T_e0=15.0):
    """z,t-averaged T_e [eV] on the guarded y grid, for STORM or Hermes-3.

    STORM writes ``T`` normalised by ``T_e0``; Hermes-3 writes ``Te`` already
    in eV once unnormalised. The guard offset differs between the two poloidal
    regions of a double null (+MYG inner, +3·MYG outer), so the caller says
    which region the leg is in.
    """
    d = netCDF4.Dataset(path)
    if "T" in d.variables:
        T = np.asarray(d["T"][:], dtype=float) * T_e0      # STORM
        code = "STORM"
    elif "Te" in d.variables:
        T = np.asarray(d["Te"][:], dtype=float)            # Hermes-3
        code = "Hermes-3"
        if T.max() < 1.0:      # still normalised -> convert with Tnorm
            T = T * float(d["Tnorm"][:]) if "Tnorm" in d.variables else T
    else:
        d.close()
        raise RuntimeError(f"{path}: neither 'T' (STORM) nor 'Te' (Hermes-3)")
    ny_full = T.shape[2]
    d.close()

    myg = m.get("y_boundary_guards") or 2
    has_guards = ny_full > m["ny"]
    off = 0 if not has_guards else (myg if region == "inner" else 3 * myg)
    return T.mean(axis=(0, 3)), off, code       # (x, y) averaged over t and z


def analyse(label, gridpath, dumppath):
    dl, m = load_grid(gridpath)
    ix = m["ixseps1"]
    y0, y1, yx, region = leg_indices(m, leg)
    jj = np.arange(y0, y1)

    out = dict(label=label, m=m, dl=dl, ix=ix, jj=jj, yx=yx, region=region,
               Lc={}, Lx={}, Lleg={})
    for off in x_offsets:
        i = ix + off
        out["Lc"][off] = float(dl[i, y0:y1].sum())
        out["Lx"][off] = float(dl[i, y0:yx + 1].sum())
        out["Lleg"][off] = float(dl[i, yx + 1:y1].sum())

    if dumppath and os.path.exists(dumppath):
        Txy, yoff, code = load_temperature(dumppath, m, region, T_e0)
        i = ix + x_ref
        Tj = Txy[i, jj + yoff]                       # T_e along the leg [eV]
        dlj = dl[i, jj]
        cs = lambda TeV: np.sqrt(E_CHARGE * np.asarray(TeV) / (A_ion * U_AMU))
        Lc = float(dlj.sum())
        out.update(code=code, Tj=Tj, dlj=dlj, Lc_ref=Lc,
                   T_omp=float(Tj[0]), T_tgt=float(Tj[-1]),
                   T_mean=float(Tj.mean()),
                   cs_omp=float(cs(Tj[0])), cs_mean=float(cs(Tj.mean())),
                   tau_omp=Lc / float(cs(Tj[0])),
                   tau_mean=Lc / float(cs(Tj.mean())),
                   tau_int=float((dlj / cs(Tj)).sum()))
    return out


# ---------------------------------------------------------------------------
res = [analyse(lab, os.path.join(_ROOT, g), os.path.join(_ROOT, f) if f else None)
       for lab, g, f in CASES]

print(f"\nleg = {leg};  c_s = sqrt(e*T_e/m_i), cold ion, A = {A_ion:g}")
for r in res:
    print(f"  {r['label']:6s} grid ny={r['m']['ny']} ny_inner={r['m']['ny_inner']} "
          f"ixseps1={r['ix']}  metric check |sqrt(g_22)/(hthe B/Bp) - 1| = "
          f"{r['m'].get('metric_check', float('nan')):.1e}")

print("\nCONNECTION LENGTH [m]  (midplane -> target)")
hdr = "  {:6s}".format("case") + "".join(f"{'sep+'+str(o):>22s}" for o in x_offsets)
print(hdr)
print("  {:6s}".format("") + "".join(f"{'L_c   X-pt   leg':>22s}" for _ in x_offsets))
for r in res:
    row = f"  {r['label']:6s}"
    for o in x_offsets:
        row += f"{r['Lc'][o]:8.2f}{r['Lx'][o]:7.2f}{r['Lleg'][o]:7.2f}"
    print(row)
if len(res) == 2:
    a, b = res
    row = f"  {'ratio':6s}"
    for o in x_offsets:
        row += (f"{a['Lc'][o]/b['Lc'][o]:8.2f}{a['Lx'][o]/b['Lx'][o]:7.2f}"
                f"{a['Lleg'][o]/b['Lleg'][o]:7.2f}")
    print(row)

if all("tau_int" in r for r in res):
    # NB "mean" spans the whole midplane->target path, the same span the
    # tau columns integrate over — not the X-point->target sub-section.
    print(f"\nTEMPERATURE along midplane -> target at sep+{x_ref} [eV]")
    print("  {:6s}{:>10s}{:>14s}{:>10s}".format(
        "case", "midplane", "path-mean", "target"))
    for r in res:
        print(f"  {r['label']:6s}{r['T_omp']:10.2f}{r['T_mean']:14.2f}{r['T_tgt']:10.2f}")

    print("\nPARALLEL RESIDENCE TIME [us]")
    print("  {:6s}{:>14s}{:>14s}{:>16s}".format(
        "case", "L_c/c_s(T_omp)", "L_c/c_s(<T>)", "INT dl/c_s(l)"))
    for r in res:
        print(f"  {r['label']:6s}{r['tau_omp']*1e6:14.0f}{r['tau_mean']*1e6:14.0f}"
              f"{r['tau_int']*1e6:16.0f}")
    if len(res) == 2:
        a, b = res
        print(f"  {'ratio':6s}{a['tau_omp']/b['tau_omp']:14.2f}"
              f"{a['tau_mean']/b['tau_mean']:14.2f}{a['tau_int']/b['tau_int']:16.2f}")
        print(f"  {'L_n ~':6s}{np.sqrt(a['tau_omp']/b['tau_omp']):14.2f}"
              f"{np.sqrt(a['tau_mean']/b['tau_mean']):14.2f}"
              f"{np.sqrt(a['tau_int']/b['tau_int']):16.2f}   <- sqrt(tau ratio)")
        print("\n  L_c/c_s is free-streaming, so all three are LOWER bounds on the")
        print("  residence time; the ratio is the robust part. c_s(<T>) is not a")
        print("  shortcut for <1/c_s> — it understates tau by "
              f"{a['tau_int']/a['tau_mean']:.1f}x ({a['label']}) and "
              f"{b['tau_int']/b['tau_mean']:.1f}x ({b['label']}) here.")

    # ---- figure --------------------------------------------------------------
    fig, ax = plt.subplots(1, 3, figsize=(14, 4.2))
    for r in res:
        s = np.concatenate(([0.0], np.cumsum(r["dlj"])))[:-1]
        ax[0].plot(s, r["Tj"], "-", label=r["label"])
        cs = np.sqrt(E_CHARGE * r["Tj"] / (A_ion * U_AMU))
        ax[1].plot(s, cs * 1e-3, "-", label=r["label"])
        ax[2].plot(s, np.cumsum(r["dlj"] / cs) * 1e6, "-", label=r["label"])
    for a_, xl, yl, ti in ((ax[0], "distance along B from midplane [m]", r"$T_e$ [eV]", "temperature"),
                           (ax[1], "distance along B from midplane [m]", r"$c_s$ [km/s]", "sound speed"),
                           (ax[2], "distance along B from midplane [m]", r"$\int dl/c_s$ [$\mu$s]", "cumulative transit time")):
        a_.set_xlabel(xl); a_.set_ylabel(yl); a_.set_title(ti); a_.legend()
    ax[0].set_yscale("log")
    fig.suptitle(f"{leg} leg, separatrix+{x_ref}")
    fig.tight_layout()
    if save_png:
        out = f"connection_length_{leg}.png"
        plt.savefig(out, dpi=140)
        print(f"\nsaved {out}")
    plt.show()

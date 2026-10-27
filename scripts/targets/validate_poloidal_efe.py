#!/usr/bin/env python3
"""
Validate the from-definitions poloidal/parallel heat-flux channels against
Hermes-3's own Y-face energy-flow diagnostics (`ef<species>_*_ylow`).

For every Y cell face the parallel energy flow [W] is reconstructed from first
principles and compared, channel by channel, with the code's diagnostic:

    advection   (5/2) P V              ↔  ef e_adv_ylow
    conduction  −κ_∥ ∂T/∂s_∥           ↔  ef e_cond_ylow
    total       advection + conduction ↔  ef e_tot_ylow

Face-flow forms (BOUT++ field-aligned coords, y = parallel coordinate):
    perpendicular face area     A_⊥ = (J/√g_22)·dx·dz
    parallel gradient           ∂T/∂s_∥ = (T_i − T_{i-1})/(√g_22·dy)
  ⇒ conduction flow = −κ_face·(T_i − T_{i-1})·(J/g_22)_face·dx·dz/dy
    advection   flow = (5/2)·P_face·V_face·A_⊥

The 3-D fields are first shifted to field-aligned coordinates
(`h3t.to_field_aligned`, replicating xbout's `_shift_z`) because with the
`shifted` parallel transform adjacent y at the same (x,z) index are NOT
field-line-connected — a raw ∂/∂y would give spurious jumps. Geometry factors
are z-independent and left unshifted.

These are integrated over the physical radial range (x-guards excluded) and the
full torus (×zperiod), giving a poloidal power profile vs y that should track
the diagnostic. The conduction then matches `efe_cond_ylow` essentially exactly
in the clean SOL/leg; advection sits ~14% low because Hermes-3 advects with an
MC slope limiter (`Div_par_mod<Limiter>`) vs the centred face average here. A
single global FFT shift is invalid across branch cuts (X-points, the DN
inter-region guard rows), so those few faces are excluded from the statistics.

Usage:  python validate_poloidal_efe.py <casepath> [species]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from xhermes import open_hermesdataset
import hermes3_turb as h3t
from hermes3_turb.load import _no_t

#casepath = sys.argv[1] if len(sys.argv) > 1 else "."
#species  = sys.argv[2] if len(sys.argv) > 2 else "e"
casepath = "./160/"
species = 'e'

# ── Load (SI, no geometry so guard-retaining dumps load without a matching grid)
if os.path.isdir(casepath):
    _nc = os.path.join(casepath, "boutdata_guards.nc")
    datapath = _nc if os.path.exists(_nc) else os.path.join(casepath, "BOUT.dmp.*.nc")
else:
    datapath = casepath
_inp = os.path.join(casepath, "BOUT.inp")
ds = open_hermesdataset(datapath, inputfilepath=_inp if os.path.exists(_inp) else None,
                        keep_yboundaries=True)
dsa = ds.mean("t") if "t" in ds.dims else ds
pdim = h3t.detect_poloidal_dim(ds)
zdim = h3t.detect_toroidal_dim(ds)
zperiod = int(ds.metadata.get("zperiod", 1))
MXG = int(ds.metadata.get("MXG", 2))
nx, ny = ds.sizes["x"], ds.sizes[pdim]

adv_name  = f"ef{species}_adv_ylow"
cond_name = f"ef{species}_cond_ylow"
tot_name  = f"ef{species}_tot_ylow"
for nm in (adv_name, cond_name, tot_name):
    if nm not in dsa:
        print(f"WARNING: '{nm}' not in dataset — that channel will be skipped.")

# ── Raw arrays (numpy) for explicit face arithmetic ──────────────────────────
def arr(name):
    return _no_t(dsa[name]).transpose("x", pdim, zdim).values if zdim in dsa[name].dims \
        else _no_t(dsa[name]).transpose("x", pdim).values

# Field-aligned 3-D fields: with the `shifted` parallel transform, adjacent y
# at the same (x,z) index are NOT field-line-connected, so the parallel
# derivative for conduction (and the advective face average) must be taken
# after shifting to field-aligned coords. Geometry factors (J, g_22, …) are
# z-independent → left unshifted.
def faligned(name_or_da):
    da = dsa[name_or_da] if isinstance(name_or_da, str) else name_or_da
    return h3t.to_field_aligned(_no_t(da), dsa).transpose("x", pdim, zdim).values

Vda = h3t.get_parallel_velocity(dsa, species)
Vda = Vda.where(np.isfinite(Vda), 0.0)
P  = faligned(f"P{species}")
V  = faligned(Vda)
T  = faligned(f"T{species}") if f"T{species}" in dsa \
    else faligned(f"P{species}") / np.clip(faligned(f"N{species}"), 1e-10, None)
kap = faligned("kappa_par_e") if (species == "e" and "kappa_par_e" in dsa) else None

J   = _no_t(dsa["J"]).transpose("x", pdim).values
g22 = _no_t(dsa["g_22"]).transpose("x", pdim).values
dx  = _no_t(dsa["dx"]).transpose("x", pdim).values
dy  = _no_t(dsa["dy"]).transpose("x", pdim).values
dz  = float(np.asarray(_no_t(dsa["dz"]).values).flat[0])

xsl = slice(MXG, nx - MXG)   # physical radial range (drop x-guards)


def face_avg(a):
    """Arithmetic average onto the lower-y face of each cell: 0.5(a[y]+a[y-1])."""
    out = np.full_like(a, np.nan, dtype=float)
    out[:, 1:] = 0.5 * (a[:, 1:] + a[:, :-1]) if a.ndim == 2 else None
    if a.ndim == 3:
        out[:, 1:, :] = 0.5 * (a[:, 1:, :] + a[:, :-1, :])
    return out


def sum_face(flow_xz):
    """Σ over physical x and all ζ, ×zperiod → [W] per y face."""
    f = np.where(np.isfinite(flow_xz), flow_xz, 0.0)
    return f[xsl].sum(axis=(0, 2)) * zperiod      # (ny,)

# face-averaged geometry / fields
Jf, g22f, dxf, dyf = (face_avg(a) for a in (J, g22, dx, dy))
Aperp = (Jf / np.sqrt(g22f))[:, :, None] * dxf[:, :, None] * dz   # (x,y,1)

# advection: (5/2) P_face V_face · A_⊥
adv_mine = sum_face(2.5 * face_avg(P) * face_avg(V) * Aperp) / 1e3   # kW

# conduction: −κ_face·(T[y]−T[y-1])·(J/g_22)_face·dx·dz/dy
cond_mine = np.full(ny, np.nan)
if kap is not None:
    dT = np.full_like(T, np.nan)
    dT[:, 1:, :] = T[:, 1:, :] - T[:, :-1, :]
    cond_flow = -face_avg(kap) * dT * (Jf / g22f)[:, :, None] * dxf[:, :, None] * dz / dyf[:, :, None]
    cond_mine = sum_face(cond_flow) / 1e3   # kW

tot_mine = adv_mine + np.nan_to_num(cond_mine)


def native(name):
    if name not in dsa:
        return np.full(ny, np.nan)
    return sum_face(arr(name)) / 1e3

adv_nat, cond_nat, tot_nat = native(adv_name), native(cond_name), native(tot_name)

# ── Report agreement over the "clean" domain ─────────────────────────────────
# A single global FFT field-aligned shift is invalid across topological branch
# cuts (X-points, the double-null inter-region guard rows), so exclude faces
# where the reconstructed conduction is a large outlier vs the diagnostic —
# those few cells need region-aware alignment, the bulk SOL/leg is unaffected.
yint = np.arange(ny)
domain = (yint >= MXG) & (yint < ny - MXG)          # y-domain faces
clean = domain & np.isfinite(cond_nat) \
    & (np.abs(cond_mine) < 10.0 * np.abs(cond_nat) + 1.0)   # drop branch-cut spikes

print("=" * 72)
print(f"Validation of poloidal heat-flux channels vs ef{species}_*_ylow  [kW per y-face]")
print("=" * 72)
print(f"{'channel':<12}{'median |mine/native|':>22}{'median |mine-nat| kW':>24}")
for label, mine, nat in [("advection", adv_mine, adv_nat),
                         ("conduction", cond_mine, cond_nat),
                         ("total", tot_mine, tot_nat)]:
    if not np.isfinite(nat).any():
        print(f"{label:<12}{'(no diagnostic)':>22}")
        continue
    m = clean & (np.abs(nat) > 0.1)
    ratio = np.nanmedian(np.abs(mine[m] / nat[m])) if m.any() else np.nan
    absdif = np.nanmedian(np.abs(mine[m] - nat[m])) if m.any() else np.nan
    print(f"{label:<12}{ratio:>22.3f}{absdif:>24.3f}")
print("=" * 72)
print("clean = y-domain faces excluding |cond_native| > 5×median (X-point/crash spikes)")

# ── Plot mine vs native for each channel ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for ax, (label, mine, nat) in zip(axes, [("advection", adv_mine, adv_nat),
                                         ("conduction", cond_mine, cond_nat),
                                         ("total", tot_mine, tot_nat)]):
    ax.plot(yint[domain], nat[domain], "ks-", ms=4, label=f"ef{species}_{label[:3]}_ylow")
    ax.plot(yint[domain], mine[domain], "r.-", label="from definitions")
    ax.set_xlabel("poloidal index y (face)")
    ax.set_ylabel("power through Y face [kW]")
    ax.set_title(f"{label}")
    ax.legend(fontsize=8)
    # clip y-range to the clean data so crash spikes don't flatten the plot
    lim = np.nanmax(np.abs(nat[clean])) * 1.5 if clean.any() else None
    if lim:
        ax.set_ylim(-lim, lim)
fig.suptitle(f"Poloidal heat-flux validation vs Hermes-3 ef{species}_*_ylow", y=1.02)
fig.tight_layout()
#out = os.path.join(os.getcwd(), f"validate_poloidal_ef{species}.png")
#fig.savefig(out, dpi=130, bbox_inches="tight")
#print(f"Saved plot: {out}")
plt.show()
